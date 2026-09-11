"""Durable, allowlisted execution selection for one run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .models import ExecutionRole, ExecutionSelection, HarnessConfig, SelectedProfile
from .profiles import ProfileError, profile_execution_fingerprint, profile_for_role


class ExecutionSelectionError(ValueError):
    """The execution selection is absent, invalid, or no longer matches config."""


class ExecutionSelectionConflict(ExecutionSelectionError):
    """A different execution selection has already been published."""


_FILENAME = "execution_selection.json"
SCHEMA_VERSION = 2
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = (
    ("planner", ExecutionRole.PLANNER),
    ("implementer", ExecutionRole.IMPLEMENTER),
    ("reviewer", ExecutionRole.REVIEWER),
)
_V1_FIELDS = frozenset({"profile_id", "driver", "model", "selection_mode", "effort", "sandbox"})
_V2_FIELDS = _V1_FIELDS | {"config_sha256"}


def is_profile_aware_run(state: Mapping[str, Any]) -> bool:
    """True when ``state.execution.planner.profile_id`` is a non-empty string."""

    if not isinstance(state, Mapping):
        return False
    execution = state.get("execution")
    if not isinstance(execution, Mapping):
        return False
    planner = execution.get("planner")
    if not isinstance(planner, Mapping):
        return False
    profile_id = planner.get("profile_id")
    return isinstance(profile_id, str) and bool(profile_id)


def _env_allowlist(config: HarnessConfig, role: ExecutionRole) -> tuple[str, ...]:
    # Only the implementer receives an agent environment.
    return tuple(config.agent.env_allowlist) if role is ExecutionRole.IMPLEMENTER else ()


def _selected(
    profile: Any,
    *,
    schema_version: int = SCHEMA_VERSION,
    agent_env_allowlist: tuple[str, ...] = (),
) -> SelectedProfile:
    return SelectedProfile(
        profile_id=profile.id,
        driver=profile.driver.value,
        model=profile.model,
        selection_mode=profile.selection_mode.value,
        effort=profile.effort,
        sandbox=profile.sandbox,
        config_sha256=(
            profile_execution_fingerprint(profile, agent_env_allowlist=agent_env_allowlist)
            if schema_version == 2
            else None
        ),
    )


def resolve_execution_selection(
    config: HarnessConfig,
    *,
    planner_profile_id: str,
    implementer_profile_id: str,
    reviewer_profile_id: str,
) -> ExecutionSelection:
    """Resolve three profiles from trusted config into a schema-2 snapshot."""

    requested = {
        "planner": planner_profile_id,
        "implementer": implementer_profile_id,
        "reviewer": reviewer_profile_id,
    }
    selected = {
        name: _selected(
            profile_for_role(config, requested[name], role),
            agent_env_allowlist=_env_allowlist(config, role),
        )
        for name, role in _ROLES
    }
    return ExecutionSelection(schema_version=SCHEMA_VERSION, **selected)


def _payload(selection: ExecutionSelection) -> dict[str, Any]:
    if not isinstance(selection, ExecutionSelection) or selection.schema_version not in (1, 2):
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    payload: dict[str, Any] = {"schema_version": selection.schema_version}
    for name, _role in _ROLES:
        value = getattr(selection, name)
        if not isinstance(value, SelectedProfile):
            raise ExecutionSelectionError(f"execution selection {name} is invalid")
        fields = asdict(value)
        if selection.schema_version == 1:
            if fields.pop("config_sha256") is not None:
                raise ExecutionSelectionError("schema 1 selection cannot carry a fingerprint")
        elif not isinstance(value.config_sha256, str) or _SHA256.fullmatch(value.config_sha256) is None:
            raise ExecutionSelectionError(f"execution selection {name}.config_sha256 is invalid")
        payload[name] = fields
    return payload


def _publish_exclusive(path: Path, content: str) -> bool:
    """Publish atomically; return False if the destination already exists.

    Same principle as ``plan_approval.json``: a fsynced private temporary file
    is hard-linked to the final name, which never replaces an existing file.
    """

    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    except OSError as exc:
        raise ExecutionSelectionError(f"could not write execution selection: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def ensure_execution_selection(
    run_dir: Path,
    selection: ExecutionSelection,
) -> ExecutionSelection:
    """Publish *selection* once; never replace a different published one.

    An identical existing snapshot is returned unchanged (idempotent retry);
    a different one raises :class:`ExecutionSelectionConflict`.
    """

    content = json.dumps(_payload(selection), ensure_ascii=False, indent=2) + "\n"
    directory = Path(run_dir).expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExecutionSelectionError(f"could not write execution selection: {exc}") from exc
    if _publish_exclusive(directory / _FILENAME, content):
        return selection
    existing = read_execution_selection(directory)
    if existing != selection:
        raise ExecutionSelectionConflict("a different execution selection is already published")
    return existing


def _selected_from(value: Any, name: str, schema_version: int) -> SelectedProfile:
    if not isinstance(value, dict):
        raise ExecutionSelectionError(f"execution selection {name} must be an object")
    required = _V2_FIELDS if schema_version == 2 else _V1_FIELDS
    if set(value) != required:
        raise ExecutionSelectionError(f"execution selection {name} has invalid fields")
    strings = ("profile_id", "driver", "model", "selection_mode")
    if any(not isinstance(value[key], str) for key in strings):
        raise ExecutionSelectionError(f"execution selection {name} has invalid metadata")
    for key in ("effort", "sandbox"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ExecutionSelectionError(f"execution selection {name}.{key} is invalid")
    if schema_version == 2 and (
        not isinstance(value["config_sha256"], str)
        or _SHA256.fullmatch(value["config_sha256"]) is None
    ):
        raise ExecutionSelectionError(f"execution selection {name}.config_sha256 is invalid")
    return SelectedProfile(**value)


def parse_execution_selection(data: bytes) -> ExecutionSelection:
    """Parse the exact bytes of an ``execution_selection.json`` artifact."""

    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise ExecutionSelectionError("execution selection must contain an object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or schema_version not in (1, 2):
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    if set(payload) != {"schema_version", "planner", "implementer", "reviewer"}:
        raise ExecutionSelectionError("execution selection has invalid fields")
    return ExecutionSelection(
        schema_version=schema_version,
        **{
            name: _selected_from(payload[name], name, schema_version)
            for name, _role in _ROLES
        },
    )


def read_execution_selection_with_sha256(
    run_dir: Path,
) -> tuple[ExecutionSelection, str]:
    """Read the selection once and return it with the SHA-256 of those bytes."""

    path = Path(run_dir).expanduser().resolve() / _FILENAME
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    return parse_execution_selection(data), hashlib.sha256(data).hexdigest()


def read_execution_selection(
    run_dir: Path,
) -> ExecutionSelection:
    return read_execution_selection_with_sha256(run_dir)[0]


def validate_execution_selection(
    config: HarnessConfig,
    selection: ExecutionSelection,
) -> None:
    """Prove the configured profiles are still exactly the selected ones.

    Schema 2 compares execution fingerprints (the implementer's includes
    ``config.agent.env_allowlist``); historic schema 1 can only compare the
    recorded metadata.  Any divergence raises :class:`ExecutionSelectionError`.
    """

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    if not isinstance(selection, ExecutionSelection) or selection.schema_version not in (1, 2):
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    for name, role in _ROLES:
        selected = getattr(selection, name)
        if not isinstance(selected, SelectedProfile):
            raise ExecutionSelectionError(f"execution selection {name} is invalid")
        if selection.schema_version == 2 and selected.config_sha256 is None:
            raise ExecutionSelectionError(f"execution selection {name}.config_sha256 is missing")
        try:
            profile = profile_for_role(config, selected.profile_id, role)
            expected = _selected(
                profile,
                schema_version=selection.schema_version,
                agent_env_allowlist=_env_allowlist(config, role),
            )
        except ProfileError as exc:
            raise ExecutionSelectionError(
                f"execution selection {name} profile is no longer available: {exc}"
            ) from exc
        if selected != expected:
            raise ExecutionSelectionError(
                f"execution selection {name} profile {selected.profile_id!r} "
                "no longer matches the configured profile"
            )


__all__ = [
    "ExecutionSelectionConflict",
    "ExecutionSelectionError",
    "SCHEMA_VERSION",
    "ensure_execution_selection",
    "is_profile_aware_run",
    "parse_execution_selection",
    "read_execution_selection",
    "read_execution_selection_with_sha256",
    "resolve_execution_selection",
    "validate_execution_selection",
]
