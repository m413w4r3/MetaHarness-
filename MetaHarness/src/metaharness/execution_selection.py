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

from .models import (
    ExecutionRole,
    ExecutionSelection,
    ExecutionSelectionV3,
    HarnessConfig,
    SelectedProfile,
    StepExecutionSelection,
)
from .profiles import ProfileError, profile_execution_fingerprint, profile_for_role


class ExecutionSelectionError(ValueError):
    """The execution selection is absent, invalid, or no longer matches config."""


class ExecutionSelectionConflict(ExecutionSelectionError):
    """A different execution selection has already been published."""


_FILENAME = "execution_selection.json"
SCHEMA_VERSION = 2
SCHEMA_VERSION_V3 = 3
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STEP_ID = re.compile(r"S0[1-6]\Z")
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
    codex_home: Path | None = None,
) -> SelectedProfile:
    return SelectedProfile(
        profile_id=profile.id,
        driver=profile.driver.value,
        model=profile.model,
        selection_mode=profile.selection_mode.value,
        effort=profile.effort,
        sandbox=profile.sandbox,
        config_sha256=(
            profile_execution_fingerprint(
                profile,
                agent_env_allowlist=agent_env_allowlist,
                codex_home=codex_home,
            )
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
            codex_home=(config.codex_runtime.home if role is ExecutionRole.IMPLEMENTER else None),
        )
        for name, role in _ROLES
    }
    return ExecutionSelection(schema_version=SCHEMA_VERSION, **selected)


def resolve_execution_selection_v3(
    config: HarnessConfig,
    *,
    planner_profile_id: str,
    step_profile_ids: Mapping[str, str],
    reviewer_profile_id: str,
) -> ExecutionSelectionV3:
    """Resolve one immutable profile snapshot for every ordered v2 step."""

    planner = _selected(
        profile_for_role(config, planner_profile_id, ExecutionRole.PLANNER),
        agent_env_allowlist=_env_allowlist(config, ExecutionRole.PLANNER),
    )
    reviewer = _selected(
        profile_for_role(config, reviewer_profile_id, ExecutionRole.REVIEWER),
        agent_env_allowlist=_env_allowlist(config, ExecutionRole.REVIEWER),
    )
    if not isinstance(step_profile_ids, Mapping) or not step_profile_ids:
        raise ExecutionSelectionError("v3 selection must contain steps")
    steps: list[StepExecutionSelection] = []
    for step_id, profile_id in step_profile_ids.items():
        if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None or not isinstance(profile_id, str):
            raise ExecutionSelectionError("v3 step selection is invalid")
        profile = profile_for_role(config, profile_id, ExecutionRole.IMPLEMENTER)
        steps.append(
            StepExecutionSelection(
                step_id=step_id,
                implementer=_selected(
                    profile,
                    agent_env_allowlist=_env_allowlist(config, ExecutionRole.IMPLEMENTER),
                    codex_home=config.codex_runtime.home,
                ),
            )
        )
    return ExecutionSelectionV3(SCHEMA_VERSION_V3, planner, tuple(steps), reviewer)


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


def _payload_v3(selection: ExecutionSelectionV3) -> dict[str, Any]:
    if not isinstance(selection, ExecutionSelectionV3) or selection.schema_version != SCHEMA_VERSION_V3:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    if not isinstance(selection.planner, SelectedProfile) or not isinstance(selection.reviewer, SelectedProfile):
        raise ExecutionSelectionError("execution selection profile is invalid")
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in selection.steps:
        if not isinstance(item, StepExecutionSelection) or not isinstance(item.step_id, str) or _STEP_ID.fullmatch(item.step_id) is None or item.step_id in seen:
            raise ExecutionSelectionError("execution selection steps are invalid")
        if not isinstance(item.implementer, SelectedProfile):
            raise ExecutionSelectionError("execution selection implementer is invalid")
        if item.implementer.config_sha256 is None or _SHA256.fullmatch(item.implementer.config_sha256) is None:
            raise ExecutionSelectionError("execution selection implementer.config_sha256 is invalid")
        seen.add(item.step_id)
        steps.append({"step_id": item.step_id, "implementer": asdict(item.implementer)})
    if not steps:
        raise ExecutionSelectionError("execution selection steps are missing")
    for name, profile in (("planner", selection.planner), ("reviewer", selection.reviewer)):
        if profile.config_sha256 is None or _SHA256.fullmatch(profile.config_sha256) is None:
            raise ExecutionSelectionError(f"execution selection {name}.config_sha256 is invalid")
    return {
        "schema_version": SCHEMA_VERSION_V3,
        "planner": asdict(selection.planner),
        "steps": steps,
        "reviewer": asdict(selection.reviewer),
    }


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

    if isinstance(selection, ExecutionSelectionV3):
        return ensure_execution_selection_v3(run_dir, selection)  # type: ignore[return-value]
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


def ensure_execution_selection_v3(run_dir: Path, selection: ExecutionSelectionV3) -> ExecutionSelectionV3:
    """Publish the v3 selection once, idempotently and exclusively."""

    content = json.dumps(_payload_v3(selection), ensure_ascii=False, indent=2) + "\n"
    directory = Path(run_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if _publish_exclusive(directory / _FILENAME, content):
        return selection
    existing = read_execution_selection_v3(directory)
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
    if schema_version == SCHEMA_VERSION_V3:
        return parse_execution_selection_v3(data)  # type: ignore[return-value]
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


def _selected_v3(value: Any, name: str) -> SelectedProfile:
    if not isinstance(value, dict) or set(value) != _V2_FIELDS:
        raise ExecutionSelectionError(f"execution selection {name} is invalid")
    return _selected_from(value, name, 2)


def parse_execution_selection_v3(data: bytes) -> ExecutionSelectionV3:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "planner", "steps", "reviewer"} or payload.get("schema_version") != SCHEMA_VERSION_V3:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ExecutionSelectionError("execution selection steps are invalid")
    steps: list[StepExecutionSelection] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict) or set(item) != {"step_id", "implementer"}:
            raise ExecutionSelectionError(f"execution selection step {index} is invalid")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None or step_id in seen:
            raise ExecutionSelectionError("execution selection step IDs are invalid")
        seen.add(step_id)
        steps.append(StepExecutionSelection(step_id, _selected_v3(item.get("implementer"), f"step {step_id}")))
    if [item.step_id for item in steps] != [f"S{index:02d}" for index in range(1, len(steps) + 1)]:
        raise ExecutionSelectionError("execution selection step IDs are not contiguous")
    return ExecutionSelectionV3(
        SCHEMA_VERSION_V3,
        _selected_v3(payload["planner"], "planner"),
        tuple(steps),
        _selected_v3(payload["reviewer"], "reviewer"),
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


def read_execution_selection_v3_with_sha256(run_dir: Path) -> tuple[ExecutionSelectionV3, str]:
    path = Path(run_dir).expanduser().resolve() / _FILENAME
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    return parse_execution_selection_v3(data), hashlib.sha256(data).hexdigest()


def read_execution_selection_v3(run_dir: Path) -> ExecutionSelectionV3:
    return read_execution_selection_v3_with_sha256(run_dir)[0]


def validate_execution_selection(
    config: HarnessConfig,
    selection: ExecutionSelection,
) -> None:
    """Prove the configured profiles are still exactly the selected ones.

    Schema 2 compares execution fingerprints (the implementer's includes
    ``config.agent.env_allowlist``); historic schema 1 can only compare the
    recorded metadata.  Any divergence raises :class:`ExecutionSelectionError`.
    """

    if isinstance(selection, ExecutionSelectionV3):
        validate_execution_selection_v3(config, selection)
        return
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
                codex_home=(config.codex_runtime.home if role is ExecutionRole.IMPLEMENTER else None),
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


def validate_execution_selection_v3(config: HarnessConfig, selection: ExecutionSelectionV3) -> None:
    if not isinstance(config, HarnessConfig) or not isinstance(selection, ExecutionSelectionV3) or selection.schema_version != SCHEMA_VERSION_V3:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    for name, role, selected in (
        ("planner", ExecutionRole.PLANNER, selection.planner),
        ("reviewer", ExecutionRole.REVIEWER, selection.reviewer),
    ):
        try:
            profile = profile_for_role(config, selected.profile_id, role)
        except ProfileError as exc:
            raise ExecutionSelectionError(f"execution selection {name} profile is unavailable") from exc
        expected = _selected(profile, agent_env_allowlist=_env_allowlist(config, role))
        if selected != expected:
            raise ExecutionSelectionError(f"execution selection {name} profile no longer matches config")
    if not selection.steps:
        raise ExecutionSelectionError("execution selection steps are missing")
    for item in selection.steps:
        try:
            profile = profile_for_role(config, item.implementer.profile_id, ExecutionRole.IMPLEMENTER)
        except ProfileError as exc:
            raise ExecutionSelectionError(f"execution selection {item.step_id} profile is unavailable") from exc
        expected = _selected(profile, agent_env_allowlist=_env_allowlist(config, ExecutionRole.IMPLEMENTER), codex_home=config.codex_runtime.home)
        if item.implementer != expected:
            raise ExecutionSelectionError(f"execution selection {item.step_id} profile no longer matches config")


__all__ = [
    "ExecutionSelectionConflict",
    "ExecutionSelectionError",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V3",
    "ensure_execution_selection",
    "ensure_execution_selection_v3",
    "is_profile_aware_run",
    "parse_execution_selection",
    "parse_execution_selection_v3",
    "read_execution_selection",
    "read_execution_selection_with_sha256",
    "read_execution_selection_v3",
    "read_execution_selection_v3_with_sha256",
    "resolve_execution_selection",
    "resolve_execution_selection_v3",
    "validate_execution_selection",
    "validate_execution_selection_v3",
]
