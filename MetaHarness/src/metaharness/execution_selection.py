"""The immutable, role-shaped execution authority for pipeline v2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ExecutionClass,
    ExecutionRole,
    ExecutionSelection,
    HarnessConfig,
    SelectedProfile,
    StepExecutionSelection,
    ImplementationStep,
    profile_driver_name,
)
from .planning.artifacts import iteration_dir
from .profiles import ProfileError, profile_execution_fingerprint, profile_for_role
from .run_options import RUN_SCHEMA_UNSUPPORTED
from .step_ids import MAX_STEPS, STEP_ID_RE, step_ids


class ExecutionSelectionError(ValueError):
    """The execution selection is absent, invalid, or no longer matches config."""

    code = "EXECUTION_SELECTION_INVALID"


class ExecutionSelectionConflict(ExecutionSelectionError):
    """A different execution selection has already been published."""


_FILENAME = "execution_selection.json"
SCHEMA_VERSION = 7
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_FIELDS = frozenset({
    "profile_id", "driver", "provider", "model", "effort", "selection_mode",
    "config_sha256", "sandbox", "permission_mode",
})


def is_profile_aware_run(state: Mapping[str, Any]) -> bool:
    execution = state.get("execution") if isinstance(state, Mapping) else None
    planner = execution.get("planner") if isinstance(execution, Mapping) else None
    return isinstance(planner, Mapping) and isinstance(planner.get("profile_id"), str) and bool(
        planner["profile_id"].strip()
    )


def _selected(config: HarnessConfig, profile_id: str, role: ExecutionRole) -> SelectedProfile:
    profile = profile_for_role(config, profile_id, role)
    return SelectedProfile(
        profile_id=profile.id,
        driver=profile_driver_name(profile.driver),
        provider=profile.provider,
        model=profile.model,
        effort=profile.effort,
        selection_mode=profile.selection_mode.value,
        sandbox=profile.sandbox,
        permission_mode=profile.permission_mode,
        config_sha256=profile_execution_fingerprint(
            profile,
            agent_env_allowlist=(
                tuple(config.codex_runtime.env_allowlist)
                if role is ExecutionRole.IMPLEMENTER
                else ()
            ),
            codex_home=(config.codex_runtime.home if profile.driver == "codex" else None),
            claude_config_home=(
                config.claude_runtime.home if profile.driver == "claude-code" else None
            ),
            provider_base_url=(
                config.codex_providers[profile.provider].base_url
                if profile.driver == "codex" and profile.provider in config.codex_providers
                else None
            ),
            provider_wire_api=(
                config.codex_providers[profile.provider].wire_api
                if profile.driver == "codex" and profile.provider in config.codex_providers
                else None
            ),
            provider_api_key_env=(
                config.codex_providers[profile.provider].api_key_env
                if profile.driver == "codex" and profile.provider in config.codex_providers
                else None
            ),
        ),
    )


def resolve_execution_selection(
    config: HarnessConfig,
    *,
    planner_profile_id: str,
    plan_steps: tuple[ImplementationStep, ...] | list[ImplementationStep] | None = None,
    step_profile_ids: Mapping[str, str] | None = None,
    audit_profile_id: str,
    fallback_authority: Any | None = None,
) -> ExecutionSelection:
    """Freeze every role used by the generic v2 state machine."""

    if not plan_steps:
        raise ExecutionSelectionError("execution plan steps are required")
    if len(plan_steps) > MAX_STEPS:
        raise ExecutionSelectionError(f"execution selection may contain at most {MAX_STEPS} steps")
    override_ids = step_profile_ids or {}
    step_items = {}
    for step in plan_steps:
        if not isinstance(step, ImplementationStep):
            raise ExecutionSelectionError("execution plan step is invalid")
        if step.id in step_items:
            raise ExecutionSelectionError("execution plan step IDs must be unique")
        profile_id = override_ids.get(step.id) or config.routing.profile_for(step.execution_class)
        step_items[step.id] = (step.execution_class, profile_id)
    if set(override_ids) - set(step_items):
        raise ExecutionSelectionError("execution selection contains an unknown step")
    fallbacks = fallback_authority or config.recovery.execution_fallbacks
    steps = tuple(
        StepExecutionSelection(
            step_id=step_id,
            implementer=_selected(config, profile_id, ExecutionRole.IMPLEMENTER),
            execution_class=execution_class,
            fallbacks=tuple(
                _selected(config, fallback_id, ExecutionRole.IMPLEMENTER)
                for fallback_id in fallbacks.for_execution_class(execution_class.value)
            ),
        )
        for step_id, (execution_class, profile_id) in sorted(
            step_items.items(), key=lambda item: int(item[0][1:])
        )
    )
    return ExecutionSelection(
        schema_version=SCHEMA_VERSION,
        planner=_selected(config, planner_profile_id, ExecutionRole.PLANNER),
        steps=steps,
        audit=_selected(config, audit_profile_id, ExecutionRole.AUDITOR),
    )



def _selected_payload(value: SelectedProfile) -> dict[str, Any]:
    if not isinstance(value, SelectedProfile) or value.config_sha256 is None:
        raise ExecutionSelectionError("selected profile is incomplete")
    return {
        "profile_id": value.profile_id,
        "driver": value.driver,
        "provider": value.provider,
        "model": value.model,
        "effort": value.effort,
        "selection_mode": value.selection_mode,
        "config_sha256": value.config_sha256,
        "sandbox": value.sandbox,
        "permission_mode": value.permission_mode,
    }


def _payload(selection: ExecutionSelection) -> dict[str, Any]:
    if not isinstance(selection, ExecutionSelection) or selection.schema_version != SCHEMA_VERSION:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    if not selection.steps:
        raise ExecutionSelectionError("execution selection steps are missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "planner": _selected_payload(selection.planner),
        "steps": [
            {"step_id": item.step_id, "execution_class": item.execution_class.value,
             "implementer": _selected_payload(item.implementer),
             "fallbacks": [_selected_payload(profile) for profile in item.fallbacks]}
            for item in selection.steps
        ],
        "audit": _selected_payload(selection.audit),
    }


def _publish_exclusive(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _unsupported_global_selection_error() -> ExecutionSelectionError:
    error = ExecutionSelectionError(
        f"{RUN_SCHEMA_UNSUPPORTED}: global execution selection is unsupported"
    )
    error.code = RUN_SCHEMA_UNSUPPORTED
    return error


def ensure_execution_selection(
    run_dir: Path, selection: ExecutionSelection, *, iteration: int = 1,
) -> ExecutionSelection:
    directory = Path(run_dir).expanduser().resolve()
    if (directory / _FILENAME).is_file():
        raise _unsupported_global_selection_error()
    path = iteration_dir(directory, iteration) / _FILENAME
    content = json.dumps(_payload(selection), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            durable = parse_execution_selection(path.read_bytes())
        except OSError as exc:
            raise ExecutionSelectionError("execution selection is unreadable") from exc
        if durable != selection:
            raise ExecutionSelectionConflict("execution selection is immutable")
        return durable
    if not _publish_exclusive(path, content):
        durable = parse_execution_selection(path.read_bytes())
        if durable != selection:
            raise ExecutionSelectionConflict("execution selection is immutable")
        return durable
    return selection



def _parse_selected(value: Any, name: str) -> SelectedProfile:
    if not isinstance(value, dict) or set(value) != _PROFILE_FIELDS:
        raise ExecutionSelectionError(f"execution selection {name} is invalid")
    if any(
        not isinstance(value.get(key), (str, type(None)))
        for key in _PROFILE_FIELDS
    ):
        raise ExecutionSelectionError(f"execution selection {name} is invalid")
    if not isinstance(value["profile_id"], str) or not value["profile_id"].strip():
        raise ExecutionSelectionError(f"execution selection {name} is invalid")
    digest = value["config_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ExecutionSelectionError(f"execution selection {name} fingerprint is invalid")
    return SelectedProfile(
        profile_id=value["profile_id"], driver=value["driver"], provider=value["provider"],
        model=value["model"], effort=value["effort"], selection_mode=value["selection_mode"],
        config_sha256=digest, sandbox=value["sandbox"], permission_mode=value["permission_mode"],
    )


def _parse_selected_list(value: Any, name: str) -> tuple[SelectedProfile, ...]:
    if not isinstance(value, list) or len(value) > 10:
        raise ExecutionSelectionError(f"execution selection {name} is invalid")
    profiles = tuple(_parse_selected(item, name) for item in value)
    ids = [item.profile_id for item in profiles]
    if len(ids) != len(set(ids)):
        raise ExecutionSelectionError(f"execution selection {name} contains duplicates")
    return profiles


def parse_execution_selection(data: bytes) -> ExecutionSelection:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    expected = {"schema_version", "planner", "steps", "audit"}
    if not isinstance(payload, dict) or schema_version != SCHEMA_VERSION or set(payload) != expected:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ExecutionSelectionError("execution selection steps are invalid")
    steps = []
    for item in raw_steps:
        allowed_keys = {"step_id", "execution_class", "implementer", "fallbacks"}
        if not isinstance(item, dict) or set(item) != allowed_keys:
            raise ExecutionSelectionError("execution selection step is invalid")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or STEP_ID_RE.fullmatch(step_id) is None:
            raise ExecutionSelectionError("execution selection step ID is invalid")
        try:
            execution_class = ExecutionClass(item["execution_class"])
        except (TypeError, ValueError) as exc:
            raise ExecutionSelectionError("execution selection execution class is invalid") from exc
        steps.append(StepExecutionSelection(
            step_id, _parse_selected(item["implementer"], f"step {step_id}"), execution_class,
            _parse_selected_list(item.get("fallbacks", []), f"step {step_id} fallbacks"),
        ))
    if [item.step_id for item in steps] != list(step_ids(len(steps))):
        raise ExecutionSelectionError("execution selection step IDs are not contiguous")
    return ExecutionSelection(
        schema_version=schema_version,
        planner=_parse_selected(payload["planner"], "planner"),
        steps=tuple(steps),
        audit=_parse_selected(payload["audit"], "audit"),
    )



def read_execution_selection_with_sha256(
    run_dir: Path, *, iteration: int = 1,
) -> tuple[ExecutionSelection, str]:
    directory = Path(run_dir).expanduser().resolve()
    if (directory / _FILENAME).is_file():
        raise _unsupported_global_selection_error()
    path = iteration_dir(directory, iteration) / _FILENAME
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    except OSError as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    return parse_execution_selection(data), hashlib.sha256(data).hexdigest()


def read_execution_selection(run_dir: Path, *, iteration: int = 1) -> ExecutionSelection:
    return read_execution_selection_with_sha256(run_dir, iteration=iteration)[0]


def validate_execution_selection(config: HarnessConfig, selection: ExecutionSelection) -> None:
    if not isinstance(config, HarnessConfig) or not isinstance(selection, ExecutionSelection):
        raise ExecutionSelectionError("execution selection is invalid")
    if selection.schema_version != SCHEMA_VERSION:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")

    def check(name: str, selected: SelectedProfile, role: ExecutionRole) -> None:
        try:
            expected = _selected(config, selected.profile_id, role)
        except ProfileError as exc:
            raise ExecutionSelectionError(f"execution selection {name} profile is unavailable") from exc
        if selected != expected:
            raise ExecutionSelectionError(f"execution selection {name} profile no longer matches config")

    check("planner", selection.planner, ExecutionRole.PLANNER)
    check("audit", selection.audit, ExecutionRole.AUDITOR)
    for item in selection.steps:
        check(f"step {item.step_id}", item.implementer, ExecutionRole.IMPLEMENTER)
        expected_ids = config.recovery.execution_fallbacks.for_execution_class(
            item.execution_class.value
        )
        if tuple(profile.profile_id for profile in item.fallbacks) != expected_ids:
            raise ExecutionSelectionError(f"step {item.step_id} fallback authority changed")
        for profile in item.fallbacks:
            check(f"step {item.step_id} fallback", profile, ExecutionRole.IMPLEMENTER)


__all__ = [
    "ExecutionSelection",
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
