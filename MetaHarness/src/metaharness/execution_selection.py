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
    CycleExecutionSelection,
    ExecutionClass,
    ExecutionRole,
    ExecutionSelection,
    HarnessConfig,
    SelectedProfile,
    StepExecutionSelection,
    ImplementationStep,
    profile_driver_name,
)
from .profiles import ProfileError, profile_execution_fingerprint, profile_for_role
from .step_ids import MAX_STEPS, STEP_ID_RE, step_ids


class ExecutionSelectionError(ValueError):
    """The execution selection is absent, invalid, or no longer matches config."""


class ExecutionSelectionConflict(ExecutionSelectionError):
    """A different execution selection has already been published."""


_FILENAME = "execution_selection.json"
SCHEMA_VERSION = 6
_CYCLE_SCHEMA_VERSION = 2
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
                if role in {ExecutionRole.IMPLEMENTER, ExecutionRole.REPAIR}
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


def _canonical_step_items(step_profile_ids: Mapping[str, str]) -> list[tuple[str, str]]:
    if not isinstance(step_profile_ids, Mapping) or not step_profile_ids:
        raise ExecutionSelectionError("execution selection steps are required")
    items = list(step_profile_ids.items())
    if len(items) > MAX_STEPS:
        raise ExecutionSelectionError(f"execution selection may contain at most {MAX_STEPS} steps")
    if any(
        not isinstance(step_id, str)
        or STEP_ID_RE.fullmatch(step_id) is None
        or not isinstance(profile_id, str)
        or not profile_id.strip()
        for step_id, profile_id in items
    ):
        raise ExecutionSelectionError("execution selection step is invalid")
    items.sort(key=lambda item: int(item[0][1:]))
    ids = [item[0] for item in items]
    if ids != list(step_ids(len(ids))):
        raise ExecutionSelectionError("execution selection step IDs must be contiguous from S01")
    return items


def resolve_execution_selection(
    config: HarnessConfig,
    *,
    planner_profile_id: str,
    plan_steps: tuple[ImplementationStep, ...] | list[ImplementationStep] | None = None,
    step_profile_ids: Mapping[str, str] | None = None,
    check_repair_profile_id: str | None,
    semantic_reviser_profile_id: str | None,
    final_reviewer_profile_id: str,
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
        check_repair=(
            _selected(config, check_repair_profile_id, ExecutionRole.REPAIR)
            if check_repair_profile_id is not None else None
        ),
        semantic_reviser=(
            _selected(config, semantic_reviser_profile_id, ExecutionRole.REVISER)
            if semantic_reviser_profile_id is not None else None
        ),
        final_reviewer=_selected(config, final_reviewer_profile_id, ExecutionRole.REVIEWER),
        check_repair_fallbacks=(
            tuple(_selected(config, profile_id, ExecutionRole.REPAIR)
                  for profile_id in fallbacks.check_repair)
            if check_repair_profile_id is not None else ()
        ),
        semantic_reviser_fallbacks=(
            tuple(_selected(config, profile_id, ExecutionRole.REVISER)
                  for profile_id in fallbacks.semantic_reviser)
            if semantic_reviser_profile_id is not None else ()
        ),
    )


def resolve_cycle_execution_selection(
    config: HarnessConfig,
    *,
    cycle: int,
    plan_steps: tuple[ImplementationStep, ...] | list[ImplementationStep] | None = None,
    step_profile_ids: Mapping[str, str] | None = None,
    fallback_authority: Any | None = None,
) -> CycleExecutionSelection:
    """Freeze implementer profiles for a single review-replan cycle."""

    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 2:
        raise ExecutionSelectionError("cycle execution selection cycle is invalid")
    if not plan_steps:
        raise ExecutionSelectionError("execution plan steps are required")
    if len(plan_steps) > MAX_STEPS:
        raise ExecutionSelectionError(f"cycle execution selection may contain at most {MAX_STEPS} steps")
    overrides = step_profile_ids or {}
    if set(overrides) - {step.id for step in plan_steps}:
        raise ExecutionSelectionError("cycle execution selection contains an unknown step")
    fallbacks = fallback_authority or config.recovery.execution_fallbacks
    steps = tuple(
        StepExecutionSelection(
            step_id=step.id,
            implementer=_selected(
                config,
                overrides.get(step.id) or config.routing.profile_for(step.execution_class),
                ExecutionRole.IMPLEMENTER,
            ),
            execution_class=step.execution_class,
            fallbacks=tuple(
                _selected(config, fallback_id, ExecutionRole.IMPLEMENTER)
                for fallback_id in fallbacks.for_execution_class(step.execution_class.value)
            ),
        )
        for step in sorted(plan_steps, key=lambda item: int(item.id[1:]))
    )
    return CycleExecutionSelection(_CYCLE_SCHEMA_VERSION, cycle, steps)


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
        "check_repair": (
            _selected_payload(selection.check_repair)
            if selection.check_repair is not None else None
        ),
        "semantic_reviser": (
            _selected_payload(selection.semantic_reviser)
            if selection.semantic_reviser is not None else None
        ),
        "final_reviewer": _selected_payload(selection.final_reviewer),
        "check_repair_fallbacks": [
            _selected_payload(profile) for profile in selection.check_repair_fallbacks
        ],
        "semantic_reviser_fallbacks": [
            _selected_payload(profile) for profile in selection.semantic_reviser_fallbacks
        ],
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


def ensure_execution_selection(run_dir: Path, selection: ExecutionSelection) -> ExecutionSelection:
    path = Path(run_dir).expanduser().resolve() / _FILENAME
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


def _cycle_payload(selection: CycleExecutionSelection) -> dict[str, Any]:
    if not isinstance(selection, CycleExecutionSelection) or selection.schema_version != _CYCLE_SCHEMA_VERSION:
        raise ExecutionSelectionError("cycle execution selection schema_version is unsupported")
    if selection.cycle < 2 or not selection.steps:
        raise ExecutionSelectionError("cycle execution selection is invalid")
    return {
        "schema_version": _CYCLE_SCHEMA_VERSION,
        "cycle": selection.cycle,
        "steps": [
            {"step_id": item.step_id, "execution_class": item.execution_class.value,
             "implementer": _selected_payload(item.implementer),
             "fallbacks": [_selected_payload(profile) for profile in item.fallbacks]}
            for item in selection.steps
        ],
    }


def _cycle_path(run_dir: Path, cycle: int) -> Path:
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 2:
        raise ExecutionSelectionError("cycle execution selection cycle is invalid")
    return Path(run_dir).expanduser().resolve() / "cycles" / f"{cycle:03d}" / "correction" / _FILENAME


def ensure_cycle_execution_selection(
    run_dir: Path, selection: CycleExecutionSelection,
) -> CycleExecutionSelection:
    path = _cycle_path(run_dir, selection.cycle)
    content = json.dumps(_cycle_payload(selection), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        durable = parse_cycle_execution_selection(path.read_bytes())
        if durable != selection:
            raise ExecutionSelectionConflict("cycle execution selection is immutable")
        return durable
    if not _publish_exclusive(path, content):
        durable = parse_cycle_execution_selection(path.read_bytes())
        if durable != selection:
            raise ExecutionSelectionConflict("cycle execution selection is immutable")
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
    expected_v5 = {
        "schema_version", "planner", "steps", "check_repair",
        "semantic_reviser", "final_reviewer",
    }
    expected_v6 = expected_v5 | {"check_repair_fallbacks", "semantic_reviser_fallbacks"}
    if not isinstance(payload, dict) or (
        (schema_version == 5 and set(payload) != expected_v5)
        or (schema_version == 6 and set(payload) != expected_v6)
        or schema_version not in {5, 6}
    ):
        raise ExecutionSelectionError("execution selection schema_version is unsupported")
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ExecutionSelectionError("execution selection steps are invalid")
    steps = []
    for item in raw_steps:
        allowed_keys = {"step_id", "execution_class", "implementer"}
        if schema_version == 6:
            allowed_keys.add("fallbacks")
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
        check_repair=(
            _parse_selected(payload["check_repair"], "check_repair")
            if payload["check_repair"] is not None else None
        ),
        semantic_reviser=(
            _parse_selected(payload["semantic_reviser"], "semantic_reviser")
            if payload["semantic_reviser"] is not None else None
        ),
        final_reviewer=_parse_selected(payload["final_reviewer"], "final_reviewer"),
        check_repair_fallbacks=(
            _parse_selected_list(payload.get("check_repair_fallbacks", []), "check_repair_fallbacks")
        ),
        semantic_reviser_fallbacks=(
            _parse_selected_list(payload.get("semantic_reviser_fallbacks", []), "semantic_reviser_fallbacks")
        ),
    )


def parse_cycle_execution_selection(data: bytes) -> CycleExecutionSelection:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionSelectionError("cycle execution selection is missing or invalid") from exc
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or (
        schema_version == 1 and set(payload) != {"schema_version", "cycle", "steps"}
    ) or (
        schema_version == 2 and set(payload) != {"schema_version", "cycle", "steps"}
    ) or schema_version not in {1, 2}:
        raise ExecutionSelectionError("cycle execution selection schema is invalid")
    cycle = payload.get("cycle")
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 2:
        raise ExecutionSelectionError("cycle execution selection cycle is invalid")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ExecutionSelectionError("cycle execution selection steps are invalid")
    steps = []
    for item in raw_steps:
        allowed_keys = {"step_id", "execution_class", "implementer"}
        if schema_version == 2:
            allowed_keys.add("fallbacks")
        if not isinstance(item, dict) or set(item) != allowed_keys:
            raise ExecutionSelectionError("cycle execution selection step is invalid")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or STEP_ID_RE.fullmatch(step_id) is None:
            raise ExecutionSelectionError("cycle execution selection step ID is invalid")
        try:
            execution_class = ExecutionClass(item["execution_class"])
        except (TypeError, ValueError) as exc:
            raise ExecutionSelectionError("cycle execution selection execution class is invalid") from exc
        steps.append(StepExecutionSelection(
            step_id, _parse_selected(item["implementer"], f"cycle step {step_id}"), execution_class,
            _parse_selected_list(item.get("fallbacks", []), f"cycle step {step_id} fallbacks"),
        ))
    if [item.step_id for item in steps] != list(step_ids(len(steps))):
        raise ExecutionSelectionError("cycle execution selection step IDs are not contiguous")
    return CycleExecutionSelection(schema_version, cycle, tuple(steps))


def read_cycle_execution_selection_with_sha256(
    run_dir: Path, cycle: int,
) -> tuple[CycleExecutionSelection, str]:
    path = _cycle_path(run_dir, cycle)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSelectionError("cycle execution selection is missing or invalid") from exc
    return parse_cycle_execution_selection(data), hashlib.sha256(data).hexdigest()


def read_cycle_execution_selection(run_dir: Path, cycle: int) -> CycleExecutionSelection:
    return read_cycle_execution_selection_with_sha256(run_dir, cycle)[0]


def validate_cycle_execution_selection(
    config: HarnessConfig, selection: CycleExecutionSelection,
) -> None:
    if not isinstance(selection, CycleExecutionSelection) or selection.schema_version not in {1, _CYCLE_SCHEMA_VERSION}:
        raise ExecutionSelectionError("cycle execution selection is invalid")
    for item in selection.steps:
        try:
            expected = _selected(config, item.implementer.profile_id, ExecutionRole.IMPLEMENTER)
        except ProfileError as exc:
            raise ExecutionSelectionError(
                f"cycle step {item.step_id} implementer profile is unavailable"
            ) from exc
        if item.implementer != expected:
            raise ExecutionSelectionError(
                f"cycle step {item.step_id} implementer profile no longer matches config"
            )
        if selection.schema_version == _CYCLE_SCHEMA_VERSION:
            expected_ids = config.recovery.execution_fallbacks.for_execution_class(
                item.execution_class.value
            )
            if tuple(profile.profile_id for profile in item.fallbacks) != expected_ids:
                raise ExecutionSelectionError(
                    f"cycle step {item.step_id} fallback authority changed"
                )
            for profile in item.fallbacks:
                try:
                    selected = _selected(config, profile.profile_id, ExecutionRole.IMPLEMENTER)
                except ProfileError as exc:
                    raise ExecutionSelectionError(
                        f"cycle step {item.step_id} fallback profile is unavailable"
                    ) from exc
                if profile != selected:
                    raise ExecutionSelectionError(
                        f"cycle step {item.step_id} fallback profile no longer matches config"
                    )


def read_execution_selection_with_sha256(run_dir: Path) -> tuple[ExecutionSelection, str]:
    path = Path(run_dir).expanduser().resolve() / _FILENAME
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExecutionSelectionError("execution selection is missing or invalid") from exc
    return parse_execution_selection(data), hashlib.sha256(data).hexdigest()


def read_execution_selection(run_dir: Path) -> ExecutionSelection:
    return read_execution_selection_with_sha256(run_dir)[0]


def validate_execution_selection(config: HarnessConfig, selection: ExecutionSelection) -> None:
    if not isinstance(config, HarnessConfig) or not isinstance(selection, ExecutionSelection):
        raise ExecutionSelectionError("execution selection is invalid")
    if selection.schema_version not in {5, SCHEMA_VERSION}:
        raise ExecutionSelectionError("execution selection schema_version is unsupported")

    def check(name: str, selected: SelectedProfile, role: ExecutionRole) -> None:
        try:
            expected = _selected(config, selected.profile_id, role)
        except ProfileError as exc:
            raise ExecutionSelectionError(f"execution selection {name} profile is unavailable") from exc
        if selected != expected:
            raise ExecutionSelectionError(f"execution selection {name} profile no longer matches config")

    check("planner", selection.planner, ExecutionRole.PLANNER)
    check("final_reviewer", selection.final_reviewer, ExecutionRole.REVIEWER)
    if selection.check_repair is not None:
        check("check_repair", selection.check_repair, ExecutionRole.REPAIR)
    if selection.semantic_reviser is not None:
        check("semantic_reviser", selection.semantic_reviser, ExecutionRole.REVISER)
    for item in selection.steps:
        check(f"step {item.step_id}", item.implementer, ExecutionRole.IMPLEMENTER)
        if selection.schema_version == SCHEMA_VERSION:
            expected_ids = config.recovery.execution_fallbacks.for_execution_class(
                item.execution_class.value
            )
            if tuple(profile.profile_id for profile in item.fallbacks) != expected_ids:
                raise ExecutionSelectionError(f"step {item.step_id} fallback authority changed")
            for profile in item.fallbacks:
                check(f"step {item.step_id} fallback", profile, ExecutionRole.IMPLEMENTER)
    if selection.schema_version == SCHEMA_VERSION:
        for name, selected_profiles, configured_ids, role in (
            ("check_repair", selection.check_repair_fallbacks,
             config.recovery.execution_fallbacks.check_repair, ExecutionRole.REPAIR),
            ("semantic_reviser", selection.semantic_reviser_fallbacks,
             config.recovery.execution_fallbacks.semantic_reviser, ExecutionRole.REVISER),
        ):
            if tuple(profile.profile_id for profile in selected_profiles) != configured_ids:
                raise ExecutionSelectionError(f"{name} fallback authority changed")
            for profile in selected_profiles:
                check(f"{name} fallback", profile, role)


__all__ = [
    "CycleExecutionSelection",
    "ExecutionSelection",
    "ExecutionSelectionConflict",
    "ExecutionSelectionError",
    "SCHEMA_VERSION",
    "ensure_execution_selection",
    "ensure_cycle_execution_selection",
    "is_profile_aware_run",
    "parse_execution_selection",
    "parse_cycle_execution_selection",
    "read_execution_selection",
    "read_execution_selection_with_sha256",
    "read_cycle_execution_selection",
    "read_cycle_execution_selection_with_sha256",
    "resolve_cycle_execution_selection",
    "resolve_execution_selection",
    "validate_execution_selection",
    "validate_cycle_execution_selection",
]
