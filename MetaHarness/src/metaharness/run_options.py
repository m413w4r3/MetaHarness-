"""The immutable, secret-free options captured by every pipeline v2 run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .models import HarnessConfig, RevisionConfig, PlanningConfig, RoutingConfig, validate_revision_budget
from .recovery_policy import ExecutionFallbacks, RecoveryBudgets
from .profiles import ProfileError, profile_for_role
from .result import atomic_write_text
from .models import ExecutionRole


class RunOptionsError(ValueError):
    """A run-options snapshot is absent, malformed, or incompatible."""


class RunOptionsConflict(RunOptionsError):
    """An immutable run-options artifact already contains different bytes."""


SCHEMA_VERSION = 6
# Deterministic refusal code for a snapshot that is not the current schema.
RUN_SCHEMA_UNSUPPORTED = "RUN_SCHEMA_UNSUPPORTED"
RUN_OPTIONS_NAME = "run_options.json"
_TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "pipeline_version", "planning", "pipeline", "profiles", "recovery",
})
_PLANNING_FIELDS = frozenset({
    "protocol", "decomposition", "execution_mode_policy",
    "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
    "max_steps_per_plan", "max_read_paths_per_step", "max_step_contract_chars",
    "max_preapproval_corrections",
})
_PIPELINE_FIELDS = frozenset({
    "max_step_contract_repairs",
})
_PROFILE_FIELDS = frozenset({
    "planner_profile", "mechanical_profile", "reasoning_profile", "agentic_profile",
    "audit_profile",
})
_RECOVERY_FIELDS = frozenset({
    "max_transient_attempts", "max_executor_fallbacks",
    "max_check_infra_retries",
    "max_workspace_setup_retries", "max_contract_repair_output_corrections",
    "max_contract_repair_planner_restarts",
})
_FALLBACK_FIELDS = frozenset({
    "mechanical", "reasoning", "agentic",
})


def _require_exact_keys(payload: Any, expected: frozenset[str], where: str) -> None:
    """One exact key set per section: a missing or extra key is a schema error."""

    if not isinstance(payload, Mapping):
        raise RunOptionsError(f"{where} schema is invalid")
    keys = set(payload)
    missing = sorted(expected - keys)
    if missing:
        raise RunOptionsError(f"{where} is missing {missing[0]}")
    unknown = sorted(keys - expected)
    if unknown:
        raise RunOptionsError(f"{where} has unknown key {unknown[0]}")


def _fallback_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise RunOptionsError("run options recovery.execution_fallbacks must be arrays of profile IDs")
    if any(not isinstance(profile_id, str) for profile_id in value):
        raise RunOptionsError("run options recovery.execution_fallbacks must be arrays of profile IDs")
    return tuple(value)


@dataclass(frozen=True)
class RunOptions:
    schema_version: int
    pipeline_version: int
    protocol: str
    decomposition: str
    execution_mode_policy: str
    single_step_max_mutable_paths: int
    staged_step_max_mutable_paths: int
    planner_profile: str
    max_step_contract_repairs: int = 2
    mechanical_profile: str = ""
    reasoning_profile: str = ""
    agentic_profile: str = ""
    audit_profile: str = ""
    max_steps_per_plan: int = 12
    max_read_paths_per_step: int = 8
    max_step_contract_chars: int = 9000
    max_preapproval_corrections: int = 2
    recovery: RecoveryBudgets = RecoveryBudgets()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RunOptionsError(
                f"{RUN_SCHEMA_UNSUPPORTED}: run options schema_version "
                f"{self.schema_version!r} is not {SCHEMA_VERSION}"
            )
        if self.pipeline_version != 2:
            raise RunOptionsError("run options pipeline_version must be 2")
        if self.protocol != "v2":
            raise RunOptionsError("run options protocol must be v2")
        if self.decomposition not in {"balanced", "aggressive"}:
            raise RunOptionsError("run options decomposition is invalid")
        if self.execution_mode_policy not in {"auto", "require-staged"}:
            raise RunOptionsError("run options execution_mode_policy is invalid")
        for name in ("single_step_max_mutable_paths", "staged_step_max_mutable_paths"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RunOptionsError(f"run options {name} must be greater than zero")
        for name in ("max_steps_per_plan", "max_read_paths_per_step", "max_step_contract_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RunOptionsError(f"run options {name} must be greater than zero")
        if self.max_steps_per_plan > 99:
            raise RunOptionsError("run options max_steps_per_plan must not exceed 99")
        try:
            validate_revision_budget(self.max_preapproval_corrections, "run options max_preapproval_corrections")
        except ValueError as exc:
            raise RunOptionsError(str(exc)) from None
        if not isinstance(self.recovery, RecoveryBudgets):
            raise RunOptionsError("run options recovery budgets are invalid")
        try:
            validate_revision_budget(
                self.max_step_contract_repairs, "run options max_step_contract_repairs"
            )
        except ValueError as exc:
            raise RunOptionsError(str(exc)) from None
        for name in (
            "planner_profile", "mechanical_profile", "reasoning_profile",
            "agentic_profile", "audit_profile",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunOptionsError(f"run options {name} is invalid")

    @classmethod
    def from_config(cls, config: HarnessConfig, **overrides: Any) -> "RunOptions":
        allowed = {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
            "planner_profile", "mechanical_profile",
            "reasoning_profile", "agentic_profile", "audit_profile",
            "max_steps_per_plan", "max_read_paths_per_step", "max_step_contract_chars",
            "max_preapproval_corrections", "max_step_contract_repairs",
            "recovery",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise RunOptionsError(f"unknown run option: {sorted(unknown)[0]}")
        route_profiles = {
            "mechanical_profile": config.routing.mechanical_profile,
            "reasoning_profile": config.routing.reasoning_profile,
            "agentic_profile": config.routing.agentic_profile,
        }
        values: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": 2,
            "protocol": config.planning.protocol,
            "decomposition": config.planning.decomposition,
            "execution_mode_policy": config.planning.execution_mode_policy,
            "single_step_max_mutable_paths": config.planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": config.planning.staged_step_max_mutable_paths,
            "max_steps_per_plan": config.planning.max_steps_per_plan,
            "max_read_paths_per_step": config.planning.max_read_paths_per_step,
            "max_step_contract_chars": config.planning.max_step_contract_chars,
            "max_preapproval_corrections": config.planning.max_preapproval_corrections,
            "max_step_contract_repairs": config.revision.max_step_contract_repairs,
            "planner_profile": config.ui.default_planner_profile,
            **route_profiles,
            "audit_profile": config.ui.default_audit_profile,
            "recovery": config.recovery,
        }
        values.update(overrides)
        result = cls(**values)
        result.validate_profiles(config)
        return result

    def validate_profiles(self, config: HarnessConfig) -> None:
        for name, role in (
            ("planner_profile", ExecutionRole.PLANNER),
            ("mechanical_profile", ExecutionRole.IMPLEMENTER),
            ("reasoning_profile", ExecutionRole.IMPLEMENTER),
            ("agentic_profile", ExecutionRole.IMPLEMENTER),
            ("audit_profile", ExecutionRole.AUDITOR),
        ):
            try:
                profile_for_role(config, getattr(self, name), role)
            except ProfileError as exc:
                raise RunOptionsError(f"{name} is invalid or incompatible") from exc
        for execution_class in ("mechanical", "reasoning", "agentic"):
            fallback_ids = self.recovery.execution_fallbacks.for_execution_class(execution_class)
            if getattr(self, f"{execution_class}_profile") in fallback_ids:
                raise RunOptionsError(
                    f"recovery.execution_fallbacks.{execution_class} repeats the primary profile"
                )
            for profile_id in fallback_ids:
                try:
                    profile_for_role(config, profile_id, ExecutionRole.IMPLEMENTER)
                except ProfileError as exc:
                    raise RunOptionsError(
                        f"recovery.execution_fallbacks.{execution_class} contains an incompatible profile"
                    ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "planning": {
                "protocol": self.protocol,
                "decomposition": self.decomposition,
                "execution_mode_policy": self.execution_mode_policy,
                "single_step_max_mutable_paths": self.single_step_max_mutable_paths,
                "staged_step_max_mutable_paths": self.staged_step_max_mutable_paths,
                "max_steps_per_plan": self.max_steps_per_plan,
                "max_read_paths_per_step": self.max_read_paths_per_step,
                "max_step_contract_chars": self.max_step_contract_chars,
                "max_preapproval_corrections": self.max_preapproval_corrections,
            },
            "pipeline": {
                "max_step_contract_repairs": self.max_step_contract_repairs,
            },
            "recovery": asdict(self.recovery),
            "profiles": {
                "planner_profile": self.planner_profile,
                "mechanical_profile": self.mechanical_profile,
                "reasoning_profile": self.reasoning_profile,
                "agentic_profile": self.agentic_profile,
                "audit_profile": self.audit_profile,
            },
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "RunOptions":
        """Read the one current snapshot shape; nothing is migrated.

        The schema 5 -> 6 break removed the former revision and gate-repair
        surfaces: every older snapshot is refused, never converted.
        """

        if not isinstance(value, Mapping):
            raise RunOptionsError("run options schema is invalid")
        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
            raise RunOptionsError(
                f"{RUN_SCHEMA_UNSUPPORTED}: run options schema_version "
                f"{schema_version!r} is not {SCHEMA_VERSION}"
            )
        _require_exact_keys(value, _TOP_LEVEL_FIELDS, "run options")
        planning, pipeline, profiles, recovery = (
            value["planning"], value["pipeline"], value["profiles"], value["recovery"],
        )
        _require_exact_keys(planning, _PLANNING_FIELDS, "run options planning")
        _require_exact_keys(pipeline, _PIPELINE_FIELDS, "run options pipeline")
        _require_exact_keys(profiles, _PROFILE_FIELDS, "run options profiles")
        _require_exact_keys(
            recovery, _RECOVERY_FIELDS | {"execution_fallbacks"}, "run options recovery",
        )
        fallbacks = recovery["execution_fallbacks"]
        _require_exact_keys(
            fallbacks, _FALLBACK_FIELDS, "run options recovery.execution_fallbacks",
        )
        normalized_recovery: dict[str, Any] = {key: recovery[key] for key in _RECOVERY_FIELDS}
        normalized_recovery["execution_fallbacks"] = ExecutionFallbacks(
            **{key: _fallback_ids(fallbacks[key]) for key in _FALLBACK_FIELDS}
        )
        try:
            return cls(
                schema_version=schema_version, pipeline_version=value["pipeline_version"],
                **planning, **pipeline, **profiles,
                recovery=RecoveryBudgets(**normalized_recovery),
            )
        except RunOptionsError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RunOptionsError("run options schema is invalid") from exc


def canonical_run_options_bytes(options: RunOptions) -> bytes:
    return (json.dumps(options.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def run_options_sha256(options: RunOptions) -> str:
    return hashlib.sha256(canonical_run_options_bytes(options)).hexdigest()


def write_run_options(run_dir: str | Path, options: RunOptions) -> str:
    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    data = canonical_run_options_bytes(options)
    if path.exists() and path.read_bytes() != data:
        raise RunOptionsConflict("run_options.json is immutable")
    if not path.exists():
        atomic_write_text(path, data.decode())
    return hashlib.sha256(data).hexdigest()


def read_run_options_with_sha256_and_raw(
    run_dir: str | Path, expected_sha256: str | None = None,
) -> tuple[RunOptions, str, Mapping[str, Any]]:
    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    try:
        data = path.read_bytes()
        raw = json.loads(data.decode(), object_pairs_hook=dict)
        options = RunOptions.from_mapping(raw)
    except RunOptionsError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunOptionsError("run_options.json is missing or malformed") from exc
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunOptionsError("run_options.json hash mismatch")
    return options, digest, raw


def read_run_options_for_state(
    run_dir: str | Path, state: Mapping[str, Any],
) -> tuple[RunOptions, str]:
    """Read the frozen options of an existing run, bound to its state hash."""

    expected = state.get("run_options_sha256")
    if not isinstance(expected, str) or not expected:
        raise RunOptionsError("run options hash is missing from the run state")
    return read_run_options_with_sha256(run_dir, expected)


def read_run_options_with_sha256(run_dir: str | Path, expected_sha256: str | None = None) -> tuple[RunOptions, str]:
    options, digest, _ = read_run_options_with_sha256_and_raw(run_dir, expected_sha256)
    return options, digest


def effective_run_config(config: HarnessConfig, options: RunOptions) -> HarnessConfig:
    options.validate_profiles(config)
    planning = PlanningConfig(
        protocol=options.protocol, decomposition=options.decomposition,
        single_step_max_mutable_paths=options.single_step_max_mutable_paths,
        staged_step_max_mutable_paths=options.staged_step_max_mutable_paths,
        execution_mode_policy=options.execution_mode_policy,
        max_steps_per_plan=options.max_steps_per_plan,
        max_read_paths_per_step=options.max_read_paths_per_step,
        max_step_contract_chars=options.max_step_contract_chars,
        max_preapproval_corrections=options.max_preapproval_corrections,
    )
    ui = replace(
        config.ui,
        default_planner_profile=options.planner_profile,
        default_audit_profile=options.audit_profile,
    )
    routing = RoutingConfig(
        mechanical_profile=options.mechanical_profile,
        reasoning_profile=options.reasoning_profile,
        agentic_profile=options.agentic_profile,
    )
    revision = RevisionConfig(
        max_step_contract_repairs=options.max_step_contract_repairs,
    )
    return replace(
        config, planning=planning, ui=ui, routing=routing, revision=revision,
        recovery=options.recovery,
    )


__all__ = [
    "SCHEMA_VERSION", "RUN_SCHEMA_UNSUPPORTED", "RUN_OPTIONS_NAME",
    "RunOptions", "RunOptionsConflict", "RunOptionsError",
    "canonical_run_options_bytes", "run_options_sha256",
    "write_run_options", "read_run_options_for_state", "read_run_options_with_sha256",
    "read_run_options_with_sha256_and_raw",
    "effective_run_config",
]
