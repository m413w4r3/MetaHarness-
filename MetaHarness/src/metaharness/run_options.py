"""The immutable, secret-free options captured by every pipeline v2 run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .models import HarnessConfig, RevisionConfig, PlanningConfig, UIConfig, validate_revision_budget
from .profiles import ProfileError, profile_for_role
from .result import atomic_write_text
from .models import ExecutionRole


class RunOptionsError(ValueError):
    """A run-options snapshot is absent, malformed, or incompatible."""


class RunOptionsConflict(RunOptionsError):
    """An immutable run-options artifact already contains different bytes."""


SCHEMA_VERSION = 2
RUN_OPTIONS_NAME = "run_options.json"
REPAIR_SCOPE_POLICIES = frozenset({"auto-bounded", "require-approval", "deny-expansion"})
REPAIR_SCOPE_OVERRIDE_NAME = "repair_scope_override.json"
REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION = 1
REPAIR_SCOPE_OVERRIDE_POLICY = "auto-bounded"
REPAIR_SCOPE_OVERRIDE_REASON = "operator-enabled historical test-scope recovery"
REPAIR_SCOPE_MAX_ADDED_PATHS = 100


@dataclass(frozen=True)
class RepairScopeOverride:
    schema_version: int
    policy: str
    max_added_paths: int
    reason: str

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION:
            raise RunOptionsError("repair scope override schema_version is unsupported")
        if self.policy != REPAIR_SCOPE_OVERRIDE_POLICY:
            raise RunOptionsError("repair scope override policy is invalid")
        if not isinstance(self.max_added_paths, int) or isinstance(self.max_added_paths, bool) or not 0 < self.max_added_paths <= REPAIR_SCOPE_MAX_ADDED_PATHS:
            raise RunOptionsError("repair scope override max_added_paths must be between 1 and 100")
        if self.reason != REPAIR_SCOPE_OVERRIDE_REASON:
            raise RunOptionsError("repair scope override reason is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "max_added_paths": self.max_added_paths,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EffectiveRepairScopePolicy:
    policy: str
    max_added_paths: int
    source: str

    def __post_init__(self) -> None:
        if self.policy not in REPAIR_SCOPE_POLICIES:
            raise RunOptionsError("effective repair scope policy is invalid")
        if not isinstance(self.max_added_paths, int) or isinstance(self.max_added_paths, bool) or self.max_added_paths <= 0:
            raise RunOptionsError("effective repair scope max_added_paths must be greater than zero")
        if self.source not in {"run-options", "operator-override"}:
            raise RunOptionsError("effective repair scope policy source is invalid")


@dataclass(frozen=True)
class RunOptions:
    schema_version: int
    pipeline_version: int
    protocol: str
    decomposition: str
    execution_mode_policy: str
    single_step_max_mutable_paths: int
    staged_step_max_mutable_paths: int
    semantic_revision_enabled: bool
    max_check_repair_attempts: int
    max_review_repair_cycles: int
    planner_profile: str
    default_implementer_profile: str
    check_repair_profile: str | None
    semantic_reviser_profile: str | None
    final_reviewer_profile: str
    repair_scope_policy: str = "auto-bounded"
    repair_scope_max_added_paths: int = 4

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.pipeline_version != 2:
            raise RunOptionsError("run options schema or pipeline version is unsupported")
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
        if not isinstance(self.semantic_revision_enabled, bool):
            raise RunOptionsError("run options semantic_revision_enabled must be boolean")
        for name in ("max_check_repair_attempts", "max_review_repair_cycles"):
            try:
                validate_revision_budget(getattr(self, name), f"run options {name}")
            except ValueError as exc:
                raise RunOptionsError(str(exc)) from None
        for name in ("planner_profile", "default_implementer_profile", "final_reviewer_profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunOptionsError(f"run options {name} is invalid")
        for name in ("check_repair_profile", "semantic_reviser_profile"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RunOptionsError(f"run options {name} is invalid")
        if self.repair_scope_policy not in REPAIR_SCOPE_POLICIES:
            raise RunOptionsError("run options repair_scope_policy is invalid")
        if not isinstance(self.repair_scope_max_added_paths, int) or isinstance(self.repair_scope_max_added_paths, bool) or self.repair_scope_max_added_paths <= 0:
            raise RunOptionsError("run options repair_scope_max_added_paths must be greater than zero")

    @classmethod
    def from_config(cls, config: HarnessConfig, **overrides: Any) -> "RunOptions":
        allowed = {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
            "semantic_revision_enabled", "max_check_repair_attempts",
            "max_review_repair_cycles", "planner_profile", "default_implementer_profile",
            "check_repair_profile", "semantic_reviser_profile", "final_reviewer_profile",
            "repair_scope_policy", "repair_scope_max_added_paths",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise RunOptionsError(f"unknown run option: {sorted(unknown)[0]}")
        values: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": 2,
            "protocol": config.planning.protocol,
            "decomposition": config.planning.decomposition,
            "execution_mode_policy": config.planning.execution_mode_policy,
            "single_step_max_mutable_paths": config.planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": config.planning.staged_step_max_mutable_paths,
            "semantic_revision_enabled": config.revision.enabled,
            "max_check_repair_attempts": config.revision.max_check_repair_attempts,
            "max_review_repair_cycles": config.revision.max_review_repair_cycles,
            "planner_profile": config.ui.default_planner_profile or "legacy-planner",
            "default_implementer_profile": config.ui.default_implementer_profile or "legacy-implementer",
            "check_repair_profile": config.ui.default_repair_profile,
            "semantic_reviser_profile": config.ui.default_reviser_profile,
            "final_reviewer_profile": config.ui.default_reviewer_profile or "legacy-reviewer",
            "repair_scope_policy": "auto-bounded",
            "repair_scope_max_added_paths": 4,
        }
        values.update(overrides)
        result = cls(**values)
        result.validate_profiles(config)
        if result.semantic_revision_enabled and result.semantic_reviser_profile is None:
            raise RunOptionsError("semantic revision requires a semantic reviser profile")
        if (result.max_check_repair_attempts or result.max_review_repair_cycles) and result.check_repair_profile is None:
            raise RunOptionsError("correction budget requires a check-repair profile")
        return result

    def validate_profiles(self, config: HarnessConfig) -> None:
        for name, role in (
            ("planner_profile", ExecutionRole.PLANNER),
            ("default_implementer_profile", ExecutionRole.IMPLEMENTER),
            ("final_reviewer_profile", ExecutionRole.REVIEWER),
            ("semantic_reviser_profile", ExecutionRole.REVISER),
            ("check_repair_profile", ExecutionRole.REPAIR),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                profile_for_role(config, value, role)
            except ProfileError as exc:
                raise RunOptionsError(f"{name} is invalid or incompatible") from exc

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
            },
            "pipeline": {
                "semantic_revision_enabled": self.semantic_revision_enabled,
                "max_check_repair_attempts": self.max_check_repair_attempts,
                "max_review_repair_cycles": self.max_review_repair_cycles,
                "repair_scope_policy": self.repair_scope_policy,
                "repair_scope_max_added_paths": self.repair_scope_max_added_paths,
            },
            "profiles": {
                "planner_profile": self.planner_profile,
                "default_implementer_profile": self.default_implementer_profile,
                "check_repair_profile": self.check_repair_profile,
                "semantic_reviser_profile": self.semantic_reviser_profile,
                "final_reviewer_profile": self.final_reviewer_profile,
            },
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "RunOptions":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "pipeline_version", "planning", "pipeline", "profiles"}:
            raise RunOptionsError("run options schema is invalid")
        planning, pipeline, profiles = value["planning"], value["pipeline"], value["profiles"]
        if not isinstance(planning, Mapping) or set(planning) != {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
        }:
            raise RunOptionsError("run options planning schema is invalid")
        if not isinstance(pipeline, Mapping) or set(pipeline) != {
            "semantic_revision_enabled", "max_check_repair_attempts", "max_review_repair_cycles",
            "repair_scope_policy", "repair_scope_max_added_paths",
        }:
            raise RunOptionsError("run options pipeline schema is invalid")
        if not isinstance(profiles, Mapping) or set(profiles) != {
            "planner_profile", "default_implementer_profile", "check_repair_profile",
            "semantic_reviser_profile", "final_reviewer_profile",
        }:
            raise RunOptionsError("run options profiles schema is invalid")
        try:
            return cls(
                schema_version=value["schema_version"], pipeline_version=value["pipeline_version"],
                **planning, **pipeline, **profiles,
            )
        except (KeyError, TypeError) as exc:
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


def read_run_options_with_sha256(run_dir: str | Path, expected_sha256: str | None = None) -> tuple[RunOptions, str]:
    options, digest, _ = read_run_options_with_sha256_and_raw(run_dir, expected_sha256)
    return options, digest


def write_repair_scope_override(run_dir: str | Path, override: RepairScopeOverride) -> str:
    data = (json.dumps(override.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path = Path(run_dir).expanduser().resolve() / REPAIR_SCOPE_OVERRIDE_NAME
    if path.exists() and path.read_bytes() != data:
        raise RunOptionsConflict("repair_scope_override.json is immutable")
    if not path.exists():
        atomic_write_text(path, data.decode())
    return hashlib.sha256(data).hexdigest()


def read_repair_scope_override(run_dir: str | Path) -> RepairScopeOverride | None:
    path = Path(run_dir).expanduser().resolve() / REPAIR_SCOPE_OVERRIDE_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "policy", "max_added_paths", "reason"}:
            raise RunOptionsError("repair scope override schema is invalid")
        return RepairScopeOverride(**payload)
    except RunOptionsError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise RunOptionsError("repair_scope_override.json is missing or malformed") from exc


def effective_repair_scope_policy(options: RunOptions, *, override: RepairScopeOverride | None = None) -> EffectiveRepairScopePolicy:
    if override is not None:
        return EffectiveRepairScopePolicy(override.policy, override.max_added_paths, "operator-override")
    return EffectiveRepairScopePolicy(options.repair_scope_policy, options.repair_scope_max_added_paths, "run-options")


def effective_run_config(config: HarnessConfig, options: RunOptions) -> HarnessConfig:
    options.validate_profiles(config)
    planning = PlanningConfig(
        protocol=options.protocol, decomposition=options.decomposition,
        single_step_max_mutable_paths=options.single_step_max_mutable_paths,
        staged_step_max_mutable_paths=options.staged_step_max_mutable_paths,
        execution_mode_policy=options.execution_mode_policy,
    )
    ui = replace(
        config.ui,
        default_planner_profile=options.planner_profile,
        default_implementer_profile=options.default_implementer_profile,
        default_reviewer_profile=options.final_reviewer_profile,
        default_reviser_profile=options.semantic_reviser_profile,
        default_repair_profile=options.check_repair_profile,
    )
    revision = RevisionConfig(
        enabled=options.semantic_revision_enabled,
        max_check_repair_attempts=options.max_check_repair_attempts,
        max_review_repair_cycles=options.max_review_repair_cycles,
    )
    return replace(config, planning=planning, ui=ui, revision=revision)


__all__ = [
    "SCHEMA_VERSION", "RUN_OPTIONS_NAME", "REPAIR_SCOPE_OVERRIDE_NAME",
    "REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION", "REPAIR_SCOPE_OVERRIDE_REASON",
    "REPAIR_SCOPE_OVERRIDE_POLICY", "REPAIR_SCOPE_MAX_ADDED_PATHS", "REPAIR_SCOPE_POLICIES",
    "RunOptions", "RunOptionsConflict", "RunOptionsError", "RepairScopeOverride",
    "EffectiveRepairScopePolicy", "canonical_run_options_bytes", "run_options_sha256",
    "write_run_options", "read_run_options_with_sha256", "read_run_options_with_sha256_and_raw",
    "write_repair_scope_override", "read_repair_scope_override",
    "effective_repair_scope_policy", "effective_run_config",
]
