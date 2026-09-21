"""Durable, secret-free options captured when a run is created."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ExecutionRole,
    HarnessConfig,
    validate_revision_budget,
)
from .profiles import ProfileError, profile_for_role
from .result import atomic_write_text


class RunOptionsError(ValueError):
    """A run-options snapshot is absent, malformed, or incompatible."""


class RunOptionsConflict(RunOptionsError):
    """An immutable run-options artifact already contains different bytes."""


SCHEMA_VERSION = 2
HISTORICAL_SCHEMA_VERSION = 1
RUN_OPTIONS_NAME = "run_options.json"
REPAIR_SCOPE_POLICIES = frozenset({"auto-bounded", "require-approval", "deny-expansion"})
REPAIR_SCOPE_OVERRIDE_NAME = "repair_scope_override.json"
REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION = 1
REPAIR_SCOPE_OVERRIDE_POLICY = "auto-bounded"
REPAIR_SCOPE_OVERRIDE_REASON = "operator-enabled historical test-scope recovery"
REPAIR_SCOPE_MAX_ADDED_PATHS = 100


@dataclass(frozen=True)
class RepairScopeOverride:
    """The explicit operator opt-in for bounded expansion of an old run."""

    schema_version: int
    policy: str
    max_added_paths: int
    reason: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION
        ):
            raise RunOptionsError("repair scope override schema_version is unsupported")
        if self.policy != REPAIR_SCOPE_OVERRIDE_POLICY:
            raise RunOptionsError("repair scope override policy is invalid")
        if (
            isinstance(self.max_added_paths, bool)
            or not isinstance(self.max_added_paths, int)
            or not 0 < self.max_added_paths <= REPAIR_SCOPE_MAX_ADDED_PATHS
        ):
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
    """The scope policy actually authoritative for orchestration."""

    policy: str
    max_added_paths: int
    source: str

    def __post_init__(self) -> None:
        if self.policy not in REPAIR_SCOPE_POLICIES:
            raise RunOptionsError("effective repair scope policy is invalid")
        if (
            isinstance(self.max_added_paths, bool)
            or not isinstance(self.max_added_paths, int)
            or self.max_added_paths <= 0
        ):
            raise RunOptionsError("effective repair scope max_added_paths must be greater than zero")
        if self.source not in {"run-options", "historical-default", "operator-override"}:
            raise RunOptionsError("effective repair scope policy source is invalid")


@dataclass(frozen=True)
class RunOptions:
    """Immutable per-run options, with schema 1 read compatibility."""

    schema_version: int
    protocol: str
    decomposition: str
    execution_mode_policy: str
    single_step_max_mutable_paths: int
    staged_step_max_mutable_paths: int
    pipeline_version: int = 2
    semantic_revision_enabled: bool = False
    max_check_repair_attempts: int = 0
    max_review_repair_cycles: int = 0
    planner_profile: str = ""
    default_implementer_profile: str = ""
    check_repair_profile: str | None = None
    semantic_reviser_profile: str | None = None
    final_reviewer_profile: str = ""
    repair_scope_policy: str = "auto-bounded"
    repair_scope_max_added_paths: int = 4

    # Constructor/read aliases for callers that still build historical v1
    # snapshots directly.  They are never emitted by a new schema-2 artifact.
    claude_revision_enabled: bool | None = None
    repair_cycles: int | None = None
    reviewer_profile: str | None = None
    reviser_profile: str | None = None
    repair_profile: str | None = None

    def __post_init__(self) -> None:
        if self.claude_revision_enabled is not None:
            object.__setattr__(self, "semantic_revision_enabled", self.claude_revision_enabled)
        if self.repair_cycles is not None:
            object.__setattr__(self, "max_review_repair_cycles", self.repair_cycles)
        if self.reviewer_profile is not None:
            object.__setattr__(self, "final_reviewer_profile", self.reviewer_profile)
        if self.reviser_profile is not None:
            object.__setattr__(self, "semantic_reviser_profile", self.reviser_profile)
        if self.repair_profile is not None:
            object.__setattr__(self, "check_repair_profile", self.repair_profile)

        if isinstance(self.schema_version, bool) or self.schema_version not in {
            HISTORICAL_SCHEMA_VERSION, SCHEMA_VERSION
        }:
            raise RunOptionsError("run options schema_version is unsupported")
        if isinstance(self.pipeline_version, bool) or self.pipeline_version not in {1, 2}:
            raise RunOptionsError("run options pipeline_version is unsupported")
        if not isinstance(self.protocol, str) or self.protocol not in {"v1", "v2"}:
            raise RunOptionsError("run options protocol is invalid")
        if not isinstance(self.decomposition, str) or self.decomposition not in {"balanced", "aggressive"}:
            raise RunOptionsError("run options decomposition is invalid")
        if not isinstance(self.execution_mode_policy, str) or self.execution_mode_policy not in {"auto", "require-staged"}:
            raise RunOptionsError("run options execution_mode_policy is invalid")
        for name in ("single_step_max_mutable_paths", "staged_step_max_mutable_paths"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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
        if (isinstance(self.repair_scope_max_added_paths, bool)
                or not isinstance(self.repair_scope_max_added_paths, int)
                or self.repair_scope_max_added_paths <= 0):
            raise RunOptionsError("run options repair_scope_max_added_paths must be greater than zero")

        # Keep old attribute reads deterministic without making them durable
        # authorities.  A positive generic review budget means the old binary
        # switch was enabled.
        object.__setattr__(self, "claude_revision_enabled", self.semantic_revision_enabled)
        object.__setattr__(self, "repair_cycles", int(self.max_review_repair_cycles > 0))
        object.__setattr__(self, "reviewer_profile", self.final_reviewer_profile)
        object.__setattr__(self, "reviser_profile", self.semantic_reviser_profile)
        object.__setattr__(self, "repair_profile", self.check_repair_profile)

    @classmethod
    def from_config(cls, config: HarnessConfig, **overrides: Any) -> "RunOptions":
        """Build and validate a schema-2 snapshot from trusted config."""

        allowed = {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
            "semantic_revision_enabled", "max_check_repair_attempts",
            "max_review_repair_cycles", "planner_profile",
            "default_implementer_profile", "check_repair_profile",
            "semantic_reviser_profile", "final_reviewer_profile",
            "repair_scope_policy", "repair_scope_max_added_paths",
            # Historical request aliases.
            "claude_revision_enabled", "repair_cycles", "reviewer_profile",
            "reviser_profile", "repair_profile",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise RunOptionsError(f"unknown run option: {sorted(unknown)[0]}")
        check_profile = config.ui.default_repair_profile
        reviser_profile = config.ui.default_reviser_profile
        # Both correction budgets use the independently selected repair
        # backend.  Semantic revision is a separate switch/profile pair.
        check_budget = config.revision.max_check_repair_attempts if check_profile else 0
        review_budget = config.revision.max_review_repair_cycles if check_profile else 0
        values: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": 2,
            "protocol": config.planning.protocol,
            "decomposition": config.planning.decomposition,
            "execution_mode_policy": config.planning.execution_mode_policy,
            "single_step_max_mutable_paths": config.planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": config.planning.staged_step_max_mutable_paths,
            "semantic_revision_enabled": config.revision.enabled,
            "max_check_repair_attempts": check_budget,
            "max_review_repair_cycles": review_budget,
            "planner_profile": config.ui.default_planner_profile or "legacy-planner",
            "default_implementer_profile": config.ui.default_implementer_profile or "legacy-implementer",
            "check_repair_profile": check_profile,
            "semantic_reviser_profile": reviser_profile,
            "final_reviewer_profile": config.ui.default_reviewer_profile or "legacy-reviewer",
            "repair_scope_policy": "auto-bounded",
            "repair_scope_max_added_paths": 4,
        }
        aliases = {
            "claude_revision_enabled": "semantic_revision_enabled",
            "repair_cycles": "max_review_repair_cycles",
            "reviewer_profile": "final_reviewer_profile",
            "reviser_profile": "semantic_reviser_profile",
            "repair_profile": "check_repair_profile",
        }
        for old_name, new_name in aliases.items():
            if old_name in overrides:
                values[new_name] = overrides[old_name]
        values.update({key: value for key, value in overrides.items() if key not in aliases})
        result = cls(**values)
        result.validate_profiles(config)
        if result.semantic_revision_enabled and result.semantic_reviser_profile is None:
            raise RunOptionsError("semantic revision requires a semantic reviser profile")
        if (
            result.max_check_repair_attempts > 0
            or result.max_review_repair_cycles > 0
        ) and result.check_repair_profile is None:
            raise RunOptionsError("correction budget requires a check-repair profile")
        return result

    def validate_profiles(self, config: HarnessConfig) -> None:
        roles = (
            ("planner_profile", ExecutionRole.PLANNER),
            ("default_implementer_profile", ExecutionRole.IMPLEMENTER),
            ("final_reviewer_profile", ExecutionRole.REVIEWER),
        )
        for name, role in roles:
            try:
                profile_for_role(config, getattr(self, name), role)
            except ProfileError as exc:
                raise RunOptionsError(f"{name} is invalid or incompatible") from exc
        for name, role in (
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
        if self.schema_version == HISTORICAL_SCHEMA_VERSION:
            return {
                "schema_version": HISTORICAL_SCHEMA_VERSION,
                "planning": {
                    "protocol": self.protocol,
                    "decomposition": self.decomposition,
                    "execution_mode_policy": self.execution_mode_policy,
                    "single_step_max_mutable_paths": self.single_step_max_mutable_paths,
                    "staged_step_max_mutable_paths": self.staged_step_max_mutable_paths,
                },
                "pipeline": {
                    "claude_revision_enabled": self.semantic_revision_enabled,
                    "repair_cycles": self.repair_cycles,
                    "repair_scope_policy": self.repair_scope_policy,
                    "repair_scope_max_added_paths": self.repair_scope_max_added_paths,
                },
                "profiles": {
                    "planner_profile": self.planner_profile,
                    "default_implementer_profile": self.default_implementer_profile,
                    "reviewer_profile": self.final_reviewer_profile,
                    "reviser_profile": self.semantic_reviser_profile,
                    "repair_profile": self.check_repair_profile,
                },
            }
        return {
            "schema_version": SCHEMA_VERSION,
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
        if not isinstance(value, Mapping):
            raise RunOptionsError("run options must be a JSON object")
        if value.get("schema_version") == HISTORICAL_SCHEMA_VERSION:
            if set(value) != {"schema_version", "planning", "pipeline", "profiles"}:
                raise RunOptionsError("historical run options contains unknown or missing fields")
            planning, pipeline, profiles = value["planning"], value["pipeline"], value["profiles"]
            if not isinstance(planning, Mapping) or set(planning) != {
                "protocol", "decomposition", "execution_mode_policy",
                "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
            }:
                raise RunOptionsError("historical run options planning schema is invalid")
            if (not isinstance(pipeline, Mapping)
                    or not {"claude_revision_enabled", "repair_cycles"}.issubset(set(pipeline))
                    or set(pipeline) - {"claude_revision_enabled", "repair_cycles",
                                        "repair_scope_policy", "repair_scope_max_added_paths"}):
                raise RunOptionsError("historical run options pipeline schema is invalid")
            if not isinstance(profiles, Mapping) or set(profiles) != {
                "planner_profile", "default_implementer_profile", "reviewer_profile",
                "reviser_profile", "repair_profile",
            }:
                raise RunOptionsError("historical run options profiles schema is invalid")
            try:
                result = cls(
                    schema_version=HISTORICAL_SCHEMA_VERSION,
                    protocol=planning["protocol"],
                    decomposition=planning["decomposition"],
                    execution_mode_policy=planning["execution_mode_policy"],
                    single_step_max_mutable_paths=planning["single_step_max_mutable_paths"],
                    staged_step_max_mutable_paths=planning["staged_step_max_mutable_paths"],
                    semantic_revision_enabled=pipeline["claude_revision_enabled"],
                    max_review_repair_cycles=pipeline["repair_cycles"],
                    planner_profile=profiles["planner_profile"],
                    default_implementer_profile=profiles["default_implementer_profile"],
                    final_reviewer_profile=profiles["reviewer_profile"],
                    semantic_reviser_profile=profiles["reviser_profile"],
                    check_repair_profile=profiles["repair_profile"],
                    repair_scope_policy=pipeline.get("repair_scope_policy", "deny-expansion"),
                    repair_scope_max_added_paths=pipeline.get("repair_scope_max_added_paths", 4),
                )
            except (KeyError, TypeError) as exc:
                raise RunOptionsError("historical run options schema is invalid") from exc
            if result.semantic_revision_enabled and result.semantic_reviser_profile is None:
                raise RunOptionsError("semantic revision requires a semantic reviser profile")
            if result.repair_cycles == 1 and result.check_repair_profile is None:
                raise RunOptionsError("check repair requires a check-repair profile")
            return result

        if set(value) != {"schema_version", "pipeline_version", "planning", "pipeline", "profiles"}:
            raise RunOptionsError("run options contains unknown or missing fields")
        planning, pipeline, profiles = value["planning"], value["pipeline"], value["profiles"]
        if not isinstance(planning, Mapping) or set(planning) != {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
        }:
            raise RunOptionsError("run options planning schema is invalid")
        if (not isinstance(pipeline, Mapping)
                or not {"semantic_revision_enabled", "max_check_repair_attempts",
                        "max_review_repair_cycles"}.issubset(set(pipeline))
                or set(pipeline) - {"semantic_revision_enabled", "max_check_repair_attempts",
                                    "max_review_repair_cycles", "repair_scope_policy",
                                    "repair_scope_max_added_paths"}):
            raise RunOptionsError("run options pipeline schema is invalid")
        if not isinstance(profiles, Mapping) or set(profiles) != {
            "planner_profile", "default_implementer_profile", "check_repair_profile",
            "semantic_reviser_profile", "final_reviewer_profile",
        }:
            raise RunOptionsError("run options profiles schema is invalid")
        try:
            result = cls(
                schema_version=value["schema_version"],
                pipeline_version=value["pipeline_version"],
                protocol=planning["protocol"],
                decomposition=planning["decomposition"],
                execution_mode_policy=planning["execution_mode_policy"],
                single_step_max_mutable_paths=planning["single_step_max_mutable_paths"],
                staged_step_max_mutable_paths=planning["staged_step_max_mutable_paths"],
                semantic_revision_enabled=pipeline["semantic_revision_enabled"],
                max_check_repair_attempts=pipeline["max_check_repair_attempts"],
                max_review_repair_cycles=pipeline["max_review_repair_cycles"],
                planner_profile=profiles["planner_profile"],
                default_implementer_profile=profiles["default_implementer_profile"],
                check_repair_profile=profiles["check_repair_profile"],
                semantic_reviser_profile=profiles["semantic_reviser_profile"],
                final_reviewer_profile=profiles["final_reviewer_profile"],
                repair_scope_policy=pipeline.get("repair_scope_policy", "deny-expansion"),
                repair_scope_max_added_paths=pipeline.get("repair_scope_max_added_paths", 4),
            )
        except (KeyError, TypeError) as exc:
            raise RunOptionsError("run options schema is invalid") from exc
        if result.semantic_revision_enabled and result.semantic_reviser_profile is None:
            raise RunOptionsError("semantic revision requires a semantic reviser profile")
        if (
            result.max_check_repair_attempts > 0
            or result.max_review_repair_cycles > 0
        ) and result.check_repair_profile is None:
            raise RunOptionsError("correction budget requires a check-repair profile")
        return result


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunOptionsError("duplicate run options field")
        result[key] = value
    return result


def _override_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunOptionsError("duplicate repair scope override field")
        result[key] = value
    return result


def canonical_run_options_bytes(options: RunOptions) -> bytes:
    return (json.dumps(options.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def run_options_sha256(options: RunOptions) -> str:
    return hashlib.sha256(canonical_run_options_bytes(options)).hexdigest()


def write_run_options(run_dir: str | Path, options: RunOptions) -> str:
    """Claim the immutable artifact, allowing only an exact idempotent write."""

    if not isinstance(options, RunOptions):
        raise TypeError("options must be RunOptions")
    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    data = canonical_run_options_bytes(options)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RunOptionsError("run options are unreadable") from exc
        if existing != data:
            raise RunOptionsConflict("run_options.json is immutable")
    else:
        atomic_write_text(path, data.decode("utf-8"))
    return hashlib.sha256(data).hexdigest()


def canonical_repair_scope_override_bytes(override: RepairScopeOverride) -> bytes:
    if not isinstance(override, RepairScopeOverride):
        raise TypeError("override must be RepairScopeOverride")
    return (json.dumps(override.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_repair_scope_override(
    run_dir: str | Path, override: RepairScopeOverride,
) -> str:
    """Claim the immutable operator override, allowing exact idempotence."""

    data = canonical_repair_scope_override_bytes(override)
    path = Path(run_dir).expanduser().resolve() / REPAIR_SCOPE_OVERRIDE_NAME
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RunOptionsError("repair scope override is unreadable") from exc
        if existing != data:
            raise RunOptionsConflict("repair_scope_override.json is immutable")
    else:
        atomic_write_text(path, data.decode("utf-8"))
    return hashlib.sha256(data).hexdigest()


def read_repair_scope_override(
    run_dir: str | Path,
) -> RepairScopeOverride | None:
    """Read the immutable operator override, or ``None`` when absent."""

    path = Path(run_dir).expanduser().resolve() / REPAIR_SCOPE_OVERRIDE_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_override_json_pairs
        )
    except RunOptionsError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunOptionsError("repair_scope_override.json is missing or malformed") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version", "policy", "max_added_paths", "reason",
    }:
        raise RunOptionsError("repair scope override schema is invalid")
    try:
        return RepairScopeOverride(
            schema_version=payload["schema_version"],
            policy=payload["policy"],
            max_added_paths=payload["max_added_paths"],
            reason=payload["reason"],
        )
    except (KeyError, TypeError) as exc:
        raise RunOptionsError("repair scope override schema is invalid") from exc


def effective_repair_scope_policy(
    options: RunOptions,
    *,
    raw_run_options: Mapping[str, Any] | None,
    override: RepairScopeOverride | None,
) -> EffectiveRepairScopePolicy:
    """Resolve scope authority while preserving the historical default."""

    if not isinstance(options, RunOptions):
        raise TypeError("options must be RunOptions")
    pipeline = raw_run_options.get("pipeline") if isinstance(raw_run_options, Mapping) else None
    if pipeline is not None and not isinstance(pipeline, Mapping):
        raise RunOptionsError("run options pipeline schema is invalid")
    explicit_keys = set(pipeline or ()) & {
        "repair_scope_policy", "repair_scope_max_added_paths",
    }
    if override is not None:
        if not isinstance(override, RepairScopeOverride):
            raise TypeError("override must be RepairScopeOverride")
        if explicit_keys:
            raise RunOptionsError(
                "repair scope override is only allowed for historical run options"
            )
        return EffectiveRepairScopePolicy(
            override.policy, override.max_added_paths, "operator-override"
        )
    if not explicit_keys:
        return EffectiveRepairScopePolicy(
            "deny-expansion", 4, "historical-default"
        )
    return EffectiveRepairScopePolicy(
        options.repair_scope_policy,
        options.repair_scope_max_added_paths,
        "run-options",
    )


def read_run_options_with_sha256_and_raw(
    run_dir: str | Path, expected_sha256: str | Mapping[str, Any] | None = None,
) -> tuple[RunOptions, str, Mapping[str, Any]]:
    if isinstance(expected_sha256, Mapping):
        expected_sha256 = expected_sha256.get("run_options_sha256")
    if expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise RunOptionsError("run options hash is invalid")
    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    try:
        data = path.read_bytes()
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_json_pairs)
        options = RunOptions.from_mapping(raw)
    except RunOptionsError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunOptionsError("run_options.json is missing or malformed") from exc
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunOptionsError("run_options.json hash mismatch")
    if not isinstance(raw, Mapping):  # pragma: no cover - guarded by from_mapping
        raise RunOptionsError("run options must be a JSON object")
    return options, digest, raw


def read_run_options_with_sha256(
    run_dir: str | Path, expected_sha256: str | Mapping[str, Any] | None = None,
) -> tuple[RunOptions, str]:
    options, digest, _raw = read_run_options_with_sha256_and_raw(
        run_dir, expected_sha256
    )
    return options, digest


def legacy_or_durable_run_options(
    config: HarnessConfig, run_dir: str | Path, *, expected_sha256: str | None = None,
) -> tuple[RunOptions, str | None]:
    """Read the snapshot, or explicitly derive historical defaults."""

    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    if not path.exists():
        if expected_sha256 is not None:
            raise RunOptionsError("run_options.json is missing")
        # A pre-snapshot run must remain on the historical no-expansion
        # policy when it is resumed.  New durable runs receive the
        # auto-bounded default through the normal creation path.
        options = RunOptions.from_config(config, repair_scope_policy="deny-expansion")
        return options, None
    options, digest, _raw = read_run_options_with_sha256_and_raw(
        run_dir, expected_sha256
    )
    return options, digest


def legacy_or_durable_run_options_with_raw(
    config: HarnessConfig, run_dir: str | Path, *, expected_sha256: str | None = None,
) -> tuple[RunOptions, str | None, Mapping[str, Any] | None]:
    """Read options with the original JSON shape needed for policy decisions."""

    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    if not path.exists():
        options, digest = legacy_or_durable_run_options(
            config, run_dir, expected_sha256=expected_sha256
        )
        return options, digest, None
    return read_run_options_with_sha256_and_raw(run_dir, expected_sha256)


def effective_run_config(config: HarnessConfig, options: RunOptions) -> HarnessConfig:
    """Return an immutable per-run view without mutating the shared config."""

    from .models import PlanningConfig, RevisionConfig, UIConfig

    options.validate_profiles(config)
    planning = PlanningConfig(
        protocol=options.protocol,
        decomposition=options.decomposition,
        single_step_max_mutable_paths=options.single_step_max_mutable_paths,
        staged_step_max_mutable_paths=options.staged_step_max_mutable_paths,
        execution_mode_policy=options.execution_mode_policy,
    )
    ui = replace(
        config.ui,
        default_planner_profile=options.planner_profile,
        default_implementer_profile=options.default_implementer_profile,
        default_reviewer_profile=options.reviewer_profile,
        default_reviser_profile=options.reviser_profile,
        default_repair_profile=options.repair_profile,
    )
    revision = RevisionConfig(
        enabled=options.semantic_revision_enabled,
        max_check_repair_attempts=options.max_check_repair_attempts,
        max_review_repair_cycles=options.max_review_repair_cycles,
    )
    return replace(config, planning=planning, ui=ui, revision=revision)


__all__ = [
    "RUN_OPTIONS_NAME", "REPAIR_SCOPE_OVERRIDE_NAME", "REPAIR_SCOPE_OVERRIDE_REASON",
    "REPAIR_SCOPE_OVERRIDE_SCHEMA_VERSION", "REPAIR_SCOPE_MAX_ADDED_PATHS",
    "SCHEMA_VERSION", "HISTORICAL_SCHEMA_VERSION", "REPAIR_SCOPE_POLICIES", "RunOptions", "RunOptionsConflict",
    "RunOptionsError", "RepairScopeOverride", "EffectiveRepairScopePolicy",
    "canonical_run_options_bytes", "canonical_repair_scope_override_bytes",
    "effective_repair_scope_policy", "effective_run_config",
    "legacy_or_durable_run_options", "legacy_or_durable_run_options_with_raw",
    "read_run_options_with_sha256", "read_run_options_with_sha256_and_raw",
    "read_repair_scope_override", "run_options_sha256", "write_run_options",
    "write_repair_scope_override",
]
