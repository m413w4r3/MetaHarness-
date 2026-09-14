"""Durable, secret-free options captured when a run is created."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .models import ExecutionRole, HarnessConfig, ProfileDriver
from .profiles import ProfileError, profile_for_role
from .result import atomic_write_text


class RunOptionsError(ValueError):
    """A run-options snapshot is absent, malformed, or incompatible."""


class RunOptionsConflict(RunOptionsError):
    """An immutable run-options artifact already contains different bytes."""


SCHEMA_VERSION = 1
RUN_OPTIONS_NAME = "run_options.json"


@dataclass(frozen=True)
class RunOptions:
    """The operator's requested, per-run configuration.

    Only validated profile IDs are stored.  Endpoint, model, credential and
    provider configuration remains exclusively in the trusted TOML catalogue.
    """

    schema_version: int
    protocol: str
    decomposition: str
    execution_mode_policy: str
    single_step_max_mutable_paths: int
    staged_step_max_mutable_paths: int
    claude_revision_enabled: bool
    repair_cycles: int
    planner_profile: str
    default_implementer_profile: str
    reviewer_profile: str
    reviser_profile: str | None
    repair_profile: str | None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != SCHEMA_VERSION:
            raise RunOptionsError("run options schema_version is unsupported")
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
        if not isinstance(self.claude_revision_enabled, bool):
            raise RunOptionsError("run options claude_revision_enabled must be boolean")
        if isinstance(self.repair_cycles, bool) or self.repair_cycles not in {0, 1}:
            raise RunOptionsError("run options repair_cycles must be 0 or 1")
        for name in ("planner_profile", "default_implementer_profile", "reviewer_profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RunOptionsError(f"run options {name} is invalid")
        for name in ("reviser_profile", "repair_profile"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RunOptionsError(f"run options {name} is invalid")

    @classmethod
    def from_config(cls, config: HarnessConfig, **overrides: Any) -> "RunOptions":
        """Build and validate a snapshot from trusted config plus overrides."""

        allowed = {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
            "claude_revision_enabled", "repair_cycles", "planner_profile",
            "default_implementer_profile", "reviewer_profile", "reviser_profile",
            "repair_profile",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise RunOptionsError(f"unknown run option: {sorted(unknown)[0]}")
        historical_repair = 1 if config.revision.enabled and config.revision.max_cycles == 2 else 0
        values: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": config.planning.protocol,
            "decomposition": config.planning.decomposition,
            "execution_mode_policy": config.planning.execution_mode_policy,
            "single_step_max_mutable_paths": config.planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": config.planning.staged_step_max_mutable_paths,
            "claude_revision_enabled": config.revision.enabled,
            "repair_cycles": historical_repair,
            "planner_profile": config.ui.default_planner_profile or "legacy-planner",
            "default_implementer_profile": config.ui.default_implementer_profile or "legacy-implementer",
            "reviewer_profile": config.ui.default_reviewer_profile or "legacy-reviewer",
            "reviser_profile": config.ui.default_reviser_profile,
            "repair_profile": config.ui.default_repair_profile,
        }
        values.update(overrides)
        result = cls(**values)
        result.validate_profiles(config)
        if result.claude_revision_enabled and result.reviser_profile is None:
            raise RunOptionsError("Claude revision requires a reviser profile")
        if result.repair_cycles == 1 and result.repair_profile is None:
            raise RunOptionsError("automatic repair requires a repair profile")
        return result

    def validate_profiles(self, config: HarnessConfig) -> None:
        roles = (
            ("planner_profile", ExecutionRole.PLANNER),
            ("default_implementer_profile", ExecutionRole.IMPLEMENTER),
            ("reviewer_profile", ExecutionRole.REVIEWER),
        )
        for name, role in roles:
            try:
                profile_for_role(config, getattr(self, name), role)
            except ProfileError as exc:
                raise RunOptionsError(f"{name} is invalid or incompatible") from exc
        for name, role in (("reviser_profile", ExecutionRole.REVISER), ("repair_profile", ExecutionRole.REPAIR)):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                resolved = profile_for_role(config, value, role)
            except ProfileError as exc:
                raise RunOptionsError(f"{name} is invalid or incompatible") from exc
            if role is ExecutionRole.REVISER and resolved.driver is not ProfileDriver.CLAUDE_CODE:
                raise RunOptionsError("reviser_profile must use claude-code")
            if role is ExecutionRole.REPAIR and resolved.driver is not ProfileDriver.CODEX:
                raise RunOptionsError("repair_profile must use codex")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planning": {
                "protocol": self.protocol,
                "decomposition": self.decomposition,
                "execution_mode_policy": self.execution_mode_policy,
                "single_step_max_mutable_paths": self.single_step_max_mutable_paths,
                "staged_step_max_mutable_paths": self.staged_step_max_mutable_paths,
            },
            "pipeline": {
                "claude_revision_enabled": self.claude_revision_enabled,
                "repair_cycles": self.repair_cycles,
            },
            "profiles": {
                "planner_profile": self.planner_profile,
                "default_implementer_profile": self.default_implementer_profile,
                "reviewer_profile": self.reviewer_profile,
                "reviser_profile": self.reviser_profile,
                "repair_profile": self.repair_profile,
            },
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "RunOptions":
        if not isinstance(value, Mapping):
            raise RunOptionsError("run options must be a JSON object")
        if set(value) != {"schema_version", "planning", "pipeline", "profiles"}:
            raise RunOptionsError("run options contains unknown or missing fields")
        planning = value["planning"]
        pipeline = value["pipeline"]
        profiles = value["profiles"]
        if not isinstance(planning, Mapping) or set(planning) != {
            "protocol", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
        }:
            raise RunOptionsError("run options planning schema is invalid")
        if not isinstance(pipeline, Mapping) or set(pipeline) != {"claude_revision_enabled", "repair_cycles"}:
            raise RunOptionsError("run options pipeline schema is invalid")
        if not isinstance(profiles, Mapping) or set(profiles) != {
            "planner_profile", "default_implementer_profile", "reviewer_profile",
            "reviser_profile", "repair_profile",
        }:
            raise RunOptionsError("run options profiles schema is invalid")
        try:
            result = cls(
                schema_version=value["schema_version"],
                protocol=planning["protocol"],
                decomposition=planning["decomposition"],
                execution_mode_policy=planning["execution_mode_policy"],
                single_step_max_mutable_paths=planning["single_step_max_mutable_paths"],
                staged_step_max_mutable_paths=planning["staged_step_max_mutable_paths"],
                claude_revision_enabled=pipeline["claude_revision_enabled"],
                repair_cycles=pipeline["repair_cycles"],
                planner_profile=profiles["planner_profile"],
                default_implementer_profile=profiles["default_implementer_profile"],
                reviewer_profile=profiles["reviewer_profile"],
                reviser_profile=profiles["reviser_profile"],
                repair_profile=profiles["repair_profile"],
            )
            if result.claude_revision_enabled and result.reviser_profile is None:
                raise RunOptionsError("Claude revision requires a reviser profile")
            if result.repair_cycles == 1 and result.repair_profile is None:
                raise RunOptionsError("automatic repair requires a repair profile")
            return result
        except (KeyError, TypeError) as exc:  # pragma: no cover - guarded by exact keys above
            raise RunOptionsError("run options schema is invalid") from exc


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunOptionsError("duplicate run options field")
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


def read_run_options_with_sha256(
    run_dir: str | Path, expected_sha256: str | Mapping[str, Any] | None = None,
) -> tuple[RunOptions, str]:
    if isinstance(expected_sha256, Mapping):
        expected_sha256 = expected_sha256.get("run_options_sha256")
    if expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise RunOptionsError("run options hash is invalid")
    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    try:
        data = path.read_bytes()
        options = RunOptions.from_mapping(json.loads(data.decode("utf-8"), object_pairs_hook=_json_pairs))
    except RunOptionsError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunOptionsError("run_options.json is missing or malformed") from exc
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunOptionsError("run_options.json hash mismatch")
    return options, digest


def legacy_or_durable_run_options(
    config: HarnessConfig, run_dir: str | Path, *, expected_sha256: str | None = None,
) -> tuple[RunOptions, str | None]:
    """Read the snapshot, or explicitly derive historical defaults."""

    path = Path(run_dir).expanduser().resolve() / RUN_OPTIONS_NAME
    if not path.exists():
        if expected_sha256 is not None:
            raise RunOptionsError("run_options.json is missing")
        options = RunOptions.from_config(config)
        return options, None
    return read_run_options_with_sha256(run_dir, expected_sha256)


def effective_run_config(config: HarnessConfig, options: RunOptions) -> HarnessConfig:
    """Return an immutable per-run view without mutating the shared config."""

    from .models import PlanningConfig, UIConfig

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
    return replace(config, planning=planning, ui=ui)


__all__ = [
    "RUN_OPTIONS_NAME", "SCHEMA_VERSION", "RunOptions", "RunOptionsConflict",
    "RunOptionsError", "canonical_run_options_bytes", "effective_run_config",
    "legacy_or_durable_run_options", "read_run_options_with_sha256",
    "run_options_sha256", "write_run_options",
]
