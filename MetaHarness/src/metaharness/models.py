"""Modèles de données de configuration et de contrôle."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class ProfileDriver(StrEnum):
    OPENAI_CHAT = "openai-chat"
    CODEX = "codex"


class SelectionMode(StrEnum):
    REQUEST = "request"
    CLI = "cli"
    EXTERNAL_UI = "external-ui"


class ExecutionRole(StrEnum):
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    REPAIR = "repair"
    AUDITOR = "auditor"


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display_name: str
    roles: tuple[ExecutionRole, ...]
    driver: ProfileDriver
    model: str
    selection_mode: SelectionMode

    base_url: str | None = None
    endpoint_path: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 300
    retries: int = 2
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    effort: str | None = None
    sandbox: str | None = None

    description: str = ""
    strengths: tuple[str, ...] = ()
    cost_tier: str = "standard"
    latency_tier: str = "standard"

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or len(self.description) > 300:
            raise ValueError("profile description must be at most 300 characters")
        if not isinstance(self.strengths, tuple) or len(self.strengths) > 8:
            raise ValueError("profile strengths must contain at most 8 entries")
        if any(not isinstance(item, str) or len(item) > 80 for item in self.strengths):
            raise ValueError("profile strengths entries must be at most 80 characters")
        if not isinstance(self.cost_tier, str) or self.cost_tier not in {"low", "standard", "high"}:
            raise ValueError("profile cost_tier is invalid")
        if not isinstance(self.latency_tier, str) or self.latency_tier not in {"fast", "standard", "slow"}:
            raise ValueError("profile latency_tier is invalid")


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    BLOCKED = "blocked"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    PLAN_REJECTED = "plan_rejected"
    WORKTREE_READY = "worktree_ready"
    PREPARING = "preparing"
    IMPLEMENTING = "implementing"
    VALIDATING = "validating"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    COMMITTED = "committed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    REVISE = "REVISE"
    FAIL = "FAIL"


class ReviewRoute(StrEnum):
    NONE = "NONE"
    IMPLEMENTATION = "IMPLEMENTATION"
    REPLAN = "REPLAN"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class LLMEndpointConfig:
    base_url: str
    endpoint_path: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: int = 300
    retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextConfig:
    always_files: tuple[str, ...] = (
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
    )
    locator_argv: tuple[str, ...] = ()
    locator_timeout_seconds: int = 120
    max_hits: int = 8
    max_bytes: int = 160_000
    require_locator_head_at_base: bool = True


@dataclass(frozen=True)
class AgentConfig:
    provider: str = "codex"
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    sandbox: str = "workspace-write"
    timeout_seconds: int = 5400
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
    )


@dataclass(frozen=True)
class ApprovalConfig:
    require_plan_approval: bool = False
    poll_interval_seconds: float = 0.5


@dataclass(frozen=True)
class UIConfig:
    max_active_runs: int = 1
    default_planner_profile: str | None = None
    default_implementer_profile: str | None = None
    default_reviewer_profile: str | None = None
    enable_profile_recommendation: bool = True


@dataclass(frozen=True)
class CheckConfig:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 3600
    required: bool = True


@dataclass(frozen=True)
class EnvironmentConfig:
    files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CodexRuntimeConfig:
    home: Path


@dataclass(frozen=True)
class WorkspaceSetupCommand:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 1200
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PNPM_HOME",
    )


@dataclass(frozen=True)
class HarnessConfig:
    repo: Path
    base_ref: str
    runs_root: Path
    worktrees_root: Path
    require_clean_base: bool
    planner: LLMEndpointConfig
    reviewer: LLMEndpointConfig
    context: ContextConfig
    agent: AgentConfig
    checks: tuple[CheckConfig, ...]
    max_diff_bytes: int = 400_000
    allow_no_required_checks: bool = False
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    model_profiles: Mapping[str, ModelProfile] = field(default_factory=dict)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runtime_environment: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    codex_runtime: CodexRuntimeConfig = field(
        default_factory=lambda: CodexRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "codex"
        )
    )
    workspace_setup: tuple[WorkspaceSetupCommand, ...] = ()


@dataclass(frozen=True)
class SelectedProfile:
    profile_id: str
    driver: str
    model: str
    selection_mode: str
    effort: str | None = None
    sandbox: str | None = None
    # SHA-256 of the execution-relevant profile configuration (schema 2).
    config_sha256: str | None = None


@dataclass(frozen=True)
class ExecutionSelection:
    schema_version: int
    planner: SelectedProfile
    implementer: SelectedProfile
    reviewer: SelectedProfile
