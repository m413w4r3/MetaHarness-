"""Modèles de données de configuration et de contrôle."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class ProfileDriver(StrEnum):
    OPENAI_CHAT = "openai-chat"
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"


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
    REVISER = "reviser"


class PlanDecision(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class ExecutionMode(StrEnum):
    SINGLE = "SINGLE"
    STAGED = "STAGED"


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
    permission_mode: str | None = None

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


@dataclass(frozen=True)
class ImplementationStep:
    id: str
    title: str
    implementer_profile: str
    depends_on: str | None
    objective: str
    read_set: tuple[str, ...]
    write_set: tuple[str, ...]
    instructions: str
    verify: str
    forbidden: str
    # New paths the step may create, existing paths it may delete.  Older
    # META PLAN v2 answers without these sections parse to empty tuples.
    create_set: tuple[str, ...] = ()
    delete_set: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlanV2:
    decision: PlanDecision
    title: str
    objective: str
    constraints: str
    execution_mode: ExecutionMode | None
    reviewer_profile: str | None
    steps: tuple[ImplementationStep, ...]
    acceptance: str
    tests: str
    risks: str
    blockers: str
    raw: str


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
    PRE_REVISION_VALIDATING = "pre_revision_validating"
    REVISING = "revising"
    REVALIDATING = "revalidating"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
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
class RepositoryConfig:
    remote: str = "origin"
    planner_remote_exploration: bool = True
    web_url: str | None = None


class ExecutionModePolicy(StrEnum):
    """Planner freedom over EXECUTION_MODE, fixed by configuration."""

    AUTO = "auto"
    REQUIRE_STAGED = "require-staged"


class PublishMode(StrEnum):
    """Where a final reviewed commit is published."""

    # Push only the run branch ``harness/<plan>/<run-id>``.
    RUN_BRANCH = "run-branch"
    # Compare-and-swap fast-forward of the local base branch to the reviewed
    # commit, then push that exact commit to the remote base branch.
    FAST_FORWARD_BASE = "fast-forward-base"


@dataclass(frozen=True)
class PublishConfig:
    enabled: bool = False
    remote: str = "origin"
    mode: str = "run-branch"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("publish.enabled must be a boolean")
        if (
            not isinstance(self.remote, str)
            or not self.remote.strip()
            or any(char.isspace() for char in self.remote)
            or "\x00" in self.remote
            or self.remote.startswith("-")
        ):
            raise ValueError("publish.remote must be a valid remote name")
        if self.mode not in {item.value for item in PublishMode}:
            raise ValueError("publish.mode must be 'run-branch' or 'fast-forward-base'")


@dataclass(frozen=True)
class PlanningConfig:
    protocol: str = "v1"
    decomposition: str = "balanced"
    single_step_max_mutable_paths: int = 2
    execution_mode_policy: str = "auto"

    def __post_init__(self) -> None:
        if self.protocol not in {"v1", "v2"}:
            raise ValueError("planning protocol must be 'v1' or 'v2'")
        if self.decomposition not in {"balanced", "aggressive"}:
            raise ValueError("planning decomposition must be 'balanced' or 'aggressive'")
        if isinstance(self.single_step_max_mutable_paths, bool) or self.single_step_max_mutable_paths <= 0:
            raise ValueError("single_step_max_mutable_paths must be greater than zero")
        if self.execution_mode_policy not in {item.value for item in ExecutionModePolicy}:
            raise ValueError("planning execution_mode_policy must be 'auto' or 'require-staged'")


@dataclass(frozen=True)
class RevisionConfig:
    """Bounded automatic correction-loop configuration.

    ``enabled`` is the only authority for the P25/P26 Claude revision and
    C02 repair cycle; profiles present in the catalogue never enable it.
    """

    enabled: bool = False
    max_cycles: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("revision.enabled must be a boolean")
        if isinstance(self.max_cycles, bool) or self.max_cycles != 2:
            raise ValueError("revision.max_cycles must be exactly 2")


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
    default_reviser_profile: str | None = None
    default_repair_profile: str | None = None
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
class ClaudeRuntimeConfig:
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
    claude_runtime: ClaudeRuntimeConfig = field(
        default_factory=lambda: ClaudeRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "claude"
        )
    )
    workspace_setup: tuple[WorkspaceSetupCommand, ...] = ()
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    # Compatibility marker for programmatic legacy configurations that do not
    # have a repository TOML section yet.
    repository_section_explicit: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class SelectedProfile:
    profile_id: str
    driver: str
    model: str
    selection_mode: str
    effort: str | None = None
    sandbox: str | None = None
    permission_mode: str | None = None
    # SHA-256 of the execution-relevant profile configuration (schema 2).
    config_sha256: str | None = None


@dataclass(frozen=True)
class ExecutionSelection:
    schema_version: int
    planner: SelectedProfile
    implementer: SelectedProfile
    reviewer: SelectedProfile
    reviser: SelectedProfile | None = None


@dataclass(frozen=True)
class StepExecutionSelection:
    """The immutable implementer snapshot selected for one v2 step."""

    step_id: str
    implementer: SelectedProfile


@dataclass(frozen=True)
class ExecutionSelectionV3:
    """Durable planner, per-step implementer and reviewer selection."""

    schema_version: int
    planner: SelectedProfile
    steps: tuple[StepExecutionSelection, ...]
    reviewer: SelectedProfile
    reviser: SelectedProfile | None = None


@dataclass(frozen=True)
class ExecutionSelectionV4:
    """Immutable execution authority for new META PLAN v2 runs."""

    schema_version: int
    planner: SelectedProfile
    steps: tuple[StepExecutionSelection, ...]
    reviser: SelectedProfile
    repair_implementer: SelectedProfile
    reviewer: SelectedProfile


@dataclass(frozen=True)
class RunCycle:
    """One bounded orchestration cycle."""

    number: int
    kind: str

    def __post_init__(self) -> None:
        expected = {1: "initial", 2: "repair"}
        if self.number not in expected or self.kind != expected[self.number]:
            raise ValueError("run cycle must be initial cycle 1 or repair cycle 2")
