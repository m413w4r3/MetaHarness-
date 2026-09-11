"""Modèles de données de configuration et de contrôle."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    BLOCKED = "blocked"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    PLAN_REJECTED = "plan_rejected"
    WORKTREE_READY = "worktree_ready"
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
        "CODEX_HOME",
    )


@dataclass(frozen=True)
class ApprovalConfig:
    require_plan_approval: bool = False
    poll_interval_seconds: float = 0.5


@dataclass(frozen=True)
class UIConfig:
    max_active_runs: int = 1


@dataclass(frozen=True)
class CheckConfig:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 3600
    required: bool = True


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
