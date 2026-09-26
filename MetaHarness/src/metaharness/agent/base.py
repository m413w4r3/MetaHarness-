"""Contrats communs aux exécuteurs d'agents.

The concrete command-line integrations live in backend-specific modules.  The
orchestrator only needs the small request/result protocol defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..models import AgentExecutorCapabilities, ExecutionRole


AgentUsage = dict[str, int]

# Durable, backend-neutral failure reasons.  Backend-specific diagnostics stay
# in ``backend_reason`` and never become pipeline state.
AGENT_START_FAILED = "AGENT_START_FAILED"
AGENT_RUNTIME_FAILED = "AGENT_RUNTIME_FAILED"
AGENT_TIMEOUT = "AGENT_TIMEOUT"
AGENT_PROTOCOL_FAILED = "AGENT_PROTOCOL_FAILED"
AGENT_SCOPE_VIOLATION = "AGENT_SCOPE_VIOLATION"
AGENT_AUTH_FAILURE = "AGENT_AUTH_FAILURE"
AGENT_GIT_VIOLATION = "AGENT_GIT_VIOLATION"
AGENT_SCOPE_REQUEST = "AGENT_SCOPE_REQUEST"


@dataclass(frozen=True)
class AgentRunRequest:
    """One bounded execution request issued by the orchestration layer."""

    role: ExecutionRole
    profile_id: str
    prompt: str
    worktree: Path
    artifact_dir: Path
    mutable_paths: tuple[str, ...]
    # This is an infrastructure hint, not a business/backend selection.  It
    # lets the adapter preserve the v1 plan prompt and v2 step prompt bytes.
    prompt_mode: str = "raw"
    # Adapter inputs for retry and prompt-contract handling.
    contract: str | None = None
    retry_addendum: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Normalized result returned by every concrete executor."""

    status: str
    exit_reason: str | None
    tree_before: str
    tree_after: str
    usage: AgentUsage | None
    external_session_id: str | None
    report_path: str | None

    # Normalized execution and diagnostic fields.  They are intentionally
    # optional in the generic contract.
    exit_code: int | None = None
    timed_out: bool = False
    final_message: str = ""
    stderr_tail: str = ""
    driver: str | None = None
    driver_version: str | None = None
    backend_reason: str | None = None
    terminal_type: str | None = None
    terminal_subtype: str | None = None
    terminal_is_error: bool | None = None
    terminal_num_turns: int | None = None
    terminal_stop_reason: str | None = None
    terminal_errors: tuple[str, ...] = ()
    raw_result: Any = field(default=None, repr=False, compare=False)


@runtime_checkable
class AgentExecutor(Protocol):
    """Backend-independent execution contract."""

    @property
    def capabilities(self) -> AgentExecutorCapabilities:
        ...

    @property
    def driver_version(self) -> str | None:
        ...

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        ...


@dataclass(frozen=True)
class AgentResult:
    """Résultat observable d'une exécution d'agent."""

    exit_code: int
    timed_out: bool
    final_message: str
    usage: dict[str, int]
    stderr_tail: str


class AgentError(RuntimeError):
    """Erreur empêchant de produire un résultat d'agent."""


class AgentScopeError(AgentError):
    """The worker crossed a Git or mutable-scope boundary."""

    code = AGENT_SCOPE_VIOLATION


class AgentProtocolError(AgentError):
    """The worker produced an invalid execution protocol result."""

    code = AGENT_PROTOCOL_FAILED
