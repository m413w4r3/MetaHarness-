"""Contrats communs aux exécuteurs d'agents.

The concrete command-line integrations live in backend-specific modules.  The
orchestrator only needs the small request/result protocol defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..models import ExecutionRole


AgentUsage = dict[str, int]

# Durable, backend-neutral failure reasons.  Historical backend-specific
# reasons remain readable in ``resume.py`` and in old artifacts.
AGENT_START_FAILED = "AGENT_START_FAILED"
AGENT_RUNTIME_FAILED = "AGENT_RUNTIME_FAILED"
AGENT_TIMEOUT = "AGENT_TIMEOUT"
AGENT_PROTOCOL_FAILED = "AGENT_PROTOCOL_FAILED"
AGENT_SCOPE_VIOLATION = "AGENT_SCOPE_VIOLATION"


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
    # Optional adapter inputs retained for compatibility with the historical
    # in-process Codex double.  They do not change the generic contract.
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

    # Compatibility/projection fields used by the existing artifact and
    # resume code.  They are intentionally optional in the generic contract.
    exit_code: int | None = None
    timed_out: bool = False
    final_message: str = ""
    stderr_tail: str = ""
    driver: str | None = None
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


def normalized_failure_reason(result: AgentRunResult) -> str | None:
    """Return the generic durable reason for a normalized result."""

    if result.timed_out or result.exit_reason == AGENT_TIMEOUT:
        return AGENT_TIMEOUT
    if result.exit_reason in {
        AGENT_START_FAILED,
        AGENT_RUNTIME_FAILED,
        AGENT_PROTOCOL_FAILED,
        AGENT_SCOPE_VIOLATION,
    }:
        return result.exit_reason
    if result.exit_code not in (None, 0):
        return AGENT_RUNTIME_FAILED
    if result.status not in {"completed", "success", "ok"}:
        return AGENT_RUNTIME_FAILED
    return None
