"""Deterministic classification of pipeline failures and recovery budgets.

One stable failure code maps to exactly one :class:`FailureClass` through the
single table :data:`FAILURE_CLASSES`; a code the table does not name is an
ordinary ``FIXABLE`` failure, never a stop.  Every class owns one ordered
ladder (:data:`RECOVERY_LADDERS`).  ``HARD_STOP`` is reachable only from the
closed ``FATAL`` allowlist and ``WAIT_HUMAN`` only from ``SPEC_DECISION``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class FailureClass(StrEnum):
    """What a stable failure code proves about who may act next."""

    TRANSIENT = "transient"
    FIXABLE = "fixable"
    SPEC_DECISION = "spec_decision"
    FATAL = "fatal"


class RecoveryStrategy(StrEnum):
    """One deterministic step of one recovery ladder."""

    RETRY_TARGETED = "retry_targeted"
    FALLBACK_EXECUTOR = "fallback_executor"
    MARK_FAILED_CONTINUE = "mark_failed_continue"
    WAIT_EXTERNAL = "wait_external"
    WAIT_HUMAN = "wait_human"
    HARD_STOP = "hard_stop"

    @property
    def terminal(self) -> bool:
        """Whether this step ends autonomous progression instead of acting."""

        return self in {
            RecoveryStrategy.WAIT_EXTERNAL, RecoveryStrategy.WAIT_HUMAN,
            RecoveryStrategy.HARD_STOP,
        }


_T, _F, _S, _X = (
    FailureClass.TRANSIENT, FailureClass.FIXABLE,
    FailureClass.SPEC_DECISION, FailureClass.FATAL,
)

# The single classification authority.  A key ending or starting with ``*`` is
# a prefix or suffix pattern; an exact key always wins over a pattern.
FAILURE_CLASSES: Mapping[str, FailureClass] = {
    # -- FATAL: secrets and content the harness cannot inspect.
    "SECRET_*": _X, "*_BLOB_NOT_REVIEWABLE": _X,
    "UNSCANNABLE_STAGED_BLOB": _X, "STAGED_BLOB_SCAN_FAILED": _X, "UNREVIEWABLE_TEXT_DIFF": _X, "COMMIT_SECURITY_FAILURE": _X,
    # -- FATAL: writes outside physical authority, foreign Git state.
    "TREE_MODIFIED_OUTSIDE_AUTHORITY": _X, "HEAD_MODIFIED_OUTSIDE_AUTHORITY": _X,
    "BRANCH_MODIFIED_OUTSIDE_AUTHORITY": _X, "REMOTE_AUTHORITY_MISMATCH": _X,
    "BASE_MOVED_SINCE_RUN": _X, "REPOSITORY_TREE_DRIFT_UNEXPLAINED": _X, "HARD_DENY_PATH_MUTATION": _X,
    # -- FATAL: the last green tree or the durable authority cannot be restored.
    "ROLLBACK_FAILED": _X, "ROLLBACK_TREE_MISMATCH": _X, "RESUME_REQUIRES_OPERATOR": _X,
    "RESUME_INTEGRITY_FAILURE": _X, "DURABLE_ARTIFACT_CORRUPTED": _X,
    "RUN_SCHEMA_UNSUPPORTED": _X,
    "AUDIT_PROFILE_NOT_WRITABLE": _X,
    # -- SPEC_DECISION: a product or policy choice the SPEC leaves open.
    "SPEC_DECISION_REQUIRED": _S,
    # -- TRANSIENT: providers, transport, executors and infrastructure.
    "LLM_*": _T, "AGENT_START_FAILED": _T, "AGENT_RUNTIME_FAILED": _T,
    "AGENT_TIMEOUT": _T, "AGENT_PROTOCOL_FAILED": _T, "AGENT_FAILURE": _T,
    "AGENT_AUTH_FAILURE": _T, "MISSING_PROVIDER_CREDENTIALS": _T,
    "PROVIDER_CREDENTIALS_MISSING": _T, "EXTERNAL_AUTH_REQUIRED": _T,
    "REMOTE_TEMPORARILY_UNAVAILABLE": _T, "REMOTE_UNAVAILABLE": _T,
    "CANDIDATE_REMOTE_UNAVAILABLE": _T, "PUSH_FAILED": _T, "CANDIDATE_PUSH_FAILED": _T,
    "GITHUB_WORKSTREAM_FAILURE": _T, "GIT_FAILURE": _T,
    "CHECK_TIMEOUT": _T, "CHECK_PREFLIGHT_FAILED": _T, "CHECK_INFRA_FAILURE": _T,
    "CHECK_INFRASTRUCTURE_UNAVAILABLE": _T,
    "CHECK_SIDE_EFFECT_REPEATED": _T, "CHECK_SIDE_EFFECT_UNSTABLE": _T,
    "WORKSPACE_SETUP_FAILED": _T, "WORKSPACE_SETUP_TIMEOUT": _T,
    # -- FIXABLE: model, contract, scope and correctness failures.
    "AGENT_CONTRACT_MISMATCH": _F, "AGENT_NO_CHANGE": _F, "AGENT_SCOPE_VIOLATION": _F,
    "AGENT_GIT_VIOLATION": _F,
    "CHECK_FAILED": _F, "DETERMINISTIC_GATE_FAILED": _F,
    "AUDIT_REMAINING": _F, "CODEX_RUNTIME_FAILURE": _F,
    "GITHUB_CONFIG_INVALID": _F, "GITHUB_ISSUE_NOT_FOUND": _F,
    "GITHUB_PR_CANDIDATE_MISMATCH": _F, "GITHUB_PR_REQUIRES_RUN_BRANCH": _F,
    "PER_STEP_GATE_REGRESSION": _F, "CHECK_SETUP_INVALID": _F,
    "PLANNER_OUTPUT_INVALID": _F,
    "PLANNER_BLOCKED_REQUIRES_OPERATOR": _F, "PLAN_APPROVAL_INVALID": _F,
    "PLAN_REPOSITORY_PRECONDITION_INVALID": _F,
    "REPOSITORY_EVIDENCE_RECOVERY_EXHAUSTED": _F,
    "EXECUTION_SELECTION_INVALID": _F, "WORKSPACE_SETUP_MUTATED": _F,
    "COMMIT_GATE_FAILED": _F, "COMMIT_SCOPE_VIOLATION": _F, "COMMIT_PARENT_MISMATCH": _F,
    "COMMIT_WORKTREE_DRIFT": _F, "COMMIT_VERIFICATION_FAILURE": _F,
    "COMMIT_TREE_MISMATCH": _F, "HEAD_MISMATCH": _F, "TREE_MISMATCH": _F, "TOCTOU_FAILURE": _F, "INVALID_PHASE_TRANSITION": _F,
    "INTERNAL_HARNESS_ERROR": _F,
}

RECOVERY_LADDERS: Mapping[FailureClass, tuple[RecoveryStrategy, ...]] = {
    FailureClass.TRANSIENT: (
        RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
        RecoveryStrategy.WAIT_EXTERNAL,
    ),
    FailureClass.FIXABLE: (
        RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
        RecoveryStrategy.MARK_FAILED_CONTINUE,
    ),
    FailureClass.SPEC_DECISION: (RecoveryStrategy.WAIT_HUMAN,),
    FailureClass.FATAL: (RecoveryStrategy.HARD_STOP,),
}

_REASONS = {
    FailureClass.TRANSIENT: "a temporary external condition; retried, then waited on",
    FailureClass.FIXABLE: "an ordinary failure; recovered autonomously",
    FailureClass.SPEC_DECISION: "a product decision the SPEC does not settle",
    FailureClass.FATAL: "a proven, irrecoverable authority or integrity boundary",
}


@dataclass(frozen=True)
class RecoveryDecision:
    """One classification: the class, the ladder step it admits, and why.

    ``known`` is false for a code :data:`FAILURE_CLASSES` does not name; the
    decision is still the ordinary ``FIXABLE`` ladder.
    """

    failure_class: FailureClass
    strategy: RecoveryStrategy
    reason: str
    known: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.failure_class, FailureClass):
            raise TypeError("recovery failure class must be a FailureClass")
        if not isinstance(self.strategy, RecoveryStrategy):
            raise TypeError("recovery strategy must be a RecoveryStrategy")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("recovery decision reason must be a non-empty string")
        # HARD_STOP only from FATAL, WAIT_HUMAN only from SPEC_DECISION.
        if (self.strategy is RecoveryStrategy.HARD_STOP) != (
            self.failure_class is FailureClass.FATAL
        ) or (self.strategy is RecoveryStrategy.WAIT_HUMAN) != (
            self.failure_class is FailureClass.SPEC_DECISION
        ):
            raise ValueError(
                f"{self.failure_class.value} failures never reach {self.strategy.value}"
            )


@dataclass(frozen=True)
class ExecutionFallbacks:
    """Configured executor fallback profile IDs, grouped by worker authority."""

    mechanical: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    agentic: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("mechanical", "reasoning", "agentic"):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(values, tuple):
                raise ValueError(f"execution_fallbacks.{name} must be a tuple of profile IDs")
            if len(values) > 10 or any(
                not isinstance(value, str)
                or not value
                or len(value) > 64
                or not value[0].isalnum()
                or any(not (char.isalnum() or char in "_.-") for char in value)
                for value in values
            ) or len(set(values)) != len(values):
                raise ValueError(f"execution_fallbacks.{name} contains an invalid profile ID")

    def for_execution_class(self, execution_class: str) -> tuple[str, ...]:
        key = str(execution_class).casefold()
        if key not in {"mechanical", "reasoning", "agentic"}:
            raise ValueError("execution class is invalid")
        return getattr(self, key)


@dataclass(frozen=True)
class RecoveryBudgets:
    """Retry limits frozen into each run's immutable options snapshot."""

    max_transient_attempts: int = 2
    max_executor_fallbacks: int = 1
    max_check_infra_retries: int = 2
    max_workspace_setup_retries: int = 2
    execution_fallbacks: ExecutionFallbacks = ExecutionFallbacks()

    def __post_init__(self) -> None:
        for name in (
            "max_transient_attempts", "max_executor_fallbacks",
            "max_check_infra_retries", "max_workspace_setup_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
                raise ValueError(f"{name} must be an integer between 0 and 10")
        if not isinstance(self.execution_fallbacks, ExecutionFallbacks):
            raise ValueError("execution_fallbacks is invalid")


def stable_code(failure: object) -> str:
    """The stable code of a failure, without its detail or check ID suffix."""

    if not isinstance(failure, str) or not failure.strip():
        raise ValueError("failure must be a non-empty code")
    return failure.strip().split(":", 1)[0].strip().upper()


def _lookup(code: str) -> FailureClass | None:
    known = FAILURE_CLASSES.get(code)
    if known is not None:
        return known
    for key, failure_class in FAILURE_CLASSES.items():
        if (key.endswith("*") and code.startswith(key[:-1])) or (
            key.startswith("*") and code.endswith(key[1:])
        ):
            return failure_class
    return None


def recovery_ladder(failure_class: FailureClass) -> tuple[RecoveryStrategy, ...]:
    """The ordered ladder of one failure class, its last step included."""

    return RECOVERY_LADDERS[FailureClass(failure_class)]


def classify_failure(failure: str, *, exhausted: bool = False) -> RecoveryDecision:
    """Classify one stable failure code; an unknown code is ``FIXABLE``.

    ``failure`` may carry a detail or check ID suffix (``CHECK_FAILED:unit``);
    only its stable code is consulted.  The decision names the first step of
    the class ladder, or its last step once every autonomous step of the
    caller's loop is ``exhausted``.  Pure: it never emits or logs anything.
    """

    known = _lookup(stable_code(failure))
    failure_class = known or FailureClass.FIXABLE
    ladder = RECOVERY_LADDERS[failure_class]
    return RecoveryDecision(
        failure_class, ladder[-1] if exhausted else ladder[0],
        _REASONS[failure_class], known is not None,
    )


__all__ = [
    "ExecutionFallbacks", "FAILURE_CLASSES", "FailureClass", "RECOVERY_LADDERS",
    "RecoveryBudgets", "RecoveryDecision", "RecoveryStrategy", "classify_failure",
    "recovery_ladder", "stable_code",
]
