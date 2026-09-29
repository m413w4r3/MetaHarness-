"""Deterministic classification of pipeline failures and recovery budgets.

One canonical stable failure code maps to exactly one :class:`FailureClass`
through the single table :data:`FAILURE_CLASSES`; input aliases resolve before
classification, while an unknown code is an ordinary ``FIXABLE`` failure.
Every class owns one ordered ladder (:data:`RECOVERY_LADDERS`).  ``HARD_STOP``
is reachable only from the closed ``FATAL`` allowlist and ``WAIT_HUMAN`` only
from ``SPEC_DECISION``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
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
    # Read-only safety for fatal boundaries recorded by current v4 checkpoints.
    # These codes are no longer produced; unknown namespaces stay FIXABLE.
    'UNSCANNABLE_STAGED_BLOB': _X,
    'STAGED_BLOB_SCAN_FAILED': _X,
    'UNREVIEWABLE_TEXT_DIFF': _X,
    'HEAD_MODIFIED_OUTSIDE_AUTHORITY': _X,
    'BRANCH_MODIFIED_OUTSIDE_AUTHORITY': _X,
    'REMOTE_AUTHORITY_MISMATCH': _X,
    'ROLLBACK_TREE_MISMATCH': _X,
    'SECRET_*': _X,
    '*_BLOB_NOT_REVIEWABLE': _X,
    'COMMIT_SECURITY_FAILURE': _X,
    'TREE_MODIFIED_OUTSIDE_AUTHORITY': _X,
    'BASE_MOVED_SINCE_RUN': _X,
    'REPOSITORY_TREE_DRIFT_UNEXPLAINED': _X,
    'HARD_DENY_PATH_MUTATION': _X,
    'ROLLBACK_FAILED': _X,
    'RESUME_REQUIRES_OPERATOR': _X,
    'RESUME_INTEGRITY_FAILURE': _X,
    'DURABLE_ARTIFACT_CORRUPTED': _X,
    'RUN_SCHEMA_UNSUPPORTED': _X,
    'AUDIT_PROFILE_NOT_WRITABLE': _X,
    'SPEC_DECISION_REQUIRED': _S,
    'LLM_*': _T,
    'AGENT_RUNTIME_FAILED': _T,
    'AGENT_TIMEOUT': _T,
    'EXTERNAL_AUTH_REQUIRED': _T,
    'PUSH_FAILED': _T,
    'GITHUB_WORKSTREAM_FAILURE': _T,
    'GIT_FAILURE': _T,
    'CHECK_INFRASTRUCTURE_UNAVAILABLE': _T,
    'CHECK_SIDE_EFFECT_REPEATED': _T,
    'AGENT_CONTRACT_MISMATCH': _F,
    'AGENT_SCOPE_VIOLATION': _F,
    'AGENT_GIT_VIOLATION': _F,
    'CHECK_FAILED': _F,
    'DETERMINISTIC_GATE_FAILED': _F,
    'INTERNAL_HARNESS_ERROR': _F,
    'CONFIGURATION_INVALID': _F,
    'PER_STEP_GATE_REGRESSION': _F,
    'PLANNER_OUTPUT_INVALID': _F,
    'PLAN_APPROVAL_INVALID': _F,
    'COMMIT_GATE_FAILED': _F,
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
    audit: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("mechanical", "reasoning", "agentic", "audit"):
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


# The exact five limits of the one autonomous budget; no other numeric
# retry or autonomy knob exists anywhere in the runtime.
_BUDGET_BOUNDS: Mapping[str, tuple[int, int]] = {
    "step_attempts": (1, 10),
    "audit_repairs": (0, 10),
    "max_iterations": (1, 99),
}


@dataclass(frozen=True)
class AutonomyBudget:
    """The single autonomous budget of a run: five limits, no hidden knob.

    ``step_attempts`` is the total ceiling of one autonomous operation: the
    primary attempt, its retries and its executor fallbacks all consume it.
    ``audit_repairs`` is the number of writable AUDIT calls one iteration may
    spend.  ``max_iterations`` counts ``M01`` as iteration one.
    ``max_wall_clock_hours`` is measured from the run's durable creation
    timestamp, so a resume never restarts it.  ``max_cost = 0`` disables the
    cost cap; a positive value is refused until a provider publishes an
    explicit cost, because MetaHarness never invents a price.
    """

    step_attempts: int = 3
    audit_repairs: int = 2
    max_iterations: int = 8
    max_wall_clock_hours: float = 12
    max_cost: float = 0

    def __post_init__(self) -> None:
        for name, (minimum, maximum) in _BUDGET_BOUNDS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"budget.{name} must be an integer between {minimum} and {maximum}")
        hours = self.max_wall_clock_hours
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= 720:
            raise ValueError("budget.max_wall_clock_hours must be a number of hours in ]0, 720]")
        cost = self.max_cost
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not 0 <= cost <= 1e9:
            raise ValueError("budget.max_cost must be a number of USD in [0, 1e9]")


def unsupported_cost_cap(max_cost: object) -> str | None:
    """Why a positive cost cap cannot be applied, or ``None`` for ``0``.

    No provider configured by this runtime publishes an explicit ``cost_usd``,
    so a positive cap would be simulated.  It is refused instead.
    """

    if max_cost in (0, 0.0):
        return None
    return (
        "budget.max_cost > 0 is not supported: no configured provider publishes "
        "an explicit cost, and MetaHarness never derives a price from tokens"
    )


def elapsed_hours(started_at: object, *, now: datetime | None = None) -> float | None:
    """Hours a run created at *started_at* has been alive, or ``None``.

    The durable creation timestamp is the only source of elapsed time: an
    in-memory timer would silently restart on a resume.  An unreadable
    timestamp yields ``None`` rather than a fabricated duration.
    """

    if not isinstance(started_at, str) or not started_at.strip():
        return None
    try:
        started = datetime.fromisoformat(started_at.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return ((now or datetime.now(timezone.utc)) - started).total_seconds() / 3600


def wall_clock_exhausted(
    started_at: object, *, max_wall_clock_hours: float, now: datetime | None = None,
) -> bool:
    """Whether a run created at *started_at* spent its wall-clock budget."""

    hours = elapsed_hours(started_at, now=now)
    return hours is not None and hours >= float(max_wall_clock_hours)


def stable_code(failure: object) -> str:
    """The stable code of a failure, without its detail or check ID suffix."""

    if not isinstance(failure, str) or not failure.strip():
        raise ValueError("failure must be a non-empty code")
    return failure.strip().split(":", 1)[0].strip().upper()


# Provider spellings accepted at input boundaries. Durable records use the
# canonical code on the right and never emit these aliases as failure reasons.
_FAILURE_CODE_ALIASES: Mapping[str, str] = {
    "AGENT_AUTH_FAILURE": "EXTERNAL_AUTH_REQUIRED",
    "LLM_401": "EXTERNAL_AUTH_REQUIRED",
    "LLM_403": "EXTERNAL_AUTH_REQUIRED",
    "LLM_AUTH_FAILURE": "EXTERNAL_AUTH_REQUIRED",
}


def canonical_failure_code(failure: object) -> str:
    """Resolve one accepted input alias to its durable stable code."""

    code = stable_code(failure)
    return _FAILURE_CODE_ALIASES.get(code, code)


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

    known = _lookup(canonical_failure_code(failure))
    failure_class = known or FailureClass.FIXABLE
    ladder = RECOVERY_LADDERS[failure_class]
    return RecoveryDecision(
        failure_class, ladder[-1] if exhausted else ladder[0],
        _REASONS[failure_class], known is not None,
    )


__all__ = [
    "AutonomyBudget", "ExecutionFallbacks", "FAILURE_CLASSES", "FailureClass",
    "RECOVERY_LADDERS", "RecoveryDecision", "RecoveryStrategy", "canonical_failure_code",
    "classify_failure",
    "elapsed_hours", "recovery_ladder", "stable_code", "unsupported_cost_cap",
    "wall_clock_exhausted",
]
