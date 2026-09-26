"""The generic recovery coordinator of one run.

:func:`metaharness.recovery_policy.classify_failure` classifies stable failure
codes.  :class:`RecoveryCoordinator` applies that classification: it owns the
durable recovery budgets, records every consumed attempt with its tree
boundary, emits the ``recovery.*`` trace and projects a disposition that left
its recovery loop onto one durable run status.

It never runs a model, interprets the SPEC, chooses a mutable scope, repairs a
tree or touches an authority artifact: phase services own those actions and
ask this coordinator only whether a bounded automatic recovery is admitted.
Its ladder vocabulary stays inside
:mod:`metaharness.recovery_policy`: this module only projects a ladder terminal
onto the durable run status that waits after it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Collection, Mapping

from ..models import RunStatus
from ..recovery_policy import (
    FailureClass,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryStrategy,
    classify_failure,
    failure_class_for,
)
from ..resume import ResumePhase
from ..state import RunStateStore
from .pipeline_v2 import PipelineFailure


@dataclass(frozen=True)
class RecoveryTerminalState:
    status: RunStatus
    resumable: bool
    reason: str


_CHECK_INFRA = frozenset({
    "CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE",
    "CHECK_INFRA_RETRIES_EXHAUSTED", "CHECK_INFRASTRUCTURE_UNAVAILABLE",
    "CHECK_SIDE_EFFECT_REPEATED", "CHECK_SIDE_EFFECT_UNSTABLE",
    "WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT",
})
_REMOTE = frozenset({
    "PUSH_FAILED", "CANDIDATE_PUSH_FAILED", "CANDIDATE_REMOTE_UNAVAILABLE",
    "REMOTE_UNAVAILABLE", "REMOTE_TEMPORARILY_UNAVAILABLE",
})
# Provider credentials are an external waiting condition, never a retry.
_AUTH_ALIASES = frozenset({
    "AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "LLM_AUTH_FAILURE",
    "MISSING_PROVIDER_CREDENTIALS", "PROVIDER_CREDENTIALS_MISSING",
})
# The trace phase of a recovery loop names its durable checkpoint phase.
_TRACE_CHECKPOINT = {
    "implementation": ResumePhase.IMPLEMENT_STEP,
    "workspace setup": ResumePhase.WORKTREE_SETUP,
    "preparing": ResumePhase.WORKTREE_SETUP,
    "review": ResumePhase.FINAL_REVIEW,
    "final_review": ResumePhase.FINAL_REVIEW,
    "checks": ResumePhase.DETERMINISTIC_GATE,
    "validation": ResumePhase.DETERMINISTIC_GATE,
    "check-repair": ResumePhase.CHECK_REPAIR,
    "semantic-revision": ResumePhase.SEMANTIC_REVISION,
    "candidate_push": ResumePhase.CANDIDATE_PUSH,
    "planning": ResumePhase.PLANNER,
}
# Bounded durable attempt history; the counters stay the budget authority.
MAX_RECOVERY_ATTEMPT_RECORDS = 256


def failure_code(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("failure reason must be a non-empty reason-code string")
    return reason.split(":", 1)[0].strip().upper()


def terminal_state_for(
    decision: RecoveryDecision, *, failure_code: str, phase: ResumePhase,
) -> RecoveryTerminalState:
    """Only terminal dispositions may cross the coordinator boundary."""

    if not isinstance(failure_code, str) or not failure_code.strip():
        raise TypeError("terminal failure_code must be a non-empty reason-code string")
    code = failure_code.split(":", 1)[0].upper()
    if decision.disposition is RecoveryDisposition.HARD_STOP:
        return RecoveryTerminalState(RunStatus.FAILED, False, decision.reason)
    if decision.disposition is RecoveryDisposition.WAIT_HUMAN:
        if code == "STEP_CONTRACT_REPAIR_OUTPUT_INVALID":
            # The pending repair slot is intact: its planner can be retried.
            return RecoveryTerminalState(RunStatus.WAITING_CONTRACT_REPAIR, True, decision.reason)
        if code == "CHECK_REPAIR_EXHAUSTED":
            # The gate checkpoint and candidate are intact; retry runs the gate
            # only and cannot admit another worker without a new authority.
            return RecoveryTerminalState(RunStatus.WAITING_CHECK_REPAIR, True, decision.reason)
        return RecoveryTerminalState(RunStatus.WAITING_HUMAN, False, decision.reason)
    if decision.disposition is RecoveryDisposition.WAIT_EXTERNAL:
        if code in _CHECK_INFRA:
            status = RunStatus.WAITING_CHECK_INFRASTRUCTURE
        elif code in _REMOTE and phase in {ResumePhase.CANDIDATE_PUSH, ResumePhase.PUBLISH}:
            status = RunStatus.WAITING_REMOTE
        else:
            status = RunStatus.WAITING_EXTERNAL
        return RecoveryTerminalState(status, True, decision.reason)
    raise ValueError(f"recovery disposition {decision.disposition} is not terminal")


# Terminal ladder steps only: an autonomous step is executed inside its own
# recovery loop and can never cross this boundary.
_TERMINAL_DISPOSITIONS = {
    RecoveryStrategy.HARD_STOP: RecoveryDisposition.HARD_STOP,
    RecoveryStrategy.WAIT_HUMAN: RecoveryDisposition.WAIT_HUMAN,
    RecoveryStrategy.WAIT_EXTERNAL: RecoveryDisposition.WAIT_EXTERNAL,
}


def strategy_terminal_state(
    strategy: RecoveryStrategy, *, failure_code: str, phase: ResumePhase,
) -> RecoveryTerminalState:
    """Project one ladder terminal onto the durable status that waits after it.

    An autonomous step is refused: the ladder may only end a run on a terminal
    step the existing authority already allowed for that failure code.
    """

    if not isinstance(strategy, RecoveryStrategy) or not strategy.terminal:
        raise ValueError(f"recovery strategy {strategy!r} is not terminal")
    decision = RecoveryDecision(
        _TERMINAL_DISPOSITIONS[strategy],
        "recovery ladder reached a terminal step", False, False,
        failure_class_for(failure_code), strategy,
    )
    return terminal_state_for(decision, failure_code=failure_code, phase=phase)


def project_exit(
    reason: str, *, phase: ResumePhase, remote_required: bool = False,
) -> tuple[RecoveryDecision, RecoveryTerminalState]:
    """Project a failure that left its recovery loop onto a durable status.

    Every automatic recovery is consumed inside its own loop, so a failure
    reaching this boundary is classified as budget-exhausted.  A disposition
    that is still not terminal escaped its coordinator and fails closed.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("project_exit expects a stable reason-code string")
    decision = classify_failure(
        reason, budget_exhausted=True,
        remote_required=remote_required, remote_unavailable=remote_required,
    )
    try:
        return decision, terminal_state_for(decision, failure_code=reason, phase=phase)
    except ValueError:
        decision = RecoveryDecision(
            RecoveryDisposition.HARD_STOP,
            "recovery operation escaped its coordinator", False, False,
        )
        return decision, terminal_state_for(decision, failure_code=reason, phase=phase)


def normalize_exit_reason(reason: str) -> str:
    """Collapse provider credential aliases onto one waiting condition."""

    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("exit reason must be a non-empty reason-code string")
    return "EXTERNAL_AUTH_REQUIRED" if failure_code(reason) in _AUTH_ALIASES else reason


@dataclass(frozen=True)
class RecoveryAttempt:
    """The durable identity of one consumed automatic recovery attempt."""

    phase: str
    reason: str
    attempt: int
    budget_key: str
    budget: int
    budget_consumed: int
    disposition: str
    operation_id: str
    cycle: int | None = None
    step_id: str | None = None
    profile_id: str | None = None
    tree_before: str | None = None
    tree_after: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operation_id, str)
            or not self.operation_id
            or len(self.operation_id) > 256
        ):
            raise ValueError("recovery attempt operation_id must be a bounded non-empty string")


def _attempt_record(attempt: RecoveryAttempt, **overrides: Any) -> dict[str, Any]:
    return {**asdict(attempt), **overrides}


_LEGACY_IDENTITY = ("phase", "reason", "attempt", "budget_key", "cycle", "step_id", "tree_before")


@dataclass(frozen=True)
class RecoveryAdmission:
    """Whether one bounded recovery was admitted, and its budget position."""

    admitted: bool
    decision: RecoveryDecision
    attempt: int
    used: int
    budget: int
    reason: str
    phase: str
    cycle: int | None = None
    step_id: str | None = None
    tree_before: str | None = None
    exhausted: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)


TraceEmit = Callable[..., None]


class RecoveryCoordinator:
    """Apply recovery decisions to one run's durable budgets and trace."""

    def __init__(self, store: RunStateStore, *, emit: TraceEmit) -> None:
        self._store = store
        self._emit = emit

    # -- durable budgets -------------------------------------------------

    @staticmethod
    def budget_key(kind: str, *parts: object) -> str:
        """One budget per recovery kind and boundary; never per resume."""

        return ":".join((kind, *(str(part) for part in parts)))

    def used(self, key: str) -> int:
        counters = self._store.load().get("recovery_counters", {})
        return self._counter_value(counters, key)

    def consume(self, key: str, attempt: RecoveryAttempt) -> int:
        state = self._store.load()
        counters = state.get("recovery_counters", {})
        value = self._counter_value(counters, key) + 1
        attempts = state.get("recovery_attempts", [])
        if not isinstance(attempts, list):
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery attempts are malformed")
        record = _attempt_record(attempt, budget_consumed=value)
        self._store.update(
            status=state.get("status", RunStatus.VALIDATING),
            recovery_counters={**counters, key: value},
            recovery_attempts=[*attempts, record][-MAX_RECOVERY_ATTEMPT_RECORDS:],
        )
        return value

    def record(self, attempt: RecoveryAttempt) -> None:
        """Record an attempt whose budget authority is a durable artifact."""

        state = self._store.load()
        attempts = state.get("recovery_attempts", [])
        if not isinstance(attempts, list):
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery attempts are malformed")
        record = _attempt_record(attempt)
        for item in attempts:
            if isinstance(item, dict) and item.get("operation_id") == attempt.operation_id:
                if item != record:
                    raise PipelineFailure(
                        "DURABLE_ARTIFACT_CORRUPTED",
                        "recovery operation_id was reused with different attempt data",
                    )
                return
        # Records written before stable identities existed describe the same
        # semantic operation once per resume; keep only this one.
        identity = tuple(record.get(key) for key in _LEGACY_IDENTITY)
        attempts = [
            item for item in attempts
            if not (
                isinstance(item, dict) and "operation_id" not in item
                and tuple(item.get(key) for key in _LEGACY_IDENTITY) == identity
            )
        ]
        self._store.update(
            status=state.get("status", RunStatus.VALIDATING),
            recovery_attempts=[*attempts, record][-MAX_RECOVERY_ATTEMPT_RECORDS:],
        )

    @staticmethod
    def _counter_value(counters: Any, key: str) -> int:
        if not isinstance(counters, dict):
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counters are malformed")
        value = counters.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counter is malformed")
        return value

    # -- admission -------------------------------------------------------

    def admit(
        self,
        key: str,
        *,
        reason: str,
        budget: int,
        phase: str,
        cycle: int | None = None,
        step_id: str | None = None,
        profile_id: str | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        decision: RecoveryDecision | None = None,
        allowed: Collection[RecoveryDisposition] | None = None,
        facts: Mapping[str, Any] | None = None,
    ) -> RecoveryAdmission:
        """Classify one failure and consume its budget when recovery is allowed.

        ``allowed`` restricts the dispositions the caller can execute; any
        other disposition is returned unadmitted without consuming budget.
        """

        facts = dict(facts or {})
        used = self.used(key)
        attempt = used + 1
        decision = decision or classify_failure(reason, **facts)
        common = {
            "attempt": attempt, "tree_before": tree_before, "tree_after": tree_after,
            "phase": phase, "cycle": cycle, "step_id": step_id,
        }
        self.trace(
            "recovery.classified", reason=reason, decision=decision,
            budget_remaining=max(0, budget - used), **common,
        )
        admission = {
            "attempt": attempt, "budget": budget, "reason": reason, "phase": phase,
            "cycle": cycle, "step_id": step_id, "tree_before": tree_before,
        }
        if allowed is not None and decision.disposition not in allowed:
            return RecoveryAdmission(False, decision, used=used, **admission)
        if used >= budget:
            exhausted = classify_failure(reason, **{**facts, "budget_exhausted": True})
            self.trace(
                "recovery.exhausted", reason=reason, decision=exhausted,
                budget_remaining=0, **common,
            )
            return RecoveryAdmission(False, exhausted, used=used, exhausted=True, **admission)
        consumed = self.consume(key, RecoveryAttempt(
            phase=phase, reason=reason, attempt=attempt, budget_key=key,
            budget=budget, budget_consumed=used + 1,
            disposition=decision.disposition.value,
            operation_id=f"recovery:{key}:{attempt:02d}",
            cycle=cycle, step_id=step_id,
            profile_id=profile_id, tree_before=tree_before, tree_after=tree_after,
        ))
        self.trace(
            "recovery.started", reason=reason, decision=decision,
            budget_remaining=max(0, budget - consumed), **common,
        )
        return RecoveryAdmission(True, decision, used=consumed, **admission)

    def complete(
        self, admission: RecoveryAdmission, *, recovered: bool, tree_after: str | None,
    ) -> None:
        self.trace(
            "recovery.completed", reason=admission.reason, decision=admission.decision,
            attempt=admission.attempt, tree_before=admission.tree_before,
            tree_after=tree_after, budget_remaining=admission.remaining,
            phase=admission.phase, cycle=admission.cycle, step_id=admission.step_id,
            recovered=recovered,
        )

    def fallback_selected(
        self,
        *,
        reason: str,
        index: int,
        available: int,
        phase: str,
        cycle: int | None = None,
        step_id: str | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
    ) -> RecoveryDecision:
        decision = RecoveryDecision(
            RecoveryDisposition.FALLBACK_EXECUTOR,
            "primary executor transient retry budget exhausted", True, False,
            FailureClass.EXTERNAL, RecoveryStrategy.FALLBACK_EXECUTOR,
        )
        self.trace(
            "recovery.executor_selected", reason=reason, decision=decision,
            attempt=index, tree_before=tree_before, tree_after=tree_after,
            budget_remaining=max(0, available - index), phase=phase,
            cycle=cycle, step_id=step_id,
        )
        return decision

    def stop(
        self,
        reason: str,
        *,
        phase: str,
        attempt: int = 1,
        cycle: int | None = None,
        step_id: str | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        facts: Mapping[str, Any] | None = None,
    ) -> RecoveryDecision:
        """Record a failure that no automatic recovery may handle."""

        decision = classify_failure(reason, **dict(facts or {}))
        self.trace(
            "recovery.classified", reason=reason, decision=decision,
            attempt=max(1, attempt), tree_before=tree_before, tree_after=tree_after,
            budget_remaining=0, phase=phase, cycle=cycle, step_id=step_id,
        )
        return decision

    # -- trace -----------------------------------------------------------

    def trace(
        self,
        event: str,
        *,
        reason: str,
        decision: RecoveryDecision,
        attempt: int,
        tree_before: str | None,
        tree_after: str | None,
        budget_remaining: int,
        phase: str,
        cycle: int | None = None,
        step_id: str | None = None,
        recovered: bool | None = None,
        terminal_status: RunStatus | None = None,
        checkpoint_phase: ResumePhase | None = None,
    ) -> None:
        """Record one secret-free recovery transition with its tree boundary."""

        data: dict[str, Any] = {
            "reason": reason,
            "disposition": decision.disposition.value,
            "attempt": attempt,
            "tree_before": tree_before,
            "tree_after": tree_after,
            "budget_remaining": budget_remaining,
        }
        if recovered is not None:
            data["recovered"] = recovered
        if event == "recovery.exhausted":
            checkpoint_phase = checkpoint_phase or _TRACE_CHECKPOINT.get(phase)
            if terminal_status is None and checkpoint_phase is not None:
                try:
                    terminal_status = terminal_state_for(
                        decision, failure_code=reason, phase=checkpoint_phase,
                    ).status
                except ValueError:
                    terminal_status = RunStatus.FAILED
            data["terminal_disposition"] = decision.disposition.value
            data["terminal_status"] = (
                terminal_status.value if terminal_status else RunStatus.FAILED.value
            )
            data["checkpoint_phase"] = checkpoint_phase.value if checkpoint_phase else phase
        self._emit(event, phase=phase, cycle=cycle, step_id=step_id, data=data)


__all__ = [
    "MAX_RECOVERY_ATTEMPT_RECORDS", "RecoveryAdmission", "RecoveryAttempt",
    "RecoveryCoordinator", "RecoveryTerminalState", "failure_code",
    "normalize_exit_reason", "project_exit", "strategy_terminal_state",
    "terminal_state_for",
]
