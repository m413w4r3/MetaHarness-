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
from typing import Any, Callable, Collection, Mapping, Protocol

from ..evidence import EvidenceBundle
from ..models import (
    GateStage,
    RunDisposition,
    RunEvent,
    RunMachineState,
    RunPhase,
    RunStatus,
    project_run_outcome,
    transition,
)
from ..recovery_policy import (
    FailureClass,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryStrategy,
    classify_failure,
    failure_class_for,
)
from ..state import RunStateStore
from .pipeline_v2 import PipelineFailure, RecoveryStepUnavailable


@dataclass(frozen=True)
class RecoveryTerminalState:
    """One terminal projection: the disposition, and its derived status.

    ``status`` is the RunStatus spelling of the projection; ``disposition``
    and ``phase`` are the durable state it was derived from.
    """

    status: RunStatus
    resumable: bool
    reason: str
    disposition: RunDisposition
    phase: RunPhase | None = None


# Provider credentials are an external waiting condition, never a retry.
_AUTH_ALIASES = frozenset({
    "AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "LLM_AUTH_FAILURE",
    "MISSING_PROVIDER_CREDENTIALS", "PROVIDER_CREDENTIALS_MISSING",
})
# The trace phase of a recovery loop names its durable phase.
_TRACE_CHECKPOINT = {
    "implementation": RunPhase.IMPLEMENT_STEP,
    "workspace setup": RunPhase.WORKTREE_SETUP,
    "preparing": RunPhase.WORKTREE_SETUP,
    "review": RunPhase.FINAL_REVIEW,
    "final_review": RunPhase.FINAL_REVIEW,
    "checks": RunPhase.DETERMINISTIC_GATE,
    "validation": RunPhase.DETERMINISTIC_GATE,
    "check-repair": RunPhase.CHECK_REPAIR,
    "semantic-revision": RunPhase.SEMANTIC_REVISION,
    "candidate_push": RunPhase.CANDIDATE_PUSH,
    "planning": RunPhase.PLANNER,
}
# Bounded durable attempt history; the counters stay the budget authority.
MAX_RECOVERY_ATTEMPT_RECORDS = 256


def failure_code(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("failure reason must be a non-empty reason-code string")
    return reason.split(":", 1)[0].strip().upper()


# The only postures a terminal recovery decision may leave behind.  Every
# finer distinction (which infrastructure is down, which bounded pass a human
# may retry) stays in the failure reason and the phase.
_TERMINAL_DISPOSITION = {
    RecoveryDisposition.HARD_STOP: RunDisposition.FAILED,
    RecoveryDisposition.WAIT_HUMAN: RunDisposition.WAIT_HUMAN,
    RecoveryDisposition.WAIT_EXTERNAL: RunDisposition.WAIT_EXTERNAL,
}


def terminal_state_for(
    decision: RecoveryDecision, *, failure_code: str, phase: RunPhase,
) -> RecoveryTerminalState:
    """Only terminal dispositions may cross the coordinator boundary.

    The decision names one of the five run dispositions; the projected status
    and resumability are derived from it, the phase and the failure reason.
    """

    if not isinstance(failure_code, str) or not failure_code.strip():
        raise TypeError("terminal failure_code must be a non-empty reason-code string")
    code = failure_code.split(":", 1)[0].upper()
    try:
        disposition = _TERMINAL_DISPOSITION[decision.disposition]
    except KeyError as exc:
        raise ValueError(f"recovery disposition {decision.disposition} is not terminal") from exc
    event = (
        RunEvent.fail(reason=code)
        if disposition is RunDisposition.FAILED
        else RunEvent.wait(disposition, reason=code)
    )
    outcome = project_run_outcome(
        transition(RunMachineState(phase, RunDisposition.RUNNING), event)
    )
    return RecoveryTerminalState(
        outcome.status, outcome.resumable, decision.reason,
        outcome.disposition, outcome.phase,
    )


# Terminal ladder steps only: an autonomous step is executed inside its own
# recovery loop and can never cross this boundary.
_TERMINAL_DISPOSITIONS = {
    RecoveryStrategy.HARD_STOP: RecoveryDisposition.HARD_STOP,
    RecoveryStrategy.WAIT_HUMAN: RecoveryDisposition.WAIT_HUMAN,
    RecoveryStrategy.WAIT_EXTERNAL: RecoveryDisposition.WAIT_EXTERNAL,
}


def strategy_terminal_state(
    strategy: RecoveryStrategy, *, failure_code: str, phase: RunPhase,
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
    reason: str, *, phase: RunPhase, remote_required: bool = False,
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


_ATTEMPT_IDENTITY = ("phase", "reason", "attempt", "budget_key", "cycle", "step_id", "tree_before")


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
        # Records without a stable operation_id describe the same semantic
        # operation once per resume; keep only this one.
        identity = tuple(record.get(key) for key in _ATTEMPT_IDENTITY)
        attempts = [
            item for item in attempts
            if not (
                isinstance(item, dict) and "operation_id" not in item
                and tuple(item.get(key) for key in _ATTEMPT_IDENTITY) == identity
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
        checkpoint_phase: RunPhase | None = None,
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


# -- the deterministic-gate ladder interface ----------------------------------


@dataclass(frozen=True)
class GateRecoveryStep:
    """One distinct ladder strategy a red deterministic gate must try next.

    ``repair_attempt`` names the bounded check-repair pass this step consumes
    when the step is a repair pass; a step that re-executes approved cycle work
    names the plan indices it replays instead.  ``exhausted`` is true only once
    every strategy of the failure class is consumed or deterministically
    inapplicable for these exact facts; the gate may then wait for an operator.
    """

    strategy: RecoveryStrategy
    tree: str = ""
    failed_check_ids: tuple[str, ...] = ()
    repair_attempt: int | None = None
    step_indices: tuple[int, ...] = ()
    added_paths: tuple[str, ...] = ()
    consumed: tuple[RecoveryStrategy, ...] = ()
    exhausted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", RecoveryStrategy(self.strategy))
        if not isinstance(self.tree, str) or len(self.tree) > 128:
            raise ValueError("gate recovery step tree must be a bounded string")
        object.__setattr__(
            self, "failed_check_ids", tuple(str(item) for item in self.failed_check_ids),
        )
        if self.repair_attempt is not None and (
            isinstance(self.repair_attempt, bool)
            or not isinstance(self.repair_attempt, int)
            or self.repair_attempt < 1
        ):
            raise ValueError("gate recovery repair attempt must be a positive integer")
        for name in ("step_indices", "added_paths", "consumed"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.step_indices
        ):
            raise ValueError("gate recovery step indices must be non-negative integers")
        if any(
            not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
            for path in self.added_paths
        ):
            raise ValueError("gate recovery added paths must be safe repository paths")
        object.__setattr__(self, "consumed", tuple(
            RecoveryStrategy(strategy) for strategy in self.consumed
        ))
        if not isinstance(self.exhausted, bool):
            raise TypeError("gate recovery exhausted flag must be a boolean")
        if self.exhausted and (self.step_indices or self.repair_attempt is not None):
            raise ValueError("an exhausted gate recovery step executes nothing")

    @property
    def is_repair_pass(self) -> bool:
        """Whether this step is executed by the bounded check-repair operation."""

        return self.repair_attempt is not None

    @property
    def fingerprint_strategy(self) -> str:
        """The exact ladder position a fixed-point fingerprint carries.

        The identity of one exhausted gate episode is its consumed ladder
        trail, never only its last rung: two episodes that reached different
        positions are different facts even on the same candidate tree.
        """

        trail = self.consumed if self.exhausted else (*self.consumed, self.strategy)
        return "+".join(item.value for item in trail)


class RecoveryOperations(Protocol):
    """The single ladder object :class:`PipelineV2Operations` references.

    The machine asks it which distinct strategy one red deterministic gate must
    try next, then reports the step as started and finished.  The
    implementation owns the durable ladder ledger, so a strategy consumed for
    an exact candidate tree and failure is never proposed twice, and it never
    widens the scope authority the operator already approved.
    """

    def gate_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        evidence: EvidenceBundle, repair_attempt: int, repair_budget: int,
    ) -> GateRecoveryStep:
        """The next distinct ladder step of one red gate, or the terminal."""

        ...

    def begin_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> None:
        """Consume the step durably before anything is executed for it."""

        ...

    def replay_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> str:
        """Execute one replan rung on approved work; return the tree after it.

        The implementation re-executes only steps of the approved cycle plan,
        under their own approved contracts and effective authority, and raises
        :class:`RecoveryStepUnavailable` when the rung cannot be executed
        deterministically.  It never widens a scope.
        """

        ...

    def finish_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle, tree_after: str,
    ) -> None:
        """Mark the consumed step as executed, without consuming it twice."""

        ...


__all__ = [
    "MAX_RECOVERY_ATTEMPT_RECORDS", "RecoveryAdmission", "RecoveryAttempt",
    "GateRecoveryStep", "RecoveryCoordinator", "RecoveryOperations",
    "RecoveryStepUnavailable",
    "RecoveryTerminalState", "failure_code",
    "normalize_exit_reason", "project_exit", "strategy_terminal_state",
    "terminal_state_for",
]
