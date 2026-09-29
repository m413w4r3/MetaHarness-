"""The generic recovery coordinator of one run.

:class:`RecoveryCoordinator` applies :func:`~metaharness.recovery_policy.classify_failure`:
it owns the durable recovery budgets, records every consumed attempt with its
tree boundary, emits the ``recovery.*`` trace and projects the last ladder step
of a failure that left its loop onto one durable run status.  Phase services
own every action; this module only admits or refuses a bounded recovery.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

from ..agent.base import AGENT_RUNTIME_FAILED, AGENT_TIMEOUT
from ..attempt_transaction import (
    AttemptBoundary,
    AttemptViolation,
    CandidateAttemptTransaction,
    GitOwnership,
    contain_trusted_process,
)
from ..baseline import PREFLIGHT_FILE, PREFLIGHT_SCHEMA_VERSION, preflight_fingerprint
from ..evidence import EvidenceBundle
from ..gitops import snapshot_candidate_state
from ..models import (
    CheckConfig,
    HarnessConfig,
    ImplementationStep,
    RunDisposition,
    RunEvent,
    RunMachineState,
    RunPhase,
    RunStatus,
    project_run_outcome,
    transition,
)
from ..recovery_policy import (
    RecoveryDecision, RecoveryStrategy, canonical_failure_code, classify_failure,
)
from ..result import ResultArtifactError, atomic_write_text
from ..scope import ScopePolicy
from ..state import RunStateStore
from ..validation import run_check_preflights
from ..workspace import WorkspaceSetupError
from .pipeline_v2 import BudgetExhausted, PipelineFailure
from .shared import (
    OrchestrationError,
    StepExecutionFailure,
    archive_attempt,
    archive_attempt_tree,
    safe_candidate_tree,
    safe_status,
)


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


# The trace phase of a recovery loop names its durable phase.
_TRACE_CHECKPOINT = {
    "implementation": RunPhase.IMPLEMENT_STEP,
    "workspace setup": RunPhase.WORKTREE_SETUP,
    "preparing": RunPhase.WORKTREE_SETUP,
    "checks": RunPhase.DETERMINISTIC_GATE,
    "validation": RunPhase.DETERMINISTIC_GATE,
    "audit": RunPhase.AUDIT,
    "candidate_push": RunPhase.CANDIDATE_PUSH,
    "planning": RunPhase.PLANNER,
}
# Bounded durable attempt history; the counters stay the budget authority.
MAX_RECOVERY_ATTEMPT_RECORDS = 256


def failure_code(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("failure reason must be a non-empty reason-code string")
    return reason.split(":", 1)[0].strip().upper()


# The only terminal run postures in a recovery ladder.  MARK_FAILED_CONTINUE
# is consumed by its phase owner and is never a run disposition.
_TERMINAL_RUN_DISPOSITIONS = {
    RecoveryStrategy.HARD_STOP: RunDisposition.FAILED,
    RecoveryStrategy.WAIT_HUMAN: RunDisposition.WAIT_HUMAN,
    RecoveryStrategy.WAIT_EXTERNAL: RunDisposition.WAIT_EXTERNAL,
}


def terminal_state_for(
    decision: RecoveryDecision, *, failure_code: str, phase: RunPhase,
) -> RecoveryTerminalState:
    """Project a terminal ladder step onto a durable run status.

    MARK_FAILED_CONTINUE belongs to phase-local unit handling and cannot be
    represented as a terminal run posture.
    """

    if not isinstance(failure_code, str) or not failure_code.strip():
        raise TypeError("terminal failure_code must be a non-empty reason-code string")
    if not isinstance(decision, RecoveryDecision):
        raise TypeError("terminal projection requires a recovery decision")
    disposition = _TERMINAL_RUN_DISPOSITIONS.get(decision.strategy)
    if disposition is None:
        raise ValueError(
            f"recovery strategy {decision.strategy.value} is not a terminal run disposition"
        )
    code = failure_code.split(":", 1)[0].upper()
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


def project_exit(
    reason: str, *, phase: RunPhase,
) -> tuple[RecoveryDecision, RecoveryTerminalState]:
    """Project a failure that left its recovery loop onto a durable status.

    Terminal strategies are projected here. MARK_FAILED_CONTINUE must already
    have been consumed by a phase-local continuable unit before this boundary.
    """

    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("project_exit expects a stable reason-code string")
    decision = classify_failure(reason, exhausted=True)
    return decision, terminal_state_for(decision, failure_code=reason, phase=phase)


def normalize_exit_reason(reason: str) -> str:
    """Collapse provider credential aliases onto one waiting condition."""

    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("exit reason must be a non-empty reason-code string")
    code = failure_code(reason)
    canonical = canonical_failure_code(code)
    return canonical if canonical != code else reason


@dataclass(frozen=True)
class RecoveryAttempt:
    """The durable identity of one consumed automatic recovery attempt."""

    phase: str
    reason: str
    attempt: int
    budget_key: str
    budget: int
    budget_consumed: int
    strategy: str
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
        self._store.update_metadata(
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
        self._store.update_metadata(
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
        allowed: Collection[RecoveryStrategy] | None = None,
        strategy: RecoveryStrategy | None = None,
    ) -> RecoveryAdmission:
        """Classify one failure and consume its budget when recovery is allowed.

        ``allowed`` restricts the ladder strategies the caller can execute; a
        terminal decision, or one naming any other strategy, is returned
        unadmitted without consuming budget.  ``strategy`` records which rung
        the caller executes when it is not the first one of the ladder.
        """

        used = self.used(key)
        attempt = used + 1
        decision = classify_failure(reason)
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
        if decision.strategy.terminal or (allowed is not None and decision.strategy not in allowed):
            return RecoveryAdmission(False, decision, used=used, **admission)
        if used >= budget:
            exhausted = classify_failure(reason, exhausted=True)
            self.trace(
                "recovery.exhausted", reason=reason, decision=exhausted,
                budget_remaining=0, **common,
            )
            return RecoveryAdmission(False, exhausted, used=used, exhausted=True, **admission)
        consumed = self.consume(key, RecoveryAttempt(
            phase=phase, reason=reason, attempt=attempt, budget_key=key,
            budget=budget, budget_consumed=used + 1,
            strategy=(strategy or decision.strategy).value,
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
            classify_failure(reason).failure_class, RecoveryStrategy.FALLBACK_EXECUTOR,
            "primary executor retry budget exhausted",
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
    ) -> RecoveryDecision:
        """Record a failure that left every automatic recovery loop."""

        decision = classify_failure(reason, exhausted=True)
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
        """Record one secret-free recovery transition with its tree boundary.

        A code the policy does not name is also reported, once, as
        ``recovery.unclassified_code``; it still walks the ``FIXABLE`` ladder.
        """

        if not decision.known:
            self._emit(
                "recovery.unclassified_code", phase=phase, cycle=cycle, step_id=step_id,
                data={"code": failure_code(reason)}, once=True,
            )
        data: dict[str, Any] = {
            "reason": reason,
            "strategy": decision.strategy.value,
            "failure_class": decision.failure_class.value,
            "attempt": attempt,
            "tree_before": tree_before,
            "tree_after": tree_after,
            "budget_remaining": budget_remaining,
        }
        if recovered is not None:
            data["recovered"] = recovered
        if event == "recovery.exhausted":
            checkpoint_phase = checkpoint_phase or _TRACE_CHECKPOINT.get(phase)
            if (
                terminal_status is None
                and checkpoint_phase is not None
                and decision.strategy in _TERMINAL_RUN_DISPOSITIONS
            ):
                try:
                    terminal_status = terminal_state_for(
                        decision, failure_code=reason, phase=checkpoint_phase,
                    ).status
                except ValueError:
                    terminal_status = None
            data["terminal_status"] = terminal_status.value if terminal_status else None
            data["checkpoint_phase"] = checkpoint_phase.value if checkpoint_phase else phase
        self._emit(event, phase=phase, cycle=cycle, step_id=step_id, data=data)


TRANSIENT_WORKER_FAILURES = frozenset({
    AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
})
# The only ladder step this recovery loop may execute: a same-executor retry on
# the exact pre-attempt tree it just restored.
_RETRY_STRATEGIES = frozenset({RecoveryStrategy.RETRY_TARGETED})


class WorkerRecovery:
    """Retry and fallback admission for one run's untrusted worker attempts."""

    def __init__(
        self,
        recovery: RecoveryCoordinator,
        *,
        store: RunStateStore,
        secrets: Sequence[str],
        scope: ScopePolicy,
    ) -> None:
        self._recovery = recovery
        self._store = store
        self._secrets = tuple(secrets)
        self._scope = scope

    # -- implementation steps -------------------------------------------

    def admit_step_retry(
        self,
        *,
        failure: StepExecutionFailure,
        retry_key: str,
        repo: Path,
        worktree: Path,
        branch_ref: str,
        base_sha: str,
        ownership_before: GitOwnership,
        step: ImplementationStep,
        artifact_dir: Path,
        cycle: int,
        retry_budget: int,
    ) -> RecoveryAdmission | None:
        """Admit a same-executor retry only after an exact, in-scope rollback.

        ``retry_budget`` is how many attempts this step may still spend after
        its primary one: the retry ladder rung is refused once the step's one
        ``step_attempts`` budget is consumed.  On refusal ``failure`` carries
        the stable reason the run must project.
        """

        reason = normalize_exit_reason(failure.reason)
        failure.reason = reason
        before = failure.tree_before
        trace = {"phase": "implementation", "cycle": cycle, "step_id": failure.step_id}
        attempt = self._recovery.used(retry_key) + 1
        if classify_failure(reason).strategy.terminal:
            return None

        def refuse(code: str, detail: str, tree_after: str | None) -> None:
            if code == "RESUME_REQUIRES_OPERATOR":
                failure.detail = (failure.detail or "worker failed") + "; " + detail
            else:
                failure.detail = detail
            failure.reason = code
            failure.tree_after = tree_after
            failure.step_dir = artifact_dir
            self._recovery.stop(
                code, attempt=attempt, tree_before=before, tree_after=tree_after, **trace,
            )

        if before is None:
            refuse(
                "RESUME_REQUIRES_OPERATOR", "pre-attempt tree identity is unavailable",
                failure.tree_after or safe_candidate_tree(worktree),
            )
            return None
        if failure.tree_after is None:
            refuse("RESUME_REQUIRES_OPERATOR", "post-attempt tree identity is unavailable", None)
            return None
        if safe_status(worktree) is None:
            refuse("RESUME_REQUIRES_OPERATOR", "post-attempt status is unreadable", failure.tree_after)
            return None
        if failure.status_before is None:
            refuse("RESUME_REQUIRES_OPERATOR", "pre-attempt status is unavailable", failure.tree_after)
            return None
        transaction = CandidateAttemptTransaction(
            repo, worktree, AttemptBoundary(before, tuple(failure.status_before), ownership_before),
            branch_ref=branch_ref, base_sha=base_sha, secrets=self._secrets,
        )
        try:
            transaction.abort(
                set((*step.write_set, *step.create_set, *step.delete_set)),
            )
        except AttemptViolation as violation:
            refuse(violation.code, violation.detail, safe_candidate_tree(worktree))
            return None

        failure.tree_after = before
        failure.index_tree_after = before
        if reason == "EXTERNAL_AUTH_REQUIRED":
            # The same credentials can never succeed: an external change is due.
            failure.step_dir = artifact_dir
            return None
        admission = self._recovery.admit(
            retry_key, reason=reason, budget=retry_budget,
            profile_id=failure.profile_id, tree_before=before, tree_after=before,
            allowed=_RETRY_STRATEGIES, **trace,
        )
        if not admission.admitted:
            if admission.exhausted:
                failure.detail = (
                    (failure.detail or "agent failure") + "; retry budget exhausted"
                )
            failure.step_dir = artifact_dir
            return None
        self._store.update_metadata(current_step=step.id)
        archive_attempt(artifact_dir)
        return admission


_INFRA_FAILURE_PREFIXES = (
    "CHECK_INFRASTRUCTURE_UNAVAILABLE:", "CHECK_INFRASTRUCTURE_UNAVAILABLE:",
    "CHECK_SIDE_EFFECT_REPEATED:", "CHECK_SIDE_EFFECT_REPEATED:",
)


def _field(check_result: Any, name: str, default: Any = None) -> Any:
    if isinstance(check_result, Mapping):
        return check_result.get(name, default)
    return getattr(check_result, name, default)


class CheckInfrastructureRecovery:
    """Bounded same-tree retries of trusted check infrastructure."""

    def __init__(
        self, recovery: RecoveryCoordinator, *, store: RunStateStore, attempts: int,
        budget_exhausted: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._recovery = recovery
        self._store = store
        # The one ``budget.step_attempts`` ceiling of the run: a retryable
        # trusted operation is bounded by it, never by a second hidden budget.
        self._attempts = attempts
        # The run's global budget guard, consulted before a costly retry.
        self._budget_exhausted = budget_exhausted

    def _guard(self) -> None:
        reason = self._budget_exhausted()
        if reason is not None:
            raise BudgetExhausted(reason)

    def prepare_workspace(
        self, *, worktree: Path, run_dir: Path, run_setup: Callable[[], Sequence[Any]],
    ) -> Sequence[Any]:
        """Run trusted workspace setup; retry transient failures exactly."""

        key = self._recovery.budget_key("workspace-setup")
        budget = self._attempts
        pending: RecoveryAdmission | None = None
        while True:
            self._guard()
            before = snapshot_candidate_state(worktree)
            try:
                results = run_setup()
                error: WorkspaceSetupError | None = None
            except WorkspaceSetupError as exc:
                results, error = exc.results, exc
            try:
                after = contain_trusted_process(worktree, before, label="workspace setup")
            except AttemptViolation as violation:
                raise OrchestrationError(f"{violation.code}: {violation.detail}") from None
            tree_after = (after.after if after else before).candidate_tree
            if error is None:
                if pending is not None:
                    self._recovery.complete(pending, recovered=True, tree_after=tree_after)
                return results
            if error.results:
                self._store.update_metadata(
                    workspace_setup=[asdict(result) for result in error.results],
                )
            admission = self._recovery.admit(
                key, reason=error.code, budget=budget, phase="workspace setup",
                tree_before=before.candidate_tree, tree_after=tree_after,
            )
            if not admission.admitted:
                raise OrchestrationError(
                    f"CHECK_INFRASTRUCTURE_UNAVAILABLE: {error.code}; "
                    "workspace setup attempts exhausted"
                ) from error
            archive_attempt_tree(run_dir / "setup")
            pending = admission

    def run_preflights(
        self,
        *,
        run_dir: Path,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Evaluate every trusted preflight exactly once per run.

        The verdict is durable, so Docker availability is never probed again on
        a resume or a later cycle.  A failed preflight means the check does not
        run: its ID is returned for the gate to record as ``SKIPPED_INFRA``
        with a durable warning.  Only a check that explicitly declares itself
        ``blocking`` turns that condition into the external wait the ladder
        already owns.
        """

        try:
            selected = check_config.select_checks(check_ids)
        except ValueError as exc:
            raise OrchestrationError("CHECK_INFRASTRUCTURE_UNAVAILABLE: trusted check selection invalid") from exc
        store = _PreflightStore(run_dir)
        skipped: list[str] = []
        warnings: list[str] = []
        for check in selected:
            if not check.preflight_argv:
                continue
            fingerprint = preflight_fingerprint(check)
            verdict = store.verdict(check.id, fingerprint)
            if verdict is None:
                verdict = self._evaluate_preflight(
                    worktree=worktree, check_config=check_config, check=check,
                )
                store.record(check.id, fingerprint, verdict)
            if verdict["status"] == "PASS":
                continue
            if check.blocking:
                raise OrchestrationError(
                    "CHECK_INFRASTRUCTURE_UNAVAILABLE: CHECK_INFRASTRUCTURE_UNAVAILABLE:"
                    f"{check.id}; the check declares itself blocking"
                )
            skipped.append(check.id)
            warnings.append(
                f"check:{check.id}:PREFLIGHT_FAILED: skipped with SKIPPED_INFRA"
            )
        if warnings:
            self._store.update_metadata(check_warnings=warnings)
        return tuple(skipped)

    def _evaluate_preflight(
        self, *, worktree: Path, check_config: HarnessConfig, check: CheckConfig,
    ) -> Mapping[str, Any]:
        """Run one preflight once, contained, and describe its verdict."""

        before = snapshot_candidate_state(worktree)
        failures = run_check_preflights(worktree, check_config, [check.id])
        after = snapshot_candidate_state(worktree)
        if before != after:
            # run_check_preflights restores its own side effects.
            raise OrchestrationError(
                "ROLLBACK_FAILED: preflight did not preserve candidate state"
            )
        if not failures:
            return {"status": "PASS", "exit_code": 0, "detail": ""}
        failure = failures[0]
        if not failure.startswith("CHECK_INFRASTRUCTURE_UNAVAILABLE:"):
            # A preflight that mutates the candidate is an integrity problem,
            # never an infrastructure condition to skip.
            raise OrchestrationError(failure)
        return {"status": "FAIL", "exit_code": -1, "detail": failure}

    def gate_retries(self, *, cycle: int, stage: str, worktree: Path) -> "GateInfraRetries":
        return GateInfraRetries(self, cycle=cycle, stage=stage, worktree=worktree)


class GateInfraRetries:
    """The same-tree infrastructure retry callback of one deterministic gate."""

    def __init__(
        self, owner: CheckInfrastructureRecovery, *, cycle: int, stage: str, worktree: Path,
    ) -> None:
        self._owner = owner
        self._cycle = cycle
        self._stage = stage
        self._worktree = worktree
        self._admissions: dict[str, RecoveryAdmission] = {}

    def __call__(self, check_id: str, failure_kind: str) -> bool:
        self._owner._guard()
        recovery = self._owner._recovery
        reason = "CHECK_INFRASTRUCTURE_UNAVAILABLE" if failure_kind == "timeout" else "CHECK_INFRASTRUCTURE_UNAVAILABLE"
        tree = safe_candidate_tree(self._worktree)
        admission = recovery.admit(
            recovery.budget_key("check-infra", f"{self._cycle:03d}", self._stage, check_id),
            reason=f"{reason}:{check_id}", budget=self._owner._attempts,
            phase="validation", cycle=self._cycle, tree_before=tree, tree_after=tree,
        )
        self._admissions[check_id] = admission
        return admission.admitted

    def settle(self, evidence: EvidenceBundle) -> None:
        """Trace retried checks and raise the waiting condition of the gate."""

        recovery = self._owner._recovery
        tree = evidence.staged_tree_sha
        infra_failures = [
            item for item in evidence.failures if item.startswith(_INFRA_FAILURE_PREFIXES)
        ]
        for check_result in evidence.checks:
            check_id = _field(check_result, "name")
            retries = _field(check_result, "infrastructure_retries", 0)
            if (
                isinstance(check_id, str) and isinstance(retries, int) and retries > 0
                and _field(check_result, "failure_kind") in {"passed", "nonzero_exit"}
                and check_id in self._admissions
            ):
                recovery.complete(self._admissions[check_id], recovered=True, tree_after=tree)
        for item in infra_failures:
            admission = self._admissions.get(item.split(":", 1)[1])
            if admission is not None:
                recovery.complete(admission, recovered=False, tree_after=tree)
        if infra_failures:
            waiting_reason = (
                "CHECK_SIDE_EFFECT_REPEATED"
                if all(item.startswith("CHECK_SIDE_EFFECT_REPEATED:") for item in infra_failures)
                else "CHECK_INFRASTRUCTURE_UNAVAILABLE"
            )
            raise PipelineFailure(waiting_reason, ", ".join(infra_failures))


class _PreflightStore:
    """The durable, tolerant cache of one run's preflight verdicts."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / PREFLIGHT_FILE

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
            or not isinstance(payload.get("checks"), dict)
        ):
            return {}
        return payload["checks"]

    def verdict(self, check_id: str, fingerprint: str) -> Mapping[str, Any] | None:
        """The cached verdict of one preflight, never a stale one."""

        entry = self._load().get(check_id)
        if (
            not isinstance(entry, dict)
            or entry.get("fingerprint") != fingerprint
            or entry.get("status") not in {"PASS", "FAIL"}
            or not isinstance(entry.get("exit_code"), int)
        ):
            return None
        return entry

    def record(self, check_id: str, fingerprint: str, verdict: Mapping[str, Any]) -> None:
        checks = self._load()
        checks[check_id] = {
            "status": verdict["status"], "exit_code": verdict["exit_code"],
            "detail": str(verdict.get("detail", ""))[:200], "fingerprint": fingerprint,
        }
        try:
            atomic_write_text(self.path, json.dumps({
                "schema_version": PREFLIGHT_SCHEMA_VERSION, "checks": checks,
            }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        except (OSError, ResultArtifactError) as exc:
            raise OrchestrationError(
                f"DURABLE_ARTIFACT_CORRUPTED: could not record preflight verdicts: {exc}"
            ) from None


__all__ = ['MAX_RECOVERY_ATTEMPT_RECORDS', 'RecoveryAdmission', 'RecoveryAttempt', 'RecoveryCoordinator', 'RecoveryTerminalState', 'failure_code', 'normalize_exit_reason', 'project_exit', 'terminal_state_for', 'TRANSIENT_WORKER_FAILURES', 'WorkerRecovery', 'CheckInfrastructureRecovery', 'GateInfraRetries']
