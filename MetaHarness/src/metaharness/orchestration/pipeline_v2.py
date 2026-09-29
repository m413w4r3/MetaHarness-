"""The generic pipeline-v2 state machine.

``PipelineV2Coordinator`` owns the order of the durable phases and the
checkpoint written before each of them.  It never owns an ``Orchestrator``:
every operation it sequences (a worker step, a deterministic gate, an audit,
a candidate commit, the publication) is injected explicitly through
:class:`PipelineV2Operations`.

A run executes one approved plan in one cycle: its steps, the deterministic
gate the audit authority answers whenever it leaves a diff, the accepted
candidate HEAD, its push and the publication.
"""

from __future__ import annotations

import dataclasses
import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias

from ..evidence import EvidenceBundle, required_checks_passed
from ..gitops import RepositoryReference, WorktreeInfo, resolve_tree
from ..models import (
    PARTIAL_REASONS,
    CycleKind,
    DROP_UNKNOWN_REQUIRED_CHECK,
    ExecutionSelection,
    GateStage,
    RunCycle,
    RunDisposition,
    RunEvent,
    RunMachineState,
    RunPhase,
    RunTransitionError,
    TaskPlanV2,
    transition,
)
from ..planning.planner_continue import ContinueDecision, PlannerContinueResult, stagnation_fingerprint
from ..planning.grammar import V2PlanParseError
from ..plan_repository_validation import PlanRepositoryPreconditionError, validate_plan_repository_topology
from ..result import RunResult
from ..resume import ResumeCheckpoint
from ..run_options import RunOptions

FailureDetail: TypeAlias = str | Mapping[str, Any]



# -- durable artifact layout -------------------------------------------------


def _cycle_number(cycle: RunCycle | int) -> int:
    number = cycle.number if isinstance(cycle, RunCycle) else cycle
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("cycle number must be a positive integer")
    return number


def _stage_name(stage: GateStage | str) -> str:
    try:
        return GateStage(stage).value.casefold().replace("_", "-")
    except ValueError as exc:
        raise ValueError("gate stage is unknown") from exc


def cycle_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return Path(run_dir) / "cycles" / f"{_cycle_number(cycle):03d}"


def implementation_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "implementation"


def implementation_steps_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return implementation_dir(run_dir, cycle) / "steps"


def step_dir(run_dir: Path, cycle: RunCycle | int, step_id: str) -> Path:
    return implementation_steps_dir(run_dir, cycle) / step_id


def gate_dir(run_dir: Path, cycle: RunCycle | int, stage: GateStage | str) -> Path:
    return cycle_dir(run_dir, cycle) / "checks" / _stage_name(stage)


def gate_acceptance_path(run_dir: Path, cycle: RunCycle | int, stage: GateStage | str) -> Path:
    return gate_dir(run_dir, cycle, stage) / "accepted.json"


def candidate_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "candidate"


def cycle_record_path(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "cycle.json"


# -- data exchanged with the operations ---------------------------------------


class RecoveryStepUnavailable(RuntimeError):
    """A recovery ladder rung cannot be executed for these exact facts.

    The deterministic gate never guesses and never widens an authority: it
    consumes the refused rung durably and asks the ladder for the next
    distinct strategy.
    """

    def __init__(self, strategy: Any, detail: str) -> None:
        super().__init__(f"{getattr(strategy, 'value', strategy)}: {detail}")
        self.strategy = strategy
        self.detail = detail


class BudgetExhausted(Exception):
    """The run's one global budget is spent; the partial path owns the exit.

    Raised at the boundary *before* an expensive operation: a new iteration, a
    planner or worker semantic attempt, an audit, or a costly gate retry.  It
    is never a failure: the coordinator projects it onto a ``PARTIAL``
    completion with its reason.
    """

    def __init__(self, reason: str) -> None:
        if reason not in PARTIAL_REASONS:
            raise ValueError(f"{reason!r} is not a PARTIAL reason")
        super().__init__(f"budget exhausted: {reason}")
        self.reason = reason


class PipelineFailure(Exception):
    """An attempt failure that left the current operation's recovery loop.

    ``reason`` is a stable failure code.  It is not necessarily terminal:
    the recovery coordinator projects it onto a hard failure (``FAILED``) or
    onto a waiting condition (``WAITING_*``) from the recovery policy.
    """

    def __init__(
        self, reason: str, detail: FailureDetail | None = None, *, step_id: str | None = None,
    ) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError("PipelineFailure reason must be a non-empty reason code string")
        if detail is not None and not isinstance(detail, (str, Mapping)):
            raise TypeError("PipelineFailure detail must be text or a structured mapping")
        if isinstance(detail, Mapping):
            if any(not isinstance(key, str) for key in detail):
                raise TypeError("PipelineFailure detail mapping keys must be strings")
            try:
                json.dumps(detail, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise TypeError("PipelineFailure detail mapping must contain JSON data") from exc
        super().__init__(f"{reason}: {detail}" if detail is not None else reason)
        self.reason = reason
        self.detail = detail
        self.step_id = step_id


@dataclass(frozen=True)
class PipelineV2Context:
    """Immutable facts of one prepared run."""

    run_dir: Path
    run_id: str
    spec: str
    context: str
    repo: Path
    base_sha: str
    base_tree_sha: str
    repository_reference: RepositoryReference
    info: WorktreeInfo
    plan: TaskPlanV2
    bundle: Mapping[str, Any]
    selection: ExecutionSelection
    options: RunOptions
    iteration: int = 1

    @property
    def branch_ref(self) -> str:
        return f"refs/heads/{self.info.branch}"


@dataclass(frozen=True)
class CyclePlan:
    """The approved authority one cycle executes."""

    cycle: RunCycle
    plan: TaskPlanV2
    bundle: Mapping[str, Any]
    # Where the approved ``steps/Sxx/contract.md`` files of this plan live.
    contracts_dir: Path
    step_profile_ids: Mapping[str, str]
    step_fallback_profile_ids: Mapping[str, tuple[str, ...]] | None = None

    @property
    def mutable_scope(self) -> tuple[str, ...]:
        return tuple(sorted({
            path for step in self.plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        }))


@dataclass(frozen=True)
class IterationOutcome:
    """The authoritative gate and writable audit result for one milestone."""

    evidence: EvidenceBundle
    audit_status: str = "NOT_RUN"
    audit_remaining: tuple[str, ...] = ()
    audit_fixed: tuple[str, ...] = ()
    audit_refactored: tuple[str, ...] = ()
    audit_risks: tuple[str, ...] = ()

    @property
    def gate_green(self) -> bool:
        return self.evidence.deterministic_passed and required_checks_passed(self.evidence)


@dataclass(frozen=True)
class PipelineV2Operations:
    """Side effects of the single implementation, audit and acceptance path."""

    checkpoint: Callable[..., None]
    current_head: Callable[[PipelineV2Context], str]
    begin_cycle: Callable[[PipelineV2Context, RunCycle, bool], None]
    initial_plan: Callable[[PipelineV2Context], CyclePlan]
    completed_steps: Callable[[PipelineV2Context, CyclePlan], list[dict[str, Any]]]
    execute_step: Callable[[PipelineV2Context, CyclePlan, int], None]
    pause_requested: Callable[[PipelineV2Context], bool]
    pause: Callable[[PipelineV2Context, CyclePlan, int], RunResult]
    run_gate: Callable[[PipelineV2Context, CyclePlan, GateStage], EvidenceBundle]
    load_gate_evidence: Callable[[PipelineV2Context, int, GateStage], EvidenceBundle | None]
    run_audit: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle, int], Any]
    accept_gate_state: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    hard_failures: Callable[[EvidenceBundle], list[str]]
    create_candidate: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    load_candidate: Callable[[PipelineV2Context, int], Mapping[str, Any]]
    push_candidate: Callable[[PipelineV2Context, int, Mapping[str, Any]], Mapping[str, Any]]
    publish: Callable[[PipelineV2Context, int, Mapping[str, Any]], RunResult]
    load_iteration_outcome: Callable[[PipelineV2Context, CyclePlan], IterationOutcome]
    planner_continue: Callable[..., PlannerContinueResult]
    prepare_next_iteration: Callable[[PipelineV2Context, PlannerContinueResult], PipelineV2Context]
    close_iteration: Callable[..., None]
    partial: Callable[..., RunResult]
    spec_decision: Callable[..., RunResult]
    failed_continued: Callable[[PipelineV2Context, CyclePlan], tuple[str, ...]]
    # The one global budget: the guard every expensive boundary consults, and
    # the partial exit that consumes it without a human decision.
    budget_exhausted: Callable[[PipelineV2Context], str | None]
    budget_partial: Callable[[PipelineV2Context, str], RunResult]


@dataclass(frozen=True)
class PipelineV2Coordinator:
    """Run milestone batches until COMPLETE, SPEC_DECISION or PARTIAL."""

    context: PipelineV2Context
    operations: PipelineV2Operations
    _cursor: list[RunPhase] = dataclasses.field(default_factory=list, repr=False, compare=False)
    # The iteration in flight; a budget exit reports on this one, never on the
    # context the coordinator was constructed with.
    _current: list[PipelineV2Context] = dataclasses.field(default_factory=list, repr=False, compare=False)

    @property
    def current(self) -> PipelineV2Context:
        return self._current[0] if self._current else self.context

    def _guard(self, ctx: PipelineV2Context) -> None:
        """Refuse to start an expensive operation once the budget is spent."""

        self._current[:] = [ctx]
        reason = self.operations.budget_exhausted(ctx)
        if reason is not None:
            raise BudgetExhausted(reason)

    def _phase(self, target: RunPhase) -> None:
        if self._cursor:
            try:
                transition(
                    RunMachineState(self._cursor[-1], RunDisposition.RUNNING),
                    RunEvent.advance(target),
                )
            except RunTransitionError as exc:
                raise PipelineFailure("INVALID_PHASE_TRANSITION", str(exc)) from exc
        self._cursor.append(target)

    def _boundary(
        self, ctx: PipelineV2Context, phase: RunPhase, plan: CyclePlan, *, step_index: int | None = None,
    ) -> None:
        self._phase(phase)
        self.operations.checkpoint(
            ctx, phase, iteration=plan.cycle.number,
            head=self.operations.current_head(ctx),
            step_index=step_index,
        )

    def _candidate_boundary(
        self, ctx: PipelineV2Context, phase: RunPhase, plan: CyclePlan, candidate: Mapping[str, Any],
    ) -> None:
        self._phase(phase)
        self.operations.checkpoint(
            ctx, phase, iteration=plan.cycle.number,
            head=candidate["commit_sha"],
        )

    def _pause_after_step(
        self, ctx: PipelineV2Context, plan: CyclePlan, index: int,
    ) -> RunResult | None:
        """Stop only after the accepted result of one complete step."""

        if not self.operations.pause_requested(ctx):
            return None
        next_index = index + 1
        if next_index < len(plan.plan.steps):
            self._boundary(ctx, RunPhase.IMPLEMENT_STEP, plan, step_index=next_index)
        else:
            self._boundary(ctx, RunPhase.DETERMINISTIC_GATE, plan)
        return self.operations.pause(ctx, plan, index)

    def run(self, start: ResumeCheckpoint, *, resumed: bool) -> RunResult:
        try:
            return self._run(start, resumed=resumed)
        except BudgetExhausted as exhausted:
            return self.operations.budget_partial(self.current, exhausted.reason)

    def _run(self, start: ResumeCheckpoint, *, resumed: bool) -> RunResult:
        ops = self.operations
        ctx = self.context
        if ctx.iteration != start.iteration:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "pipeline iteration does not match checkpoint")
        if start.phase in {
            RunPhase.CONTEXT, RunPhase.PLANNER, RunPhase.PLAN_APPROVAL,
            RunPhase.WORKTREE_SETUP,
        } and not (
            start.phase is RunPhase.PLANNER and start.plan_sha256 is not None
        ):
            raise ValueError(f"{start.phase.value} is not an execution checkpoint")
        if start.phase is RunPhase.PLANNER and start.plan_sha256 is not None:
            plan = ops.initial_plan(ctx)
            ops.begin_cycle(ctx, plan.cycle, False)
            outcome = ops.load_iteration_outcome(ctx, plan)
            return self._continue(
                ctx, plan, outcome,
                start_commit=iteration_start_commit(ctx, start.last_green_commit or ctx.base_sha),
            )

        while True:
            self._guard(ctx)
            iteration = ctx.iteration
            cycle = RunCycle(iteration, CycleKind.INITIAL)
            ops.begin_cycle(ctx, cycle, not resumed if iteration == start.iteration else True)
            plan = ops.initial_plan(ctx)
            stage = GateStage.POST_IMPLEMENTATION
            if start.phase is RunPhase.PUBLISH:
                return ops.publish(ctx, iteration, ops.load_candidate(ctx, iteration))
            if start.phase is RunPhase.IMPLEMENT_STEP:
                index = start.step_index
                if index is None or index > len(plan.plan.steps):
                    raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "implementation step_index is outside the plan")
                for index in range(index, len(plan.plan.steps)):
                    self._boundary(ctx, RunPhase.IMPLEMENT_STEP, plan, step_index=index)
                    ops.execute_step(ctx, plan, index)
                    if paused := self._pause_after_step(ctx, plan, index):
                        return paused
                self._boundary(ctx, RunPhase.DETERMINISTIC_GATE, plan)
                outcome = self._gate(ctx, plan, stage, None)
            elif start.phase in {RunPhase.DETERMINISTIC_GATE, RunPhase.AUDIT}:
                outcome = self._gate(ctx, plan, stage, start)
            elif start.phase in {RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH}:
                outcome = None
            else:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "checkpoint phase is not executable")

            if start.phase in {RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH}:
                candidate = ops.load_candidate(ctx, iteration)
                if start.phase is RunPhase.CANDIDATE_READY:
                    self._candidate_boundary(ctx, RunPhase.CANDIDATE_PUSH, plan, candidate)
            else:
                if outcome is None:
                    raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "iteration outcome is missing")
                return self._continue(
                    ctx, plan, outcome,
                    start_commit=iteration_start_commit(ctx, start.last_green_commit or ctx.base_sha),
                )
            if start.phase is not RunPhase.CANDIDATE_PUSH:
                if start.phase not in {RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH}:
                    self._boundary(ctx, RunPhase.CANDIDATE_READY, plan)
                    candidate = ops.create_candidate(ctx, plan, stage, outcome.evidence if outcome else None)
                    self._candidate_boundary(ctx, RunPhase.CANDIDATE_PUSH, plan, candidate)
            candidate = ops.push_candidate(ctx, iteration, candidate)
            self._candidate_boundary(ctx, RunPhase.PUBLISH, plan, candidate)
            return ops.publish(ctx, iteration, candidate)

    def _continue(
        self, ctx: PipelineV2Context, plan: CyclePlan, outcome: IterationOutcome,
        *, start_commit: str,
    ) -> RunResult:
        ops = self.operations
        current_head = ops.current_head(ctx)
        self._boundary(ctx, RunPhase.PLANNER, plan)
        def validate_decision(decision: PlannerContinueResult) -> PlannerContinueResult:
            if decision.decision is ContinueDecision.COMPLETE:
                invalid: list[str] = []
                if not outcome.gate_green:
                    invalid.append("deterministic gate is red or required checks are unsatisfied")
                if outcome.audit_remaining:
                    invalid.append("audit REMAINING is not empty")
                if decision.remaining:
                    invalid.append("continuation REMAINING is not empty")
                if str(plan.plan.project_remainder).strip().casefold() not in {
                    "none", "n/a", "na", "-", "—", "nil",
                }:
                    invalid.append("current project_remainder is not NONE")
                remaining = (*outcome.audit_remaining, *decision.remaining)
                unresolved_failed = [
                    item for item in failed
                    if item.partition(":")[0].casefold() in " ".join(remaining).casefold()
                ]
                if unresolved_failed:
                    invalid.append("FAILED_CONTINUED steps remain in continuation REMAINING")
                if invalid:
                    raise V2PlanParseError("COMPLETE refused: " + "; ".join(invalid))
                return decision
            if decision.decision is not ContinueDecision.NEXT or decision.next_plan is None:
                if decision.decision is ContinueDecision.SPEC_DECISION:
                    return decision
                raise V2PlanParseError("continuation decision is incomplete")
            current = plan.plan.milestone_id
            if current.startswith("M") and current[1:].isdigit():
                n = int(current[1:])
                allowed = {current, f"M{n + 1:02d}"}
                if decision.next_milestone not in allowed:
                    raise V2PlanParseError(
                        f"NEXT must stay on {current} or progress to M{n + 1:02d}"
                    )
            else:
                expected_milestone = f"M{ctx.iteration + 1:02d}"
                if decision.next_milestone != expected_milestone:
                    raise V2PlanParseError(f"NEXT must progress monotonically to {expected_milestone}")
            if len(decision.next_plan.steps) > ctx.options.max_steps_per_plan:
                raise V2PlanParseError("NEXT exceeds planning.max_steps_per_plan")
            unknown_checks = [
                item.detail for item in decision.next_plan.normalizations
                if item.code == DROP_UNKNOWN_REQUIRED_CHECK
            ]
            if unknown_checks:
                raise V2PlanParseError(f"NEXT contains untrusted checks: {unknown_checks}")
            try:
                normalized = validate_plan_repository_topology(
                    ctx.info.worktree,
                    resolve_tree(ctx.info.worktree, ops.current_head(ctx)),
                    decision.next_plan,
                )
            except PlanRepositoryPreconditionError as exc:
                raise V2PlanParseError(f"NEXT cannot run against the current HEAD: {exc}") from exc
            return dataclasses.replace(decision, next_plan=normalized)

        failed = ops.failed_continued(ctx, plan)
        decision = ops.planner_continue(ctx, plan, outcome, validate_decision)
        failed = ops.failed_continued(ctx, plan)

        if decision.decision is ContinueDecision.SPEC_DECISION:
            ops.close_iteration(ctx, plan, outcome, decision, start_commit, None, "SPEC_DECISION")
            return ops.spec_decision(ctx, decision, outcome)

        if decision.decision is ContinueDecision.COMPLETE:
            fingerprint = stagnation_fingerprint((), resolve_candidate_tree(ctx), outcome.evidence.failures)
            ops.close_iteration(ctx, plan, outcome, decision, start_commit, fingerprint, "COMPLETE")
            self._boundary(ctx, RunPhase.CANDIDATE_READY, plan)
            candidate = ops.create_candidate(ctx, plan, GateStage.POST_IMPLEMENTATION, outcome.evidence)
            self._candidate_boundary(ctx, RunPhase.CANDIDATE_PUSH, plan, candidate)
            candidate = ops.push_candidate(ctx, ctx.iteration, candidate)
            self._candidate_boundary(ctx, RunPhase.PUBLISH, plan, candidate)
            result = ops.publish(ctx, ctx.iteration, candidate)
            return result

        if decision.decision is not ContinueDecision.NEXT or decision.next_plan is None:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", "continuation decision is incomplete")
        remaining = (*outcome.audit_remaining, *decision.remaining)
        fingerprint = stagnation_fingerprint(
            remaining, resolve_candidate_tree(ctx), outcome.evidence.failures,
        )
        previous = load_stagnation_fingerprint(ctx.run_dir, ctx.iteration - 1)
        stagnant = previous is not None and previous == fingerprint
        partial_reason = (
            "max_iterations" if ctx.iteration >= ctx.options.budget.max_iterations
            else "stagnation" if stagnant else None
        )
        ops.close_iteration(
            ctx, plan, outcome, decision, start_commit, fingerprint,
            "PARTIAL" if partial_reason else "NEXT",
        )
        if partial_reason:
            failures = tuple(dict.fromkeys((*outcome.evidence.failures, *failed)))
            return ops.partial(ctx, partial_reason, remaining, failures)
        next_ctx = ops.prepare_next_iteration(ctx, decision)
        self._guard(next_ctx)
        next_plan = ops.initial_plan(next_ctx)
        self._boundary(next_ctx, RunPhase.IMPLEMENT_STEP, next_plan, step_index=0)
        ctx = next_ctx
        plan = next_plan
        self.operations.begin_cycle(ctx, plan.cycle, True)
        for index in range(len(plan.plan.steps)):
            self._boundary(ctx, RunPhase.IMPLEMENT_STEP, plan, step_index=index)
            ops.execute_step(ctx, plan, index)
            if paused := self._pause_after_step(ctx, plan, index):
                return paused
        self._boundary(ctx, RunPhase.DETERMINISTIC_GATE, plan)
        outcome = self._gate(ctx, plan, GateStage.POST_IMPLEMENTATION, None)
        return self._continue(ctx, plan, outcome, start_commit=current_head)

    def _gate(
        self, ctx: PipelineV2Context, plan: CyclePlan, stage: GateStage,
        start: ResumeCheckpoint | None,
    ) -> IterationOutcome:
        ops = self.operations
        reports = sorted((cycle_dir(ctx.run_dir, plan.cycle) / "audit").glob("*/report.json"))
        if start is not None and start.phase is RunPhase.AUDIT and not reports:
            evidence = ops.load_gate_evidence(ctx, plan.cycle.number, stage)
            if evidence is None:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "diagnostic gate evidence is missing")
        else:
            evidence = ops.run_gate(ctx, plan, stage)
        for report_path in reports:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "audit report is unreadable") from exc
            if report.get("status") == "SPEC_DECISION":
                raise PipelineFailure("SPEC_DECISION_REQUIRED", {
                    "remaining": report.get("remaining", []),
                    "failure_ids": report.get("failure_ids", []),
                })
        summary = audit_summary(reports)
        if reports and evidence.deterministic_passed and required_checks_passed(evidence):
            ops.accept_gate_state(ctx, plan, stage, evidence)
            return IterationOutcome(evidence, **summary)
        for attempt in range(len(reports) + 1, ctx.options.budget.audit_repairs + 1):
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if not evidence.changed_files and not ops.failed_continued(ctx, plan):
                if evidence.deterministic_passed and required_checks_passed(evidence):
                    ops.accept_gate_state(ctx, plan, stage, evidence)
                    return IterationOutcome(evidence, **summary)
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
            self._boundary(ctx, RunPhase.AUDIT, plan)
            audit = ops.run_audit(ctx, plan, stage, evidence, attempt)
            summary = {
                "audit_status": audit.status,
                "audit_remaining": tuple(audit.remaining),
                "audit_fixed": tuple(audit.fixed),
                "audit_refactored": tuple(audit.refactored),
                "audit_risks": tuple(audit.risks),
            }
            if audit.status == "SPEC_DECISION":
                raise PipelineFailure("SPEC_DECISION_REQUIRED", {
                    "remaining": list(audit.remaining), "failure_ids": list(evidence.failures),
                })
            self._boundary(ctx, RunPhase.DETERMINISTIC_GATE, plan)
            evidence = ops.run_gate(ctx, plan, stage)
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if evidence.deterministic_passed:
                if evidence.failures or not required_checks_passed(evidence):
                    raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "PASS gate has incomplete evidence")
                ops.accept_gate_state(ctx, plan, stage, evidence)
                return IterationOutcome(evidence, **summary)
        return IterationOutcome(evidence, **summary)


def resolve_candidate_tree(ctx: PipelineV2Context) -> str:
    """Return the exact tracked tree at the current worktree HEAD."""

    from ..gitops import candidate_tree_sha
    return candidate_tree_sha(ctx.info.worktree)


def load_stagnation_fingerprint(run_dir: Path, iteration: int) -> str | None:
    if iteration < 1:
        return None
    path = Path(run_dir) / "iterations" / f"{iteration:02d}" / "iteration.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = record.get("stagnation_fingerprint") if isinstance(record, dict) else None
    return value if isinstance(value, str) else None


def iteration_start_commit(ctx: PipelineV2Context, fallback: str) -> str:
    if ctx.iteration == 1:
        return ctx.base_sha
    path = ctx.run_dir / "iterations" / f"{ctx.iteration - 1:02d}" / "iteration.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    value = record.get("end_commit") if isinstance(record, dict) else None
    return value if isinstance(value, str) else fallback


def audit_summary(report_paths: list[Path]) -> dict[str, Any]:
    """Read the latest compact audit outcome when resuming at PLANNER."""

    if not report_paths:
        return {"audit_status": "NOT_RUN"}
    try:
        report = json.loads(report_paths[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "audit report is unreadable") from exc
    return {
        "audit_status": report.get("status", "NOT_RUN"),
        "audit_remaining": tuple(report.get("remaining", ())),
        "audit_fixed": tuple(report.get("fixed", ())),
        "audit_refactored": tuple(report.get("refactored", ())),
        "audit_risks": tuple(report.get("risks", ())),
    }


__all__ = [
    "BudgetExhausted", "CyclePlan", "PipelineFailure", "PipelineV2Context", "PipelineV2Coordinator",
    "RecoveryStepUnavailable",
    "IterationOutcome", "PipelineV2Operations", "candidate_dir",
    "cycle_dir", "cycle_record_path", "gate_acceptance_path", "gate_dir",
    "implementation_dir", "implementation_steps_dir", "step_dir",
]
