"""The generic pipeline-v2 state machine.

``PipelineV2Coordinator`` owns the order of the durable phases and the
checkpoint written before each of them.  It never owns an ``Orchestrator``:
every operation it sequences (a worker step, a deterministic gate, a
candidate commit, a review, the publication) is injected explicitly through
:class:`PipelineV2Operations`.

A run is a sequence of cycles ``001, 002, ...``.  Each cycle executes one
approved plan (the operator-approved plan for the initial cycle, a
review-driven or red-gate correction plan afterwards), optionally a semantic
revision, one or two deterministic gate episodes with their recovery ladder,
the accepted candidate HEAD, its push and one final review.  The number of
cycles is bounded only by the frozen run options, never by this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeAlias

from ..evidence import EvidenceBundle, required_checks_passed
from ..gitops import RepositoryReference, WorktreeInfo
from ..models import (
    CycleKind,
    ExecutionSelection,
    GateStage,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    RunDisposition,
    RunEvent,
    RunMachineState,
    RunPhase,
    RunTransitionError,
    TaskPlanV2,
    correction_cycles_used,
    transition,
)
from ..recovery_policy import RecoveryStrategy
from ..result import RunResult
from ..resume import ResumeCheckpoint
from ..review import ReviewResult
from ..run_options import RunOptions

FailureDetail: TypeAlias = str | Mapping[str, Any]

if TYPE_CHECKING:  # pragma: no cover - the ladder protocol lives in recovery.py
    from .recovery import GateRecoveryStep, RecoveryOperations


def check_repair_fingerprint(
    candidate_tree_sha: str, failed_check_ids: Sequence[str], stage: GateStage | str,
    strategy: str = "",
) -> tuple[str, tuple[str, ...], str, str]:
    """Stable identity of one exhausted deterministic-gate failure.

    The ladder position is part of the identity: two exhausted episodes that
    stopped after different strategies are different facts, so an operator
    retry that opened a new rung is never mistaken for a fixed point.
    """

    stage_name = stage.value if isinstance(stage, GateStage) else str(stage)
    return (
        candidate_tree_sha, tuple(sorted(set(failed_check_ids))), stage_name, str(strategy),
    )


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


def correction_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "correction"


def semantic_revision_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "semantic-revision"


def gate_dir(run_dir: Path, cycle: RunCycle | int, stage: GateStage | str) -> Path:
    return cycle_dir(run_dir, cycle) / "checks" / _stage_name(stage)


def gate_acceptance_path(run_dir: Path, cycle: RunCycle | int, stage: GateStage | str) -> Path:
    return gate_dir(run_dir, cycle, stage) / "accepted.json"


def check_repair_dir(run_dir: Path, cycle: RunCycle | int, stage: GateStage | str) -> Path:
    return cycle_dir(run_dir, cycle) / "check-repair" / _stage_name(stage)


def check_repair_root(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "check-repair"


def check_repair_attempts_dir(
    run_dir: Path, cycle: RunCycle | int, stage: GateStage | str,
) -> Path:
    return check_repair_dir(run_dir, cycle, stage) / "attempts"


def check_repair_attempt_dir(
    run_dir: Path, cycle: RunCycle | int, stage: GateStage | str, attempt: int,
) -> Path:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("check-repair attempt must be a positive integer")
    return check_repair_attempts_dir(run_dir, cycle, stage) / f"{attempt:03d}"


def candidate_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "candidate"


def review_dir(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "review"


def cycle_record_path(run_dir: Path, cycle: RunCycle | int) -> Path:
    return cycle_dir(run_dir, cycle) / "cycle.json"


def pre_semantic_gate_stage(kind: CycleKind) -> GateStage:
    """Return the gate immediately following implementation work."""

    return {
        CycleKind.INITIAL: GateStage.POST_IMPLEMENTATION,
        CycleKind.REVIEW_REPLAN: GateStage.POST_REVIEW_REPLAN,
        CycleKind.CHECK_REPLAN: GateStage.POST_CHECK_REPLAN,
    }[CycleKind(kind)]


def final_gate_stage(kind: CycleKind) -> GateStage:
    """Return the gate which authorizes the candidate HEAD."""

    kind = CycleKind(kind)
    if kind is CycleKind.REVIEW_IMPLEMENTATION:
        return GateStage.POST_REVIEW_IMPLEMENTATION
    return GateStage.POST_SEMANTIC_REVISION


def correction_kind(route: ReviewRoute) -> CycleKind:
    if route is ReviewRoute.IMPLEMENTATION:
        return CycleKind.REVIEW_IMPLEMENTATION
    if route is ReviewRoute.REPLAN:
        return CycleKind.REVIEW_REPLAN
    raise ValueError("only IMPLEMENTATION and REPLAN reviews open a correction cycle")


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


# The phases that restate a cycle boundary, and the ordinary step cycles.
_CYCLE_ENTRY_PHASES = frozenset({RunPhase.REVIEW_REPLAN, RunPhase.CHECK_REPLAN})
_IMPLEMENT_PHASES = frozenset({CycleKind.INITIAL, CycleKind.CHECK_REPLAN})


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
    correction_bundle_sha256: str | None = None
    step_fallback_profile_ids: Mapping[str, tuple[str, ...]] | None = None

    @property
    def mutable_scope(self) -> tuple[str, ...]:
        return tuple(sorted({
            path for step in self.plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        }))


@dataclass(frozen=True)
class PipelineV2Operations:
    """Every side effect the coordinator sequences, injected explicitly."""

    # Durable boundaries and Git identity.
    checkpoint: Callable[..., None]
    current_head: Callable[[PipelineV2Context], str]
    candidate_tree: Callable[[PipelineV2Context], str]
    # Cycle bookkeeping.
    begin_cycle: Callable[[PipelineV2Context, RunCycle, bool], None]
    load_cycle: Callable[[PipelineV2Context, int], RunCycle]
    initial_plan: Callable[[PipelineV2Context], CyclePlan]
    # The flag is true when the correction itself is about to run, false
    # when a later checkpoint of the same cycle is resumed.
    review_implementation_correction: Callable[
        [PipelineV2Context, RunCycle, bool], tuple[CyclePlan, ReviewResult]]
    plan_correction: Callable[[PipelineV2Context, RunCycle], CyclePlan]
    load_correction: Callable[[PipelineV2Context, RunCycle, str | None], CyclePlan]
    # Implementation.
    completed_steps: Callable[[PipelineV2Context, CyclePlan], list[dict[str, Any]]]
    execute_step: Callable[[PipelineV2Context, CyclePlan, int], None]
    unresolved_mismatches: Callable[[PipelineV2Context, CyclePlan], bool]
    semantic_revision: Callable[[PipelineV2Context, CyclePlan], None]
    semantic_review_correction: Callable[[PipelineV2Context, CyclePlan, ReviewResult], None]
    # Deterministic gate episode.
    run_gate: Callable[[PipelineV2Context, CyclePlan, GateStage], EvidenceBundle]
    load_gate_evidence: Callable[[PipelineV2Context, int, GateStage], EvidenceBundle | None]
    load_accepted_gate_evidence: Callable[
        [PipelineV2Context, int, GateStage], EvidenceBundle | None]
    accept_gate_state: Callable[
        [PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    check_repair_attempts: Callable[[PipelineV2Context, int, GateStage], tuple[Any, ...]]
    check_repair_attempt: Callable[
        [PipelineV2Context, CyclePlan, GateStage, int, EvidenceBundle], None]
    hard_failures: Callable[[EvidenceBundle], list[str]]
    soft_failures: Callable[[EvidenceBundle], list[str]]
    # Candidate, review and publication.
    create_candidate: Callable[
        [PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    load_candidate: Callable[[PipelineV2Context, int], Mapping[str, Any]]
    push_candidate: Callable[[PipelineV2Context, int, Mapping[str, Any]], Mapping[str, Any]]
    review_candidate: Callable[
        [PipelineV2Context, CyclePlan, Mapping[str, Any], EvidenceBundle], ReviewResult]
    record_review: Callable[[PipelineV2Context, int, ReviewResult, EvidenceBundle], None]
    request_human: Callable[[PipelineV2Context, int, ReviewResult, str], RunResult]
    review_repair_exhausted: Callable[
        [PipelineV2Context, int, ReviewResult, Mapping[str, Any]], RunResult]
    publish: Callable[[PipelineV2Context, int, Mapping[str, Any]], RunResult]
    # The single red-gate recovery ladder object of the run: every red
    # deterministic gate asks it which distinct strategy to try next and
    # reports each step, and there is no other engine.
    recovery_operations: "RecoveryOperations"
    # Accept the durable candidate of a ``STEP_ACCEPTANCE``: no model call.
    accept_step: Callable[[PipelineV2Context, CyclePlan, int], None] | None = None

    def __post_init__(self) -> None:
        if self.recovery_operations is None:
            raise TypeError("PipelineV2Operations requires the red-gate recovery ladder")


# ``RunPhase`` is the state machine's own phase vocabulary, never a posture:
# the phase this module may hand over to is decided by
# :func:`metaharness.models.transition`, in one place.


@dataclass(frozen=True)
class PipelineV2Coordinator:
    """Sequence the generic cycles of one prepared pipeline-v2 run."""

    context: PipelineV2Context
    operations: PipelineV2Operations
    # The phase this coordinator made durable last, in this pass: the single
    # transition table decides which phase may follow it.
    _cursor: list[RunPhase] = dataclasses.field(
        default_factory=list, repr=False, compare=False,
    )

    def _phase(self, target: RunPhase) -> RunPhase:
        """Validate one durable phase against the single transition table."""

        if self._cursor:
            try:
                transition(
                    RunMachineState(self._cursor[-1], RunDisposition.RUNNING),
                    RunEvent.advance(target),
                )
            except RunTransitionError as exc:
                raise PipelineFailure("INVALID_PHASE_TRANSITION", str(exc)) from exc
        self._cursor.append(target)
        return target

    def run(self, start: ResumeCheckpoint, *, resumed: bool) -> RunResult:
        """Execute from *start*: the first step of a new run, or a checkpoint.

        Every operation before *start* is durable and is never replayed; the
        operations read back what they need from their durable artifacts.
        """

        if start.phase in {
            RunPhase.CONTEXT, RunPhase.PLANNER, RunPhase.PLAN_APPROVAL,
            RunPhase.WORKTREE_SETUP,
        }:
            raise ValueError(f"{start.phase.value} is not an execution checkpoint")
        cycle = self._cycle_at(start)
        entry: ResumeCheckpoint | None = start
        fresh = not resumed
        while True:
            outcome = self._run_cycle(cycle, entry, fresh=fresh)
            if isinstance(outcome, RunResult):
                return outcome
            if isinstance(outcome, CyclePlan):
                # A re-decomposed cycle: its plan is already durable.
                cycle = outcome.cycle
            else:
                cycle = RunCycle(cycle.number + 1, correction_kind(outcome.route))
            entry, fresh = None, True

    # -- one cycle -----------------------------------------------------------

    def _cycle_at(self, start: ResumeCheckpoint) -> RunCycle:
        if start.review_cycle == 1:
            return RunCycle(1, CycleKind.INITIAL)
        return self.operations.load_cycle(self.context, start.review_cycle)

    def _run_cycle(
        self, cycle: RunCycle, start: ResumeCheckpoint | None, *, fresh: bool,
    ) -> RunResult | ReviewResult | CyclePlan:
        ops, ctx = self.operations, self.context
        ops.begin_cycle(ctx, cycle, fresh)
        accept_step_id: str | None = None
        if start is not None and start.phase is RunPhase.STEP_ACCEPTANCE:
            # Resumed inside the implementation phase of this cycle: the step
            # candidate is accepted, then the following steps run normally.
            accept_step_id = start.step_id
            start = dataclasses.replace(
                start,
                phase=(
                    RunPhase.IMPLEMENT_STEP if cycle.kind in _IMPLEMENT_PHASES
                    else RunPhase.REVIEW_IMPLEMENTATION
                ),
            )
        if start is not None and start.phase is RunPhase.PUBLISH:
            return ops.publish(ctx, cycle.number, ops.load_candidate(ctx, cycle.number))

        if cycle.kind is CycleKind.INITIAL:
            cycle_plan = ops.initial_plan(ctx)
            previous_review = None
        elif cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            cycle_plan, previous_review = ops.review_implementation_correction(
                ctx, cycle, start is None or start.phase is RunPhase.SEMANTIC_REVISION,
            )
        elif start is None or start.phase in _CYCLE_ENTRY_PHASES:
            if start is None:
                # A cycle boundary names its own kind: both kinds recover an
                # already durable plan here, never a second planner call.
                entry = (
                    RunPhase.CHECK_REPLAN if cycle.kind is CycleKind.CHECK_REPLAN
                    else RunPhase.REVIEW_REPLAN
                )
                self._phase(entry)
                ops.checkpoint(
                    ctx, entry, cycle=cycle.number,
                    head=ops.current_head(ctx), tree=ops.candidate_tree(ctx),
                )
            cycle_plan = ops.plan_correction(ctx, cycle)
            previous_review = None
        else:
            cycle_plan = ops.load_correction(ctx, cycle, start.correction_bundle_sha256)
            previous_review = None

        phase = (
            RunPhase.IMPLEMENT_STEP
            if cycle.kind in _IMPLEMENT_PHASES
            else (
                RunPhase.SEMANTIC_REVISION
                if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION
                else RunPhase.REVIEW_IMPLEMENTATION
            )
        )
        pre_stage = (
            pre_semantic_gate_stage(cycle.kind)
            if cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION else None
        )
        final_stage = final_gate_stage(cycle.kind)
        semantic_enabled = (
            ctx.options.semantic_revision_enabled
            and cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION
        )

        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            if start is None or start.phase is phase:
                self._boundary(phase, cycle_plan)
                ops.semantic_review_correction(ctx, cycle_plan, previous_review)
                self._boundary(RunPhase.DETERMINISTIC_GATE, cycle_plan, stage=final_stage)
        elif start is None or start.phase is phase or start.phase in _CYCLE_ENTRY_PHASES:
            self._implement(
                cycle_plan, next_stage=pre_stage or final_stage, accept_step_id=accept_step_id,
            )
        if ops.unresolved_mismatches(ctx, cycle_plan) and not semantic_enabled:
            raise PipelineFailure(
                "UNRESOLVED_CONTRACT_MISMATCH",
                "HUMAN_REQUIRED: semantic revision is disabled while contract mismatches are deferred",
            )

        # Initial and replan cycles have a gate before semantic revision.
        if pre_stage is not None and (
            start is None or start.phase is phase or start.phase in _CYCLE_ENTRY_PHASES
            or self._is_gate_checkpoint(start, pre_stage)
        ):
            outcome = self._gate_episode(
                cycle_plan, pre_stage,
                start if self._is_gate_checkpoint(start, pre_stage) else None,
            )
            if isinstance(outcome, CyclePlan):
                return outcome

        if semantic_enabled and cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION and (
            start is None or start.phase in {
                phase, RunPhase.SEMANTIC_REVISION, *_CYCLE_ENTRY_PHASES,
            } or self._is_gate_checkpoint(start, pre_stage)
        ):
            self._boundary(RunPhase.SEMANTIC_REVISION, cycle_plan)
            ops.semantic_revision(ctx, cycle_plan)
            self._boundary(RunPhase.DETERMINISTIC_GATE, cycle_plan, stage=final_stage)

        final_start = start if self._is_gate_checkpoint(start, final_stage) else None
        should_run_final_gate = (
            final_start is not None
            or start is None
            or start.phase is phase
            or start.phase in _CYCLE_ENTRY_PHASES
            or start.phase is RunPhase.SEMANTIC_REVISION
            or self._is_gate_checkpoint(start, pre_stage)
        )
        if semantic_enabled:
            # A pre-semantic gate was consumed above; the final gate is still
            # needed after the revision, and its own checkpoint resumes that
            # episode without replaying the revision.
            outcome = self._gate_episode(cycle_plan, final_stage, final_start) if should_run_final_gate else None
        elif pre_stage is not None:
            outcome = ops.load_gate_evidence(ctx, cycle.number, pre_stage)
        else:
            outcome = self._gate_episode(cycle_plan, final_stage, final_start) if should_run_final_gate else None
        if isinstance(outcome, CyclePlan):
            return outcome
        evidence = outcome
        if evidence is None:
            evidence = ops.load_gate_evidence(ctx, cycle.number, final_stage if semantic_enabled else (pre_stage or final_stage))
        if evidence is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "the final gate evidence is missing")

        stage = final_stage if semantic_enabled else (pre_stage or final_stage)
        if start is None or start.phase not in {
            RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH,
            RunPhase.FINAL_REVIEW, RunPhase.PUBLISH,
        }:
            self._boundary(RunPhase.CANDIDATE_READY, cycle_plan, tree=evidence.staged_tree_sha)
            candidate = ops.create_candidate(ctx, cycle_plan, stage, evidence)
        else:
            candidate = ops.load_candidate(ctx, cycle.number)
        if start is None or start.phase in {
            phase, RunPhase.SEMANTIC_REVISION, RunPhase.DETERMINISTIC_GATE,
            RunPhase.CHECK_REPAIR, RunPhase.CANDIDATE_READY, *_CYCLE_ENTRY_PHASES,
        }:
            self._candidate_boundary(RunPhase.CANDIDATE_PUSH, cycle_plan, candidate)
            candidate = ops.push_candidate(ctx, cycle.number, candidate)
        elif start.phase is RunPhase.CANDIDATE_PUSH:
            candidate = ops.push_candidate(ctx, cycle.number, candidate)
        if start is None or start.phase in {
            phase, RunPhase.SEMANTIC_REVISION, RunPhase.DETERMINISTIC_GATE,
            RunPhase.CHECK_REPAIR, RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH,
            RunPhase.FINAL_REVIEW, *_CYCLE_ENTRY_PHASES,
        }:
            self._candidate_boundary(RunPhase.FINAL_REVIEW, cycle_plan, candidate)
        review = ops.review_candidate(ctx, cycle_plan, candidate, evidence)
        ops.record_review(ctx, cycle.number, review, evidence)
        return self._route(cycle_plan, candidate, evidence, review)

    @staticmethod
    def _is_gate_checkpoint(
        start: ResumeCheckpoint | None, stage: GateStage | None,
    ) -> bool:
        return bool(
            start is not None and stage is not None
            and start.phase in {RunPhase.DETERMINISTIC_GATE, RunPhase.CHECK_REPAIR}
            and start.stage is stage
        )

    def _route(
        self, cycle_plan: CyclePlan, candidate: Mapping[str, Any],
        evidence: EvidenceBundle, review: ReviewResult,
    ) -> RunResult | ReviewResult:
        ops, ctx, cycle = self.operations, self.context, cycle_plan.cycle
        if review.verdict is ReviewVerdict.PASS:
            if review.route is not ReviewRoute.NONE:
                raise PipelineFailure("REVIEW_ROUTE_NOT_NONE")
            if not evidence.deterministic_passed:
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED")
            return ops.publish(ctx, cycle.number, candidate)
        if review.verdict is ReviewVerdict.FAIL:
            raise PipelineFailure(
                "REVIEW_EVIDENCE_UNRESOLVED",
                {"summary": review.summary, "findings": review.findings},
            )
        if review.route is ReviewRoute.HUMAN:
            return ops.request_human(ctx, cycle.number, review, "HUMAN_REQUIRED")
        if review.route not in {ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN}:
            return ops.request_human(ctx, cycle.number, review, "HUMAN_REQUIRED")
        # One budget bounds every cycle after INITIAL, review or red-gate.
        budget = ctx.options.max_correction_cycles
        corrections_used = correction_cycles_used(cycle.number)
        if corrections_used >= budget:
            return ops.review_repair_exhausted(
                ctx, cycle.number, review,
                {
                    "last_review_cycle": cycle.number,
                    "corrections_used": corrections_used,
                    "max_correction_cycles": budget,
                    "last_route": review.route.value,
                },
            )
        return review

    # -- phases ----------------------------------------------------------------

    def _implement(
        self, cycle_plan: CyclePlan, *, next_stage: GateStage, accept_step_id: str | None = None,
    ) -> None:
        ops, ctx = self.operations, self.context
        phase = (
            RunPhase.IMPLEMENT_STEP if cycle_plan.cycle.kind in _IMPLEMENT_PHASES
            else RunPhase.REVIEW_IMPLEMENTATION
        )
        steps = cycle_plan.plan.steps
        if accept_step_id is not None:
            if ops.accept_step is None or accept_step_id not in {step.id for step in steps}:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    "the step acceptance checkpoint does not name a step of this cycle",
                )
            # Idempotent: a candidate committed before a crash is only
            # re-recorded, never committed or executed again.
            ops.accept_step(
                ctx, cycle_plan, [step.id for step in steps].index(accept_step_id),
            )
        done = {record["id"] for record in ops.completed_steps(ctx, cycle_plan)}
        for index, step in enumerate(steps):
            if step.id in done:
                continue
            if step.id == accept_step_id:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", "the accepted step is not durably completed",
                    step_id=step.id,
                )
            self._boundary(phase, cycle_plan, step_id=step.id)
            ops.execute_step(ctx, cycle_plan, index)
        self._boundary(RunPhase.DETERMINISTIC_GATE, cycle_plan, stage=next_stage)

    def _gate_episode(
        self, cycle_plan: CyclePlan, stage: GateStage, start: ResumeCheckpoint | None,
    ) -> EvidenceBundle | CyclePlan:
        """One deterministic gate and its recovery ladder; a red gate whose
        rungs are durably spent returns the new cycle plan it re-decomposed.
        """

        ops, ctx = self.operations, self.context
        number = cycle_plan.cycle.number
        budget = ctx.options.max_check_repair_attempts
        if start is not None and start.phase is RunPhase.CHECK_REPAIR:
            evidence = ops.load_gate_evidence(ctx, number, stage)
            if evidence is None or evidence.staged_tree_sha != start.expected_tree_sha:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    "the red gate evidence of the check-repair attempt is missing",
                )
            attempt = int(start.check_repair_attempt or 1)
            # Only durable, protocol-verified completed repair records spend
            # this budget. The checkpoint names a pending attempt or one
            # already recorded whose worker must never be called again.
            durable = len(ops.check_repair_attempts(ctx, number, stage))
            if attempt not in {durable, durable + 1}:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    "the check-repair attempt does not follow the durable attempts",
                )
            repair_boundary_written = True
            attempt_recorded = attempt == durable
        elif start is not None and start.phase is RunPhase.DETERMINISTIC_GATE:
            evidence = ops.load_accepted_gate_evidence(ctx, number, stage)
            if evidence is None:
                evidence = ops.run_gate(ctx, cycle_plan, stage)
            durable = len(ops.check_repair_attempts(ctx, number, stage))
            attempt = durable + 1
            repair_boundary_written = False
            attempt_recorded = False
        else:
            evidence = ops.run_gate(ctx, cycle_plan, stage)
            durable = len(ops.check_repair_attempts(ctx, number, stage))
            attempt = durable + 1
            repair_boundary_written = False
            attempt_recorded = False
        recovery = ops.recovery_operations
        while True:
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if evidence.deterministic_passed:
                if evidence.failures or not required_checks_passed(evidence):
                    raise PipelineFailure(
                        "DURABLE_ARTIFACT_CORRUPTED",
                        "deterministic gate claims PASS without PASS evidence for every required check",
                    )
                ops.accept_gate_state(ctx, cycle_plan, stage, evidence)
                return evidence
            soft = ops.soft_failures(evidence)
            if not soft:
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
            # Ladder-first: one distinct strategy per red gate, consumed
            # durably before anything runs for it and never proposed twice for
            # the same candidate tree and failure.  The frozen budget only
            # bounds the rungs that run a check-repair worker: an inapplicable
            # rung consumes nothing and the ladder advances.
            while True:
                red = evidence
                step = recovery.gate_step(
                    ctx=ctx, cycle_plan=cycle_plan, stage=stage, evidence=red,
                    repair_attempt=attempt, repair_budget=budget,
                    correction_budget=ctx.options.max_correction_cycles,
                )
                self._require_ladder_step(step, red)
                if step.exhausted:
                    raise PipelineFailure(
                        "CHECK_REPAIR_EXHAUSTED",
                        self._check_repair_exhaustion_detail(
                            ctx, cycle_plan, stage, red, soft,
                            attempt=attempt, budget=budget, step=step,
                        ),
                    )
                if step.is_repair_pass and step.repair_attempt != attempt:
                    # A pending ladder pass the checkpoint predates is adopted:
                    # a recorded pass is never replayed, the pending one still
                    # runs exactly once.
                    attempt = self._adopt_pending_ladder_attempt(
                        step, attempt=attempt, durable=durable,
                    )
                    attempt_recorded = False
                recovery.begin_step(
                    ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step, evidence=red,
                )
                if step.is_repair_pass:
                    if step.repair_attempt != attempt:
                        raise PipelineFailure(
                            "RESUME_INTEGRITY_FAILURE",
                            "the ladder repair pass does not follow the durable check-repair attempts",
                        )
                    if not repair_boundary_written:
                        self._boundary(
                            RunPhase.CHECK_REPAIR, cycle_plan, stage=stage,
                            check_repair_attempt=attempt, tree=red.staged_tree_sha,
                        )
                    if not attempt_recorded:
                        ops.check_repair_attempt(ctx, cycle_plan, stage, attempt, red)
                    attempt_recorded = False
                    self._boundary(
                        RunPhase.DETERMINISTIC_GATE, cycle_plan, stage=stage,
                        check_repair_attempt=attempt,
                    )
                    evidence = ops.run_gate(ctx, cycle_plan, stage)
                    recovery.finish_step(
                        ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step,
                        evidence=red, tree_after=evidence.staged_tree_sha,
                    )
                    attempt += 1
                    repair_boundary_written = False
                    break
                # A replan rung rewrites approved work -- the responsible
                # step's contract, or once that rung is spent too the whole
                # decomposition -- from this gate's failure evidence.  A rung
                # these facts do not admit is consumed and the ladder moves on.
                try:
                    if step.strategy is RecoveryStrategy.REPLAN_CYCLE:
                        # The new plan is durable before the rung is done, so a
                        # crash here resumes it without paying again.
                        planned = recovery.replan_cycle(
                            ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step,
                            evidence=red,
                        )
                        recovery.finish_step(
                            ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step,
                            evidence=red, tree_after=red.staged_tree_sha,
                        )
                        return planned
                    tree_after = recovery.replan_step(
                        ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step, evidence=red,
                    )
                except RecoveryStepUnavailable:
                    recovery.finish_step(
                        ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step,
                        evidence=red, tree_after=red.staged_tree_sha,
                    )
                    continue
                recovery.finish_step(
                    ctx=ctx, cycle_plan=cycle_plan, stage=stage, step=step,
                    evidence=red, tree_after=tree_after,
                )
                # The checkpoint names the last durable pass only when its
                # recorded result is exactly the tree this gate verifies.
                records = ops.check_repair_attempts(ctx, number, stage)
                self._boundary(
                    RunPhase.DETERMINISTIC_GATE, cycle_plan, stage=stage,
                    check_repair_attempt=(
                        len(records)
                        if records and records[-1].tree_after == tree_after else None
                    ),
                )
                evidence = ops.run_gate(ctx, cycle_plan, stage)
                break

    def _require_ladder_step(self, step: GateRecoveryStep, evidence: EvidenceBundle) -> None:
        """Fail closed when a ladder step does not describe the red gate."""

        if step.exhausted:
            return
        failed = tuple(
            item.split(":", 1)[1] for item in self.operations.soft_failures(evidence) if ":" in item
        )
        if step.tree != evidence.staged_tree_sha or step.failed_check_ids != failed:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                "the gate recovery ladder step does not describe the red gate evidence",
            )

    @staticmethod
    def _adopt_pending_ladder_attempt(
        step: GateRecoveryStep, *, attempt: int, durable: int,
    ) -> int:
        """Adopt the pending ladder pass a checkpoint may predate.

        The ladder consumes a rung durably before it runs, so a checkpoint that
        stopped before that pending pass is reconciled to it instead of the
        reverse.  A pass that is already recorded is never replayed: only the
        next unrecorded bounded pass of the ladder is ever adopted.
        """

        if (
            step.repair_attempt is not None
            and step.repair_attempt == durable + 1
            and attempt <= durable
        ):
            return step.repair_attempt
        raise PipelineFailure(
            "RESUME_INTEGRITY_FAILURE",
            "the ladder repair pass does not follow the durable check-repair attempts",
        )

    @staticmethod
    def _check_repair_exhaustion_detail(
        ctx: PipelineV2Context, cycle_plan: CyclePlan, stage: GateStage,
        evidence: EvidenceBundle, soft: Sequence[str], *,
        attempt: int, budget: int, step: GateRecoveryStep | None = None,
    ) -> dict[str, Any]:
        """The durable detail of one exhausted deterministic gate episode."""

        try:
            evidence_sha256 = hashlib.sha256(
                (gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage) / "evidence.json").read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "latest deterministic evidence is unreadable",
            ) from exc
        detail: dict[str, Any] = {
            "failed_check_ids": [item.split(":", 1)[1] for item in soft if ":" in item],
            "candidate_tree": evidence.staged_tree_sha,
            "attempt_count": attempt - 1,
            "budget": budget,
            "latest_evidence_sha256": evidence_sha256,
        }
        if step is not None:
            detail["strategy"] = step.fingerprint_strategy
            detail["strategies"] = [item.value for item in step.consumed]
        return detail

    # -- durable boundaries ------------------------------------------------------

    def _boundary(
        self, phase: RunPhase, cycle_plan: CyclePlan, *, stage: GateStage | None = None,
        step_id: str | None = None, check_repair_attempt: int | None = None,
        tree: str | None = None,
    ) -> None:
        ctx = self.context
        self._phase(phase)
        self.operations.checkpoint(
            ctx, phase,
            cycle=cycle_plan.cycle.number,
            head=self.operations.current_head(ctx),
            tree=tree or self.operations.candidate_tree(ctx),
            stage=stage, step_id=step_id, check_repair_attempt=check_repair_attempt,
            correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
        )

    def _candidate_boundary(
        self, phase: RunPhase, cycle_plan: CyclePlan, candidate: Mapping[str, Any],
    ) -> None:
        self._phase(phase)
        self.operations.checkpoint(
            self.context, phase,
            cycle=cycle_plan.cycle.number,
            head=candidate["commit_sha"], tree=candidate["tree_sha"],
            expected_parent_sha=candidate["parent_sha"],
            correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
        )


__all__ = [
    "CyclePlan", "PipelineFailure", "PipelineV2Context", "PipelineV2Coordinator",
    "RecoveryStepUnavailable",
    "PipelineV2Operations", "candidate_dir", "check_repair_attempt_dir",
    "check_repair_attempts_dir", "check_repair_dir", "check_repair_root", "correction_dir", "correction_kind", "cycle_dir",
    "cycle_record_path", "final_gate_stage", "gate_acceptance_path", "gate_dir", "implementation_dir", "implementation_steps_dir",
    "pre_semantic_gate_stage", "step_dir",
    "review_dir", "semantic_revision_dir",
]
