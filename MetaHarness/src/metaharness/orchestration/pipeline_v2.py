"""The generic pipeline-v2 state machine.

``PipelineV2Coordinator`` owns the order of the durable phases and the
checkpoint written before each of them.  It never owns an ``Orchestrator``:
every operation it sequences (a worker step, a deterministic gate, a
candidate commit, a review, the publication) is injected explicitly through
:class:`PipelineV2Operations`.

A run is a sequence of cycles ``001, 002, ...``.  Each cycle executes one
approved plan (the operator-approved plan for the initial cycle, a
review-driven correction plan afterwards), optionally a semantic revision,
one or two deterministic gate episodes with their bounded check-repair
attempts, the accepted candidate HEAD, its push and one final review.  The number of
cycles is bounded only by the frozen run options, never by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..evidence import EvidenceBundle
from ..gitops import RepositoryReference, WorktreeInfo
from ..models import (
    CycleKind,
    ExecutionSelection,
    GateStage,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    TaskPlanV2,
)
from ..result import RunResult
from ..resume import ResumeCheckpoint, ResumePhase
from ..review import ReviewResult
from ..run_options import RunOptions


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


class PipelineFailure(Exception):
    """A terminal, classified failure of the current operation."""

    def __init__(self, reason: str, detail: Any = None, *, step_id: str | None = None) -> None:
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
        [PipelineV2Context, RunCycle, bool], tuple[CyclePlan, ReviewResult]
    ]
    plan_correction: Callable[[PipelineV2Context, RunCycle], CyclePlan]
    load_correction: Callable[[PipelineV2Context, RunCycle, str | None], CyclePlan]
    # Implementation.
    completed_steps: Callable[[PipelineV2Context, CyclePlan], list[dict[str, Any]]]
    execute_step: Callable[[PipelineV2Context, CyclePlan, int], None]
    unresolved_mismatches: Callable[[PipelineV2Context, CyclePlan], bool]
    semantic_revision: Callable[[PipelineV2Context, CyclePlan], None]
    semantic_review_correction: Callable[
        [PipelineV2Context, CyclePlan, ReviewResult], None
    ]
    # Deterministic gate episode.
    run_gate: Callable[[PipelineV2Context, CyclePlan, GateStage], EvidenceBundle]
    load_gate_evidence: Callable[[PipelineV2Context, int, GateStage], EvidenceBundle | None]
    accept_gate_state: Callable[
        [PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]
    ]
    check_repair_attempts: Callable[[PipelineV2Context, int, GateStage], tuple[Any, ...]]
    check_repair_attempt: Callable[
        [PipelineV2Context, CyclePlan, GateStage, int, EvidenceBundle], None
    ]
    hard_failures: Callable[[EvidenceBundle], list[str]]
    soft_failures: Callable[[EvidenceBundle], list[str]]
    # Candidate, review and publication.
    create_candidate: Callable[
        [PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]
    ]
    load_candidate: Callable[[PipelineV2Context, int], Mapping[str, Any]]
    push_candidate: Callable[[PipelineV2Context, int, Mapping[str, Any]], Mapping[str, Any]]
    review_candidate: Callable[
        [PipelineV2Context, CyclePlan, Mapping[str, Any], EvidenceBundle], ReviewResult
    ]
    record_review: Callable[[PipelineV2Context, int, ReviewResult, EvidenceBundle], None]
    request_human: Callable[[PipelineV2Context, int, ReviewResult, str], RunResult]
    review_repair_exhausted: Callable[
        [PipelineV2Context, int, ReviewResult, Mapping[str, Any]], RunResult
    ]
    publish: Callable[[PipelineV2Context, int, Mapping[str, Any]], RunResult]


@dataclass(frozen=True)
class PipelineV2Coordinator:
    """Sequence the generic cycles of one prepared pipeline-v2 run."""

    context: PipelineV2Context
    operations: PipelineV2Operations

    def run(self, start: ResumeCheckpoint, *, resumed: bool) -> RunResult:
        """Execute from *start*: the first step of a new run, or a checkpoint.

        Every operation before *start* is durable and is never replayed; the
        operations read back what they need from their durable artifacts.
        """

        if start.phase in {
            ResumePhase.CONTEXT, ResumePhase.PLANNER, ResumePhase.PLAN_APPROVAL,
            ResumePhase.WORKTREE_SETUP,
        }:
            raise ValueError(f"{start.phase.value} is not an execution checkpoint")
        cycle = self._cycle_at(start)
        entry: ResumeCheckpoint | None = start
        fresh = not resumed
        while True:
            outcome = self._run_cycle(cycle, entry, fresh=fresh)
            if isinstance(outcome, RunResult):
                return outcome
            cycle = RunCycle(cycle.number + 1, correction_kind(outcome.route))
            entry, fresh = None, True

    # -- one cycle -----------------------------------------------------------

    def _cycle_at(self, start: ResumeCheckpoint) -> RunCycle:
        if start.review_cycle == 1:
            return RunCycle(1, CycleKind.INITIAL)
        return self.operations.load_cycle(self.context, start.review_cycle)

    def _run_cycle(
        self, cycle: RunCycle, start: ResumeCheckpoint | None, *, fresh: bool,
    ) -> RunResult | ReviewResult:
        ops, ctx = self.operations, self.context
        ops.begin_cycle(ctx, cycle, fresh)
        if start is not None and start.phase is ResumePhase.PUBLISH:
            return ops.publish(ctx, cycle.number, ops.load_candidate(ctx, cycle.number))

        if cycle.kind is CycleKind.INITIAL:
            cycle_plan = ops.initial_plan(ctx)
            previous_review = None
        elif cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            cycle_plan, previous_review = ops.review_implementation_correction(
                ctx, cycle, start is None or start.phase is ResumePhase.SEMANTIC_REVISION,
            )
        elif start is None or start.phase is ResumePhase.REVIEW_REPLAN:
            if start is None:
                ops.checkpoint(
                    ctx, ResumePhase.REVIEW_REPLAN, cycle=cycle.number,
                    head=ops.current_head(ctx), tree=ops.candidate_tree(ctx),
                )
            cycle_plan = ops.plan_correction(ctx, cycle)
            previous_review = None
        else:
            cycle_plan = ops.load_correction(ctx, cycle, start.correction_bundle_sha256)
            previous_review = None

        phase = (
            ResumePhase.IMPLEMENT_STEP
            if cycle.kind is CycleKind.INITIAL
            else (
                ResumePhase.SEMANTIC_REVISION
                if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION
                else ResumePhase.REVIEW_IMPLEMENTATION
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
                self._boundary(ResumePhase.DETERMINISTIC_GATE, cycle_plan, stage=final_stage)
        elif start is None or start.phase is phase or start.phase is ResumePhase.REVIEW_REPLAN:
            self._implement(cycle_plan, next_stage=pre_stage or final_stage)
        if ops.unresolved_mismatches(ctx, cycle_plan) and not semantic_enabled:
            raise PipelineFailure(
                "UNRESOLVED_CONTRACT_MISMATCH",
                "HUMAN_REQUIRED: semantic revision is disabled while contract mismatches are deferred",
            )

        # Initial and replan cycles have a gate before semantic revision.
        if pre_stage is not None and (
            start is None or start.phase in {phase, ResumePhase.REVIEW_REPLAN}
            or self._is_gate_checkpoint(start, pre_stage)
        ):
            self._gate_episode(
                cycle_plan, pre_stage,
                start if self._is_gate_checkpoint(start, pre_stage) else None,
            )

        if semantic_enabled and cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION and (
            start is None or start.phase in {
                phase, ResumePhase.REVIEW_REPLAN, ResumePhase.SEMANTIC_REVISION,
            } or self._is_gate_checkpoint(start, pre_stage)
        ):
            self._boundary(ResumePhase.SEMANTIC_REVISION, cycle_plan)
            ops.semantic_revision(ctx, cycle_plan)
            self._boundary(ResumePhase.DETERMINISTIC_GATE, cycle_plan, stage=final_stage)

        final_start = start if self._is_gate_checkpoint(start, final_stage) else None
        should_run_final_gate = (
            final_start is not None
            or start is None
            or start.phase is phase
            or start.phase is ResumePhase.REVIEW_REPLAN
            or start.phase is ResumePhase.SEMANTIC_REVISION
            or self._is_gate_checkpoint(start, pre_stage)
        )
        if semantic_enabled:
            # A pre-semantic gate was already consumed above; the final gate is
            # still needed after the revision.  A final gate checkpoint resumes
            # its own episode without replaying the revision.
            evidence = self._gate_episode(cycle_plan, final_stage, final_start) if should_run_final_gate else None
        elif pre_stage is not None:
            evidence = ops.load_gate_evidence(ctx, cycle.number, pre_stage)
        else:
            evidence = self._gate_episode(cycle_plan, final_stage, final_start) if should_run_final_gate else None
        if evidence is None:
            evidence = ops.load_gate_evidence(ctx, cycle.number, final_stage if semantic_enabled else (pre_stage or final_stage))
        if evidence is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "the final gate evidence is missing")

        stage = final_stage if semantic_enabled else (pre_stage or final_stage)
        if start is None or start.phase not in {
            ResumePhase.CANDIDATE_READY, ResumePhase.CANDIDATE_PUSH,
            ResumePhase.FINAL_REVIEW, ResumePhase.PUBLISH,
        }:
            self._boundary(ResumePhase.CANDIDATE_READY, cycle_plan, tree=evidence.staged_tree_sha)
            candidate = ops.create_candidate(ctx, cycle_plan, stage, evidence)
        else:
            candidate = ops.load_candidate(ctx, cycle.number)
        if start is None or start.phase in {
            phase, ResumePhase.REVIEW_REPLAN, ResumePhase.SEMANTIC_REVISION,
            ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR,
            ResumePhase.CANDIDATE_READY,
        }:
            self._candidate_boundary(ResumePhase.CANDIDATE_PUSH, cycle_plan, candidate)
            candidate = ops.push_candidate(ctx, cycle.number, candidate)
        elif start.phase is ResumePhase.CANDIDATE_PUSH:
            candidate = ops.push_candidate(ctx, cycle.number, candidate)
        if start is None or start.phase in {
            phase, ResumePhase.SEMANTIC_REVISION, ResumePhase.DETERMINISTIC_GATE,
            ResumePhase.CHECK_REPAIR, ResumePhase.CANDIDATE_READY,
            ResumePhase.CANDIDATE_PUSH, ResumePhase.FINAL_REVIEW,
        }:
            self._candidate_boundary(ResumePhase.FINAL_REVIEW, cycle_plan, candidate)
        review = ops.review_candidate(ctx, cycle_plan, candidate, evidence)
        ops.record_review(ctx, cycle.number, review, evidence)
        return self._route(cycle_plan, candidate, evidence, review)

    @staticmethod
    def _is_gate_checkpoint(
        start: ResumeCheckpoint | None, stage: GateStage | None,
    ) -> bool:
        return bool(
            start is not None and stage is not None
            and start.phase in {ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR}
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
                "REVIEW_FAILED",
                {
                    "summary": review.summary,
                    "findings": review.findings,
                    "required_fixes": review.required_fixes,
                    "missing_tests": review.missing_tests,
                },
            )
        if review.route is ReviewRoute.HUMAN:
            return ops.request_human(ctx, cycle.number, review, "HUMAN_REQUIRED")
        if review.route not in {ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN}:
            return ops.request_human(ctx, cycle.number, review, "HUMAN_REQUIRED")
        budget = ctx.options.max_review_repair_cycles
        # Cycle 001 is the initial implementation; every later cycle spends
        # one review-repair unit of the frozen budget.
        corrections_used = cycle.number - 1
        if corrections_used >= budget:
            return ops.review_repair_exhausted(
                ctx, cycle.number, review,
                {
                    "last_review_cycle": cycle.number,
                    "corrections_used": corrections_used,
                    "max_review_repair_cycles": budget,
                    "last_route": review.route.value,
                },
            )
        return review

    # -- phases ----------------------------------------------------------------

    def _implement(self, cycle_plan: CyclePlan, *, next_stage: GateStage) -> None:
        ops, ctx = self.operations, self.context
        phase = (
            ResumePhase.IMPLEMENT_STEP if cycle_plan.cycle.kind is CycleKind.INITIAL
            else ResumePhase.REVIEW_IMPLEMENTATION
        )
        done = {record["id"] for record in ops.completed_steps(ctx, cycle_plan)}
        steps = cycle_plan.plan.steps
        for index, step in enumerate(steps):
            if step.id in done:
                continue
            self._boundary(phase, cycle_plan, step_id=step.id)
            ops.execute_step(ctx, cycle_plan, index)
        self._boundary(ResumePhase.DETERMINISTIC_GATE, cycle_plan, stage=next_stage)

    def _gate_episode(
        self, cycle_plan: CyclePlan, stage: GateStage, start: ResumeCheckpoint | None,
    ) -> EvidenceBundle:
        """One deterministic gate and its bounded check-repair attempts."""

        ops, ctx = self.operations, self.context
        number = cycle_plan.cycle.number
        budget = ctx.options.max_check_repair_attempts
        if start is not None and start.phase is ResumePhase.CHECK_REPAIR:
            evidence = ops.load_gate_evidence(ctx, number, stage)
            if evidence is None or evidence.staged_tree_sha != start.expected_tree_sha:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    "the red gate evidence of the check-repair attempt is missing",
                )
            attempt = int(start.check_repair_attempt or 1)
            # The budget is spent from durable attempt records, never from the
            # checkpoint alone: it names the next attempt, or one already
            # recorded whose worker is never called again.
            durable = len(ops.check_repair_attempts(ctx, number, stage))
            if attempt not in {durable, durable + 1}:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    "the check-repair attempt does not follow the durable attempts",
                )
            repair_boundary_written = True
            attempt_recorded = attempt == durable
        else:
            evidence = ops.run_gate(ctx, cycle_plan, stage)
            attempt = len(ops.check_repair_attempts(ctx, number, stage)) + 1
            repair_boundary_written = False
            attempt_recorded = False
        while True:
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if evidence.deterministic_passed:
                ops.accept_gate_state(ctx, cycle_plan, stage, evidence)
                return evidence
            soft = ops.soft_failures(evidence)
            if not soft:
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
            if budget <= 0:
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
            if attempt > budget:
                raise PipelineFailure(
                    "CHECK_REPAIR_EXHAUSTED",
                    {
                        "remaining_failed_check_ids": [
                            item.split(":", 1)[1] for item in soft if ":" in item
                        ],
                        "attempts": attempt - 1,
                    },
                )
            if not repair_boundary_written:
                self._boundary(
                    ResumePhase.CHECK_REPAIR, cycle_plan, stage=stage,
                    check_repair_attempt=attempt, tree=evidence.staged_tree_sha,
                )
            if not attempt_recorded:
                ops.check_repair_attempt(ctx, cycle_plan, stage, attempt, evidence)
            attempt_recorded = False
            self._boundary(
                ResumePhase.DETERMINISTIC_GATE, cycle_plan, stage=stage,
                check_repair_attempt=attempt,
            )
            evidence = ops.run_gate(ctx, cycle_plan, stage)
            attempt += 1
            repair_boundary_written = False

    # -- durable boundaries ------------------------------------------------------

    def _boundary(
        self, phase: ResumePhase, cycle_plan: CyclePlan, *, stage: GateStage | None = None,
        step_id: str | None = None, check_repair_attempt: int | None = None,
        tree: str | None = None,
    ) -> None:
        ctx = self.context
        self.operations.checkpoint(
            ctx, phase,
            cycle=cycle_plan.cycle.number,
            head=self.operations.current_head(ctx),
            tree=tree or self.operations.candidate_tree(ctx),
            stage=stage, step_id=step_id, check_repair_attempt=check_repair_attempt,
            correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
        )

    def _candidate_boundary(
        self, phase: ResumePhase, cycle_plan: CyclePlan, candidate: Mapping[str, Any],
    ) -> None:
        self.operations.checkpoint(
            self.context, phase,
            cycle=cycle_plan.cycle.number,
            head=candidate["commit_sha"], tree=candidate["tree_sha"],
            expected_parent_sha=candidate["parent_sha"],
            correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
        )


__all__ = [
    "CyclePlan", "PipelineFailure", "PipelineV2Context", "PipelineV2Coordinator",
    "PipelineV2Operations", "candidate_dir", "check_repair_attempt_dir",
    "check_repair_attempts_dir", "check_repair_dir", "check_repair_root", "correction_dir", "correction_kind", "cycle_dir",
    "cycle_record_path", "final_gate_stage", "gate_acceptance_path", "gate_dir", "implementation_dir", "implementation_steps_dir",
    "pre_semantic_gate_stage", "step_dir",
    "review_dir", "semantic_revision_dir",
]
