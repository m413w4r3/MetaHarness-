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
    """Side effects of the single implementation, audit and acceptance path."""

    checkpoint: Callable[..., None]
    current_head: Callable[[PipelineV2Context], str]
    candidate_tree: Callable[[PipelineV2Context], str]
    begin_cycle: Callable[[PipelineV2Context, RunCycle, bool], None]
    initial_plan: Callable[[PipelineV2Context], CyclePlan]
    completed_steps: Callable[[PipelineV2Context, CyclePlan], list[dict[str, Any]]]
    execute_step: Callable[[PipelineV2Context, CyclePlan, int], None]
    accept_step: Callable[[PipelineV2Context, CyclePlan, int], None]
    run_gate: Callable[[PipelineV2Context, CyclePlan, GateStage], EvidenceBundle]
    load_gate_evidence: Callable[[PipelineV2Context, int, GateStage], EvidenceBundle | None]
    run_audit: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle, int], Any]
    accept_gate_state: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    hard_failures: Callable[[EvidenceBundle], list[str]]
    create_candidate: Callable[[PipelineV2Context, CyclePlan, GateStage, EvidenceBundle], Mapping[str, Any]]
    load_candidate: Callable[[PipelineV2Context, int], Mapping[str, Any]]
    push_candidate: Callable[[PipelineV2Context, int, Mapping[str, Any]], Mapping[str, Any]]
    publish: Callable[[PipelineV2Context, int, Mapping[str, Any]], RunResult]


@dataclass(frozen=True)
class PipelineV2Coordinator:
    """One batch: implement, diagnose, audit, rerun, accept and publish."""

    context: PipelineV2Context
    operations: PipelineV2Operations
    _cursor: list[RunPhase] = dataclasses.field(default_factory=list, repr=False, compare=False)

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
        self, phase: RunPhase, plan: CyclePlan, *, stage: GateStage | None = None,
        step_id: str | None = None,
    ) -> None:
        self._phase(phase)
        self.operations.checkpoint(
            self.context, phase, cycle=plan.cycle.number,
            head=self.operations.current_head(self.context),
            tree=self.operations.candidate_tree(self.context),
            stage=stage, step_id=step_id,
        )

    def _candidate_boundary(
        self, phase: RunPhase, plan: CyclePlan, candidate: Mapping[str, Any],
    ) -> None:
        self._phase(phase)
        self.operations.checkpoint(
            self.context, phase, cycle=plan.cycle.number,
            head=candidate["commit_sha"], tree=candidate["tree_sha"],
            expected_parent_sha=candidate["parent_sha"],
        )

    def run(self, start: ResumeCheckpoint, *, resumed: bool) -> RunResult:
        ops, ctx = self.operations, self.context
        if start.review_cycle != 1:
            raise PipelineFailure("RUN_SCHEMA_UNSUPPORTED", "old correction cycles cannot resume")
        if start.phase in {
            RunPhase.CONTEXT, RunPhase.PLANNER, RunPhase.PLAN_APPROVAL,
            RunPhase.WORKTREE_SETUP,
        }:
            raise ValueError(f"{start.phase.value} is not an execution checkpoint")
        cycle = RunCycle(1, CycleKind.INITIAL)
        ops.begin_cycle(ctx, cycle, not resumed)
        plan = ops.initial_plan(ctx)
        stage = GateStage.POST_IMPLEMENTATION
        if start.phase is RunPhase.PUBLISH:
            return ops.publish(ctx, 1, ops.load_candidate(ctx, 1))
        if start.phase in {RunPhase.IMPLEMENT_STEP, RunPhase.STEP_ACCEPTANCE}:
            if start.phase is RunPhase.STEP_ACCEPTANCE:
                ids = [step.id for step in plan.plan.steps]
                if start.step_id not in ids:
                    raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted step is absent from the plan")
                ops.accept_step(ctx, plan, ids.index(start.step_id))
            done = {item["id"] for item in ops.completed_steps(ctx, plan)}
            for index, step in enumerate(plan.plan.steps):
                if step.id in done:
                    continue
                self._boundary(RunPhase.IMPLEMENT_STEP, plan, step_id=step.id)
                ops.execute_step(ctx, plan, index)
            self._boundary(RunPhase.DETERMINISTIC_GATE, plan, stage=stage)
            evidence = self._gate(plan, stage, None)
        elif start.phase in {RunPhase.DETERMINISTIC_GATE, RunPhase.AUDIT}:
            evidence = self._gate(plan, stage, start)
        else:
            evidence = ops.load_gate_evidence(ctx, 1, stage)
            if evidence is None or not evidence.deterministic_passed:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted gate evidence is missing")
        if start.phase in {RunPhase.CANDIDATE_READY, RunPhase.CANDIDATE_PUSH}:
            candidate = ops.load_candidate(ctx, 1)
        else:
            self._boundary(RunPhase.CANDIDATE_READY, plan)
            candidate = ops.create_candidate(ctx, plan, stage, evidence)
        if start.phase is not RunPhase.CANDIDATE_PUSH:
            self._candidate_boundary(RunPhase.CANDIDATE_PUSH, plan, candidate)
        candidate = ops.push_candidate(ctx, 1, candidate)
        self._candidate_boundary(RunPhase.PUBLISH, plan, candidate)
        return ops.publish(ctx, 1, candidate)

    def _gate(
        self, plan: CyclePlan, stage: GateStage, start: ResumeCheckpoint | None,
    ) -> EvidenceBundle:
        ops, ctx = self.operations, self.context
        reports = sorted((cycle_dir(ctx.run_dir, plan.cycle) / "audit").glob("*/report.json"))
        if start is not None and start.phase is RunPhase.AUDIT and not reports:
            evidence = ops.load_gate_evidence(ctx, 1, stage)
            if evidence is None or evidence.staged_tree_sha != start.expected_tree_sha:
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
        remaining = reports and report.get("remaining", []) or []
        if reports and evidence.deterministic_passed and required_checks_passed(evidence):
            ops.accept_gate_state(ctx, plan, stage, evidence)
            return evidence
        for attempt in range(len(reports) + 1, 3):
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if not evidence.changed_files:
                if evidence.deterministic_passed and required_checks_passed(evidence):
                    ops.accept_gate_state(ctx, plan, stage, evidence)
                    return evidence
                raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
            self._boundary(RunPhase.AUDIT, plan)
            audit = ops.run_audit(ctx, plan, stage, evidence, attempt)
            remaining = list(audit.remaining)
            if audit.status == "SPEC_DECISION":
                raise PipelineFailure("SPEC_DECISION_REQUIRED", {
                    "remaining": remaining, "failure_ids": list(evidence.failures),
                })
            self._boundary(RunPhase.DETERMINISTIC_GATE, plan, stage=stage)
            evidence = ops.run_gate(ctx, plan, stage)
            hard = ops.hard_failures(evidence)
            if hard:
                raise PipelineFailure(hard[0].split(":", 1)[0], ", ".join(hard))
            if evidence.deterministic_passed:
                if evidence.failures or not required_checks_passed(evidence):
                    raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "PASS gate has incomplete evidence")
                ops.accept_gate_state(ctx, plan, stage, evidence)
                return evidence
        raise PipelineFailure("AUDIT_REMAINING", {
            "remaining": remaining,
            "failure_ids": list(evidence.failures),
            "candidate_tree": evidence.staged_tree_sha,
        })


__all__ = [
    "CyclePlan", "PipelineFailure", "PipelineV2Context", "PipelineV2Coordinator",
    "RecoveryStepUnavailable",
    "PipelineV2Operations", "candidate_dir", "check_repair_attempt_dir",
    "check_repair_attempts_dir", "check_repair_dir", "check_repair_root", "correction_dir", "correction_kind", "cycle_dir",
    "cycle_record_path", "final_gate_stage", "gate_acceptance_path", "gate_dir", "implementation_dir", "implementation_steps_dir",
    "pre_semantic_gate_stage", "step_dir",
    "review_dir", "semantic_revision_dir",
]
