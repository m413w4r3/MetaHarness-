"""The red-gate step replan: rewrite one step's contract and re-run its suffix.

The rung names the evidence-proven responsible step and its descendants.  Their
accepted boundary is re-derived from their own durable records, the run branch
is rewound to exactly that boundary, the step's contract is rewritten through
the durable contract repair transaction inside the operator-approved scope, its
new authority is proven from the durable slot, and the responsible step with
its descendants is re-executed under that authority before the gate runs again.

A rung whose durable facts do not prove this boundary raises
:class:`RecoveryStepUnavailable` before the repository is touched, and a resume
re-enters the same durable slot with the same operation id.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Sequence,
    TYPE_CHECKING,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    GitError,
    candidate_tree_sha,
    current_head,
    is_ancestor,
    resolve_tree,
    rewind_worktree,
    status_porcelain,
    symbolic_head,
)
from ..models import (
    CycleKind,
    GateStage,
    RunStatus,
)
from ..result import atomic_write_text
from ..resume import ResumePhase, read_checkpoint
from ..state import RunStateStore
from . import contract_repair
from .candidate import accepted_chain_records
from .check_repair import (
    replan_failure_evidence,
    replan_mismatch,
    replan_problem,
)
from .contract_recovery import (
    GATE_REPLAN_EVIDENCE,
    ContractRecoveryService,
)
from .contract_repair import ContractRepairIntegrityError
from .recovery import GateRecoveryStep
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    RecoveryStepUnavailable,
    gate_dir,
    step_dir as cycle_step_dir,
)
from .revision import future_step_ownership
from .shared import (
    archive_attempt,
    bounded_v2_report,
    is_object_id,
    json_text,
    read_json_artifact,
    status_has_unstaged_or_untracked,
)
from .step_authority import approved_step_contract


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


class StepReplanService:
    """One owner of the red-gate step replan transaction described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def replan_cycle_step(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> str:
        """Rewrite the contract of the responsible approved step and re-run it.

        The rung names the evidence-proven responsible step and its
        descendants.  Their accepted boundary is re-derived from their own
        durable records, the run branch is rewound to exactly that boundary,
        the step's contract is rewritten through the existing durable contract
        repair transaction inside the operator-approved scope, its new
        authority is proven from the durable slot, and the responsible step
        with its descendants is re-executed under that authority before the
        gate runs again.  A resume re-enters the same slot with the same
        operation id: the planner is never called twice for one repair, and no
        second slot is ever opened.

        A rung whose durable facts do not prove this boundary raises
        :class:`RecoveryStepUnavailable` before the repository is touched, so
        the ladder advances without consuming anything.
        """

        strategy = step.strategy
        worktree = ctx.info.worktree
        steps = list(cycle_plan.plan.steps)
        indices = tuple(step.step_indices)
        number = cycle_plan.cycle.number
        red_tree = evidence.staged_tree_sha
        if not indices or indices != tuple(range(indices[0], len(steps))):
            raise RecoveryStepUnavailable(
                strategy, "the rung does not replan a suffix of the approved cycle",
            )
        first = indices[0]
        records = self._replan_records(ctx, cycle_plan, indices, strategy)
        anchor_tree = records[0]["tree_before"]
        anchor_commit = self._replan_anchor(ctx, cycle_plan, first, anchor_tree, strategy)
        for index, (previous, record) in enumerate(zip(records, records[1:], strict=False)):
            if record["tree_before"] != previous["tree_after"]:
                raise RecoveryStepUnavailable(
                    strategy,
                    f"the durable records of step {steps[indices[index + 1]].id} are not contiguous",
                )
        step_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, steps[first].id)
        # The slot this rung already opened is the durable identity of the
        # replan: a resume never opens a second one, and it may find the
        # worktree anywhere between the accepted boundary and the tree the
        # interrupted re-execution produced.
        directory = self._gate_replan_slot(
            step_dir, cycle=number, stage=stage, step_id=steps[first].id,
            anchor_tree=anchor_tree,
        )
        current_tree = candidate_tree_sha(worktree)
        if directory is None and current_tree not in {red_tree, anchor_tree}:
            raise RecoveryStepUnavailable(
                strategy, "the rung does not describe the current candidate tree",
            )
        self._rewind_replan_boundary(
            store, ctx, cycle_plan, anchor_commit=anchor_commit, anchor_tree=anchor_tree,
            step_ids=[steps[index].id for index in indices],
        )
        # The re-execution starts exactly at the accepted boundary: the
        # checkpoint describes that tree before any worker runs again.
        self.runtime.write_checkpoint(
            ctx.run_dir, self._replan_phase(cycle_plan),
            head=anchor_commit, tree=anchor_tree, cycle=number,
            step_id=steps[first].id,
            correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
        )
        approved_contract = approved_step_contract(cycle_plan, steps[first])
        authority = self.runtime.contract_recovery.resolve_step_authority(
            step_dir, steps[first], approved_contract, expected_tree=anchor_tree,
            expected_plan_step_count=len(steps),
        )
        effective_step = authority.effective_step
        effective_contract = authority.effective_contract
        problem = replan_problem(
            evidence=evidence, evidence_dir=gate_dir(ctx.run_dir, number, stage),
            repo=ctx.repo, worktree=worktree,
            approved_scope=tuple(cycle_plan.mutable_scope),
        )
        mismatch = bounded_v2_report(replan_mismatch(
            problem=problem, step_id=steps[first].id, cycle=number, stage=stage,
            anchor_tree=anchor_tree,
        ))
        resumed = directory is not None
        repair_number = (
            contract_repair.next_repair_number(step_dir) if directory is None
            else int(directory.name)
        )
        previous_repairs = tuple(
            contract_repair.episode_summary(item)
            for item in contract_repair.repair_dirs(step_dir)
            if item.name.isdigit() and int(item.name) < repair_number
        )
        failure_evidence = replan_failure_evidence(
            problem=problem, repo=ctx.repo, step=effective_step, anchor_tree=anchor_tree,
            previous_repairs=previous_repairs,
        )
        if directory is None:
            directory = step_dir / "contract_repairs" / f"{repair_number:02d}"
            try:
                contract_repair.begin(
                    directory, number=repair_number, cycle=number,
                    step_id=steps[first].id, current_contract=effective_contract,
                    mismatch=mismatch, tree_sha=anchor_tree,
                    output_correction_limit=(
                        self.runtime.run_options.recovery.max_contract_repair_output_corrections
                    ),
                )
                # The evidence is part of the slot's durable identity: a
                # resume rebuilds the same planner request from it.
                atomic_write_text(directory / GATE_REPLAN_EVIDENCE, failure_evidence)
            except (ContractRepairIntegrityError, OSError) as exc:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", str(exc), step_id=steps[first].id,
                ) from exc
        checkpoint = read_checkpoint(ctx.run_dir)
        self.runtime.contract_recovery.repair_transaction(
            store=store, recovery=self.runtime.recovery(store), repo=ctx.repo,
            worktree=worktree, run_dir=ctx.run_dir, artifact_dir=step_dir,
            directory=directory, cycle=number, number=repair_number,
            step=effective_step, current_contract=effective_contract, mismatch=mismatch,
            tree_before=anchor_tree, original_spec=ctx.spec,
            original_plan_identity=json_text(
                asdict(checkpoint.plan_identity) if checkpoint and checkpoint.plan_identity else {}
            ),
            future_ownership=future_step_ownership(steps, first),
            max_repairs=getattr(self.runtime.run_options, "max_step_contract_repairs", 0),
            profile_id=cycle_plan.step_profile_ids[steps[first].id],
            resumed=resumed, expected_plan_step_count=len(steps),
            failure_evidence=failure_evidence, require_new_contract=True,
        )
        repaired = self.runtime.contract_recovery.repaired_authority(
            self.runtime.contract_recovery.resolve_step_authority(
                step_dir, steps[first], approved_contract, expected_tree=anchor_tree,
                expected_plan_step_count=len(steps),
            ),
            repair_number,
        )
        self.runtime.observability.trace_emit(
            "recovery.replanned", phase="implementation", cycle=number,
            step_id=steps[first].id,
            data={
                "strategy": strategy.value, "stage": GateStage(stage).value,
                "anchor_commit": anchor_commit, "tree_before": anchor_tree,
                "candidate_tree": problem.red_tree,
                "step_ids": [steps[index].id for index in indices],
                "contract_repair": repair_number,
                "repair_id": contract_repair.repair_identity(
                    number, steps[first].id, repair_number,
                ),
                "effective_contract_sha256": repaired.effective_contract_sha256,
                "resumed": resumed,
            },
        )
        return self._execute_replanned_suffix(
            store, ctx, cycle_plan, indices, anchor_tree=anchor_tree,
        )
    @staticmethod
    def _replan_records(
        ctx: PipelineV2Context, cycle_plan: CyclePlan, indices: tuple[int, ...],
        strategy: Any,
    ) -> list[dict[str, Any]]:
        """The durable accepted records that prove one replan boundary."""

        steps = list(cycle_plan.plan.steps)
        records: list[dict[str, Any]] = []
        for index in indices:
            record = read_json_artifact(
                cycle_step_dir(ctx.run_dir, cycle_plan.cycle, steps[index].id) / "step.json",
                128 * 1024,
            )
            if (
                not isinstance(record, dict) or record.get("id") != steps[index].id
                or not is_object_id(record.get("tree_before"))
                or not is_object_id(record.get("tree_after"))
            ):
                raise RecoveryStepUnavailable(
                    strategy, f"step {steps[index].id} has no replannable durable record",
                )
            records.append(record)
        return records
    @staticmethod
    def _replan_anchor(
        ctx: PipelineV2Context, cycle_plan: CyclePlan, first: int, anchor_tree: str,
        strategy: Any,
    ) -> str:
        """The accepted commit the responsible step started from."""

        worktree = ctx.info.worktree
        if first == 0:
            anchor_commit = ctx.base_sha
        else:
            record = read_json_artifact(
                cycle_step_dir(
                    ctx.run_dir, cycle_plan.cycle, cycle_plan.plan.steps[first - 1].id,
                )
                / "step.json",
                128 * 1024,
            )
            anchor_commit = record.get("commit_sha") if isinstance(record, dict) else None
            if not is_object_id(anchor_commit):
                raise RecoveryStepUnavailable(
                    strategy, "the accepted boundary of the responsible step is not a commit",
                )
        try:
            boundary_tree = resolve_tree(worktree, anchor_commit)
        except GitError as exc:
            raise RecoveryStepUnavailable(
                strategy, f"the accepted boundary of the responsible step is unreadable: {exc}",
            ) from exc
        if boundary_tree != anchor_tree:
            raise RecoveryStepUnavailable(
                strategy, "the accepted boundary of the responsible step is another tree",
            )
        if not is_ancestor(worktree, anchor_commit, current_head(worktree)):
            raise RecoveryStepUnavailable(
                strategy, "the responsible step boundary is not on the run branch",
            )
        return anchor_commit
    @staticmethod
    def _gate_replan_slot(
        step_dir: Path, *, cycle: int, stage: GateStage, step_id: str, anchor_tree: str,
    ) -> Path | None:
        """The durable contract repair slot a previous try of this rung opened."""

        for directory in reversed(contract_repair.repair_dirs(step_dir)):
            try:
                transaction = contract_repair.read_transaction(directory)
            except ContractRepairIntegrityError:
                continue
            if transaction is None or transaction["status"] == contract_repair.SUPERSEDED:
                continue
            origin = ContractRecoveryService.repair_slot_origin(directory)
            if (
                origin is None
                or origin.get("stage") != GateStage(stage).value
                or origin.get("cycle") != cycle
                or origin.get("step_id") != step_id
                or origin.get("step_boundary_tree_sha") != anchor_tree
                or transaction.get("step_id") != step_id
                or transaction.get("cycle") != cycle
                or transaction.get("tree_sha") != anchor_tree
            ):
                continue
            return directory
        return None
    @staticmethod
    def _rewind_replan_boundary(
        store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan, *,
        anchor_commit: str, anchor_tree: str, step_ids: Sequence[str],
    ) -> None:
        """Rewind the run branch to one accepted boundary and drop what it replaced.

        The rewritten steps write their own commits and records: the commits
        the rewound branch abandoned are removed from the durable accepted
        chain and from the state, and the step records of the suffix are
        archived so an interrupted replan resumes by re-executing exactly the
        steps whose work the branch no longer holds.
        """

        worktree = ctx.info.worktree
        if symbolic_head(worktree) != ctx.branch_ref:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "the step replan boundary is not the run branch",
            )
        chain = accepted_chain_records(ctx.run_dir)
        replaced = {
            record["commit_sha"] for record in chain
            if isinstance(record, dict) and isinstance(record.get("commit_sha"), str)
            and record["commit_sha"] != anchor_commit
            and is_ancestor(worktree, anchor_commit, record["commit_sha"])
        }
        try:
            rewind_worktree(worktree, anchor_commit)
        except GitError as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                f"the step replan boundary could not be restored: {exc}",
            ) from exc
        if (
            current_head(worktree) != anchor_commit
            or candidate_tree_sha(worktree) != anchor_tree
            or status_has_unstaged_or_untracked(status_porcelain(worktree))
        ):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                "the step replan boundary does not match the responsible step tree",
            )
        kept = [
            record for record in chain
            if isinstance(record, dict) and record.get("commit_sha") not in replaced
        ]
        chain_path = ctx.run_dir / "accepted-chain.json"
        if kept:
            atomic_write_text(chain_path, json_text({"commits": kept}))
        elif chain_path.exists():
            chain_path.unlink()
        state = store.load()

        def pruned(records: Any) -> list[dict[str, Any]]:
            return [
                item for item in (records or [])
                if not (isinstance(item, dict) and item.get("commit_sha") in replaced)
            ]

        for step_id in step_ids:
            archive_attempt(cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step_id))
        store.update(
            status=RunStatus.IMPLEMENTING,
            accepted_steps=pruned(state.get("accepted_steps")),
            accepted_commits=pruned(state.get("accepted_commits")),
            expected_head_sha=anchor_commit, expected_tree_sha=anchor_tree,
        )
    @staticmethod
    def _replan_phase(cycle_plan: CyclePlan) -> ResumePhase:
        """The step-phase boundary of a cycle's own step execution."""

        return (
            ResumePhase.REVIEW_IMPLEMENTATION if cycle_plan.cycle.kind is CycleKind.REVIEW_IMPLEMENTATION
            else ResumePhase.IMPLEMENT_STEP
        )
    def _execute_replanned_suffix(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        indices: tuple[int, ...], *, anchor_tree: str,
    ) -> str:
        """Re-execute the responsible step and its descendants under the new authority."""

        steps = list(cycle_plan.plan.steps)
        worktree = ctx.info.worktree
        phase = self._replan_phase(cycle_plan)
        produced = anchor_tree
        for index in indices:
            # Each re-executed step gets its own durable boundary, so an
            # interrupted replan resumes at the exact step it stopped in.
            self.runtime.write_checkpoint(
                ctx.run_dir, phase, head=current_head(worktree),
                tree=candidate_tree_sha(worktree), cycle=cycle_plan.cycle.number,
                step_id=steps[index].id,
                correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
            )
            self.runtime.step_execution.execute_cycle_step(store, ctx, cycle_plan, index)
            produced = candidate_tree_sha(worktree)
        return produced
