"""Step execution: worker requests and their durable acceptance.

This module owns exactly one approved step: the worker attempts with their
provider-neutral fallbacks, the effective step authority (the approved
contract plus every validated contract repair), the contract-mismatch route
and the ``STEP_ACCEPTANCE`` boundary where a successful worker tree becomes a
commit.  It never plans a cycle, reviews a candidate or runs a check.
"""

from __future__ import annotations

import dataclasses, functools, hashlib, time
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    NoReturn,
    Sequence,
    TYPE_CHECKING,
)
from ..agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_RUNTIME_FAILED,
    AGENT_PROTOCOL_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentRunRequest,
    AgentScopeError,
)
from ..agent.diagnostics import write_token_diagnostics
from ..agent.protocol import (
    contract_mismatch_explanation,
    deferred_verify_dependency,
)
from ..prompt_contracts import (
    build_implementer_payload,
    write_prompt_diagnostics,
)
from ..approval import (
    ApprovalDecision,
    read_scope_approval,
)
from ..gitops import (
    GitError,
    WorktreeInfo,
    commit_message,
    commit_parents,
    restore_paths_from_tree,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_step_tree,
    current_head,
    index_tree_sha,
    is_ancestor,
    path_exists_in_tree,
    resolve_tree,
    rewind_worktree,
    status_porcelain,
    symbolic_head,
    stage_all,
    staged_diff,
)
from ..commit_gate import (
    COMMIT_GATE_FAILED,
    COMMIT_PARENT_MISMATCH,
    COMMIT_TREE_MISMATCH,
    CommitSafetyError,
    StepVerification,
    accepted_step_record,
    commit_safety_gate,
    step_verification,
)
from ..repository_topology import RepositoryTopology
from ..llm.chat import LLMError
from ..models import (
    CycleKind,
    ExecutionRole,
    GateStage,
    ModelProfile,
    ImplementationStep,
    RunStatus,
)
from ..planning.artifacts import (
    STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
    StepContractRepairArtifactError,
    read_approved_step_contract,
)
from ..planning.contract_repair import (
    StepContractRepairOutputInvalid,
    StepContractRepairPlanner,
    StepRepairIdentity,
)
from ..planning.protocol import (
    V2PlanParseError,
    read_set_paths,
)
from ..resume import (
    ResumePhase,
    read_checkpoint,
)
from ..usage import normalize_usage
from ..redaction import redact
from ..profiles import (
    build_llm_endpoint,
    profile_for_role,
)
from ..result import (
    ResultArtifactError,
    atomic_write_text,
)
from ..recovery_policy import (
    RecoveryStrategy,
    classify_failure,
)
from ..state import RunStateStore
from ..evidence import EvidenceBundle
from .shared import (
    GitOwnership,
    ScopeApprovalRequired,
    StepExecutionFailure,
    StepExecutionOutcome,
    _BOUNDED_NO_CHANGE_MISMATCH,
    _SYNTHETIC_NO_CHANGE_MISMATCH,
    _archive_attempt,
    bounded_v2_report,
    _git_ownership,
    is_object_id,
    _json_text,
    _new_status_lines,
    _ownership_violations,
    _paths_detail,
    _read_json_artifact,
    _record_failure_tree,
    _safe_candidate_tree,
    _safe_index_tree,
    _safe_status,
    _status_has_unstaged_or_untracked,
    bounded_parse_detail,
    chat_client,
)
from .revision import future_step_ownership
from .candidate import accepted_chain_records
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    RecoveryStepUnavailable,
    gate_dir,
    step_dir as cycle_step_dir,
)
from .recovery import (
    GateRecoveryStep,
    RecoveryAdmission,
    RecoveryAttempt,
    RecoveryCoordinator,
)
from . import contract_repair
from .check_repair import (
    replan_failure_evidence,
    replan_mismatch,
    replan_problem,
    replan_slot_origin,
)
from .contract_repair import ContractRepairIntegrityError
from .step_authority import (
    STEP_ACCEPTANCE_NAME,
    EffectiveStepAuthority,
    EffectiveStepExecution,
    StepAuthorityError,
    build_step_candidate,
    read_step_candidate,
    resolve_effective_step_authority,
    write_authority_diagnostic,
    write_step_candidate,
)
from .worker_recovery import TRANSIENT_WORKER_FAILURES


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


# The bounded red-gate evidence a contract replan planner was given, kept with
# its slot so an interrupted replan resumes with the same evidence instead of
# a second, weaker request.
_GATE_REPLAN_EVIDENCE = "gate_evidence.txt"
_MAX_GATE_REPLAN_EVIDENCE_BYTES = 64 * 1024


class ImplementationService:
    """One owner of the pipeline operations described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def execute_cycle_step(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan, index: int,
    ) -> None:
        """Execute, verify and accept exactly one approved step of a cycle."""

        step = cycle_plan.plan.steps[index]
        step_artifact_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        # A previous failed attempt of this same step keeps its artifacts.
        _archive_attempt(step_artifact_dir)
        contract = self.approved_step_contract(cycle_plan, step)
        store.update(
            status=RunStatus.IMPLEMENTING, current_step=step.id,
            steps=self.runtime.state_steps(ctx, cycle_plan, running=step.id),
        )
        checkpoint = read_checkpoint(ctx.run_dir)
        parent_sha = current_head(ctx.info.worktree)
        planner_profile = profile_for_role(
            self.runtime.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        reviewer_profile = profile_for_role(
            self.runtime.config, ctx.selection.final_reviewer.profile_id, ExecutionRole.REVIEWER
        )
        execution = self._execute_step_attempts(
            store=store, run_dir=ctx.run_dir, original_spec=ctx.spec,
            original_plan_identity=_json_text(
                asdict(checkpoint.plan_identity) if checkpoint and checkpoint.plan_identity else {}
            ),
            repo=ctx.repo, worktree=ctx.info.worktree, base_sha=parent_sha,
            branch_ref=ctx.branch_ref,
            ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
            expected_tree=(
                checkpoint.expected_tree_sha if checkpoint is not None
                else candidate_tree_sha(ctx.info.worktree)
            ),
            step=step, contract=contract,
            expected_plan_step_count=len(cycle_plan.plan.steps),
            profile_id=cycle_plan.step_profile_ids[step.id],
            fallback_profile_ids=(cycle_plan.step_fallback_profile_ids or {}).get(step.id, ()),
            artifact_dir=step_artifact_dir,
            forbidden_env_names=(planner_profile.api_key_env, reviewer_profile.api_key_env),
            future_ownership=future_step_ownership(cycle_plan.plan.steps, index),
        )
        self._accept_step_execution(
            store, ctx, cycle_plan, index, execution, parent_sha=parent_sha,
        )
        store.update(
            status=RunStatus.IMPLEMENTING, current_step=None,
            steps=self.runtime.state_steps(ctx, cycle_plan),
        )
        self.runtime.update_v2_usage(store, ctx.run_dir)
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
        approved_contract = self.approved_step_contract(cycle_plan, steps[first])
        authority = self.resolve_step_authority(
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
                atomic_write_text(directory / _GATE_REPLAN_EVIDENCE, failure_evidence)
            except (ContractRepairIntegrityError, OSError) as exc:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", str(exc), step_id=steps[first].id,
                ) from exc
        checkpoint = read_checkpoint(ctx.run_dir)
        self._contract_repair_transaction(
            store=store, recovery=self.runtime.recovery(store), repo=ctx.repo,
            worktree=worktree, run_dir=ctx.run_dir, artifact_dir=step_dir,
            directory=directory, cycle=number, number=repair_number,
            step=effective_step, current_contract=effective_contract, mismatch=mismatch,
            tree_before=anchor_tree, original_spec=ctx.spec,
            original_plan_identity=_json_text(
                asdict(checkpoint.plan_identity) if checkpoint and checkpoint.plan_identity else {}
            ),
            future_ownership=future_step_ownership(steps, first),
            max_repairs=getattr(self.runtime.run_options, "max_step_contract_repairs", 0),
            profile_id=cycle_plan.step_profile_ids[steps[first].id],
            resumed=resumed, expected_plan_step_count=len(steps),
            failure_evidence=failure_evidence, require_new_contract=True,
        )
        repaired = self._repaired_authority(
            self.resolve_step_authority(
                step_dir, steps[first], approved_contract, expected_tree=anchor_tree,
                expected_plan_step_count=len(steps),
            ),
            repair_number,
        )
        self.runtime.trace_emit(
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
            record = _read_json_artifact(
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
            record = _read_json_artifact(
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
    def _replan_slot_origin(directory: Path) -> Mapping[str, Any] | None:
        """The red-gate replan identity of one contract repair slot, if it is one."""

        archived = _read_json_artifact(directory / contract_repair.MISMATCH_NAME, 64 * 1024)
        if not isinstance(archived, dict) or not isinstance(archived.get("mismatch"), str):
            return None
        return replan_slot_origin(archived["mismatch"])

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
            origin = ImplementationService._replan_slot_origin(directory)
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
            or _status_has_unstaged_or_untracked(status_porcelain(worktree))
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
            atomic_write_text(chain_path, _json_text({"commits": kept}))
        elif chain_path.exists():
            chain_path.unlink()
        state = store.load()

        def pruned(records: Any) -> list[dict[str, Any]]:
            return [
                item for item in (records or [])
                if not (isinstance(item, dict) and item.get("commit_sha") in replaced)
            ]

        for step_id in step_ids:
            _archive_attempt(cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step_id))
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
            self.execute_cycle_step(store, ctx, cycle_plan, index)
            produced = candidate_tree_sha(worktree)
        return produced

    def approved_step_contract(self, cycle_plan: CyclePlan, step: ImplementationStep) -> str:
        """The hash-bound approved contract: immutable evidence of the step."""

        try:
            return read_approved_step_contract(cycle_plan.contracts_dir, cycle_plan.bundle, step.id)
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise PipelineFailure("PLAN_APPROVAL_INVALID", str(exc), step_id=step.id) from exc
    def _accept_step_execution(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        index: int, execution: EffectiveStepExecution, *, parent_sha: str,
    ) -> None:
        """Cross the durable worker-success -> commit boundary of one step.

        A changed tree is first frozen as ``step_candidate.json`` and the
        checkpoint moves to ``STEP_ACCEPTANCE``; only then does the commit
        gate run, with the very authority the worker executed under.
        """

        authority, outcome = execution.authority, execution.outcome
        step_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, authority.step_id)
        future = tuple(item.id for item in cycle_plan.plan.steps[index + 1:])
        accept = functools.partial(
            self._accept_v2_step_tree,
            store=store, run_dir=ctx.run_dir, info=ctx.info, authority=authority,
            outcome=outcome, parent_sha=parent_sha, future_step_ids=future,
            run_id=ctx.run_id, step_dir=step_dir,
        )
        try:
            if outcome.no_change or outcome.tree_after == outcome.tree_before:
                # Nothing to commit: no candidate crosses a commit boundary.
                accept()
                return
            verification = self._step_verification(authority, outcome, future)
            self._persist_step_candidate(
                ctx, cycle_plan, step_dir, authority, outcome, verification,
                parent_sha=parent_sha, source="worker_success",
            )
            self.runtime.write_checkpoint(
                ctx.run_dir, ResumePhase.STEP_ACCEPTANCE,
                head=parent_sha, tree=outcome.tree_after, cycle=cycle_plan.cycle.number,
                step_id=authority.step_id,
                correction_bundle_sha256=cycle_plan.correction_bundle_sha256,
            )
            accept(verification=verification)
        except (CommitSafetyError, GitError) as exc:
            raise self._step_acceptance_failure(step_dir, authority, exc) from exc
    def resume_step_acceptance(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan, index: int,
    ) -> None:
        """Accept the durable worker candidate of a ``STEP_ACCEPTANCE`` checkpoint.

        No worker, planner or reviewer is called.  Every proof is re-derived:
        the checkpoint, the self-hashed candidate, the step record and report,
        the effective authority (from its artifacts, never from state.json),
        and the exact Git boundary.  Then the normal commit gate runs.
        """

        step = cycle_plan.plan.steps[index]
        step_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        worktree = ctx.info.worktree

        def refuse(message: str) -> NoReturn:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", f"step acceptance: {message}", step_id=step.id,
            )

        checkpoint = read_checkpoint(ctx.run_dir)
        if (
            checkpoint is None or checkpoint.phase is not ResumePhase.STEP_ACCEPTANCE
            or checkpoint.step_id != step.id
            or checkpoint.review_cycle != cycle_plan.cycle.number
        ):
            refuse("the checkpoint does not name this step")
        try:
            candidate = read_step_candidate(step_dir)
        except StepAuthorityError as exc:
            refuse(str(exc))
        if candidate is None:
            refuse("the durable step candidate is missing")
        parent_sha, tree_before, tree_after = (
            candidate["parent_head_sha"], candidate["tree_before"], candidate["tree_after"],
        )
        changed = tuple(candidate["changed_paths"])
        if (
            candidate["step_id"] != step.id
            or candidate.get("cycle") != cycle_plan.cycle.number
            or candidate.get("run_id") != ctx.run_id
            or parent_sha != checkpoint.expected_head_sha
            or tree_after != checkpoint.expected_tree_sha
        ):
            refuse("the step candidate is not bound to its checkpoint")
        authority = self.resolve_step_authority(
            step_dir, step, self.approved_step_contract(cycle_plan, step),
            expected_tree=tree_before, expected_plan_step_count=len(cycle_plan.plan.steps),
        )
        if (
            authority.authority_sha256 != candidate["effective_authority_sha256"]
            or authority.effective_contract_sha256 != candidate["effective_contract_sha256"]
        ):
            refuse("the effective step authority changed since the worker succeeded")
        if any(path not in authority.mutable_scope for path in changed):
            refuse("the candidate changed paths outside its effective authority")
        try:
            verification = StepVerification.from_payload(candidate.get("verification"))
        except ValueError as exc:
            refuse(str(exc))
        outcome_refs = candidate["outcome"]
        record_path, final_path = step_dir / "step.json", step_dir / "agent.final.md"
        try:
            record_bytes = record_path.read_bytes()
            final_bytes = final_path.read_bytes() if final_path.is_file() else None
        except OSError:
            refuse("the step record is unreadable")
        final_sha = hashlib.sha256(final_bytes).hexdigest() if final_bytes is not None else None
        if final_sha != outcome_refs.get("final_report_sha256"):
            refuse("the worker report changed")
        record = _read_json_artifact(record_path, 128 * 1024)
        if (
            not isinstance(record, dict) or record.get("id") != step.id
            or record.get("tree_before") != tree_before or record.get("tree_after") != tree_after
            or sorted(record.get("changed_paths") or []) != sorted(changed)
        ):
            refuse("the step record does not match the candidate")
        future = tuple(item.id for item in cycle_plan.plan.steps[index + 1:])
        self.runtime.trace_emit(
            "recovery.resumed", phase="implementation", cycle=cycle_plan.cycle.number,
            step_id=step.id,
            data={
                "operation": "step_acceptance", "tree_after": tree_after,
                "effective_authority_sha256": authority.authority_sha256,
                "source": candidate.get("source"),
            },
        )
        try:
            head = current_head(worktree)
            if symbolic_head(worktree) != ctx.branch_ref:
                refuse("the worktree HEAD is not the run branch")
            if head != parent_sha:
                self._recover_committed_step(
                    store, ctx, step_dir, candidate, authority, verification,
                    head=head, future_step_ids=future,
                )
                store.update(
                    status=RunStatus.IMPLEMENTING, current_step=None,
                    steps=self.runtime.state_steps(ctx, cycle_plan),
                )
                return
            if hashlib.sha256(record_bytes).hexdigest() != outcome_refs.get("step_record_sha256"):
                refuse("the step record changed")
            if (
                resolve_tree(worktree, parent_sha) != tree_before
                or index_tree_sha(worktree) != tree_after
                or candidate_tree_sha(worktree) != tree_after
                or _status_has_unstaged_or_untracked(status_porcelain(worktree))
                or tuple(sorted(changed_paths_between_trees(ctx.repo, tree_before, tree_after))) != tuple(sorted(changed))
            ):
                refuse("the worktree is not exactly the durable worker candidate")
        except GitError as exc:
            refuse(f"Git state is unreadable: {exc}")
        outcome = StepExecutionOutcome(
            step_id=step.id, profile_id=str(candidate.get("profile_id") or ""),
            tree_before=tree_before, tree_after=tree_after, changed_paths=changed,
            usage=normalize_usage(record.get("usage")),
            final_report=(final_bytes or b"").decode("utf-8", errors="replace"),
            deferred_verify=str(record.get("deferred_verify") or ""),
            mismatch_retry_count=int(record.get("mismatch_retry_count") or 0),
        )
        store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
        try:
            self._accept_v2_step_tree(
                store=store, run_dir=ctx.run_dir, info=ctx.info, authority=authority,
                outcome=outcome, parent_sha=parent_sha, future_step_ids=future,
                run_id=ctx.run_id, step_dir=step_dir, verification=verification,
            )
        except (CommitSafetyError, GitError) as exc:
            raise self._step_acceptance_failure(step_dir, authority, exc) from exc
        store.update(
            status=RunStatus.IMPLEMENTING, current_step=None,
            steps=self.runtime.state_steps(ctx, cycle_plan),
        )
        self.runtime.update_v2_usage(store, ctx.run_dir)
    def _recover_committed_step(
        self, store: RunStateStore, ctx: PipelineV2Context, step_dir: Path,
        candidate: Mapping[str, Any], authority: EffectiveStepAuthority,
        verification: StepVerification, *, head: str, future_step_ids: Sequence[str],
    ) -> None:
        """A crash landed after the step commit: prove it, then only record it."""

        worktree = ctx.info.worktree
        parent_sha, tree_after = candidate["parent_head_sha"], candidate["tree_after"]
        message = commit_message(worktree, head)
        subject = message.splitlines()[0] if message else ""
        if (
            commit_parents(worktree, head) != (parent_sha,)
            or resolve_tree(worktree, head) != tree_after
            or not subject.startswith(f"metaharness({authority.step_id}):")
            or f"MetaHarness-Run: {ctx.run_id}" not in message
            or index_tree_sha(worktree) != tree_after
            or candidate_tree_sha(worktree) != tree_after
            or _status_has_unstaged_or_untracked(status_porcelain(worktree))
        ):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                "step acceptance: HEAD moved to a commit that is not this step candidate",
                step_id=authority.step_id,
            )
        record = accepted_step_record(
            step_id=authority.step_id,
            verification_status=verification.status,
            parent_sha=parent_sha,
            commit_sha=head,
            tree_before=candidate["tree_before"],
            tree_after=tree_after,
            changed_paths=candidate["changed_paths"],
            deferred=verification.deferred,
            authority=authority.summary(),
        )
        diff_path = step_dir / "diff.patch"
        self._finalize_accepted_step(
            store, ctx.run_dir, step_dir, record, future_step_ids=future_step_ids,
            authority=authority, diff_path=diff_path if diff_path.is_file() else None,
        )
    def _step_verification(
        self, authority: EffectiveStepAuthority, outcome: StepExecutionOutcome,
        future_step_ids: Sequence[str],
    ) -> StepVerification:
        try:
            return step_verification(
                outcome.final_report, step_id=authority.step_id,
                future_step_ids=future_step_ids,
                reported_status=getattr(outcome, "verification_status", None),
                deferred_requested=bool(
                    getattr(outcome, "deferred_verify", "")
                    or getattr(outcome, "status", "") == "DEFERRED_CONTRACT_MISMATCH"
                ),
            )
        except CommitSafetyError:
            self.runtime.trace_emit(
                "step.verification.completed", phase="implementation",
                cycle=self.runtime.trace_cycle, step_id=authority.step_id,
                data={
                    "status": "failed", "tree_before": outcome.tree_before,
                    "tree_after": outcome.tree_after, "deferred": False,
                },
            )
            raise
    def _persist_step_candidate(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan, step_dir: Path,
        authority: EffectiveStepAuthority, outcome: StepExecutionOutcome,
        verification: StepVerification, *, parent_sha: str, source: str,
    ) -> dict[str, Any]:
        """Freeze a successful worker candidate before its commit boundary."""

        record_path = step_dir / "step.json"
        final_path = step_dir / "agent.final.md"
        try:
            record_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
            final_sha = (
                hashlib.sha256(final_path.read_bytes()).hexdigest()
                if final_path.is_file() else None
            )
        except OSError as exc:
            raise PipelineFailure(
                "DURABLE_ARTIFACT_CORRUPTED", f"step record is unreadable: {exc}",
                step_id=authority.step_id,
            ) from exc
        payload = build_step_candidate(
            run_id=ctx.run_id, cycle=cycle_plan.cycle.number, step_id=authority.step_id,
            parent_head_sha=parent_sha, tree_before=outcome.tree_before,
            tree_after=outcome.tree_after, changed_paths=outcome.changed_paths,
            profile_id=outcome.profile_id, authority=authority,
            verification=verification.payload(), step_record_sha256=record_sha,
            final_report_sha256=final_sha, source=source,
        )
        try:
            write_step_candidate(step_dir, payload)
        except StepAuthorityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=authority.step_id) from exc
        write_authority_diagnostic(step_dir, authority)
        self.runtime.trace_emit(
            "step.candidate.persisted", phase="implementation",
            cycle=cycle_plan.cycle.number, step_id=authority.step_id,
            data={
                "tree_after": outcome.tree_after, "source": source,
                "effective_authority_sha256": authority.authority_sha256,
                "effective_contract_sha256": authority.effective_contract_sha256,
                "authority_source": authority.authority_source,
                "repair_slot": authority.repair_slot,
            },
        )
        return payload
    def _step_acceptance_failure(
        self, step_dir: Path, authority: EffectiveStepAuthority, exc: Exception,
    ) -> PipelineFailure:
        """Record which authority refused the commit; the gate stays strict."""

        code = getattr(exc, "code", None) if isinstance(exc, CommitSafetyError) else None
        code = code or COMMIT_GATE_FAILED
        message = bounded_parse_detail(exc)
        try:
            atomic_write_text(step_dir / STEP_ACCEPTANCE_NAME, _json_text({
                "schema_version": 1, "status": "refused", "step_id": authority.step_id,
                "code": code, "detail": message,
                "paths": list(getattr(exc, "paths", ()))[:20],
                "commit_gate_authority_sha256": authority.authority_sha256,
                "effective_contract_sha256": authority.effective_contract_sha256,
                "effective_mutable_paths": list(authority.mutable_scope),
            }))
        except OSError:
            pass
        return PipelineFailure(
            "COMMIT_GATE_FAILED",
            f"{code}: {message} (authority {authority.authority_sha256[:16]})",
            step_id=authority.step_id,
        )
    @staticmethod
    def _repair_failure_evidence(directory: Path) -> str:
        """The durable failure evidence of a slot a red gate opened, if any."""

        try:
            data = (directory / _GATE_REPLAN_EVIDENCE).read_bytes()[:_MAX_GATE_REPLAN_EVIDENCE_BYTES]
        except OSError:
            return "NONE"
        return data.decode("utf-8", errors="replace") or "NONE"

    def _step_profile(self, profile_id: str) -> tuple[ModelProfile, ExecutionRole]:
        """The approved implementation profile of one plan step."""

        return profile_for_role(self.runtime.config, profile_id, ExecutionRole.IMPLEMENTER), ExecutionRole.IMPLEMENTER
    def _execute_step_attempts(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        original_spec: str,
        original_plan_identity: str,
        repo: Path,
        worktree: Path,
        base_sha: str,
        branch_ref: str,
        ownership_before: GitOwnership,
        expected_tree: str,
        step: ImplementationStep,
        contract: str,
        expected_plan_step_count: int | None = None,
        profile_id: str,
        artifact_dir: Path,
        fallback_profile_ids: Sequence[str] = (),
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
    ) -> EffectiveStepExecution:
        """Execute a step, repairing semantic contract mismatches in-place.

        A mismatch is a recoverable transaction: its attempt artifacts are
        archived, all in-scope edits are restored to the pre-attempt tree, a
        bounded planner repair is validated, and the worker receives the
        effective contract.  The approved plan and original contract remain
        immutable evidence throughout.  The result carries the exact
        :class:`EffectiveStepAuthority` the successful worker executed under,
        so acceptance never re-derives it from the approved step.
        """

        def resolve() -> EffectiveStepAuthority:
            return self.resolve_step_authority(
                artifact_dir, step, contract, expected_tree=expected_tree,
                expected_plan_step_count=expected_plan_step_count,
            )

        authority = resolve()
        effective_step, effective_contract = authority.effective_step, authority.effective_contract
        # Semantic budget excludes superseded slots and transport retries.
        try:
            repair_count = contract_repair.semantic_repair_count(artifact_dir)
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
        max_repairs = getattr(self.runtime.run_options, "max_step_contract_repairs", 0)
        recovery = self.runtime.recovery(store)
        cycle_number = self.runtime.trace_cycle
        active_profile_id = profile_id
        fallback_ids = tuple(fallback_profile_ids)[
            :self.runtime.run_options.recovery.max_executor_fallbacks
        ]
        fallback_index = 0
        retry_key = recovery.budget_key("agent-step", f"{cycle_number:03d}", step.id)
        pending_transient: RecoveryAdmission | None = None
        repair_context = {
            "store": store, "recovery": recovery, "repo": repo, "worktree": worktree,
            "run_dir": run_dir, "artifact_dir": artifact_dir, "cycle": cycle_number,
            "original_spec": original_spec,
            "original_plan_identity": original_plan_identity,
            "future_ownership": future_ownership, "max_repairs": max_repairs,
            "expected_plan_step_count": expected_plan_step_count,
        }
        try:
            pending_repair = contract_repair.find_pending(
                artifact_dir, cycle=cycle_number, step_id=step.id,
                current_contract=effective_contract,
            )
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
        if pending_repair is not None:
            # Resume the incomplete repair; the worker that produced its
            # mismatch is never replayed.
            drift = self._pre_step_boundary_drift(
                repo, worktree, ownership_before,
                branch_ref=branch_ref, base_sha=base_sha,
                tree_before=pending_repair.tree_sha,
            )
            if drift:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE",
                    f"pending contract repair {pending_repair.number:02d}: {drift}",
                    step_id=step.id,
                )
            self._contract_repair_transaction(
                **repair_context, directory=pending_repair.directory,
                number=pending_repair.number, step=effective_step,
                current_contract=effective_contract, mismatch=pending_repair.mismatch,
                tree_before=pending_repair.tree_sha, profile_id=active_profile_id,
                resumed=True,
                failure_evidence=self._repair_failure_evidence(pending_repair.directory),
                # A slot a red gate opened owes a contract different from the
                # one that gate failed under, on a resume as well: an
                # unchanged answer stays an output defect, never a replay.
                require_new_contract=self._replan_slot_origin(pending_repair.directory) is not None,
            )
            authority = self._repaired_authority(resolve(), pending_repair.number)
            effective_step, effective_contract = authority.effective_step, authority.effective_contract
        while True:
            common = {
                "repo": repo, "worktree": worktree, "base_sha": base_sha,
                "branch_ref": branch_ref, "ownership_before": ownership_before,
                "expected_tree": expected_tree, "step": effective_step,
                "contract": effective_contract, "profile_id": active_profile_id,
                "artifact_dir": artifact_dir,
                "forbidden_env_names": forbidden_env_names,
                "future_ownership": future_ownership,
                "original_spec": original_spec,
            }
            try:
                write_authority_diagnostic(artifact_dir, authority)
                outcome = self._run_step_attempt(
                    **common, initial_mismatch=None,
                    mismatch_retry_count=repair_count,
                )
                if pending_transient is not None:
                    recovery.complete(
                        pending_transient, recovered=True, tree_after=outcome.tree_after,
                    )
                return EffectiveStepExecution(outcome, authority)
            except StepExecutionFailure as failure:
                atomic_write_text(artifact_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "reason": failure.reason[:120],
                    "step_id": failure.step_id,
                    "profile_id": failure.profile_id,
                    "tree_before": failure.tree_before,
                    "tree_after": failure.tree_after,
                    "status_before": list(failure.status_before or ()),
                }))
                if pending_transient is not None:
                    recovery.complete(
                        pending_transient, recovered=False,
                        tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                    )
                    pending_transient = None
                retry = self.runtime.worker_recovery(store).admit_step_retry(
                    failure=failure, retry_key=retry_key, repo=repo, worktree=worktree,
                    branch_ref=branch_ref, base_sha=base_sha,
                    ownership_before=ownership_before, step=effective_step,
                    artifact_dir=artifact_dir, cycle=cycle_number,
                )
                if retry is not None:
                    pending_transient = retry
                    continue
                if failure.reason == AGENT_AUTH_FAILURE:
                    failure.reason = "EXTERNAL_AUTH_REQUIRED"
                    failure.detail = "executor credentials or external authorization are required"
                    failure.step_dir = artifact_dir
                    raise
                if failure.reason == "AGENT_NO_CHANGE":
                    return EffectiveStepExecution(self._finish_no_change_step(
                        failure, artifact_dir, worktree=worktree, expected_head=base_sha,
                    ), authority)
                if (
                    failure.reason in TRANSIENT_WORKER_FAILURES
                    and recovery.used(retry_key) >= self.runtime.run_options.recovery.max_transient_attempts
                    and fallback_index < len(fallback_ids)
                ):
                    fallback_index += 1
                    active_profile_id = fallback_ids[fallback_index - 1]
                    retry_key = recovery.budget_key(
                        "agent-step", f"{cycle_number:03d}", step.id,
                        "fallback", active_profile_id,
                    )
                    recovery.fallback_selected(
                        reason=failure.reason, index=fallback_index,
                        available=len(fallback_ids), phase="implementation",
                        cycle=cycle_number, step_id=step.id,
                        tree_before=failure.tree_before,
                        tree_after=_safe_candidate_tree(worktree),
                    )
                    _archive_attempt(artifact_dir)
                    atomic_write_text(artifact_dir / "executor.json", _json_text({
                        "profile_id": active_profile_id,
                        "role": "implementer",
                        "selection_source": "frozen execution fallback authority",
                    }))
                    store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
                    continue
                if failure.reason != "AGENT_CONTRACT_MISMATCH":
                    failure.step_dir = artifact_dir
                    raise
                if failure.tree_before is None or failure.mismatch is None:
                    if failure.tree_before is None:
                        failure.reason = "RESUME_REQUIRES_OPERATOR"
                    failure.step_dir = artifact_dir
                    raise
                allowed = set(authority.mutable_scope)
                changed: list[str] = []
                if failure.tree_after is not None:
                    try:
                        changed = changed_paths_between_trees(repo, failure.tree_before, failure.tree_after)
                    except GitError:
                        changed = []
                    unexpected = [path for path in changed if path not in allowed]
                    if unexpected:
                        failure.reason = AGENT_SCOPE_VIOLATION
                        failure.detail = "worker changed paths outside scope: " + _paths_detail(unexpected)
                        failure.step_dir = artifact_dir
                        raise
                drift = self._pre_step_boundary_drift(
                    repo, worktree, ownership_before,
                    branch_ref=branch_ref, base_sha=base_sha,
                    tree_before=failure.tree_before,
                )
                if drift and not self._restore_failed_step_attempt(
                    worktree, failure.tree_before, allowed,
                ):
                    failure.reason = "RESUME_REQUIRES_OPERATOR"
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; " + drift
                    failure.step_dir = artifact_dir
                    raise
                if not self._restore_failed_step_attempt(worktree, failure.tree_before, allowed):
                    failure.reason = "RESUME_REQUIRES_OPERATOR"
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; failed to restore attempt tree"
                    failure.step_dir = artifact_dir
                    raise
                recovery_decision = classify_failure(
                    "AGENT_CONTRACT_MISMATCH",
                    tree_changed_in_scope=bool(changed),
                    clean_contract_mismatch=True,
                    rollback_succeeded=True,
                )
                recovery_attempt = repair_count + 1
                recovery.trace(
                    "recovery.classified", reason="AGENT_CONTRACT_MISMATCH",
                    decision=recovery_decision, attempt=recovery_attempt,
                    tree_before=failure.tree_before, tree_after=failure.tree_before,
                    budget_remaining=max(0, max_repairs - repair_count),
                    phase="implementation", cycle=cycle_number,
                    step_id=step.id,
                )
                if recovery_decision.strategy is not RecoveryStrategy.REPAIR_TARGETED:
                    failure.step_dir = artifact_dir
                    raise
                _archive_attempt(artifact_dir)
                if repair_count >= max_repairs:
                    if failure.mismatch in {
                        _SYNTHETIC_NO_CHANGE_MISMATCH,
                        _BOUNDED_NO_CHANGE_MISMATCH,
                    }:
                        return EffectiveStepExecution(self._finish_no_change_step(
                            failure, artifact_dir, worktree=worktree, expected_head=base_sha,
                        ), authority)
                    exhausted = classify_failure(
                        "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True,
                        budget_exhausted=True,
                    )
                    recovery.trace(
                        "recovery.exhausted", reason="AGENT_CONTRACT_MISMATCH",
                        decision=exhausted, attempt=recovery_attempt,
                        tree_before=failure.tree_before, tree_after=failure.tree_before,
                        budget_remaining=0, phase="implementation",
                        cycle=cycle_number, step_id=step.id,
                    )
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; contract repair budget exhausted"
                    failure.tree_after = failure.tree_before
                    failure.index_tree_after = failure.tree_before
                    failure.step_dir = artifact_dir
                    raise
                number = contract_repair.next_repair_number(artifact_dir)
                repair_dir = artifact_dir / "contract_repairs" / f"{number:02d}"
                bounded_mismatch = bounded_v2_report(failure.mismatch)
                try:
                    contract_repair.begin(
                        repair_dir, number=number, cycle=cycle_number, step_id=step.id,
                        current_contract=effective_contract, mismatch=bounded_mismatch,
                        tree_sha=failure.tree_before,
                        output_correction_limit=self.runtime.run_options.recovery.max_contract_repair_output_corrections,
                    )
                except (ContractRepairIntegrityError, OSError) as exc:
                    raise PipelineFailure(
                        "RESUME_INTEGRITY_FAILURE", str(exc), step_id=step.id,
                    ) from exc
                repair_count += 1
                recovery.trace(
                    "recovery.started", reason="AGENT_CONTRACT_MISMATCH",
                    decision=recovery_decision, attempt=recovery_attempt,
                    tree_before=failure.tree_before, tree_after=failure.tree_before,
                    budget_remaining=max(0, max_repairs - repair_count),
                    phase="implementation", cycle=cycle_number,
                    step_id=step.id,
                )
                self._contract_repair_transaction(
                    **repair_context, directory=repair_dir, number=number,
                    step=effective_step, current_contract=effective_contract,
                    mismatch=bounded_mismatch, tree_before=failure.tree_before,
                    profile_id=active_profile_id, resumed=False, usage=failure.usage,
                )
                authority = self._repaired_authority(resolve(), number)
                effective_step, effective_contract = authority.effective_step, authority.effective_contract
                # The next attempt starts at the exact restored tree and uses
                # no blind retry addendum.
    def _finish_no_change_step(
        self, failure: StepExecutionFailure, artifact_dir: Path, *,
        worktree: Path, expected_head: str,
    ) -> StepExecutionOutcome:
        """Record a clean empty delta after the bounded worker/repair path."""

        before = failure.tree_before
        after = _safe_candidate_tree(worktree)
        index_after = _safe_index_tree(worktree)
        status_after = _safe_status(worktree)
        if (
            before is None or after != before or index_after != before
            or current_head(worktree) != expected_head
            or _status_has_unstaged_or_untracked(status_after or ())
            or (failure.status_before is not None and status_after != failure.status_before)
        ):
            failure.step_dir = artifact_dir
            raise failure
        usage = normalize_usage(failure.usage)
        report = bounded_v2_report(failure.detail or "candidate delta is empty")
        atomic_write_text(artifact_dir / "step.json", _json_text({
            "id": failure.step_id,
            "status": "COMPLETED",
            "no_change": True,
            "reason": "bounded execution left the exact candidate tree unchanged",
            "profile_id": failure.profile_id,
            "tree_before": before,
            "tree_after": after,
            "changed_paths": [],
            "mismatch_retry_count": failure.mismatch_retry_count,
            **({"initial_mismatch": failure.initial_mismatch}
               if failure.initial_mismatch else {}),
            "usage": usage,
        }))
        self.runtime.trace_emit(
            "step.completed_no_change", phase="implementation",
            cycle=self.runtime.trace_cycle, step_id=failure.step_id,
            data={"tree_sha": after, "mismatch_retry_count": failure.mismatch_retry_count},
        )
        return StepExecutionOutcome(
            step_id=failure.step_id,
            profile_id=failure.profile_id or "",
            tree_before=before,
            tree_after=after,
            changed_paths=(),
            usage=usage,
            final_report=report,
            mismatch_retry_count=failure.mismatch_retry_count,
            no_change=True,
        )
    def resolve_step_authority(
        self, artifact_dir: Path, step: ImplementationStep, contract: str, *,
        expected_tree: str | None, expected_plan_step_count: int | None,
    ) -> EffectiveStepAuthority:
        """The single effective authority of *step*; corruption fails closed."""

        try:
            return resolve_effective_step_authority(
                artifact_dir, step, contract,
                max_read_paths_per_step=self.runtime.config.planning.max_read_paths_per_step,
                expected_plan_step_count=expected_plan_step_count,
                expected_tree_sha=expected_tree,
                authorize_added=self._authorize_repair_additions,
            )
        except StepAuthorityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
    @staticmethod
    def _repaired_authority(authority: EffectiveStepAuthority, number: int) -> EffectiveStepAuthority:
        if authority.repair_slot != number:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                f"validated contract repair {number:02d} is not the effective step authority",
                step_id=authority.step_id,
            )
        return authority
    def _authorize_repair_additions(self, repair_dir: Path, added: list[str]) -> None:
        """The frozen scope policy of one validated repair's added paths."""

        policy = self.runtime.repair_scope
        if policy.policy == "deny-expansion":
            raise PipelineFailure("REPAIR_SCOPE_EXPANSION")
        if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
            delta_path = repair_dir / "scope_delta.json"
            delta = _read_json_artifact(delta_path, 64 * 1024)
            if not isinstance(delta, dict) or delta.get("added_paths") != sorted(added):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "step contract repair scope delta is malformed")
            approval = read_scope_approval(
                repair_dir,
                expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
            )
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                raise ScopeApprovalRequired()
    @staticmethod
    def _restore_failed_step_attempt(
        worktree: Path, tree_before: str, allowed: set[str],
    ) -> bool:
        try:
            restore_paths_from_tree(worktree, tree_before, sorted(allowed))
            stage_all(worktree)
            return (
                candidate_tree_sha(worktree) == tree_before
                and index_tree_sha(worktree) == tree_before
                and not _status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except (GitError, OSError):
            return False
    def _contract_repair_transaction(
        self, *, store: RunStateStore, recovery: RecoveryCoordinator,
        repo: Path, worktree: Path, run_dir: Path, artifact_dir: Path,
        directory: Path, cycle: int, number: int, step: ImplementationStep,
        current_contract: str, mismatch: str, tree_before: str,
        original_spec: str, original_plan_identity: str,
        future_ownership: Mapping[str, tuple[str, ...]] | None,
        max_repairs: int, profile_id: str, resumed: bool,
        expected_plan_step_count: int | None,
        usage: dict[str, int] | None = None,
        failure_evidence: str = "NONE",
        require_new_contract: bool = False,
    ) -> tuple[ImplementationStep, str]:
        """Drive one durable semantic repair slot to ``completed``.

        A planner transport failure parks the slot in ``waiting_external``
        and propagates; the resume re-enters this same slot.  A durable but
        invalid planner answer is corrected inside the same slot within
        ``recovery.max_contract_repair_output_corrections``; it is never a
        worker contract mismatch.  Only a validated repair consumes the
        semantic ``contract_repairs`` budget.
        """

        max_corrections = self.runtime.run_options.recovery.max_contract_repair_output_corrections
        try:
            semantic_attempt = contract_repair.semantic_repair_count(artifact_dir)
            transaction = contract_repair.read_transaction(directory) or {}
            attempt = contract_repair.output_attempt(transaction)
            if contract_repair.planner_response_durable(directory, attempt):
                transaction = contract_repair.ensure(directory, contract_repair.PLANNER_RESPONSE_DURABLE)
            if "output_correction_limit" not in transaction:
                # A slot opened before output corrections existed.
                transaction = contract_repair.advance(
                    directory, transaction["status"], output_correction_limit=max_corrections,
                )
            if resumed and transaction.get("status") == contract_repair.OUTPUT_CORRECTION_EXHAUSTED:
                # The operator retried the planner: one more bounded round of
                # output corrections, still inside this semantic slot.
                transaction = contract_repair.advance(
                    directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                    output_attempt=attempt + 1, output_correction_attempt=attempt,
                    output_correction_limit=int(transaction["output_correction_limit"]) + max(1, max_corrections),
                    operator_output_retries=int(transaction.get("operator_output_retries") or 0) + 1,
                    planner_transport_attempt=0,
                )
            if contract_repair.is_awaiting_planner(transaction):
                transaction = contract_repair.advance(
                    directory, contract_repair.awaiting_status(transaction),
                    planner_transport_attempt=int(transaction.get("planner_transport_attempt") or 0) + 1,
                )
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc

        def progress_of(current: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "step_id": step.id, "attempt": semantic_attempt,
                "pending_operation": "contract_repair",
                "contract_repair_number": number,
                "repair_id": current.get("repair_id"),
                "planner_transport_attempt": current.get("planner_transport_attempt"),
                "output_attempt": contract_repair.output_attempt(dict(current)),
                "output_correction_attempt": current.get("output_correction_attempt", 0),
                "output_correction_limit": current.get("output_correction_limit"),
            }

        def publish(status: str, current: Mapping[str, Any]) -> dict[str, Any]:
            progress = progress_of(current)
            store.update(
                status=RunStatus.CONTRACT_REPAIRING, current_step=step.id,
                contract_repair={"status": status, **progress},
            )
            return progress

        progress = publish(
            "correcting_output" if contract_repair.output_attempt(transaction) > 1 else "running",
            transaction,
        )
        if resumed:
            self.runtime.trace_emit(
                "recovery.resumed", phase="implementation", cycle=cycle, step_id=step.id,
                data={"operation": "contract_repair", "transaction_status": transaction.get("status"), **progress},
            )

        def on_request(output_attempt: int) -> None:
            current = contract_repair.read_transaction(directory) or {}
            if contract_repair.output_attempt(current) >= output_attempt:
                return
            current = contract_repair.advance(
                directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                output_attempt=output_attempt,
                output_correction_attempt=output_attempt - 1,
                planner_transport_attempt=1,
            )
            data = publish("correcting_output", current)
            self.runtime.trace_emit(
                "contract_repair.output_correction.started", phase="implementation",
                cycle=cycle, step_id=step.id, data=data,
            )

        def on_response_durable(_output_attempt: int) -> None:
            contract_repair.ensure(directory, contract_repair.PLANNER_RESPONSE_DURABLE)

        def on_output_invalid(output_attempt: int, detail: str) -> None:
            error = {
                "code": STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
                "detail": detail[:500], "output_attempt": output_attempt,
            }
            current = contract_repair.ensure(
                directory, contract_repair.PLANNER_OUTPUT_INVALID, last_output_error=error,
            )
            data = publish("output_invalid", current)
            self.runtime.trace_emit(
                "contract_repair.output_invalid", phase="implementation",
                cycle=cycle, step_id=step.id, data={**data, "error": error},
            )

        while True:
            try:
                repaired = self._repair_step_contract(
                    repo=repo, worktree=worktree, run_dir=run_dir,
                    artifact_dir=directory, original_spec=original_spec,
                    original_plan_identity=original_plan_identity,
                    original_step=step, current_contract=current_contract,
                    expected_plan_step_count=expected_plan_step_count,
                    mismatch=mismatch, tree_before=tree_before,
                    future_ownership=future_ownership,
                    resume_request=contract_repair.durable_request_matches(directory, tree_before) is True,
                    max_output_corrections=int(
                        (contract_repair.read_transaction(directory) or {}).get(
                            "output_correction_limit", max_corrections,
                        )
                    ),
                    on_request=on_request, on_response_durable=on_response_durable,
                    on_output_invalid=on_output_invalid,
                    failure_evidence=failure_evidence,
                    require_new_contract=require_new_contract,
                )
                effective_contract = (directory / "contract.md").read_text(encoding="utf-8")
                break
            except LLMError as exc:
                detail = redact(str(exc), self.runtime.secrets)[:500]
                try:
                    current = contract_repair.read_transaction(directory) or {}
                    if contract_repair.is_awaiting_planner(current):
                        current = contract_repair.advance(
                            directory, contract_repair.WAITING_EXTERNAL,
                            last_transport_failure=detail,
                        )
                except ContractRepairIntegrityError as marker:
                    raise PipelineFailure(marker.code, str(marker), step_id=step.id) from exc
                data = publish("waiting_external", current)
                self.runtime.trace_emit(
                    "recovery.waiting_external", phase="implementation", cycle=cycle,
                    step_id=step.id,
                    data={"operation": "contract_repair", "reason": "LLM_FAILURE", **data},
                )
                raise
            except ScopeApprovalRequired:
                delta = _read_json_artifact(directory / "scope_delta.json", 64 * 1024)
                store.update(
                    status=RunStatus.WAITING_SCOPE_APPROVAL,
                    current_step=step.id,
                    scope_delta=delta if isinstance(delta, dict) else {},
                )
                raise
            except (ContractRepairIntegrityError, StepContractRepairArtifactError) as exc:
                raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
            except StepContractRepairOutputInvalid as exc:
                # Never a new AGENT_CONTRACT_MISMATCH.  One bounded,
                # self-contained planner restart first, in this same semantic
                # slot; then the slot waits for an operator planner retry.
                if self._restart_contract_repair_planner(directory, cycle, step.id, publish):
                    continue
                try:
                    current = contract_repair.ensure(
                        directory, contract_repair.OUTPUT_CORRECTION_EXHAUSTED,
                        last_output_error={
                            "code": exc.code, "detail": exc.detail[:500],
                            "output_attempt": exc.output_attempt,
                        },
                    )
                except ContractRepairIntegrityError as marker:
                    raise PipelineFailure(marker.code, str(marker), step_id=step.id) from exc
                data = publish("output_correction_exhausted", current)
                self.runtime.trace_emit(
                    "contract_repair.output_correction.exhausted", phase="implementation",
                    cycle=cycle, step_id=step.id, data=data,
                )
                raise PipelineFailure(
                    exc.code,
                    bounded_v2_report(
                        f"contract repair {progress['repair_id']} planner output is invalid after "
                        f"{exc.corrections} of {exc.limit} output corrections: {exc.detail}"
                    ),
                    step_id=step.id,
                ) from exc
            except V2PlanParseError as exc:
                # A planner answer that could not even be made durable.
                raise PipelineFailure(
                    STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
                    bounded_v2_report(f"contract repair planner output is unusable: {exc}"),
                    step_id=step.id,
                ) from exc
            except (AgentError, GitError, OSError) as exc:
                raise StepExecutionFailure(
                    "AGENT_CONTRACT_MISMATCH", step.id,
                    bounded_v2_report(f"contract repair failed: {exc}"),
                    profile_id=profile_id, tree_before=tree_before,
                    tree_after=tree_before, usage=usage, mismatch=mismatch,
                    mismatch_retry_count=semantic_attempt, step_dir=artifact_dir,
                ) from exc
        progress = progress_of(contract_repair.read_transaction(directory) or {})
        decision = classify_failure(
            "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True, rollback_succeeded=True,
        )
        # One semantic record per repair, whatever the number of resumes.
        recovery.record(RecoveryAttempt(
            phase="implementation", reason="AGENT_CONTRACT_MISMATCH",
            attempt=semantic_attempt, budget_key="contract_repairs",
            budget=max_repairs, budget_consumed=semantic_attempt,
            strategy=decision.strategy.value,
            cycle=cycle, step_id=step.id, profile_id=profile_id,
            tree_before=tree_before, tree_after=tree_before,
            operation_id=progress["repair_id"],
        ))
        contract_repair.ensure(directory, contract_repair.COMPLETED)
        store.update(
            status=RunStatus.CONTRACT_REPAIRING, current_step=step.id,
            contract_repair={"status": "completed", **progress},
        )
        self.runtime.trace_emit(
            "step.contract_repair.completed", phase="implementation",
            cycle=cycle, step_id=step.id,
            data={
                "repair": number, "repair_id": progress["repair_id"], "tree_sha": tree_before,
                "output_corrections": progress["output_correction_attempt"],
            },
        )
        recovery.trace(
            "recovery.completed", reason="AGENT_CONTRACT_MISMATCH",
            decision=decision, attempt=semantic_attempt,
            tree_before=tree_before, tree_after=candidate_tree_sha(worktree),
            budget_remaining=max(0, max_repairs - semantic_attempt),
            phase="implementation", cycle=cycle,
            step_id=step.id, recovered=True,
        )
        return repaired, effective_contract
    def _restart_contract_repair_planner(
        self, directory: Path, cycle: int, step_id: str,
        publish: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    ) -> bool:
        """Admit one bounded planner restart of an exhausted output budget.

        The restart stays inside the same semantic repair slot: it consumes
        no ``contract_repairs`` budget, replays no worker, keeps every raw
        answer, and re-sends a standalone request built from the durable
        mismatch, tree, identity and repository topology evidence.
        """

        limit = self.runtime.run_options.recovery.max_contract_repair_planner_restarts
        max_corrections = self.runtime.run_options.recovery.max_contract_repair_output_corrections
        try:
            current = contract_repair.read_transaction(directory) or {}
            used = int(current.get("planner_restarts") or 0)
            if used >= limit:
                return False
            attempt = contract_repair.output_attempt(current)
            current = contract_repair.advance(
                directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                output_attempt=attempt + 1, output_correction_attempt=attempt,
                output_correction_limit=int(current.get("output_correction_limit") or 0)
                + max(1, max_corrections),
                planner_restarts=used + 1, planner_transport_attempt=0,
            )
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step_id) from exc
        data = publish("correcting_output", current)
        self.runtime.trace_emit(
            "contract_repair.planner_restart", phase="implementation",
            cycle=cycle, step_id=step_id, data={**data, "planner_restarts": used + 1},
        )
        return True
    def _repair_step_contract(
        self, *, repo: Path, worktree: Path, run_dir: Path,
        artifact_dir: Path, original_spec: str, original_plan_identity: str,
        original_step: ImplementationStep, current_contract: str,
        expected_plan_step_count: int | None,
        mismatch: str, tree_before: str,
        future_ownership: Mapping[str, tuple[str, ...]] | None,
        resume_request: bool = False,
        max_output_corrections: int = 0,
        on_request: Callable[[int], None] | None = None,
        on_response_durable: Callable[[int], None] | None = None,
        on_output_invalid: Callable[[int, str], None] | None = None,
        failure_evidence: str = "NONE",
        require_new_contract: bool = False,
    ) -> ImplementationStep:
        """Run and validate one durable StepContractRepairPlanner transaction.

        ``resume_request`` re-sends (or re-parses the answer to) the exact
        durable request of an interrupted transaction instead of rebuilding it.
        Deterministic answer defects (protocol, identity, removed approved
        paths, anchors absent from the tree) are planner output corrections;
        a mutable-scope expansion is then decided by the scope policy only.
        ``require_new_contract`` marks a repair a red deterministic gate
        opened: an answer identical to the contract that gate failed under is
        one more answer defect, never a silent replay of the approved step.
        """

        planner_profile = profile_for_role(
            self.runtime.config, self.runtime.run_options.planner_profile, ExecutionRole.PLANNER
        )
        original_mutable = set((*original_step.write_set, *original_step.create_set, *original_step.delete_set))

        def validate(repaired: ImplementationStep) -> None:
            removed = original_mutable - set((*repaired.write_set, *repaired.create_set, *repaired.delete_set))
            if removed:
                raise V2PlanParseError(
                    "contract repair removed approved mutable paths: " + ", ".join(sorted(removed))[:300]
                )
            if require_new_contract and repaired == original_step:
                raise V2PlanParseError(
                    "contract repair is identical to the contract the deterministic gate "
                    "failed under: a replan repairs the step contract, it never replays it"
                )
            drift = self._step_contract_drift(repo, tree_before, tree_before, repaired)
            if drift:
                raise V2PlanParseError("repaired contract is not executable on current tree: " + drift)

        planner = StepContractRepairPlanner(
            self.runtime.planner_client or chat_client(
                build_llm_endpoint(planner_profile), self.runtime.environment, self.runtime.trace_transport
            ),
            max_read_paths_per_step=self.runtime.config.planning.max_read_paths_per_step,
            max_output_corrections=max_output_corrections,
        )
        sets = {
            "read_set": "\n".join(f"- {item}" for item in original_step.read_set),
            "write_set": "\n".join(f"- {item}" for item in original_step.write_set) or "NONE",
            "create_set": "\n".join(f"- {item}" for item in original_step.create_set) or "NONE",
            "delete_set": "\n".join(f"- {item}" for item in original_step.delete_set) or "NONE",
        }
        try:
            # Deterministic evidence for the planner, never a path decision.
            topology: RepositoryTopology | None = RepositoryTopology.from_tree(repo, tree_before)
        except GitError:
            topology = None
        hooks = {
            "identity": StepRepairIdentity.of(original_step, expected_plan_step_count),
            "validate": validate,
            "on_request": on_request, "on_response_durable": on_response_durable,
            "on_output_invalid": on_output_invalid,
            "topology": topology,
        }
        if resume_request:
            repaired = planner.resume(
                artifacts_dir=artifact_dir,
                original_plan_identity=original_plan_identity,
                current_contract=current_contract,
                mismatch_explanation=bounded_v2_report(mismatch),
                current_tree_sha=tree_before,
                **sets, **hooks,
            )
        else:
            evidence_parts = [
                "TREE SHA: " + tree_before,
                "STATUS: " + "; ".join(status_porcelain(worktree)[:20]),
            ]
            for path in read_set_paths(original_step.read_set):
                target = worktree / path
                try:
                    data = target.read_bytes()[:8192]
                    evidence_parts.append(
                        f"PATH {path}\n" + data.decode("utf-8", errors="replace")
                    )
                except (OSError, UnicodeError):
                    evidence_parts.append(f"PATH {path}\n<unavailable>")
            repaired = planner.repair(
                original_spec=original_spec, current_tree_sha=tree_before,
                original_plan_identity=original_plan_identity,
                current_contract=current_contract,
                mismatch_explanation=bounded_v2_report(mismatch),
                future_ownership=_json_text(future_ownership or {}),
                repository_evidence="\n\n".join(evidence_parts),
                artifacts_dir=artifact_dir,
                failure_evidence=failure_evidence,
                **sets, **hooks,
            )
        contract_repair.ensure(artifact_dir, contract_repair.PLANNER_VALIDATED)
        repaired_mutable = set((*repaired.write_set, *repaired.create_set, *repaired.delete_set))
        added = repaired_mutable - original_mutable
        if added:
            policy = self.runtime.repair_scope
            if len(added) > policy.max_added_paths or policy.policy == "deny-expansion":
                # A valid answer the scope policy does not authorize: an
                # operator decision, never a planner output correction.
                raise PipelineFailure(
                    "CONTRACT_REPAIR_SCOPE_DENIED",
                    f"contract repair requested {len(added)} additional mutable path(s) "
                    f"beyond the {policy.policy} bound of {policy.max_added_paths}: "
                    + ", ".join(sorted(added))[:500],
                    step_id=original_step.id,
                )
            if policy.policy == "require-approval":
                delta = {
                    "schema_version": 1, "step_id": original_step.id,
                    "added_paths": sorted(added), "tree_sha": tree_before,
                    "repair_number": int(artifact_dir.name),
                }
                delta_path = artifact_dir / "scope_delta.json"
                atomic_write_text(delta_path, _json_text(delta))
                approval = read_scope_approval(
                    artifact_dir,
                    expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
                )
                if approval is None:
                    contract_repair.ensure(artifact_dir, contract_repair.SCOPE_WAITING)
                    raise ScopeApprovalRequired()
                if approval.decision is not ApprovalDecision.APPROVE:
                    raise PipelineFailure("HUMAN_REQUIRED", "contract repair scope rejected")
        validation_path = artifact_dir / "validation.json"
        validation = _read_json_artifact(validation_path, 64 * 1024)
        if isinstance(validation, dict):
            validation.update({
                "status": "validated", "added_mutable_paths": sorted(added),
                "removed_mutable_paths": [],
                "original_step_contract_sha256": hashlib.sha256(current_contract.encode("utf-8")).hexdigest(),
                "repaired_contract_sha256": hashlib.sha256(
                    (artifact_dir / "contract.md").read_bytes()
                ).hexdigest(),
            })
            atomic_write_text(validation_path, _json_text(validation))
            contract_repair.ensure(artifact_dir, contract_repair.VALIDATED)
        return repaired
    def _pre_step_boundary_drift(
        self, repo: Path, worktree: Path, ownership_before: GitOwnership, *,
        branch_ref: str, base_sha: str, tree_before: str,
    ) -> str | None:
        """Why the repository is no longer exactly in its pre-step state."""

        try:
            if candidate_tree_sha(worktree) != tree_before:
                return "the candidate tree is no longer the pre-step tree"
            if index_tree_sha(worktree) != tree_before:
                return "the index is no longer the pre-step index"
            if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
                return "the worktree has unstaged or untracked modifications"
        except GitError as exc:
            return f"Git state is unreadable: {exc}"
        violations = _ownership_violations(
            ownership_before, _git_ownership(repo, worktree),
            branch_ref=branch_ref, base_sha=base_sha,
        )
        return "; ".join(violations) or None
    def _run_step_attempt(
        self,
        *,
        repo: Path,
        worktree: Path,
        base_sha: str,
        branch_ref: str,
        ownership_before: GitOwnership,
        expected_tree: str,
        step: ImplementationStep,
        contract: str,
        profile_id: str,
        artifact_dir: Path,
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
        original_spec: str = "",
        initial_mismatch: str | None = None,
        mismatch_retry_count: int = 0,
    ) -> StepExecutionOutcome:
        """The single authoritative execution of one step, in any cycle.

        Gates run in a fixed order and every failure raises
        :class:`StepExecutionFailure`; the caller owns the run status.  The
        worker's final report is data only and never drives a decision.
        """

        step_id = step.id
        # The retry mode of this invocation, recorded with every failure it
        # can raise so the durable step record says which attempt failed.
        retry_mode: dict[str, Any] = (
            {
                "mismatch_retry_count": mismatch_retry_count,
                "initial_mismatch": bounded_v2_report(initial_mismatch or "") or None,
            }
            if mismatch_retry_count else {}
        )
        # 1-2. The exact tree the worker will receive, and the contract's Git
        # preconditions on it.
        tree_before = candidate_tree_sha(worktree)
        drift = self._step_contract_drift(repo, tree_before, expected_tree, step)
        if drift:
            raise StepExecutionFailure(
                "STEP_CONTRACT_DRIFT", step_id, drift,
                profile_id=profile_id, tree_before=tree_before, **retry_mode,
            )
        # The complete Git boundary a no-op mismatch must leave untouched.
        # Accumulated modifications from the earlier steps are legitimate, so
        # the gate is "unchanged", never "empty".
        index_before = index_tree_sha(worktree)
        status_before = status_porcelain(worktree)
        # 3-4. Approved profile and isolated environment.
        profile, step_role = self._step_profile(profile_id)
        executor = self.runtime.executor_for_profile(
            profile.id,
            step_role,
            forbidden_env_names=forbidden_env_names,
        )
        selected_executor = self.runtime.trace_selected_profile(profile.id, step_role, step_id=step_id)
        selection_source = "primary execution authority"
        frozen_selection = self.runtime.last_selection
        if frozen_selection is not None and step_id is not None:
            for selected_step in frozen_selection.steps:
                if selected_step.step_id == step_id and any(
                    fallback.profile_id == profile.id for fallback in selected_step.fallbacks
                ):
                    selection_source = "frozen execution fallback authority"
                    break
        atomic_write_text(artifact_dir / "executor.json", _json_text({
            "profile_id": profile.id,
            "role": step_role.value,
            "config_sha256": getattr(selected_executor, "config_sha256", None),
            "selection_source": selection_source,
        }))
        # 5. One fresh worker process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
        # The selected adapter owns the single execution call.
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompt_payload = build_implementer_payload(
                original_spec=original_spec,
                step_identity=f"{step.id}\nTITLE\n{step.title}",
                step_title=step.title,
                step_objective=step.objective,
                read_set="\n".join(step.read_set),
                write_set="\n".join(step.write_set) or "NONE",
                create_set="\n".join(step.create_set) or "NONE",
                delete_set="\n".join(step.delete_set) or "NONE",
                mutable_scope=_json_text({
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                }),
                repository_instructions=step.instructions,
                verify_instructions=step.verify,
                instructions=step.instructions,
                verify_contract=step.verify,
                forbidden_contract=step.forbidden,
                budget_bytes=self.runtime.config.prompt_budget.implementer_max_bytes,
            )
            request_prompt = prompt_payload.rendered
            write_prompt_diagnostics(artifact_dir, prompt_payload)
            trace_started_at = self.runtime.trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = selected_executor
            self.runtime.trace_emit(
                "step.started",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
                    "tree_before": tree_before,
                    "session": self.runtime.trace_session(
                        profile=profile,
                        selected=trace_selected,
                        role=step_role,
                        prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before,
                    ),
                },
            )
            result = executor.run(
                AgentRunRequest(
                    role=step_role,
                    profile_id=profile.id,
                    prompt=request_prompt,
                    worktree=worktree,
                    artifact_dir=artifact_dir,
                    mutable_paths=tuple(
                        sorted({*step.write_set, *step.create_set, *step.delete_set})
                    ),
                    prompt_mode="raw",
                    contract=contract,
                    retry_addendum=None,
                )
            )
        except AgentScopeError as exc:
            self.runtime.trace_emit(
                "step.agent.completed",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
                    "status": "failed",
                    "session": self.runtime.trace_session(
                        profile=profile,
                        selected=trace_selected,
                        role=step_role,
                        prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before,
                        exit_reason=getattr(exc, "code", type(exc).__name__),
                    ),
                },
            )
            self.runtime.redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id, redact(str(exc), self.runtime.secrets),
                profile_id=profile.id, tree_before=tree_before, **retry_mode,
            ) from None
        except AgentError as exc:
            reason = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
            tree_after = _safe_candidate_tree(worktree)
            _record_failure_tree(artifact_dir, worktree)
            atomic_write_text(artifact_dir / "failure.json", _json_text({
                "schema_version": 1,
                "reason": str(reason)[:120],
                "profile_id": profile.id,
                "tree_before": tree_before,
                "tree_after": tree_after,
                "mutable_scope": sorted({
                    *step.write_set, *step.create_set, *step.delete_set,
                }),
            }))
            self.runtime.redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                reason, step_id, redact(str(exc), self.runtime.secrets),
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, status_before=status_before,
                **retry_mode,
            ) from None
        # 6. Complete and redact the durable artifacts.
        self.runtime.ensure_step_artifacts(artifact_dir, result)
        self.runtime.redact_step_artifacts(artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self.runtime.secrets),
            stderr_tail=redact(result.stderr_tail, self.runtime.secrets),
        )
        self.runtime.trace_emit(
            "step.agent.completed",
            phase="implementation",
            cycle=self.runtime.trace_cycle,
            step_id=step_id,
            data={
                "attempt": mismatch_retry_count + 1,
                "status": result.status,
                "session": self.runtime.trace_session(
                    profile=profile,
                    selected=trace_selected,
                    role=step_role,
                    prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                    started_at=trace_started_at,
                    started_mono=trace_started_mono,
                    tree_before=tree_before,
                    result=result,
                ),
            },
        )
        usage = normalize_usage(result.usage)
        # Advisory, argument-free context diagnostics (never a gate).
        try:
            write_token_diagnostics(artifact_dir, usage, worktree=worktree)
        except (OSError, ResultArtifactError):
            pass
        failed = {
            "profile_id": profile.id, "tree_before": tree_before, "usage": usage,
            "status_before": status_before,
            **retry_mode,
        }
        # 7. Authentication classification from fixed markers only.
        auth_failure = result.backend_reason == "AGENT_AUTH_FAILURE"
        # Capture ownership before interpreting the worker's structural report.
        # A clean mismatch is allowed to defer only when the complete Git
        # boundary is untouched.
        ownership_after = _git_ownership(repo, worktree)
        ownership_violations = _ownership_violations(
            ownership_before, ownership_after, branch_ref=branch_ref, base_sha=base_sha
        )
        # A structural mismatch is the only worker report that has protocol
        # meaning.  It is checked before any staging and its explanation stays
        # bounded and non-authoritative.
        mismatch = None
        if not result.timed_out and result.exit_code == 0:
            mismatch = contract_mismatch_explanation(result.final_message)
            if mismatch is None:
                # A successful, completely clean no-op has the same semantic
                # meaning as a worker-declared structural mismatch.  Feed it
                # through the existing bounded retry path; any boundary drift
                # remains fail-closed below.
                no_change_tree = _safe_candidate_tree(worktree)
                no_change_index = _safe_index_tree(worktree)
                no_change_status = _safe_status(worktree)
                if (
                    no_change_tree == tree_before
                    and index_before == tree_before
                    and no_change_index == tree_before
                    and not _status_has_unstaged_or_untracked(status_before)
                    and no_change_status == status_before
                    and not ownership_violations
                ):
                    mismatch = (
                        _BOUNDED_NO_CHANGE_MISMATCH
                        if mismatch_retry_count else _SYNTHETIC_NO_CHANGE_MISMATCH
                    )
        if mismatch is not None:
            # Freeze even unstaged worker edits into a durable candidate tree
            # before the repair transaction.  This makes a crash at the
            # mismatch boundary recoverable by the normal resume validator.
            try:
                stage_all(worktree)
            except GitError as exc:
                raise StepExecutionFailure(
                    AGENT_RUNTIME_FAILED, step_id, "could not freeze mismatch tree",
                    **failed, tree_after=_safe_candidate_tree(worktree),
                ) from exc
            tree_after = _safe_candidate_tree(worktree)
            index_after = _safe_index_tree(worktree)
            status_after = _safe_status(worktree)
            if ownership_violations:
                _record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    "AGENT_GIT_VIOLATION", step_id,
                    "; ".join(ownership_violations),
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=tree_after, usage=usage,
                    mismatch=bounded_v2_report(mismatch),
                    mismatch_retry_count=mismatch_retry_count,
                )
            # Clean means "the worker changed nothing": the candidate tree,
            # the index, the porcelain status and Git ownership are all exactly
            # what this step received.  It is deliberately not "git status is
            # empty": the cumulative modifications of the earlier approved
            # steps are legitimate and untracked-but-ignored files are never a
            # gate here.
            clean = (
                bool(mismatch.strip())
                and tree_after == tree_before
                and index_after == index_before
                and status_after == status_before
                and not ownership_violations
            )
            changed = []
            if tree_after is not None:
                try:
                    changed = changed_paths_between_trees(repo, tree_before, tree_after)
                except GitError:
                    changed = []
            allowed = {*step.write_set, *step.create_set, *step.delete_set}
            unexpected = [path for path in changed if path not in allowed]
            if unexpected:
                _record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    AGENT_SCOPE_VIOLATION, step_id,
                    "worker changed paths outside scope: " + _paths_detail(unexpected),
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=tree_after, usage=usage,
                    mismatch=bounded_v2_report(mismatch),
                    index_tree_after=index_after,
                    mismatch_retry_count=mismatch_retry_count,
                )
            atomic_write_text(artifact_dir / "step.json", _json_text({
                "id": step_id, "status": "FAILED", "reason": "AGENT_CONTRACT_MISMATCH",
                "profile_id": profile.id, "tree_before": tree_before,
                "tree_after": tree_after, "index_tree_after": index_after,
                "changed_paths": list(changed),
                "mismatch": bounded_v2_report(mismatch), "usage": usage,
            }))
            _record_failure_tree(artifact_dir, worktree)
            details = []
            if mismatch:
                details.append(bounded_v2_report(mismatch))
            if tree_after is None:
                details.append("failure tree could not be read")
            residual = _new_status_lines(status_before, status_after)
            raise StepExecutionFailure(
                "AGENT_CONTRACT_MISMATCH", step_id,
                "; ".join(details) or "worker reported a contract mismatch",
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, usage=usage,
                mismatch=bounded_v2_report(mismatch),
                clean_contract_mismatch=clean,
                mismatch_retry_count=mismatch_retry_count,
                initial_mismatch=(
                    bounded_v2_report(initial_mismatch)
                    if mismatch_retry_count and initial_mismatch else None
                ),
                index_tree_after=index_after,
            )
        # 8-9. Git ownership: HEAD, branch, branches and worktrees.
        if ownership_after.head != base_sha:
            raise StepExecutionFailure("AGENT_GIT_VIOLATION", step_id, "worktree HEAD changed", **failed)
        if ownership_violations:
            raise StepExecutionFailure(
                "AGENT_GIT_VIOLATION", step_id, "; ".join(ownership_violations), **failed
            )
        if auth_failure:
            raise StepExecutionFailure(
                AGENT_AUTH_FAILURE, step_id, "worker authentication failed", **failed,
                tree_after=_safe_candidate_tree(worktree),
            )
        # 10-11. Process outcome.  The tree left behind is recorded so that a
        # resume can tell a clean retry from partial worker changes.
        if result.timed_out or result.exit_reason == AGENT_TIMEOUT:
            raise StepExecutionFailure(
                AGENT_TIMEOUT,
                step_id, **failed, tree_after=_safe_candidate_tree(worktree)
            )
        if result.exit_code not in (0, None) or result.exit_reason in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            reason = result.exit_reason or AGENT_RUNTIME_FAILED
            raise StepExecutionFailure(
                reason, step_id, f"exit status {result.exit_code}", **failed,
                tree_after=_safe_candidate_tree(worktree),
            )
        # 12-14. Freeze the candidate; a step must change it.
        stage_all(worktree)
        tree_after = index_tree_sha(worktree)
        if tree_after == tree_before:
            raise StepExecutionFailure(
                "AGENT_NO_CHANGE", step_id,
                bounded_v2_report(result.final_message) or "candidate delta is empty",
                tree_after=tree_after, index_tree_after=_safe_index_tree(worktree), **failed,
            )
        # 15-16. Git, not the prompt, is the scope barrier: every changed path
        # must be authorized by this step's WRITE, CREATE or DELETE set.
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        allowed = {*step.write_set, *step.create_set, *step.delete_set}
        unexpected = [path for path in changed_paths if path not in allowed]
        if unexpected:
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id,
                f"unexpected={_paths_detail(unexpected)}", **failed, tree_after=tree_after,
            )
        # 17-18. Durable step record, then the outcome.  A deferred verify
        # dependency is recorded as data for the reviser and the reviewer; it never
        # relaxes a deterministic gate.
        deferred_verify = bounded_v2_report(
            deferred_verify_dependency(result.final_message) or ""
        )
        atomic_write_text(artifact_dir / "step.json", _json_text({
            "id": step_id, "status": "COMPLETED", "profile_id": profile.id,
            "tree_before": tree_before, "tree_after": tree_after,
            "changed_paths": list(changed_paths),
            **({"mismatch_retry_count": mismatch_retry_count}
               if mismatch_retry_count else {}),
            **({"initial_mismatch": bounded_v2_report(initial_mismatch)}
               if mismatch_retry_count and initial_mismatch else {}),
            **({"deferred_verify": deferred_verify} if deferred_verify else {}),
            "usage": usage,
        }))
        return StepExecutionOutcome(
            step_id=step_id,
            profile_id=profile.id,
            tree_before=tree_before,
            tree_after=tree_after,
            changed_paths=tuple(changed_paths),
            usage=usage,
            final_report=result.final_message,
            deferred_verify=deferred_verify,
            mismatch_retry_count=mismatch_retry_count,
        )
    def _accept_v2_step_tree(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: WorktreeInfo,
        authority: EffectiveStepAuthority,
        outcome: StepExecutionOutcome,
        parent_sha: str,
        future_step_ids: Sequence[str],
        run_id: str,
        step_dir: Path,
        verification: StepVerification | None = None,
    ) -> str | None:
        """Run the reusable safety gate and accept one normal step tree.

        Worker attempts never call this method until their structural and
        scope gates have passed.  A red/failed attempt therefore remains an
        artifact tree only.  The explicit deferred contract is the sole
        exception to a passed verification status.  The commit gate receives
        the effective authority the worker executed under, never the
        approved step it was repaired from.
        """

        step_id = authority.step_id
        if current_head(info.worktree) != parent_sha:
            raise CommitSafetyError(
                "step parent HEAD changed before acceptance", code=COMMIT_PARENT_MISMATCH,
            )
        if outcome.no_change:
            if (
                outcome.tree_after != outcome.tree_before
                or candidate_tree_sha(info.worktree) != outcome.tree_after
                or index_tree_sha(info.worktree) != outcome.tree_after
                or _status_has_unstaged_or_untracked(status_porcelain(info.worktree))
            ):
                raise CommitSafetyError(
                    "no-change outcome does not match the exact current tree",
                    code=COMMIT_TREE_MISMATCH,
                )
            parent_list = commit_parents(info.worktree, parent_sha)
            expected_parent = parent_list[0] if parent_list else None
            store.update(
                status=RunStatus.IMPLEMENTING,
                expected_head_sha=parent_sha,
                expected_parent_sha=expected_parent,
                expected_tree_sha=outcome.tree_after,
                next_step_id=(future_step_ids[0] if future_step_ids else None),
                no_change_step=step_id,
            )
            self.runtime.trace_emit(
                "step.no_change.accepted", phase="implementation",
                cycle=self.runtime.trace_cycle, step_id=step_id,
                data={"head_sha": parent_sha, "tree_sha": outcome.tree_after},
            )
            return None
        if verification is None:
            verification = self._step_verification(authority, outcome, future_step_ids)
        verification_status, deferred = verification.status, verification.deferred

        self.runtime.trace_emit(
            "step.verification.completed",
            phase="implementation",
            cycle=self.runtime.trace_cycle,
            step_id=step_id,
            data={
                "status": verification_status,
                "tree_before": outcome.tree_before,
                "tree_after": outcome.tree_after,
                "deferred": deferred is not None,
            },
        )

        # A no-change deferred mismatch is a traceable worker outcome, not a
        # new Git state.  There is no legal empty commit; the later candidate
        # gate still sees the durable mismatch artifact.
        if outcome.tree_after == outcome.tree_before:
            if deferred is None:
                raise CommitSafetyError(
                    "an unchanged step tree is neither a no-change nor a deferred outcome",
                    code=COMMIT_TREE_MISMATCH,
                )
            state = store.load()
            deferred_records = list(state.get("deferred_verifications") or [])
            deferred_records.append({
                "step_id": step_id,
                "verification_status": "deferred",
                "tree_before": outcome.tree_before,
                "tree_after": outcome.tree_after,
                "changed_paths": [],
                "deferred_reason": deferred.reason,
                "dependent_step_ids": list(deferred.dependent_step_ids),
                "deferred_verify_command_or_contract": deferred.command_or_contract,
                "commit_sha": None,
            })
            parents = commit_parents(info.worktree, parent_sha)
            expected_parent = parents[0] if parents else None
            store.update(
                status=RunStatus.IMPLEMENTING,
                deferred_verifications=deferred_records,
                expected_head_sha=parent_sha,
                expected_parent_sha=expected_parent,
                expected_tree_sha=outcome.tree_after,
                next_step_id=(future_step_ids[0] if future_step_ids else None),
            )
            return None

        gate = commit_safety_gate(
            info.worktree,
            tree_sha=outcome.tree_after,
            parent_sha=parent_sha,
            mutable_scope=authority.mutable_scope,
            verification_status=verification_status,
            deferred_reason=deferred.reason if deferred is not None else None,
            dependent_step_ids=deferred.dependent_step_ids if deferred is not None else (),
            deferred_command_or_contract=(
                deferred.command_or_contract if deferred is not None else None
            ),
            secrets=self.runtime.secrets,
            # v2 deliberately treats a large diff as bounded review
            # evidence; the canonical blob-size and binary policies still
            # apply in the shared scanner.
            max_diff_bytes=None,
        )
        diff_path = step_dir / "diff.patch"
        atomic_write_text(diff_path, redact(staged_diff(info.worktree), self.runtime.secrets))
        commit_sha = commit_step_tree(
            info.worktree,
            tree_sha=gate.tree_sha,
            parent_sha=gate.parent_sha,
            step_id=step_id,
            step_title=authority.title,
            body=f"MetaHarness-Run: {run_id}",
        )
        record = accepted_step_record(
            step_id=step_id,
            verification_status=verification_status,
            parent_sha=parent_sha,
            commit_sha=commit_sha,
            tree_before=outcome.tree_before,
            tree_after=outcome.tree_after,
            changed_paths=gate.changed_paths,
            deferred=deferred,
            authority=authority.summary(),
        )
        self._finalize_accepted_step(
            store, run_dir, step_dir, record, future_step_ids=future_step_ids,
            authority=authority, diff_path=diff_path,
        )
        return commit_sha
    def _finalize_accepted_step(
        self, store: RunStateStore, run_dir: Path, step_dir: Path,
        record: Mapping[str, Any], *, future_step_ids: Sequence[str],
        authority: EffectiveStepAuthority, diff_path: Path | None,
    ) -> None:
        """Record one committed step durably; idempotent across a resume."""

        commit_sha = record["commit_sha"]
        step_path = step_dir / "step.json"
        step_payload = _read_json_artifact(step_path)
        if not isinstance(step_payload, dict):
            step_payload = {"id": authority.step_id}
        step_payload.update(record)
        step_payload["verification_status"] = record["verification_status"]
        atomic_write_text(step_path, _json_text(step_payload))

        def merged(records: Any) -> list[dict[str, Any]]:
            kept = [
                item for item in (records or [])
                if not (isinstance(item, dict) and item.get("commit_sha") == commit_sha)
            ]
            return [*kept, dict(record)]

        state = store.load()
        chain = merged(accepted_chain_records(run_dir))
        atomic_write_text(run_dir / "accepted-chain.json", _json_text({"commits": chain}))
        atomic_write_text(step_dir / STEP_ACCEPTANCE_NAME, _json_text({
            "schema_version": 1, "status": "accepted", "step_id": authority.step_id,
            "commit_sha": commit_sha, "parent_sha": record["parent_sha"],
            "tree_after": record["tree_after"],
            "commit_gate_authority_sha256": authority.authority_sha256,
            "effective_contract_sha256": authority.effective_contract_sha256,
            "authority_source": authority.authority_source,
            "repair_slot": authority.repair_slot,
        }))
        store.update(
            status=RunStatus.IMPLEMENTING,
            accepted_steps=merged(state.get("accepted_steps")),
            accepted_commits=merged(state.get("accepted_commits")),
            expected_head_sha=commit_sha,
            expected_parent_sha=record["parent_sha"],
            expected_tree_sha=record["tree_after"],
            next_step_id=future_step_ids[0] if future_step_ids else None,
        )
        self.runtime.trace_emit(
            "step.committed",
            phase="implementation",
            cycle=self.runtime.trace_cycle,
            step_id=authority.step_id,
            data={
                "parent_sha": record["parent_sha"],
                "commit_sha": commit_sha,
                "tree_sha": record["tree_after"],
                "changed_paths": list(record["changed_paths"]),
                "effective_authority_sha256": authority.authority_sha256,
                **(self.runtime.trace_diff_reference(diff_path) if diff_path is not None else {}),
            },
        )
    def _step_contract_drift(
        self, repo: Path, before_tree: str, expected_tree: str, step: ImplementationStep,
    ) -> str | None:
        """Check the step's Git preconditions on the tree Codex will receive.

        Every READ/WRITE/DELETE path must exist in *before_tree* and no CREATE
        path may exist.  The tree must also be exactly the base or the tree
        frozen after the previous step.
        """

        if before_tree != expected_tree:
            return "worktree changed outside a step"
        problems: list[str] = []
        for label, paths, must_exist in (
            ("read_missing", read_set_paths(step.read_set), True),
            ("write_missing", step.write_set, True),
            ("delete_missing", step.delete_set, True),
            ("create_exists", step.create_set, False),
        ):
            wrong = [path for path in paths
                     if path_exists_in_tree(repo, before_tree, path) is not must_exist]
            if wrong:
                problems.append(f"{label}={_paths_detail(wrong)}")
        return " ".join(problems) or None
