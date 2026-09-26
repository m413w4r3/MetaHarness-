"""Step execution: one approved step, run as a bounded ladder of attempts.

This module owns exactly one pipeline operation, ``execute_cycle_step``.  It
drives the bounded worker attempts of one approved step: the transient retry
and executor-fallback admissions stay with
:mod:`~metaharness.orchestration.worker_recovery`, the single worker request and
its normalized candidate result with
:mod:`~metaharness.orchestration.worker_attempt`, the semantic contract-mismatch
route with :mod:`~metaharness.orchestration.contract_recovery`.  The successful
attempt is then handed to :mod:`~metaharness.orchestration.step_acceptance`.

It never plans a cycle, never reviews a candidate, never repairs a contract
itself and never crosses a commit boundary.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import (
    Mapping,
    Sequence,
    TYPE_CHECKING,
)
from ..agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_SCOPE_VIOLATION,
)
from ..attempt_transaction import (
    git_ownership,
    ownership_violations,
    status_has_unstaged_or_untracked,
)
from ..gitops import (
    GitError,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    restore_paths_from_tree,
    stage_all,
    status_porcelain,
)
from ..models import ExecutionRole, ImplementationStep
from ..profiles import profile_for_role
from ..recovery_policy import (
    RecoveryStrategy,
    classify_failure,
)
from ..result import atomic_write_text
from ..resume import read_checkpoint
from ..state import RunStateStore
from . import contract_repair
from .contract_repair import ContractRepairIntegrityError
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    step_dir as cycle_step_dir,
)
from .recovery import RecoveryAdmission
from .revision import future_step_ownership
from .shared import (
    BOUNDED_NO_CHANGE_MISMATCH,
    SYNTHETIC_NO_CHANGE_MISMATCH,
    GitOwnership,
    StepExecutionFailure,
    archive_attempt,
    bounded_v2_report,
    json_text,
    paths_detail,
    safe_candidate_tree,
)
from .step_authority import (
    EffectiveStepAuthority,
    EffectiveStepExecution,
    approved_step_contract,
    write_authority_diagnostic,
)
from .worker_recovery import TRANSIENT_WORKER_FAILURES


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


class StepExecutionService:
    """One owner of the step execution transactions described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def execute_cycle_step(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan, index: int,
    ) -> None:
        """Execute, verify and accept exactly one approved step of a cycle."""

        step = cycle_plan.plan.steps[index]
        step_artifact_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        # A previous failed attempt of this same step keeps its artifacts.
        archive_attempt(step_artifact_dir)
        contract = approved_step_contract(cycle_plan, step)
        store.update_metadata(
            current_step=step.id,
            steps=self.runtime.composition.state_steps(ctx, cycle_plan, running=step.id),
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
            original_plan_identity=json_text(
                asdict(checkpoint.plan_identity) if checkpoint and checkpoint.plan_identity else {}
            ),
            repo=ctx.repo, worktree=ctx.info.worktree, base_sha=parent_sha,
            branch_ref=ctx.branch_ref,
            ownership_before=git_ownership(ctx.repo, ctx.info.worktree),
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
        self.runtime.step_acceptance.accept_execution(
            store, ctx, cycle_plan, index, execution, parent_sha=parent_sha,
        )
        store.update_metadata(
            current_step=None,
            steps=self.runtime.composition.state_steps(ctx, cycle_plan),
        )
        self.runtime.observability.update_v2_usage(store, ctx.run_dir)
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
            return self.runtime.contract_recovery.resolve_step_authority(
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
            self.runtime.contract_recovery.repair_transaction(
                **repair_context, directory=pending_repair.directory,
                number=pending_repair.number, step=effective_step,
                current_contract=effective_contract, mismatch=pending_repair.mismatch,
                tree_before=pending_repair.tree_sha, profile_id=active_profile_id,
                resumed=True,
                failure_evidence=self.runtime.contract_recovery.repair_failure_evidence(pending_repair.directory),
                # A slot a red gate opened owes a contract different from the
                # one that gate failed under, on a resume as well: an
                # unchanged answer stays an output defect, never a replay.
                require_new_contract=self.runtime.contract_recovery.repair_slot_origin(pending_repair.directory) is not None,
            )
            authority = self.runtime.contract_recovery.repaired_authority(resolve(), pending_repair.number)
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
                outcome = self.runtime.worker_attempt.run_attempt(
                    **common, initial_mismatch=None,
                    mismatch_retry_count=repair_count,
                )
                if pending_transient is not None:
                    recovery.complete(
                        pending_transient, recovered=True, tree_after=outcome.tree_after,
                    )
                return EffectiveStepExecution(outcome, authority)
            except StepExecutionFailure as failure:
                atomic_write_text(artifact_dir / "failure.json", json_text({
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
                        tree_after=failure.tree_after or safe_candidate_tree(worktree),
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
                    return EffectiveStepExecution(self.runtime.worker_attempt.no_change_outcome(
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
                        tree_after=safe_candidate_tree(worktree),
                    )
                    archive_attempt(artifact_dir)
                    atomic_write_text(artifact_dir / "executor.json", json_text({
                        "profile_id": active_profile_id,
                        "role": "implementer",
                        "selection_source": "frozen execution fallback authority",
                    }))
                    store.update_metadata(current_step=step.id)
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
                        failure.detail = "worker changed paths outside scope: " + paths_detail(unexpected)
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
                archive_attempt(artifact_dir)
                if repair_count >= max_repairs:
                    if failure.mismatch in {
                        SYNTHETIC_NO_CHANGE_MISMATCH,
                        BOUNDED_NO_CHANGE_MISMATCH,
                    }:
                        return EffectiveStepExecution(self.runtime.worker_attempt.no_change_outcome(
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
                self.runtime.contract_recovery.repair_transaction(
                    **repair_context, directory=repair_dir, number=number,
                    step=effective_step, current_contract=effective_contract,
                    mismatch=bounded_mismatch, tree_before=failure.tree_before,
                    profile_id=active_profile_id, resumed=False, usage=failure.usage,
                )
                authority = self.runtime.contract_recovery.repaired_authority(resolve(), number)
                effective_step, effective_contract = authority.effective_step, authority.effective_contract
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
                and not status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except (GitError, OSError):
            return False
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
            if status_has_unstaged_or_untracked(status_porcelain(worktree)):
                return "the worktree has unstaged or untracked modifications"
        except GitError as exc:
            return f"Git state is unreadable: {exc}"
        violations = ownership_violations(
            ownership_before, git_ownership(repo, worktree),
            branch_ref=branch_ref, base_sha=base_sha,
        )
        return "; ".join(violations) or None
