"""Step execution: one approved step, run as a bounded ladder of attempts.

This module owns exactly one pipeline operation, ``execute_cycle_step``.  It
drives the bounded worker attempts of one approved step: the transient retry
and executor-fallback admissions stay with
:mod:`~metaharness.orchestration.worker_recovery`, the single worker request and
its normalized candidate result with
:mod:`~metaharness.orchestration.worker_attempt`.  The successful attempt is
then handed to :mod:`~metaharness.orchestration.step_acceptance`.

It never plans a cycle, never reviews a candidate, never repairs a contract
itself and never crosses a commit boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Mapping,
    Sequence,
    TYPE_CHECKING,
)
from ..agent.base import (
    AGENT_AUTH_FAILURE,
)
from ..attempt_transaction import (
    AttemptBoundary,
    AttemptViolation,
    CandidateAttemptTransaction,
    git_ownership,
    ownership_violations,
    status_has_unstaged_or_untracked,
)
from ..gitops import (
    GitError,
    candidate_tree_sha,
    current_head,
    index_tree_sha,
    resolve_tree,
    restore_paths_from_tree,
    stage_all,
    status_porcelain,
)
from ..models import ExecutionRole, ImplementationStep
from ..profiles import profile_for_role
from ..recovery_policy import (
    FailureClass,
    RecoveryStrategy,
    classify_failure,
)
from ..result import atomic_write_text
from ..usage import normalize_usage
from ..resume import read_checkpoint
from ..state import RunStateStore
from .durable_readers import FAILED_CONTINUED, SKIPPED_DEPENDENCY, settled_step_status
from .per_step_gate import per_step_check_ids
from .pipeline_v2 import (
    BudgetExhausted,
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    step_dir as cycle_step_dir,
)
from .recovery import RecoveryAdmission
from .step_authority import future_step_ownership
from .shared import (
    GitOwnership,
    StepExecutionFailure,
    archive_attempt,
    bounded_v2_report,
    json_text,
    safe_candidate_tree,
)
from .step_authority import (
    EffectiveStepAuthority,
    EffectiveStepExecution,
    approved_step_contract,
    approved_step_authority,
    write_authority_diagnostic,
)


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
        # A dependent of a settled step is skipped; a skip settles it too, so
        # the whole transitive chain of DEPENDS_ON is skipped in plan order.
        dependency = step.depends_on
        if dependency and settled_step_status(
            cycle_step_dir(ctx.run_dir, cycle_plan.cycle, dependency), dependency,
        ):
            self._settle(store, ctx, cycle_plan, step, {
                "status": SKIPPED_DEPENDENCY, "depends_on": dependency,
            })
            return
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
        audit_profile = profile_for_role(
            self.runtime.config, ctx.selection.audit.profile_id, ExecutionRole.AUDITOR
        )
        # The last green tree this step must leave behind if it is abandoned.
        green_commit = checkpoint.last_green_commit if checkpoint is not None else None
        green = AttemptBoundary(
            resolve_tree(ctx.info.worktree, green_commit) if green_commit
            else candidate_tree_sha(ctx.info.worktree),
            status_porcelain(ctx.info.worktree),
            git_ownership(ctx.repo, ctx.info.worktree),
        )
        try:
            execution = self._execute_step_attempts(
                store=store, run_dir=ctx.run_dir, original_spec=ctx.spec,
                repo=ctx.repo, worktree=ctx.info.worktree, base_sha=parent_sha,
                branch_ref=ctx.branch_ref, ownership_before=green.ownership,
                expected_tree=green.tree, step=step, contract=contract,
                profile_id=cycle_plan.step_profile_ids[step.id],
                fallback_profile_ids=(cycle_plan.step_fallback_profile_ids or {}).get(step.id, ()),
                artifact_dir=step_artifact_dir,
                forbidden_env_names=(planner_profile.api_key_env, audit_profile.api_key_env),
                future_ownership=future_step_ownership(cycle_plan.plan.steps, index),
            )
        except StepExecutionFailure as failure:
            if classify_failure(failure.reason).failure_class is not FailureClass.FIXABLE:
                raise
            self._mark_failed_continue(
                store, ctx, cycle_plan, step, failure, green=green, parent_sha=parent_sha,
            )
            return
        self.runtime.step_acceptance.accept_execution(
            store, ctx, cycle_plan, index, execution, parent_sha=parent_sha,
        )
        store.update_metadata(
            current_step=None,
            steps=self.runtime.composition.state_steps(ctx, cycle_plan),
        )
        self.runtime.observability.update_v2_usage(store, ctx.run_dir)

    def _mark_failed_continue(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        step: ImplementationStep, failure: StepExecutionFailure, *,
        green: AttemptBoundary, parent_sha: str,
    ) -> None:
        """MARK_FAILED_CONTINUE: prove the last green tree, then settle the step.

        Only an exact restoration of the pre-step tree, index and Git ownership
        lets the run continue; anything else is ``ROLLBACK_FAILED``.
        """

        transaction = CandidateAttemptTransaction(
            ctx.repo, ctx.info.worktree, green, branch_ref=ctx.branch_ref,
            base_sha=parent_sha, secrets=self.runtime.secrets,
        )
        try:
            transaction.audit_ownership()
            tree, changed = transaction.freeze()
            if tree != green.tree:
                transaction.scan()
            transaction.rollback(changed)
        except AttemptViolation as violation:
            code = (
                violation.code if classify_failure(violation.code).failure_class is FailureClass.FATAL
                and violation.code != "RESUME_REQUIRES_OPERATOR" else "ROLLBACK_FAILED"
            )
            raise PipelineFailure(
                code, f"{failure.reason}: {violation.detail}", step_id=step.id,
            ) from violation
        directory = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        decision = classify_failure(failure.reason, exhausted=True)
        strategies = [
            item.get("strategy") for item in store.load().get("recovery_attempts") or ()
            if isinstance(item, dict) and item.get("step_id") == step.id
            and item.get("cycle") == self.runtime.trace_cycle
        ]
        if failure.profile_id not in {None, cycle_plan.step_profile_ids.get(step.id)}:
            strategies.append(RecoveryStrategy.FALLBACK_EXECUTOR.value)
        self._settle(store, ctx, cycle_plan, step, {
            "status": FAILED_CONTINUED, "reason": failure.reason,
            "detail": bounded_v2_report(str(failure.detail or "")),
            "strategies": list(dict.fromkeys((*strategies, decision.strategy.value))),
            "profile_id": failure.profile_id, "changed_paths": list(changed),
            "tree_before": green.tree, "tree_after": green.tree, "tree_restored": green.tree,
            "usage": normalize_usage(failure.usage),
        })
        self.runtime.recovery(store).trace(
            "recovery.failed_continued", reason=failure.reason, decision=decision,
            attempt=1, tree_before=tree, tree_after=green.tree, budget_remaining=0,
            phase="implementation", cycle=self.runtime.trace_cycle, step_id=step.id,
        )

    def _settle(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        step: ImplementationStep, record: Mapping[str, object],
    ) -> None:
        """Persist a step the run abandons while its independent steps go on."""

        directory = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        archive_attempt(directory)
        atomic_write_text(directory / "step.json", json_text({"id": step.id, **record}))
        self.runtime.observability.trace_emit(
            "step." + str(record["status"]).casefold(), phase="implementation",
            cycle=self.runtime.trace_cycle, step_id=step.id,
            data={key: value for key, value in record.items() if key != "usage"},
        )
        store.update_metadata(
            current_step=None,
            steps=self.runtime.composition.state_steps(ctx, cycle_plan),
        )

    def _execute_step_attempts(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        original_spec: str,
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
        fallback_profile_ids: Sequence[str] = (),
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
    ) -> EffectiveStepExecution:
        """Run the one bounded worker ladder of a plan step.

        ``budget.step_attempts`` is the total ceiling of this autonomous
        operation: the primary attempt, its same-executor retries and its
        executor fallbacks all consume it, so a step never spends more than
        that many semantic worker calls.  One attempt stays reserved per
        remaining fallback, so a configured escalation is always reachable
        inside the budget.
        """

        authority = approved_step_authority(step, contract)
        recovery = self.runtime.recovery(store)
        cycle_number = self.runtime.trace_cycle
        attempts = self.runtime.run_options.budget.step_attempts
        active_profile_id = profile_id
        fallback_ids = tuple(fallback_profile_ids)[:max(0, attempts - 1)]
        fallback_index = 0
        retry_key = recovery.budget_key("agent-step", f"{cycle_number:03d}", step.id)
        pending_retry: RecoveryAdmission | None = None
        retry_addendum: str | None = None
        gate_check_ids = per_step_check_ids(self.runtime, run_dir)
        attempt_number = 0
        while True:
            exhausted = self.runtime.budget_exhausted(store)
            if exhausted is not None:
                raise BudgetExhausted(exhausted)
            attempt_number += 1
            common = {
                "repo": repo, "worktree": worktree, "base_sha": base_sha,
                "branch_ref": branch_ref, "ownership_before": ownership_before,
                "expected_tree": expected_tree, "step": step,
                "contract": contract, "profile_id": active_profile_id,
                "artifact_dir": artifact_dir,
                "forbidden_env_names": forbidden_env_names,
                "future_ownership": future_ownership,
                "original_spec": original_spec,
                "attempt_number": attempt_number,
                "retry_addendum": retry_addendum,
            }
            try:
                write_authority_diagnostic(artifact_dir, authority)
                outcome = self.runtime.worker_attempt.run_attempt(**common)
                # The optional fast gate runs on the mutated step tree, before
                # this step's own commit, and only its new regressions count.
                per_step = self.runtime.gates.run_per_step_gate(
                    run_dir=run_dir, worktree=worktree, base_sha=base_sha,
                    step_dir=artifact_dir,
                    check_ids=gate_check_ids,
                    changed_paths=outcome.changed_paths,
                )
                if not per_step.passed:
                    raise StepExecutionFailure(
                        "PER_STEP_GATE_REGRESSION", step_id=step.id,
                        detail=per_step.feedback,
                        profile_id=active_profile_id,
                        tree_before=outcome.tree_before, tree_after=outcome.tree_after,
                        status_before=outcome.status_before,
                        retry_feedback=per_step.feedback,
                    )
                if per_step.warnings:
                    self.runtime.gates.record_step_warnings(store, per_step.warnings)
                if pending_retry is not None:
                    recovery.complete(
                        pending_retry, recovered=True, tree_after=outcome.tree_after,
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
                if pending_retry is not None:
                    recovery.complete(
                        pending_retry, recovered=False,
                        tree_after=failure.tree_after or safe_candidate_tree(worktree),
                    )
                    pending_retry = None
                remaining_fallbacks = len(fallback_ids) - fallback_index
                retry = self.runtime.worker_recovery(store).admit_step_retry(
                    failure=failure, retry_key=retry_key, repo=repo, worktree=worktree,
                    branch_ref=branch_ref, base_sha=base_sha,
                    ownership_before=ownership_before, step=step,
                    artifact_dir=artifact_dir, cycle=cycle_number,
                    retry_budget=max(0, attempts - 1 - remaining_fallbacks),
                )
                if retry is not None:
                    pending_retry = retry
                    feedback = failure.retry_feedback or failure.mismatch or failure.detail
                    if feedback:
                        retry_addendum = bounded_v2_report(feedback)
                    continue
                if failure.reason == AGENT_AUTH_FAILURE:
                    failure.reason = "EXTERNAL_AUTH_REQUIRED"
                    failure.detail = "executor credentials or external authorization are required"
                    failure.step_dir = artifact_dir
                    raise
                rolled_back = (
                    failure.tree_before is not None
                    and failure.tree_after == failure.tree_before
                )
                retryable = (
                    rolled_back
                    and not classify_failure(failure.reason).strategy.terminal
                )
                if (
                    retryable
                    and fallback_index < len(fallback_ids)
                    and recovery.used(retry_key) < attempts - 1
                ):
                    # The next rung of the ladder, paid for by the same one
                    # step budget as every other semantic worker attempt.
                    admission = recovery.admit(
                        retry_key, reason=failure.reason, budget=attempts - 1,
                        phase="implementation", cycle=cycle_number, step_id=step.id,
                        tree_before=failure.tree_before,
                        tree_after=safe_candidate_tree(worktree),
                        strategy=RecoveryStrategy.FALLBACK_EXECUTOR,
                    )
                    if not admission.admitted:
                        failure.step_dir = artifact_dir
                        raise
                    fallback_index += 1
                    active_profile_id = fallback_ids[fallback_index - 1]
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
                failure.step_dir = artifact_dir
                raise
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
