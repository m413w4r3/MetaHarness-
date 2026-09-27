"""One worker attempt of one approved step, and its normalized result.

The single authoritative execution of a step, in any cycle: the exact tree the
worker receives, the approved profile and its isolated environment, one fresh
worker process, the durable artifacts it leaves, and the
:class:`~metaharness.orchestration.shared.StepExecutionOutcome` its caller
consumes.  Gates run in a fixed order and every failure raises
:class:`~metaharness.orchestration.shared.StepExecutionFailure`.

The retry ladder and commit boundary stay with their own transactions; this
module only runs one attempt and normalizes what it produced.
"""

from __future__ import annotations

import dataclasses
import time

from pathlib import Path
from typing import (
    Any,
    Mapping,
    TYPE_CHECKING,
)
from ..agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
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
from ..attempt_transaction import (
    AttemptViolation,
    GitOwnership,
    audit_git_mutation,
    git_ownership,
    ownership_violations,
    paths_detail,
    recover_worker_git_state,
)
from ..gitops import (
    GitError,
    candidate_tree_sha,
    changed_paths_between_trees,
    index_tree_sha,
    restore_paths_from_tree,
    stage_all,
    status_porcelain,
)
from ..models import (
    ExecutionRole,
    ImplementationStep,
    ModelProfile,
)
from ..plan_repository_validation import repository_tree_facts
from ..scope import ScopeViolation
from ..planning.normalization import normalization_entries, normalize_step_contract
from ..profiles import profile_for_role
from ..prompt_contracts import (
    build_implementer_payload,
    write_prompt_diagnostics,
)
from ..redaction import redact
from ..result import (
    ResultArtifactError,
    atomic_write_text,
)
from ..usage import normalize_usage
from .shared import (
    StepExecutionFailure,
    StepExecutionOutcome,
    bounded_v2_report,
    json_text,
    record_failure_tree,
    safe_candidate_tree,
    safe_index_tree,
)


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


class WorkerAttemptService:
    """One owner of the worker attempt described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def run_attempt(
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
        attempt_number: int = 1,
        retry_addendum: str | None = None,
    ) -> StepExecutionOutcome:
        """The single authoritative execution of one step, in any cycle.

        Gates run in a fixed order and every failure raises
        :class:`StepExecutionFailure`; the caller owns the run status.  The
        worker's final report is data only and never drives a decision.
        """

        step_id = step.id
        # 1-2. The exact tree the worker will receive, and the effective
        # contract Git determines on it.  A mechanical path misclassification
        # -- a CREATE_SET entry for a path that tree already holds, a WRITE_SET
        # entry for a path it does not -- is normalized here, against the real
        # tree, instead of being reported as an impossible contract.
        tree_before = candidate_tree_sha(worktree)
        if tree_before != expected_tree:
            raise StepExecutionFailure(
                "REPOSITORY_TREE_DRIFT_UNEXPLAINED", step_id,
                "worktree changed outside a step",
                profile_id=profile_id, tree_before=tree_before,
            )
        contract_normalization = normalize_step_contract(
            step, repository_tree_facts(repo, tree_before),
        )
        step = contract_normalization.step
        # The complete Git boundary a no-op mismatch must leave untouched.
        # Accumulated modifications from the earlier steps are legitimate, so
        # the gate is "unchanged", never "empty".
        status_before = status_porcelain(worktree)
        # 3-4. Approved profile and isolated environment.
        profile, step_role = self._step_profile(profile_id)
        executor = self.runtime.composition.executor_for_profile(
            profile.id,
            step_role,
            forbidden_env_names=forbidden_env_names,
        )
        selected_executor = self.runtime.observability.trace_selected_profile(profile.id, step_role, step_id=step_id)
        selection_source = "primary execution authority"
        frozen_selection = self.runtime.last_selection
        if frozen_selection is not None and step_id is not None:
            for selected_step in frozen_selection.steps:
                if selected_step.step_id == step_id and any(
                    fallback.profile_id == profile.id for fallback in selected_step.fallbacks
                ):
                    selection_source = "frozen execution fallback authority"
                    break
        atomic_write_text(artifact_dir / "executor.json", json_text({
            "profile_id": profile.id,
            "role": step_role.value,
            "config_sha256": getattr(selected_executor, "config_sha256", None),
            "selection_source": selection_source,
        }))
        # 5. One fresh worker process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
        # The selected adapter owns the single execution call.
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if contract_normalization.changed:
            # The plan artifact records what the harness normalized at planning
            # time; this one records only what this tree changed since, so an
            # identical second normalization writes a single, distinct record.
            atomic_write_text(artifact_dir / "contract_normalization.json", json_text({
                "schema": 1,
                "step_id": step_id,
                "normalizations": normalization_entries(contract_normalization.normalizations),
                "contradictions": list(contract_normalization.contradictions),
            }))
        try:
            prompt_payload = build_implementer_payload(
                original_spec=original_spec,
                step_identity=f"{step.id}\nTITLE\n{step.title}",
                step_title=step.title,
                context=step.context,
                read_set="\n".join(step.read_set) or "NONE",
                write_set="\n".join(step.write_set) or "NONE",
                create_set="\n".join(step.create_set) or "NONE",
                delete_set="\n".join(step.delete_set) or "NONE",
                mutable_scope=json_text({
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                }),
                instructions=step.instructions,
                interfaces=step.interfaces,
                examples=step.examples,
                tests=step.tests,
                pitfalls=step.pitfalls,
                done_when=step.done_when,
                verify_contract=step.verify,
                retry_addendum=retry_addendum or "",
                budget_bytes=self.runtime.config.prompt_budget.implementer_max_bytes,
            )
            request_prompt = prompt_payload.rendered
            write_prompt_diagnostics(artifact_dir, prompt_payload)
            trace_started_at = self.runtime.observability.trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = selected_executor
            self.runtime.observability.trace_emit(
                "step.started",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": attempt_number,
                    "tree_before": tree_before,
                    "session": self.runtime.observability.trace_session(
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
                    retry_addendum=retry_addendum,
                )
            )
        except AgentScopeError as exc:
            self.runtime.observability.trace_emit(
                "step.agent.completed",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": attempt_number,
                    "status": "failed",
                    "session": self.runtime.observability.trace_session(
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
            self.runtime.observability.redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id, redact(str(exc), self.runtime.secrets),
                profile_id=profile.id, tree_before=tree_before,
            ) from None
        except AgentError as exc:
            reason = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
            tree_after = safe_candidate_tree(worktree)
            record_failure_tree(artifact_dir, worktree)
            atomic_write_text(artifact_dir / "failure.json", json_text({
                "schema_version": 1,
                "reason": str(reason)[:120],
                "profile_id": profile.id,
                "tree_before": tree_before,
                "tree_after": tree_after,
                "mutable_scope": sorted({
                    *step.write_set, *step.create_set, *step.delete_set,
                }),
            }))
            self.runtime.observability.redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                reason, step_id, redact(str(exc), self.runtime.secrets),
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, status_before=status_before,
            ) from None
        # 6. Complete and redact the durable artifacts.
        self.runtime.observability.ensure_step_artifacts(artifact_dir, result)
        self.runtime.observability.redact_step_artifacts(artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self.runtime.secrets),
            stderr_tail=redact(result.stderr_tail, self.runtime.secrets),
        )
        self.runtime.observability.trace_emit(
            "step.agent.completed",
            phase="implementation",
            cycle=self.runtime.trace_cycle,
            step_id=step_id,
            data={
                "attempt": attempt_number,
                "status": result.status,
                "session": self.runtime.observability.trace_session(
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
        }
        # 7. Authentication classification from fixed markers only.
        auth_failure = result.backend_reason == "AGENT_AUTH_FAILURE"
        # Capture ownership before interpreting the worker's structural report.
        ownership_after = git_ownership(repo, worktree)
        mutation = audit_git_mutation(
            ownership_before, ownership_after, branch_ref=branch_ref, base_sha=base_sha,
            repo=repo,
        )
        if mutation.fatal_code is not None:
            record_failure_tree(artifact_dir, worktree)
            raise StepExecutionFailure(
                mutation.fatal_code, step_id, mutation.fatal_detail,
                profile_id=profile.id, tree_before=tree_before,
                tree_after=safe_candidate_tree(worktree), usage=usage,
            )
        if mutation.recoverable:
            # A local commit or a parasite branch is not a boundary: the
            # harness takes Git back, keeps the worker's content and goes on.
            try:
                recover_worker_git_state(
                    repo, worktree, mutation, branch_ref=branch_ref, base_sha=base_sha,
                )
            except AttemptViolation as violation:
                raise StepExecutionFailure(
                    violation.code, step_id, violation.detail,
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=safe_candidate_tree(worktree), usage=usage,
                ) from None
            ownership_after = git_ownership(repo, worktree)
        boundary_violations = ownership_violations(
            ownership_before, ownership_after, branch_ref=branch_ref, base_sha=base_sha
        )
        # A structural mismatch is the only worker report that has protocol
        # meaning.  It is checked before any staging and its explanation stays
        # bounded and non-authoritative.
        mismatch = None
        if not result.timed_out and result.exit_code == 0:
            mismatch = contract_mismatch_explanation(result.final_message)
        if mismatch is not None:
            # Freeze the attempt tree so the ordinary retry can prove and
            # restore its exact boundary.
            try:
                stage_all(worktree)
            except GitError as exc:
                raise StepExecutionFailure(
                    AGENT_RUNTIME_FAILED, step_id, "could not freeze mismatch tree",
                    **failed, tree_after=safe_candidate_tree(worktree),
                ) from exc
            tree_after = safe_candidate_tree(worktree)
            index_after = safe_index_tree(worktree)
            if boundary_violations:
                record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    "AGENT_GIT_VIOLATION", step_id,
                    "; ".join(boundary_violations),
                    **failed, tree_after=tree_after,
                    mismatch=bounded_v2_report(mismatch),
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
                record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    AGENT_SCOPE_VIOLATION, step_id,
                    "worker changed paths outside scope: " + paths_detail(unexpected),
                    **failed, tree_after=tree_after,
                    mismatch=bounded_v2_report(mismatch),
                    index_tree_after=index_after,
                )
            atomic_write_text(artifact_dir / "step.json", json_text({
                "id": step_id, "status": "FAILED", "reason": "AGENT_CONTRACT_MISMATCH",
                "profile_id": profile.id, "tree_before": tree_before,
                "tree_after": tree_after, "index_tree_after": index_after,
                "changed_paths": list(changed),
                "mismatch": bounded_v2_report(mismatch), "usage": usage,
            }))
            record_failure_tree(artifact_dir, worktree)
            details = []
            if mismatch:
                details.append(bounded_v2_report(mismatch))
            if tree_after is None:
                details.append("failure tree could not be read")
            raise StepExecutionFailure(
                "AGENT_CONTRACT_MISMATCH", step_id,
                "; ".join(details) or "worker reported a contract mismatch",
                **failed, tree_after=tree_after,
                mismatch=bounded_v2_report(mismatch),
                retry_feedback=bounded_v2_report(mismatch),
                index_tree_after=index_after,
            )
        # 8-9. Git ownership: HEAD, branch, branches and worktrees.
        if ownership_after.head != base_sha:
            raise StepExecutionFailure("AGENT_GIT_VIOLATION", step_id, "worktree HEAD changed", **failed)
        if boundary_violations:
            raise StepExecutionFailure(
                "AGENT_GIT_VIOLATION", step_id, "; ".join(boundary_violations), **failed
            )
        if auth_failure:
            raise StepExecutionFailure(
                AGENT_AUTH_FAILURE, step_id, "worker authentication failed", **failed,
                tree_after=safe_candidate_tree(worktree),
            )
        # 10-11. Process outcome.  The tree left behind is recorded so that a
        # resume can tell a clean retry from partial worker changes.
        if result.timed_out or result.exit_reason == AGENT_TIMEOUT:
            raise StepExecutionFailure(
                AGENT_TIMEOUT,
                step_id, **failed, tree_after=safe_candidate_tree(worktree)
            )
        if result.exit_code not in (0, None) or result.exit_reason in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            reason = result.exit_reason or AGENT_RUNTIME_FAILED
            raise StepExecutionFailure(
                reason, step_id, f"exit status {result.exit_code}", **failed,
                tree_after=safe_candidate_tree(worktree),
            )
        # 12-14. Freeze the candidate; a step must change it.
        stage_all(worktree)
        tree_after = index_tree_sha(worktree)
        if tree_after == tree_before:
            feedback = (
                "No in-scope candidate change was produced. Complete the approved "
                "step if work remains."
            )
            raise StepExecutionFailure(
                "AGENT_NO_CHANGE", step_id,
                feedback,
                tree_after=tree_after, index_tree_after=safe_index_tree(worktree), **failed,
                retry_feedback=feedback,
            )
        # 15-16. Git, not the prompt, is the scope authority.  A forbidden
        # path is fatal in both modes; an ordinary path outside the declared
        # sets is a signal: it is admitted and recorded (``soft``) or restored
        # to its pre-attempt content (``strict``).
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        policy = self.runtime.config.scope
        try:
            policy.check(changed_paths, worktree=worktree)
        except ScopeViolation as violation:
            raise StepExecutionFailure(
                violation.code, step_id, violation.detail,
                **failed, tree_after=tree_after,
            ) from None
        allowed = {*step.write_set, *step.create_set, *step.delete_set}
        unexpected = sorted(path for path in changed_paths if path not in allowed)
        if unexpected and policy.strict:
            try:
                restore_paths_from_tree(worktree, tree_before, unexpected)
                stage_all(worktree)
            except (GitError, OSError) as exc:
                raise StepExecutionFailure(
                    "ROLLBACK_FAILED", step_id,
                    f"out-of-scope paths could not be discarded: {exc}",
                    **failed, tree_after=safe_candidate_tree(worktree),
                ) from None
            tree_after = index_tree_sha(worktree)
            if tree_after == tree_before:
                # Nothing but out-of-scope work: the step is retried, with the
                # bounded feedback the discarded paths carry.
                raise StepExecutionFailure(
                    AGENT_SCOPE_VIOLATION, step_id,
                    "changes outside mutable scope were discarded: "
                    + paths_detail(unexpected),
                    retry_feedback=(
                        "changes outside mutable scope were discarded: "
                        + paths_detail(unexpected)
                    ),
                    **failed, tree_after=tree_after,
                )
            changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        out_of_scope_paths = () if policy.strict else tuple(unexpected)
        # 17-18. Durable step record, then the outcome.  A deferred verify
        # dependency is recorded as data for the audit; it never relaxes a
        # deterministic gate.
        deferred_verify = bounded_v2_report(
            deferred_verify_dependency(result.final_message) or ""
        )
        atomic_write_text(artifact_dir / "step.json", json_text({
            "id": step_id, "status": "COMPLETED", "profile_id": profile.id,
            "tree_before": tree_before, "tree_after": tree_after,
            "changed_paths": list(changed_paths),
            "out_of_scope_paths": list(out_of_scope_paths),
            "attempt": attempt_number,
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
            out_of_scope_paths=out_of_scope_paths,
            status_before=tuple(status_before),
        )
    def _step_profile(self, profile_id: str) -> tuple[ModelProfile, ExecutionRole]:
        """The approved implementation profile of one plan step."""

        return profile_for_role(self.runtime.config, profile_id, ExecutionRole.IMPLEMENTER), ExecutionRole.IMPLEMENTER
