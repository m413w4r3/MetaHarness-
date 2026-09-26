"""One worker attempt of one approved step, and its normalized result.

The single authoritative execution of a step, in any cycle: the exact tree the
worker receives, the approved profile and its isolated environment, one fresh
worker process, the durable artifacts it leaves, and the
:class:`~metaharness.orchestration.shared.StepExecutionOutcome` its caller
consumes.  Gates run in a fixed order and every failure raises
:class:`~metaharness.orchestration.shared.StepExecutionFailure`.

The retry ladder, the semantic contract repair and the commit boundary stay
with their own transactions; this module only runs one attempt and normalizes
what it produced.
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
    GitOwnership,
    git_ownership,
    ownership_violations,
    paths_detail,
    status_has_unstaged_or_untracked,
)
from ..gitops import (
    GitError,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    stage_all,
    status_porcelain,
)
from ..models import (
    ExecutionRole,
    ImplementationStep,
    ModelProfile,
)
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
    BOUNDED_NO_CHANGE_MISMATCH,
    SYNTHETIC_NO_CHANGE_MISMATCH,
    StepExecutionFailure,
    StepExecutionOutcome,
    bounded_v2_report,
    json_text,
    new_status_lines,
    record_failure_tree,
    safe_candidate_tree,
    safe_index_tree,
    safe_status,
)
from .step_authority import step_contract_drift


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
        drift = step_contract_drift(repo, tree_before, expected_tree, step)
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
                mutable_scope=json_text({
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
            trace_started_at = self.runtime.observability.trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = selected_executor
            self.runtime.observability.trace_emit(
                "step.started",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
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
                    retry_addendum=None,
                )
            )
        except AgentScopeError as exc:
            self.runtime.observability.trace_emit(
                "step.agent.completed",
                phase="implementation",
                cycle=self.runtime.trace_cycle,
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
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
                profile_id=profile.id, tree_before=tree_before, **retry_mode,
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
                **retry_mode,
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
                "attempt": mismatch_retry_count + 1,
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
            **retry_mode,
        }
        # 7. Authentication classification from fixed markers only.
        auth_failure = result.backend_reason == "AGENT_AUTH_FAILURE"
        # Capture ownership before interpreting the worker's structural report.
        # A clean mismatch is allowed to defer only when the complete Git
        # boundary is untouched.
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
            if mismatch is None:
                # A successful, completely clean no-op has the same semantic
                # meaning as a worker-declared structural mismatch.  Feed it
                # through the existing bounded retry path; any boundary drift
                # remains fail-closed below.
                no_change_tree = safe_candidate_tree(worktree)
                no_change_index = safe_index_tree(worktree)
                no_change_status = safe_status(worktree)
                if (
                    no_change_tree == tree_before
                    and index_before == tree_before
                    and no_change_index == tree_before
                    and not status_has_unstaged_or_untracked(status_before)
                    and no_change_status == status_before
                    and not boundary_violations
                ):
                    mismatch = (
                        BOUNDED_NO_CHANGE_MISMATCH
                        if mismatch_retry_count else SYNTHETIC_NO_CHANGE_MISMATCH
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
                    **failed, tree_after=safe_candidate_tree(worktree),
                ) from exc
            tree_after = safe_candidate_tree(worktree)
            index_after = safe_index_tree(worktree)
            status_after = safe_status(worktree)
            if boundary_violations:
                record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    "AGENT_GIT_VIOLATION", step_id,
                    "; ".join(boundary_violations),
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
                and not boundary_violations
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
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=tree_after, usage=usage,
                    mismatch=bounded_v2_report(mismatch),
                    index_tree_after=index_after,
                    mismatch_retry_count=mismatch_retry_count,
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
            residual = new_status_lines(status_before, status_after)
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
            raise StepExecutionFailure(
                "AGENT_NO_CHANGE", step_id,
                bounded_v2_report(result.final_message) or "candidate delta is empty",
                tree_after=tree_after, index_tree_after=safe_index_tree(worktree), **failed,
            )
        # 15-16. Git, not the prompt, is the scope barrier: every changed path
        # must be authorized by this step's WRITE, CREATE or DELETE set.
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        allowed = {*step.write_set, *step.create_set, *step.delete_set}
        unexpected = [path for path in changed_paths if path not in allowed]
        if unexpected:
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id,
                f"unexpected={paths_detail(unexpected)}", **failed, tree_after=tree_after,
            )
        # 17-18. Durable step record, then the outcome.  A deferred verify
        # dependency is recorded as data for the reviser and the reviewer; it never
        # relaxes a deterministic gate.
        deferred_verify = bounded_v2_report(
            deferred_verify_dependency(result.final_message) or ""
        )
        atomic_write_text(artifact_dir / "step.json", json_text({
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
    def _step_profile(self, profile_id: str) -> tuple[ModelProfile, ExecutionRole]:
        """The approved implementation profile of one plan step."""

        return profile_for_role(self.runtime.config, profile_id, ExecutionRole.IMPLEMENTER), ExecutionRole.IMPLEMENTER
    def no_change_outcome(
        self, failure: StepExecutionFailure, artifact_dir: Path, *,
        worktree: Path, expected_head: str,
    ) -> StepExecutionOutcome:
        """Record a clean empty delta after the bounded worker/repair path."""

        before = failure.tree_before
        after = safe_candidate_tree(worktree)
        index_after = safe_index_tree(worktree)
        status_after = safe_status(worktree)
        if (
            before is None or after != before or index_after != before
            or current_head(worktree) != expected_head
            or status_has_unstaged_or_untracked(status_after or ())
            or (failure.status_before is not None and status_after != failure.status_before)
        ):
            failure.step_dir = artifact_dir
            raise failure
        usage = normalize_usage(failure.usage)
        report = bounded_v2_report(failure.detail or "candidate delta is empty")
        atomic_write_text(artifact_dir / "step.json", json_text({
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
        self.runtime.observability.trace_emit(
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
