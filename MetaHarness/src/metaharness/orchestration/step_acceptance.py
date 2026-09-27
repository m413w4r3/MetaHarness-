"""The acceptance and commit boundary of one successful worker step."""

from __future__ import annotations

import functools
import hashlib

from pathlib import Path
from typing import (
    Any,
    Mapping,
    Sequence,
    TYPE_CHECKING,
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
from ..gitops import (
    GitError,
    WorktreeInfo,
    candidate_tree_sha,
    commit_parents,
    commit_step_tree,
    current_head,
    index_tree_sha,
    staged_diff,
    status_porcelain,
)
from ..redaction import redact
from ..recovery_policy import FailureClass, classify_failure
from ..result import atomic_write_text
from ..state import RunStateStore
from ..usage import normalize_usage
from .candidate import accepted_chain_records
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    step_dir as cycle_step_dir,
)
from .shared import (
    StepExecutionOutcome,
    bounded_parse_detail,
    json_text,
    read_json_artifact,
    status_has_unstaged_or_untracked,
)
from .step_authority import (
    STEP_ACCEPTANCE_NAME,
    EffectiveStepAuthority,
    EffectiveStepExecution,
    StepAuthorityError,
    build_step_candidate,
    write_authority_diagnostic,
    write_step_candidate,
)


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


class StepAcceptanceService:
    """One owner of the step acceptance transactions described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def accept_execution(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        index: int, execution: EffectiveStepExecution, *, parent_sha: str,
    ) -> None:
        """Cross the durable worker-success -> commit boundary of one step.

        A changed tree is first frozen as ``step_candidate.json`` and then
        passed through the commit gate. Until the next checkpoint is written,
        a crash replays this step from its last accepted Git commit.
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
            accept(verification=verification)
        except (CommitSafetyError, GitError) as exc:
            raise self._step_acceptance_failure(step_dir, authority, exc) from exc

    def _step_verification(
        self, authority: EffectiveStepAuthority, outcome: StepExecutionOutcome,
        future_step_ids: Sequence[str],
    ) -> StepVerification:
        try:
            return step_verification(
                outcome.final_report, step_id=authority.step_id,
                future_step_ids=future_step_ids,
                reported_status=getattr(outcome, "verification_status", None),
                deferred_requested=bool(getattr(outcome, "deferred_verify", "")),
            )
        except CommitSafetyError:
            self.runtime.observability.trace_emit(
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
        self.runtime.observability.trace_emit(
            "step.candidate.persisted", phase="implementation",
            cycle=cycle_plan.cycle.number, step_id=authority.step_id,
            data={
                "tree_after": outcome.tree_after, "source": source,
                "effective_authority_sha256": authority.authority_sha256,
                "effective_contract_sha256": authority.effective_contract_sha256,
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
            atomic_write_text(step_dir / STEP_ACCEPTANCE_NAME, json_text({
                "schema_version": 1, "status": "refused", "step_id": authority.step_id,
                "code": code, "detail": message,
                "paths": list(getattr(exc, "paths", ()))[:20],
                "commit_gate_authority_sha256": authority.authority_sha256,
                "effective_contract_sha256": authority.effective_contract_sha256,
                "effective_mutable_paths": list(authority.mutable_scope),
            }))
        except OSError:
            pass
        # A security refusal keeps its own fatal code; every other refusal is
        # the ordinary, fixable commit-gate failure.
        fatal = classify_failure(code).failure_class is FailureClass.FATAL
        return PipelineFailure(
            code if fatal else COMMIT_GATE_FAILED,
            f"{code}: {message} (authority {authority.authority_sha256[:16]})",
            step_id=authority.step_id,
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
                or status_has_unstaged_or_untracked(status_porcelain(info.worktree))
            ):
                raise CommitSafetyError(
                    "no-change outcome does not match the exact current tree",
                    code=COMMIT_TREE_MISMATCH,
                )
            parent_list = commit_parents(info.worktree, parent_sha)
            expected_parent = parent_list[0] if parent_list else None
            store.update_metadata(
                expected_head_sha=parent_sha,
                expected_parent_sha=expected_parent,
                expected_tree_sha=outcome.tree_after,
                next_step_id=(future_step_ids[0] if future_step_ids else None),
                no_change_step=step_id,
            )
            self.runtime.observability.trace_emit(
                "step.no_change.accepted", phase="implementation",
                cycle=self.runtime.trace_cycle, step_id=step_id,
                data={"head_sha": parent_sha, "tree_sha": outcome.tree_after},
            )
            return None
        if verification is None:
            verification = self._step_verification(authority, outcome, future_step_ids)
        verification_status, deferred = verification.status, verification.deferred

        self.runtime.observability.trace_emit(
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
            store.update_metadata(
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
            mutable_scope=(
                *authority.mutable_scope, *outcome.out_of_scope_paths
            ),
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
        step_payload = read_json_artifact(step_path)
        if not isinstance(step_payload, dict):
            step_payload = {"id": authority.step_id}
        step_payload.update(record)
        step_payload["verification_status"] = record["verification_status"]
        atomic_write_text(step_path, json_text(step_payload))

        def merged(records: Any) -> list[dict[str, Any]]:
            kept = [
                item for item in (records or [])
                if not (isinstance(item, dict) and item.get("commit_sha") == commit_sha)
            ]
            return [*kept, dict(record)]

        state = store.load()
        chain = merged(accepted_chain_records(run_dir))
        atomic_write_text(run_dir / "accepted-chain.json", json_text({"commits": chain}))
        atomic_write_text(step_dir / STEP_ACCEPTANCE_NAME, json_text({
            "schema_version": 1, "status": "accepted", "step_id": authority.step_id,
            "commit_sha": commit_sha, "parent_sha": record["parent_sha"],
            "tree_after": record["tree_after"],
            "commit_gate_authority_sha256": authority.authority_sha256,
            "effective_contract_sha256": authority.effective_contract_sha256,
        }))
        store.update_metadata(
            accepted_steps=merged(state.get("accepted_steps")),
            accepted_commits=merged(state.get("accepted_commits")),
            expected_head_sha=commit_sha,
            expected_parent_sha=record["parent_sha"],
            expected_tree_sha=record["tree_after"],
            next_step_id=future_step_ids[0] if future_step_ids else None,
        )
        self.runtime.observability.trace_emit(
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
                **(self.runtime.observability.trace_diff_reference(diff_path) if diff_path is not None else {}),
            },
        )
