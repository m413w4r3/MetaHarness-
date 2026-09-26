"""The durable acceptance boundary of one approved step.

A successful worker tree becomes durable evidence first and a commit second:
the candidate is frozen as ``step_candidate.json``, the checkpoint moves to
``STEP_ACCEPTANCE``, and only then does the commit gate run, with the very
:class:`~metaharness.orchestration.step_authority.EffectiveStepAuthority` the
worker executed under.  ``resume_step_acceptance`` re-derives every proof of an
interrupted acceptance instead of replaying a worker, a planner or a reviewer.
"""

from __future__ import annotations

import functools
import hashlib

from pathlib import Path
from typing import (
    Any,
    Mapping,
    NoReturn,
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
    changed_paths_between_trees,
    commit_message,
    commit_parents,
    commit_step_tree,
    current_head,
    index_tree_sha,
    resolve_tree,
    staged_diff,
    status_porcelain,
    symbolic_head,
)
from ..models import RunStatus
from ..redaction import redact
from ..resume import ResumePhase, read_checkpoint
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
    approved_step_contract,
    build_step_candidate,
    read_step_candidate,
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
        authority = self.runtime.contract_recovery.resolve_step_authority(
            step_dir, step, approved_step_contract(cycle_plan, step),
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
        record = read_json_artifact(record_path, 128 * 1024)
        if (
            not isinstance(record, dict) or record.get("id") != step.id
            or record.get("tree_before") != tree_before or record.get("tree_after") != tree_after
            or sorted(record.get("changed_paths") or []) != sorted(changed)
        ):
            refuse("the step record does not match the candidate")
        future = tuple(item.id for item in cycle_plan.plan.steps[index + 1:])
        self.runtime.observability.trace_emit(
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
                    steps=self.runtime.composition.state_steps(ctx, cycle_plan),
                )
                return
            if hashlib.sha256(record_bytes).hexdigest() != outcome_refs.get("step_record_sha256"):
                refuse("the step record changed")
            if (
                resolve_tree(worktree, parent_sha) != tree_before
                or index_tree_sha(worktree) != tree_after
                or candidate_tree_sha(worktree) != tree_after
                or status_has_unstaged_or_untracked(status_porcelain(worktree))
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
            steps=self.runtime.composition.state_steps(ctx, cycle_plan),
        )
        self.runtime.observability.update_v2_usage(store, ctx.run_dir)
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
            or status_has_unstaged_or_untracked(status_porcelain(worktree))
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
        return PipelineFailure(
            "COMMIT_GATE_FAILED",
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
            store.update(
                status=RunStatus.IMPLEMENTING,
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
