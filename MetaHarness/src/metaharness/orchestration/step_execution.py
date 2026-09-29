"""Step execution: one approved step, run as a bounded ladder of attempts.

This module owns exactly one pipeline operation, ``execute_cycle_step``.  It
drives the bounded worker attempts of one approved step: the transient retry
and executor-fallback admissions stay with
:mod:`~metaharness.orchestration.recovery`, the single worker request and its
normalized candidate result with :mod:`~metaharness.orchestration.worker_attempt`.
The successful attempt then crosses the acceptance boundary in this module.

It never plans a cycle, never reviews a candidate, never repairs a contract
itself and never crosses a commit boundary.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..attempt_transaction import (
    AttemptBoundary,
    AttemptViolation,
    CandidateAttemptTransaction,
    git_ownership,
    status_has_unstaged_or_untracked,
)
from ..commit_gate import COMMIT_GATE_FAILED, CommitSafetyError, StepVerification, accepted_step_record, commit_safety_gate, step_verification
from ..gitops import (
    GitError,
    WorktreeInfo,
    candidate_tree_sha,
    commit_parents,
    commit_step_tree,
    current_head,
    index_tree_sha,
    resolve_tree,
    staged_diff,
    status_porcelain,
)
from ..models import ExecutionRole, ImplementationStep
from ..planning.artifacts import read_approved_step_contract
from ..planning.protocol import V2PlanParseError
from ..profiles import profile_for_role
from ..recovery_policy import FailureClass, RecoveryStrategy, classify_failure
from ..redaction import redact
from ..result import atomic_write_text
from ..resume import read_checkpoint
from ..state import RunStateStore
from ..usage import normalize_usage
from .gates import per_step_check_ids
from .pipeline_v2 import BudgetExhausted, CyclePlan, PipelineFailure, PipelineV2Context
from .pipeline_v2 import step_dir as cycle_step_dir
from .publication import accepted_chain_records
from .recovery import RecoveryAdmission, normalize_exit_reason
from .shared import (
    FAILED_CONTINUED,
    SKIPPED_DEPENDENCY,
    GitOwnership,
    StepExecutionFailure,
    StepExecutionOutcome,
    archive_attempt,
    bounded_parse_detail,
    bounded_v2_report,
    json_text,
    read_json_artifact,
    safe_candidate_tree,
    settled_step_status,
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
            failure.reason = normalize_exit_reason(failure.reason)
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
                failure.reason = normalize_exit_reason(failure.reason)
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
                if failure.reason == "EXTERNAL_AUTH_REQUIRED":
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


if TYPE_CHECKING:  # pragma: no cover - the plan authority is the coordinator
    from .pipeline_v2 import CyclePlan


STEP_AUTHORITY_NAME = "step_authority.json"
STEP_CANDIDATE_NAME = "step_candidate.json"
STEP_ACCEPTANCE_NAME = "step_acceptance.json"
STEP_CANDIDATE_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 256 * 1024
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StepAuthorityError(Exception):
    """Durable candidate evidence is incomplete or inconsistent."""

    code = "RESUME_INTEGRITY_FAILURE"


def mutable_paths(step: ImplementationStep) -> tuple[str, ...]:
    return tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))


def future_step_ownership(
    steps: Sequence[ImplementationStep], index: int,
) -> dict[str, tuple[str, ...]]:
    """Paths assigned to later approved steps, for verification dependencies."""

    return {
        step.id: paths
        for step in steps[index + 1:]
        if (paths := mutable_paths(step))
    }


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


@dataclass(frozen=True)
class EffectiveStepAuthority:
    """The plan step and its hash-bound, approved contract."""

    step_id: str
    title: str
    execution_class: str
    depends_on: str | None
    effective_step: ImplementationStep = field(repr=False)
    effective_contract: str = field(repr=False)
    approved_contract_sha256: str
    effective_contract_sha256: str

    @property
    def read_set(self) -> tuple[str, ...]:
        return self.effective_step.read_set

    @property
    def write_set(self) -> tuple[str, ...]:
        return self.effective_step.write_set

    @property
    def create_set(self) -> tuple[str, ...]:
        return self.effective_step.create_set

    @property
    def delete_set(self) -> tuple[str, ...]:
        return self.effective_step.delete_set

    @property
    def mutable_scope(self) -> tuple[str, ...]:
        return mutable_paths(self.effective_step)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "step_id": self.step_id,
            "title": self.title,
            "execution_class": self.execution_class,
            "depends_on": self.depends_on,
            "read_set": list(self.read_set),
            "write_set": list(self.write_set),
            "create_set": list(self.create_set),
            "delete_set": list(self.delete_set),
            "approved_contract_sha256": self.approved_contract_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
        }

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def summary(self) -> dict[str, Any]:
        return {
            "effective_authority_sha256": self.authority_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
            "approved_contract_sha256": self.approved_contract_sha256,
            "effective_mutable_paths": list(self.mutable_scope),
        }


@dataclass(frozen=True)
class EffectiveStepExecution:
    """A worker outcome together with the authority it executed under."""

    outcome: StepExecutionOutcome
    authority: EffectiveStepAuthority


def approved_step_contract(cycle_plan: "CyclePlan", step: ImplementationStep) -> str:
    """The hash-bound approved contract: immutable evidence of the step."""

    try:
        return read_approved_step_contract(
            cycle_plan.contracts_dir, cycle_plan.bundle, step.id,
        )
    except (V2PlanParseError, OSError, UnicodeError) as exc:
        raise PipelineFailure("PLAN_APPROVAL_INVALID", str(exc), step_id=step.id) from exc


def approved_step_authority(
    step: ImplementationStep, approved_contract: str,
) -> EffectiveStepAuthority:
    digest = hashlib.sha256(approved_contract.encode("utf-8")).hexdigest()
    return EffectiveStepAuthority(
        step_id=step.id, title=step.title,
        execution_class=step.execution_class.value, depends_on=step.depends_on,
        effective_step=step, effective_contract=approved_contract,
        approved_contract_sha256=digest, effective_contract_sha256=digest,
    )


def build_step_candidate(
    *, run_id: str, cycle: int, step_id: str, parent_head_sha: str,
    tree_before: str, tree_after: str, changed_paths: Sequence[str], profile_id: str,
    authority: EffectiveStepAuthority, verification: Mapping[str, Any],
    step_record_sha256: str, final_report_sha256: str | None,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STEP_CANDIDATE_SCHEMA_VERSION,
        "run_id": run_id,
        "cycle": cycle,
        "step_id": step_id,
        "parent_head_sha": parent_head_sha,
        "tree_before": tree_before,
        "tree_after": tree_after,
        "changed_paths": sorted(changed_paths),
        "profile_id": profile_id,
        "effective_authority_sha256": authority.authority_sha256,
        "effective_contract_sha256": authority.effective_contract_sha256,
        "approved_contract_sha256": authority.approved_contract_sha256,
        "effective_mutable_paths": list(authority.mutable_scope),
        "verification": dict(verification),
        "outcome": {
            "step_record": "step.json",
            "step_record_sha256": step_record_sha256,
            "final_report": "agent.final.md",
            "final_report_sha256": final_report_sha256,
        },
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    payload["candidate_sha256"] = canonical_sha256(payload)
    return payload


def write_step_candidate(step_dir: Path, payload: Mapping[str, Any]) -> str:
    """Write the candidate, read it back and return the sha of its bytes."""

    path = step_dir / STEP_CANDIDATE_NAME
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)
    if read_step_candidate(step_dir) != dict(payload):
        raise StepAuthorityError("step candidate could not be durably written")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_step_candidate(step_dir: Path) -> dict[str, Any] | None:
    """The self-hashed candidate, or ``None`` when absent; corruption raises."""

    path = step_dir / STEP_CANDIDATE_NAME
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != STEP_CANDIDATE_SCHEMA_VERSION:
        raise StepAuthorityError("step candidate is unreadable or has an unknown schema")
    body = {key: value for key, value in payload.items() if key != "candidate_sha256"}
    if payload.get("candidate_sha256") != canonical_sha256(body):
        raise StepAuthorityError("step candidate hash changed")
    changed = payload.get("changed_paths")
    outcome = payload.get("outcome")
    if (
        not all(
            isinstance(payload.get(key), str) and _OBJECT_ID.fullmatch(payload[key])
            for key in ("parent_head_sha", "tree_before", "tree_after")
        )
        or not all(
            isinstance(payload.get(key), str) and _SHA256.fullmatch(payload[key])
            for key in ("effective_authority_sha256", "effective_contract_sha256")
        )
        or not isinstance(changed, list) or not changed
        or any(not isinstance(item, str) for item in changed)
        or not isinstance(outcome, dict)
        or not isinstance(payload.get("step_id"), str)
    ):
        raise StepAuthorityError("step candidate is incomplete")
    return payload


def write_authority_diagnostic(step_dir: Path, authority: EffectiveStepAuthority) -> None:
    """Advisory copy of the approved authority an attempt ran with."""

    atomic_write_text(
        step_dir / STEP_AUTHORITY_NAME,
        json.dumps({"step_id": authority.step_id, **authority.summary()},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
                "step parent HEAD changed before acceptance", code=COMMIT_GATE_FAILED,
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
                    code=COMMIT_GATE_FAILED,
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
                    code=COMMIT_GATE_FAILED,
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


__all__ = ['EffectiveStepAuthority', 'EffectiveStepExecution', 'STEP_ACCEPTANCE_NAME', 'STEP_AUTHORITY_NAME', 'STEP_CANDIDATE_NAME', 'StepAuthorityError', 'approved_step_authority', 'approved_step_contract', 'build_step_candidate', 'canonical_sha256', 'mutable_paths', 'read_step_candidate', 'write_authority_diagnostic', 'write_step_candidate']
