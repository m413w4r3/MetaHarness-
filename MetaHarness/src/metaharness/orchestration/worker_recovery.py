"""Recovery of untrusted worker attempts: implementer, reviser, check repair.

A failed worker attempt is first proven harmless by the shared
:class:`~metaharness.attempt_transaction.CandidateAttemptTransaction`; only
then does the :class:`RecoveryCoordinator` decide whether the same executor
is retried or a configured fallback executor is selected.  Model semantics
(contract repair, check repair scope, review correction) stay with callers.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
)
from ..attempt_transaction import (
    AttemptBoundary,
    AttemptViolation,
    CandidateAttemptTransaction,
    GitOwnership,
)
from ..gitops import GitError, candidate_tree_sha, index_tree_sha, status_porcelain
from ..models import ExecutionRole, ImplementationStep, RunStatus
from ..recovery_policy import RecoveryBudgets, RecoveryDisposition, classify_failure
from ..result import atomic_write_text
from ..state import RunStateStore
from .pipeline_v2 import PipelineFailure
from .recovery import RecoveryAdmission, RecoveryCoordinator
from .revision import _SCOPE_REQUEST_ROUTE
from .shared import (
    StepExecutionFailure,
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _json_text,
    _read_json_artifact,
    _record_failure_tree,
    _safe_candidate_tree,
    _safe_status,
)

TRANSIENT_WORKER_FAILURES = frozenset({
    AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AGENT_PROTOCOL_FAILED,
})
_RETRY_DISPOSITIONS = frozenset({
    RecoveryDisposition.RETRY_SAME, RecoveryDisposition.RETRY_AFTER_ROLLBACK,
})


def safe_scope_request_path(path: str) -> bool:
    return bool(
        isinstance(path, str) and path and path == path.strip()
        and path not in {".", ".."}
        and "\x00" not in path and "\\" not in path
        and not path.startswith("/") and "//" not in path
        and all(part not in {"", ".", "..", ".git"} for part in Path(path).parts)
    )


class WorkerRecovery:
    """Transient-failure recovery of one run's untrusted worker attempts."""

    def __init__(
        self,
        recovery: RecoveryCoordinator,
        *,
        store: RunStateStore,
        budgets: RecoveryBudgets,
        secrets: Sequence[str],
    ) -> None:
        self._recovery = recovery
        self._store = store
        self._budgets = budgets
        self._secrets = tuple(secrets)

    # -- implementation steps -------------------------------------------

    def admit_step_retry(
        self,
        *,
        failure: StepExecutionFailure,
        retry_key: str,
        repo: Path,
        worktree: Path,
        branch_ref: str,
        base_sha: str,
        ownership_before: GitOwnership,
        step: ImplementationStep,
        artifact_dir: Path,
        cycle: int,
    ) -> RecoveryAdmission | None:
        """Admit a same-executor retry only after an exact, in-scope rollback.

        On refusal ``failure`` carries the stable reason the run must project.
        """

        reason = failure.reason
        before = failure.tree_before
        trace = {"phase": "implementation", "cycle": cycle, "step_id": failure.step_id}
        attempt = self._recovery.used(retry_key) + 1
        if reason not in TRANSIENT_WORKER_FAILURES | {AGENT_AUTH_FAILURE}:
            facts = {"tree_changed_out_of_scope": reason == AGENT_SCOPE_VIOLATION}
            if classify_failure(reason, **facts).disposition is RecoveryDisposition.HARD_STOP:
                self._recovery.stop(
                    reason, attempt=attempt, tree_before=before,
                    tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                    facts=facts, **trace,
                )
            return None

        def refuse(code: str, detail: str, tree_after: str | None) -> None:
            if code == "RESUME_REQUIRES_OPERATOR":
                failure.detail = (failure.detail or "worker failed") + "; " + detail
                facts = {"rollback_succeeded": False}
            else:
                failure.detail = detail
                facts = {"tree_changed_out_of_scope": code == AGENT_SCOPE_VIOLATION}
            failure.reason = code
            failure.tree_after = tree_after
            failure.step_dir = artifact_dir
            self._recovery.stop(
                code, attempt=attempt, tree_before=before, tree_after=tree_after,
                facts=facts, **trace,
            )

        if before is None:
            refuse(
                "RESUME_REQUIRES_OPERATOR", "pre-attempt tree identity is unavailable",
                failure.tree_after or _safe_candidate_tree(worktree),
            )
            return None
        if failure.tree_after is None:
            refuse("RESUME_REQUIRES_OPERATOR", "post-attempt tree identity is unavailable", None)
            return None
        if _safe_status(worktree) is None:
            refuse("RESUME_REQUIRES_OPERATOR", "post-attempt status is unreadable", failure.tree_after)
            return None
        if failure.status_before is None:
            refuse("RESUME_REQUIRES_OPERATOR", "pre-attempt status is unavailable", failure.tree_after)
            return None
        transaction = CandidateAttemptTransaction(
            repo, worktree, AttemptBoundary(before, tuple(failure.status_before), ownership_before),
            branch_ref=branch_ref, base_sha=base_sha, secrets=self._secrets,
        )
        try:
            rollback = transaction.abort(
                set((*step.write_set, *step.create_set, *step.delete_set)),
            )
        except AttemptViolation as violation:
            refuse(violation.code, violation.detail, _safe_candidate_tree(worktree))
            return None

        failure.tree_after = before
        failure.index_tree_after = before
        admission = self._recovery.admit(
            retry_key, reason=reason, budget=self._budgets.max_transient_attempts,
            profile_id=failure.profile_id, tree_before=before, tree_after=before,
            allowed=_RETRY_DISPOSITIONS,
            facts={"tree_changed_in_scope": rollback.changed}, **trace,
        )
        if not admission.admitted:
            if admission.exhausted:
                failure.detail = (
                    (failure.detail or "transient agent failure")
                    + "; transient attempt budget exhausted"
                )
            failure.step_dir = artifact_dir
            return None
        self._store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
        _archive_attempt(artifact_dir)
        return admission

    # -- semantic reviser and check repair -------------------------------

    def run_revision(
        self,
        *,
        request: Mapping[str, Any],
        is_check_repair: bool,
        cycle: int,
        attempt: int | None,
        fallbacks_limit: int,
        run_attempt: Callable[..., tuple[Any | None, str | None]],
    ) -> tuple[Any | None, str | None]:
        """Run one reviser/repair contract, retrying only after exact rollback."""

        selection = request["selection"]
        primary = selection.check_repair if is_check_repair else selection.semantic_reviser
        if primary is None:
            return None, (
                "CHECK_REPAIR_PROFILE_MISSING" if is_check_repair
                else "SEMANTIC_REVISER_PROFILE_MISSING"
            )
        fallbacks = tuple(
            selection.check_repair_fallbacks if is_check_repair
            else selection.semantic_reviser_fallbacks
        )[:fallbacks_limit]
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        phase = "check-repair" if is_check_repair else "semantic-revision"
        stage_name = request.get("gate_stage", "")
        base_key = self._recovery.budget_key(phase, f"{cycle:03d}", stage_name, attempt or 0)
        retry_key = base_key
        active = primary
        fallback_index = 0
        artifact_dir = Path(request["artifact_dir"])
        repo = Path(request["repo"])
        worktree = Path(request["info"].worktree)
        allowed = set(request["mutable_scope"])

        while True:
            active_selection = dataclasses.replace(
                selection,
                check_repair=active if is_check_repair else selection.check_repair,
                semantic_reviser=active if not is_check_repair else selection.semantic_reviser,
            )
            current_request = {
                key: value for key, value in request.items() if key != "gate_stage"
            }
            current_request["selection"] = active_selection
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(artifact_dir / "executor.json", _json_text({
                "profile_id": active.profile_id,
                "role": role.value,
                "config_sha256": active.config_sha256,
                "selection_source": (
                    "primary execution authority" if active == primary
                    else "frozen execution fallback authority"
                ),
            }))
            try:
                transaction = CandidateAttemptTransaction.begin(
                    repo, worktree, branch_ref=request["branch_ref"], secrets=self._secrets,
                )
            except AttemptViolation as violation:
                raise PipelineFailure(violation.code, "revision " + violation.detail) from violation
            tree_before = transaction.boundary.tree
            try:
                result, error = run_attempt(**current_request)
            except AgentError as exc:
                result, error = None, getattr(exc, "code", AGENT_RUNTIME_FAILED)
            except (GitError, OSError) as exc:
                result, error = None, getattr(exc, "code", AGENT_RUNTIME_FAILED)

            if error is None:
                return result, None

            failed_tree_after = _safe_candidate_tree(worktree)
            changed = self._rollback_revision(
                transaction, allowed=allowed, artifact_dir=artifact_dir,
                allow_requested_paths=(error == _SCOPE_REQUEST_ROUTE),
                discard_scope_violations=is_check_repair,
            )
            if error in TRANSIENT_WORKER_FAILURES | {AGENT_AUTH_FAILURE}:
                atomic_write_text(artifact_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "reason": str(error)[:120],
                    "profile_id": active.profile_id,
                    "tree_before": tree_before,
                    "tree_after": failed_tree_after,
                    "mutable_scope": sorted(allowed),
                    "rollback_tree_sha": candidate_tree_sha(worktree),
                    "rollback_index_tree_sha": index_tree_sha(worktree),
                    "rollback_status": list(status_porcelain(worktree)),
                }))
            if error == AGENT_AUTH_FAILURE:
                raise PipelineFailure(
                    "EXTERNAL_AUTH_REQUIRED",
                    "executor credentials or external authorization are required",
                )
            if error not in TRANSIENT_WORKER_FAILURES:
                return result, error

            admission = self._recovery.admit(
                retry_key, reason=error, budget=self._budgets.max_transient_attempts,
                phase=phase, cycle=cycle, profile_id=active.profile_id,
                tree_before=tree_before, tree_after=candidate_tree_sha(worktree),
                facts={"tree_changed_in_scope": changed},
            )
            if admission.admitted:
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                continue
            if fallback_index < len(fallbacks):
                fallback_index += 1
                self._recovery.fallback_selected(
                    reason=error, index=fallback_index, available=len(fallbacks),
                    phase=phase, cycle=cycle, tree_before=tree_before,
                    tree_after=candidate_tree_sha(worktree),
                )
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                active = fallbacks[fallback_index - 1]
                retry_key = f"{base_key}:{active.profile_id}"
                continue
            return result, error

    def _rollback_revision(
        self,
        transaction: CandidateAttemptTransaction,
        *,
        allowed: set[str],
        artifact_dir: Path,
        allow_requested_paths: bool,
        discard_scope_violations: bool = False,
    ) -> bool:
        """Audit a failed revision and restore only a fully in-scope tree."""

        requested: set[str] = set()
        if allow_requested_paths:
            report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
            scope_request = report.get("scope_request") if isinstance(report, dict) else None
            paths = scope_request.get("paths") if isinstance(scope_request, dict) else None
            if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
                requested = set(paths)
        if any(not safe_scope_request_path(path) for path in requested):
            raise PipelineFailure("AGENT_SCOPE_VIOLATION", "scope request contains an unsafe path")
        try:
            if discard_scope_violations:
                rollback = transaction.discard()
                transaction.enforce_scope(rollback.changed_paths, set(allowed) | requested)
            else:
                rollback = transaction.abort(allowed, requested=requested)
        except AttemptViolation as violation:
            if violation.code == "RESUME_REQUIRES_OPERATOR":
                _record_failure_tree(artifact_dir, transaction.worktree)
            raise PipelineFailure(violation.code, violation.detail) from violation
        atomic_write_text(
            artifact_dir / "tree_after_failure.txt", transaction.boundary.tree + "\n",
        )
        return rollback.changed


__all__ = ["TRANSIENT_WORKER_FAILURES", "WorkerRecovery", "safe_scope_request_path"]
