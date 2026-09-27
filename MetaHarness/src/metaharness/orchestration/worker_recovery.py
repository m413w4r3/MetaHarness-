"""Recovery of untrusted implementer attempts.

A failed worker attempt is first proven harmless by the shared
:class:`~metaharness.attempt_transaction.CandidateAttemptTransaction`; only
then does the :class:`RecoveryCoordinator` decide whether the same executor
is retried or a configured fallback executor is selected.  Model semantics stay with callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
)
from ..attempt_transaction import (
    AttemptBoundary,
    AttemptViolation,
    CandidateAttemptTransaction,
    GitOwnership,
)
from ..models import ImplementationStep
from ..recovery_policy import RecoveryBudgets, RecoveryStrategy, classify_failure
from ..scope import ScopePolicy
from ..state import RunStateStore
from .recovery import RecoveryAdmission, RecoveryCoordinator
from .shared import (
    StepExecutionFailure,
    _archive_attempt,
    _safe_candidate_tree,
    _safe_status,
)

TRANSIENT_WORKER_FAILURES = frozenset({
    AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AGENT_PROTOCOL_FAILED,
})
# The only ladder step this recovery loop may execute: a same-executor retry on
# the exact pre-attempt tree it just restored.
_RETRY_STRATEGIES = frozenset({RecoveryStrategy.RETRY_TARGETED})
# Failures the semantic contract route of the step owns instead.
_CONTRACT_ROUTE = frozenset({"AGENT_CONTRACT_MISMATCH", "AGENT_NO_CHANGE"})


class WorkerRecovery:
    """Transient-failure recovery of one run's untrusted worker attempts."""

    def __init__(
        self,
        recovery: RecoveryCoordinator,
        *,
        store: RunStateStore,
        budgets: RecoveryBudgets,
        secrets: Sequence[str],
        scope: ScopePolicy,
    ) -> None:
        self._recovery = recovery
        self._store = store
        self._budgets = budgets
        self._secrets = tuple(secrets)
        self._scope = scope

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
        if reason in _CONTRACT_ROUTE or classify_failure(reason).strategy.terminal:
            # The contract route owns mismatches; a terminal leaves the loop.
            return None

        def refuse(code: str, detail: str, tree_after: str | None) -> None:
            if code == "RESUME_REQUIRES_OPERATOR":
                failure.detail = (failure.detail or "worker failed") + "; " + detail
            else:
                failure.detail = detail
            failure.reason = code
            failure.tree_after = tree_after
            failure.step_dir = artifact_dir
            self._recovery.stop(
                code, attempt=attempt, tree_before=before, tree_after=tree_after, **trace,
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
            transaction.abort(
                set((*step.write_set, *step.create_set, *step.delete_set)),
            )
        except AttemptViolation as violation:
            refuse(violation.code, violation.detail, _safe_candidate_tree(worktree))
            return None

        failure.tree_after = before
        failure.index_tree_after = before
        if reason == AGENT_AUTH_FAILURE:
            # The same credentials can never succeed: an external change is due.
            failure.step_dir = artifact_dir
            return None
        admission = self._recovery.admit(
            retry_key, reason=reason, budget=self._budgets.max_transient_attempts,
            profile_id=failure.profile_id, tree_before=before, tree_after=before,
            allowed=_RETRY_STRATEGIES, **trace,
        )
        if not admission.admitted:
            if admission.exhausted:
                failure.detail = (
                    (failure.detail or "agent failure") + "; retry budget exhausted"
                )
            failure.step_dir = artifact_dir
            return None
        self._store.update_metadata(current_step=step.id)
        _archive_attempt(artifact_dir)
        return admission



__all__ = ["TRANSIENT_WORKER_FAILURES", "WorkerRecovery"]
