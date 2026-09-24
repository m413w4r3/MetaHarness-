"""Recovery of trusted check infrastructure: workspace setup, preflights, checks.

Infrastructure failures are retried against the same candidate tree only.
Every trusted process is contained by the shared attempt transaction, so a
retry can never observe a mutation of an earlier attempt.  Exhausted budgets
become ``CHECK_INFRASTRUCTURE_UNAVAILABLE``, a waiting condition.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..attempt_transaction import AttemptViolation, contain_trusted_process
from ..evidence import EvidenceBundle
from ..gitops import snapshot_candidate_state
from ..models import HarnessConfig, RunStatus
from ..recovery_policy import RecoveryBudgets
from ..state import RunStateStore
from ..validation import run_check_preflights
from ..workspace import WorkspaceSetupError
from .pipeline_v2 import PipelineFailure
from .recovery import RecoveryAdmission, RecoveryCoordinator
from .shared import OrchestrationError, _archive_attempt_tree, _safe_candidate_tree

_INFRA_FAILURE_PREFIXES = (
    "CHECK_TIMEOUT:", "CHECK_INFRA_FAILURE:",
    "CHECK_SIDE_EFFECT_REPEATED:", "CHECK_SIDE_EFFECT_UNSTABLE:",
)


def _field(check_result: Any, name: str, default: Any = None) -> Any:
    if isinstance(check_result, Mapping):
        return check_result.get(name, default)
    return getattr(check_result, name, default)


class CheckInfrastructureRecovery:
    """Bounded same-tree retries of trusted check infrastructure."""

    def __init__(
        self, recovery: RecoveryCoordinator, *, store: RunStateStore, budgets: RecoveryBudgets,
    ) -> None:
        self._recovery = recovery
        self._store = store
        self._budgets = budgets

    def prepare_workspace(
        self, *, worktree: Path, run_dir: Path, run_setup: Callable[[], Sequence[Any]],
    ) -> Sequence[Any]:
        """Run trusted workspace setup; retry transient failures exactly."""

        key = self._recovery.budget_key("workspace-setup")
        budget = self._budgets.max_workspace_setup_retries
        pending: RecoveryAdmission | None = None
        while True:
            before = snapshot_candidate_state(worktree)
            try:
                results = run_setup()
                error: WorkspaceSetupError | None = None
            except WorkspaceSetupError as exc:
                results, error = exc.results, exc
            try:
                after = contain_trusted_process(worktree, before, label="workspace setup")
            except AttemptViolation as violation:
                raise OrchestrationError(f"{violation.code}: {violation.detail}") from None
            tree_after = (after.after if after else before).candidate_tree
            if error is None:
                if pending is not None:
                    self._recovery.complete(pending, recovered=True, tree_after=tree_after)
                return results
            if error.results:
                self._store.update(
                    status=RunStatus.PREPARING,
                    workspace_setup=[asdict(result) for result in error.results],
                )
            admission = self._recovery.admit(
                key, reason=error.code, budget=budget, phase="workspace setup",
                tree_before=before.candidate_tree, tree_after=tree_after,
            )
            if not admission.admitted:
                raise OrchestrationError(
                    f"CHECK_INFRASTRUCTURE_UNAVAILABLE: {error.code}; "
                    "workspace setup retry budget exhausted"
                ) from error
            _archive_attempt_tree(run_dir / "setup")
            pending = admission

    def run_preflights(
        self,
        *,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        counter_key: str,
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Retry each trusted preflight itself under a durable infra budget."""

        budget = self._budgets.max_check_infra_retries
        try:
            selected = check_config.select_checks(check_ids)
        except ValueError as exc:
            raise OrchestrationError("CHECK_PREFLIGHT_FAILED: trusted check selection invalid") from exc
        for check in selected:
            if not check.preflight_argv:
                continue
            key = self._recovery.budget_key(counter_key, check.id)
            while True:
                before = snapshot_candidate_state(worktree)
                failures = run_check_preflights(worktree, check_config, [check.id])
                after = snapshot_candidate_state(worktree)
                if before != after:
                    # run_check_preflights restores its own side effects.
                    raise OrchestrationError(
                        "ROLLBACK_TREE_MISMATCH: preflight did not preserve candidate state"
                    )
                if not failures:
                    break
                failure = failures[0]
                reason = (
                    "CHECK_PREFLIGHT_FAILED"
                    if failure.startswith("CHECK_PREFLIGHT_FAILED:")
                    else "CHECK_INFRA_FAILURE"
                )
                admission = self._recovery.admit(
                    key, reason=f"{reason}:{check.id}", budget=budget, phase=phase,
                    cycle=cycle, tree_before=before.candidate_tree,
                    tree_after=after.candidate_tree,
                )
                if not admission.admitted:
                    raise OrchestrationError(
                        f"CHECK_INFRASTRUCTURE_UNAVAILABLE: {failure}; retry budget exhausted"
                    )
        return ()

    def gate_retries(self, *, cycle: int, stage: str, worktree: Path) -> "GateInfraRetries":
        return GateInfraRetries(self, cycle=cycle, stage=stage, worktree=worktree)


class GateInfraRetries:
    """The same-tree infrastructure retry callback of one deterministic gate."""

    def __init__(
        self, owner: CheckInfrastructureRecovery, *, cycle: int, stage: str, worktree: Path,
    ) -> None:
        self._owner = owner
        self._cycle = cycle
        self._stage = stage
        self._worktree = worktree
        self._admissions: dict[str, RecoveryAdmission] = {}

    def __call__(self, check_id: str, failure_kind: str) -> bool:
        recovery = self._owner._recovery
        reason = "CHECK_TIMEOUT" if failure_kind == "timeout" else "CHECK_INFRA_FAILURE"
        tree = _safe_candidate_tree(self._worktree)
        admission = recovery.admit(
            recovery.budget_key("check-infra", f"{self._cycle:03d}", self._stage, check_id),
            reason=f"{reason}:{check_id}", budget=self._owner._budgets.max_check_infra_retries,
            phase="validation", cycle=self._cycle, tree_before=tree, tree_after=tree,
        )
        self._admissions[check_id] = admission
        return admission.admitted

    def settle(self, evidence: EvidenceBundle) -> None:
        """Trace retried checks and raise the waiting condition of the gate."""

        recovery = self._owner._recovery
        tree = evidence.staged_tree_sha
        infra_failures = [
            item for item in evidence.failures if item.startswith(_INFRA_FAILURE_PREFIXES)
        ]
        for check_result in evidence.checks:
            check_id = _field(check_result, "name")
            retries = _field(check_result, "infrastructure_retries", 0)
            if (
                isinstance(check_id, str) and isinstance(retries, int) and retries > 0
                and _field(check_result, "failure_kind") in {"passed", "nonzero_exit"}
                and check_id in self._admissions
            ):
                recovery.complete(self._admissions[check_id], recovered=True, tree_after=tree)
        for item in infra_failures:
            admission = self._admissions.get(item.split(":", 1)[1])
            if admission is not None:
                recovery.complete(admission, recovered=False, tree_after=tree)
        if infra_failures:
            waiting_reason = (
                "CHECK_SIDE_EFFECT_REPEATED"
                if all(item.startswith("CHECK_SIDE_EFFECT_REPEATED:") for item in infra_failures)
                else "CHECK_INFRASTRUCTURE_UNAVAILABLE"
            )
            raise PipelineFailure(waiting_reason, ", ".join(infra_failures))


__all__ = ["CheckInfrastructureRecovery", "GateInfraRetries"]
