"""Recovery of trusted check infrastructure: workspace setup, preflights, checks.

Infrastructure failures are retried against the same candidate tree only.
Every trusted process is contained by the shared attempt transaction, so a
retry can never observe a mutation of an earlier attempt.  Exhausted budgets
become ``CHECK_INFRASTRUCTURE_UNAVAILABLE``, a waiting condition.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..attempt_transaction import AttemptViolation, contain_trusted_process
from ..baseline import PREFLIGHT_FILE, PREFLIGHT_SCHEMA_VERSION, preflight_fingerprint
from ..evidence import EvidenceBundle
from ..gitops import snapshot_candidate_state
from ..models import CheckConfig, HarnessConfig
from ..recovery_policy import RecoveryBudgets
from ..result import ResultArtifactError, atomic_write_text
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
                self._store.update_metadata(
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
        run_dir: Path,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Evaluate every trusted preflight exactly once per run.

        The verdict is durable, so Docker availability is never probed again on
        a resume or a later cycle.  A failed preflight means the check does not
        run: its ID is returned for the gate to record as ``SKIPPED_INFRA``
        with a durable warning.  Only a check that explicitly declares itself
        ``blocking`` turns that condition into the external wait the ladder
        already owns.
        """

        try:
            selected = check_config.select_checks(check_ids)
        except ValueError as exc:
            raise OrchestrationError("CHECK_PREFLIGHT_FAILED: trusted check selection invalid") from exc
        store = _PreflightStore(run_dir)
        skipped: list[str] = []
        warnings: list[str] = []
        for check in selected:
            if not check.preflight_argv:
                continue
            fingerprint = preflight_fingerprint(check)
            verdict = store.verdict(check.id, fingerprint)
            if verdict is None:
                verdict = self._evaluate_preflight(
                    worktree=worktree, check_config=check_config, check=check,
                )
                store.record(check.id, fingerprint, verdict)
            if verdict["status"] == "PASS":
                continue
            if check.blocking:
                raise OrchestrationError(
                    "CHECK_INFRASTRUCTURE_UNAVAILABLE: CHECK_PREFLIGHT_FAILED:"
                    f"{check.id}; the check declares itself blocking"
                )
            skipped.append(check.id)
            warnings.append(
                f"check:{check.id}:PREFLIGHT_FAILED: skipped with SKIPPED_INFRA"
            )
        if warnings:
            self._store.update_metadata(check_warnings=warnings)
        return tuple(skipped)

    def _evaluate_preflight(
        self, *, worktree: Path, check_config: HarnessConfig, check: CheckConfig,
    ) -> Mapping[str, Any]:
        """Run one preflight once, contained, and describe its verdict."""

        before = snapshot_candidate_state(worktree)
        failures = run_check_preflights(worktree, check_config, [check.id])
        after = snapshot_candidate_state(worktree)
        if before != after:
            # run_check_preflights restores its own side effects.
            raise OrchestrationError(
                "ROLLBACK_TREE_MISMATCH: preflight did not preserve candidate state"
            )
        if not failures:
            return {"status": "PASS", "exit_code": 0, "detail": ""}
        failure = failures[0]
        if not failure.startswith("CHECK_PREFLIGHT_FAILED:"):
            # A preflight that mutates the candidate is an integrity problem,
            # never an infrastructure condition to skip.
            raise OrchestrationError(failure)
        return {"status": "FAIL", "exit_code": -1, "detail": failure}

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


class _PreflightStore:
    """The durable, tolerant cache of one run's preflight verdicts."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / PREFLIGHT_FILE

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
            or not isinstance(payload.get("checks"), dict)
        ):
            return {}
        return payload["checks"]

    def verdict(self, check_id: str, fingerprint: str) -> Mapping[str, Any] | None:
        """The cached verdict of one preflight, never a stale one."""

        entry = self._load().get(check_id)
        if (
            not isinstance(entry, dict)
            or entry.get("fingerprint") != fingerprint
            or entry.get("status") not in {"PASS", "FAIL"}
            or not isinstance(entry.get("exit_code"), int)
        ):
            return None
        return entry

    def record(self, check_id: str, fingerprint: str, verdict: Mapping[str, Any]) -> None:
        checks = self._load()
        checks[check_id] = {
            "status": verdict["status"], "exit_code": verdict["exit_code"],
            "detail": str(verdict.get("detail", ""))[:200], "fingerprint": fingerprint,
        }
        try:
            atomic_write_text(self.path, json.dumps({
                "schema_version": PREFLIGHT_SCHEMA_VERSION, "checks": checks,
            }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        except (OSError, ResultArtifactError) as exc:
            raise OrchestrationError(
                f"DURABLE_ARTIFACT_CORRUPTED: could not record preflight verdicts: {exc}"
            ) from None
