"""Harness-owned deterministic gate and baseline evidence."""

from __future__ import annotations

import hashlib, time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    TYPE_CHECKING,
)
from ..evidence import (
    EvidenceBundle,
    collect_evidence,
    required_checks_passed,
)
from ..baseline import (
    BaselineCache,
    BaselineRecord,
    baseline_of_result,
    compare_check,
    junit_report_path,
    preflight_skips,
)
from ..validation import config_with_check_authority
from ..gitops import (
    GitError,
    commit_parents,
    candidate_tree_sha,
    current_head,
    index_tree_sha,
    resolve_tree,
)
from ..models import (
    GateStage,
    HarnessConfig,
)
from ..result import atomic_write_text
from ..state import RunStateStore
from .shared import (
    _CHECK_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _check_payload,
    _json_text,
    _read_json_artifact,
    _safe_candidate_tree,
)
from .per_step_gate import (
    PerStepGateOutcome,
    record_step_warnings,
    run_per_step_gate,
)
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    gate_dir,
    gate_acceptance_path,
)
from .durable_readers import (
    load_evidence,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime




class GateService:
    """One owner of the pipeline operations described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def run_per_step_gate(
        self,
        *,
        run_dir: Path,
        worktree: Path,
        base_sha: str,
        step_dir: Path,
        check_ids: Sequence[str],
        changed_paths: Sequence[str],
    ) -> PerStepGateOutcome:
        """The fast gate a step pays *before* its own commit.

        Only the configured ``gate.per_step`` checks run, only their *new*
        regressions count, and the worker receives the bounded evidence of a
        refusal -- never the whole log.
        """

        return run_per_step_gate(
            self, run_dir=run_dir, worktree=worktree, base_sha=base_sha,
            step_dir=step_dir, check_ids=check_ids, changed_paths=changed_paths,
        )

    # The fast gate is its own authority: this service only delegates to it.
    record_step_warnings = staticmethod(record_step_warnings)

    def run_gate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage,
    ) -> EvidenceBundle:
        """Run the authoritative deterministic checks for the current tree."""

        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        attempts_dir = directory / "attempts"
        archived_gate_attempts = sum(
            1 for item in attempts_dir.iterdir()
            if item.is_dir() and item.name.isdigit()
        ) if attempts_dir.is_dir() else 0
        gate_attempt = archived_gate_attempts + (1 if (directory / "evidence.json").is_file() else 0) + 1
        retry_check = self.runtime.check_recovery(store).gate_retries(
            cycle=cycle_plan.cycle.number, stage=stage.value, worktree=ctx.info.worktree,
        )

        _archive_attempt(directory, names=_CHECK_ATTEMPT_ARTIFACTS)
        store.update_metadata(current_step=None)
        evidence, baseline_payload = self._final_evidence(
            ctx.info.worktree, ctx.base_sha, directory, run_dir=ctx.run_dir,
            check_failures_hard=False, reuse=True, stage=stage,
            expected_head_sha=current_head(ctx.info.worktree),
            required_check_ids=cycle_plan.plan.required_checks or None,
            enforce_diff_size=False,
            retry_check_infrastructure=retry_check,
        )
        gate = {
            "stage": stage.value,
            "attempt": gate_attempt,
            "passed": evidence.deterministic_passed,
            "required_check_ids": list(evidence.required_check_ids),
            "failures": list(evidence.failures),
        }
        store.update_metadata(
            checks=_check_payload(evidence),
            staged_tree_sha=evidence.staged_tree_sha,
            changed_files=list(evidence.changed_files),
            deterministic_gate=gate,
            deterministic_gate_attempt=gate_attempt,
            baseline=baseline_payload,
        )
        self.runtime.cycle_update(store, cycle_plan.cycle, deterministic_gate=gate)
        retry_check.settle(evidence)
        return evidence
    @staticmethod
    def load_accepted_gate_evidence(
        ctx: PipelineV2Context, number: int, stage: GateStage,
    ) -> EvidenceBundle | None:
        """Return a green evidence bundle only when its gate acceptance binds it."""

        directory = gate_dir(ctx.run_dir, number, stage)
        acceptance_path = gate_acceptance_path(ctx.run_dir, number, stage)
        if not acceptance_path.is_file():
            return None
        try:
            payload = _read_json_artifact(acceptance_path)
            evidence_path = directory / "evidence.json"
            evidence_bytes = evidence_path.read_bytes()
            evidence = load_evidence(directory)
            current = current_head(ctx.info.worktree)
            current_tree = resolve_tree(ctx.info.worktree, current)
            parents = commit_parents(ctx.info.worktree, current)
        except (OSError, GitError, ValueError) as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "accepted gate evidence is unreadable",
            ) from exc
        digest = hashlib.sha256(evidence_bytes).hexdigest()
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in {1, 2}
            or payload.get("review_cycle") != number
            or payload.get("stage") != stage.value
            or evidence is None
            or evidence.base_sha != ctx.base_sha
            or not evidence.deterministic_passed
            or not required_checks_passed(evidence)
            or bool(evidence.failures)
            or payload.get("tree_sha") != evidence.staged_tree_sha
            or payload.get("commit_sha") != current
            or current_tree != evidence.staged_tree_sha
            or (
                payload.get("no_change") is not True
                and parents != (payload.get("parent_sha"),)
            )
            or not isinstance(payload.get("no_change", False), bool)
            or (
                payload.get("no_change") is True
                and (
                    payload.get("parent_sha") is not None
                    or evidence.diff != ""
                    or bool(evidence.changed_files)
                    or payload.get("commit_created") is not False
                )
            )
            or (
                payload.get("parent_sha") is None
                and (
                    payload.get("no_change") is not True
                    or bool(evidence.changed_files)
                )
            )
            or (
                payload.get("schema_version") == 2
                and payload.get("evidence_sha256") != digest
            )
        ):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "gate acceptance does not bind its evidence",
            )
        return evidence
    def run_check_preflights_recoverably(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Evaluate each trusted preflight once per run; return the skipped IDs."""

        return self.runtime.check_recovery(store).run_preflights(
            run_dir=run_dir, worktree=worktree, check_config=check_config,
            check_ids=check_ids, phase=phase, cycle=cycle,
        )
    def gate_baseline(
        self,
        *,
        run_dir: Path,
        worktree: Path,
        base_sha: str,
        check_config: HarnessConfig,
        checks: Sequence[Any],
        skipped: Mapping[str, str],
    ) -> BaselineRecord:
        """The run's baseline for one check subset, captured at most once.

        The baseline is always established *before* the subset is judged, so a
        gate never has to guess what the base commit already answered.
        """

        return BaselineCache(self.runtime.config.runs_root).ensure(
            repo=worktree, base_sha=base_sha, config=check_config,
            check_ids=[check.id for check in checks],
            environment=self.runtime.environment,
            setup_commands=self.runtime.config.workspace_setup,
            secrets=self.runtime.secrets, skipped=skipped,
        )

    def _final_evidence(
        self,
        worktree: Path,
        base_sha: str,
        evidence_dir: Path,
        *,
        run_dir: Path,
        check_failures_hard: bool,
        reuse: bool,
        expected_head_sha: str | None = None,
        required_check_ids: tuple[str, ...] | None = None,
        enforce_diff_size: bool = False,
        stage: GateStage | None = None,
        retry_check_infrastructure: Callable[[str, str], bool] | None = None,
    ) -> tuple[EvidenceBundle, dict[str, Any]]:
        """Final checks for the exact current candidate.

        With *reuse*, durable evidence already frozen for exactly this index
        tree is reused: checks are never replayed for a tree whose evidence is
        complete.  Every candidate check is judged against the baseline of the
        base commit: only a *new* regression can close the gate.
        """

        if reuse:
            stored = load_evidence(evidence_dir)
            if (
                stored is not None
                and stored.base_sha == base_sha
                and current_head(worktree) == (expected_head_sha or base_sha)
                and stored.staged_tree_sha == index_tree_sha(worktree)
                and stored.staged_tree_sha == candidate_tree_sha(worktree)
            ):
                self.runtime.observability.trace_emit(
                    "checks.completed",
                    phase="validation",
                    cycle=self.runtime.trace_cycle,
                    data={
                        "reused": True,
                        "stage": stage.value if stage is not None else None,
                        "passed": stored.deterministic_passed,
                        "failures": list(stored.failures),
                        "required_check_ids": list(stored.required_check_ids),
                        "tree_sha": stored.staged_tree_sha,
                        "changed_paths": list(stored.changed_files),
                    },
                )
                return stored, {}
        checks_started_at = self.runtime.observability.trace_time()
        checks_started_mono = time.perf_counter()
        checks_tree_before = _safe_candidate_tree(worktree)
        self.runtime.observability.trace_emit(
            "checks.started",
            phase="validation",
            cycle=self.runtime.trace_cycle,
            data={
                "stage": stage.value if stage is not None else None,
                "required_check_ids": list(required_check_ids or ()),
                "tree_before": checks_tree_before,
            },
        )
        check_config, check_ids = config_with_check_authority(
            self.runtime.config, evidence_dir, requested_check_ids=required_check_ids,
            expected_sha256=self.runtime.approved_check_authority_sha256(evidence_dir),
        )
        selected = check_config.select_checks(check_ids)
        skipped = preflight_skips(run_dir, selected)
        baseline = self.gate_baseline(
            run_dir=run_dir, worktree=worktree, base_sha=base_sha,
            check_config=check_config, checks=selected, skipped=skipped,
        )
        verdicts: list[dict[str, Any]] = []
        recorded: set[str] = set()

        def record(check: Any, result: Any) -> Any:
            """Compare one candidate check against its baseline, once."""

            candidate = baseline_of_result(
                check.id, result, junit_path=junit_report_path(check, result),
            )
            judgement = compare_check(check.id, candidate, baseline.entry(check.id))
            recorded.add(check.id)
            verdicts.append({
                "id": check.id,
                "verdict": judgement.verdict.value,
                "baseline_status": (
                    baseline.entry(check.id).status if baseline.entry(check.id) else None
                ),
                "candidate_status": candidate.status,
                "exit_code": candidate.exit_code,
                "failure_ids": list(candidate.failure_ids),
                "failure_ids_parsed": candidate.failure_ids_parsed,
                "new_failure_ids": list(judgement.new_failure_ids),
                "warning": judgement.warning,
            })
            return judgement

        def judge(check: Any, result: Any) -> tuple[str | None, str | None]:
            judgement = record(check, result)
            return judgement.failure, judgement.warning

        try:
            evidence = collect_evidence(
                worktree, base_sha, check_config, evidence_dir=evidence_dir,
                secrets=self.runtime.secrets, check_failures_hard=check_failures_hard,
                expected_head_sha=expected_head_sha, required_check_ids=check_ids,
                enforce_diff_size=enforce_diff_size,
                # Accepted step commits make the current HEAD itself the
                # candidate.  There is no staged diff against that HEAD, but the
                # authoritative checks still must run and their tree is exact.
                allow_empty_diff=(
                    stage is not None or current_head(worktree) != base_sha
                ),
                retry_check_infrastructure=retry_check_infrastructure,
                skip_checks=skipped, judge_check=judge,
            )
        except Exception as exc:
            self.runtime.observability.trace_emit(
                "checks.completed",
                phase="validation",
                cycle=self.runtime.trace_cycle,
                data={
                    "stage": stage.value if stage is not None else None,
                    "passed": False,
                    "failures": [type(exc).__name__],
                    "tree_sha": _safe_candidate_tree(worktree),
                    "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                },
            )
            raise
        self.runtime.observability.trace_emit(
            "checks.completed",
            phase="validation",
            cycle=self.runtime.trace_cycle,
            data={
                "stage": stage.value if stage is not None else None,
                "passed": evidence.deterministic_passed,
                "failures": list(evidence.failures),
                "required_check_ids": list(evidence.required_check_ids),
                "tree_sha": evidence.staged_tree_sha,
                "changed_paths": list(evidence.changed_files),
                "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                "started_at": checks_started_at,
            },
        )
        for check, result in zip(selected, evidence.checks):
            # A green candidate check is judged too: the durable baseline
            # artifact describes every check of this gate, never only the red
            # ones the failure callback happens to see.
            if check.id not in recorded:
                record(check, result)
        payload = {
            "schema_version": 1,
            "base_sha": baseline.base_sha,
            "check_config_sha": baseline.check_config_sha,
            "baseline_unavailable_reason": baseline.unavailable_reason,
            "checks": verdicts,
            "warnings": list(evidence.warnings),
        }
        atomic_write_text(evidence_dir / "baseline.json", _json_text(payload))
        return evidence, payload
