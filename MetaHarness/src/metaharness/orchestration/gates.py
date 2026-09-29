"""Harness-owned deterministic gate and baseline evidence."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ..approval import ApprovalError, read_check_authority
from ..baseline import (
    BaselineCache,
    BaselineRecord,
    baseline_of_result,
    compare_check,
    judge_results,
    junit_report_path,
    preflight_skips,
)
from ..evidence import EvidenceBundle, collect_evidence, required_checks_passed
from ..gitops import (
    candidate_tree_sha,
    commit_parents,
    current_head,
    index_tree_sha,
    resolve_tree,
)
from ..models import GateStage, HarnessConfig
from ..recovery_policy import FailureClass, classify_failure
from ..result import atomic_write_text
from ..state import RunStateStore
from ..validation import (
    ValidationError,
    bounded_tail,
    config_with_check_authority,
    run_checks,
)
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    gate_acceptance_path,
    gate_dir,
)
from .shared import (
    _CHECK_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _check_payload,
    _json_text,
    _safe_candidate_tree,
    gate_mutable_authority,
    is_object_id,
    json_text,
    load_evidence,
    read_json_artifact,
)

if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


if TYPE_CHECKING:  # pragma: no cover - the service is the composition root
    from .gates import GateService


_FEEDBACK_EXCERPT_BYTES = 2000


@dataclass(frozen=True)
class PerStepGateOutcome:
    """What the optional fast gate answered for one mutated step tree."""

    enabled: bool
    passed: bool
    feedback: str = ""
    warnings: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()


def _excerpt_text(result: Any) -> str:
    """The bounded log text of one check result, whatever shape it has."""

    stdout = str(getattr(result, "stdout_tail", "") or "")
    stderr = str(getattr(result, "stderr_tail", "") or "")
    return f"{stdout}\n{stderr}".strip()


def _regression_feedback(
    *,
    regressions: Any,
    results: Any,
    changed_paths: Any,
) -> str:
    """The bounded evidence one refused step hands back to its worker."""

    by_id = {result.name: result for result in results}
    lines = ["The fast per-step gate refused this step tree."]
    # Put durable path facts before the bounded log excerpt. The retry
    # addendum has its own size limit and should not clip this actionable data.
    if changed_paths:
        lines.append("CHANGED PATHS:")
        lines.extend(f"- {path}" for path in tuple(changed_paths)[:20])
    lines.append("Fix every regression listed above, including in files outside WRITE_SET that this step's change broke; MetaHarness admits and reports those paths.")
    for judgement in regressions:
        lines.append(f"CHECK: {judgement.check_id}")
        if judgement.new_failure_ids:
            lines.append("NEW FAILING TEST IDS:")
            lines.extend(f"- {item}" for item in judgement.new_failure_ids[:10])
        result = by_id.get(judgement.check_id)
        if result is not None:
            excerpt = bounded_tail(_excerpt_text(result), _FEEDBACK_EXCERPT_BYTES).strip()
            if excerpt:
                lines.append("FAILURE EXCERPT:")
                lines.append(excerpt)
    return "\n".join(lines)


def record_step_warnings(store: RunStateStore, warnings: Any) -> None:
    """Keep the non-blocking verdicts of the fast gate durable."""

    state = store.load()
    existing = [
        item for item in (state.get("check_warnings") or []) if isinstance(item, str)
    ]
    merged = list(dict.fromkeys((*existing, *warnings)))
    if merged != existing:
        store.update_metadata(check_warnings=merged)


def per_step_check_ids(runtime: Any, run_dir: Path) -> tuple[str, ...]:
    """Keep live gate additions outside an older run's approved catalogue."""

    store = RunStateStore(run_dir / "state.json")
    state = store.load()
    if "per_step_check_ids" in state:
        ids = state["per_step_check_ids"]
        if (
            not isinstance(ids, list)
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValidationError("the run's per-step check policy is invalid")
        return tuple(ids)
    # Legacy runs have no gate snapshot. Retain only live selections that
    # were already in their original catalogue; never import today's argv.
    try:
        policy = read_check_authority(
            run_dir, expected_sha256=runtime.approved_check_authority_sha256(run_dir),
        )
    except ApprovalError as exc:
        raise ValidationError(str(exc)) from exc
    if policy is None:
        raise ValidationError("the run has no check authority")
    approved = policy.by_id()
    requested = runtime.config.gate.per_step
    omitted = tuple(item for item in requested if item not in approved)
    if omitted:
        record_step_warnings(store, (
            "Per-step check excluded because it was added after this run's "
            "check authority was frozen: " + item for item in omitted
        ))
    return tuple(item for item in requested if item in approved)


def run_per_step_gate(
    service: "GateService",
    *,
    run_dir: Path,
    worktree: Path,
    base_sha: str,
    step_dir: Path,
    check_ids: Any,
    changed_paths: Any,
) -> PerStepGateOutcome:
    """Run the configured fast checks and judge only their *new* failures."""

    if not check_ids:
        return PerStepGateOutcome(enabled=False, passed=True)
    runtime = service.runtime
    check_config, _ids = config_with_check_authority(
        runtime.config, run_dir, requested_check_ids=list(check_ids),
        expected_sha256=runtime.approved_check_authority_sha256(run_dir),
    )
    selected = check_config.select_checks(list(check_ids))
    skipped = preflight_skips(run_dir, selected)
    baseline = service.gate_baseline(
        run_dir=run_dir, worktree=worktree, base_sha=base_sha,
        check_config=check_config, checks=selected, skipped=skipped,
    )
    directory = step_dir / "per-step-gate"
    directory.mkdir(parents=True, exist_ok=True)
    evaluated = tuple(check for check in selected if check.id not in skipped)
    results = run_checks(
        worktree, check_config, required_check_ids=[check.id for check in evaluated],
        logs_dir=directory / "checks", secrets=runtime.secrets,
    )
    judgements = judge_results(baseline, results, evaluated, skipped=skipped)
    regressions = tuple(item for item in judgements if item.blocking)
    warnings = tuple(item.warning for item in judgements if item.warning)
    payload = {
        "schema_version": 1,
        "step_dir": step_dir.name,
        "check_ids": list(check_ids),
        "baseline_check_config_sha": baseline.check_config_sha,
        "regressions": [item.check_id for item in regressions],
        "checks": [
            {
                "id": item.check_id,
                "verdict": item.verdict.value,
                "new_failure_ids": list(item.new_failure_ids),
                "warning": item.warning,
            }
            for item in judgements
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(directory / f"attempt-{_attempt_index(step_dir):02d}.json", text)
    atomic_write_text(directory / "per-step-gate.json", text)
    if not regressions:
        return PerStepGateOutcome(
            enabled=True, passed=True, warnings=warnings, regressions=(),
        )
    feedback = _regression_feedback(
        regressions=regressions, results=results, changed_paths=changed_paths,
    )
    return PerStepGateOutcome(
        enabled=True, passed=False, feedback=feedback, warnings=warnings,
        regressions=tuple(item.check_id for item in regressions),
    )


def _attempt_index(step_dir: Path) -> int:
    """The 1-based index of the worker attempt this gate just judged."""

    root = step_dir / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    return index


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


def hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if isinstance(item, str)
        and classify_failure(item.split(":", 1)[0]).failure_class is FailureClass.FATAL
    ]


def hard_integrity_failures(bundle: EvidenceBundle) -> list[str]:
    return hard_failure_items(bundle.failures)


class GateAcceptanceService:
    """Persist and validate the green tree accepted by one gate episode."""

    def __init__(
        self, *, authorize_candidate_tree: Callable[..., None],
        trace_emit: Callable[..., None],
    ) -> None:
        self._authorize_candidate_tree = authorize_candidate_tree
        self._trace_emit = trace_emit

    def accept(
        self, store: Any, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: EvidenceBundle, *, base_paths: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if (
            not evidence.deterministic_passed
            or not required_checks_passed(evidence)
            or evidence.staged_tree_sha is None
        ):
            raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
        no_change = not evidence.changed_files
        if no_change and (evidence.diff != "" or evidence.base_sha != ctx.base_sha):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change evidence is not bound to the run base",
            )
        worktree = ctx.info.worktree
        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = gate_acceptance_path(ctx.run_dir, cycle_plan.cycle, stage)
        authority = gate_mutable_authority(
            ctx.run_dir, cycle_plan.cycle.number, stage,
            base_paths=(
                cycle_plan.mutable_scope if base_paths is None else base_paths
            ),
        )
        stored = read_json_artifact(path) if path.is_file() else None
        if path.is_file() and stored is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance is corrupted")
        if stored is not None:
            stored_no_change = stored.get("no_change", False) if isinstance(stored, dict) else False
            stored_parent = stored.get("parent_sha") if isinstance(stored, dict) else None
            parent_valid = is_object_id(stored_parent) or (
                stored_no_change is True
                and stored_parent is None
                and not evidence.changed_files
            )
            evidence_sha256 = self._durable_evidence_sha256(directory)
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 2
                or stored.get("review_cycle") != cycle_plan.cycle.number
                or not all(is_object_id(stored.get(key)) for key in ("tree_sha", "commit_sha"))
                or not isinstance(stored_no_change, bool)
                or not parent_valid
                or stored_no_change is not (not evidence.changed_files)
                or (
                    stored_no_change
                    and (
                        stored_parent is not None
                        or stored.get("commit_created") is not False
                        or stored.get("acceptance_kind") != "existing-head"
                    )
                )
                or stored.get("stage") != stage.value
                or stored.get("acceptance_kind") != "existing-head"
                or not isinstance(stored.get("commit_created"), bool)
                or stored.get("tree_sha") != evidence.staged_tree_sha
                or stored.get("mutable_scope") != list(authority.effective_paths)
                or stored.get("mutable_scope_sha256") != authority.sha256
                or stored.get("evidence_sha256") != evidence_sha256
            ):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance does not match evidence")
            if current_head(worktree) != stored["commit_sha"]:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted gate HEAD moved")
            if (
                resolve_tree(worktree, stored["commit_sha"]) != stored["tree_sha"]
                or (
                    not stored_no_change
                    and commit_parents(worktree, stored["commit_sha"]) != (stored["parent_sha"],)
                )
            ):
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", "gate acceptance does not match its commit",
                )
            self._emit_acceptance(ctx, cycle_plan, stage, stored)
            return stored

        self._authorize_candidate_tree(
            evidence, worktree, current_head(worktree), ctx.branch_ref,
        )
        head = current_head(worktree)
        current_tree = resolve_tree(worktree, head)
        if no_change and current_tree != evidence.staged_tree_sha:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change HEAD tree differs from gate evidence",
            )
        if current_tree != evidence.staged_tree_sha:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "gate evidence differs from the committed audit tree",
            )
        parents = () if no_change else commit_parents(worktree, head)
        if not no_change and len(parents) != 1:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted HEAD has no single parent")
        commit_sha, parent_sha = head, None if no_change else parents[0]
        acceptance_kind = "existing-head"
        commit_created = False

        acceptance = {
            "schema_version": 2,
            "review_cycle": cycle_plan.cycle.number,
            "stage": stage.value,
            "tree_sha": evidence.staged_tree_sha,
            "commit_sha": commit_sha,
            "parent_sha": parent_sha,
            "no_change": not evidence.changed_files,
            "commit_created": commit_created,
            "acceptance_kind": acceptance_kind,
            "mutable_scope": list(authority.effective_paths),
            "mutable_scope_sha256": authority.sha256,
            "evidence_sha256": self._durable_evidence_sha256(directory),
        }
        atomic_write_text(path, json_text(acceptance))
        store.update_metadata(
            approved_tree_sha=evidence.staged_tree_sha,
            expected_head_sha=commit_sha,
            expected_parent_sha=parent_sha,
            expected_tree_sha=evidence.staged_tree_sha,
        )
        self._emit_acceptance(ctx, cycle_plan, stage, acceptance)
        return acceptance

    @staticmethod
    def _durable_evidence_sha256(directory: Path) -> str:
        try:
            return hashlib.sha256((directory / "evidence.json").read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "accepted gate evidence is unreadable",
            ) from exc

    def _emit_acceptance(self, ctx: Any, cycle_plan: Any, stage: GateStage, payload: Mapping[str, Any]) -> None:
        self._trace_emit(
            "gate.accepted",
            phase="validation",
            cycle=cycle_plan.cycle.number,
            data={
                "stage": stage.value,
                "parent_sha": payload["parent_sha"],
                "commit_sha": payload["commit_sha"],
                "tree_sha": payload["tree_sha"],
                "commit_created": payload.get("commit_created"),
                "acceptance_kind": payload.get("acceptance_kind"),
            },
        )


__all__ = ['PerStepGateOutcome', 'record_step_warnings', 'run_per_step_gate']
