"""The revision sub-domain: semantic revision and check-repair passes."""

from __future__ import annotations

import dataclasses
import json

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)
from .shared import (
    CheckRepairScope,
    _PROMPTS_DIR,
    _bounded_report,
    bounded_v2_report,
    _check_payload,
    _git_ownership,
    _json_text,
    _ownership_violations,
    _read_bounded_text,
    _record_failure_tree,
)
from ..evidence import (
    EvidenceBundle,
    collect_evidence,
    required_checks_passed,
)
from ..gitops import (
    GitError,
    RepositoryReference,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    restore_paths_from_tree,
    stage_all,
    status_porcelain,
)
from ..models import (
    ExecutionSelection,
    ExecutionRole,
    HarnessConfig,
    ImplementationStep,
    TaskPlanV2,
)
from ..prompt_contracts import (
    build_semantic_revision_payload,
    write_prompt_diagnostics,
)
from ..redaction import redact
from ..result import atomic_write_text
from ..run_options import EffectiveRepairScopePolicy
from ..state import RunStateStore
from ..usage import normalize_usage
from ..validation import config_with_check_authority
from ..review import ReviewResult
from .pipeline_v2 import (
    check_repair_root,
    correction_dir,
    review_dir,
    semantic_revision_dir,
)
from ..agent.base import (
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_SCOPE_REQUEST,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
)
from ..agent.protocol import (
    ScopeRequest,
    parse_check_repair_result,
    parse_scope_request,
)


_MAX_PREVIOUS_REVISION_REPORT_BYTES = 16 * 1024


_SCOPE_REQUEST_HEADER = "META SCOPE REQUEST v1"


SCOPE_REQUEST_ROUTE = AGENT_SCOPE_REQUEST


def _bounded_previous_revision_report(text: str) -> str:
    """Bound the advisory revision report handed to the correction planner.

    The report is consultative; the immutable candidate commit is the code
    authority, so truncating it can never hide a fact the planner must know.
    """

    data = text.encode("utf-8", errors="replace")

    if len(data) <= _MAX_PREVIOUS_REVISION_REPORT_BYTES:
        return text

    marker = (
        "\n[... revision report truncated for correction planner; "
        "candidate commit is authoritative ...]\n"
    ).encode("utf-8")

    head = data[
        : max(
            0,
            _MAX_PREVIOUS_REVISION_REPORT_BYTES - len(marker),
        )
    ].decode("utf-8", errors="ignore")

    return head + marker.decode("utf-8")


def _scope_request_payload(request: ScopeRequest) -> dict[str, Any]:
    return {
        "reason": request.reason,
        "paths": list(request.paths),
        "evidence": list(request.evidence),
        "authoritative": False,
        "requires_scope_decision": True,
    }


def _scope_request_diagnostic(request: ScopeRequest) -> str:
    return "\n".join([
        "The revision worker requested scope expansion:",
        f"  paths: {len(request.paths)}",
        "  authoritative: NO",
        "  requires operator decision: YES",
    ])


def _revision_execution_anomalies(results: list[dict[str, Any]]) -> str:
    """Render only exceptional step execution facts for the semantic reviser.

    The resulting worktree is the authority for successful implementation
    details.  Reports are retained for reviewer/audit flows, but normal worker
    narration, usage and tree metadata do not belong in the revision prompt.
    """

    records: list[dict[str, Any]] = []
    for item in results:
        status = item.get("status", "COMPLETED")
        exceptional = (
            status != "COMPLETED"
            or bool(item.get("mismatch"))
            or bool(item.get("initial_mismatch"))
            or bool(item.get("deferred_verify"))
            or bool(item.get("mismatch_retry_count"))
        )
        if not exceptional:
            continue

        record: dict[str, Any] = {
            "id": item.get("id"),
            "status": status,
            "changed_paths": list(item.get("changed_paths") or []),
        }
        for key in ("mismatch", "initial_mismatch", "deferred_verify"):
            if item.get(key):
                record[key] = bounded_v2_report(str(item[key]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        records.append(record)

    return _json_text(records) if records else "NONE\n"


def _revision_contract_index(plan: TaskPlanV2) -> str:
    """Render a compact deterministic index of approved step contracts."""

    lines: list[str] = []
    for step in plan.steps:
        writes = ",".join((*step.write_set, *step.create_set)) or "NONE"
        deletes = ",".join(step.delete_set) or "NONE"
        lines.extend((
            f"{step.id} | title={step.title} | writes={writes} | deletes={deletes}",
            f"  objective={step.objective}",
            f"  invariants={step.forbidden}",
            f"  verify={step.verify}",
        ))
    return "\n".join(lines) + ("\n" if lines else "NONE\n")


def _deferred_contract_mismatches(
    plan: TaskPlanV2, results: list[dict[str, Any]],
) -> str:
    """Render bounded summaries for the semantic reviser and the reviewer."""

    steps = {step.id: step for step in plan.steps}
    summaries: list[dict[str, Any]] = []
    for item in results:
        deferred = item.get("status") == "DEFERRED_CONTRACT_MISMATCH"
        verify = bounded_v2_report(str(item.get("deferred_verify") or ""))
        if not deferred and not verify:
            continue
        step = steps.get(item.get("id"))
        if step is None:
            continue
        scope = {
            "write": list(step.write_set),
            "create": list(step.create_set),
            "delete": list(step.delete_set),
        }
        record: dict[str, Any] = {
            "step_id": step.id,
            "step_title": step.title,
            "kind": (
                "DEFERRED_CONTRACT_MISMATCH" if deferred
                else "DEFERRED_VERIFY_DEPENDENCY"
            ),
            "original_scope": scope,
        }
        if deferred:
            record["mismatch"] = bounded_v2_report(str(item.get("mismatch") or ""))
            record["tree_at_mismatch"] = item.get("tree_before")
        if item.get("initial_mismatch"):
            record["initial_mismatch"] = bounded_v2_report(str(item["initial_mismatch"]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        if verify:
            # The step completed inside its approved scope but one VERIFY
            # command still fails on a path a later step owns.  The reviser
            # and the reviewer must decide; nothing here accepts that failure.
            record["deferred_verify_dependency"] = verify
        summaries.append(record)
    return _json_text(summaries) if summaries else "NONE\n"


def future_step_ownership(
    steps: Sequence[ImplementationStep], index: int,
) -> dict[str, tuple[str, ...]]:
    """The mutation paths the approved plan assigns to the remaining steps.

    Informative only: a bounded retry uses it to recognize an out-of-scope
    verification dependency.  It never grants write authority.
    """

    ownership: dict[str, tuple[str, ...]] = {}
    for step in steps[index + 1:]:
        paths = tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))
        if paths:
            ownership[step.id] = paths
    return ownership


def has_deferred_contract_mismatches(results: list[dict[str, Any]]) -> bool:
    return any(item.get("status") == "DEFERRED_CONTRACT_MISMATCH" for item in results)


def _review_payload(review: ReviewResult) -> dict[str, Any]:
    payload = dataclasses.asdict(review)
    payload["verdict"] = review.verdict.value
    payload["route"] = review.route.value
    payload.pop("raw", None)
    return payload


def _review_step_reports_text(results: list[dict[str, Any]]) -> str:
    records: list[dict[str, Any]] = []
    for item in results:
        record: dict[str, Any] = {
            "id": item.get("id"),
            "status": item.get("status", "COMPLETED"),
            "changed_paths": list(item.get("changed_paths") or []),
        }
        for key in ("mismatch", "initial_mismatch", "deferred_verify"):
            if item.get(key):
                record[key] = bounded_v2_report(str(item.get(key) or ""))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        records.append(record)
    return _json_text(records)


def _plan_mutable_scope(plan: TaskPlanV2) -> set[str]:
    return {
        path
        for step in plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }


def _cycle_kind_value(cycle: Any) -> str:
    kind = getattr(cycle, "kind", "")
    return str(getattr(kind, "value", kind))


@dataclasses.dataclass(frozen=True)
class EffectivePlanView:
    """The current plan authority without cumulative historical plan text."""

    original_objective: str
    original_constraints: str
    current_cumulative_approved_mutable_scope: tuple[str, ...]
    current_step_index: tuple[dict[str, Any], ...]
    required_deterministic_check_ids: tuple[str, ...]
    accepted_correction_plans: tuple[dict[str, Any], ...]
    current_cycle_correction_objective: str | None = None

    @classmethod
    def from_cycle_plans(
        cls, original_plan: TaskPlanV2, cycle_plans: Sequence[Any],
        *, correction_plan_hashes: Mapping[int, str] | None = None,
    ) -> "EffectivePlanView":
        if not cycle_plans:
            raise ValueError("at least one cycle plan is required")

        cumulative_scope: set[str] = set()
        accepted_corrections: list[dict[str, Any]] = []
        for item in cycle_plans:
            plan = item.plan
            before = set(cumulative_scope)
            current_scope = _plan_mutable_scope(plan)
            cumulative_scope.update(current_scope)
            if _cycle_kind_value(item.cycle) != "review-replan":
                continue
            plan_sha = (
                correction_plan_hashes.get(item.cycle.number)
                if correction_plan_hashes is not None
                else None
            )
            if plan_sha is None:
                plan_sha = getattr(item, "correction_plan_sha256", None)
            if plan_sha is None:
                plan_sha = getattr(item, "correction_bundle_sha256", None)
            if not isinstance(plan_sha, str) or not plan_sha:
                plan_sha = "UNAVAILABLE"
            accepted_corrections.append({
                "cycle": item.cycle.number,
                "plan_sha256": plan_sha,
                "cycle_kind": _cycle_kind_value(item.cycle),
                "approved_scope_delta": {
                    "added_paths": sorted(current_scope - before),
                    "unchanged_paths": sorted(current_scope & before),
                    "effective_paths": sorted(cumulative_scope),
                },
            })

        current = cycle_plans[-1]
        current_index = tuple(
            {
                "id": step.id,
                "title": step.title,
                "depends_on": step.depends_on,
                "objective": step.objective,
                "mutation_scope": {
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                },
                "verify": step.verify,
                "invariants": step.forbidden,
            }
            for step in current.plan.steps
        )
        correction_objective = (
            current.plan.objective
            if _cycle_kind_value(current.cycle) == "review-replan"
            else None
        )
        return cls(
            original_objective=original_plan.objective,
            original_constraints=original_plan.constraints,
            current_cumulative_approved_mutable_scope=tuple(sorted(cumulative_scope)),
            current_step_index=current_index,
            required_deterministic_check_ids=tuple(current.plan.required_checks),
            accepted_correction_plans=tuple(accepted_corrections),
            current_cycle_correction_objective=correction_objective,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_objective": self.original_objective,
            "original_constraints": self.original_constraints,
            "current_cumulative_approved_mutable_scope": list(
                self.current_cumulative_approved_mutable_scope
            ),
            "current_step_index": list(self.current_step_index),
            "required_deterministic_check_ids": list(
                self.required_deterministic_check_ids
            ),
            "accepted_correction_plans": list(self.accepted_correction_plans),
            "current_cycle_correction_objective": self.current_cycle_correction_objective,
        }

    def render(self) -> str:
        return _json_text(self.as_dict())


def _structural_check_state(bundle: EvidenceBundle) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required = set(bundle.required_check_ids)
    for raw in _check_payload(bundle):
        item = dict(raw)
        name = item.get("name")
        checks.append({
            "id": name,
            "required": bool(item.get("required", name in required)),
            "exit_code": item.get("exit_code"),
            "timed_out": bool(item.get("timed_out", False)),
            "workspace_mutated": bool(item.get("workspace_mutated", False)),
            "status": (
                "timed_out" if item.get("timed_out") else
                "passed" if item.get("exit_code") == 0 else "failed"
            ),
        })
    return {
        "required_check_ids": list(bundle.required_check_ids),
        "deterministic_passed": bundle.deterministic_passed,
        "failures": list(bundle.failures),
        "checks": checks,
    }


@dataclasses.dataclass(frozen=True)
class ReviewCycleInput:
    """Durable, bounded context presented to the independent reviewer."""

    iteration: int
    plan_text: str
    step_reports: str
    revision_report: str
    cycle_history: str
    scope_delta: str = ""
    deferred_mismatches: str = ""


@dataclasses.dataclass(frozen=True)
class ReviewContextBuilder:
    """Assemble reviewer correction context from cycle artifacts only."""

    cycle_plan: Callable[[Any, int], Any]
    completed_steps: Callable[[Any, Any], list[dict[str, Any]]]
    load_revision: Callable[[Path], Any]
    # ``None`` is the honest record of a cycle a red gate re-decomposed: that
    # cycle never reached a candidate, so it has no reviewed history.
    read_candidate: Callable[[Path, int], Mapping[str, Any] | None]
    candidate_evidence: Callable[[Path, int], EvidenceBundle | None]
    accepted_review: Callable[[Path, EvidenceBundle, str], ReviewResult | None]

    @staticmethod
    def _correction_plan_hashes(
        run_dir: Path, plans: Sequence[Any],
    ) -> dict[int, str]:
        hashes: dict[int, str] = {}
        for item in plans:
            if _cycle_kind_value(item.cycle) != "review-replan":
                continue
            text = _read_bounded_text(
                correction_dir(run_dir, item.cycle.number) / "scope_delta.json",
                limit=256 * 1024,
            )
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                continue
            plan_sha = payload.get("repair_plan_sha256") if isinstance(payload, Mapping) else None
            if isinstance(plan_sha, str) and plan_sha:
                hashes[item.cycle.number] = plan_sha
        return hashes

    def build(self, ctx: Any, cycle_plan: Any) -> ReviewCycleInput:
        number = cycle_plan.cycle.number
        plans = [self.cycle_plan(ctx, item) for item in range(1, number)] + [cycle_plan]
        effective_plan = EffectivePlanView.from_cycle_plans(
            ctx.plan,
            plans,
            correction_plan_hashes=self._correction_plan_hashes(ctx.run_dir, plans),
        )
        plan_text = effective_plan.render()
        scope_delta = ""
        if number > 1 and _cycle_kind_value(cycle_plan.cycle) == "review-replan":
            scope_delta = _json_text(
                effective_plan.accepted_correction_plans[-1]["approved_scope_delta"]
            )

        history: dict[str, Any] = {}
        for item in plans[:-1]:
            earlier = item.cycle.number
            candidate = self.read_candidate(ctx.run_dir, earlier)
            if candidate is None:
                history[f"{earlier:03d}"] = {
                    "cycle_kind": _cycle_kind_value(item.cycle),
                    "previous_route": "check-replan",
                    "previous_check_state": None,
                    "previous_candidate_sha": None,
                    "changed_paths": [],
                }
                continue
            evidence = self.candidate_evidence(ctx.run_dir, earlier)
            review = (
                self.accepted_review(
                    review_dir(ctx.run_dir, earlier), evidence, candidate["commit_sha"]
                ) if evidence is not None else None
            )
            history[f"{earlier:03d}"] = {
                "cycle_kind": _cycle_kind_value(item.cycle),
                "previous_route": (
                    review.route.value if review is not None else None
                ),
                "previous_check_state": (
                    _structural_check_state(evidence) if evidence is not None else None
                ),
                "previous_candidate_sha": candidate.get("commit_sha"),
                "changed_paths": list(evidence.changed_files) if evidence is not None else [],
            }
        cycle_history = (
            _json_text(history)
            if history else "001 is the initial implementation cycle."
        )
        step_reports, revision_reports, mismatches = [], [], []
        for item in plans:
            label = f"{item.cycle.number:03d}"
            steps = self.completed_steps(ctx, item)
            step_reports.append(f"CYCLE {label} STEP REPORTS\n{_review_step_reports_text(steps)}")
            revision_reports.append(
                f"CYCLE {label} REVISION REPORTS\n"
                f"{review_cycle_revision_report(ctx.run_dir, item.cycle.number, self.load_revision) or 'NONE'}"
            )
            mismatches.append(
                f"CYCLE {label} DEFERRED CONTRACT MISMATCHES\n"
                f"{_deferred_contract_mismatches(item.plan, steps)}"
            )
        return ReviewCycleInput(
            iteration=number,
            plan_text=plan_text,
            step_reports="\n".join(step_reports),
            revision_report="\n".join(revision_reports),
            cycle_history=cycle_history,
            scope_delta=scope_delta,
            deferred_mismatches="\n".join(mismatches),
        )



def review_cycle_revision_report(
    run_dir: Path, number: int, load_revision: Callable[[Path], Any],
) -> str:
    reports: dict[str, Any] = {}
    revision_dir = semantic_revision_dir(run_dir, number)
    try:
        status_payload = json.loads((revision_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        status_payload = None
    if isinstance(status_payload, dict) and status_payload.get("status") == "UNAVAILABLE":
        reason = status_payload.get("reason")
        reports["semantic_revision"] = (
            "SEMANTIC REVISION: UNAVAILABLE\nreason="
            + (str(reason)[:120] if isinstance(reason, str) else "infrastructure failure")
        )
    elif isinstance(status_payload, dict) and status_payload.get("status") == "REPLAN_REQUIRED":
        reports["semantic_revision"] = (
            "SEMANTIC REVISION: SCOPE EXPANSION DENIED\nroute=REPLAN\nreason="
            + str(status_payload.get("reason", "scope expansion denied"))[:200]
        )
    elif isinstance(status_payload, dict) and status_payload.get("status") == "SCOPE_REQUEST_RECORDED":
        requested = status_payload.get("requested_paths")
        reports["semantic_revision"] = (
            "SEMANTIC REVISION: SCOPE REQUEST RECORDED\nreason="
            + str(status_payload.get("reason", "scope already authorized"))[:200]
            + "\npaths="
            + ", ".join(path for path in requested if isinstance(path, str))[:500]
            if isinstance(requested, list) else
            "SEMANTIC REVISION: SCOPE REQUEST RECORDED"
        )
    revision = load_revision(revision_dir)
    if revision is not None and "semantic_revision" not in reports:
        reports["semantic_revision"] = _revision_report_text(revision, revision_dir)
    for stage_dir in sorted(check_repair_root(run_dir, number).glob("*")):
        for attempt_dir in sorted((stage_dir / "attempts").glob("[0-9][0-9][0-9]")):
            attempt = load_revision(attempt_dir)
            if attempt is not None:
                reports[f"check_repair/{stage_dir.name}/{attempt_dir.name}"] = (
                    _revision_report_text(attempt, attempt_dir)
                )
    return "\n\n".join(
        f"{key}\n{value}" for key, value in reports.items()
    ) if reports else ""


def _revision_prompt(
    *,
    repository_reference: RepositoryReference,
    spec: str,
    plan: TaskPlanV2,
    changed_files: str,
    execution_anomalies: str,
    pre_checks: str,
    mutable_scope: str,
    deferred_mismatches: str,
    candidate_identity: str = "",
    bounded_diff_evidence: str = "",
    reviewer_correction_evidence: str = "NONE\n",
    diagnostics_dir: str | Path | None = None,
    budget_bytes: int = 120_000,
) -> str:
    template = (_PROMPTS_DIR / "reviser.txt").read_text(encoding="utf-8")
    payload = build_semantic_revision_payload(
        spec=spec,
        compact_approved_contract_index=_revision_contract_index(plan),
        candidate_identity=candidate_identity or "UNKNOWN",
        changed_files=changed_files,
        required_checks_summary=pre_checks,
        mutable_scope=mutable_scope,
        bounded_diff_evidence=bounded_diff_evidence or "NONE\n",
        reviewer_correction_evidence=reviewer_correction_evidence or "NONE\n",
        template=template,
        budget_bytes=budget_bytes,
    )
    if diagnostics_dir is not None:
        write_prompt_diagnostics(diagnostics_dir, payload)
    return payload.rendered


def _revision_report_text(result: Any, artifact_dir: Path) -> str:
    """Bounded, reviewer-facing record of one revision or repair pass."""

    def tree(name: str) -> str | None:
        try:
            return (artifact_dir / name).read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeError):
            return None

    return _json_text({
        "final": _bounded_report(result.final_message),
        "tree_before": tree("tree_before.txt"),
        "tree_after": tree("tree_after.txt"),
        "usage": normalize_usage(result.usage),
    })


_REVISION_CHECK_LOG_BYTES = 16 * 1024


def _revision_check_context(payload: Mapping[str, Any]) -> str:
    """Render compact pre-revision check state for the semantic reviser.

    Successful checks contribute status only.  Output tails are decision
    evidence only for failed checks and stay bounded for prompt safety; the
    complete logs remain in the durable check artifacts.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("check payload must be a mapping")

    failures = [
        item for item in payload.get("failures", [])
        if isinstance(item, str)
    ]
    failed_names = {
        item.split(":", 1)[1]
        for item in failures
        if item.startswith("CHECK_FAILED:")
    }
    checks: list[dict[str, Any]] = []
    raw_checks = payload.get("checks", [])
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raw_checks = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, Mapping):
            continue
        name = raw_check.get("name")
        failed = name in failed_names
        check: dict[str, Any] = {
            "name": name,
            "required": bool(raw_check.get("required", False)),
            "exit_code": raw_check.get("exit_code"),
            "timed_out": bool(raw_check.get("timed_out", False)),
            "workspace_mutated": bool(raw_check.get("workspace_mutated", False)),
        }
        if failed:
            for key in ("stdout_tail", "stderr_tail"):
                value = raw_check.get(key)
                if isinstance(value, str) and value:
                    data = value.encode("utf-8", errors="replace")
                    if len(data) > _REVISION_CHECK_LOG_BYTES:
                        data = data[-_REVISION_CHECK_LOG_BYTES:]
                    check[key] = data.decode("utf-8", errors="replace")
        checks.append(check)

    return _json_text({
        "deterministic_passed": bool(payload.get("deterministic_passed", False)),
        "failure_ids": failures,
        "checks": checks,
    })


@dataclasses.dataclass(frozen=True)
class RevisionRunner:
    """Runs one semantic-revision or check-repair executor pass.

    Every dependency is injected explicitly: the runner never receives the
    ``Orchestrator`` instance and never writes a resume checkpoint -- the
    pipeline coordinator owns every durable boundary around this pass.
    """

    config: HarnessConfig
    secrets: tuple[str, ...]
    effective_repair_scope: EffectiveRepairScopePolicy
    approved_check_authority_sha256: Callable[[Path], str | None]
    run_revision: Callable[..., Any]
    ensure_revision_artifacts: Callable[[Path, Any], None]
    redact_revision_artifacts: Callable[[Path], None]
    reusable_pre_checks: Callable[[Path, str], dict[str, Any] | None]
    hard_integrity_failures: Callable[[EvidenceBundle], list[str]]
    soft_check_failures: Callable[[EvidenceBundle], list[str]]
    check_repair_prompt: Callable[..., str]

    def run(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        repo: Path,
        base_sha: str,
        base_tree_sha: str,
        spec: str,
        plan: TaskPlanV2,
        repository_reference: RepositoryReference,
        info: Any,
        branch_ref: str,
        ownership_before: Any,
        selection: ExecutionSelection,
        artifact_dir: Path,
        mutable_scope: list[str],
        step_results: Sequence[dict[str, Any]] = (),
        deferred_mismatches: str | None = None,
        deferred_mismatch_present: bool = False,
        reviewer_correction_evidence: str | None = None,
        candidate_identity: str | None = None,
        bounded_diff_evidence: str | None = None,
        pre_check_evidence: EvidenceBundle | None = None,
        check_repair_evidence: EvidenceBundle | None = None,
        check_repair_scope: CheckRepairScope | None = None,
        check_evidence_dir: Path | None = None,
    ) -> tuple[Any | None, str | None]:
        """Run one pass and return ``(result, failure_reason)``.

        A semantic revision first freezes pre-revision checks for the exact
        current tree (reused when already durable); a check-repair pass
        answers to the red gate evidence it is given.
        """

        is_check_repair = check_repair_evidence is not None
        artifact_dir.mkdir(parents=True, exist_ok=True)
        expected_head = current_head(info.worktree)
        stage_all(info.worktree)
        tree_before = candidate_tree_sha(info.worktree)
        index_before = index_tree_sha(info.worktree)
        status_before = status_porcelain(info.worktree)
        if index_before != tree_before:
            return None, "RESUME_REQUIRES_OPERATOR"
        if is_check_repair and check_repair_evidence.staged_tree_sha != tree_before:
            return None, "TOCTOU_FAILURE"
        if is_check_repair:
            check_repair_scope = check_repair_scope or CheckRepairScope(
                approved_mutable_scope=tuple(mutable_scope),
                initial_repair_scope=tuple(mutable_scope),
                added_paths=(),
                effective_repair_scope=tuple(mutable_scope),
                policy=self.effective_repair_scope.policy,
                bound=self.effective_repair_scope.max_added_paths,
                source="human-approved mutable scope",
            )
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "schema_version": 3,
                "approved_mutable_scope": list(check_repair_scope.approved_mutable_scope),
                "initial_repair_scope": list(check_repair_scope.initial_repair_scope),
                "added_paths": list(check_repair_scope.added_paths),
                "effective_repair_scope": list(check_repair_scope.effective_repair_scope),
                "policy": check_repair_scope.policy,
                "bound": check_repair_scope.bound,
                "source": check_repair_scope.source,
            }))
            revision_prompt = self.check_repair_prompt(
                spec=spec,
                plan=plan,
                approved_contract_index=_revision_contract_index(plan),
                changed_files="\n".join(check_repair_evidence.changed_files),
                evidence=check_repair_evidence,
                evidence_dir=check_evidence_dir or artifact_dir,
                repo=repo,
                worktree=info.worktree,
                effective_repair_scope=mutable_scope,
                candidate_identity=candidate_identity or tree_before,
                budget_bytes=self.config.prompt_budget.check_repair_max_bytes,
                diagnostics_dir=artifact_dir,
            )
        else:
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "approved_mutable_scope": mutable_scope,
                "source": "human-approved mutable scope",
            }))
            pre_payload = self.reusable_pre_checks(artifact_dir, tree_before)
            if pre_check_evidence is not None:
                if (
                    pre_check_evidence.staged_tree_sha != tree_before
                    or not pre_check_evidence.deterministic_passed
                    or not required_checks_passed(pre_check_evidence)
                ):
                    return None, "TOCTOU_FAILURE"
                pre_payload = {
                    "checks": _check_payload(pre_check_evidence),
                    "failures": list(pre_check_evidence.failures),
                    "deterministic_passed": pre_check_evidence.deterministic_passed,
                    "staged_tree_sha": pre_check_evidence.staged_tree_sha,
                }
                atomic_write_text(artifact_dir / "pre_checks.json", _json_text(pre_payload))
            elif pre_payload is None:
                store.update_metadata(current_step=None)
                check_config, check_ids = config_with_check_authority(
                    self.config, run_dir, requested_check_ids=plan.required_checks or None,
                    expected_sha256=self.approved_check_authority_sha256(run_dir),
                )
                pre_evidence = collect_evidence(
                    info.worktree, base_sha, check_config,
                    required_check_ids=check_ids,
                    evidence_dir=artifact_dir, secrets=self.secrets,
                    check_failures_hard=False,
                    expected_head_sha=expected_head,
                    enforce_diff_size=False,
                    allow_empty_diff=expected_head != base_sha,
                )
                pre_payload = {
                    "checks": _check_payload(pre_evidence),
                    "failures": list(pre_evidence.failures),
                    "deterministic_passed": pre_evidence.deterministic_passed,
                    "staged_tree_sha": pre_evidence.staged_tree_sha,
                }
                atomic_write_text(artifact_dir / "pre_checks.json", _json_text(pre_payload))
                pre_hard = self.hard_integrity_failures(pre_evidence)
                # A clean deferred mismatch intentionally leaves no candidate
                # delta for the pre-revision gate.  The semantic reviser owns
                # that recovery, so EMPTY_DIFF is evidence here, not a gate.
                if deferred_mismatch_present:
                    pre_hard = [item for item in pre_hard if item != "EMPTY_DIFF"]
                if pre_hard:
                    return None, pre_hard[0].split(":", 1)[0]
            revision_prompt = _revision_prompt(
                repository_reference=repository_reference,
                spec=spec,
                plan=plan,
                changed_files="\n".join(changed_paths_between_trees(repo, base_tree_sha, tree_before)),
                execution_anomalies=_revision_execution_anomalies(list(step_results)),
                pre_checks=_revision_check_context(pre_payload),
                mutable_scope=_json_text(mutable_scope),
                deferred_mismatches=deferred_mismatches or "NONE\n",
                candidate_identity=tree_before,
                # The semantic reviser can inspect the current worktree
                # directly.  Keep the secondary diff excerpt optional so a
                # large or adversarial diff never becomes the default prompt.
                bounded_diff_evidence=bounded_diff_evidence or "NONE\n",
                reviewer_correction_evidence=reviewer_correction_evidence or "NONE\n",
                diagnostics_dir=artifact_dir,
                budget_bytes=self.config.prompt_budget.semantic_revision_max_bytes,
            )
        store.update_metadata(current_step=None)
        atomic_write_text(artifact_dir / "tree_before.txt", tree_before.rstrip() + "\n")
        selected = selection.check_repair if is_check_repair else selection.semantic_reviser
        if selected is None:
            return None, (
                "CHECK_REPAIR_PROFILE_MISSING"
                if is_check_repair else "SEMANTIC_REVISER_PROFILE_MISSING"
            )
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        result = self.run_revision(
            worktree=Path(info.worktree),
            prompt=revision_prompt,
            artifact_dir=artifact_dir,
            profile_id=selected.profile_id,
            role=role,
            mutable_paths=tuple(mutable_scope),
        )
        self.ensure_revision_artifacts(artifact_dir, result)
        self.redact_revision_artifacts(artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self.secrets),
            stderr_tail=redact(result.stderr_tail, self.secrets),
        )

        if getattr(result, "timed_out", False) or getattr(result, "exit_reason", None) == AGENT_TIMEOUT:
            _record_failure_tree(artifact_dir, info.worktree)
            return result, AGENT_TIMEOUT
        terminal_is_error = getattr(result, "terminal_is_error", None) is True
        # A structured terminal marked ``is_error`` is a failure on its own.
        # Some CLI versions and wrappers still exit 0 after one, so exit code
        # is the last signal consulted, never the gate for the others.
        if terminal_is_error or getattr(result, "exit_code", None) not in (None, 0) or getattr(result, "exit_reason", None) in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            _record_failure_tree(artifact_dir, info.worktree)
            if getattr(result, "terminal_subtype", None) == "error_max_turns":
                return result, AGENT_PROTOCOL_FAILED
            return result, getattr(result, "exit_reason", None) or AGENT_RUNTIME_FAILED
        revision_ownership = _git_ownership(repo, info.worktree)
        if revision_ownership.head != expected_head:
            return result, AGENT_SCOPE_VIOLATION
        violations = _ownership_violations(
            ownership_before, revision_ownership,
            branch_ref=branch_ref, base_sha=expected_head,
        )
        if violations:
            return result, "AGENT_GIT_VIOLATION"
        stage_all(info.worktree)
        tree_after = candidate_tree_sha(info.worktree)
        atomic_write_text(artifact_dir / "tree_after.txt", tree_after.rstrip() + "\n")
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        outside_scope = [path for path in changed_paths if path not in set(mutable_scope)]
        # A valid request is evidence for the orchestration layer, never an
        # authorization by itself. The pass is rolled back before policy can
        # authorize a new scope.
        scope_request = parse_scope_request(result.final_message)
        check_repair_result = (
            parse_check_repair_result(result.final_message) if is_check_repair else None
        )
        malformed_scope_request = (
            _SCOPE_REQUEST_HEADER in result.final_message
            and scope_request is None
        )
        check_repair_unverified = bool(
            is_check_repair and (
                check_repair_result is None
                or (
                    check_repair_result.result == "DONE"
                    and check_repair_result.targeted_check == "NOT_RUN"
                )
                or (
                    malformed_scope_request
                    and not (
                        check_repair_result is not None
                        and check_repair_result.result == "BLOCKED"
                        and check_repair_result.blocked_kind == "SCOPE"
                    )
                )
                or (
                    scope_request is not None
                    and (
                        check_repair_result.result != "BLOCKED"
                        or check_repair_result.blocked_kind != "SCOPE"
                    )
                )
            )
        )
        usage = normalize_usage(result.usage)
        revision_state = {
            "profile_id": selected.profile_id,
            "status": (
                "BLOCKED" if check_repair_result is not None
                and check_repair_result.result == "BLOCKED" else
                "UNVERIFIED" if check_repair_unverified else
                "NO_CHANGE" if tree_after == tree_before else "COMPLETED"
            ),
            "tree_before": tree_before,
            "tree_after": tree_after,
            "usage": usage,
            **(
                {"check_repair_result": dataclasses.asdict(check_repair_result)}
                if check_repair_result is not None else {}
            ),
            **(
                {"check_repair_protocol_error": "result block missing or invalid"}
                if check_repair_unverified else {}
            ),
            **(
                {
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                }
                if scope_request is not None else {}
            ),
            **(
                {"scope_request_warning": "malformed scope request ignored as authority"}
                if malformed_scope_request else {}
            ),
        }
        atomic_write_text(artifact_dir / "usage.json", _json_text(usage))
        atomic_write_text(artifact_dir / "report.json", _json_text({
            **revision_state,
            "final": _bounded_report(result.final_message),
            "stderr_tail": result.stderr_tail,
            "changed_paths": list(changed_paths),
            "outside_scope_paths": list(outside_scope),
            **({"failure_ids": self.soft_check_failures(check_repair_evidence)}
               if is_check_repair else {}),
        }))
        store.update_metadata(revision=revision_state)
        if outside_scope:
            requested_paths = set(scope_request.paths) if scope_request is not None else set()
            unrequested = [path for path in outside_scope if path not in requested_paths]
            if scope_request is not None and not unrequested:
                _record_failure_tree(artifact_dir, info.worktree)
                try:
                    restore_paths_from_tree(info.worktree, tree_before, list(changed_paths))
                    stage_all(info.worktree)
                    if (
                        candidate_tree_sha(info.worktree) != tree_before
                        or index_tree_sha(info.worktree) != index_before
                        or status_porcelain(info.worktree) != status_before
                    ):
                        raise GitError("scope-request rollback did not restore the exact tree")
                except (GitError, OSError):
                    return result, "RESUME_REQUIRES_OPERATOR"
                return result, SCOPE_REQUEST_ROUTE
            # A successful transport with an unsafe candidate: the exact failed
            # tree stays durable as evidence and the run stops for an operator.
            _record_failure_tree(artifact_dir, info.worktree)
            return result, "REVISION_SCOPE_VIOLATION"
        if is_check_repair and check_repair_unverified:
            _record_failure_tree(artifact_dir, info.worktree)
            return result, "CHECK_REPAIR_UNAVAILABLE"
        if is_check_repair and check_repair_result is not None:
            if check_repair_result.result == "BLOCKED":
                _record_failure_tree(artifact_dir, info.worktree)
                if check_repair_result.blocked_kind == "INFRASTRUCTURE":
                    return result, "CHECK_REPAIR_UNAVAILABLE"
                if check_repair_result.blocked_kind == "SCOPE":
                    return result, (
                        SCOPE_REQUEST_ROUTE if scope_request is not None
                        else "HUMAN_REQUIRED"
                    )
                return result, "HUMAN_REQUIRED"
        if malformed_scope_request and not is_check_repair:
            # A semantic reviser that emits the scope marker without a valid
            # protocol is not allowed to turn malformed data into authority.
            _record_failure_tree(artifact_dir, info.worktree)
            try:
                restore_paths_from_tree(info.worktree, tree_before, list(changed_paths))
                stage_all(info.worktree)
            except (GitError, OSError):
                pass
            return result, "HUMAN_REQUIRED"
        if scope_request is not None:
            # A valid request is advisory evidence, never an authorization
            # delta.  Even an in-scope attempt is rolled back atomically, so
            # the durable candidate stays the tree the gate answered for.
            _record_failure_tree(artifact_dir, info.worktree)
            try:
                restore_paths_from_tree(info.worktree, tree_before, list(changed_paths))
                stage_all(info.worktree)
                if (
                    candidate_tree_sha(info.worktree) != tree_before
                    or index_tree_sha(info.worktree) != index_before
                    or status_porcelain(info.worktree) != status_before
                ):
                    raise GitError("scope-request rollback did not restore the exact tree")
            except (GitError, OSError):
                pass
            return result, SCOPE_REQUEST_ROUTE
        return result, None
