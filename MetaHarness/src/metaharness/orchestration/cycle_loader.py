"""Cycle-scoped durable loaders: records, bindings, plans and scopes.

One cycle's durable identity, the review or re-decomposition that authorizes
it, the correction plan and bundle it runs and the mutable scope it was
granted are read here and nowhere else.  Every loader fails closed: an
artifact that does not reproduce the exact authority of its cycle raises
:class:`~metaharness.resume.ResumeIntegrityError`.
"""

from __future__ import annotations

import hashlib
import re

from pathlib import Path
from typing import Any, NoReturn
from .durable_readers import (
    accepted_review,
    candidate_evidence,
    read_candidate_record,
)
from .pipeline_v2 import (
    candidate_dir,
    correction_dir,
    cycle_record_path,
    review_dir,
    semantic_revision_dir,
)
from .shared import (
    OrchestrationError,
    create_file_once,
    is_object_id,
    json_text,
    read_json_artifact,
)
from ..execution_selection import (
    ExecutionSelectionError,
    read_cycle_execution_selection,
    validate_cycle_execution_selection,
)
from ..models import (
    CycleKind,
    ExecutionSelection,
    HarnessConfig,
    PlanDecision,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    TaskPlanV2,
)
from ..planning.artifacts import validate_implementation_bundle
from ..planning.check_replan import (
    PLAN_ARTIFACT as CHECK_REPLAN_PLAN_ARTIFACT,
    check_replan_dir,
)
from ..planning.protocol import V2PlanParseError, parse_task_plan_v2
from ..resume import ResumeIntegrityError
from ..scope import normalize_repo_paths
from ..review import ReviewResult


def _refuse(message: str) -> NoReturn:
    raise ResumeIntegrityError(message)


def read_cycle_record(run_dir: Path, number: int) -> RunCycle:
    """The durable identity of one cycle, written when the cycle started."""

    payload = read_json_artifact(cycle_record_path(run_dir, number), 4096)
    expected_schema = 1 if number == 1 else 2
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != expected_schema
        or payload.get("number") != number
    ):
        raise ResumeIntegrityError(f"cycle {number:03d} record is missing or invalid")
    try:
        return RunCycle(number, CycleKind(payload.get("kind")))
    except ValueError as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} kind is invalid") from exc


def _accepted_review_binding(run_dir: Path, number: int) -> tuple[dict[str, Any], ReviewResult, str]:
    """Read the accepted review and hash the exact bytes of review.json."""

    candidate = read_candidate_record(run_dir, number)
    evidence = candidate_evidence(run_dir, number)
    if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
        raise ResumeIntegrityError(f"cycle {number:03d} candidate evidence is missing")
    directory = review_dir(run_dir, number)
    review = accepted_review(directory, evidence, candidate["commit_sha"])
    if review is None or review.verdict is not ReviewVerdict.REVISE:
        raise ResumeIntegrityError(f"cycle {number:03d} accepted correction review is invalid")
    if review.route not in {ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN}:
        raise ResumeIntegrityError(f"cycle {number:03d} accepted correction route is invalid")
    try:
        digest = hashlib.sha256((directory / "review.json").read_bytes()).hexdigest()
    except OSError as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} review artifact is unreadable") from exc
    return candidate, review, digest


# The source route a re-decomposed cycle records: its plan was decided by a red
# deterministic gate, never by a review.
CHECK_REPLAN_ROUTE = "check-replan"


def check_replan_binding(run_dir: Path, number: int) -> dict[str, Any]:
    """The durable binding of a cycle one red gate re-decomposed."""

    path = check_replan_dir(run_dir, number) / CHECK_REPLAN_PLAN_ARTIFACT
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ResumeIntegrityError(
            f"cycle {number:03d} has no durable check-replan plan"
        ) from exc
    record = read_json_artifact(path, 64 * 1024)
    if (
        not isinstance(record, dict)
        or record.get("cycle") != number
        or not is_object_id(record.get("candidate_tree_sha"))
    ):
        raise ResumeIntegrityError(f"cycle {number:03d} check-replan record is invalid")
    return {
        "source_review_cycle": number - 1,
        "source_route": CHECK_REPLAN_ROUTE,
        "source_candidate_sha": record["candidate_tree_sha"],
        "source_check_replan_sha256": digest,
        "kind": CycleKind.CHECK_REPLAN.value,
    }


def _correction_plan_dir(run_dir: Path, number: int) -> Path:
    """Where the plan authority of correction cycle *number* lives."""

    if number > 1 and read_cycle_record(run_dir, number).kind is CycleKind.CHECK_REPLAN:
        return check_replan_dir(run_dir, number)
    return correction_dir(run_dir, number)


def correction_binding(run_dir: Path, number: int) -> dict[str, Any]:
    """Return the exact durable binding required by correction cycle number."""

    if number < 2:
        raise ValueError("correction cycle number must be greater than one")
    if not (candidate_dir(run_dir, number - 1) / "commit.json").is_file():
        # The cycle this one corrects never reached a candidate: it ended on a
        # red deterministic gate whose ladder re-decomposed it.  A source cycle
        # that *did* reach a candidate is bound by its accepted review, never by
        # a later gate episode of that same cycle.
        return check_replan_binding(run_dir, number)
    candidate, review, review_sha256 = _accepted_review_binding(run_dir, number - 1)
    kind = (
        CycleKind.REVIEW_IMPLEMENTATION
        if review.route is ReviewRoute.IMPLEMENTATION
        else CycleKind.REVIEW_REPLAN
    )
    return {
        "source_review_cycle": number - 1,
        "source_route": review.route.value,
        "source_candidate_sha": candidate["commit_sha"],
        "source_review_sha256": review_sha256,
        "kind": kind.value,
    }


def validate_correction_bindings(run_dir: Path, through_cycle: int) -> None:
    """Validate every persisted correction binding through through_cycle."""

    if through_cycle < 2:
        return
    for number in range(2, through_cycle + 1):
        payload = read_json_artifact(cycle_record_path(run_dir, number), 4096)
        expected = correction_binding(run_dir, number)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 2
            or payload.get("number") != number
            or any(payload.get(key) != value for key, value in expected.items())
        ):
            raise ResumeIntegrityError(f"cycle {number:03d} correction binding diverges")


def load_correction_plan(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path, number: int,
    *, inherited_check_ids: tuple[str, ...],
) -> tuple[TaskPlanV2, dict[str, Any], str]:
    """Parse the durable correction plan of cycle *number* (> 1)."""

    directory = _correction_plan_dir(run_dir, number)
    try:
        plan = parse_task_plan_v2(
            (directory / "planner.raw.md").read_text(encoding="utf-8"),
            planning=config.planning,
            check_catalog=config.check_catalog,
            inherited_check_ids=inherited_check_ids,
        )
        bundle, bundle_sha = validate_implementation_bundle(
            directory, expected_step_ids=[step.id for step in plan.steps]
        )
    except (OSError, UnicodeError, V2PlanParseError, ValueError, AttributeError) as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} correction plan is unreadable: {exc}") from exc
    if plan.decision is not PlanDecision.READY:
        raise ResumeIntegrityError(f"cycle {number:03d} correction plan is not READY")
    try:
        cycle_selection = read_cycle_execution_selection(run_dir, number)
        validate_cycle_execution_selection(config, cycle_selection)
    except (ExecutionSelectionError, OSError) as exc:
        raise ResumeIntegrityError(
            f"cycle {number:03d} execution selection is unreadable: {exc}"
        ) from exc
    if [item.step_id for item in cycle_selection.steps] != [step.id for step in plan.steps]:
        raise ResumeIntegrityError(
            f"cycle {number:03d} execution selection does not match the plan"
        )
    return plan, bundle, bundle_sha


def _correction_mutation_sets(plan: TaskPlanV2) -> tuple[list[str], list[str], list[str]]:
    """The correction's plan-declared mutation sets, canonical and sorted.

    The plan grammar already refused a glob and validated every path as
    repository-relative; the scope authority is the one that refuses a path
    this delta may not admit, so nothing is re-validated here.
    """

    return (
        sorted({path for step in plan.steps for path in step.write_set}),
        sorted({path for step in plan.steps for path in step.create_set}),
        sorted({path for step in plan.steps for path in step.delete_set}),
    )


def build_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, repair_bundle_sha: str, justification: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta, in memory only: from the parsed plan sets.

    The delta is derived from the parsed plan alone; *justification* names the
    durable failure the added paths answer, and is never reviewer or planner
    prose.  Every plan that may widen an approved envelope -- a review
    correction and a red-gate cycle replan alike -- goes through this one
    builder, so the same plan always produces the same delta bytes.
    """

    writes, creates, deletes = _correction_mutation_sets(plan)
    requested = sorted(set(writes) | set(creates) | set(deletes))
    original = sorted(set(original_scope))
    added = sorted(set(requested) - set(original))
    findings = justification.strip()
    reasons: dict[str, Any] = {}
    for path in added:
        step = next(
            step for step in plan.steps
            if path in set(step.write_set) | set(step.create_set) | set(step.delete_set)
        )
        reasons[path] = {
            "reason": f"{step.title}: {step.objective}",
            "source_finding": findings,
        }
    try:
        plan_sha = hashlib.sha256((repair_dir / "planner.raw.md").read_bytes()).hexdigest()
    except OSError as exc:
        raise OrchestrationError("REPAIR_SCOPE_PLAN_UNREADABLE") from exc
    payload: dict[str, Any] = {
        "schema_version": 1,
        "original_mutable_paths": original,
        "requested_write_paths": writes,
        "requested_create_paths": creates,
        "requested_delete_paths": deletes,
        "added_paths": added,
        "unchanged_paths": sorted(set(requested) & set(original)),
        "added_path_reasons": reasons,
        "source_finding": findings,
        "candidate_commit_sha": candidate_commit_sha,
        "repair_plan_sha256": plan_sha,
        "correction_bundle_sha256": repair_bundle_sha,
    }
    return payload, json_text(payload)


def ensure_scope_delta(
    repair_dir: Path, content: str, *, expected_sha256: str | None,
) -> str:
    """Persist ``scope_delta.json`` exactly once, then only verify it.

    The first creation writes the canonical bytes atomically.  An existing
    artifact is never rewritten: its bytes must equal the canonical bytes and,
    when the checkpoint binds one, the checkpoint hash.  Any difference is a
    :class:`ResumeIntegrityError` and the file is left as found.
    """

    expected = content.encode("utf-8")
    digest = hashlib.sha256(expected).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ResumeIntegrityError("the correction scope delta changed")
    path = repair_dir / "scope_delta.json"
    if expected_sha256 is None:
        try:
            create_file_once(path, expected)
            return digest
        except FileExistsError:
            pass
        except OSError as exc:
            raise OrchestrationError("REPAIR_SCOPE_DELTA_UNWRITABLE") from exc
    try:
        if path.stat().st_size > 256 * 1024:
            raise ResumeIntegrityError("the correction scope delta is too large")
        existing = path.read_bytes()
    except OSError as exc:
        raise ResumeIntegrityError(f"the correction scope delta is unreadable: {exc}") from exc
    if existing != expected:
        raise ResumeIntegrityError("the correction scope delta changed")
    return digest


def verify_correction_scope(run_dir: Path, number: int, bundle_sha: str) -> list[str]:
    """The added paths of one correction scope delta, with their authority.

    The delta itself is the authority: it is derived from the parsed plan and
    recorded once, so a resume only re-proves that the recorded bytes are the
    canonical, path-safe ones it would derive again.
    """

    directory = _correction_plan_dir(run_dir, number)
    delta = read_json_artifact(directory / "scope_delta.json", 256 * 1024)
    if not isinstance(delta, dict) or delta.get("correction_bundle_sha256") != bundle_sha:
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta is missing or not bound to its plan")
    added = delta.get("added_paths")
    if not isinstance(added, list) or any(not isinstance(path, str) for path in added):
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta is malformed")
    try:
        canonical = sorted(set(normalize_repo_paths(added)))
    except ValueError as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta has invalid paths") from exc
    if added != canonical:
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta is not canonical")
    return list(added)


def semantic_revision_scope(run_dir: Path, number: int) -> set[str]:
    """Read the reviser-requested paths one cycle durably auto-admitted."""

    root = semantic_revision_dir(run_dir, number) / "scope_requests"
    if not root.is_dir():
        return set()
    revision_dir = semantic_revision_dir(run_dir, number)
    scope: set[str] = set()
    for request_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not request_dir.is_dir() or not request_dir.name.isdigit():
            continue
        authority = read_json_artifact(request_dir / "authority.json", 64 * 1024)
        if not isinstance(authority, dict):
            _refuse("semantic scope authority is missing")
        added = authority.get("added_paths")
        requested = authority.get("requested_paths")
        tree_sha = authority.get("tree_sha")
        source_report_sha = authority.get("source_report_sha256")
        if (
            authority.get("schema_version") != 1
            or authority.get("cycle") != number
            or not isinstance(added, list) or not isinstance(requested, list)
            or any(not _safe_semantic_scope_path(path) for path in (*added, *requested))
            or not isinstance(tree_sha, str) or not is_object_id(tree_sha)
            or not isinstance(source_report_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_report_sha)
        ):
            _refuse("semantic scope authority is malformed")
        if (
            added != sorted(set(added))
            or requested != sorted(set(requested))
            or set(added) - set(requested)
        ):
            _refuse("semantic scope authority is not canonical")
        source_reports = [revision_dir / "report.json"]
        attempts_root = revision_dir / "attempts"
        if attempts_root.is_dir():
            source_reports.extend(sorted(attempts_root.glob("[0-9][0-9]/report.json")))
        source_matches = False
        for report_path in source_reports:
            try:
                raw_report = report_path.read_bytes()
            except OSError:
                continue
            if hashlib.sha256(raw_report).hexdigest() != source_report_sha:
                continue
            report = read_json_artifact(report_path, 256 * 1024)
            scope_request = report.get("scope_request") if isinstance(report, dict) else None
            if not isinstance(scope_request, dict):
                continue
            source_paths = scope_request.get("paths")
            evidence = scope_request.get("evidence")
            source_reason = scope_request.get("reason")
            if (
                not isinstance(source_paths, list)
                or any(not isinstance(path, str) for path in source_paths)
                or not isinstance(source_reason, str)
            ):
                continue
            source_matches = bool(
                sorted(source_paths) == requested
                and source_reason[:2000] == authority.get("reason")
                and isinstance(evidence, list)
                and [str(item)[:1000] for item in evidence[:16]] == authority.get("evidence")
                and report.get("tree_before") == tree_sha
            )
            if source_matches:
                break
        if not source_matches:
            _refuse("semantic scope authority is not bound to its reviser report")
        decision = read_json_artifact(request_dir / "decision.json", 64 * 1024)
        if (
            not isinstance(decision, dict)
            or decision.get("decision") != "auto-admitted"
            or decision.get("added_paths") != added
        ):
            _refuse("semantic scope decision is missing or invalid")
        scope.update(added)
    return scope


def _safe_semantic_scope_path(path: Any) -> bool:
    """Whether one requested path already is its own canonical repository path."""

    try:
        return normalize_repo_paths((path,))[0] == path
    except ValueError:
        return False


__all__ = [
    "check_replan_binding", "correction_binding", "load_correction_plan",
    "read_cycle_record", "semantic_revision_scope",
    "validate_correction_bindings", "verify_correction_scope",
]
