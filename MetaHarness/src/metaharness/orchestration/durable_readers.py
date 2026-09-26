"""Fail-closed readers of the durable artifacts a resume depends on.

Every reader here is bounded and side-effect free: an artifact that does not
prove exactly what its caller needs yields ``None``, never a partially trusted
value.  Nothing in this module writes Git, the run state or any durable
artifact.  The one exception is the durable mutation authority of a gate
episode: it re-proves no path outside the envelope was ever authorized, so it
refuses a tampered artifact instead of returning a partial value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from pathlib import Path
from typing import Any, Sequence
from .check_failure import hard_failure_items
from .pipeline_v2 import (
    candidate_dir,
    check_repair_attempt_dir,
    check_repair_attempts_dir,
    gate_dir,
    step_dir,
)
from .shared import (
    MAX_AGENT_REPORT_BYTES,
    PLANNER_CONVERSATION,
    CheckRepairScope,
    GateMutableAuthority,
    bounded_v2_report,
    is_object_id,
    json_text,
    read_bounded_text,
    read_json_artifact,
)
from ..models import GateStage
from ..scope import normalize_repo_paths
from ..evidence import EvidenceBundle
from ..gitops import RepositoryReference
from ..resume import ResumeIntegrityError
from ..review import (
    ReviewParseError,
    ReviewResult,
    parse_review,
)
from ..usage import (
    normalize_usage,
    read_usage_artifact,
)
from ..llm.chat import LLMConversationHandle


@dataclasses.dataclass(frozen=True)
class _PersistedRevision:
    """A completed revision or check-repair pass read back from its artifacts."""

    final_message: str
    usage: dict[str, int]
    tree_before: str
    tree_after: str
    exit_code: int = 0
    timed_out: bool = False
    stderr_tail: str = ""


def reusable_pre_checks(artifact_dir: Path, tree: str) -> dict[str, Any] | None:
    """Durable pre-revision evidence frozen for exactly *tree*, if any."""

    payload = read_json_artifact(artifact_dir / "pre_checks.json")
    if not isinstance(payload, dict) or payload.get("staged_tree_sha") != tree:
        return None
    failures = payload.get("failures")
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        return None
    if hard_failure_items(failures):
        return None
    evidence = read_json_artifact(artifact_dir / "evidence.json")
    if not isinstance(evidence, dict) or evidence.get("staged_tree_sha") != tree:
        return None
    try:
        # Keep the durable evidence existence check, but never return its
        # contents to a worker prompt.
        (artifact_dir / "diff.patch").read_bytes()
    except OSError:
        return None
    return payload


def load_evidence(directory: Path) -> EvidenceBundle | None:
    """Rebuild a frozen evidence bundle from ``evidence.json``."""

    payload = read_json_artifact(directory / "evidence.json")
    if not isinstance(payload, dict):
        return None
    changed = payload.get("changed_files")
    checks = payload.get("checks")
    failures = payload.get("failures")
    if (
        not is_object_id(payload.get("base_sha"))
        or not is_object_id(payload.get("staged_tree_sha"))
        or not isinstance(payload.get("diff"), str)
        or not isinstance(payload.get("deterministic_passed"), bool)
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
        or not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks)
        or not isinstance(failures, list) or any(not isinstance(item, str) for item in failures)
    ):
        return None
    return EvidenceBundle(
        base_sha=payload["base_sha"],
        staged_tree_sha=payload["staged_tree_sha"],
        changed_files=tuple(changed),
        diff=payload["diff"],
        checks=tuple(checks),
        deterministic_passed=payload["deterministic_passed"],
        failures=tuple(failures),
        required_check_ids=tuple(
            item for item in payload.get("required_check_ids", [])
            if isinstance(item, str)
        ),
    )


def accepted_review(
    directory: Path, evidence: EvidenceBundle, candidate_sha: str | None = None,
) -> ReviewResult | None:
    """A reviewer answer already accepted for exactly this candidate tree."""

    if not (directory / "review.json").is_file():
        return None
    try:
        request = (directory / "reviewer.request.txt").read_text(encoding="utf-8")
        raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
        persisted = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return None
    except json.JSONDecodeError:
        return None
    # The request is the exact evidence the answer was given: it must name
    # both the reviewed tree and the immutable candidate commit.
    if evidence.staged_tree_sha not in request:
        return None
    if candidate_sha is not None and candidate_sha not in request:
        return None
    try:
        review = parse_review(raw, deterministic_passed=evidence.deterministic_passed)
    except ReviewParseError:
        return None
    normalized = dataclasses.asdict(review)
    normalized["verdict"] = review.verdict.value
    normalized["route"] = review.route.value
    if persisted != normalized:
        return None
    return review


def load_completed_step(step_dir: Path, step_id: str) -> dict[str, Any] | None:
    """One completed step record."""

    record = read_json_artifact(step_dir / "step.json", 128 * 1024)
    if not isinstance(record, dict) or record.get("id") != step_id:
        return None
    status = record.get("status")
    if status != "COMPLETED":
        return None
    changed = record.get("changed_paths")
    if (
        not is_object_id(record.get("tree_before"))
        or not is_object_id(record.get("tree_after"))
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
    ):
        return None
    no_change = record.get("no_change", False)
    if not isinstance(no_change, bool) or (no_change and (
        record["tree_before"] != record["tree_after"] or changed
    )):
        return None
    if record["tree_before"] != record["tree_after"] and not is_object_id(record.get("commit_sha")):
        # A successful worker whose candidate was not accepted yet: the step
        # is complete only once its commit crossed the acceptance boundary.
        return None
    extra = record.get("out_of_scope_paths")
    if not isinstance(extra, list) or any(not isinstance(path, str) for path in extra):
        extra = []
    return {
        "id": step_id, "status": status, "profile_id": record.get("profile_id"),
        "tree_before": record["tree_before"], "tree_after": record["tree_after"],
        "changed_paths": list(changed),
        **({"no_change": True} if no_change else {}),
        **({"out_of_scope_paths": sorted(set(extra))} if extra else {}),
        "usage": normalize_usage(record.get("usage")),
        "final": bounded_v2_report(read_bounded_text(step_dir / "agent.final.md")),
        **({"initial_mismatch": bounded_v2_report(str(record["initial_mismatch"]))}
           if isinstance(record.get("initial_mismatch"), str) and record["initial_mismatch"].strip()
           else {}),
        **({"mismatch_retry_count": record["mismatch_retry_count"]}
           if isinstance(record.get("mismatch_retry_count"), int) else {}),
        **({"deferred_verify": bounded_v2_report(str(record["deferred_verify"]))}
           if isinstance(record.get("deferred_verify"), str) and record["deferred_verify"].strip()
           else {}),
    }


# Durable outcomes of a step the run settled without completing it.
FAILED_CONTINUED = "FAILED_CONTINUED"
SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"


def settled_step_status(directory: Path, step_id: str) -> str | None:
    """``FAILED_CONTINUED`` or ``SKIPPED_DEPENDENCY`` once a step is settled."""

    record = read_json_artifact(directory / "step.json", 128 * 1024)
    if not isinstance(record, dict) or record.get("id") != step_id:
        return None
    status = record.get("status")
    return status if status in {FAILED_CONTINUED, SKIPPED_DEPENDENCY} else None


def completed_step_records(
    run_dir: Path, cycle: int, step_ids: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """The durable completed prefix of one cycle's approved steps.

    A settled step (failed and continued, or skipped for its dependency) is
    stepped over: the steps after it still belong to the prefix.
    """

    records: list[dict[str, Any]] = []
    for step_id in step_ids:
        directory = step_dir(run_dir, cycle, step_id)
        record = load_completed_step(directory, step_id)
        if record is None:
            if settled_step_status(directory, step_id) is not None:
                continue
            break
        records.append(record)
    return records


def load_revision(directory: Path) -> _PersistedRevision | None:
    report = read_json_artifact(directory / "report.json", 1024 * 1024)
    if not isinstance(report, dict) or report.get("status") not in {"COMPLETED", "NO_CHANGE"}:
        return None
    if not is_object_id(report.get("tree_before")) or not is_object_id(report.get("tree_after")):
        return None
    final = read_bounded_text(directory / "agent.final.md", MAX_AGENT_REPORT_BYTES * 2)
    if not final and isinstance(report.get("final"), str):
        final = report["final"]
    usage = read_usage_artifact(directory / "usage.json") or normalize_usage(report.get("usage"))
    return _PersistedRevision(final, usage, report["tree_before"], report["tree_after"])


def read_candidate_record(run_dir: Path, number: int) -> dict[str, Any]:
    """One cycle's immutable candidate commit record."""

    payload = read_json_artifact(candidate_dir(run_dir, number) / "commit.json")
    no_change = payload.get("no_change", False) if isinstance(payload, dict) else False
    if (
        not isinstance(payload, dict)
        or not is_object_id(payload.get("commit_sha"))
        or not is_object_id(payload.get("tree_sha"))
        or not isinstance(no_change, bool)
        or (
            not is_object_id(payload.get("parent_sha"))
            and not (no_change and payload.get("parent_sha") is None)
        )
    ):
        raise ResumeIntegrityError(f"cycle {number:03d} candidate commit record is missing")
    return payload


def candidate_evidence(run_dir: Path, number: int) -> EvidenceBundle | None:
    """The gate evidence one cycle's candidate commit answers for."""

    stage = read_candidate_record(run_dir, number).get("gate_stage")
    try:
        return load_evidence(gate_dir(run_dir, number, stage))
    except ValueError:
        return None


def read_planner_conversation(run_dir: Path) -> LLMConversationHandle | None:
    payload = read_json_artifact(run_dir / PLANNER_CONVERSATION, 4096)
    if not isinstance(payload, dict):
        return None
    try:
        return LLMConversationHandle(payload.get("provider_id"), payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


def read_repository_reference(run_dir: Path) -> RepositoryReference | None:
    payload = read_json_artifact(run_dir / "repository_reference.json", 16 * 1024)
    if not isinstance(payload, dict) or set(payload) != {"remote_name", "web_url", "base_sha", "immutable_url"}:
        return None
    if not isinstance(payload["remote_name"], str) or not is_object_id(payload["base_sha"]):
        return None
    if any(payload[key] is not None and not isinstance(payload[key], str) for key in ("web_url", "immutable_url")):
        return None
    return RepositoryReference(**payload)


CYCLE_SCOPE_SOURCE = "cycle mutable scope"
EVIDENCE_SCOPE_SOURCE = "failure evidence within approved mutable scope"
SCOPE_REQUEST_SOURCE = "explicit META SCOPE REQUEST v1"
GATE_SCOPE_EXPANSION_ARTIFACT = "scope-expansion.json"


def mutable_scope_sha256(paths: Sequence[str]) -> str:
    """Hash the canonical, sorted JSON representation of a mutable scope."""

    return hashlib.sha256(json_text(sorted(set(paths))).encode("utf-8")).hexdigest()


def _read_scope_artifact(directory: Path, *, fallback_base: Sequence[str]) -> CheckRepairScope:
    """Read one recorded check-repair scope; a malformed one fails closed."""

    payload = read_json_artifact(directory / "scope.json", 64 * 1024)
    approved = tuple(sorted(set(fallback_base)))
    if not isinstance(payload, dict) or payload.get("schema_version") != 4:
        raise ResumeIntegrityError("check-repair scope artifact is malformed or unsupported")
    raw = [
        payload.get("approved_mutable_scope"), payload.get("initial_repair_scope"),
        payload.get("added_paths"), payload.get("effective_repair_scope"),
    ]
    if not all(isinstance(value, list) for value in raw):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    try:
        parsed = [tuple(sorted(set(normalize_repo_paths(paths)))) for paths in raw]
    except ValueError as exc:
        raise ResumeIntegrityError(
            "check-repair scope artifact contains invalid paths"
        ) from exc
    if any(raw[index] != list(value) for index, value in enumerate(parsed)):
        raise ResumeIntegrityError("check-repair scope artifact is not canonical")
    parsed_approved, parsed_initial, parsed_added, parsed_effective = parsed
    if (
        not set(approved).issubset(parsed_approved)
        or not set(parsed_initial).issubset(parsed_approved)
        or not set(parsed_added).issubset(parsed_approved)
        or set(parsed_initial) & set(parsed_added)
        or parsed_effective != tuple(sorted(set(parsed_initial) | set(parsed_added)))
    ):
        raise ResumeIntegrityError("check-repair scope artifact does not match its envelope")
    source = payload.get("source")
    if source not in {CYCLE_SCOPE_SOURCE, EVIDENCE_SCOPE_SOURCE, SCOPE_REQUEST_SOURCE}:
        raise ResumeIntegrityError("check-repair scope artifact has invalid provenance")
    if parsed_added and source != SCOPE_REQUEST_SOURCE:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    if not parsed_added and source not in {CYCLE_SCOPE_SOURCE, EVIDENCE_SCOPE_SOURCE}:
        raise ResumeIntegrityError("check-repair initial scope has an invalid provenance")
    return CheckRepairScope(
        approved_mutable_scope=parsed_approved,
        initial_repair_scope=parsed_initial,
        added_paths=parsed_added,
        effective_repair_scope=parsed_effective,
        source=source,
    )


def _pending_scope_expansion(
    run_dir: Path, cycle: int, stage: GateStage, attempt: int,
) -> tuple[str, ...]:
    """The ladder expansion authorized for one not-yet-recorded repair pass."""

    directory = check_repair_attempt_dir(run_dir, cycle, stage, attempt)
    if (directory / "scope.json").is_file():
        # The pass is recorded: its own scope artifact is authoritative.
        return ()
    payload = read_json_artifact(directory / GATE_SCOPE_EXPANSION_ARTIFACT, 64 * 1024)
    if payload is None:
        return ()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ResumeIntegrityError("gate recovery scope expansion artifact is malformed")
    if payload.get("attempt") != attempt:
        raise ResumeIntegrityError("gate recovery scope expansion belongs to another attempt")
    failed = payload.get("failed_check_ids")
    if not isinstance(failed, list) or any(not isinstance(name, str) or not name for name in failed):
        raise ResumeIntegrityError("gate recovery scope expansion has invalid failed checks")
    added = payload.get("added_paths")
    if not isinstance(added, list) or any(not isinstance(item, str) for item in added):
        raise ResumeIntegrityError("gate recovery scope expansion contains invalid paths")
    canonical = tuple(sorted(set(added)))
    if list(canonical) != added:
        raise ResumeIntegrityError("gate recovery scope expansion is not canonical")
    try:
        normalize_repo_paths(canonical)
    except ValueError as exc:
        raise ResumeIntegrityError(
            "gate recovery scope expansion contains unsafe paths"
        ) from exc
    return canonical


def _with_ladder_expansion(
    authority: GateMutableAuthority, *,
    run_dir: Path, cycle: int, stage: GateStage, through_attempt: int | None,
) -> GateMutableAuthority:
    """Fold the ladder expansion of the pending repair pass into its authority."""

    if through_attempt is None:
        return authority
    added = _pending_scope_expansion(run_dir, cycle, stage, through_attempt + 1)
    if not added:
        return authority
    effective = tuple(sorted(set(authority.effective_paths) | set(added)))
    return GateMutableAuthority(
        base_paths=authority.base_paths,
        added_paths=tuple(sorted(set(authority.added_paths) | set(added))),
        effective_paths=effective,
        source=SCOPE_REQUEST_SOURCE,
        sha256=mutable_scope_sha256(effective),
        initial_paths=authority.initial_paths,
    )


def gate_mutable_authority(
    run_dir: Path,
    cycle: int,
    stage: GateStage | str,
    *,
    base_paths: Sequence[str],
    through_attempt: int | None = None,
    require_attempt_records: bool = False,
) -> GateMutableAuthority:
    """Rebuild and validate the exact mutation authority of one gate episode."""

    base = tuple(sorted(set(base_paths)))
    root = check_repair_attempts_dir(run_dir, cycle, stage)
    if not root.is_dir():
        return GateMutableAuthority(
            base_paths=base, added_paths=(), effective_paths=base,
            source=CYCLE_SCOPE_SOURCE, sha256=mutable_scope_sha256(base), initial_paths=(),
        )
    if through_attempt is not None and (
        isinstance(through_attempt, bool)
        or not isinstance(through_attempt, int)
        or through_attempt < 0
    ):
        raise ResumeIntegrityError("check-repair scope attempt bound is invalid")
    directories = sorted(
        (
            path for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (through_attempt is None or int(path.name) <= through_attempt)
        ),
        key=lambda path: int(path.name),
    )
    if not directories:
        if through_attempt:
            raise ResumeIntegrityError("check-repair scope attempts are missing")
        return _with_ladder_expansion(GateMutableAuthority(
            base_paths=base, added_paths=(), effective_paths=base,
            source=CYCLE_SCOPE_SOURCE, sha256=mutable_scope_sha256(base), initial_paths=(),
        ), run_dir=run_dir, cycle=cycle, stage=stage, through_attempt=through_attempt)
    scopes: list[CheckRepairScope] = []
    for expected, directory in enumerate(directories, start=1):
        if int(directory.name) != expected:
            raise ResumeIntegrityError("check-repair scope attempts are not contiguous")
        if not (directory / "scope.json").is_file():
            raise ResumeIntegrityError("check-repair scope attempt artifacts are incomplete")
        scope = _read_scope_artifact(directory, fallback_base=base)
        if require_attempt_records:
            attempt = read_json_artifact(directory / "attempt.json", 128 * 1024)
            if (
                not isinstance(attempt, dict)
                or attempt.get("number") != expected
                or attempt.get("mutable_scope") != list(scope.effective_repair_scope)
            ):
                raise ResumeIntegrityError("check-repair attempt is not bound to its scope")
        if scopes and not set(scopes[-1].added_paths).issubset(scope.added_paths):
            raise ResumeIntegrityError("check-repair scope additions are not cumulative")
        if scopes and scopes[-1].initial_repair_scope != scope.initial_repair_scope:
            raise ResumeIntegrityError("check-repair initial scope changed between attempts")
        scopes.append(scope)
    if through_attempt is not None and len(scopes) != through_attempt:
        raise ResumeIntegrityError("check-repair scope attempts are not contiguous")
    final = scopes[-1]
    return _with_ladder_expansion(GateMutableAuthority(
        base_paths=final.approved_mutable_scope,
        added_paths=final.added_paths,
        effective_paths=final.effective_repair_scope,
        source=final.source,
        sha256=mutable_scope_sha256(final.effective_repair_scope),
        initial_paths=final.initial_repair_scope,
    ), run_dir=run_dir, cycle=cycle, stage=stage, through_attempt=through_attempt)


__all__ = [
    "CYCLE_SCOPE_SOURCE", "EVIDENCE_SCOPE_SOURCE", "GATE_SCOPE_EXPANSION_ARTIFACT",
    "SCOPE_REQUEST_SOURCE",
    "FAILED_CONTINUED", "SKIPPED_DEPENDENCY",
    "accepted_review", "candidate_evidence", "completed_step_records",
    "gate_mutable_authority", "load_completed_step", "load_evidence", "load_revision",
    "mutable_scope_sha256", "settled_step_status",
    "read_candidate_record", "read_planner_conversation",
    "read_repository_reference", "reusable_pre_checks",
]
