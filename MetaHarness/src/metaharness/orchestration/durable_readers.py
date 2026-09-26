"""Fail-closed readers of the durable artifacts a resume depends on.

Every reader here is bounded and side-effect free: an artifact that does not
prove exactly what its caller needs yields ``None``, never a partially trusted
value.  Nothing in this module writes Git, the run state or any durable
artifact.
"""

from __future__ import annotations

import dataclasses
import json

from pathlib import Path
from typing import Any
from .check_failure import hard_failure_items
from .pipeline_v2 import (
    candidate_dir,
    gate_dir,
    step_dir,
)
from .shared import (
    MAX_AGENT_REPORT_BYTES,
    PLANNER_CONVERSATION,
    bounded_v2_report,
    is_object_id,
    read_bounded_text,
    read_json_artifact,
)
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
    return {
        "id": step_id, "status": status, "profile_id": record.get("profile_id"),
        "tree_before": record["tree_before"], "tree_after": record["tree_after"],
        "changed_paths": list(changed),
        **({"no_change": True} if no_change else {}),
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


def completed_step_records(
    run_dir: Path, cycle: int, step_ids: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """The durable completed prefix of one cycle's approved steps."""

    records: list[dict[str, Any]] = []
    for step_id in step_ids:
        record = load_completed_step(step_dir(run_dir, cycle, step_id), step_id)
        if record is None:
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


__all__ = [
    "accepted_review", "candidate_evidence", "completed_step_records",
    "load_completed_step", "load_evidence", "load_revision",
    "read_candidate_record", "read_planner_conversation",
    "read_repository_reference", "reusable_pre_checks",
]
