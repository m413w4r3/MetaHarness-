"""Fail-closed readers of the durable artifacts a resume depends on.

Every reader here is bounded and side-effect free: an artifact that does not
prove exactly what its caller needs yields ``None``, never a partially trusted
value.  Nothing in this module writes Git, the run state or any durable
artifact.  The one exception is the durable mutation authority of a gate
episode: it re-proves no path outside the envelope was ever authorized, so it
refuses a tampered artifact instead of returning a partial value.
"""

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import Any, Sequence
from .pipeline_v2 import (
    candidate_dir,
    gate_dir,
    step_dir,
)
from .shared import (
    PLANNER_CONVERSATION,
    GateMutableAuthority,
    bounded_v2_report,
    is_object_id,
    json_text,
    read_bounded_text,
    read_json_artifact,
)
from ..models import GateStage
from ..evidence import EvidenceBundle
from ..gitops import RepositoryReference
from ..resume import ResumeIntegrityError
from ..usage import normalize_usage
from ..llm.chat import LLMConversationHandle


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
        warnings=tuple(
            item for item in payload.get("warnings", []) if isinstance(item, str)
        ),
        baseline_cleared=tuple(
            item for item in payload.get("baseline_cleared", []) if isinstance(item, str)
        ),
    )


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


def mutable_scope_sha256(paths: Sequence[str]) -> str:
    return hashlib.sha256(json_text(sorted(set(paths))).encode("utf-8")).hexdigest()


def gate_mutable_authority(
    run_dir: Path, cycle: int, stage: GateStage | str, *,
    base_paths: Sequence[str],
) -> GateMutableAuthority:
    """The plan's immutable gate scope; audit edits have their own proof."""

    base = tuple(sorted(set(base_paths)))
    return GateMutableAuthority(
        base_paths=base, added_paths=(), effective_paths=base,
        source=CYCLE_SCOPE_SOURCE, sha256=mutable_scope_sha256(base), initial_paths=(),
    )


__all__ = [
    "CYCLE_SCOPE_SOURCE", "FAILED_CONTINUED", "SKIPPED_DEPENDENCY",
    "candidate_evidence", "completed_step_records",
    "gate_mutable_authority", "load_completed_step", "load_evidence",
    "mutable_scope_sha256", "settled_step_status", "read_candidate_record",
    "read_planner_conversation", "read_repository_reference",
]
