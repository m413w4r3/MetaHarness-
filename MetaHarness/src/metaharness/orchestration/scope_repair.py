"""Correction-cycle mutable-scope deltas and their approval."""

from __future__ import annotations

import hashlib

from pathlib import (
    Path,
    PurePosixPath,
)
from typing import Any
from .shared import (
    OrchestrationError,
    _create_file_once,
    _json_text,
)
from ..models import TaskPlanV2
from ..resume import ResumeIntegrityError
from ..review import ReviewResult


def _repair_mutation_sets(plan: TaskPlanV2) -> tuple[list[str], list[str], list[str]]:
    """Return canonical correction mutation sets and reject structural ambiguity."""

    writes = sorted({path for step in plan.steps for path in step.write_set})
    creates = sorted({path for step in plan.steps for path in step.create_set})
    deletes = sorted({path for step in plan.steps for path in step.delete_set})
    if (set(writes) & set(creates)) or (set(writes) & set(deletes)) or (set(creates) & set(deletes)):
        raise OrchestrationError("REPAIR_SCOPE_MUTATION_SETS_OVERLAP")
    for path in (*writes, *creates, *deletes):
        posix = PurePosixPath(path)
        if (not path or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in posix.parts)
                or any(char in path for char in "*?[")):
            raise OrchestrationError("REPAIR_SCOPE_UNSAFE_PATH")
    return writes, creates, deletes


def _build_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, review: ReviewResult, repair_bundle_sha: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta of one review-driven correction cycle."""

    return build_scope_delta(
        repair_dir, original_scope=original_scope, plan=plan,
        candidate_commit_sha=candidate_commit_sha, repair_bundle_sha=repair_bundle_sha,
        justification=review.required_fixes.strip() or review.findings.strip(),
    )


def build_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, repair_bundle_sha: str, justification: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta, in memory only: from parsed plan sets.

    The delta is derived from the parsed plan alone; *justification* names the
    durable failure the added paths answer, and is never reviewer or planner
    prose.  Every plan that may widen an approved envelope -- a review
    correction and a red-gate cycle replan alike -- goes through this one
    builder, so the same plan always produces the same delta bytes.
    """

    writes, creates, deletes = _repair_mutation_sets(plan)
    requested = sorted(set(writes) | set(creates) | set(deletes))
    original = sorted(set(original_scope))
    added = sorted(set(requested) - set(original))
    unchanged = sorted(set(requested) & set(original))
    findings = justification.strip()
    reasons: dict[str, Any] = {}
    for path in added:
        steps = [step for step in plan.steps if path in set(step.write_set) | set(step.create_set) | set(step.delete_set)]
        step = steps[0]
        reasons[path] = {
            "reason": f"{step.title}: {step.objective}",
            "source_finding": findings,
        }
    try:
        raw_plan = (repair_dir / "planner.raw.md").read_bytes()
    except OSError as exc:
        raise OrchestrationError("REPAIR_SCOPE_PLAN_UNREADABLE") from exc
    plan_sha = hashlib.sha256(raw_plan).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "original_mutable_paths": original,
        "requested_write_paths": writes,
        "requested_create_paths": creates,
        "requested_delete_paths": deletes,
        "added_paths": added,
        "unchanged_paths": unchanged,
        "added_path_reasons": reasons,
        "source_finding": findings,
        "candidate_commit_sha": candidate_commit_sha,
        "repair_plan_sha256": plan_sha,
        "correction_bundle_sha256": repair_bundle_sha,
    }
    return payload, _json_text(payload)


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
            _create_file_once(path, expected)
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
