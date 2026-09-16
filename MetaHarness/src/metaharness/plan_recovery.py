"""Operator plan recovery for a run whose planner answer was not executable.

A META PLAN v2 run that failed at its PLANNER checkpoint can receive a
corrected ``STATUS: READY`` plan from the operator.  The replacement goes
through exactly the parser and policies a planner answer goes through, is
published as the run's plan authority and moves the checkpoint to
PLAN_APPROVAL; the normal approval and resume workflow then continues.  No
model is ever called by a recovery.

This module holds the cheap, read-only eligibility test and the durable
``planner_recovery.json`` record.  The Git-bound validation and the artifact
transaction live in :meth:`metaharness.orchestrator.Orchestrator.recover_plan`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .result import atomic_write_text
from .resume import (
    ResumeCheckpointError,
    ResumeNotAllowedError,
    ResumePhase,
    read_checkpoint_record,
)
from .step_ids import is_step_id

PLAN_RECOVERY_ARTIFACT = "planner_recovery.json"
PLAN_RECOVERY_SCHEMA_VERSION = 1
MAX_REPLACEMENT_PLAN_BYTES = 128 * 1024
# Failures whose PLANNER checkpoint proves that no plan was ever published.
RECOVERABLE_PLANNER_FAILURES = frozenset({"PLANNER_OUTPUT_INVALID", "LLM_FAILURE"})
PLAN_SOURCE_OPERATOR = "operator_recovery"
PLAN_SOURCE_PLANNER = "planner_model"
_MAX_RECORD_BYTES = 16 * 1024
# Any of these proves the run went past planning: replacing the plan would
# orphan an approval, a selection or execution evidence derived from it.
_EXECUTION_ARTIFACTS = (
    "plan_approval.json", "execution_selection.json", "scope_approval.json",
    "agent.prompt.txt", "agent.events.jsonl", "agent.result.json", "agent.final.md",
    "agent.stderr.log", "evidence.json", "checks.json", "changed-files.txt", "diff.patch",
    "reviewer.request.txt", "reviewer.raw.md", "reviewer.usage.json", "review.json",
    "repair_task.md", "publish.json",
    "revision", "review", "checks", "repair", "setup", "candidate",
)


class PlanRecoveryError(ResumeNotAllowedError):
    """The run or the replacement plan is not eligible; nothing was changed."""


@dataclass(frozen=True)
class PlanRecoveryInfo:
    eligible: bool
    reason: str | None = None


def _step_directory_has_execution(run_dir: Path) -> bool:
    """True when ``steps/`` holds anything but planning-time contracts."""

    root = run_dir / "steps"
    if not root.exists():
        return False
    if not root.is_dir():
        return True
    for path in root.rglob("*"):
        relative = path.relative_to(root).parts
        if path.is_dir():
            if len(relative) != 1 or not is_step_id(relative[0]):
                return True
            continue
        if len(relative) != 2 or not is_step_id(relative[0]) or relative[1] != "contract.md":
            return True
    return False


def plan_recovery_info(run_dir: str | Path, state: Mapping[str, Any]) -> PlanRecoveryInfo:
    """Cheap, read-only eligibility for REPLACE PLAN; no Git, no model.

    ``eligible`` only means the durable run shape allows a recovery: the
    orchestrator re-checks everything, plus the exact BASE SHA/tree, the
    absence of any run branch or worktree and the replacement plan itself.
    """

    directory = Path(run_dir)
    if state.get("planning_protocol") != "v2":
        return PlanRecoveryInfo(False, "only META PLAN v2 runs can recover a plan")
    if state.get("status") != "failed":
        return PlanRecoveryInfo(False, "run has not failed")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    if failure.get("reason") not in RECOVERABLE_PLANNER_FAILURES:
        return PlanRecoveryInfo(False, "failure is not a recoverable planner failure")
    try:
        record = read_checkpoint_record(directory)
    except ResumeCheckpointError:
        return PlanRecoveryInfo(False, "resume checkpoint is invalid")
    if record is None or record[1] != "pending" or record[0].phase is not ResumePhase.PLANNER:
        return PlanRecoveryInfo(False, "run is not at its PLANNER checkpoint")
    if state.get("branch") or state.get("worktree"):
        return PlanRecoveryInfo(False, "run already has a branch or worktree")
    if any((directory / name).exists() for name in _EXECUTION_ARTIFACTS):
        return PlanRecoveryInfo(False, "run already has approval or execution artifacts")
    if _step_directory_has_execution(directory):
        return PlanRecoveryInfo(False, "run already has step execution artifacts")
    return PlanRecoveryInfo(True)


def validate_replacement_text(value: object) -> str:
    """Bound the operator input; the strict parser remains the authority."""

    if not isinstance(value, str):
        raise PlanRecoveryError("replacement plan must be text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise PlanRecoveryError("replacement plan must be valid UTF-8 text") from exc
    if size > MAX_REPLACEMENT_PLAN_BYTES:
        raise PlanRecoveryError(
            f"replacement plan exceeds {MAX_REPLACEMENT_PLAN_BYTES} bytes"
        )
    if not value.strip():
        raise PlanRecoveryError("replacement plan is empty")
    return value


def read_plan_recovery_record(run_dir: str | Path) -> dict[str, Any] | None:
    """The durable recovery record, or ``None`` when absent or malformed."""

    path = Path(run_dir) / PLAN_RECOVERY_ARTIFACT
    try:
        if path.stat().st_size > _MAX_RECORD_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PLAN_RECOVERY_SCHEMA_VERSION
        or payload.get("source") != "operator"
    ):
        return None
    return payload


def plan_source(run_dir: str | Path) -> str:
    """Whether the current plan came from the operator or the planner model."""

    return PLAN_SOURCE_OPERATOR if read_plan_recovery_record(run_dir) is not None else PLAN_SOURCE_PLANNER


def write_plan_recovery_record(
    run_dir: str | Path,
    *,
    previous_raw_sha256: str | None,
    replacement_raw_sha256: str,
    archived_attempt: str | None,
) -> dict[str, Any]:
    payload = {
        "schema_version": PLAN_RECOVERY_SCHEMA_VERSION,
        "source": "operator",
        "previous_raw_sha256": previous_raw_sha256,
        "replacement_raw_sha256": replacement_raw_sha256,
        "recovered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "archived_attempt": archived_attempt,
        # Explicit: this plan is not a planner completion.
        "planner_called": False,
    }
    atomic_write_text(
        Path(run_dir) / PLAN_RECOVERY_ARTIFACT,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


__all__ = [
    "MAX_REPLACEMENT_PLAN_BYTES",
    "PLAN_RECOVERY_ARTIFACT",
    "PLAN_SOURCE_OPERATOR",
    "PLAN_SOURCE_PLANNER",
    "PlanRecoveryError",
    "PlanRecoveryInfo",
    "RECOVERABLE_PLANNER_FAILURES",
    "plan_recovery_info",
    "plan_source",
    "read_plan_recovery_record",
    "validate_replacement_text",
    "write_plan_recovery_record",
]
