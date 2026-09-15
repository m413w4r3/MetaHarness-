"""Durable resume checkpoints.

A checkpoint always names the *next operation that has not yet succeeded*:
after Luna S01 completes, the checkpoint is ``checks_c01`` with the post-Luna
candidate tree; with Claude enabled, ``checks_c01`` means the pre-revision
checks are pending, ``claude_c01`` that Claude is next, and
``final_checks_c01`` (post-Claude tree) that Claude completed durably and only
the final checks remain.  After the final checks it is
``candidate_commit_c01``.  Candidate commit and push checkpoints precede
``reviewer_c01``; C02 uses the same shape.  An operation is never marked
complete before all of its mandatory artifacts are durable, so a resume never
replays a phase that already succeeded.

This module is deliberately read-only with respect to Git and never calls a
model: it persists and reads ``resume_checkpoint.json`` and decides, cheaply,
whether a failed run *may* be resumed.  The authoritative fail-closed
integrity validation happens in the orchestrator immediately before a resume.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .approval import ApprovalError, PlanIdentity
from .result import atomic_write_text

CHECKPOINT_NAME = "resume_checkpoint.json"
_SCHEMA_VERSION = 2
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STEP_ID = re.compile(r"S0[1-6]")
_MAX_CHECKPOINT_BYTES = 16 * 1024


class ResumePhase(StrEnum):
    CONTEXT = "context"
    PLANNER = "planner"
    PLAN_APPROVAL = "plan_approval"
    WORKTREE_SETUP = "worktree_setup"
    INITIAL_STEP = "initial_step"
    CHECKS_C01 = "checks_c01"
    CLAUDE_C01 = "claude_c01"
    FINAL_CHECKS_C01 = "final_checks_c01"
    CANDIDATE_COMMIT_C01 = "candidate_commit_c01"
    CANDIDATE_PUSH_C01 = "candidate_push_c01"
    REVIEWER_C01 = "reviewer_c01"
    REPAIR_PLANNER = "repair_planner"
    SCOPE_APPROVAL = "scope_approval"
    REPAIR_STEP = "repair_step"
    CHECKS_C02 = "checks_c02"
    CLAUDE_C02 = "claude_c02"
    FINAL_CHECKS_C02 = "final_checks_c02"
    CANDIDATE_COMMIT_C02 = "candidate_commit_c02"
    CANDIDATE_PUSH_C02 = "candidate_push_c02"
    REVIEWER_C02 = "reviewer_c02"
    COMMIT = "com" + "mit"
    PUBLISH = "publish"


_PHASE_ORDER = {phase: index for index, phase in enumerate(ResumePhase)}
_PHASE_CYCLE = {
    ResumePhase.INITIAL_STEP: 1,
    ResumePhase.CLAUDE_C01: 1,
    ResumePhase.CANDIDATE_COMMIT_C01: 1,
    ResumePhase.CANDIDATE_PUSH_C01: 1,
    ResumePhase.REVIEWER_C01: 1,
    ResumePhase.REPAIR_PLANNER: 2,
    ResumePhase.SCOPE_APPROVAL: 2,
    ResumePhase.REPAIR_STEP: 2,
    ResumePhase.CLAUDE_C02: 2,
    ResumePhase.CANDIDATE_COMMIT_C02: 2,
    ResumePhase.CANDIDATE_PUSH_C02: 2,
    ResumePhase.REVIEWER_C02: 2,
}
_PRE_PLAN_PHASES = frozenset({ResumePhase.CONTEXT, ResumePhase.PLANNER})
_PRE_APPROVAL_PHASES = frozenset({ResumePhase.CONTEXT, ResumePhase.PLANNER, ResumePhase.PLAN_APPROVAL})
_NO_WORKTREE_PHASES = _PRE_APPROVAL_PHASES | frozenset({ResumePhase.WORKTREE_SETUP})
_STEP_PHASES = frozenset({ResumePhase.INITIAL_STEP, ResumePhase.REPAIR_STEP})


def phase_index(phase: ResumePhase) -> int:
    """Position of *phase* in the fixed pipeline order."""

    return _PHASE_ORDER[ResumePhase(phase)]


class ResumeCheckpointError(ValueError):
    """``resume_checkpoint.json`` is malformed."""


@dataclass(frozen=True)
class ResumeCheckpoint:
    phase: ResumePhase
    cycle: int
    step_id: str | None
    expected_head_sha: str | None = None
    expected_tree_sha: str | None = None
    execution_selection_sha256: str | None = None
    plan_identity: PlanIdentity | None = None
    # SHA-256 of ``repair/C02/implementation_bundle.json`` once the repair
    # planner succeeded.  C02 bundles are not human-approved, so this is the
    # only binding between a resumed C02 phase and the repair plan it runs.
    repair_bundle_sha256: str | None = None
    scope_delta_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            phase = ResumePhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ResumeCheckpointError("checkpoint phase is unknown") from exc
        object.__setattr__(self, "phase", phase)
        expected_cycle = _PHASE_CYCLE.get(phase)
        if isinstance(self.cycle, bool) or self.cycle not in (1, 2) or (
            expected_cycle is not None and self.cycle != expected_cycle
        ):
            raise ResumeCheckpointError("checkpoint cycle does not match its phase")
        if phase in _STEP_PHASES:
            if not isinstance(self.step_id, str) or _STEP_ID.fullmatch(self.step_id) is None:
                raise ResumeCheckpointError("checkpoint step_id is invalid")
        elif self.step_id is not None:
            raise ResumeCheckpointError("checkpoint step_id is only valid for step phases")
        for label, value in (
            ("expected_head_sha", self.expected_head_sha),
            ("expected_tree_sha", self.expected_tree_sha),
        ):
            if value is not None and (not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None):
                raise ResumeCheckpointError(f"checkpoint {label} is invalid")
        if self.execution_selection_sha256 is not None and (
            not isinstance(self.execution_selection_sha256, str)
            or _SHA256.fullmatch(self.execution_selection_sha256) is None
        ):
            raise ResumeCheckpointError("checkpoint execution_selection_sha256 is invalid")
        if self.plan_identity is not None and not isinstance(self.plan_identity, PlanIdentity):
            raise ResumeCheckpointError("checkpoint plan_identity is invalid")
        if phase not in _PRE_PLAN_PHASES and self.plan_identity is None:
            raise ResumeCheckpointError("checkpoint plan identity is required for this phase")
        if phase not in _PRE_APPROVAL_PHASES and self.execution_selection_sha256 is None:
            raise ResumeCheckpointError("checkpoint execution selection hash is required for this phase")
        if phase not in _NO_WORKTREE_PHASES and (
            self.expected_head_sha is None or self.expected_tree_sha is None
        ):
            raise ResumeCheckpointError("checkpoint Git identity is required for this phase")
        if self.repair_bundle_sha256 is not None and (
            not isinstance(self.repair_bundle_sha256, str)
            or _SHA256.fullmatch(self.repair_bundle_sha256) is None
        ):
            raise ResumeCheckpointError("checkpoint repair_bundle_sha256 is invalid")
        if self.scope_delta_sha256 is not None and (
            not isinstance(self.scope_delta_sha256, str)
            or _SHA256.fullmatch(self.scope_delta_sha256) is None
        ):
            raise ResumeCheckpointError("checkpoint scope_delta_sha256 is invalid")
        if self.cycle == 2 and phase not in {
            ResumePhase.REPAIR_PLANNER, ResumePhase.SCOPE_APPROVAL,
            ResumePhase.REPAIR_STEP, ResumePhase.PUBLISH,
        }:
            if self.repair_bundle_sha256 is None:
                raise ResumeCheckpointError("C02 checkpoint requires the repair bundle hash")


def _identity_payload(identity: PlanIdentity) -> dict[str, str | None]:
    return {
        "raw_sha256": identity.raw_sha256,
        "contract_sha256": identity.contract_sha256,
        "bundle_sha256": identity.bundle_sha256,
        "execution_sha256": identity.execution_sha256,
    }


def plan_identity_from_mapping(value: Any) -> PlanIdentity:
    """Strictly rebuild a :class:`PlanIdentity` from persisted JSON."""

    if not isinstance(value, Mapping):
        raise ResumeCheckpointError("plan identity must be an object")
    try:
        return PlanIdentity(
            raw_sha256=value["raw_sha256"],
            contract_sha256=value["contract_sha256"],
            bundle_sha256=value.get("bundle_sha256"),
            execution_sha256=value.get("execution_sha256"),
        )
    except (KeyError, TypeError, ApprovalError) as exc:
        raise ResumeCheckpointError("plan identity is invalid") from exc


def checkpoint_payload(checkpoint: ResumeCheckpoint, *, status: str = "pending") -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "phase": checkpoint.phase.value,
        "cycle": checkpoint.cycle,
        "step_id": checkpoint.step_id,
        "expected_head_sha": checkpoint.expected_head_sha,
        "expected_tree_sha": checkpoint.expected_tree_sha,
        "execution_selection_sha256": checkpoint.execution_selection_sha256,
        "plan_identity": _identity_payload(checkpoint.plan_identity) if checkpoint.plan_identity else None,
        "repair_bundle_sha256": checkpoint.repair_bundle_sha256,
        "scope_delta_sha256": checkpoint.scope_delta_sha256,
    }


def write_checkpoint(run_dir: str | Path, checkpoint: ResumeCheckpoint) -> None:
    """Atomically persist the next operation that has not yet succeeded."""

    if not isinstance(checkpoint, ResumeCheckpoint):
        raise TypeError("checkpoint must be a ResumeCheckpoint")
    atomic_write_text(
        Path(run_dir) / CHECKPOINT_NAME,
        json.dumps(checkpoint_payload(checkpoint), indent=2, sort_keys=True) + "\n",
    )


def _parse(payload: Any) -> tuple[ResumeCheckpoint, str]:
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, _SCHEMA_VERSION}:
        raise ResumeCheckpointError("checkpoint schema_version is unsupported")
    status = payload.get("status")
    if status not in {"pending", "completed"}:
        raise ResumeCheckpointError("checkpoint status is invalid")
    checkpoint = ResumeCheckpoint(
        phase=payload.get("phase"),
        cycle=payload.get("cycle"),
        step_id=payload.get("step_id"),
        expected_head_sha=payload.get("expected_head_sha"),
        expected_tree_sha=payload.get("expected_tree_sha"),
        execution_selection_sha256=payload.get("execution_selection_sha256"),
        plan_identity=(
            plan_identity_from_mapping(payload.get("plan_identity"))
            if payload.get("plan_identity") is not None else None
        ),
        repair_bundle_sha256=payload.get("repair_bundle_sha256"),
        scope_delta_sha256=payload.get("scope_delta_sha256"),
    )
    return checkpoint, status


def read_checkpoint_record(run_dir: str | Path) -> tuple[ResumeCheckpoint, str] | None:
    """Return ``(checkpoint, status)`` or ``None`` when no file exists.

    A malformed file raises :class:`ResumeCheckpointError`: it is never
    silently treated as absent.
    """

    path = Path(run_dir) / CHECKPOINT_NAME
    try:
        if path.stat().st_size > _MAX_CHECKPOINT_BYTES:
            raise ResumeCheckpointError("checkpoint is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeCheckpointError("checkpoint is unreadable") from exc
    return _parse(payload)


def read_checkpoint(run_dir: str | Path) -> ResumeCheckpoint | None:
    """The pending checkpoint, or ``None`` when absent or completed."""

    record = read_checkpoint_record(run_dir)
    if record is None or record[1] != "pending":
        return None
    return record[0]


def mark_checkpoint_completed(run_dir: str | Path) -> None:
    """Mark the run's checkpoint completed (nothing left to resume)."""

    try:
        record = read_checkpoint_record(run_dir)
    except ResumeCheckpointError:
        raise
    if record is None:
        return
    atomic_write_text(
        Path(run_dir) / CHECKPOINT_NAME,
        json.dumps(checkpoint_payload(record[0], status="completed"), indent=2, sort_keys=True) + "\n",
    )


# Kept as a source-compatibility export for integrations that imported it in
# P29.  Resumability is no longer decided from this table: the authoritative
# criterion is the valid pending checkpoint and its phase-specific invariants.
_CLAUDE_PHASES = frozenset({ResumePhase.CLAUDE_C01, ResumePhase.CLAUDE_C02})
_CODEX_PHASES = _STEP_PHASES
_REVIEWER_PHASES = frozenset({ResumePhase.REVIEWER_C01, ResumePhase.REVIEWER_C02})
RESUMABLE_FAILURES: Mapping[str, frozenset[ResumePhase]] = {
    "CLAUDE_FAILED": _CLAUDE_PHASES,
    "CLAUDE_AUTH_FAILURE": _CLAUDE_PHASES,
    "CLAUDE_TIMEOUT": _CLAUDE_PHASES,
    "CODEX_AUTH_FAILURE": _CODEX_PHASES,
    "AGENT_TIMEOUT": _CODEX_PHASES,
    "AGENT_FAILED": _CODEX_PHASES,
    "REVIEWER_TRANSPORT_FAILURE": _REVIEWER_PHASES,
    "LLM_FAILURE": frozenset({ResumePhase.REPAIR_PLANNER}),
    "WAITING_SCOPE_APPROVAL": frozenset({ResumePhase.SCOPE_APPROVAL}),
    "PUSH_FAILED": frozenset({
        ResumePhase.CANDIDATE_PUSH_C01,
        ResumePhase.CANDIDATE_PUSH_C02,
        ResumePhase.PUBLISH,
    }),
}
# These are not ordinary retryable operation failures.  They mean that the
# authority needed to prove a retry has been lost (or an agent crossed a Git
# boundary), so an operator/new run is required.
_NON_RESUMABLE_FAILURES = frozenset({
    "AGENT_GIT_VIOLATION", "AGENT_COMMITTED", "CLAUDE_COMMITTED",
    "BASE_MOVED_SINCE_RUN", "TOCTOU_FAILURE", "RESUME_INTEGRITY_FAILURE",
    "RESUME_REQUIRES_OPERATOR",
    "STEP_WRITE_SET_VIOLATION", "STEP_CONTRACT_DRIFT", "AGENT_NO_CHANGE",
    "AGENT_CONTRACT_MISMATCH",
    "REVISION_SCOPE_VIOLATION",
    "HUMAN_REQUIRED", "REPAIR_SCOPE_EXPANSION", "REPAIR_SCOPE_BOUND_EXCEEDED",
})
_RESUMABLE_STATUSES = frozenset({"failed", "interrupted", "waiting_scope_approval"})
# Status a claimed resume starts in (the orchestrator refines it afterwards).
PHASE_STATUS = {
    ResumePhase.CONTEXT: "planning",
    ResumePhase.PLANNER: "planning",
    ResumePhase.PLAN_APPROVAL: "awaiting_plan_approval",
    ResumePhase.WORKTREE_SETUP: "preparing",
    ResumePhase.INITIAL_STEP: "implementing",
    ResumePhase.CHECKS_C01: "validating",
    ResumePhase.CLAUDE_C01: "revising",
    ResumePhase.FINAL_CHECKS_C01: "revalidating",
    ResumePhase.CANDIDATE_COMMIT_C01: "approved",
    ResumePhase.CANDIDATE_PUSH_C01: "approved",
    ResumePhase.REVIEWER_C01: "reviewing",
    ResumePhase.REPAIR_PLANNER: "planning",
    ResumePhase.SCOPE_APPROVAL: "waiting_scope_approval",
    ResumePhase.REPAIR_STEP: "implementing",
    ResumePhase.CHECKS_C02: "revalidating",
    ResumePhase.CLAUDE_C02: "revising",
    ResumePhase.FINAL_CHECKS_C02: "revalidating",
    ResumePhase.CANDIDATE_COMMIT_C02: "approved",
    ResumePhase.CANDIDATE_PUSH_C02: "approved",
    ResumePhase.REVIEWER_C02: "reviewing",
    ResumePhase.COMMIT: "approved",
    ResumePhase.PUBLISH: "publishing",
}


def resume_label(checkpoint: ResumeCheckpoint) -> str:
    """The single primary resume action shown for *checkpoint*."""

    phase = checkpoint.phase
    if phase is ResumePhase.CONTEXT:
        return "Retry context"
    if phase is ResumePhase.PLANNER:
        return "Retry planner"
    if phase is ResumePhase.PLAN_APPROVAL:
        return "Resume plan approval"
    if phase is ResumePhase.WORKTREE_SETUP:
        return "Retry workspace setup"
    if phase is ResumePhase.INITIAL_STEP:
        return f"Retry {checkpoint.step_id}"
    if phase in {ResumePhase.CHECKS_C01, ResumePhase.FINAL_CHECKS_C01}:
        return "Retry checks C01"
    if phase is ResumePhase.REPAIR_STEP:
        return f"Retry C02 {checkpoint.step_id}"
    if phase is ResumePhase.CLAUDE_C01:
        return "Reprendre à partir de Claude"
    if phase is ResumePhase.CANDIDATE_COMMIT_C01:
        return "Create candidate commit C01"
    if phase is ResumePhase.CANDIDATE_PUSH_C01:
        return "Push candidate C01"
    if phase is ResumePhase.CLAUDE_C02:
        return "Reprendre à partir de Claude C02"
    if phase is ResumePhase.CANDIDATE_COMMIT_C02:
        return "Create candidate commit C02"
    if phase is ResumePhase.CANDIDATE_PUSH_C02:
        return "Push candidate C02"
    if phase is ResumePhase.REVIEWER_C01:
        return "Retry reviewer #1"
    if phase is ResumePhase.REVIEWER_C02:
        return "Retry reviewer #2"
    if phase in {ResumePhase.CHECKS_C02, ResumePhase.FINAL_CHECKS_C02}:
        return "Retry checks C02"
    if phase is ResumePhase.COMMIT:
        return "Retry commit"
    if phase is ResumePhase.REPAIR_PLANNER:
        return "Retry repair planner"
    if phase is ResumePhase.SCOPE_APPROVAL:
        return "Resume scope approval"
    return "Retry publish"


def _read_text(path: Path, limit: int = 4096) -> str | None:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        return None
    if len(data) > limit:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeError:
        return None


def _read_json(path: Path, limit: int = 128 * 1024) -> Any:
    text = _read_text(path, limit)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _selection_sha256(run_dir: Path) -> str | None:
    try:
        return hashlib.sha256((run_dir / "execution_selection.json").read_bytes()).hexdigest()
    except OSError:
        return None


def infer_legacy_checkpoint(run_dir: str | Path, state: Mapping[str, Any]) -> ResumeCheckpoint | None:
    """Rebuild a checkpoint for a historical run that predates checkpoints.

    Only two unambiguous shapes are recognized, and only for cycle 1: a
    Claude C01 failure (``revision/tree_before.txt`` present, no
    ``revision/tree_after.txt``) and a Codex step failure recorded in
    ``steps/Sxx/step.json``.  Every value is still re-verified against Git
    and the approval before anything runs.
    """

    directory = Path(run_dir)
    if (directory / CHECKPOINT_NAME).exists():
        return None
    if state.get("planning_protocol") != "v2" or state.get("status") != "failed":
        return None
    if state.get("cycle", 1) != 1:
        return None
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = failure.get("reason")
    base_sha = state.get("base_sha")
    if not isinstance(base_sha, str) or _OBJECT_ID.fullmatch(base_sha) is None:
        return None
    execution_sha = _selection_sha256(directory)
    if execution_sha is None:
        return None
    try:
        identity = plan_identity_from_mapping(state.get("plan_identity"))
    except ResumeCheckpointError:
        return None
    try:
        if reason in {"CLAUDE_FAILED", "CLAUDE_AUTH_FAILURE", "CLAUDE_TIMEOUT"}:
            if (directory / "revision" / "tree_after.txt").exists():
                return None
            tree = (_read_text(directory / "revision" / "tree_before.txt") or "").strip()
            return ResumeCheckpoint(
                ResumePhase.CLAUDE_C01, 1, None, base_sha, tree, execution_sha, identity
            )
        if reason in {"CODEX_AUTH_FAILURE", "AGENT_TIMEOUT", "AGENT_FAILED"}:
            detail = failure.get("detail")
            match = re.match(r"step=(S0[1-6])\b", detail) if isinstance(detail, str) else None
            if match is None:
                return None
            record = _read_json(directory / "steps" / match.group(1) / "step.json")
            if not isinstance(record, dict) or record.get("status") != "FAILED":
                return None
            return ResumeCheckpoint(
                ResumePhase.INITIAL_STEP, 1, match.group(1), base_sha,
                str(record.get("tree_before") or ""), execution_sha, identity,
            )
    except ResumeCheckpointError:
        return None
    return None


def load_resume_checkpoint(
    run_dir: str | Path, state: Mapping[str, Any]
) -> ResumeCheckpoint | None:
    """The pending checkpoint, or an inferred one for a historical run."""

    record = read_checkpoint_record(run_dir)
    if record is not None:
        return record[0] if record[1] == "pending" else None
    return infer_legacy_checkpoint(run_dir, state)


@dataclass(frozen=True)
class ResumeInfo:
    resumable: bool
    phase: str | None = None
    label: str | None = None
    reason: str | None = None
    expected_tree: str | None = None
    cycle: int | None = None
    step_id: str | None = None


def resume_info(run_dir: str | Path, state: Mapping[str, Any]) -> ResumeInfo:
    """Cheap, read-only resumability for the API/UI; no Git, no model.

    ``True`` only means "a validable checkpoint exists for this failure": the
    orchestrator still performs the complete integrity validation and fails
    closed with ``RESUME_INTEGRITY_FAILURE`` when it does not hold.
    """

    directory = Path(run_dir)
    status = state.get("status")
    if status not in _RESUMABLE_STATUSES:
        return ResumeInfo(False, reason="run is not failed or interrupted")
    if state.get("planning_protocol") != "v2":
        return ResumeInfo(False, reason="only META PLAN v2 runs can be resumed")
    try:
        checkpoint = load_resume_checkpoint(directory, state)
    except ResumeCheckpointError:
        return ResumeInfo(False, reason="resume checkpoint is invalid")
    if checkpoint is None:
        return ResumeInfo(False, reason="no resume checkpoint")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    if str(failure.get("reason")) in _NON_RESUMABLE_FAILURES:
        return ResumeInfo(False, reason="failure requires operator intervention")
    # Requirements are deliberately phase-specific.  In particular, context
    # and planner failures are resumable before an approval or worktree exists.
    if checkpoint.phase.value not in {phase.value for phase in _PRE_APPROVAL_PHASES}:
        approval = _read_json(directory / "plan_approval.json", 16 * 1024)
        auto_selected = (directory / "execution_selection.json").is_file()
        if (not isinstance(approval, dict) or approval.get("decision") != "APPROVE") and not auto_selected:
            return ResumeInfo(False, reason="plan approval was not APPROVE")
    if checkpoint.phase not in _NO_WORKTREE_PHASES:
        worktree = state.get("worktree")
        if not isinstance(worktree, str) or not Path(worktree).is_dir():
            return ResumeInfo(False, reason="run worktree is missing")
    return ResumeInfo(
        True, checkpoint.phase.value, resume_label(checkpoint),
        expected_tree=checkpoint.expected_tree_sha, cycle=checkpoint.cycle,
        step_id=checkpoint.step_id,
    )


class ResumeError(RuntimeError):
    """A resume could not start; the run state was not modified."""


class ResumeNotAllowedError(ResumeError):
    """The run has no resumable failure or no validable checkpoint."""


class ResumeIntegrityError(ResumeError):
    """A persisted invariant no longer holds: ``RESUME_INTEGRITY_FAILURE``."""

    code = "RESUME_INTEGRITY_FAILURE"


class ResumeRequiresOperatorError(ResumeError):
    """The state is explainable but unsafe to continue automatically."""

    code = "RESUME_REQUIRES_OPERATOR"


__all__ = [
    "CHECKPOINT_NAME",
    "PHASE_STATUS",
    "RESUMABLE_FAILURES",
    "ResumeCheckpoint",
    "ResumeCheckpointError",
    "ResumeError",
    "ResumeInfo",
    "ResumeIntegrityError",
    "ResumeNotAllowedError",
    "ResumePhase",
    "ResumeRequiresOperatorError",
    "checkpoint_payload",
    "infer_legacy_checkpoint",
    "load_resume_checkpoint",
    "mark_checkpoint_completed",
    "phase_index",
    "plan_identity_from_mapping",
    "read_checkpoint",
    "read_checkpoint_record",
    "resume_info",
    "resume_label",
    "write_checkpoint",
]
