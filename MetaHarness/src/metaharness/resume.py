"""Durable, provider-neutral checkpoints for the pipeline v2 state machine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .approval import ApprovalError, PlanIdentity
from .models import GateStage
from .result import atomic_write_text
from .step_ids import STEP_ID_RE

CHECKPOINT_NAME = "resume_checkpoint.json"
_SCHEMA_VERSION = 3
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def pipeline_version_from_state(state: Mapping[str, Any]) -> int:
    if not isinstance(state, Mapping) or state.get("pipeline_version") != 2:
        raise ResumeCheckpointError("pipeline_version must be 2")
    return 2


class ResumePhase(StrEnum):
    CONTEXT = "context"
    PLANNER = "planner"
    PLAN_APPROVAL = "plan_approval"
    WORKTREE_SETUP = "worktree_setup"
    IMPLEMENT_STEP = "implement_step"
    DETERMINISTIC_GATE = "deterministic_gate"
    CHECK_REPAIR = "check_repair"
    SEMANTIC_REVISION = "semantic_revision"
    CANDIDATE_READY = "candidate_ready"
    CANDIDATE_PUSH = "candidate_push"
    FINAL_REVIEW = "final_review"
    REVIEW_IMPLEMENTATION = "review_implementation"
    REVIEW_REPLAN = "review_replan"
    PUBLISH = "publish"


_GATE_STAGES = frozenset(stage.value for stage in GateStage)
_STAGED_PHASES = frozenset({ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR})
_PRE_PLAN = frozenset({ResumePhase.CONTEXT, ResumePhase.PLANNER})
_PRE_APPROVAL = _PRE_PLAN | frozenset({ResumePhase.PLAN_APPROVAL})
_NO_WORKTREE = _PRE_APPROVAL | frozenset({ResumePhase.WORKTREE_SETUP})


class ResumeCheckpointError(ValueError):
    """A checkpoint is malformed or does not bind the next operation."""


@dataclass(frozen=True)
class ResumeCheckpoint:
    phase: ResumePhase
    review_cycle: int = 1
    stage: GateStage | None = None
    step_id: str | None = None
    next_step_id: str | None = None
    check_repair_attempt: int | None = None
    expected_head_sha: str | None = None
    expected_parent_sha: str | None = None
    expected_tree_sha: str | None = None
    execution_selection_sha256: str | None = None
    plan_identity: PlanIdentity | None = None
    correction_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            phase = ResumePhase(self.phase)
        except (TypeError, ValueError) as exc:
            raise ResumeCheckpointError("checkpoint phase is unknown") from exc
        object.__setattr__(self, "phase", phase)
        if isinstance(self.review_cycle, bool) or not isinstance(self.review_cycle, int) or self.review_cycle < 1:
            raise ResumeCheckpointError("checkpoint review_cycle must be a positive integer")
        if self.stage is not None:
            if not isinstance(self.stage, str) or self.stage not in _GATE_STAGES:
                raise ResumeCheckpointError("checkpoint stage is invalid")
            object.__setattr__(self, "stage", GateStage(self.stage))
        if phase in _STAGED_PHASES and self.stage is None:
            raise ResumeCheckpointError("gate and check-repair checkpoints require a stage")
        if phase not in _STAGED_PHASES and self.stage is not None:
            raise ResumeCheckpointError("checkpoint stage is only valid for gate phases")
        step_phases = {ResumePhase.IMPLEMENT_STEP, ResumePhase.REVIEW_IMPLEMENTATION}
        if phase in step_phases:
            if not isinstance(self.step_id, str) or STEP_ID_RE.fullmatch(self.step_id) is None:
                raise ResumeCheckpointError("checkpoint step_id is invalid")
        elif self.step_id is not None:
            raise ResumeCheckpointError("checkpoint step_id is only valid for step phases")
        for name, value in (
            ("next_step_id", self.next_step_id),
        ):
            if value is not None and (not isinstance(value, str) or STEP_ID_RE.fullmatch(value) is None):
                raise ResumeCheckpointError(f"checkpoint {name} is invalid")
        for name, value in (
            ("expected_head_sha", self.expected_head_sha),
            ("expected_parent_sha", self.expected_parent_sha),
            ("expected_tree_sha", self.expected_tree_sha),
        ):
            if value is not None and (not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None):
                raise ResumeCheckpointError(f"checkpoint {name} is invalid")
        for name, value in (
            ("execution_selection_sha256", self.execution_selection_sha256),
            ("correction_bundle_sha256", self.correction_bundle_sha256),
        ):
            if value is not None and (not isinstance(value, str) or _SHA256.fullmatch(value) is None):
                raise ResumeCheckpointError(f"checkpoint {name} is invalid")
        if self.check_repair_attempt is not None and (
            isinstance(self.check_repair_attempt, bool)
            or not isinstance(self.check_repair_attempt, int)
            or self.check_repair_attempt < 1
        ):
            raise ResumeCheckpointError("checkpoint check_repair_attempt is invalid")
        if phase is ResumePhase.CHECK_REPAIR and self.check_repair_attempt is None:
            raise ResumeCheckpointError("check-repair checkpoint requires an attempt")
        if self.plan_identity is not None and not isinstance(self.plan_identity, PlanIdentity):
            raise ResumeCheckpointError("checkpoint plan_identity is invalid")
        if phase not in _PRE_PLAN and self.plan_identity is None:
            raise ResumeCheckpointError("checkpoint plan identity is required for this phase")
        if phase not in _PRE_APPROVAL and self.execution_selection_sha256 is None:
            raise ResumeCheckpointError("checkpoint execution selection hash is required for this phase")
        if phase not in _NO_WORKTREE and (
            self.expected_head_sha is None or self.expected_tree_sha is None
        ):
            raise ResumeCheckpointError("checkpoint Git identity is required for this phase")


def _identity_payload(identity: PlanIdentity) -> dict[str, str | None]:
    return {
        "raw_sha256": identity.raw_sha256,
        "contract_sha256": identity.contract_sha256,
        "bundle_sha256": identity.bundle_sha256,
        "execution_sha256": identity.execution_sha256,
        "checks_sha256": identity.checks_sha256,
    }


def plan_identity_from_mapping(value: Any) -> PlanIdentity:
    if not isinstance(value, Mapping):
        raise ResumeCheckpointError("plan identity must be an object")
    try:
        return PlanIdentity(
            raw_sha256=value["raw_sha256"], contract_sha256=value["contract_sha256"],
            bundle_sha256=value.get("bundle_sha256"), execution_sha256=value.get("execution_sha256"),
            checks_sha256=value.get("checks_sha256"),
        )
    except (KeyError, TypeError, ApprovalError) as exc:
        raise ResumeCheckpointError("plan identity is invalid") from exc


def checkpoint_payload(checkpoint: ResumeCheckpoint, *, status: str = "pending") -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "phase": checkpoint.phase.value,
        "review_cycle": checkpoint.review_cycle,
        "stage": checkpoint.stage.value if checkpoint.stage else None,
        "step_id": checkpoint.step_id,
        "next_step_id": checkpoint.next_step_id,
        "check_repair_attempt": checkpoint.check_repair_attempt,
        "expected_head_sha": checkpoint.expected_head_sha,
        "expected_parent_sha": checkpoint.expected_parent_sha,
        "expected_tree_sha": checkpoint.expected_tree_sha,
        "execution_selection_sha256": checkpoint.execution_selection_sha256,
        "plan_identity": _identity_payload(checkpoint.plan_identity) if checkpoint.plan_identity else None,
        "correction_bundle_sha256": checkpoint.correction_bundle_sha256,
    }


def write_checkpoint(run_dir: str | Path, checkpoint: ResumeCheckpoint) -> None:
    if not isinstance(checkpoint, ResumeCheckpoint):
        raise TypeError("checkpoint must be a ResumeCheckpoint")
    atomic_write_text(
        Path(run_dir) / CHECKPOINT_NAME,
        json.dumps(checkpoint_payload(checkpoint), indent=2, sort_keys=True) + "\n",
    )


def _parse(payload: Any) -> tuple[ResumeCheckpoint, str]:
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ResumeCheckpointError("checkpoint schema_version is unsupported")
    status = payload.get("status")
    if status not in {"pending", "completed"}:
        raise ResumeCheckpointError("checkpoint status is invalid")
    try:
        checkpoint = ResumeCheckpoint(
            phase=payload["phase"], review_cycle=payload.get("review_cycle", 1),
            stage=payload.get("stage"), step_id=payload.get("step_id"),
            next_step_id=payload.get("next_step_id"), check_repair_attempt=payload.get("check_repair_attempt"),
            expected_head_sha=payload.get("expected_head_sha"), expected_parent_sha=payload.get("expected_parent_sha"),
            expected_tree_sha=payload.get("expected_tree_sha"), execution_selection_sha256=payload.get("execution_selection_sha256"),
            plan_identity=(plan_identity_from_mapping(payload["plan_identity"]) if payload.get("plan_identity") is not None else None),
            correction_bundle_sha256=payload.get("correction_bundle_sha256"),
        )
    except KeyError as exc:
        raise ResumeCheckpointError("checkpoint schema is incomplete") from exc
    return checkpoint, status


def read_checkpoint_record(run_dir: str | Path) -> tuple[ResumeCheckpoint, str] | None:
    path = Path(run_dir) / CHECKPOINT_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeCheckpointError("checkpoint is unreadable") from exc
    return _parse(payload)


def read_checkpoint(run_dir: str | Path) -> ResumeCheckpoint | None:
    record = read_checkpoint_record(run_dir)
    return record[0] if record is not None and record[1] == "pending" else None


def mark_checkpoint_completed(run_dir: str | Path) -> None:
    record = read_checkpoint_record(run_dir)
    if record is not None:
        atomic_write_text(
            Path(run_dir) / CHECKPOINT_NAME,
            json.dumps(checkpoint_payload(record[0], status="completed"), indent=2, sort_keys=True) + "\n",
        )


PHASE_STATUS = {phase: "planning" for phase in ResumePhase}
PHASE_STATUS.update({
    ResumePhase.WORKTREE_SETUP: "preparing", ResumePhase.IMPLEMENT_STEP: "implementing",
    ResumePhase.DETERMINISTIC_GATE: "validating", ResumePhase.CHECK_REPAIR: "revising",
    ResumePhase.SEMANTIC_REVISION: "revising", ResumePhase.CANDIDATE_READY: "approved",
    ResumePhase.CANDIDATE_PUSH: "approved", ResumePhase.FINAL_REVIEW: "reviewing",
    ResumePhase.REVIEW_IMPLEMENTATION: "implementing", ResumePhase.REVIEW_REPLAN: "planning",
    ResumePhase.PUBLISH: "publishing",
})
_RESUMABLE_STATUSES = frozenset({
    "failed", "interrupted", "waiting_scope_approval", "waiting_check_infrastructure",
    "waiting_remote", "waiting_external", "waiting_contract_repair",
})
# An operator retry of the StepContractRepairPlanner of one pending slot.
CONTRACT_REPAIR_OPERATION = "contract_repair"
# A stranded contract repair whose evidence no longer proves its identity.
CONTRACT_REPAIR_INTEGRITY_OPERATION = "contract_repair_integrity"
_STRANDED_DETAIL = re.compile(r"step=(S[0-9]{2}) contract repair failed: ")
# Failures that a checkpoint can never repair: the run needs an operator.
_TERMINAL_FAILURES = frozenset({
    "RESUME_INTEGRITY_FAILURE", "RESUME_REQUIRES_OPERATOR",
    "AGENT_SCOPE_VIOLATION", "AGENT_GIT_VIOLATION",
    "SECRET_IN_DIFF", "SECRET_IN_STAGED_BLOB", "UNSCANNABLE_STAGED_BLOB",
    "STAGED_BLOB_SCAN_FAILED", "UNREVIEWABLE_TEXT_DIFF",
    "HEAD_MISMATCH", "TREE_MISMATCH", "UNEXPECTED_HEAD", "UNEXPECTED_TREE",
    "COMMIT_TREE_MISMATCH", "INTEGRITY_MISMATCH",
    "DURABLE_ARTIFACT_CORRUPTED", "CORRUPTED_DURABLE_ARTIFACT",
    "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH",
    "CHECK_MUTATED_FORBIDDEN_FILES",
})


def resume_label(checkpoint: ResumeCheckpoint) -> str:
    cycle = f" (cycle {checkpoint.review_cycle:03d})" if checkpoint.review_cycle > 1 else ""
    labels = {
        ResumePhase.CONTEXT: "Retry context", ResumePhase.PLANNER: "Retry planner",
        ResumePhase.PLAN_APPROVAL: "Resume plan approval", ResumePhase.WORKTREE_SETUP: "Retry workspace setup",
        ResumePhase.IMPLEMENT_STEP: f"Retry {checkpoint.step_id}",
        ResumePhase.DETERMINISTIC_GATE: f"Retry deterministic gate ({checkpoint.stage})",
        ResumePhase.CHECK_REPAIR: f"Retry check-repair attempt {checkpoint.check_repair_attempt}",
        ResumePhase.SEMANTIC_REVISION: "Retry semantic revision", ResumePhase.CANDIDATE_READY: "Prepare candidate",
        ResumePhase.CANDIDATE_PUSH: "Push candidate", ResumePhase.FINAL_REVIEW: "Retry final review",
        ResumePhase.REVIEW_IMPLEMENTATION: f"Retry correction {checkpoint.step_id}",
        ResumePhase.REVIEW_REPLAN: "Retry correction planner", ResumePhase.PUBLISH: "Retry publish",
    }
    return labels[checkpoint.phase] + cycle


@dataclass(frozen=True)
class ResumeInfo:
    resumable: bool
    phase: str | None = None
    label: str | None = None
    reason: str | None = None
    expected_tree: str | None = None
    review_cycle: int | None = None
    step_id: str | None = None
    operation: str | None = None


def _max_read_paths(run_dir: Path) -> int | None:
    try:
        path = run_dir / "run_options.json"
        if path.stat().st_size > 256 * 1024:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))["planning"]["max_read_paths_per_step"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def stranded_contract_repair(run_dir: str | Path, state: Mapping[str, Any]) -> str:
    """``proven``, ``corrupt`` or ``absent`` for a stranded contract repair.

    Only the legacy projection of an invalid StepContractRepairPlanner answer
    onto ``AGENT_CONTRACT_MISMATCH`` in ``waiting_human`` qualifies, and it is
    ``proven`` only when the checkpoint, repair slot, durable request, paid
    answer and recorded parse error all belong together.  Every other
    ``waiting_human`` state is a genuine operator decision (``absent``) and
    stays non-resumable.
    """

    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    detail = failure.get("detail")
    if (
        state.get("status") != "waiting_human"
        or state.get("planning_protocol") != "v2"
        or failure.get("reason") != "AGENT_CONTRACT_MISMATCH"
        or not isinstance(detail, str)
    ):
        return "absent"
    match = _STRANDED_DETAIL.match(detail)
    directory = Path(run_dir)
    max_read_paths = _max_read_paths(directory)
    if match is None:
        return "absent"
    try:
        checkpoint = read_checkpoint(directory)
    except ResumeCheckpointError:
        return "corrupt"
    if (
        max_read_paths is None
        or checkpoint is None
        or checkpoint.phase is not ResumePhase.IMPLEMENT_STEP
        or checkpoint.step_id != match.group(1)
        or checkpoint.expected_tree_sha is None
    ):
        return "absent"
    from .orchestration.contract_repair import stranded_output_failure

    step_dir = (
        directory / "cycles" / f"{checkpoint.review_cycle:03d}" / "implementation"
        / "steps" / checkpoint.step_id
    )
    return stranded_output_failure(
        step_dir, step_id=checkpoint.step_id, tree_sha=checkpoint.expected_tree_sha,
        failure_detail=detail, max_read_paths_per_step=max_read_paths,
    )


def resume_info(run_dir: str | Path, state: Mapping[str, Any]) -> ResumeInfo:
    stranded_shape = stranded_contract_repair(run_dir, state)
    stranded = stranded_shape == "proven"
    if stranded_shape == "corrupt":
        return ResumeInfo(
            False, reason="pending contract repair evidence is inconsistent",
            operation=CONTRACT_REPAIR_INTEGRITY_OPERATION,
        )
    if state.get("status") not in _RESUMABLE_STATUSES and not stranded:
        return ResumeInfo(False, reason="run has no resumable waiting state")
    if state.get("planning_protocol") != "v2":
        return ResumeInfo(False, reason="only pipeline v2 runs can be resumed")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    if failure.get("reason") in _TERMINAL_FAILURES:
        return ResumeInfo(False, reason="the run requires an operator")
    if state.get("recovery_resumable") is False and not stranded:
        return ResumeInfo(False, reason="the run stopped at a non-resumable failure")
    try:
        checkpoint = read_checkpoint(run_dir)
    except ResumeCheckpointError:
        return ResumeInfo(False, reason="resume checkpoint is invalid")
    if checkpoint is None:
        return ResumeInfo(False, reason="no resume checkpoint")
    label = resume_label(checkpoint)
    operation = None
    if stranded or (
        state.get("status") == "waiting_contract_repair"
        and checkpoint.phase is ResumePhase.IMPLEMENT_STEP
    ):
        operation = CONTRACT_REPAIR_OPERATION
        label = f"Retry contract repair planner ({checkpoint.step_id})"
    return ResumeInfo(
        True, checkpoint.phase.value, label,
        expected_tree=checkpoint.expected_tree_sha,
        review_cycle=checkpoint.review_cycle,
        step_id=checkpoint.step_id,
        operation=operation,
    )


class ResumeError(RuntimeError):
    pass


class ResumeNotAllowedError(ResumeError):
    pass


class ResumeIntegrityError(ResumeError):
    code = "RESUME_INTEGRITY_FAILURE"


class ResumeRequiresOperatorError(ResumeError):
    code = "RESUME_REQUIRES_OPERATOR"


__all__ = [
    "CHECKPOINT_NAME", "CONTRACT_REPAIR_INTEGRITY_OPERATION", "CONTRACT_REPAIR_OPERATION",
    "PHASE_STATUS", "ResumeCheckpoint",
    "ResumeCheckpointError", "ResumeError", "ResumeInfo", "ResumeIntegrityError",
    "ResumeNotAllowedError", "ResumePhase", "ResumeRequiresOperatorError",
    "checkpoint_payload", "mark_checkpoint_completed", "pipeline_version_from_state",
    "plan_identity_from_mapping", "read_checkpoint", "read_checkpoint_record",
    "resume_info", "resume_label", "stranded_contract_repair", "write_checkpoint",
]
