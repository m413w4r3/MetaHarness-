"""Git-first resume checkpoint format and eligibility projection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_identity import CHECKPOINT_SCHEMA_VERSION, CheckpointFormatError, checkpoint_sha256, read_checkpoint_file, stamp_from_payload
from .models import RunDisposition, RunIdentity, RunMachineState, RunPhase, RUN_CHECKPOINT_NAME, assemble_run_state, project_run_outcome
from .recovery_policy import FailureClass, classify_failure
from .result import atomic_write_text
from .run_options import RUN_SCHEMA_UNSUPPORTED

ResumePhase = RunPhase
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EARLY = frozenset({RunPhase.CONTEXT, RunPhase.PLANNER})
_FIELDS = frozenset({"schema_version", "iteration", "phase", "step_index", "last_green_commit", "plan_sha256"})
CHECKPOINT_INTEGRITY_OPERATION = "checkpoint_integrity"
STEP_ACCEPTANCE_OPERATION = "step_acceptance"


class ResumeCheckpointError(ValueError):
    """A checkpoint is malformed or cannot name a resume boundary."""


class ResumeSchemaUnsupportedError(ResumeCheckpointError):
    """The run was written by a checkpoint schema this runtime cannot read."""


@dataclass(frozen=True)
class ResumeCheckpoint:
    phase: ResumePhase
    iteration: int = 1
    step_index: int | None = None
    last_green_commit: str | None = None
    plan_sha256: str | None = None

    def __post_init__(self) -> None:
        try: phase = ResumePhase(self.phase)
        except (TypeError, ValueError) as exc: raise ResumeCheckpointError("checkpoint phase is unknown") from exc
        object.__setattr__(self, "phase", phase)
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration < 1:
            raise ResumeCheckpointError("checkpoint iteration must be a positive integer")
        if phase is RunPhase.IMPLEMENT_STEP and (isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0):
            raise ResumeCheckpointError("implementation checkpoint needs a 0-based step_index")
        if phase is not RunPhase.IMPLEMENT_STEP and self.step_index is not None:
            raise ResumeCheckpointError("step_index is only valid during implementation")
        if self.last_green_commit is not None and not _OBJECT_ID.fullmatch(self.last_green_commit):
            raise ResumeCheckpointError("last_green_commit is invalid")
        if self.plan_sha256 is not None and not _SHA256.fullmatch(self.plan_sha256):
            raise ResumeCheckpointError("plan_sha256 is invalid")
        if phase not in _EARLY and self.plan_sha256 is None:
            raise ResumeCheckpointError("this phase requires the durable effective plan hash")
        if phase not in {*_EARLY, RunPhase.PLAN_APPROVAL, RunPhase.WORKTREE_SETUP} and self.last_green_commit is None:
            raise ResumeCheckpointError("this phase requires last_green_commit")


def pipeline_version_from_state(state: Mapping[str, Any]) -> int:
    if not isinstance(state, Mapping) or state.get("pipeline_version") != 2:
        raise ResumeCheckpointError("pipeline_version must be 2")
    return 2


def checkpoint_payload(checkpoint: ResumeCheckpoint) -> dict[str, Any]:
    return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "iteration": checkpoint.iteration,
            "phase": checkpoint.phase.value, "step_index": checkpoint.step_index,
            "last_green_commit": checkpoint.last_green_commit, "plan_sha256": checkpoint.plan_sha256}


def write_checkpoint(run_dir: str | Path, checkpoint: ResumeCheckpoint) -> None:
    if not isinstance(checkpoint, ResumeCheckpoint): raise TypeError("checkpoint must be a ResumeCheckpoint")
    atomic_write_text(Path(run_dir) / RUN_CHECKPOINT_NAME, json.dumps(checkpoint_payload(checkpoint), indent=2, sort_keys=True) + "\n")


def _parse(payload: Any, *, sha256: str) -> ResumeCheckpoint:
    if not isinstance(payload, Mapping): raise ResumeCheckpointError("checkpoint is not an object")
    version = payload.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeSchemaUnsupportedError(f"{RUN_SCHEMA_UNSUPPORTED}: checkpoint schema_version {version} is not {CHECKPOINT_SCHEMA_VERSION}")
    try: stamp = stamp_from_payload(payload, sha256=sha256)
    except CheckpointFormatError as exc: raise ResumeCheckpointError(str(exc)) from exc
    if stamp.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeSchemaUnsupportedError(f"{RUN_SCHEMA_UNSUPPORTED}: checkpoint schema is unsupported")
    if set(payload) != _FIELDS: raise ResumeCheckpointError("checkpoint fields do not match the current schema")
    try:
        return ResumeCheckpoint(
            phase=stamp.phase, iteration=payload["iteration"], step_index=payload["step_index"],
            last_green_commit=payload["last_green_commit"], plan_sha256=payload["plan_sha256"],
        )
    except (KeyError, TypeError, ResumeCheckpointError) as exc:
        raise ResumeCheckpointError(str(exc)) from exc


def read_checkpoint_record(run_dir: str | Path) -> ResumeCheckpoint | None:
    try: record = read_checkpoint_file(run_dir)
    except CheckpointFormatError as exc: raise ResumeCheckpointError(str(exc)) from exc
    return _parse(record.payload, sha256=record.sha256) if record is not None else None


read_checkpoint = read_checkpoint_record
PHASE_STATUS = {phase: project_run_outcome(RunMachineState(phase, RunDisposition.RUNNING)).status.value for phase in ResumePhase}


def resume_label(checkpoint: ResumeCheckpoint) -> str:
    return {
        RunPhase.CONTEXT: "Retry context", RunPhase.PLANNER: "Retry planner",
        RunPhase.PLAN_APPROVAL: "Resume plan approval", RunPhase.WORKTREE_SETUP: "Retry workspace setup",
        RunPhase.IMPLEMENT_STEP: f"Retry step {checkpoint.step_index}", RunPhase.STEP_ACCEPTANCE: "Retry step acceptance",
        RunPhase.DETERMINISTIC_GATE: "Retry deterministic gate", RunPhase.AUDIT: "Retry audit",
        RunPhase.CANDIDATE_READY: "Prepare candidate", RunPhase.CANDIDATE_PUSH: "Push candidate", RunPhase.PUBLISH: "Retry publish",
    }[checkpoint.phase]


@dataclass(frozen=True)
class ResumeInfo:
    resumable: bool
    phase: str | None = None
    label: str | None = None
    reason: str | None = None
    iteration: int | None = None
    step_index: int | None = None
    last_green_commit: str | None = None
    operation: str | None = None
    disposition: str | None = None


def machine_state_for_run(state: Mapping[str, Any], checkpoint: ResumeCheckpoint | None = None) -> RunMachineState:
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    phase = checkpoint.phase if checkpoint is not None else None
    if phase is None and state.get("phase") is not None:
        try: phase = RunPhase(state["phase"])
        except (TypeError, ValueError) as exc: raise ResumeCheckpointError("the recorded run phase is unknown") from exc
    try:
        return assemble_run_state(
            phase, disposition=state.get("disposition"), status=state.get("status"),
            reason=state.get("reason"), failure_reason=failure.get("reason"),
        )
    except ValueError as exc: raise ResumeCheckpointError("the durable run state is unreadable") from exc


def run_identity(state: Mapping[str, Any], run_dir: str | Path, checkpoint: ResumeCheckpoint | None = None) -> RunIdentity:
    updated_at = state.get("updated_at")
    return RunIdentity.of(
        machine_state_for_run(state, checkpoint),
        updated_at=updated_at if isinstance(updated_at, str) else None,
        checkpoint_sha256=checkpoint_sha256(run_dir),
    )


def resume_info(run_dir: str | Path, state: Mapping[str, Any]) -> ResumeInfo:
    try:
        checkpoint = read_checkpoint(run_dir)
    except ResumeSchemaUnsupportedError as exc: return ResumeInfo(False, reason=str(exc), operation=RUN_SCHEMA_UNSUPPORTED)
    except ResumeCheckpointError as exc: return ResumeInfo(False, reason=f"the current resume checkpoint is inconsistent: {exc}", operation=CHECKPOINT_INTEGRITY_OPERATION)
    try:
        outcome = project_run_outcome(machine_state_for_run(state, checkpoint))
    except (ValueError, TypeError) as exc: return ResumeInfo(False, reason=f"the durable run state is unreadable: {exc}", operation=CHECKPOINT_INTEGRITY_OPERATION)
    if not outcome.resume_eligible:
        return ResumeInfo(False, reason="run has no resumable waiting state")
    if state.get("planning_protocol") != "v2":
        return ResumeInfo(False, reason="only pipeline v2 runs can be resumed")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = failure.get("reason")
    if isinstance(reason, str) and reason.strip() and classify_failure(reason).failure_class is FailureClass.FATAL:
        return ResumeInfo(False, reason="the run stopped at a fatal boundary")
    if state.get("recovery_resumable") is False:
        return ResumeInfo(False, reason="the run stopped at a non-resumable failure")
    if checkpoint is None:
        return ResumeInfo(False, reason="no resume checkpoint")
    operation = STEP_ACCEPTANCE_OPERATION if checkpoint.phase is RunPhase.STEP_ACCEPTANCE else None
    return ResumeInfo(
        True, checkpoint.phase.value, resume_label(checkpoint), iteration=checkpoint.iteration,
        step_index=checkpoint.step_index, last_green_commit=checkpoint.last_green_commit,
        operation=operation, disposition=outcome.disposition.value,
    )


class ResumeError(RuntimeError): pass
class ResumeNotAllowedError(ResumeError): pass
class ResumeIntegrityError(ResumeError):
    code = "RESUME_INTEGRITY_FAILURE"
class ResumeRequiresOperatorError(ResumeError):
    code = "RESUME_REQUIRES_OPERATOR"


__all__ = [
    "CHECKPOINT_INTEGRITY_OPERATION", "PHASE_STATUS", "RUN_SCHEMA_UNSUPPORTED",
    "STEP_ACCEPTANCE_OPERATION", "ResumeCheckpoint", "ResumeCheckpointError",
    "ResumeError", "ResumeInfo", "ResumeIntegrityError", "ResumeNotAllowedError",
    "ResumePhase", "ResumeRequiresOperatorError", "ResumeSchemaUnsupportedError",
    "checkpoint_payload", "pipeline_version_from_state", "read_checkpoint",
    "read_checkpoint_record", "resume_info", "resume_label", "run_identity", "write_checkpoint",
]
