"""Durable, provider-neutral checkpoints for the pipeline v2 state machine.

Le checkpoint porte la phase qui fait autorité : l'opération courante ou
prochaine.  ``state.json`` n'en stocke qu'une projection dérivée (la posture et
son statut historique), calculée par
:func:`~metaharness.models.project_run_outcome`.  Aucune matrice implicite
status/phase ne subsiste : :func:`machine_state_for_run` assemble l'état
durable, :func:`resume_info` le projette.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .approval import ApprovalError, PlanIdentity
from .checkpoint_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointFormatError,
    checkpoint_sha256,
    read_checkpoint_file,
    stamp_from_payload,
)
from .models import (
    CONTRACT_REPAIR_WAIT_REASON,
    GateStage,
    RunDisposition,
    RunIdentity,
    RunMachineState,
    RunPhase,
    RUN_CHECKPOINT_NAME,
    assemble_run_state,
    project_run_outcome,
)
from .recovery_policy import FailureClass, classify_failure
from .result import atomic_write_text
from .run_options import RUN_SCHEMA_UNSUPPORTED
from .step_ids import STEP_ID_RE

CHECKPOINT_NAME = RUN_CHECKPOINT_NAME
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def pipeline_version_from_state(state: Mapping[str, Any]) -> int:
    if not isinstance(state, Mapping) or state.get("pipeline_version") != 2:
        raise ResumeCheckpointError("pipeline_version must be 2")
    return 2


# The canonical phase vocabulary lives in the model (``RunPhase``); the
# checkpoint exposes the same vocabulary as ``ResumePhase``.
ResumePhase = RunPhase


_GATE_STAGES = frozenset(stage.value for stage in GateStage)
_STAGED_PHASES = frozenset({ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR})
_PRE_PLAN = frozenset({ResumePhase.CONTEXT, ResumePhase.PLANNER})
_PRE_APPROVAL = _PRE_PLAN | frozenset({ResumePhase.PLAN_APPROVAL})
_NO_WORKTREE = _PRE_APPROVAL | frozenset({ResumePhase.WORKTREE_SETUP})


class ResumeCheckpointError(ValueError):
    """A checkpoint is malformed or does not bind the next operation."""


class ResumeSchemaUnsupportedError(ResumeCheckpointError):
    """The checkpoint was written by an incompatible runtime.

    It is never an integrity incident: the run is simply not resumable by
    this runtime and every artifact stays readable for an operator.
    """


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
        step_phases = {
            ResumePhase.IMPLEMENT_STEP, ResumePhase.STEP_ACCEPTANCE,
            ResumePhase.REVIEW_IMPLEMENTATION,
        }
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
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
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


def _parse(payload: Any, *, sha256: str) -> tuple[ResumeCheckpoint, str]:
    """The business identity of one decoded checkpoint.

    The structural frame (schema, status, phase) and the digest come from the
    shared reader this module and the state store both use; only the business
    validation below is owned here.
    """

    try:
        stamp = stamp_from_payload(payload, sha256=sha256)
    except CheckpointFormatError as exc:
        raise ResumeCheckpointError(str(exc)) from exc
    if stamp.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ResumeSchemaUnsupportedError(
            f"{RUN_SCHEMA_UNSUPPORTED}: checkpoint schema_version "
            f"{stamp.schema_version} is not {CHECKPOINT_SCHEMA_VERSION}"
        )
    checkpoint = ResumeCheckpoint(
        phase=stamp.phase, review_cycle=payload.get("review_cycle", 1),
        stage=payload.get("stage"), step_id=payload.get("step_id"),
        next_step_id=payload.get("next_step_id"), check_repair_attempt=payload.get("check_repair_attempt"),
        expected_head_sha=payload.get("expected_head_sha"), expected_parent_sha=payload.get("expected_parent_sha"),
        expected_tree_sha=payload.get("expected_tree_sha"), execution_selection_sha256=payload.get("execution_selection_sha256"),
        plan_identity=(plan_identity_from_mapping(payload["plan_identity"]) if payload.get("plan_identity") is not None else None),
        correction_bundle_sha256=payload.get("correction_bundle_sha256"),
    )
    return checkpoint, stamp.status


def read_checkpoint_record(run_dir: str | Path) -> tuple[ResumeCheckpoint, str] | None:
    """The full record of the run's checkpoint, or ``None`` when it has none.

    A checkpoint that exists but cannot be read is refused here, never treated
    as absent: the resume gate maps that refusal to ``RESUME_INTEGRITY_FAILURE``.
    """

    try:
        checkpoint = read_checkpoint_file(run_dir)
    except CheckpointFormatError as exc:
        raise ResumeCheckpointError(str(exc)) from exc
    if checkpoint is None:
        return None
    return _parse(checkpoint.payload, sha256=checkpoint.sha256)


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


# Derived, never authored: the status of a phase that is still running.
PHASE_STATUS = {
    phase: project_run_outcome(
        RunMachineState(phase, RunDisposition.RUNNING),
    ).status.value
    for phase in ResumePhase
}
# An operator retry of the StepContractRepairPlanner of one pending slot.
CONTRACT_REPAIR_OPERATION = "contract_repair"
# The deterministic acceptance of a durable, successful worker candidate.
STEP_ACCEPTANCE_OPERATION = "step_acceptance"
# A current checkpoint whose durable identity no longer holds.
CHECKPOINT_INTEGRITY_OPERATION = "checkpoint_integrity"
# Exact exhausted check-repair state.


def resume_label(checkpoint: ResumeCheckpoint) -> str:
    cycle = f" (cycle {checkpoint.review_cycle:03d})" if checkpoint.review_cycle > 1 else ""
    labels = {
        ResumePhase.CONTEXT: "Retry context", ResumePhase.PLANNER: "Retry planner",
        ResumePhase.PLAN_APPROVAL: "Resume plan approval", ResumePhase.WORKTREE_SETUP: "Retry workspace setup",
        ResumePhase.IMPLEMENT_STEP: f"Retry {checkpoint.step_id}",
        ResumePhase.STEP_ACCEPTANCE: f"Retry step acceptance ({checkpoint.step_id})",
        ResumePhase.DETERMINISTIC_GATE: f"Retry deterministic gate ({checkpoint.stage})",
        ResumePhase.AUDIT: "Retry audit",
        ResumePhase.CHECK_REPLAN: "Retry re-decomposition planner",
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
    disposition: str | None = None


def machine_state_for_run(
    state: Mapping[str, Any], checkpoint: ResumeCheckpoint | None = None,
) -> RunMachineState:
    """Assemble the durable run state: one phase, one disposition.

    The checkpoint owns the phase; the run state owns the posture.  A stored
    status is read through the single bridge
    :func:`~metaharness.models.assemble_run_state`.
    """

    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    phase = checkpoint.phase if checkpoint is not None else None
    if phase is None:
        recorded = state.get("phase")
        if recorded is not None:
            try:
                phase = RunPhase(recorded)
            except (TypeError, ValueError) as exc:
                raise ResumeCheckpointError("the recorded run phase is unknown") from exc
    try:
        return assemble_run_state(
            phase,
            disposition=state.get("disposition"),
            status=state.get("status"),
            reason=state.get("reason"),
            failure_reason=failure.get("reason"),
        )
    except ValueError as exc:
        raise ResumeCheckpointError("the durable run state is unreadable") from exc


def run_identity(
    state: Mapping[str, Any], run_dir: str | Path,
    checkpoint: ResumeCheckpoint | None = None,
) -> RunIdentity:
    """The canonical identity of one observed run.

    A claim compares the machine state, its generation (``updated_at``) and the
    exact checkpoint bytes it was validated against; a status spelling is never
    part of it.
    """

    machine = machine_state_for_run(state, checkpoint)
    updated_at = state.get("updated_at")
    return RunIdentity.of(
        machine,
        updated_at=updated_at if isinstance(updated_at, str) else None,
        checkpoint_sha256=checkpoint_sha256(run_dir),
    )


def resume_info(run_dir: str | Path, state: Mapping[str, Any]) -> ResumeInfo:
    """Whether this runtime may resume the run, and at which boundary.

    The current checkpoint schema plus its durable identities decide: a
    checkpoint written by an incompatible runtime is refused as
    ``RUN_SCHEMA_UNSUPPORTED`` with every artifact left readable, while an
    incoherent current checkpoint is an integrity incident.
    """

    try:
        checkpoint = read_checkpoint(run_dir)
    except ResumeSchemaUnsupportedError as exc:
        return ResumeInfo(False, reason=str(exc), operation=RUN_SCHEMA_UNSUPPORTED)
    except ResumeCheckpointError as exc:
        return ResumeInfo(
            False, reason=f"the current resume checkpoint is inconsistent: {exc}",
            operation=CHECKPOINT_INTEGRITY_OPERATION,
        )
    try:
        machine = machine_state_for_run(state, checkpoint)
        outcome = project_run_outcome(machine)
    except (ValueError, TypeError) as exc:
        return ResumeInfo(
            False, reason=f"the durable run state is unreadable: {exc}",
            operation=CHECKPOINT_INTEGRITY_OPERATION,
        )
    if not outcome.resume_eligible:
        return ResumeInfo(False, reason="run has no resumable waiting state")
    if state.get("planning_protocol") != "v2":
        return ResumeInfo(False, reason="only pipeline v2 runs can be resumed")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = failure.get("reason")
    if isinstance(reason, str) and reason.strip() and (
        classify_failure(reason).failure_class is FailureClass.FATAL
    ):
        return ResumeInfo(False, reason="the run stopped at a fatal boundary")
    if state.get("recovery_resumable") is False:
        return ResumeInfo(False, reason="the run stopped at a non-resumable failure")
    if checkpoint is None:
        return ResumeInfo(False, reason="no resume checkpoint")
    label = resume_label(checkpoint)
    operation = None
    # The projected outcome names the durable boundary; no caller compares the
    # stored status to the checkpoint phase any more.
    if (
        machine.reason == CONTRACT_REPAIR_WAIT_REASON
        and checkpoint.phase is RunPhase.IMPLEMENT_STEP
    ):
        operation = CONTRACT_REPAIR_OPERATION
        label = f"Retry contract repair planner ({checkpoint.step_id})"
    elif checkpoint.phase is RunPhase.STEP_ACCEPTANCE:
        operation = STEP_ACCEPTANCE_OPERATION
    return ResumeInfo(
        True, checkpoint.phase.value, label,
        expected_tree=checkpoint.expected_tree_sha,
        review_cycle=checkpoint.review_cycle,
        step_id=checkpoint.step_id,
        operation=operation,
        disposition=outcome.disposition.value,
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
    "CHECKPOINT_NAME",
    "CHECKPOINT_INTEGRITY_OPERATION", "CONTRACT_REPAIR_OPERATION",
    "PHASE_STATUS", "RUN_SCHEMA_UNSUPPORTED", "checkpoint_sha256", "machine_state_for_run",
    "run_identity", "STEP_ACCEPTANCE_OPERATION", "ResumeCheckpoint",
    "ResumeCheckpointError", "ResumeError", "ResumeInfo", "ResumeIntegrityError",
    "ResumeNotAllowedError", "ResumePhase", "ResumeRequiresOperatorError",
    "ResumeSchemaUnsupportedError",
    "checkpoint_payload", "mark_checkpoint_completed", "pipeline_version_from_state",
    "plan_identity_from_mapping", "read_checkpoint", "read_checkpoint_record",
    "resume_info", "resume_label", "write_checkpoint",
]
