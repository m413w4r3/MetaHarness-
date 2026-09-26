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
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .approval import ApprovalError, PlanIdentity
from .models import (
    GateStage,
    RunDisposition,
    RunMachineState,
    RunPhase,
    RunStatus,
    disposition_for_status,
    project_run_outcome,
    wait_reason_for_status,
)
from .result import atomic_write_text
from .run_options import RUN_SCHEMA_UNSUPPORTED
from .step_ids import STEP_ID_RE

CHECKPOINT_NAME = "resume_checkpoint.json"
_SCHEMA_VERSION = 4
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
    if not isinstance(payload, dict):
        raise ResumeCheckpointError("checkpoint is not an object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ResumeCheckpointError("checkpoint schema_version is not an integer")
    if schema_version != _SCHEMA_VERSION:
        raise ResumeSchemaUnsupportedError(
            f"{RUN_SCHEMA_UNSUPPORTED}: checkpoint schema_version "
            f"{schema_version} is not {_SCHEMA_VERSION}"
        )
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
CHECK_REPAIR_RETRY_OPERATION = "check_repair_retry"
CHECK_REPAIR_INTEGRITY_OPERATION = "check_repair_integrity"
# Failures that a checkpoint can never repair: the run needs an operator.
_TERMINAL_FAILURES = frozenset({
    "RESUME_INTEGRITY_FAILURE", "RESUME_REQUIRES_OPERATOR",
    "AGENT_SCOPE_VIOLATION", "AGENT_GIT_VIOLATION",
    "SECRET_IN_DIFF", "SECRET_IN_STAGED_BLOB", "UNSCANNABLE_STAGED_BLOB",
    "STAGED_BLOB_SCAN_FAILED", "UNREVIEWABLE_TEXT_DIFF",
    "HEAD_MISMATCH", "TREE_MISMATCH", "UNEXPECTED_HEAD", "UNEXPECTED_TREE",
    "COMMIT_TREE_MISMATCH", "INTEGRITY_MISMATCH",
    "COMMIT_GATE_FAILED",
    "DURABLE_ARTIFACT_CORRUPTED", "CORRUPTED_DURABLE_ARTIFACT",
    "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH",
    "CHECK_MUTATED_FORBIDDEN_FILES",
    "CHECK_REPAIR_FIXED_POINT",
})


def resume_label(checkpoint: ResumeCheckpoint) -> str:
    cycle = f" (cycle {checkpoint.review_cycle:03d})" if checkpoint.review_cycle > 1 else ""
    labels = {
        ResumePhase.CONTEXT: "Retry context", ResumePhase.PLANNER: "Retry planner",
        ResumePhase.PLAN_APPROVAL: "Resume plan approval", ResumePhase.WORKTREE_SETUP: "Retry workspace setup",
        ResumePhase.IMPLEMENT_STEP: f"Retry {checkpoint.step_id}",
        ResumePhase.STEP_ACCEPTANCE: f"Retry step acceptance ({checkpoint.step_id})",
        ResumePhase.DETERMINISTIC_GATE: f"Retry deterministic gate ({checkpoint.stage})",
        ResumePhase.CHECK_REPAIR: f"Retry check-repair attempt {checkpoint.check_repair_attempt}",
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
    :func:`~metaharness.models.disposition_for_status`.
    """

    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = failure.get("reason")
    stored = state.get("disposition")
    if isinstance(stored, str):
        try:
            disposition = RunDisposition(stored)
        except ValueError as exc:
            raise ResumeCheckpointError("run disposition is unknown") from exc
    else:
        disposition = disposition_for_status(state.get("status"))
    if not isinstance(reason, str) or not reason:
        reason = wait_reason_for_status(state.get("status")) if disposition.waiting else None
    return RunMachineState(
        checkpoint.phase if checkpoint is not None else None,
        disposition,
        reason,
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
    check_repair = _check_repair_exhaustion_info(run_dir, state, checkpoint)
    if check_repair is not None:
        return check_repair
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
    if failure.get("reason") in _TERMINAL_FAILURES:
        return ResumeInfo(False, reason="the run requires an operator")
    if state.get("recovery_resumable") is False:
        return ResumeInfo(False, reason="the run stopped at a non-resumable failure")
    if checkpoint is None:
        return ResumeInfo(False, reason="no resume checkpoint")
    label = resume_label(checkpoint)
    operation = None
    # The projected outcome names the durable boundary; no caller compares the
    # stored status to the checkpoint phase any more.
    if (
        outcome.status is RunStatus.WAITING_CONTRACT_REPAIR
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


def _check_repair_exhaustion_info(
    run_dir: str | Path, state: Mapping[str, Any],
    checkpoint: ResumeCheckpoint | None,
) -> ResumeInfo | None:
    """Recognize only the durable, exhausted deterministic gate."""

    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    detail = failure.get("detail")
    if not (
        state.get("planning_protocol") == "v2"
        and failure.get("reason") == "CHECK_REPAIR_EXHAUSTED"
        and checkpoint is not None
        and project_run_outcome(machine_state_for_run(state, checkpoint)).status
        is RunStatus.WAITING_CHECK_REPAIR
    ):
        return None

    def invalid(reason: str) -> ResumeInfo:
        return ResumeInfo(
            False, reason=reason, operation=CHECK_REPAIR_INTEGRITY_OPERATION,
        )

    directory = Path(run_dir)
    try:
        from .run_options import read_run_options_for_state
        options, _ = read_run_options_for_state(directory, state)
    except (OSError, ValueError):
        return invalid("exhausted check-repair authority is unreadable")
    budget = options.max_check_repair_attempts
    if (
        checkpoint is None
        or checkpoint.phase is not ResumePhase.DETERMINISTIC_GATE
        or checkpoint.stage is None
        or checkpoint.stage.value != "POST_IMPLEMENTATION"
        or budget < 1
        or checkpoint.check_repair_attempt != budget
        or checkpoint.expected_head_sha is None
        or checkpoint.expected_tree_sha is None
    ):
        return invalid("exhausted check-repair checkpoint does not match its budget")

    cycle = checkpoint.review_cycle
    stage_dir = checkpoint.stage.value.casefold().replace("_", "-")
    attempt_path = (
        directory / "cycles" / f"{cycle:03d}" / "check-repair" / stage_dir
        / "attempts" / f"{budget:03d}" / "attempt.json"
    )
    evidence_path = (
        directory / "cycles" / f"{cycle:03d}" / "checks" / stage_dir / "evidence.json"
    )
    try:
        if attempt_path.stat().st_size > 128 * 1024 or evidence_path.stat().st_size > 4 * 1024 * 1024:
            return invalid("latest check-repair evidence exceeds its size limit")
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
    except (OSError, UnicodeError, ValueError):
        return invalid("latest check-repair attempt or gate evidence is missing")
    if (
        not isinstance(attempt, dict)
        or attempt.get("number") != budget
        or attempt.get("tree_after") != checkpoint.expected_tree_sha
        or not isinstance(attempt.get("failed_check_ids_before"), list)
        or any(not isinstance(item, str) for item in attempt["failed_check_ids_before"])
        or not isinstance(evidence, dict)
        or evidence.get("staged_tree_sha") != checkpoint.expected_tree_sha
        or evidence.get("deterministic_passed") is not False
        or not isinstance(evidence.get("failures"), list)
        or any(
            not isinstance(item, str) or not item.startswith("CHECK_FAILED:")
            for item in evidence["failures"]
        )
        or not evidence["failures"]
    ):
        return invalid("latest gate evidence is not a red product-check result")

    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    failed_ids = [item.split(":", 1)[1] for item in evidence["failures"]]
    check_repair_state = state.get("check_repair")
    stored_sha = None
    if not isinstance(check_repair_state, Mapping):
        return invalid("check-repair attempt summary is missing")
    else:
        stored_sha = check_repair_state.get("latest_evidence_sha256")
        if stored_sha is not None and stored_sha != evidence_sha256:
            return invalid("latest deterministic evidence hash changed")
        stored_attempts = check_repair_state.get("attempt_count")
        attempt_summaries = check_repair_state.get("attempts")
        if (
            stored_attempts != budget
            or not isinstance(attempt_summaries, list)
            or len(attempt_summaries) != budget
            or any(
                not isinstance(item, Mapping) or item.get("number") != index
                for index, item in enumerate(attempt_summaries, start=1)
            )
        ):
            return invalid("check-repair attempt count does not match its budget")
        reports = check_repair_state.get("repair_reports")
        if reports is not None:
            if not isinstance(reports, list):
                return invalid("check-repair report references are malformed")
            for report in reports:
                if not isinstance(report, Mapping):
                    return invalid("check-repair report reference is malformed")
                number = report.get("attempt")
                relative = report.get("artifact")
                digest = report.get("sha256")
                expected_relative = (
                    f"cycles/{cycle:03d}/check-repair/{stage_dir}/attempts/{number:03d}/report.json"
                    if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= budget
                    else None
                )
                if (
                    relative != expected_relative
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    return invalid("check-repair report reference identity is invalid")
                try:
                    if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
                        return invalid("check-repair report hash changed")
                except OSError:
                    return invalid("check-repair report artifact is unreadable")
    detail_sha = detail.get("latest_evidence_sha256") if isinstance(detail, Mapping) else None
    if detail_sha is not None and detail_sha != evidence_sha256:
        return invalid("latest failure evidence hash changed")
    if stored_sha != evidence_sha256 or detail_sha != evidence_sha256:
        return invalid("exhausted check-repair state has no evidence hash authority")
    if (
        not isinstance(detail, Mapping)
        or detail.get("attempt_count") != budget
        or detail.get("budget") != budget
        or detail.get("candidate_tree") != checkpoint.expected_tree_sha
        or detail.get("failed_check_ids") != failed_ids
        or check_repair_state.get("candidate_tree") != checkpoint.expected_tree_sha
        or check_repair_state.get("failed_check_ids") != failed_ids
    ):
        return invalid("exhausted check-repair summary does not match its gate evidence")

    worktree_value = state.get("worktree")
    if not isinstance(worktree_value, str) or not worktree_value:
        return invalid("check-repair candidate worktree is missing")
    try:
        from .attempt_transaction import status_has_unstaged_or_untracked
        from .gitops import (
            candidate_tree_sha, current_head, index_tree_sha, resolve_tree, status_porcelain,
        )
        worktree = Path(worktree_value)
        if (
            current_head(worktree) != checkpoint.expected_head_sha
            or candidate_tree_sha(worktree) != checkpoint.expected_tree_sha
            or index_tree_sha(worktree) != checkpoint.expected_tree_sha
            or status_has_unstaged_or_untracked(status_porcelain(worktree))
        ):
            return invalid("check-repair candidate HEAD or tree changed")
    except (OSError, ValueError, RuntimeError):
        return invalid("check-repair candidate integrity cannot be verified")

    if state.get("expected_head_sha") not in {None, checkpoint.expected_head_sha}:
        return invalid("check-repair expected HEAD does not match its checkpoint")
    try:
        head_tree = resolve_tree(worktree, checkpoint.expected_head_sha)
    except (OSError, ValueError, RuntimeError):
        return invalid("check-repair expected HEAD tree cannot be verified")
    if state.get("expected_tree_sha") not in {None, head_tree}:
        return invalid("check-repair expected tree does not match its accepted HEAD")
    if state.get("staged_tree_sha") not in {None, checkpoint.expected_tree_sha}:
        return invalid("check-repair candidate tree does not match its checkpoint")

    return ResumeInfo(
        True,
        ResumePhase.DETERMINISTIC_GATE.value,
        resume_label(checkpoint),
        expected_tree=checkpoint.expected_tree_sha,
        review_cycle=cycle,
        operation=CHECK_REPAIR_RETRY_OPERATION,
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
    "CHECKPOINT_NAME", "CHECK_REPAIR_INTEGRITY_OPERATION", "CHECK_REPAIR_RETRY_OPERATION",
    "CHECKPOINT_INTEGRITY_OPERATION", "CONTRACT_REPAIR_OPERATION",
    "PHASE_STATUS", "RUN_SCHEMA_UNSUPPORTED", "machine_state_for_run", "STEP_ACCEPTANCE_OPERATION", "ResumeCheckpoint",
    "ResumeCheckpointError", "ResumeError", "ResumeInfo", "ResumeIntegrityError",
    "ResumeNotAllowedError", "ResumePhase", "ResumeRequiresOperatorError",
    "ResumeSchemaUnsupportedError",
    "checkpoint_payload", "mark_checkpoint_completed", "pipeline_version_from_state",
    "plan_identity_from_mapping", "read_checkpoint", "read_checkpoint_record",
    "resume_info", "resume_label", "write_checkpoint",
]
