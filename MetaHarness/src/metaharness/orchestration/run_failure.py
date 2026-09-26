"""The durable projection of every failure that left its recovery loop.

``RunFailure`` owns the single outcome a failure becomes: the durable step
artifact, the state fields that close the run, the status the recovery policy
authorizes, the fixed-point fingerprint of an exhausted check-repair episode
and the projection of an exception that escaped the coordinator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from ..agent.base import AgentError
from ..approval import ApprovalError
from ..execution_selection import ExecutionSelectionError
from ..gitops import GitError
from ..integrations.github import GitHubIntegrationError
from ..llm.chat import LLMError
from ..models import RunDisposition, RunMachineState
from ..plan_repository_validation import PlanRepositoryPreconditionError
from ..planning.protocol import PlanParseError
from ..recovery_policy import RecoveryStrategy
from ..redaction import redact, redact_mapping
from ..result import RunResult, atomic_write_text
from ..resume import (
    ResumeCheckpoint, ResumeCheckpointError, ResumeIntegrityError, ResumePhase,
    ResumeRequiresOperatorError,
    read_checkpoint,
)
from ..review import ReviewParseError
from ..run_options import (
    effective_repair_scope_policy, effective_run_config,
    read_run_options_for_state,
)
from ..state import RunStateStore
from ..usage import empty_usage, normalize_usage
from ..validation import ValidationError
from ..workspace import WorkspaceSetupError
from .pipeline_v2 import (
    FailureDetail, PipelineFailure, PipelineV2Context, PipelineV2Coordinator,
    check_repair_dir,
    check_repair_fingerprint, step_dir as cycle_step_dir,
)
from .recovery import normalize_exit_reason, project_exit
from .resume_integrity import validate_resume
from .shared import (
    CandidatePushError, CommitBoundaryError, OrchestrationError,
    ScopeApprovalRequired,
    StepExecutionFailure, bounded_v2_report, json_text, safe_candidate_tree,
)

if TYPE_CHECKING:
    from .runtime import RunRuntime

def _terminal_step_fields(
    state: Mapping[str, Any], failed_step: str | None, terminal: str = "failed"
) -> dict[str, Any]:
    """State fields that close every step when a run becomes terminal.

    The failed step and any step still marked ``running`` take *terminal*;
    later steps stay ``waiting``; ``current_step`` is cleared.
    """

    steps = state.get("steps") if isinstance(state, Mapping) else None
    if not isinstance(steps, list):
        return {"current_step": None}
    closed: list[Any] = []
    for item in steps:
        if isinstance(item, dict) and (
            item.get("id") == failed_step or item.get("status") == "running"
        ):
            item = {**item, "status": terminal}
        closed.append(item)
    return {"steps": closed, "current_step": None}


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (ResumeIntegrityError, ResumeRequiresOperatorError)):
        # Their stable ``.code`` is the authority: these reasons must stay
        # recognizable as permanently non-resumable.
        return exc.code
    if isinstance(exc, OrchestrationError) and str(exc).startswith("CHECK_PREFLIGHT_FAILED:"):
        return str(exc).split()[0]
    if isinstance(exc, OrchestrationError) and str(exc).startswith(
        "CHECK_INFRASTRUCTURE_UNAVAILABLE:"
    ):
        return "CHECK_INFRASTRUCTURE_UNAVAILABLE"
    if isinstance(exc, (OrchestrationError, ValidationError)):
        detail = str(exc)
        for code in (
            "AGENT_GIT_VIOLATION", "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH",
            "TREE_MISMATCH", "HEAD_MISMATCH", "DURABLE_ARTIFACT_CORRUPTED",
        ):
            if detail.startswith(code + ":"):
                return code
    if isinstance(exc, CandidatePushError):
        return exc.code
    if isinstance(exc, CommitBoundaryError):
        return "TOCTOU_FAILURE"
    if isinstance(exc, GitError):
        return "GIT_FAILURE"
    if isinstance(exc, AgentError):
        return getattr(exc, "code", "AGENT_FAILURE")
    if isinstance(exc, PlanRepositoryPreconditionError):
        return exc.code
    if isinstance(exc, ApprovalError):
        return "PLAN_APPROVAL_INVALID"
    if isinstance(exc, ExecutionSelectionError):
        return "EXECUTION_SELECTION_INVALID"
    if isinstance(exc, PlanParseError):
        return "PLANNER_OUTPUT_INVALID"
    if isinstance(exc, ReviewParseError):
        return "REVIEWER_OUTPUT_INVALID"
    if isinstance(exc, LLMError):
        # A transport that names its own stable condition (the exhausted
        # horizon) keeps that code; every other transport error stays the
        # generic ``LLM_FAILURE`` classification.
        return getattr(exc, "code", "LLM_FAILURE")
    if isinstance(exc, ValidationError):
        return "CHECK_SETUP_INVALID"
    if isinstance(exc, WorkspaceSetupError):
        return exc.code
    if isinstance(exc, GitHubIntegrationError):
        return getattr(exc, "code", "GITHUB_WORKSTREAM_FAILURE")
    return "INTERNAL_HARNESS_ERROR"


class RunFailure:
    """The durable outcome of every failure that left its recovery loop."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    @staticmethod
    def closing_step_fields(store: RunStateStore, terminal: str) -> dict[str, Any]:
        try:
            return _terminal_step_fields(store.load(), None, terminal)
        except (OSError, ValueError):
            return {}

    def step_failed(
        self, store: RunStateStore, run_dir: Path, failure: StepExecutionFailure,
    ) -> RunResult:
        _decision, terminal = project_exit(
            failure.reason, phase=self.checkpoint_phase(run_dir),
        )
        self.runtime.observability.trace_emit(
            "step.failed",
            phase="implementation",
            cycle=self.runtime.trace_cycle,
            step_id=failure.step_id,
            data={
                "reason": failure.reason,
                "detail": failure.detail,
                "profile_id": failure.profile_id,
                "tree_before": failure.tree_before,
                "tree_after": failure.tree_after,
                "changed_paths": [],
                "commit_sha": None,
            },
        )
        return self.v2_failed(
            store, run_dir, failure.reason, failure.step_id, failure.detail,
            usage=failure.usage, profile_id=failure.profile_id,
            tree_before=failure.tree_before, tree_after=failure.tree_after,
            mismatch=failure.mismatch,
            mismatch_clean=failure.clean_contract_mismatch,
            mismatch_retry_count=failure.mismatch_retry_count,
            initial_mismatch=failure.initial_mismatch,
            index_tree_after=failure.index_tree_after,
            step_dir=failure.step_dir,
            terminal_disposition=terminal.disposition,
            auto_resumable=terminal.resumable,
        )

    def v2_failed(
        self, store: RunStateStore, run_dir: Path, reason: str,
        step_id: str | None, detail: FailureDetail | None = None, *,
        usage: Mapping[str, int] | None = None,
        profile_id: str | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        mismatch: str | None = None,
        mismatch_clean: bool = False,
        mismatch_retry_count: int = 0,
        initial_mismatch: str | None = None,
        index_tree_after: str | None = None,
        step_dir: Path | None = None,
        terminal_disposition: RunDisposition | None = None,
        auto_resumable: bool | None = None,
    ) -> RunResult:
        """Persist a failure that left its recovery loop, with its step record.

        Without an explicit ``terminal_disposition`` the failure is projected
        by the recovery policy: only a hard stop becomes ``FAILED``.
        """

        repeated_fixed_point = False
        retry_fingerprint: list[Any] | None = None
        if reason == "CHECK_REPAIR_EXHAUSTED" and isinstance(detail, Mapping):
            raw_failed_ids = detail.get("failed_check_ids")
            candidate_tree = detail.get("candidate_tree")
            try:
                checkpoint = read_checkpoint(run_dir)
            except ResumeCheckpointError:
                checkpoint = None
            if (
                isinstance(raw_failed_ids, list)
                and all(isinstance(item, str) for item in raw_failed_ids)
                and isinstance(candidate_tree, str)
                and checkpoint is not None
                and checkpoint.stage is not None
            ):
                fingerprint = check_repair_fingerprint(
                    candidate_tree, raw_failed_ids, checkpoint.stage,
                    detail.get("strategy") if isinstance(detail.get("strategy"), str) else "",
                )
                # The durable fingerprint is compared against a JSON round
                # trip: its failed-check set is normalized to a list so a
                # reloaded state is the same identity as the written one.
                retry_fingerprint = [fingerprint[0], list(fingerprint[1]), *fingerprint[2:]]
                prior_check_repair = store.load().get("check_repair")
                repeated_fixed_point = (
                    isinstance(prior_check_repair, Mapping)
                    and prior_check_repair.get("operator_retry_fingerprint") == retry_fingerprint
                )
                if repeated_fixed_point:
                    reason = "CHECK_REPAIR_FIXED_POINT"
                    detail = {
                        **dict(detail),
                        "fixed_point_fingerprint": retry_fingerprint,
                        "operator_message": "Code change or additional repair authority required",
                    }
                    terminal_disposition = None
                    auto_resumable = False
        if terminal_disposition is None:
            reason = normalize_exit_reason(reason)
            terminal_disposition = project_exit(
                reason, phase=self.checkpoint_phase(run_dir),
                remote_required=reason == "PUSH_FAILED",
            )[1].disposition
        if reason == "CHECK_REPAIR_EXHAUSTED":
            self.runtime.observability.trace_emit(
                "check_repair.exhausted",
                phase="repair",
                cycle=self.runtime.trace_cycle,
                data={"reason": reason, "step_id": step_id},
                once=True,
            )
        elif reason == "CHECK_REPAIR_FIXED_POINT":
            self.runtime.observability.trace_emit(
                "check_repair.fixed_point",
                phase="repair",
                cycle=self.runtime.trace_cycle,
                data={
                    "step_id": step_id,
                    "fingerprint": retry_fingerprint,
                    "operator_message": "Code change or additional repair authority required",
                },
                once=True,
            )
        try:
            self.runtime.observability.update_v2_usage(store, run_dir)
        except (OSError, ValueError):
            pass
        state = store.load()
        fields = _terminal_step_fields(
            state, step_id,
            "waiting" if terminal_disposition is not RunDisposition.FAILED else "failed",
        )
        if reason in {"CHECK_REPAIR_EXHAUSTED", "CHECK_REPAIR_FIXED_POINT"} and isinstance(detail, Mapping):
            failure_detail = dict(detail)
            evidence_sha = failure_detail.get("latest_evidence_sha256")
            failed_ids = failure_detail.get("failed_check_ids")
            candidate_tree = failure_detail.get("candidate_tree")
            attempt_count = failure_detail.get("attempt_count")
            budget = failure_detail.get("budget")
            try:
                checkpoint = read_checkpoint(run_dir)
            except ResumeCheckpointError:
                checkpoint = None
            reports: list[dict[str, Any]] = []
            if (
                checkpoint is not None and checkpoint.stage is not None
                and isinstance(attempt_count, int) and not isinstance(attempt_count, bool)
            ):
                root = check_repair_dir(run_dir, checkpoint.review_cycle, checkpoint.stage) / "attempts"
                for number in range(1, attempt_count + 1):
                    report_path = root / f"{number:03d}" / "report.json"
                    if report_path.is_file():
                        try:
                            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
                        except OSError:
                            digest = None
                        reports.append({
                            "attempt": number,
                            "artifact": report_path.relative_to(run_dir).as_posix(),
                            "sha256": digest,
                        })
            failure_detail["repair_reports"] = reports
            prior_check_repair = state.get("check_repair")
            if retry_fingerprint is None:
                fingerprint = check_repair_fingerprint(
                    candidate_tree if isinstance(candidate_tree, str) else "",
                    failed_ids if isinstance(failed_ids, list) else [],
                    checkpoint.stage if checkpoint is not None and checkpoint.stage is not None else "",
                    failure_detail.get("strategy")
                    if isinstance(failure_detail.get("strategy"), str) else "",
                )
                retry_fingerprint = [fingerprint[0], list(fingerprint[1]), *fingerprint[2:]]
            fixed_point = reason == "CHECK_REPAIR_FIXED_POINT"
            if fixed_point:
                failure_detail["fixed_point_fingerprint"] = retry_fingerprint
                failure_detail["operator_message"] = "Code change or additional repair authority required"
            fields["check_repair"] = {
                **(dict(prior_check_repair) if isinstance(prior_check_repair, Mapping) else {}),
                "status": "fixed_point" if fixed_point else "exhausted",
                "attempt_count": attempt_count,
                "budget": budget,
                "failed_check_ids": list(failed_ids) if isinstance(failed_ids, list) else [],
                "candidate_tree": candidate_tree,
                "repair_reports": reports,
                "latest_evidence_sha256": evidence_sha,
                "failure_classification": "product_check",
                "operator_retry_fingerprint": retry_fingerprint,
                "next_action": (
                    "Code change or additional repair authority required"
                    if fixed_point else "Retry deterministic gate"
                ),
            }
            detail = failure_detail
        if auto_resumable is not None:
            fields["recovery_resumable"] = auto_resumable
        cycles = list(state.get("cycles") or [])
        current_cycle = state.get("cycle", 1)
        for index, cycle in enumerate(cycles):
            if isinstance(cycle, dict) and cycle.get("number") == current_cycle:
                cycles[index] = {
                    **cycle,
                    "status": (
                        "waiting" if terminal_disposition is not RunDisposition.FAILED else "failed"
                    ),
                    "failure": reason,
                }
                break
        if cycles:
            fields["cycles"] = cycles
        if step_id is not None:
            detail = f"step={step_id}" + (f" {detail}" if detail is not None else "")
            step_usage = normalize_usage(usage) if usage is not None else empty_usage()
            if isinstance(fields.get("steps"), list):
                fields["steps"] = [
                    {**item, "usage": step_usage,
                     "input_tokens": step_usage["input_tokens"],
                     "output_tokens": step_usage["output_tokens"]}
                    if isinstance(item, dict) and item.get("id") == step_id else item
                    for item in fields["steps"]
                ]
            step_dir = step_dir or (
                cycle_step_dir(run_dir, current_cycle, step_id)
            )
            if not (step_dir / "step.json").exists():
                if profile_id is None:
                    state_steps = state.get("steps")
                    profile_id = next(
                        (item.get("profile_id") for item in state_steps
                         if isinstance(item, dict) and item.get("id") == step_id),
                        None,
                    ) if isinstance(state_steps, list) else None
                atomic_write_text(step_dir / "step.json", json_text({
                    "id": step_id, "status": "FAILED", "reason": reason,
                    "profile_id": profile_id,
                    "tree_before": tree_before, "tree_after": tree_after,
                    **({"changed_paths": []} if reason == "AGENT_CONTRACT_MISMATCH" and tree_before == tree_after else {}),
                    **({"mismatch": bounded_v2_report(mismatch)} if mismatch else {}),
                    **({"mismatch_clean": mismatch_clean}
                       if reason == "AGENT_CONTRACT_MISMATCH" else {}),
                    **({"mismatch_retry_count": mismatch_retry_count}
                       if mismatch_retry_count else {}),
                    **({"initial_mismatch": bounded_v2_report(initial_mismatch)}
                       if initial_mismatch else {}),
                    **({"index_tree_after": index_tree_after}
                       if index_tree_after else {}),
                    "usage": step_usage,
                }))
        state = self.persist_exit(store, reason, detail, terminal_disposition, **fields)
        return RunResult.of(run_dir, state)

    @staticmethod
    def checkpoint_phase(
        run_dir: Path, *, default: ResumePhase = ResumePhase.IMPLEMENT_STEP,
    ) -> ResumePhase:
        try:
            checkpoint = read_checkpoint(run_dir)
        except ResumeCheckpointError:
            return default
        return checkpoint.phase if checkpoint is not None else default

    def persist_exit(
        self, store: RunStateStore, reason: str, detail: FailureDetail | None,
        disposition: RunDisposition, **fields: Any,
    ) -> dict[str, Any]:
        """Write one terminal or waiting outcome; FAILED only for hard stops."""

        if isinstance(detail, str):
            detail = redact(detail, self.runtime.secrets)
        elif isinstance(detail, Mapping):
            detail = redact_mapping(detail, self.runtime.secrets)
        elif detail is not None:
            raise TypeError("failure detail must be text or a structured mapping")
        if disposition is RunDisposition.FAILED:
            return store.record_failure(reason, detail, **fields)
        failure = {"reason": reason}
        if detail is not None:
            failure["detail"] = detail
        return store.set_run_state(
            RunMachineState(disposition=disposition, reason=reason),
            failure=failure, **fields,
        )

    def project_exception(
        self, store: RunStateStore, run_dir: Path, exc: Exception, *,
        operation: str = "orchestrator",
    ) -> RunResult:
        """Project an exception that escaped every recovery loop."""

        reason = normalize_exit_reason(_failure_reason(exc))
        phase = self.checkpoint_phase(run_dir)
        _decision, terminal = project_exit(reason, phase=phase)
        detail: FailureDetail
        auto_resumable: bool | None = None
        if reason == "INTERNAL_HARNESS_ERROR":
            message = " ".join(str(exc).split())[:500]
            detail = {
                "exception_type": type(exc).__name__,
                "message": message,
                "phase": phase.value,
                "operation": operation,
            }
            auto_resumable = self._checkpoint_is_retryable(store, run_dir)
            disposition = (
                RunDisposition.WAIT_EXTERNAL if auto_resumable else RunDisposition.FAILED
            )
        else:
            detail = " ".join(str(exc).split())[:500]
            disposition = terminal.disposition
        fields = self.closing_step_fields(
            store, "waiting" if disposition is not RunDisposition.FAILED else "failed",
        )
        if auto_resumable is not None:
            fields["recovery_resumable"] = auto_resumable
        state = self.persist_exit(store, reason, detail, disposition, **fields)
        return RunResult.of(run_dir, state)

    def _checkpoint_is_retryable(self, store: RunStateStore, run_dir: Path) -> bool:
        """Only make an internal crash resumable after the full resume gate passes."""

        try:
            state = store.load()
            checkpoint = read_checkpoint(run_dir)
            if checkpoint is None:
                return False
            options, _digest = read_run_options_for_state(run_dir, state)
            config = effective_run_config(self.runtime.config, options)
            validate_resume(
                config=config,
                repair_scope=effective_repair_scope_policy(options),
                run_dir=run_dir,
                state=state,
                checkpoint=checkpoint,
                staging_remote=config.repository.remote,
            )
        except Exception:
            return False
        return True

    def run_pipeline(
        self, store: RunStateStore, pipeline: PipelineV2Context,
        start: ResumeCheckpoint, engine: PipelineV2Coordinator, *, resumed: bool,
    ) -> RunResult:
        """Run the generic coordinator and project a failure that left it."""

        self.runtime.last_selection = pipeline.selection
        try:
            return engine.run(start, resumed=resumed)
        except PipelineFailure as failure:
            reason = normalize_exit_reason(failure.reason)
            if reason != failure.reason:
                failure.reason = reason
                failure.detail = "external executor authorization is required"
            checkpoint_phase = self.checkpoint_phase(pipeline.run_dir, default=start.phase)
            decision, terminal = project_exit(
                failure.reason, phase=checkpoint_phase,
                remote_required=failure.reason == "PUSH_FAILED",
            )
            state = store.load()
            tree = state.get("staged_tree_sha")
            if not isinstance(tree, str):
                tree = safe_candidate_tree(pipeline.info.worktree)
            cycle = state.get("cycle") if isinstance(state.get("cycle"), int) else None
            phase = state.get("status") if isinstance(state.get("status"), str) else "run"
            recovery = self.runtime.recovery(store)
            initial = recovery.stop(
                failure.reason, phase=phase, cycle=cycle, step_id=failure.step_id,
                tree_before=tree, tree_after=safe_candidate_tree(pipeline.info.worktree),
            )
            if (
                initial.strategy is not RecoveryStrategy.HARD_STOP
                or terminal.disposition is not RunDisposition.FAILED
            ):
                recovery.trace(
                    "recovery.exhausted", reason=failure.reason, decision=decision,
                    attempt=1, tree_before=tree,
                    tree_after=safe_candidate_tree(pipeline.info.worktree),
                    budget_remaining=0, phase=phase, cycle=cycle,
                    step_id=failure.step_id, terminal_status=terminal.status,
                    checkpoint_phase=checkpoint_phase,
                )
            return self.v2_failed(
                store, pipeline.run_dir, failure.reason, failure.step_id, failure.detail,
                terminal_disposition=terminal.disposition,
                auto_resumable=terminal.resumable,
            )
        except StepExecutionFailure as failure:
            return self.step_failed(store, pipeline.run_dir, failure)
        except ScopeApprovalRequired:
            return RunResult.of(pipeline.run_dir, store.load())
        except (ResumeIntegrityError, ResumeRequiresOperatorError):
            raise
        except Exception as exc:
            return self.project_exception(
                store, pipeline.run_dir, exc, operation="pipeline_coordinator",
            )
