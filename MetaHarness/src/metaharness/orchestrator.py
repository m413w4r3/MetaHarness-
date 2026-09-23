"""The MetaHarness façade: prepare one run and drive its generic pipeline."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

from .agent.base import (
    AGENT_AUTH_FAILURE,
    AGENT_RUNTIME_FAILED,
    AGENT_PROTOCOL_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentRunRequest,
    AgentScopeError,
)
from .agent.diagnostics import write_token_diagnostics
from .agent.protocol import contract_mismatch_explanation, deferred_verify_dependency
from .prompt_contracts import (
    build_implementer_payload,
    build_final_review_payload,
    write_prompt_diagnostics,
)
from .agent.execution import (
    ExecutorRuntimeConfig,
    executor_for_profile,
)
from .approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    read_plan_approval,
    read_scope_approval,
    write_check_authority,
    wait_for_plan_approval,
)
from .config import load_config
from .context import build_context, render_context
from .evidence import (
    EvidenceBundle,
    bounded_semantic_diff,
    collect_evidence,
    scan_staged_security,
)
from .validation import (
    ValidationError,
    check_result_json,
    config_with_check_authority,
    run_check_preflights,
)
from .gitops import (
    BaseMovedError,
    BasePushError,
    GitError,
    WorktreeInfo,
    assert_clean,
    branch_exists,
    commit_parents,
    publish_fast_forward_base,
    restore_paths_from_tree,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_step_tree,
    create_run_worktree,
    current_head,
    delete_run_branch,
    git_root,
    index_tree_sha,
    local_branches,
    build_repository_reference,
    build_run_branch,
    immutable_commit_web_url,
    compare_commits_web_url,
    render_repository_reference,
    path_exists_in_tree,
    push_run_branch,
    remote_run_branch_tip,
    registered_worktrees,
    resolve_commit,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
    status_porcelain,
    symbolic_head,
    stage_all,
    staged_diff,
    repository_remote_url,
    validate_run_branch,
)
from .commit_gate import (
    CommitSafetyError,
    accepted_step_record,
    assert_deferred_verifications_resolved,
    commit_safety_gate,
    parse_deferred_verification,
)
from .llm.chat import LLMError, OpenAIChatTextClient
from .integrations.github import (
    GitHubIntegrationError,
    GitHubWorkstreamError,
    GitHubWorkstreamClient,
    NullGitHubWorkstreamClient,
)
from .recommendation import (
    ExecutionRecommender,
    RecommendationError,
    write_recommendation_error,
)
from .execution_selection import (
    SCHEMA_VERSION as EXECUTION_SELECTION_SCHEMA,
    ExecutionSelectionError,
    ensure_execution_selection,
    ensure_cycle_execution_selection,
    read_execution_selection_with_sha256,
    read_cycle_execution_selection,
    resolve_execution_selection,
    resolve_cycle_execution_selection,
    validate_cycle_execution_selection,
    validate_execution_selection,
)
from .models import (
    CycleKind,
    ExecutionRole,
    ExecutionSelection,
    GateStage,
    ModelProfile,
    HarnessConfig,
    ImplementationStep,
    PlanDecision,
    PublishMode,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    RunStatus,
    profile_driver_name,
)
from .planning_v2 import (
    PlanParseError,
    PlannerV2,
    RepairPlannerV2,
    StepContractRepairPlanner,
    TaskPlanV2,
    V2PlanParseError,
    parse_task_plan_v2,
    persist_recovered_plan_artifacts,
    read_approved_step_contract,
    read_set_paths,
    validate_decomposition_policy,
    validate_execution_mode_policy,
    validate_implementation_bundle,
    render_repair_plan_summary,
    render_repair_step_index,
    render_step_contract,
    parse_step_contract_repair,
)
from .plan_repository_validation import (
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
    validate_plan_repository_topology,
)
from .plan_recovery import (
    PLAN_SOURCE_OPERATOR,
    PlanRecoveryError,
    plan_recovery_info,
    plan_source,
    recoverable_plan_source_status,
    validate_replacement_text,
    write_plan_recovery_record,
)
from .resume import (
    PHASE_STATUS,
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeError,
    ResumeIntegrityError,
    ResumeNotAllowedError,
    ResumePhase,
    ResumeRequiresOperatorError,
    mark_checkpoint_completed,
    plan_identity_from_mapping,
    read_checkpoint,
    read_checkpoint_record,
    pipeline_version_from_state,
    resume_info,
    resume_label,
    write_checkpoint,
)
from .usage import normalize_usage, empty_usage, phase_usage_summary, read_usage_artifact
from .redaction import config_secret_values, redact, redact_file
from .diagnostics import write_run_diagnostics
from .profiles import (
    ProfileError,
    build_llm_endpoint,
    profile_for_role,
    profile_execution_fingerprint,
    profiles_for_config,
)
from .trace import TraceSink, TraceStream
from .result import RunResult, ResultArtifactError, atomic_write_text, write_repair_task
from .review import Reviewer, ReviewParseError, ReviewResult
from .recovery_policy import RecoveryDecision, RecoveryDisposition, classify_failure
from .state import RunStateStore
from .workspace import WorkspaceSetupError, prepare_workspace
from .run_options import (
    EffectiveRepairScopePolicy,
    RunOptions,
    RunOptionsError,
    effective_repair_scope_policy,
    effective_run_config,
    read_run_options_for_state,
    write_run_options,
)
from .orchestration.shared import (
    CandidatePushError,
    CheckRepairScope,
    CommitBoundaryError,
    CycleArtifactService,
    DeferredStepExecutionOutcome,
    GitOwnership,
    OrchestrationError,
    ScopeApprovalRequired,
    StepExecutionFailure,
    StepExecutionOutcome,
    _AGENT_ARTIFACTS,
    _BOUNDED_NO_CHANGE_MISMATCH,
    _CHECK_ATTEMPT_ARTIFACTS,
    _PLANNER_ATTEMPT_ARTIFACTS,
    _RECOVERY_ATTEMPT_ARTIFACTS,
    _REVIEW_ATTEMPT_ARTIFACTS,
    _REVISION_ARTIFACTS,
    _REVISION_ATTEMPT_ARTIFACTS,
    _SYNTHETIC_NO_CHANGE_MISMATCH,
    _archive_attempt,
    _archive_attempt_target,
    _archive_attempt_tree,
    _bounded_report,
    _bounded_v2_report,
    _check_payload,
    _git_ownership,
    _git_ownership_payload,
    _is_object_id,
    _json_text,
    _new_status_lines,
    _ownership_violations,
    _paths_detail,
    _read_json_artifact,
    _record_failure_tree,
    _repair_checks_payload,
    _safe_candidate_tree,
    _safe_index_tree,
    _safe_status,
    _status_has_unstaged_or_untracked,
)

from .orchestration.revision import (
    ReviewContextBuilder,
    ReviewCycleInput,
    RevisionRunner,
    _SCOPE_REQUEST_ROUTE,
    _bounded_previous_revision_report,
    _deferred_contract_mismatches,
    _future_step_ownership,
    _has_deferred_contract_mismatches,
    _review_payload,
    review_cycle_revision_report,
)
from .orchestration.check_repair import (
    CheckRepairAttempt,
    CheckRepairCoordinator,
    GateAcceptanceService,
    _check_repair_prompt,
    _check_repair_scope_candidates,
    _hard_integrity_failures,
    gate_mutable_authority,
    _soft_check_failures,
)
from .orchestration.scope_repair import (
    _build_scope_delta,
    _ensure_scope_delta,
)
from .orchestration.candidate import (
    CandidateLifecycle,
    accepted_chain_records,
    validate_accepted_chain,
    _candidate_commit_path,
    _commit_web_url,
)
from .orchestration.pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    PipelineV2Coordinator,
    PipelineV2Operations,
    check_repair_attempt_dir,
    check_repair_dir,
    correction_dir,
    gate_dir,
    implementation_steps_dir,
    review_dir,
    semantic_revision_dir,
    step_dir as cycle_step_dir,
)
from .orchestration.resume_validation import (
    ResumedRun,
    _accepted_review,
    _load_evidence,
    _load_revision,
    _persist_planner_conversation,
    _read_planner_conversation,
    _read_repository_reference,
    _reusable_pre_checks,
    _semantic_revision_scope,
    candidate_evidence,
    completed_step_records,
    load_correction_plan,
    read_candidate_record,
    read_cycle_record,
    validate_resume,
    verify_correction_scope,
)


_OUTPUT_DISCIPLINE_TARGETS = {
    ExecutionRole.PLANNER: "META PLAN v2 only",
    ExecutionRole.IMPLEMENTER: "<=8 lines; <=1200 characters",
    ExecutionRole.REPAIR: "<=6 lines; <=800 characters",
    ExecutionRole.REVISER: "<=10 lines; <=1500 characters",
    ExecutionRole.REVIEWER: "META REVIEW v1; terse material findings only",
}

def _chat_client(endpoint: Any, environment: Mapping[str, str]) -> OpenAIChatTextClient:
    """Construct the production client with the runtime mapping.

    The constructor is inspected once so embedded clients can receive the
    runtime environment when they declare it.
    """

    constructor = OpenAIChatTextClient
    try:
        parameters = inspect.signature(constructor).parameters.values()
        accepts_environment = any(
            parameter.name == "environment"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_environment = True
    if accepts_environment:
        return constructor(endpoint, environment=environment)
    return constructor(endpoint)


_MAX_REVIEW_FALLBACK_DIFF_BYTES = 32 * 1024
def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"



def _safe_run_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationError("run_id must be a non-empty path component")
    value = value.strip()
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise OrchestrationError("run_id must be one safe path component")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise OrchestrationError("run_id contains unsupported characters")
    # The run id is also the last component of the run branch name.
    if ".." in value or value.endswith(".") or value.endswith(".lock"):
        raise OrchestrationError("run_id must be a valid Git ref component")
    return value


def _required_checks_summary(evidence: EvidenceBundle) -> str:
    """Keep required-check authority while excluding stdout/stderr blobs."""

    rows: list[dict[str, Any]] = []
    required = set(evidence.required_check_ids)
    for raw in evidence.checks:
        item = dict(raw) if isinstance(raw, Mapping) else check_result_json(raw)
        name = item.get("name")
        if name not in required and required:
            continue
        rows.append({
            "id": name,
            "required": bool(item.get("required", name in required)),
            "exit_code": item.get("exit_code"),
            "timed_out": bool(item.get("timed_out", False)),
            "workspace_mutated": bool(item.get("workspace_mutated", False)),
            "status": (
                "timed_out" if item.get("timed_out") else
                "passed" if item.get("exit_code") == 0 else "failed"
            ),
        })
    return _json_text({
        "required_check_ids": list(evidence.required_check_ids),
        "failures": list(evidence.failures),
        "checks": rows,
        "deterministic_passed": evidence.deterministic_passed,
    })


def _diffstat(diff: str, changed_files: Sequence[str]) -> str:
    additions = sum(
        1 for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return _json_text({
        "files": len(tuple(changed_files)),
        "insertions": additions,
        "deletions": deletions,
    })


def _compact_cycle_summary(text: str) -> str:
    """Remove check output and worker narration from cycle history."""

    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {
                    "stdout", "stderr", "stdout_tail", "stderr_tail",
                    "final", "final_message", "agent_report", "step_reports",
                    "semantic_revision_report",
                }
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return _json_text(clean(value))


def _review_code_evidence(
    *,
    repository_reference: RepositoryReference,
    base_sha: str,
    candidate_sha: str,
    evidence: EvidenceBundle,
    remote_sha: str | None = None,
    remote_branch: str | None = None,
    remote_name: str | None = None,
) -> str:
    diff_bytes = evidence.diff.encode("utf-8", errors="replace")

    candidate_url = immutable_commit_web_url(
        repository_reference,
        candidate_sha,
    )
    compare_url = compare_commits_web_url(
        repository_reference,
        base_sha,
        candidate_sha,
    )

    # A web URL describes where a candidate might be inspectable; it does not
    # prove that the exact candidate was pushed. Durable remote authority is
    # required before remote exploration can be offered to a reviewer.
    remote_pushed = (
        remote_sha == candidate_sha
        and isinstance(remote_branch, str)
        and bool(remote_branch)
        and isinstance(remote_name, str)
        and bool(remote_name)
    )
    remote_available = (
        remote_pushed and candidate_url is not None and compare_url is not None
    )

    payload: dict[str, Any] = {
        "authority": "immutable_candidate_commit",
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "candidate_tree_sha": evidence.staged_tree_sha,
        "candidate_url": candidate_url,
        "compare_url": compare_url,
        "remote_exploration": "ALLOWED" if remote_available else "UNAVAILABLE",
        "remote_authority": {
            "remote": remote_name,
            "remote_branch": remote_branch,
            "remote_sha": remote_sha,
            "verified": remote_pushed,
        },
        "full_diff_bytes": len(diff_bytes),
        "full_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "diffstat": json.loads(_diffstat(evidence.diff, evidence.changed_files)),
        "inline_full_diff": False,
    }

    if not remote_available:
        excerpt, truncated, full_bytes = bounded_semantic_diff(
            evidence.diff,
            _MAX_REVIEW_FALLBACK_DIFF_BYTES,
        )
        payload["inline_fallback"] = {
            "truncated": truncated,
            "full_diff_bytes": full_bytes,
            "excerpt": excerpt,
        }

    return _json_text(payload)




def _safe_agent_result_payload(result: Any) -> dict[str, Any]:
    """Persist bounded protocol metadata, never the raw backend result."""

    return {
        "status": getattr(result, "status", None),
        "exit_reason": getattr(result, "exit_reason", None),
        "exit_code": getattr(result, "exit_code", None),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "usage": normalize_usage(getattr(result, "usage", None)),
        "driver": getattr(result, "driver", None),
        "backend_reason": getattr(result, "backend_reason", None),
    }




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


def _bounded_parse_detail(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


@dataclasses.dataclass(frozen=True)
class _V2Setup:
    """A planned, approved and prepared v2 run, ready for its first step."""

    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: Any
    info: WorktreeInfo
    ownership_before: GitOwnership
    base_tree_sha: str
    checkpoint: ResumeCheckpoint | None


class Orchestrator:
    """The façade of one pipeline-v2 run.

    It prepares the run, freezes its options and execution selection, then
    delegates every execution phase to :class:`PipelineV2Coordinator`.
    """

    def __init__(
        self,
        config: HarnessConfig,
        *,
        planner_client: Any | None = None,
        reviewer_client: Any | None = None,
        recommender_client: Any | None = None,
        github_client: GitHubWorkstreamClient | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self._planner_client = planner_client
        self._reviewer_client = reviewer_client
        self._recommender_client = recommender_client
        self._github_client = (
            github_client if github_client is not None else NullGitHubWorkstreamClient()
        )
        # The local JSONL sink is always created per run.  This optional sink
        # is an observation-only extension point (for example Nimbalyst).
        self._trace_sink = trace_sink
        self._secrets: tuple[str, ...] = ()
        self._effective_repair_scope = EffectiveRepairScopePolicy(
            "deny-expansion", 4, "run-options"
        )
        self._runtime_environment = (
            config.runtime_environment
            if config.runtime_environment
            else os.environ
        )

    def _begin_trace(
        self, run_dir: Path, run_id: str, *, created: bool, pipeline_version: int = 2,
    ) -> None:
        """Attach the observation stream without changing run authority."""

        self._trace = TraceStream(
            run_dir,
            run_id,
            pipeline_version=pipeline_version,
            sink=getattr(self, "_trace_sink", None),
            secrets=getattr(self, "_secrets", ()),
        )
        if created:
            self._trace.emit(
                "run.created",
                phase="run",
                cycle=1,
                data={"status": RunStatus.CREATED.value},
            )

    def _trace_emit(
        self,
        event: str,
        *,
        phase: str | None = None,
        cycle: int | None = None,
        step_id: str | None = None,
        data: Mapping[str, Any] | None = None,
        once: bool = False,
    ) -> None:
        stream = getattr(self, "_trace", None)
        if stream is None:
            return
        kwargs = {
            "phase": phase,
            "cycle": cycle,
            "step_id": step_id,
            "data": data or {},
        }
        if once:
            # Terminal events describe the run, not a cycle.  A resumed
            # terminal projection must remain idempotent even when the
            # durable state exposes a different cycle number.
            if event in {"run.created", "run.failed", "run.completed"}:
                if stream.has_event(event):
                    return
                stream.emit(event, **kwargs)
            else:
                stream.emit_once(event, **kwargs)
        else:
            stream.emit(event, **kwargs)

    def _trace_recovery(
        self,
        event: str,
        *,
        reason: str,
        decision: RecoveryDecision,
        attempt: int,
        tree_before: str | None,
        tree_after: str | None,
        budget_remaining: int,
        phase: str,
        cycle: int | None = None,
        step_id: str | None = None,
        recovered: bool | None = None,
    ) -> None:
        """Record one secret-free recovery transition with its tree boundary."""

        data: dict[str, Any] = {
            "reason": reason,
            "disposition": decision.disposition.value,
            "attempt": attempt,
            "tree_before": tree_before,
            "tree_after": tree_after,
            "budget_remaining": budget_remaining,
        }
        if recovered is not None:
            data["recovered"] = recovered
        self._trace_emit(
            event, phase=phase, cycle=cycle, step_id=step_id, data=data,
        )

    @staticmethod
    def _trace_time() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _trace_selected_profile(
        self, profile_id: str, role: ExecutionRole, *, step_id: str | None = None,
    ) -> Any | None:
        selection = getattr(self, "_last_selection", None)
        if selection is None:
            return None
        if step_id is not None:
            for item in getattr(selection, "steps", ()):
                if getattr(item, "step_id", None) == step_id:
                    selected = getattr(item, "implementer", None)
                    if getattr(selected, "profile_id", None) == profile_id:
                        return selected
                    for fallback in getattr(item, "fallbacks", ()):
                        if getattr(fallback, "profile_id", None) == profile_id:
                            return fallback
        name = {
            ExecutionRole.PLANNER: "planner",
            ExecutionRole.REPAIR: "check_repair",
            ExecutionRole.REVIEWER: "final_reviewer",
            ExecutionRole.REVISER: "semantic_reviser",
        }.get(role)
        selected = getattr(selection, name, None) if name is not None else None
        if getattr(selected, "profile_id", None) == profile_id:
            return selected
        fallback_name = {
            ExecutionRole.REPAIR: "check_repair_fallbacks",
            ExecutionRole.REVISER: "semantic_reviser_fallbacks",
        }.get(role)
        for fallback in getattr(selection, fallback_name, ()) if fallback_name else ():
            if getattr(fallback, "profile_id", None) == profile_id:
                return fallback
        return None

    @staticmethod
    def _github_result_number(value: Any, kind: str) -> int:
        """Extract only the public numeric identifier from an integration result."""

        number = value if isinstance(value, int) and not isinstance(value, bool) else None
        if number is None and isinstance(value, Mapping):
            candidate = value.get("number")
            number = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
        if number is None:
            candidate = getattr(value, "number", None)
            number = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
        if number is None or number <= 0:
            raise GitHubWorkstreamError(f"GitHub {kind} response did not contain a valid number")
        return number

    @staticmethod
    def _github_issue_title(plan_title: str, run_id: str) -> str:
        first_line = next((line.strip() for line in plan_title.splitlines() if line.strip()), "MetaHarness run")
        title = f"MetaHarness: {first_line} ({run_id})"
        return title[:240]

    @staticmethod
    def _github_metadata_payload(state: Mapping[str, Any]) -> dict[str, int | str]:
        payload: dict[str, int | str] = {}
        for key in (
            "remote_branch", "issue_number", "pull_request_number",
            "reviewed_candidate_sha",
        ):
            value = state.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                payload[key] = value
        return payload

    def _ensure_github_issue_metadata(
        self,
        *,
        store: RunStateStore,
        run_id: str,
        plan_title: str,
        info: WorktreeInfo,
    ) -> None:
        """Resolve optional issue metadata without reading issue content into authority."""

        github = self.config.github
        if not github.enabled or github.issue_mode == "off":
            return
        state = store.load()
        current = state.get("issue_number")
        if isinstance(current, int) and not isinstance(current, bool) and current > 0:
            if github.issue_mode == "link-existing" and github.issue_number not in {None, current}:
                raise GitHubWorkstreamError(
                    "configured GitHub issue does not match the persisted workstream issue",
                    code="GITHUB_CONFIG_INVALID",
                )
            return
        if github.issue_mode == "link-existing":
            requested = github.issue_number
            if requested is None:
                raise GitHubWorkstreamError(
                    "github.issue_number is required for link-existing",
                    code="GITHUB_CONFIG_INVALID",
                )
            try:
                issue = self._github_client.read_issue(requested)
            except Exception:
                raise GitHubWorkstreamError("GitHub issue operation failed") from None
            if issue is None:
                raise GitHubWorkstreamError(
                    "requested GitHub issue was not found",
                    code="GITHUB_ISSUE_NOT_FOUND",
                )
            number = self._github_result_number(issue, "issue")
            if number != requested:
                raise GitHubWorkstreamError(
                    "GitHub issue response did not match the requested issue",
                    code="GITHUB_ISSUE_NOT_FOUND",
                )
            event = "workstream.issue.linked"
        else:
            title = self._github_issue_title(plan_title, run_id)
            body = (
                "MetaHarness workstream metadata.\n\n"
                f"Run ID: {run_id}\n"
                f"Branch: {info.branch}\n"
                f"Base ref: {self.config.base_ref}\n"
            )
            try:
                issue = self._github_client.create_issue(title, body)
            except Exception:
                raise GitHubWorkstreamError("GitHub issue operation failed") from None
            number = self._github_result_number(issue, "issue")
            event = "workstream.issue.created"

        state = store.update(status=state.get("status", RunStatus.CREATED), issue_number=number)
        self._trace_emit(
            event,
            phase="setup",
            cycle=1,
            data={"issue_number": number},
            once=True,
        )
        self._trace_emit(
            "workstream.metadata",
            phase="setup",
            cycle=1,
            data=self._github_metadata_payload(state),
        )

    def _ensure_github_pull_request_metadata(
        self,
        *,
        store: RunStateStore,
        run_id: str,
        info: WorktreeInfo,
        commit_sha: str,
        cycle: int | None,
    ) -> None:
        """Create the requested PR from the exact reviewed run branch."""

        github = self.config.github
        if not github.enabled or github.pull_request_mode == "off":
            return
        state = store.load()
        current = state.get("pull_request_number")
        if isinstance(current, int) and not isinstance(current, bool) and current > 0:
            return
        if (
            not self.config.publish.enabled
            or self.config.publish.mode != PublishMode.RUN_BRANCH.value
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request creation requires a published run branch",
                code="GITHUB_PR_REQUIRES_RUN_BRANCH",
            )
        review = state.get("review")
        reviewed_candidate_sha = state.get("reviewed_candidate_sha")
        accepted_candidate_sha = state.get("candidate_commit_sha")
        if (
            not isinstance(review, Mapping)
            or review.get("verdict") != ReviewVerdict.PASS.value
            or review.get("route") not in {None, ReviewRoute.NONE.value}
            or not _is_object_id(reviewed_candidate_sha)
            or not _is_object_id(accepted_candidate_sha)
            or reviewed_candidate_sha != accepted_candidate_sha
            or reviewed_candidate_sha != commit_sha
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request candidate authority is not an exact PASS candidate",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            )
        try:
            if current_head(info.worktree) != accepted_candidate_sha:
                raise GitHubWorkstreamError(
                    "local accepted candidate does not match the reviewed candidate",
                    code="GITHUB_PR_CANDIDATE_MISMATCH",
                )
            remote_tip = remote_run_branch_tip(
                info.source_repo,
                remote=getattr(
                    getattr(self.config, "repository", None),
                    "remote",
                    self.config.publish.remote,
                ),
                branch=info.branch,
            )
        except GitHubWorkstreamError:
            raise
        except (GitError, OSError, ValueError):
            raise GitHubWorkstreamError(
                "remote run branch tip could not be verified",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            ) from None
        if remote_tip != reviewed_candidate_sha:
            raise GitHubWorkstreamError(
                "remote run branch tip does not match the reviewed candidate",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            )
        plan_state = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
        plan_title = plan_state.get("title") if isinstance(plan_state.get("title"), str) else "MetaHarness run"
        title = self._github_issue_title(plan_title, run_id)
        body = (
            "MetaHarness reviewed workstream metadata.\n\n"
            f"Run ID: {run_id}\n"
            f"Reviewed commit: {commit_sha}\n"
            f"Head branch: {info.branch}\n"
            f"Base branch: {self.config.base_ref}\n"
        )
        try:
            pull_request = self._github_client.create_pull_request(
                title, body, self.config.base_ref, info.branch
            )
        except Exception:
            raise GitHubWorkstreamError("GitHub pull-request operation failed") from None
        number = self._github_result_number(pull_request, "pull request")

        state = store.update(
            status=state.get("status", RunStatus.CREATED),
            remote_branch=info.branch,
            pull_request_number=number,
        )
        self._trace_emit(
            "workstream.pull_request.created",
            phase="publication",
            cycle=cycle,
            data={
                "remote_branch": info.branch,
                "pull_request_number": number,
                "reviewed_candidate_sha": reviewed_candidate_sha,
            },
            once=True,
        )
        self._trace_emit(
            "workstream.metadata",
            phase="publication",
            cycle=cycle,
            data=self._github_metadata_payload(state),
        )

    def _validate_github_publication_mode(self) -> None:
        github = self.config.github
        if (
            github.enabled
            and github.pull_request_mode == "create"
            and (
                not self.config.publish.enabled
                or self.config.publish.mode != PublishMode.RUN_BRANCH.value
            )
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request creation requires a published run branch",
                code="GITHUB_PR_REQUIRES_RUN_BRANCH",
            )

    def _trace_session(
        self,
        *,
        profile: ModelProfile | None,
        selected: Any | None,
        role: ExecutionRole,
        prompt_bytes: int | None,
        started_at: str,
        started_mono: float,
        tree_before: str | None = None,
        result: Any | None = None,
        exit_reason: str | None = None,
        final_message: str | None = None,
    ) -> dict[str, Any]:
        """Build session metadata without deriving unavailable metrics."""

        fingerprint = getattr(selected, "config_sha256", None)
        if fingerprint is None and profile is not None:
            try:
                fingerprint = profile_execution_fingerprint(
                    profile,
                    agent_env_allowlist=self.config.codex_runtime.env_allowlist,
                    codex_home=self.config.codex_runtime.home,
                    claude_config_home=self.config.claude_runtime.home,
                )
            except (TypeError, ValueError, AttributeError):
                fingerprint = None
        raw_result = getattr(result, "raw_result", None) if result is not None else None
        if final_message is None and result is not None:
            candidate_message = getattr(result, "final_message", None)
            if isinstance(candidate_message, str):
                final_message = candidate_message
        raw_usage = getattr(raw_result, "usage", None) if raw_result is not None else None
        usage = raw_usage if isinstance(raw_usage, Mapping) else (
            getattr(result, "usage", None)
            if result is not None and raw_result is None else None
        )

        aliases = {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
            "cache_write_input_tokens": (
                "cache_write_input_tokens", "cache_creation_input_tokens",
            ),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "reasoning_output_tokens": ("reasoning_output_tokens",),
        }
        nested = {
            "cached_input_tokens": (
                ("prompt_tokens_details", "cached_tokens"),
                ("input_tokens_details", "cached_tokens"),
            ),
            "cache_write_input_tokens": (
                ("prompt_tokens_details", "cache_write_tokens"),
                ("input_tokens_details", "cache_write_tokens"),
            ),
            "reasoning_output_tokens": (
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_tokens_details", "reasoning_tokens"),
            ),
        }

        def metric(name: str) -> int | None:
            if not isinstance(usage, Mapping):
                return None
            for alias in aliases.get(name, (name,)):
                value = usage.get(alias)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
            for container, key in nested.get(name, ()):
                details = usage.get(container)
                if isinstance(details, Mapping):
                    value = details.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        return value
            return None

        return {
            "driver": (
                profile_driver_name(profile.driver) if profile is not None
                else getattr(selected, "driver", None)
                or getattr(result, "driver", None)
            ),
            "driver_version": (
                getattr(result, "driver_version", None) if result is not None else None
            ) or getattr(selected, "driver_version", None)
            or (profile.driver_version if profile is not None else None),
            "provider": (
                profile.provider if profile is not None
                else getattr(selected, "provider", None)
            ),
            "model": profile.model if profile is not None else getattr(selected, "model", None),
            "effort": profile.effort if profile is not None else getattr(selected, "effort", None),
            "profile_id": getattr(selected, "profile_id", None) or (profile.id if profile is not None else None),
            "profile_fingerprint": fingerprint,
            "role": role.value,
            "started_at": started_at,
            "finished_at": self._trace_time() if result is not None or exit_reason is not None else None,
            "wall_time_ms": round((time.perf_counter() - started_mono) * 1000) if result is not None or exit_reason is not None else None,
            "prompt_bytes": prompt_bytes,
            "final_message_bytes": (
                len(final_message.encode("utf-8", errors="replace"))
                if isinstance(final_message, str) else None
            ),
            "output_discipline_target": _OUTPUT_DISCIPLINE_TARGETS.get(role),
            "input_tokens": metric("input_tokens"),
            "cached_input_tokens": metric("cached_input_tokens"),
            "cache_write_input_tokens": metric("cache_write_input_tokens"),
            "output_tokens": metric("output_tokens"),
            "reasoning_output_tokens": metric("reasoning_output_tokens"),
            "tool_call_count": None,
            "exit_reason": exit_reason if exit_reason is not None else getattr(result, "exit_reason", None),
            "tree_before": tree_before or getattr(result, "tree_before", None),
            "tree_after": getattr(result, "tree_after", None) if result is not None else None,
            "external_session_id": getattr(result, "external_session_id", None) if result is not None else None,
        }

    def _trace_finished_model_session(
        self,
        *,
        profile: ModelProfile,
        selected: Any | None,
        role: ExecutionRole,
        prompt_bytes: int | None,
        started_at: str,
        started_mono: float,
        usage: Mapping[str, Any] | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        exit_reason: str | None = None,
        final_message: str | None = None,
    ) -> dict[str, Any]:
        session = self._trace_session(
            profile=profile,
            selected=selected,
            role=role,
            prompt_bytes=prompt_bytes,
            started_at=started_at,
            started_mono=started_mono,
            tree_before=tree_before,
            result=None,
            exit_reason=exit_reason,
            final_message=final_message,
        )
        session.update(
            finished_at=self._trace_time(),
            wall_time_ms=round((time.perf_counter() - started_mono) * 1000),
            tree_after=tree_after,
        )
        aliases = {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
            "cache_write_input_tokens": (
                "cache_write_input_tokens", "cache_creation_input_tokens",
            ),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "reasoning_output_tokens": ("reasoning_output_tokens",),
        }
        nested = {
            "cached_input_tokens": (
                ("prompt_tokens_details", "cached_tokens"),
                ("input_tokens_details", "cached_tokens"),
            ),
            "cache_write_input_tokens": (
                ("prompt_tokens_details", "cache_write_tokens"),
                ("input_tokens_details", "cache_write_tokens"),
            ),
            "reasoning_output_tokens": (
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_tokens_details", "reasoning_tokens"),
            ),
        }
        for name in aliases:
            value = None
            if isinstance(usage, Mapping):
                for alias in aliases[name]:
                    value = usage.get(alias)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        break
                    value = None
                if value is None:
                    for container, key in nested.get(name, ()):
                        details = usage.get(container)
                        if isinstance(details, Mapping):
                            value = details.get(key)
                            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                                break
                            value = None
            session[name] = value
        return session

    @staticmethod
    def _trace_diff_reference(path: Path) -> dict[str, Any]:
        try:
            payload = path.read_bytes()
        except OSError:
            return {"diff_artifact": str(path), "diff_sha256": None}
        return {
            "diff_artifact": str(path),
            "diff_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _reviewer_for_profile(self, profile_id: str) -> Reviewer:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.REVIEWER)
        client = self._reviewer_client
        if client is None:
            client = _chat_client(
                build_llm_endpoint(profile), self._runtime_environment
            )
        return Reviewer(client, allow_format_repair=False)

    def _recommender_for_profile(self, profile_id: str) -> ExecutionRecommender:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.PLANNER)
        client = self._recommender_client
        if client is None:
            # This is deliberately a new client: the recommender has no
            # planner conversation/history, while using the same profile
            # endpoint and transport policy.
            client = _chat_client(
                build_llm_endpoint(profile), self._runtime_environment
            )
        return ExecutionRecommender(client)

    def _maybe_recommend_profiles(
        self,
        store: RunStateStore,
        run_dir: Path,
        planner_profile_id: str,
    ) -> None:
        if not self.config.ui.enable_profile_recommendation:
            return
        profiles = profiles_for_config(self.config)
        implementers = tuple(
            profile for profile in profiles.values() if ExecutionRole.IMPLEMENTER in profile.roles
        )
        reviewers = tuple(
            profile for profile in profiles.values() if ExecutionRole.REVIEWER in profile.roles
        )
        if len(implementers) <= 1 and len(reviewers) <= 1:
            return
        try:
            contract = (run_dir / "implementation_contract.md").read_text(encoding="utf-8")
            recommendation = self._recommender_for_profile(planner_profile_id).recommend(
                contract,
                implementers,
                reviewers,
                artifacts_dir=run_dir,
            )
        except (LLMError, RecommendationError, OSError, UnicodeError) as exc:
            message = redact(" ".join(str(exc).split()), self._secrets)
            warning = f"{type(exc).__name__}: {message}"[:1000]
            try:
                write_recommendation_error(run_dir, warning)
            except (OSError, UnicodeError):
                pass
            store.update(
                status=RunStatus.PLANNING,
                recommendation={"status": "FAILED", "warning": warning},
            )
            return
        store.update(
            status=RunStatus.PLANNING,
            recommendation={
                "status": "READY",
                "implementer_profile": recommendation.implementer_profile,
                "reviewer_profile": recommendation.reviewer_profile,
                "rationale": recommendation.rationale,
            },
        )

    def _executor_for_profile(
        self,
        profile_id: str,
        role: ExecutionRole,
        *,
        forbidden_env_names: tuple[str | None, ...] = (),
    ) -> Any:
        """Resolve one generic executor through the infrastructure boundary."""

        profile = profile_for_role(self.config, profile_id, role)
        runtime = ExecutorRuntimeConfig(
            config=self.config,
            environment=self._runtime_environment,
            codex_home=self.config.codex_runtime.home,
            claude_home=self.config.claude_runtime.home,
            forbidden_env_names=forbidden_env_names,
        )
        return executor_for_profile(profile, runtime)

    def _run_revision(
        self,
        *,
        worktree: Path,
        prompt: str,
        artifact_dir: Path,
        profile_id: str,
        role: ExecutionRole,
        mutable_paths: tuple[str, ...] = (),
    ) -> Any:
        """Run one semantic-revision or check-repair worker pass."""

        profile = profile_for_role(self.config, profile_id, role)
        executor = self._executor_for_profile(profile.id, role)
        return executor.run(
            AgentRunRequest(
                role=role,
                profile_id=profile.id,
                prompt=redact(prompt, self._secrets),
                worktree=Path(worktree),
                artifact_dir=Path(artifact_dir),
                mutable_paths=mutable_paths,
                prompt_mode="revision",
            )
        )

    @staticmethod
    def _cycle_update(store: RunStateStore, cycle: RunCycle | int, **fields: Any) -> None:
        """Merge one cycle record without replacing the other cycle records."""

        number = cycle.number if isinstance(cycle, RunCycle) else cycle
        state = store.load()
        cycles = list(state.get("cycles") or [])
        index = next(
            (i for i, item in enumerate(cycles)
             if isinstance(item, dict) and item.get("number") == number),
            None,
        )
        record: dict[str, Any] = {"number": number}
        if index is not None and isinstance(cycles[index], dict):
            record.update(cycles[index])
        if isinstance(cycle, RunCycle):
            record["kind"] = cycle.kind.value
        record.update(fields)
        if index is None:
            cycles.append(record)
        else:
            cycles[index] = record
        store.update(status=state.get("status", RunStatus.PLANNING), cycles=cycles)

    def _update_v2_usage(self, store: RunStateStore, run_dir: Path) -> None:
        """Publish the per-cycle token totals.

        Always derived from persisted artifacts
        (``cycles/NNN/implementation/steps/Sxx/step.json`` and every worker
        report), never from ``state.steps``.
        """

        store.update(
            status=store.load().get("status", RunStatus.PLANNING),
            usage=phase_usage_summary(run_dir),
        )

    @staticmethod
    def _ensure_step_artifacts(step_dir: Path, result: Any) -> None:
        """Make the durable step envelope complete for test doubles too."""

        if not (step_dir / "agent.final.md").exists():
            atomic_write_text(step_dir / "agent.final.md", str(getattr(result, "final_message", "")))
        if not (step_dir / "agent.stderr.log").exists():
            atomic_write_text(step_dir / "agent.stderr.log", str(getattr(result, "stderr_tail", "")))
        if not (step_dir / "agent.events.jsonl").exists():
            atomic_write_text(step_dir / "agent.events.jsonl", "")
        if not (step_dir / "agent.result.json").exists():
            payload = _safe_agent_result_payload(result)
            atomic_write_text(step_dir / "agent.result.json", _json_text(payload))

    @staticmethod
    def _ensure_revision_artifacts(artifact_dir: Path, result: Any) -> None:
        """Complete the revision artifact set without replacing provider files."""

        if not (artifact_dir / "agent.final.md").exists():
            atomic_write_text(artifact_dir / "agent.final.md", str(getattr(result, "final_message", "")))
        if not (artifact_dir / "agent.stderr.log").exists():
            atomic_write_text(artifact_dir / "agent.stderr.log", str(getattr(result, "stderr_tail", "")))
        if not (artifact_dir / "agent.events.jsonl").exists():
            atomic_write_text(artifact_dir / "agent.events.jsonl", "")
        if not (artifact_dir / "agent.result.json").exists():
            payload = _safe_agent_result_payload(result)
            atomic_write_text(artifact_dir / "agent.result.json", _json_text(payload))

    def run(
        self, spec: str | Path, *, run_id: str | None = None,
        planner_profile: str | None = None,
        run_options: RunOptions | None = None,
    ) -> RunResult:
        """Read one SPEC file and delegate execution to :meth:`run_text`."""

        spec_path = Path(spec).expanduser().resolve()
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise OrchestrationError(f"could not read spec {spec_path}: {exc}") from exc
        kwargs: dict[str, Any] = {"run_id": run_id}
        if planner_profile is not None:
            kwargs["planner_profile"] = planner_profile
        if run_options is not None:
            kwargs["run_options"] = run_options
        return self.run_text(spec_content, **kwargs)

    def run_text(
        self,
        spec_content: str,
        *,
        run_id: str | None = None,
        planner_profile: str | None = None,
        run_options: RunOptions | None = None,
        on_created: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Run one in-memory SPEC and return its durable final state.

        A worktree is deliberately never removed.  This keeps failed and
        interrupted runs inspectable and makes the run directory the handoff
        point for operators.
        """

        if not isinstance(spec_content, str):
            raise OrchestrationError("spec must be a string")
        if not spec_content.strip():
            raise OrchestrationError("spec must not be empty")

        original_config = self.config
        try:
            if run_options is None:
                overrides = {"planner_profile": planner_profile} if planner_profile is not None else {}
                run_options = RunOptions.from_config(original_config, **overrides)
            elif planner_profile is not None and planner_profile != run_options.planner_profile:
                raise OrchestrationError("planner profile conflicts with run options")
            run_options.validate_profiles(original_config)
            selected_planner = profile_for_role(
                original_config, run_options.planner_profile, ExecutionRole.PLANNER
            )
        except RunOptionsError as exc:
            raise OrchestrationError(str(exc)) from exc

        selected_run_id = _safe_run_id(run_id) if run_id is not None else generate_run_id()
        run_dir = (self.config.runs_root / selected_run_id).expanduser().resolve()
        if run_dir.exists():
            raise OrchestrationError(f"run directory already exists: {run_dir}")
        self._secrets = config_secret_values(
            self.config, self._runtime_environment
        )

        store: RunStateStore | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            # The SPEC copy is created before state initialization, as the
            # CREATED phase contract requires both to exist together.
            (run_dir / "spec.md").write_text(spec_content, encoding="utf-8")
            options_sha256 = write_run_options(run_dir, run_options)
            store = RunStateStore(run_dir / "state.json")
            store.initialize(selected_run_id, pipeline_version=2)
            self._begin_trace(run_dir, selected_run_id, created=True)
            # This is the first durable boundary.  It intentionally carries
            # no Git/plan identity yet: context and repository discovery are
            # themselves resumable operations.
            write_checkpoint(run_dir, ResumeCheckpoint(phase=ResumePhase.CONTEXT))
            # All downstream methods use this frozen per-run view.  The
            # caller's HarnessConfig object is never mutated.
            self._run_options = run_options
            self._effective_repair_scope = effective_repair_scope_policy(run_options)
            self.config = effective_run_config(original_config, run_options)
            store.update(
                status=RunStatus.CREATED,
                spec_path="spec.md",
                repo=str(self.config.repo),
                base_ref=self.config.base_ref,
                run_options_sha256=options_sha256,
                run_options=run_options.to_dict(),
                execution={
                    "planner": {
                        "profile_id": selected_planner.id,
                        "model": selected_planner.model,
                        "selection_mode": selected_planner.selection_mode.value,
                    }
                },
            )
            if on_created is not None:
                on_created(run_dir)
            return self._diagnose_result(
                self._execute(store, run_dir, selected_run_id, spec_content)
            )
        except KeyboardInterrupt:
            if store is None:
                raise
            state = store.update(
                status=RunStatus.INTERRUPTED,
                failure={"reason": "INTERRUPTED"},
                **self._closing_step_fields(store, "interrupted"),
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.INTERRUPTED, state))
        except Exception as exc:
            if store is None:
                raise
            state = store.record_failure(
                _failure_reason(exc),
                redact(str(exc), self._secrets),
                **self._closing_step_fields(store, "failed"),
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.FAILED, state))

    def _diagnose_result(self, result: RunResult) -> RunResult:
        """Best-effort terminal projection; diagnostics never changes a run result."""

        if result.status in {RunStatus.COMMITTED, RunStatus.PUBLISHED}:
            state = result.state
            self._trace_emit(
                "run.completed",
                phase="run",
                cycle=state.get("cycle") if isinstance(state.get("cycle"), int) else None,
                data={
                    "status": result.status.value,
                    "commit_sha": state.get("commit_sha"),
                    "published": result.status is RunStatus.PUBLISHED,
                },
                once=True,
            )
        elif result.status in {
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.BLOCKED,
            RunStatus.PLAN_REJECTED,
        }:
            failure = result.state.get("failure") if isinstance(result.state, Mapping) else None
            self._trace_emit(
                "run.failed",
                phase="run",
                cycle=result.state.get("cycle") if isinstance(result.state.get("cycle"), int) else None,
                data={
                    "status": result.status.value,
                    "reason": failure.get("reason") if isinstance(failure, Mapping) else result.status.value,
                },
                once=True,
            )

        if result.status in {
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.BLOCKED,
            RunStatus.PLAN_REJECTED, RunStatus.COMMITTED, RunStatus.PUBLISHED,
        }:
            try:
                write_run_diagnostics(self.config, result.run_dir)
            except Exception:
                # ``write_run_diagnostics`` records diagnostics.error.txt when
                # possible.  A reporting failure must not alter the pipeline.
                pass
        return result

    @staticmethod
    def _closing_step_fields(store: RunStateStore, terminal: str) -> dict[str, Any]:
        try:
            return _terminal_step_fields(store.load(), None, terminal)
        except (OSError, ValueError):
            return {}

    execute = run

    def _execute(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
    ) -> RunResult:
        repo = git_root(self.config.repo)
        state = store.load()
        planner_profile_id = state.get("execution", {}).get("planner", {}).get("profile_id")
        if not isinstance(planner_profile_id, str):
            raise OrchestrationError("run planner profile is missing")
        if self.config.require_clean_base:
            assert_clean(repo)
        base_sha = resolve_commit(repo, self.config.base_ref)
        base_tree_sha = resolve_tree(repo, base_sha)
        store.update(
            status=RunStatus.CREATED, repo=str(repo), base_sha=base_sha,
            planning_protocol="v2",
        )
        try:
            repository_reference = build_repository_reference(
                repo, base_sha=base_sha, config=self.config.repository
            )
        except GitError:
            repository_reference = RepositoryReference(
                self.config.repository.remote, None, base_sha, None
            )
        atomic_write_text(
            run_dir / "repository_reference.json",
            json.dumps(repository_reference_dict(repository_reference), indent=2) + "\n",
        )
        context_bundle = build_context(repo, base_sha, spec, self.config.context)
        context = render_context(context_bundle)
        atomic_write_text(run_dir / "context.txt", context)
        store.update(
            status=RunStatus.PLANNING,
            context={
                "base_sha": context_bundle.base_sha,
                "locator_used": context_bundle.locator_used,
                "locator_warning": context_bundle.locator_warning,
                "omitted": list(context_bundle.omitted),
                "total_bytes": context_bundle.total_bytes,
            },
        )
        self._write_checkpoint(
            run_dir, ResumePhase.PLANNER, head=base_sha, tree=base_tree_sha
        )
        return self._execute_v2(
            store, run_dir, run_id, spec, repo, base_sha, context,
            repository_reference,
        )

    def _prepare_v2_run(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        context: str,
        repository_reference: RepositoryReference,
        planner_profile: ModelProfile,
        existing_plan: TaskPlanV2 | None = None,
        existing_info: WorktreeInfo | None = None,
    ) -> "_V2Setup | RunResult":
        """Plan, obtain the approved selection, create the worktree and set up.

        Returns a terminal :class:`RunResult` for BLOCKED/REJECTED plans.  None
        of this is ever replayed by a resume.
        """

        revision_enabled = self._run_options.semantic_revision_enabled
        repair_enabled = self._run_options.max_review_repair_cycles > 0
        check_repair_enabled = self._run_options.max_check_repair_attempts > 0
        planner_profile_id = planner_profile.id
        if existing_plan is None:
            planner = PlannerV2(
                self._planner_client or _chat_client(build_llm_endpoint(planner_profile), self._runtime_environment),
                repository_reference=repository_reference,
                planning=self.config.planning,
                check_catalog=self.config.check_catalog,
                default_check_ids=self.config.default_check_ids,
                prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
                repository_preconditions=RepositoryPreconditions(
                    repo, resolve_tree(repo, base_sha),
                ),
                on_event=lambda name, data: self._trace_emit(name, phase="planning", cycle=1, data=data),
            )
            plan_started_at = self._trace_time()
            plan_started_mono = time.perf_counter()
            self._trace_emit(
                "plan.started",
                phase="planning",
                cycle=1,
                data={
                    "session": self._trace_session(
                        profile=planner_profile,
                        selected=None,
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=None,
                        started_at=plan_started_at,
                        started_mono=plan_started_mono,
                        tree_before=resolve_tree(repo, base_sha),
                    )
                },
            )
            try:
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
            except Exception:
                # The failed planner answer is deliberately left in place and
                # will be archived by the next planner attempt.
                self._write_checkpoint(
                    run_dir, ResumePhase.PLANNER, head=base_sha,
                    tree=resolve_tree(repo, base_sha),
                )
                raise
            _persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            self._trace_emit(
                "plan.completed",
                phase="planning",
                cycle=1,
                data={
                    "decision": plan.decision.value,
                    "title": plan.title,
                    "session": self._trace_finished_model_session(
                        profile=planner_profile,
                        selected=None,
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=(
                            (run_dir / "planner.request.txt").stat().st_size
                            if (run_dir / "planner.request.txt").is_file() else None
                        ),
                        started_at=plan_started_at,
                        started_mono=plan_started_mono,
                        usage=(
                            getattr(planner, "last_usage", None)
                            if getattr(planner, "last_usage", None) is not None
                            else read_usage_artifact(run_dir / "planner.usage.json")
                        ),
                        tree_before=resolve_tree(repo, base_sha),
                        tree_after=resolve_tree(repo, base_sha),
                        final_message=getattr(plan, "raw", None),
                    ),
                },
            )
        else:
            # Resume of PLANNER has already produced and durably parsed this
            # plan.  Re-entering setup must never call the planner again.
            plan = existing_plan
        store.update(
            status=RunStatus.PLANNING,
            planning_protocol="v2",
            planner={
                "decision": plan.decision.value,
                "title": plan.title,
                "model": planner_profile.model,
                "profile_id": planner_profile.id,
                "selection_mode": planner_profile.selection_mode.value,
                # operator_recovery: the plan was pasted by the operator and
                # no planner completion produced it.
                "source": plan_source(run_dir),
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
                "required_checks": list(plan.required_checks),
                "steps": [
                    {"id": step.id, "title": step.title,
                     "execution_class": step.execution_class.value,
                     "recommended_profile": self.config.routing.profile_for(step.execution_class),
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": self._run_options.final_reviewer_profile,
            },
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": self.config.routing.profile_for(step.execution_class)}
                for step in plan.steps
            ],
            current_step=None,
        )
        self._cycle_update(
            store, 1, status="running", plan_summary=plan.title,
            steps_summary=[{"id": step.id, "title": step.title} for step in plan.steps],
        )
        if plan.decision is PlanDecision.BLOCKED:
            state = store.update(
                status=RunStatus.BLOCKED,
                failure={"reason": "PLANNER_BLOCKED", "detail": plan.blockers},
            )
            return RunResult(run_dir, RunStatus.BLOCKED, state)

        # Every source of this plan (planner, resumed planner answer, operator
        # recovery) must be possible against the base tree before it can be
        # offered for approval.  Resume re-enters here, so it is checked again.
        validate_plan_repository_topology(repo, resolve_tree(repo, base_sha), plan)
        try:
            # REQUIRED_CHECKS has already been parsed against the trusted
            # catalogue.  Materialize those exact trusted definitions before
            # the plan can become approval authority.
            selected_checks = self.config.select_checks(plan.required_checks)
            # Freeze the whole trusted catalogue, not just this selection: a
            # correction plan may legitimately require another approved check,
            # and it must still run the argv approved at this boundary.
            write_check_authority(
                run_dir, tuple(self.config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            _bundle, _bundle_sha = validate_implementation_bundle(run_dir)
            plan_identity = compute_plan_identity_from_run(run_dir)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update(status=RunStatus.PLANNING, plan_identity=asdict(plan_identity))
        self._write_checkpoint(
            run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha,
            tree=resolve_tree(repo, base_sha), plan_identity=plan_identity,
        )

        if self.config.approval.require_plan_approval:
            store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
            approval = wait_for_plan_approval(
                run_dir, identity=plan_identity,
                poll_interval_seconds=self.config.approval.poll_interval_seconds,
            )
            if approval.decision is ApprovalDecision.REJECT:
                state = store.update(status=RunStatus.PLAN_REJECTED)
                return RunResult(run_dir, RunStatus.PLAN_REJECTED, state)
            try:
                selection, execution_sha = read_execution_selection_with_sha256(run_dir)
                validate_execution_selection(self.config, selection)
                durable_identity = compute_plan_identity_from_run(run_dir)
                if durable_identity.execution_sha256 != execution_sha:
                    raise ApprovalError("execution selection hash mismatch")
                read = compute_plan_identity_from_run(run_dir)
                bound = read_plan_approval(run_dir, expected_identity=read)
                if bound is None or bound.bundle_sha256 != durable_identity.bundle_sha256:
                    raise ApprovalError("v2 approval is not bound to the exact bundle")
            except (ExecutionSelectionError, ApprovalError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        else:
            requested = resolve_execution_selection(
                self.config,
                planner_profile_id=planner_profile_id,
                plan_steps=plan.steps,
                semantic_reviser_profile_id=(
                    self._run_options.semantic_reviser_profile
                    if revision_enabled or repair_enabled else None
                ),
                check_repair_profile_id=(
                    self._run_options.check_repair_profile
                    if check_repair_enabled else None
                ),
                final_reviewer_profile_id=self._run_options.final_reviewer_profile,
                fallback_authority=self._run_options.recovery.execution_fallbacks,
            )
            selection = ensure_execution_selection(run_dir, requested)
            durable_identity = compute_plan_identity_from_run(run_dir)

        # Re-read the complete manifest after the approval transaction.  The
        # first validation protects the approval surface; this one closes the
        # race between approval and worktree creation.  The bundle bytes must
        # still be the ones hashed at planning time and bound by the approval.
        try:
            bundle, bundle_sha = validate_implementation_bundle(
                run_dir, expected_step_ids=[step.id for step in plan.steps]
            )
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        if bundle_sha != plan_identity.bundle_sha256:
            raise ApprovalError("PLAN_APPROVAL_INVALID: implementation bundle changed after planning")

        if selection.planner.profile_id != planner_profile_id:
            raise ExecutionSelectionError("execution selection planner is not the run planner")
        if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
            raise ExecutionSelectionError("execution selection steps do not match the plan")
        if [item.execution_class for item in selection.steps] != [step.execution_class for step in plan.steps]:
            raise ExecutionSelectionError("execution selection classes do not match the plan")
        if not isinstance(selection, ExecutionSelection) or selection.schema_version != EXECUTION_SELECTION_SCHEMA:
            raise ExecutionSelectionError("v2 execution requires the generic execution selection")
        validate_execution_selection(self.config, selection)
        self._last_selection = selection
        execution_state: dict[str, Any] = {
            "planner": asdict(selection.planner),
            "steps": [
                {"step_id": item.step_id, "implementer": asdict(item.implementer)}
                for item in selection.steps
            ],
        }
        if selection.semantic_reviser is not None:
            execution_state["semantic_reviser"] = asdict(selection.semantic_reviser)
        if selection.check_repair is not None:
            execution_state["check_repair"] = asdict(selection.check_repair)
        execution_state["final_reviewer"] = asdict(selection.final_reviewer)
        store.update(status=RunStatus.PLANNING, execution=execution_state,
                     plan_identity=asdict(durable_identity))
        self._trace_emit(
            "plan.approved",
            phase="planning",
            cycle=1,
            data={
                "plan_identity": asdict(durable_identity),
                "execution_selection_sha256": durable_identity.execution_sha256,
            },
            once=True,
        )
        base_tree_sha = resolve_tree(repo, base_sha)
        self._write_checkpoint(
            run_dir, ResumePhase.WORKTREE_SETUP, head=base_sha,
            tree=base_tree_sha, plan_identity=durable_identity,
            execution_selection_sha256=durable_identity.execution_sha256,
        )
        # Plan approval complete: the next operation is the first step.
        checkpoint = self._initial_checkpoint(plan, base_sha, base_tree_sha, durable_identity)
        if checkpoint is not None:
            write_checkpoint(run_dir, checkpoint)

        branch = build_run_branch(plan.title, run_id)
        if existing_info is None:
            info = create_run_worktree(
                repo, base_ref=base_sha, branch=branch,
                worktree_path=self.config.worktrees_root / run_id,
                require_clean_base=self.config.require_clean_base,
            )
        else:
            info = existing_info
        store.update(status=RunStatus.WORKTREE_READY, branch=info.branch,
                     worktree=str(info.worktree), base_sha=info.base_sha)
        self._ensure_github_issue_metadata(
            store=store, run_id=run_id, plan_title=plan.title, info=info,
        )
        self._trace_emit(
            "worktree.created",
            phase="setup",
            cycle=1,
            data={
                "branch": info.branch,
                "worktree": str(info.worktree),
                "base_sha": info.base_sha,
                "tree_sha": base_tree_sha,
                "resumed_setup": existing_info is not None,
            },
            once=True,
        )
        ownership_before = _git_ownership(repo, info.worktree)
        # Persisted with an explicit status so a later resume has the stronger
        # durable ownership proof of the pre-execution boundary.
        store.update(
            status=RunStatus.PREPARING,
            git_ownership=_git_ownership_payload(ownership_before),
        )
        try:
            setup_results = prepare_workspace(
                info.worktree, self.config.workspace_setup,
                environment=self._runtime_environment, artifacts_dir=run_dir,
                secrets=self._secrets,
            )
        except WorkspaceSetupError as exc:
            if exc.results:
                store.update(status=RunStatus.PREPARING,
                             workspace_setup=[asdict(result) for result in exc.results])
            raise
        store.update(status=RunStatus.PREPARING,
                     workspace_setup=[asdict(result) for result in setup_results])
        # The candidate starts as the base tree and all later gates use the
        # actual index identity, never an inferred file list.
        stage_all(info.worktree)
        candidate_tree = index_tree_sha(info.worktree)
        if candidate_tree != base_tree_sha:
            raise OrchestrationError("initial candidate tree does not match base")
        # The workspace preflight is the first live use of the authority; the
        # expected hash is the one this run's identity already durably bound.
        check_config, check_ids = config_with_check_authority(
            self.config, run_dir, expected_sha256=durable_identity.checks_sha256,
        )
        preflight_failures = self._run_check_preflights_recoverably(
            store=store, worktree=info.worktree, check_config=check_config,
            check_ids=check_ids or plan.required_checks,
            counter_key="check-preflight:workspace", phase="preparing",
        )
        if preflight_failures:
            raise OrchestrationError(preflight_failures[0])
        # Worktree and setup complete: still the first step.
        if checkpoint is not None:
            write_checkpoint(run_dir, checkpoint)
        return _V2Setup(plan, bundle, selection, info, ownership_before, base_tree_sha, checkpoint)

    @staticmethod
    def _initial_checkpoint(
        plan: TaskPlanV2, base_sha: str, base_tree_sha: str, identity: PlanIdentity,
    ) -> ResumeCheckpoint | None:
        if identity.execution_sha256 is None or not plan.steps:
            return None
        return ResumeCheckpoint(
            phase=ResumePhase.IMPLEMENT_STEP, review_cycle=1,
            step_id=plan.steps[0].id, expected_head_sha=base_sha,
            expected_tree_sha=base_tree_sha,
            execution_selection_sha256=identity.execution_sha256,
            plan_identity=identity,
        )

    @staticmethod
    def _approved_check_authority_sha256(run_dir: Path) -> str | None:
        """The check authority hash the run's durable boundary already binds.

        This is deliberately *not* a hash of the file being read: the expected
        value comes from the checkpoint written before the approval, so a
        rewritten ``check_authority.json`` is rejected on every live use, not
        only on resume.  A run created before the artifact existed has no hash
        and keeps its current behavior.
        """

        directory = Path(run_dir).expanduser().resolve()
        while not (directory / "check_authority.json").is_file():
            parent = directory.parent
            if parent == directory:
                return None
            directory = parent
        try:
            record = read_checkpoint_record(directory)
        except ResumeCheckpointError as exc:
            raise ValidationError(f"the run checkpoint is unreadable: {exc}") from exc
        identity = record[0].plan_identity if record is not None else None
        approved = identity.checks_sha256 if identity is not None else None
        if approved is None:
            raise ValidationError(
                "the run has a check authority but no durable approved hash"
            )
        return approved

    @staticmethod
    def _write_checkpoint(
        run_dir: Path,
        phase: ResumePhase,
        *,
        head: str | None,
        tree: str | None,
        cycle: int | None = None,
        stage: GateStage | None = None,
        step_id: str | None = None,
        check_repair_attempt: int | None = None,
        correction_bundle_sha256: str | None = None,
        expected_parent_sha: str | None = None,
        next_step_id: str | None = None,
        plan_identity: PlanIdentity | None = None,
        execution_selection_sha256: str | None = None,
    ) -> None:
        """Persist the next operation that has not succeeded yet.

        Only the run identity (plan identity and execution selection) is
        carried over from the current checkpoint; every other field describes
        exactly the new boundary.  A run without a pending checkpoint has
        nothing to resume and is left untouched.
        """

        record = read_checkpoint_record(run_dir)
        if record is None or record[1] != "pending":
            return
        previous = record[0]
        write_checkpoint(run_dir, ResumeCheckpoint(
            phase=phase,
            review_cycle=cycle or previous.review_cycle,
            stage=stage,
            step_id=step_id,
            next_step_id=next_step_id,
            check_repair_attempt=check_repair_attempt,
            expected_head_sha=head,
            expected_parent_sha=expected_parent_sha,
            expected_tree_sha=tree,
            execution_selection_sha256=(
                execution_selection_sha256 or previous.execution_selection_sha256
            ),
            plan_identity=plan_identity or previous.plan_identity,
            correction_bundle_sha256=correction_bundle_sha256,
        ))

    def _execute_v2(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        context: str,
        repository_reference: RepositoryReference,
        *,
        prepared: "_V2Setup | RunResult | None" = None,
    ) -> RunResult:
        """Prepare a new run, then delegate its execution to the coordinator."""

        if prepared is None:
            planner_profile = profile_for_role(
                self.config, store.load()["execution"]["planner"]["profile_id"],
                ExecutionRole.PLANNER,
            )
            prepared = self._prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                repository_reference, planner_profile,
            )
        if isinstance(prepared, RunResult):
            return prepared
        if prepared.checkpoint is None:
            # A v2 run always has an execution selection bound to its plan
            # identity once approved; never run without one.
            raise OrchestrationError("v2 run has no execution selection identity")
        pipeline = PipelineV2Context(
            run_dir=run_dir, run_id=run_id, spec=spec, context=context, repo=repo,
            base_sha=base_sha, base_tree_sha=prepared.base_tree_sha,
            repository_reference=repository_reference, info=prepared.info,
            plan=prepared.plan, bundle=prepared.bundle, selection=prepared.selection,
            options=self._run_options,
        )
        return self._run_pipeline(store, pipeline, prepared.checkpoint, resumed=False)

    def _run_pipeline(
        self, store: RunStateStore, pipeline: PipelineV2Context,
        start: ResumeCheckpoint, *, resumed: bool,
    ) -> RunResult:
        """Run the generic coordinator and classify its terminal failure."""

        self._last_selection = pipeline.selection
        coordinator = PipelineV2Coordinator(pipeline, self._pipeline_operations(store))
        try:
            return coordinator.run(start, resumed=resumed)
        except PipelineFailure as failure:
            if failure.reason in {
                AGENT_AUTH_FAILURE, "LLM_401", "LLM_403", "LLM_AUTH_FAILURE",
                "MISSING_PROVIDER_CREDENTIALS", "PROVIDER_CREDENTIALS_MISSING",
            }:
                failure.reason = "EXTERNAL_AUTH_REQUIRED"
                failure.detail = "external executor authorization is required"
            exhausted = failure.reason in {
                "CHECK_REPAIR_EXHAUSTED", "CHECK_INFRA_RETRIES_EXHAUSTED",
                "TRANSIENT_ATTEMPTS_EXHAUSTED", "REVIEW_REPAIR_EXHAUSTED",
            }
            if (
                failure.reason == "DETERMINISTIC_GATE_FAILED"
                and pipeline.options.max_check_repair_attempts == 0
                and isinstance(failure.detail, str)
                and "CHECK_FAILED:" in failure.detail
            ):
                exhausted = True
            if failure.reason == "CHECK_TIMEOUT":
                stage_value = getattr(start, "stage", None)
                try:
                    stage = GateStage(stage_value)
                    cycle_number = store.load().get("cycle", 1)
                    if isinstance(cycle_number, bool) or not isinstance(cycle_number, int):
                        raise ValueError("cycle is invalid")
                    retry_key = f"check-infra:{cycle_number:03d}:{stage.value}"
                    exhausted = self._recovery_counter(store, retry_key) >= (
                        pipeline.options.recovery.max_check_infra_retries
                    )
                except (TypeError, ValueError):
                    exhausted = False
            remote_required = failure.reason in {"PUSH_FAILED", "CANDIDATE_PUSH_FAILED"}
            decision = classify_failure(
                failure.reason, budget_exhausted=exhausted,
                remote_required=remote_required, remote_unavailable=remote_required,
            )
            if failure.reason != "CHECK_TIMEOUT" and decision.disposition in {
                RecoveryDisposition.HARD_STOP, RecoveryDisposition.WAIT_EXTERNAL,
            }:
                state = store.load()
                tree = state.get("staged_tree_sha")
                if not isinstance(tree, str):
                    tree = _safe_candidate_tree(pipeline.info.worktree)
                repair_state = state.get("check_repair")
                used = repair_state.get("attempt_count", 0) if isinstance(repair_state, dict) else 0
                attempt = used + 1 if isinstance(used, int) and not isinstance(used, bool) else 1
                phase = state.get("status")
                self._trace_recovery(
                    "recovery.classified", reason=failure.reason,
                    decision=decision, attempt=attempt,
                    tree_before=tree, tree_after=_safe_candidate_tree(pipeline.info.worktree),
                    budget_remaining=(
                        0 if exhausted else pipeline.options.recovery.max_transient_attempts
                    ),
                    phase=phase if isinstance(phase, str) else "run",
                    cycle=state.get("cycle") if isinstance(state.get("cycle"), int) else None,
                    step_id=failure.step_id,
                )
                if exhausted:
                    self._trace_recovery(
                        "recovery.exhausted", reason=failure.reason,
                        decision=decision, attempt=attempt,
                        tree_before=tree, tree_after=_safe_candidate_tree(pipeline.info.worktree),
                        budget_remaining=0,
                        phase=phase if isinstance(phase, str) else "run",
                        cycle=state.get("cycle") if isinstance(state.get("cycle"), int) else None,
                        step_id=failure.step_id,
                    )
            return self._v2_failed(
                store, pipeline.run_dir, failure.reason, failure.step_id, failure.detail,
            )
        except StepExecutionFailure as failure:
            return self._step_failed(store, pipeline.run_dir, failure)
        except ScopeApprovalRequired:
            return RunResult(pipeline.run_dir, RunStatus.WAITING_SCOPE_APPROVAL, store.load())
        except (ResumeIntegrityError, ResumeRequiresOperatorError):
            raise
        except Exception as exc:
            return self._v2_failed(
                store, pipeline.run_dir, _failure_reason(exc), None,
                redact(str(exc), self._secrets),
            )

    def _pipeline_operations(self, store: RunStateStore) -> PipelineV2Operations:
        """Bind every operation the coordinator sequences to this run."""

        bind = functools.partial
        candidate_lifecycle = CandidateLifecycle(
            staging_remote=self.config.repository.remote,
            authorize_tree=self._authorize_candidate_tree,
            gate_mutable_authority=lambda ctx, cycle_plan, stage: gate_mutable_authority(
                ctx.run_dir, cycle_plan.cycle.number, stage,
                base_paths=self._effective_cycle_scope(ctx, cycle_plan),
                policy_config=self._effective_repair_scope,
                require_attempt_records=True,
            ),
            push_tree=self._push_candidate,
            cycle_update=self._cycle_update,
        )
        gate_acceptance = GateAcceptanceService(
            secrets=self._secrets,
            repair_scope_policy=self._effective_repair_scope,
            authorize_candidate_tree=self._authorize_candidate_tree,
            check_repair_attempts=self._check_repair_attempt_records,
            load_revision=_load_revision,
            trace_emit=self._trace_emit,
            bounded_detail=_bounded_parse_detail,
        )
        cycle_artifacts = CycleArtifactService(
            cycle_update=self._cycle_update,
            trace_emit=self._trace_emit,
            set_trace_cycle=lambda number: setattr(self, "_trace_cycle", number),
        )
        return PipelineV2Operations(
            checkpoint=lambda ctx, phase, **fields: self._write_checkpoint(
                ctx.run_dir, phase, **fields
            ),
            current_head=lambda ctx: current_head(ctx.info.worktree),
            candidate_tree=lambda ctx: candidate_tree_sha(ctx.info.worktree),
            begin_cycle=bind(cycle_artifacts.begin, store),
            load_cycle=lambda ctx, number: read_cycle_record(ctx.run_dir, number),
            initial_plan=self._initial_cycle_plan,
            review_implementation_correction=self._review_implementation_correction,
            plan_correction=bind(self._plan_correction, store),
            load_correction=self._load_correction,
            completed_steps=self._completed_steps,
            execute_step=bind(self._execute_cycle_step, store),
            unresolved_mismatches=lambda ctx, plan: _has_deferred_contract_mismatches(
                self._completed_steps(ctx, plan)
            ),
            semantic_revision=bind(self._semantic_revision, store),
            semantic_review_correction=bind(self._semantic_review_correction, store),
            run_gate=bind(self._run_gate, store),
            load_gate_evidence=lambda ctx, number, stage: _load_evidence(
                gate_dir(ctx.run_dir, number, stage)
            ),
            accept_gate_state=lambda ctx, cycle_plan, stage, evidence: gate_acceptance.accept(
                store, ctx, cycle_plan, stage, evidence,
                base_paths=self._effective_cycle_scope(ctx, cycle_plan),
            ),
            check_repair_attempts=lambda ctx, number, stage: self._check_repair_attempt_records(
                ctx.run_dir, number, stage
            ),
            check_repair_attempt=bind(self._run_check_repair_attempt, store),
            hard_failures=_hard_integrity_failures,
            soft_failures=_soft_check_failures,
            create_candidate=bind(candidate_lifecycle.create, store),
            load_candidate=lambda ctx, number: read_candidate_record(ctx.run_dir, number),
            push_candidate=bind(candidate_lifecycle.push, store),
            review_candidate=bind(self._review_candidate, store),
            record_review=bind(self._record_review, store),
            request_human=bind(self._request_human, store),
            review_repair_exhausted=bind(self._review_repair_exhausted, store),
            publish=bind(self._publish_candidate, store),
        )

    # -- cycle operations -----------------------------------------------------

    def _initial_cycle_plan(self, ctx: PipelineV2Context) -> CyclePlan:
        return CyclePlan(
            cycle=RunCycle(1, CycleKind.INITIAL),
            plan=ctx.plan,
            bundle=ctx.bundle,
            contracts_dir=ctx.run_dir,
            step_profile_ids={
                item.step_id: item.implementer.profile_id for item in ctx.selection.steps
            },
            step_fallback_profile_ids={
                item.step_id: tuple(profile.profile_id for profile in item.fallbacks)
                for item in ctx.selection.steps
            },
        )

    def _cycle_plan(self, ctx: PipelineV2Context, number: int) -> CyclePlan:
        """The approved plan any durable cycle executed."""

        if number == 1:
            return self._initial_cycle_plan(ctx)
        cycle = read_cycle_record(ctx.run_dir, number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            previous = self._cycle_plan(ctx, number - 1)
            return CyclePlan(
                cycle=cycle,
                plan=previous.plan,
                bundle=previous.bundle,
                contracts_dir=previous.contracts_dir,
                step_profile_ids=previous.step_profile_ids,
                step_fallback_profile_ids=previous.step_fallback_profile_ids,
            )
        plan, bundle, bundle_sha = load_correction_plan(
            self.config, ctx.selection, ctx.run_dir, number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        return self._correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )

    def _correction_cycle_plan(
        self, ctx: PipelineV2Context, cycle: RunCycle, plan: TaskPlanV2,
        bundle: Mapping[str, Any], bundle_sha: str, *, creating: bool,
    ) -> CyclePlan:
        if cycle.kind is not CycleKind.REVIEW_REPLAN:
            raise PipelineFailure("REPLAN_CYCLE_REQUIRED")
        planned_profile_ids = {
            step.id: self.config.routing.profile_for(step.execution_class) for step in plan.steps
        }
        try:
            if creating:
                cycle_selection = ensure_cycle_execution_selection(
                    ctx.run_dir,
                    resolve_cycle_execution_selection(
                        self.config, cycle=cycle.number,
                        plan_steps=plan.steps,
                        step_profile_ids=planned_profile_ids,
                        fallback_authority=self._run_options.recovery.execution_fallbacks,
                    ),
                )
            else:
                cycle_selection = read_cycle_execution_selection(ctx.run_dir, cycle.number)
                validate_cycle_execution_selection(self.config, cycle_selection)
                if [item.step_id for item in cycle_selection.steps] != [step.id for step in plan.steps]:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
                if [item.execution_class for item in cycle_selection.steps] != [step.execution_class for step in plan.steps]:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
                if {
                    item.step_id: item.implementer.profile_id for item in cycle_selection.steps
                } != planned_profile_ids:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
        except (ExecutionSelectionError, OSError) as exc:
            raise PipelineFailure("REPLAN_EXECUTION_SELECTION_INVALID", str(exc)) from exc
        step_profile_ids = {
            item.step_id: item.implementer.profile_id for item in cycle_selection.steps
        }
        step_fallback_profile_ids = {
            item.step_id: tuple(profile.profile_id for profile in item.fallbacks)
            for item in cycle_selection.steps
        }
        return CyclePlan(
            cycle=cycle, plan=plan, bundle=bundle,
            contracts_dir=correction_dir(ctx.run_dir, cycle),
            step_profile_ids=step_profile_ids,
            correction_bundle_sha256=bundle_sha,
            step_fallback_profile_ids=step_fallback_profile_ids,
        )

    def _review_implementation_correction(
        self, ctx: PipelineV2Context, cycle: RunCycle, starting: bool,
    ) -> tuple[CyclePlan, ReviewResult]:
        """Load the previous candidate and reviewer report for direct correction."""

        if cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION:
            raise PipelineFailure("IMPLEMENTATION_CORRECTION_CYCLE_REQUIRED")
        previous = cycle.number - 1
        candidate = read_candidate_record(ctx.run_dir, previous)
        evidence = candidate_evidence(ctx.run_dir, previous)
        review = _accepted_review(
            review_dir(ctx.run_dir, previous), evidence, candidate["commit_sha"]
        ) if evidence is not None else None
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route is not ReviewRoute.IMPLEMENTATION:
            raise ResumeIntegrityError(
                f"cycle {previous:03d} review did not route direct implementation correction"
            )
        # The correction starts exactly on the reviewed candidate.  Once its
        # green tree is accepted, HEAD is the single accepted child of it.
        head = current_head(ctx.info.worktree)
        if head != candidate["commit_sha"] and (
            starting or commit_parents(ctx.info.worktree, head) != (candidate["commit_sha"],)
        ):
            raise ResumeIntegrityError(
                f"cycle {cycle.number:03d} does not start from the reviewed candidate"
            )
        previous_plan = self._cycle_plan(ctx, previous)
        return CyclePlan(
            cycle=cycle,
            plan=previous_plan.plan,
            bundle=previous_plan.bundle,
            contracts_dir=previous_plan.contracts_dir,
            step_profile_ids=previous_plan.step_profile_ids,
            step_fallback_profile_ids=previous_plan.step_fallback_profile_ids,
        ), review

    def _approved_scope_before(self, ctx: PipelineV2Context, number: int) -> list[str]:
        """Every mutable path the plans of cycles ``1..number-1`` approved."""

        scope: set[str] = set()
        for earlier in range(1, number):
            earlier_plan = self._cycle_plan(ctx, earlier)
            scope |= set(self._effective_cycle_scope(ctx, earlier_plan))
        return sorted(scope)

    def _plan_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
    ) -> CyclePlan:
        """Plan one review-driven correction cycle from the reviewed candidate."""

        previous = cycle.number - 1
        repair_dir = correction_dir(ctx.run_dir, cycle)
        repair_dir.mkdir(parents=True, exist_ok=True)
        if cycle.kind is not CycleKind.REVIEW_REPLAN:
            raise PipelineFailure("REPLAN_CYCLE_REQUIRED")
        candidate = read_candidate_record(ctx.run_dir, previous)
        evidence = candidate_evidence(ctx.run_dir, previous)
        if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
            raise ResumeIntegrityError(f"cycle {previous:03d} candidate evidence is missing")
        review = _accepted_review(review_dir(ctx.run_dir, previous), evidence, candidate["commit_sha"])
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route is not ReviewRoute.REPLAN:
            raise ResumeIntegrityError(
                f"cycle {previous:03d} review did not route replan correction"
            )
        head = current_head(ctx.info.worktree)
        if head != candidate["commit_sha"]:
            raise ResumeIntegrityError(
                f"cycle {cycle.number:03d} does not start from the reviewed candidate"
            )
        tree_before = candidate_tree_sha(ctx.info.worktree)
        planner_profile = profile_for_role(
            self.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        approved_scope = self._approved_scope_before(ctx, cycle.number)
        store.update(status=RunStatus.PLANNING, current_step=None)
        current_state = _json_text({
            "BASE_SHA": ctx.base_sha,
            "HEAD_SHA": head,
            "CANDIDATE_TREE_SHA": tree_before,
            "CHANGED_FILES": changed_paths_between_trees(ctx.repo, ctx.base_tree_sha, tree_before),
            "GIT_STATUS": status_porcelain(ctx.info.worktree),
        })
        planner = RepairPlannerV2(
            self._planner_client or _chat_client(
                build_llm_endpoint(planner_profile), self._runtime_environment
            ),
            planning=self.config.planning,
            check_catalog=self.config.check_catalog,
            original_required_check_ids=ctx.plan.required_checks,
            # The correction starts from the reviewed candidate, not the base.
            repository_preconditions=RepositoryPreconditions(ctx.repo, candidate["tree_sha"]),
        )
        # The reviewed candidate commit is the code authority: the planner
        # gets its immutable candidate/compare URLs instead of an inline diff.
        remote_available = (
            candidate.get("remote_sha") == head
            and candidate.get("remote_branch") == ctx.info.branch
            and candidate.get("remote") == self.config.repository.remote
            and immutable_commit_web_url(ctx.repository_reference, head) is not None
            and compare_commits_web_url(ctx.repository_reference, ctx.base_sha, head) is not None
        )
        started_at, started_mono = self._trace_time(), time.perf_counter()
        planner_selected = self._trace_selected_profile(
            ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        self._trace_emit(
            "plan.started", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "tree_before": tree_before,
                "session": self._trace_session(
                    profile=planner_profile, selected=planner_selected,
                    role=ExecutionRole.PLANNER, prompt_bytes=None,
                    started_at=started_at, started_mono=started_mono,
                    tree_before=tree_before,
                ),
            },
        )
        try:
            plan = planner.plan(
                repository_reference=render_repository_reference(ctx.repository_reference),
                original_spec=ctx.spec,
                original_plan_summary=render_repair_plan_summary(ctx.plan),
                original_step_index=render_repair_step_index(ctx.plan),
                current_repository_state=current_state,
                candidate_code_evidence=_review_code_evidence(
                    repository_reference=ctx.repository_reference,
                    base_sha=ctx.base_sha, candidate_sha=head, evidence=evidence,
                    remote_sha=candidate.get("remote_sha"),
                    remote_branch=candidate.get("remote_branch"),
                    remote_name=candidate.get("remote"),
                ),
                previous_cycle_checks=_json_text(_repair_checks_payload(evidence)),
                previous_revision_report=_bounded_previous_revision_report(
                    review_cycle_revision_report(ctx.run_dir, previous, _load_revision) or "NONE"
                ),
                original_approved_mutable_scope=_json_text(approved_scope),
                reviewer_result=_json_text(_review_payload(review)),
                artifacts_dir=repair_dir,
                fallback_candidate_diff="" if remote_available else evidence.diff,
            )
        except PlanRepositoryPreconditionError as exc:
            raise PipelineFailure(exc.code, _bounded_parse_detail(exc)) from exc
        except V2PlanParseError as exc:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", _bounded_parse_detail(exc)) from exc
        self._trace_emit(
            "plan.completed", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "decision": plan.decision.value, "title": plan.title,
                "session": self._trace_finished_model_session(
                    profile=planner_profile, selected=planner_selected,
                    role=ExecutionRole.PLANNER,
                    prompt_bytes=(
                        (repair_dir / "planner.request.txt").stat().st_size
                        if (repair_dir / "planner.request.txt").is_file() else None
                    ),
                    started_at=started_at, started_mono=started_mono,
                    usage=getattr(planner, "last_usage", None),
                    tree_before=tree_before, tree_after=tree_before,
                    final_message=getattr(plan, "raw", None),
                ),
            },
        )
        self._cycle_update(store, cycle, status="planning", plan_summary=plan.title)
        if plan.decision is PlanDecision.BLOCKED:
            self._cycle_update(store, cycle, status="blocked", blockers=plan.blockers)
            raise PipelineFailure("REPAIR_PLANNER_BLOCKED", plan.blockers)
        bundle, bundle_sha = validate_implementation_bundle(
            repair_dir, expected_step_ids=[step.id for step in plan.steps]
        )
        self._authorize_correction_scope(
            store, ctx, cycle, plan, bundle_sha, candidate["commit_sha"], review, approved_scope,
        )
        return self._correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=True,
        )

    def _authorize_correction_scope(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, bundle_sha: str, candidate_sha: str, review: ReviewResult,
        approved_scope: list[str],
    ) -> None:
        """Bind the correction scope delta and apply the run's scope policy."""

        repair_dir = correction_dir(ctx.run_dir, cycle)
        try:
            delta, content = _build_scope_delta(
                repair_dir, original_scope=approved_scope, plan=plan,
                candidate_commit_sha=candidate_sha, review=review,
                repair_bundle_sha=bundle_sha,
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        # Created once; on every later pass (resume included) the persisted
        # bytes are only compared, never repaired.
        delta_sha = _ensure_scope_delta(repair_dir, content, expected_sha256=None)
        for path in delta["requested_write_paths"] + delta["requested_delete_paths"]:
            if not path_exists_in_tree(ctx.repo, candidate_sha, path):
                raise PipelineFailure("REPAIR_SCOPE_EXISTING_PATH_MISSING", path)
        for path in delta["requested_create_paths"]:
            if path_exists_in_tree(ctx.repo, candidate_sha, path):
                raise PipelineFailure("REPAIR_SCOPE_CREATE_PATH_EXISTS", path)
        if delta["added_paths"] and not review.required_fixes.strip() and not review.findings.strip():
            raise PipelineFailure("REPAIR_SCOPE_UNJUSTIFIED")
        requested = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        atomic_write_text(repair_dir / "scope.json", _json_text({
            "repair_mutable_scope": requested,
            "approved_mutable_scope_before": approved_scope,
            "scope_delta_sha256": delta_sha,
        }))
        added = delta["added_paths"]
        policy = self._effective_repair_scope
        if added and policy.policy == "deny-expansion":
            self._cycle_update(store, cycle, status="failed", failure="REPAIR_SCOPE_EXPANSION",
                               scope_delta=delta)
            raise PipelineFailure("REPAIR_SCOPE_EXPANSION")
        if added and (
            policy.policy == "require-approval"
            or (policy.policy == "auto-bounded" and len(added) > policy.max_added_paths)
        ):
            approval = read_scope_approval(repair_dir, expected_sha256=delta_sha)
            if approval is None:
                self._cycle_update(store, cycle, status="waiting_scope_approval", scope_delta=delta)
                store.update(
                    status=RunStatus.WAITING_SCOPE_APPROVAL, scope_delta=delta, current_step=None,
                )
                raise ScopeApprovalRequired()
            if approval.decision is not ApprovalDecision.APPROVE:
                raise PipelineFailure("HUMAN_REQUIRED", "correction scope rejected")
        elif added:
            self._cycle_update(store, cycle, status="scope_auto_approved", scope_delta=delta)
        # The correction plan may require trusted checks the initial plan did
        # not; their config-only preflights run before any expensive worker.
        check_config, check_ids = config_with_check_authority(
            self.config, ctx.run_dir, requested_check_ids=plan.required_checks,
            expected_sha256=self._approved_check_authority_sha256(ctx.run_dir),
        )
        preflight_failures = self._run_check_preflights_recoverably(
            store=store, worktree=ctx.info.worktree, check_config=check_config,
            check_ids=check_ids or plan.required_checks,
            counter_key=f"check-preflight:cycle:{cycle.number:03d}",
            phase="planning", cycle=cycle.number,
        )
        if preflight_failures:
            raise PipelineFailure(
                preflight_failures[0].split(":", 1)[0], preflight_failures[0],
            )

    def _load_correction(
        self, ctx: PipelineV2Context, cycle: RunCycle, expected_sha: str | None,
    ) -> CyclePlan:
        """Read back the correction plan a checkpoint of this cycle is bound to."""

        plan, bundle, bundle_sha = load_correction_plan(
            self.config, ctx.selection, ctx.run_dir, cycle.number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        if expected_sha is None or bundle_sha != expected_sha:
            raise ResumeIntegrityError(f"cycle {cycle.number:03d} correction plan changed")
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha, self._effective_repair_scope)
        return self._correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )

    def _completed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        return completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number, [step.id for step in cycle_plan.plan.steps],
        )

    def _effective_cycle_scope(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> tuple[str, ...]:
        """Include only durable, policy-approved contract-repair additions."""

        scope = set(cycle_plan.mutable_scope)
        root = implementation_steps_dir(ctx.run_dir, cycle_plan.cycle)
        if not root.is_dir():
            return tuple(sorted(scope))
        for step_dir in root.iterdir():
            repairs = step_dir / "contract_repairs"
            if not repairs.is_dir():
                continue
            for repair_dir in repairs.iterdir():
                validation = _read_json_artifact(repair_dir / "validation.json", 64 * 1024)
                if isinstance(validation, dict) and validation.get("status") == "validated":
                    added = validation.get("added_mutable_paths")
                    if isinstance(added, list) and all(isinstance(path, str) for path in added):
                        contract_path = repair_dir / "contract.md"
                        digest = validation.get("repaired_contract_sha256")
                        if (
                            not isinstance(digest, str) or not contract_path.is_file()
                            or hashlib.sha256(contract_path.read_bytes()).hexdigest() != digest
                        ):
                            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "step contract repair contract hash changed")
                        if added:
                            policy = self._effective_repair_scope
                            if policy.policy == "deny-expansion":
                                raise PipelineFailure("REPAIR_SCOPE_EXPANSION")
                            if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
                                delta_path = repair_dir / "scope_delta.json"
                                delta = _read_json_artifact(delta_path, 64 * 1024)
                                if not isinstance(delta, dict) or delta.get("added_paths") != sorted(added):
                                    raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "step contract repair scope delta is malformed")
                                approval = read_scope_approval(
                                    repair_dir,
                                    expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
                                )
                                if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                                    raise ScopeApprovalRequired()
                        scope.update(added)
        scope.update(_semantic_revision_scope(
            ctx.repo, ctx.run_dir, cycle_plan.cycle.number,
            self._effective_repair_scope,
        ))
        return tuple(sorted(scope))

    def _state_steps(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan, *, running: str | None = None,
    ) -> list[dict[str, Any]]:
        """The ``state.steps`` view of one cycle, derived from durable records."""

        ctx_steps = {record["id"]: record for record in self._completed_steps(ctx, cycle_plan)}
        rows: list[dict[str, Any]] = []
        for step in cycle_plan.plan.steps:
            record = ctx_steps.get(step.id)
            row: dict[str, Any] = {
                "id": step.id, "title": step.title,
                "profile_id": cycle_plan.step_profile_ids.get(step.id),
                "status": "waiting",
            }
            if record is not None:
                row.update(
                    status="deferred" if record["status"] == "DEFERRED_CONTRACT_MISMATCH" else "completed",
                    usage=record["usage"],
                    input_tokens=record["usage"]["input_tokens"],
                    output_tokens=record["usage"]["output_tokens"],
                )
            elif step.id == running:
                row["status"] = "running"
            rows.append(row)
        return rows

    def _execute_cycle_step(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan, index: int,
    ) -> None:
        """Execute, verify and accept exactly one approved step of a cycle."""

        step = cycle_plan.plan.steps[index]
        step_artifact_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
        # A previous failed attempt of this same step keeps its artifacts.
        _archive_attempt(step_artifact_dir)
        try:
            contract = read_approved_step_contract(cycle_plan.contracts_dir, cycle_plan.bundle, step.id)
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise PipelineFailure("PLAN_APPROVAL_INVALID", str(exc), step_id=step.id) from exc
        store.update(
            status=RunStatus.IMPLEMENTING, current_step=step.id,
            steps=self._state_steps(ctx, cycle_plan, running=step.id),
        )
        checkpoint = read_checkpoint(ctx.run_dir)
        parent_sha = current_head(ctx.info.worktree)
        planner_profile = profile_for_role(
            self.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        reviewer_profile = profile_for_role(
            self.config, ctx.selection.final_reviewer.profile_id, ExecutionRole.REVIEWER
        )
        outcome = self._execute_step_attempts(
            store=store, run_dir=ctx.run_dir, original_spec=ctx.spec,
            original_plan_identity=_json_text(
                asdict(checkpoint.plan_identity) if checkpoint and checkpoint.plan_identity else {}
            ),
            repo=ctx.repo, worktree=ctx.info.worktree, base_sha=parent_sha,
            branch_ref=ctx.branch_ref,
            ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
            expected_tree=(
                checkpoint.expected_tree_sha if checkpoint is not None
                else candidate_tree_sha(ctx.info.worktree)
            ),
            step=step, contract=contract,
            profile_id=cycle_plan.step_profile_ids[step.id],
            fallback_profile_ids=(cycle_plan.step_fallback_profile_ids or {}).get(step.id, ()),
            artifact_dir=step_artifact_dir,
            forbidden_env_names=(planner_profile.api_key_env, reviewer_profile.api_key_env),
            future_ownership=_future_step_ownership(cycle_plan.plan.steps, index),
        )
        try:
            self._accept_v2_step_tree(
                store=store, run_dir=ctx.run_dir, info=ctx.info, step=step,
                outcome=outcome, parent_sha=parent_sha,
                future_step_ids=tuple(item.id for item in cycle_plan.plan.steps[index + 1:]),
                run_id=ctx.run_id, step_dir=step_artifact_dir,
            )
        except (CommitSafetyError, GitError) as exc:
            raise PipelineFailure(
                "COMMIT_GATE_FAILED", _bounded_parse_detail(exc), step_id=step.id,
            ) from exc
        store.update(
            status=RunStatus.IMPLEMENTING, current_step=None,
            steps=self._state_steps(ctx, cycle_plan),
        )
        self._update_v2_usage(store, ctx.run_dir)

    def _semantic_revision(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> None:
        """One semantic revision pass over the implemented cycle."""

        number = cycle_plan.cycle.number
        artifact_dir = semantic_revision_dir(ctx.run_dir, number)
        steps = self._completed_steps(ctx, cycle_plan)
        mutable_scope = list(self._effective_cycle_scope(ctx, cycle_plan))
        while True:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            try:
                result, error = self._run_revision_with_recovery(
                    store=store, cycle=number, is_check_repair=False,
                    request={
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": cycle_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": artifact_dir,
                "mutable_scope": mutable_scope, "step_results": steps,
                "deferred_mismatches": _deferred_contract_mismatches(cycle_plan.plan, steps),
                "deferred_mismatch_present": _has_deferred_contract_mismatches(steps),
                    },
                )
            except PipelineFailure:
                raise
            except (AgentScopeError, AgentError) as exc:
                self._redact_revision_artifacts(artifact_dir)
                _record_failure_tree(artifact_dir, ctx.info.worktree)
                raise PipelineFailure(
                    getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self._secrets),
                ) from exc
            if error == _SCOPE_REQUEST_ROUTE:
                outcome, mutable_scope = self._authorize_semantic_scope_request(
                    store, ctx, cycle_plan, artifact_dir, mutable_scope,
                )
                if outcome == "expanded":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    continue
                if outcome == "replan":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    return
                report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
                final = report.get("final", "") if isinstance(report, dict) else ""
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                self._cycle_update(
                    store, cycle_plan.cycle, status="revised",
                    semantic_revision_status="SCOPE_REQUEST_RECORDED",
                    semantic_revision_report=_bounded_report(str(final)),
                )
                return
            if error is not None:
                if error in {
                    AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                    AGENT_PROTOCOL_FAILED,
                }:
                    unavailable = {"status": "UNAVAILABLE", "reason": error[:120]}
                    atomic_write_text(artifact_dir / "status.json", _json_text(unavailable))
                    self._cycle_update(
                        store, cycle_plan.cycle, status="revision_unavailable",
                        semantic_revision_status="UNAVAILABLE",
                        semantic_revision_reason=unavailable["reason"],
                        semantic_revision_report=(
                            "SEMANTIC REVISION: UNAVAILABLE\nreason=" + unavailable["reason"]
                        ),
                    )
                    store.update(status=RunStatus.REVISING, semantic_revision=unavailable)
                    return
                if error in {"REVISION_SCOPE_VIOLATION", AGENT_SCOPE_VIOLATION}:
                    raise PipelineFailure(
                        AGENT_SCOPE_VIOLATION,
                        "semantic revision changed a path outside approved authority",
                    )
                raise PipelineFailure(error)
            self._cycle_update(
                store, cycle_plan.cycle, status="revised",
                semantic_revision_status="COMPLETED",
                semantic_revision_report=_bounded_report(result.final_message) if result else "",
            )
            return

    def _authorize_semantic_scope_request(
        self,
        store: RunStateStore,
        ctx: PipelineV2Context,
        cycle_plan: CyclePlan,
        artifact_dir: Path,
        current_scope: list[str],
    ) -> tuple[str, list[str]]:
        """Persist and apply a strict, policy-bounded reviser scope request."""

        report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
        request = report.get("scope_request") if isinstance(report, dict) else None
        paths = request.get("paths") if isinstance(request, dict) else None
        reason = request.get("reason") if isinstance(request, dict) else None
        evidence = request.get("evidence") if isinstance(request, dict) else None
        tree_sha = report.get("tree_before") if isinstance(report, dict) else None
        if (
            not isinstance(paths, list) or not paths or any(
                not isinstance(path, str) or not self._safe_scope_request_path(path)
                for path in paths
            ) or len(paths) != len(set(paths))
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence)
            or not isinstance(tree_sha, str) or not _is_object_id(tree_sha)
        ):
            raise PipelineFailure(AGENT_SCOPE_VIOLATION, "semantic scope request is malformed")
        source_report = artifact_dir / "report.json"
        try:
            source_report_sha = hashlib.sha256(source_report.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_REQUIRES_OPERATOR", "semantic scope request report is unreadable",
            ) from exc
        try:
            exists = {
                path: path_exists_in_tree(ctx.repo, tree_sha, path) for path in paths
            }
        except GitError as exc:
            raise PipelineFailure("RESUME_REQUIRES_OPERATOR", "scope request tree semantics are unreadable") from exc
        base = tuple(sorted(set(current_scope)))
        requested = tuple(sorted(set(paths)))
        added = tuple(path for path in requested if path not in base)
        root = artifact_dir / "scope_requests"
        root.mkdir(parents=True, exist_ok=True)
        existing_added: set[str] = set()
        next_number = 1
        prior_request_dir: Path | None = None
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.isdigit():
                continue
            next_number = max(next_number, int(path.name) + 1)
            saved = _read_json_artifact(path / "authority.json", 64 * 1024)
            if isinstance(saved, dict):
                saved_added = saved.get("added_paths")
                if isinstance(saved_added, list):
                    existing_added.update(item for item in saved_added if isinstance(item, str))
                if (
                    saved.get("tree_sha") == tree_sha
                    and saved.get("base_mutable_scope") == list(base)
                    and saved.get("requested_paths") == list(requested)
                    and saved.get("reason") == reason
                    and saved.get("evidence") == [item[:1000] for item in evidence[:16]]
                ):
                    prior_request_dir = path
        target = prior_request_dir or root / f"{next_number:03d}"
        target.mkdir(parents=True, exist_ok=True)
        added_all = tuple(sorted(existing_added | set(added)))
        policy = self._effective_repair_scope
        authority = {
            "schema_version": 1,
            "cycle": cycle_plan.cycle.number,
            "tree_sha": tree_sha,
            "source_report_sha256": source_report_sha,
            "base_mutable_scope": list(base),
            "requested_paths": list(requested),
            "added_paths": list(added),
            "existing_paths": [path for path in requested if exists[path]],
            "create_paths": [path for path in requested if not exists[path]],
            "reason": reason[:2000],
            "evidence": [item[:1000] for item in evidence[:16]],
            "policy": policy.policy,
            "bound": policy.max_added_paths,
        }
        authority_path = target / "authority.json"
        authority_content = _json_text(authority)
        if authority_path.exists():
            saved_authority = _read_json_artifact(authority_path, 64 * 1024)
            saved_semantics = (
                {key: value for key, value in saved_authority.items()
                 if key != "source_report_sha256"}
                if isinstance(saved_authority, dict) else None
            )
            current_semantics = {
                key: value for key, value in authority.items()
                if key != "source_report_sha256"
            }
            if saved_semantics != current_semantics:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "semantic scope request authority changed")
            authority = saved_authority
        else:
            atomic_write_text(authority_path, authority_content)

        if not added:
            atomic_write_text(target / "decision.json", _json_text({
                **authority, "decision": "recorded-in-scope",
            }))
            atomic_write_text(artifact_dir / "status.json", _json_text({
                "status": "SCOPE_REQUEST_RECORDED",
                "reason": reason[:2000],
                "requested_paths": list(requested),
            }))
            return "recorded", current_scope
        if policy.policy == "deny-expansion":
            denied = {**authority, "decision": "denied-expansion"}
            atomic_write_text(target / "decision.json", _json_text(denied))
            status = {
                "status": "REPLAN_REQUIRED",
                "reason": "semantic scope expansion denied by recovery policy",
            }
            atomic_write_text(artifact_dir / "status.json", _json_text(status))
            report_text = (
                "SEMANTIC REVISION: SCOPE EXPANSION DENIED\nroute=REPLAN\n"
                f"reason={_bounded_v2_report(reason)}"
            )
            self._cycle_update(
                store, cycle_plan.cycle, status="scope_expansion_denied",
                semantic_revision_status="REPLAN_REQUIRED",
                semantic_revision_report=report_text,
            )
            return "replan", current_scope

        delta_path = target / "scope_delta.json"
        delta = {
            "schema_version": 1, "cycle": cycle_plan.cycle.number,
            "tree_sha": tree_sha, "added_paths": list(added),
            "requested_paths": list(requested), "reason": reason[:2000],
            "evidence": [item[:1000] for item in evidence[:16]],
            "policy": policy.policy, "bound": policy.max_added_paths,
        }
        delta_content = _json_text(delta)
        if delta_path.exists() and delta_path.read_text(encoding="utf-8") != delta_content:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "semantic scope delta changed")
        if not delta_path.exists():
            atomic_write_text(delta_path, delta_content)
        delta_sha = hashlib.sha256(delta_path.read_bytes()).hexdigest()
        approval = read_scope_approval(target, expected_sha256=delta_sha)
        requires_approval = policy.policy == "require-approval" or len(added_all) > policy.max_added_paths
        if requires_approval and approval is None:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            delta["approval_artifact"] = delta_path.relative_to(ctx.run_dir).as_posix()
            store.update(
                status=RunStatus.WAITING_SCOPE_APPROVAL,
                scope_delta=delta,
                current_step=None,
            )
            self._cycle_update(
                store, cycle_plan.cycle, status="waiting_scope_approval", scope_delta=delta,
            )
            raise ScopeApprovalRequired()
        if approval is not None and approval.decision is not ApprovalDecision.APPROVE:
            raise PipelineFailure("HUMAN_REQUIRED", "semantic scope request was rejected")
        if requires_approval and approval is None:
            raise ScopeApprovalRequired()
        return "expanded", sorted(set(current_scope) | set(added))

    def _semantic_review_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        review: ReviewResult,
    ) -> None:
        """Apply one direct semantic correction requested by the reviewer.

        This route deliberately has no new plan or implementation step.  The
        reviewer report is evidence for the reviser, while the approved plan,
        candidate commit and cumulative mutable scope remain the authorities.
        """

        if review is None or review.route is not ReviewRoute.IMPLEMENTATION:
            raise ResumeIntegrityError("direct semantic correction has no implementation review")
        number = cycle_plan.cycle.number
        previous_plan = self._cycle_plan(ctx, number - 1)
        candidate = read_candidate_record(ctx.run_dir, number - 1)
        evidence = candidate_evidence(ctx.run_dir, number - 1)
        if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
            raise ResumeIntegrityError(f"cycle {number - 1:03d} candidate evidence is missing")
        approved_scope = self._approved_scope_before(ctx, number)
        code_evidence = _review_code_evidence(
            repository_reference=ctx.repository_reference,
            base_sha=ctx.base_sha,
            candidate_sha=candidate["commit_sha"],
            evidence=evidence,
            remote_sha=candidate.get("remote_sha"),
            remote_branch=candidate.get("remote_branch"),
            remote_name=candidate.get("remote"),
        )
        candidate_identity = _json_text({
            "authority": "immutable_candidate_commit",
            "commit_sha": candidate["commit_sha"],
            "tree_sha": candidate["tree_sha"],
            "parent_sha": candidate["parent_sha"],
            "repository_reference": repository_reference_dict(ctx.repository_reference),
        })
        reviewer_evidence = "\n\n".join((
            "SUMMARY\n" + review.summary,
            "FINDINGS\n" + review.findings,
            "REQUIRED FIXES\n" + review.required_fixes,
            "MISSING TESTS\n" + review.missing_tests,
            "CORRECTION EVIDENCE\n" + code_evidence,
        ))
        artifact_dir = semantic_revision_dir(ctx.run_dir, number)
        mutable_scope = list(self._effective_cycle_scope(ctx, cycle_plan))
        step_results = self._completed_steps(ctx, previous_plan)
        while True:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            try:
                result, error = self._run_revision_with_recovery(
                    store=store, cycle=number, is_check_repair=False,
                    request={
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": previous_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": artifact_dir,
                "mutable_scope": mutable_scope,
                "step_results": step_results,
                "deferred_mismatches": _deferred_contract_mismatches(
                    previous_plan.plan, step_results
                ),
                "deferred_mismatch_present": _has_deferred_contract_mismatches(
                    step_results
                ),
                "reviewer_correction_evidence": reviewer_evidence,
                "candidate_identity": candidate_identity,
                "bounded_diff_evidence": bounded_semantic_diff(evidence.diff, 16 * 1024)[0],
                    },
                )
            except PipelineFailure:
                raise
            except AgentError as exc:
                self._redact_revision_artifacts(artifact_dir)
                raise PipelineFailure(
                    getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self._secrets),
                ) from exc
            if error == _SCOPE_REQUEST_ROUTE:
                outcome, mutable_scope = self._authorize_semantic_scope_request(
                    store, ctx, cycle_plan, artifact_dir, mutable_scope,
                )
                if outcome == "expanded":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    continue
                if outcome == "replan":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    return
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                self._cycle_update(
                    store, cycle_plan.cycle, status="revised",
                    semantic_revision_status="SCOPE_REQUEST_RECORDED",
                    semantic_revision_report="SEMANTIC REVISION: SCOPE REQUEST RECORDED",
                )
                return
            if error in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AGENT_PROTOCOL_FAILED,
            }:
                unavailable = {"status": "UNAVAILABLE", "reason": error[:120]}
                atomic_write_text(artifact_dir / "status.json", _json_text(unavailable))
                self._cycle_update(
                    store, cycle_plan.cycle, status="revision_unavailable",
                    semantic_revision_status="UNAVAILABLE",
                    semantic_revision_reason=error[:120],
                    semantic_revision_report=(
                        "SEMANTIC REVISION: UNAVAILABLE\nreason=" + error[:120]
                    ),
                )
                return
            if error is not None:
                if error in {"REVISION_SCOPE_VIOLATION", AGENT_SCOPE_VIOLATION}:
                    raise PipelineFailure(AGENT_SCOPE_VIOLATION, "semantic correction changed a path outside approved scope")
                raise PipelineFailure(error)
            self._cycle_update(
                store, cycle_plan.cycle, status="revised",
                semantic_revision_status="COMPLETED",
                semantic_revision_report=_bounded_report(result.final_message) if result else "",
            )
            return

    def _run_gate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage,
    ) -> EvidenceBundle:
        """Run the authoritative deterministic checks for the current tree."""

        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        retry_key = f"check-infra:{cycle_plan.cycle.number:03d}:{stage.value}"
        retries_used = self._recovery_counter(store, retry_key)
        retry_budget = ctx.options.recovery.max_check_infra_retries
        retries_left = max(0, retry_budget - retries_used)
        pending_retry: tuple[int, str, str, RecoveryDecision] | None = None
        evidence: EvidenceBundle | None = None
        for _ in range(retries_left + 1):
            # Each infrastructure attempt has a fresh artifact directory. The
            # failed evidence remains archived and cannot be mistaken for the
            # result of a resumed check invocation.
            _archive_attempt(directory, names=_CHECK_ATTEMPT_ARTIFACTS)
            store.update(status=RunStatus.VALIDATING, current_step=None)
            tree_before = _safe_candidate_tree(ctx.info.worktree)
            try:
                evidence = self._final_evidence(
                    ctx.info.worktree, ctx.base_sha, directory,
                    check_failures_hard=False, reuse=True, stage=stage,
                    expected_head_sha=current_head(ctx.info.worktree),
                    required_check_ids=cycle_plan.plan.required_checks or None,
                    enforce_diff_size=False,
                )
            except Exception as exc:
                reason = _failure_reason(exc)
                if reason != "CHECK_PREFLIGHT_FAILED":
                    raise
                decision = classify_failure(reason)
                attempt = retries_used + 1
                tree_after = _safe_candidate_tree(ctx.info.worktree)
                self._trace_recovery(
                    "recovery.classified", reason=reason, decision=decision,
                    attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                    budget_remaining=max(0, retry_budget - retries_used),
                    phase="validation", cycle=cycle_plan.cycle.number,
                )
                if retries_used >= retry_budget:
                    exhausted = classify_failure(reason, budget_exhausted=True)
                    self._trace_recovery(
                        "recovery.exhausted", reason=reason, decision=exhausted,
                        attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                        budget_remaining=0, phase="validation",
                        cycle=cycle_plan.cycle.number,
                    )
                    raise PipelineFailure(
                        reason, "check preflight recovery budget exhausted",
                    ) from exc
                retries_used = self._consume_recovery_counter(store, retry_key)
                remaining = retry_budget - retries_used
                self._trace_recovery(
                    "recovery.started", reason=reason, decision=decision,
                    attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                    budget_remaining=remaining, phase="validation",
                    cycle=cycle_plan.cycle.number,
                )
                pending_retry = (attempt, reason, tree_before, decision)
                continue

            timeouts = [item for item in evidence.failures if item.startswith("CHECK_TIMEOUT:")]
            tree_after = evidence.staged_tree_sha
            if pending_retry is not None:
                attempt, retry_reason, retry_tree, decision = pending_retry
                self._trace_recovery(
                    "recovery.completed", reason=retry_reason,
                    decision=decision, attempt=attempt, tree_before=retry_tree,
                    tree_after=tree_after,
                    budget_remaining=max(0, retry_budget - retries_used),
                    phase="validation", cycle=cycle_plan.cycle.number,
                    recovered=not timeouts,
                )
                pending_retry = None
            if not timeouts:
                break
            reason = "CHECK_TIMEOUT"
            decision = classify_failure(reason)
            attempt = retries_used + 1
            self._trace_recovery(
                "recovery.classified", reason=reason, decision=decision,
                attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                budget_remaining=max(0, retry_budget - retries_used),
                phase="validation", cycle=cycle_plan.cycle.number,
            )
            if retries_used >= retry_budget:
                exhausted = classify_failure(reason, budget_exhausted=True)
                self._trace_recovery(
                    "recovery.exhausted", reason=reason, decision=exhausted,
                    attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                    budget_remaining=0, phase="validation",
                    cycle=cycle_plan.cycle.number,
                )
                break
            retries_used = self._consume_recovery_counter(store, retry_key)
            self._trace_recovery(
                "recovery.started", reason=reason, decision=decision,
                attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                budget_remaining=retry_budget - retries_used,
                phase="validation", cycle=cycle_plan.cycle.number,
            )
            pending_retry = (attempt, reason, tree_before, decision)
        if evidence is None:  # pragma: no cover - each non-retry exit assigns evidence
            raise PipelineFailure("CHECK_PREFLIGHT_FAILED", "check preflight produced no evidence")
        gate = {
            "stage": stage.value,
            "passed": evidence.deterministic_passed,
            "required_check_ids": list(evidence.required_check_ids),
            "failures": list(evidence.failures),
        }
        store.update(
            status=RunStatus.VALIDATING, checks=_check_payload(evidence),
            staged_tree_sha=evidence.staged_tree_sha,
            changed_files=list(evidence.changed_files),
            deterministic_gate=gate,
        )
        self._cycle_update(store, cycle_plan.cycle, deterministic_gate=gate)
        return evidence

    def _run_check_preflights_recoverably(
        self,
        *,
        store: RunStateStore,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        counter_key: str,
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Retry trusted read-only preflights under a durable infrastructure budget."""

        reason = "CHECK_PREFLIGHT_FAILED"
        retry_budget = self._run_options.recovery.max_check_infra_retries
        retries_used = self._recovery_counter(store, counter_key)
        retries_left = max(0, retry_budget - retries_used)
        pending_retry: tuple[int, str, RecoveryDecision] | None = None
        for _ in range(retries_left + 1):
            tree_before = _safe_candidate_tree(worktree)
            failures = run_check_preflights(worktree, check_config, check_ids)
            tree_after = _safe_candidate_tree(worktree)
            if tree_before != tree_after:
                decision = classify_failure("CHECK_MUTATED")
                self._trace_recovery(
                    "recovery.classified", reason="CHECK_MUTATED", decision=decision,
                    attempt=retries_used + 1, tree_before=tree_before, tree_after=tree_after,
                    budget_remaining=0, phase=phase, cycle=cycle,
                )
                raise OrchestrationError("CHECK_MUTATED: preflight changed the candidate tree")
            if pending_retry is not None:
                attempt, retry_tree, pending_decision = pending_retry
                self._trace_recovery(
                    "recovery.completed", reason=reason, decision=pending_decision,
                    attempt=attempt, tree_before=retry_tree, tree_after=tree_after,
                    budget_remaining=max(0, retry_budget - retries_used),
                    phase=phase, cycle=cycle, recovered=not failures,
                )
                pending_retry = None
            if not failures:
                return ()
            decision = classify_failure(reason)
            attempt = retries_used + 1
            self._trace_recovery(
                "recovery.classified", reason=reason, decision=decision,
                attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                budget_remaining=max(0, retry_budget - retries_used),
                phase=phase, cycle=cycle,
            )
            if retries_used >= retry_budget:
                exhausted = classify_failure(reason, budget_exhausted=True)
                self._trace_recovery(
                    "recovery.exhausted", reason=reason, decision=exhausted,
                    attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                    budget_remaining=0, phase=phase, cycle=cycle,
                )
                raise OrchestrationError(
                    f"{failures[0]}: check infrastructure recovery budget exhausted"
                )
            retries_used = self._consume_recovery_counter(store, counter_key)
            self._trace_recovery(
                "recovery.started", reason=reason, decision=decision,
                attempt=attempt, tree_before=tree_before, tree_after=tree_after,
                budget_remaining=retry_budget - retries_used,
                phase=phase, cycle=cycle,
            )
            pending_retry = (attempt, tree_before, decision)
        return failures

    @staticmethod
    def _recovery_counter(store: RunStateStore, key: str) -> int:
        state = store.load()
        counters = state.get("recovery_counters", {})
        if not isinstance(counters, dict):
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counters are malformed")
        value = counters.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counter is malformed")
        return value

    @classmethod
    def _consume_recovery_counter(cls, store: RunStateStore, key: str) -> int:
        state = store.load()
        counters = state.get("recovery_counters", {})
        if not isinstance(counters, dict):
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counters are malformed")
        value = counters.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "recovery counter is malformed")
        updated = {**counters, key: value + 1}
        store.update(
            status=state.get("status", RunStatus.VALIDATING),
            recovery_counters=updated,
        )
        return value + 1

    @staticmethod
    def _check_repair_attempt_records(
        run_dir: Path, cycle: int, stage: GateStage,
    ) -> tuple[CheckRepairAttempt, ...]:
        """The contiguous durable attempts of one gate episode."""

        root = check_repair_dir(run_dir, cycle, stage) / "attempts"
        records: list[CheckRepairAttempt] = []
        if not root.is_dir():
            return ()
        for directory in sorted(root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or not directory.name.isdigit():
                continue
            record_path = directory / "attempt.json"
            if not record_path.is_file():
                # A worker can be interrupted after its prompt was persisted
                # but before the successful attempt record.  That boundary is
                # retryable; a present but malformed record is not.
                continue
            payload = _read_json_artifact(record_path, 128 * 1024)
            if not isinstance(payload, dict):
                raise ResumeIntegrityError("check-repair attempt record is malformed")
            number = payload.get("number")
            failed = payload.get("failed_check_ids_before")
            scope = payload.get("mutable_scope")
            before, after = payload.get("tree_before"), payload.get("tree_after")
            if (
                isinstance(number, bool) or not isinstance(number, int) or number < 1
                or number != int(directory.name)
                or not isinstance(failed, list) or any(not isinstance(item, str) for item in failed)
                or not isinstance(scope, list) or any(not isinstance(item, str) for item in scope)
                or not _is_object_id(before) or not _is_object_id(after)
            ):
                raise ResumeIntegrityError("check-repair attempt record is invalid")
            records.append(CheckRepairAttempt(
                number=number,
                failed_check_ids_before=tuple(failed),
                tree_before=before,
                tree_after=after,
                mutable_scope=tuple(scope),
            ))
        records.sort(key=lambda item: item.number)
        if any(item.number != index for index, item in enumerate(records, start=1)):
            raise ResumeIntegrityError("check-repair attempt records are not contiguous")
        return tuple(records)

    def _run_check_repair_attempt(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, attempt: int, evidence: EvidenceBundle,
    ) -> None:
        """One bounded check-repair worker pass on the red gate evidence."""

        number = cycle_plan.cycle.number
        attempt_dir = check_repair_attempt_dir(ctx.run_dir, number, stage, attempt)
        # A failed earlier try of this same attempt keeps its artifacts.
        _archive_attempt_tree(attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        selected = ctx.selection.check_repair
        if selected is None:
            raise PipelineFailure("CHECK_REPAIR_PROFILE_MISSING")
        records = self._check_repair_attempt_records(ctx.run_dir, number, stage)
        previous_scope = None
        if records:
            previous_authority = gate_mutable_authority(
                ctx.run_dir, number, stage,
                base_paths=self._effective_cycle_scope(ctx, cycle_plan),
                policy_config=self._effective_repair_scope,
                through_attempt=len(records),
                require_attempt_records=True,
            )
            previous_scope = CheckRepairScope(
                base_paths=previous_authority.base_paths,
                added_paths=previous_authority.added_paths,
                effective_paths=previous_authority.effective_paths,
                policy=self._effective_repair_scope.policy,
                bound=self._effective_repair_scope.max_added_paths,
                source=previous_authority.source,
            )
        soft = _soft_check_failures(evidence)
        failed_ids = tuple(item.split(":", 1)[1] for item in soft if ":" in item)
        scope = CheckRepairCoordinator(
            effective_repair_scope=self._effective_repair_scope,
        ).resolve_scope(
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=evidence.staged_tree_sha,
            run_dir=ctx.run_dir, evidence=evidence,
            base_mutable_scope=self._effective_cycle_scope(ctx, cycle_plan), previous=previous_scope,
        )
        decision = classify_failure(soft[0] if soft else "CHECK_FAILED")
        if decision.disposition is not RecoveryDisposition.CHECK_REPAIR:
            raise PipelineFailure("CHECK_REPAIR_NOT_AUTHORIZED", "check failure is not repairable")
        atomic_write_text(
            attempt_dir / "failed_check_evidence_before.json",
            _json_text(_check_payload(evidence)),
        )
        atomic_write_text(attempt_dir / "profile.json", _json_text({
            "profile_id": selected.profile_id,
            "profile_fingerprint": selected.config_sha256,
            "selected_profile": asdict(selected),
        }))
        progress = {
            "stage": stage.value,
            "attempt_number": attempt,
            "failure_ids": list(soft),
            "repair_profile_id": selected.profile_id,
            "repair_profile_fingerprint": selected.config_sha256,
            "mutable_scope": list(scope.effective_paths),
        }
        self._cycle_update(
            store, cycle_plan.cycle,
            check_repair={"status": "running", "attempt_count": len(records), **progress},
        )
        store.update(
            status=RunStatus.REVISING,
            check_repair={"status": "running", "attempt_count": len(records), **progress},
        )
        self._trace_recovery(
            "recovery.classified", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=evidence.staged_tree_sha,
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt + 1),
            phase="validation", cycle=number,
        )
        self._trace_recovery(
            "recovery.started", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=_safe_candidate_tree(ctx.info.worktree),
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt),
            phase="validation", cycle=number,
        )
        try:
            _result, error = self._run_revision_with_recovery(
                store=store, cycle=number, is_check_repair=True, attempt=attempt,
                request={
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": cycle_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": attempt_dir,
                "mutable_scope": list(scope.effective_paths),
                "check_repair_evidence": evidence, "check_repair_scope": scope,
                "check_repair_attempt": attempt,
                "gate_stage": stage.value,
                },
            )
        except (AgentError, GitError, OSError) as exc:
            error = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
            _record_failure_tree(attempt_dir, ctx.info.worktree)
        effective_executor = _read_json_artifact(attempt_dir / "executor.json", 16 * 1024)
        effective_profile_id = (
            effective_executor.get("profile_id")
            if isinstance(effective_executor, dict)
            and isinstance(effective_executor.get("profile_id"), str)
            else selected.profile_id
        )
        effective_selected = self._trace_selected_profile(effective_profile_id, ExecutionRole.REPAIR)
        effective_fingerprint = getattr(effective_selected, "config_sha256", None)
        if error is not None:
            if error in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                AGENT_PROTOCOL_FAILED,
            }:
                error_detail = f"CHECK_REPAIR_UNAVAILABLE after {error}"
                atomic_write_text(attempt_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "number": attempt,
                    "failed_check_ids_before": list(failed_ids),
                    "tree_before": evidence.staged_tree_sha,
                    "tree_after": candidate_tree_sha(ctx.info.worktree),
                    "mutable_scope": list(scope.effective_paths),
                    "profile_id": effective_profile_id,
                    "profile_fingerprint": effective_fingerprint,
                    "reason": "CHECK_REPAIR_UNAVAILABLE",
                    "infrastructure_reason": error[:120],
                }))
                store.update(
                    status=RunStatus.REVISING,
                    check_repair={
                        "status": "unavailable", "error": error[:120], **progress,
                    },
                )
                raise PipelineFailure("CHECK_REPAIR_UNAVAILABLE", error_detail)
            reason = "REVISION_SCOPE_VIOLATION" if error == _SCOPE_REQUEST_ROUTE else error
            self._trace_recovery(
                "recovery.completed", reason="CHECK_FAILED", decision=decision,
                attempt=attempt, tree_before=evidence.staged_tree_sha,
                tree_after=_safe_candidate_tree(ctx.info.worktree),
                budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt),
                phase="validation", cycle=number, recovered=False,
            )
            atomic_write_text(attempt_dir / "failure.json", _json_text({
                "schema_version": 1,
                "number": attempt,
                "failed_check_ids_before": list(failed_ids),
                "tree_before": evidence.staged_tree_sha,
                "tree_after": _safe_candidate_tree(ctx.info.worktree),
                "mutable_scope": list(scope.effective_paths),
                "profile_id": effective_profile_id,
                "profile_fingerprint": effective_fingerprint,
                "reason": reason,
            }))
            store.update(
                status=RunStatus.REVISING,
                check_repair={"status": "failed", "error": reason, **progress},
            )
            raise PipelineFailure(reason, ", ".join(soft))
        record = CheckRepairAttempt(
            number=attempt,
            failed_check_ids_before=failed_ids,
            tree_before=evidence.staged_tree_sha,
            tree_after=candidate_tree_sha(ctx.info.worktree),
            mutable_scope=tuple(scope.effective_paths),
        )
        atomic_write_text(attempt_dir / "attempt.json", _json_text({
            "schema_version": 1,
            **asdict(record),
            "profile_id": effective_profile_id,
            "profile_fingerprint": effective_fingerprint,
            "status": "completed",
        }))
        attempts = [*records, record]
        self._cycle_update(
            store, cycle_plan.cycle,
            check_repair={
                "status": "completed", "attempt_count": len(attempts),
                "attempts": [asdict(item) for item in attempts], **progress,
            },
        )
        store.update(
            status=RunStatus.REVALIDATING,
            check_repair={
                "status": "completed", "attempt_count": len(attempts),
                "attempts": [asdict(item) for item in attempts], **progress,
            },
        )
        self._trace_recovery(
            "recovery.completed", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=record.tree_after,
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt),
            phase="validation", cycle=number, recovered=True,
        )
        self._update_v2_usage(store, ctx.run_dir)

    def _review_context_builder(self) -> ReviewContextBuilder:
        return ReviewContextBuilder(
            cycle_plan=self._cycle_plan,
            completed_steps=self._completed_steps,
            load_revision=_load_revision,
            read_candidate=read_candidate_record,
            candidate_evidence=candidate_evidence,
            accepted_review=_accepted_review,
        )

    def _review_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        candidate: Mapping[str, Any], evidence: EvidenceBundle,
    ) -> ReviewResult:
        """Review exactly the pushed candidate; an accepted answer is reused."""

        self._assert_candidate_review_authority(ctx, candidate, evidence)
        directory = review_dir(ctx.run_dir, cycle_plan.cycle)
        accepted = _accepted_review(directory, evidence, candidate["commit_sha"])
        if accepted is not None:
            store.update(
                status=store.load().get("status", RunStatus.REVIEWING),
                reviewed_candidate_sha=candidate["commit_sha"],
            )
            return accepted
        _archive_attempt(directory, names=_REVIEW_ATTEMPT_ARTIFACTS)
        reviewer = self._reviewer_for_profile(ctx.selection.final_reviewer.profile_id)
        store.update(status=RunStatus.REVIEWING, current_step=None)
        try:
            review = self._run_v2_reviewer(
                reviewer=reviewer, spec=ctx.spec, run_dir=ctx.run_dir,
                repository_reference=ctx.repository_reference, evidence=evidence,
                input=self._review_context_builder().build(ctx, cycle_plan), artifacts_dir=directory,
                worktree=ctx.info.worktree, base_sha=ctx.base_sha,
                candidate_commit=candidate,
            )
            store.update(
                status=store.load().get("status", RunStatus.REVIEWING),
                reviewed_candidate_sha=candidate["commit_sha"],
            )
            return review
        except LLMError as exc:
            # Transport only: no reviewer answer was accepted, so the same
            # exact candidate can be reviewed again on resume.
            raise PipelineFailure(
                "REVIEWER_TRANSPORT_FAILURE", _bounded_parse_detail(exc),
            ) from exc

    def _assert_candidate_review_authority(
        self,
        ctx: PipelineV2Context,
        candidate: Mapping[str, Any],
        evidence: EvidenceBundle,
    ) -> None:
        """Fail closed unless the reviewer has an exact pushed candidate."""

        candidate_sha = candidate.get("commit_sha")
        candidate_tree = candidate.get("tree_sha")
        if not _is_object_id(candidate_sha) or not _is_object_id(candidate_tree):
            raise PipelineFailure(
                "CANDIDATE_PUSH_FAILED", "candidate identity is incomplete before review"
            )
        try:
            if symbolic_head(ctx.info.worktree) != ctx.branch_ref:
                raise GitError("candidate worktree is not on the run branch")
            if current_head(ctx.info.worktree) != candidate_sha:
                raise GitError("local HEAD does not equal the candidate commit")
            if resolve_tree(ctx.info.worktree, candidate_sha) != candidate_tree:
                raise GitError("candidate tree does not match the candidate commit")
            if evidence.staged_tree_sha != candidate_tree:
                raise GitError("candidate tree does not match accepted gate evidence")
            if candidate.get("remote") != self.config.repository.remote:
                raise GitError("candidate remote is not the repository staging remote")
            if candidate.get("remote_branch") != ctx.info.branch:
                raise GitError("candidate remote branch is not the run branch")
            if candidate.get("remote_sha") != candidate_sha:
                raise GitError("candidate record has no exact remote SHA proof")
            if not isinstance(candidate.get("pushed_at"), str) or not candidate["pushed_at"]:
                raise GitError("candidate record has no push timestamp")
            remote_tip = remote_run_branch_tip(
                ctx.info.source_repo,
                remote=self.config.repository.remote,
                branch=ctx.info.branch,
            )
            if remote_tip != candidate_sha:
                raise GitError("remote run branch does not point to the candidate commit")
        except (GitError, OSError, ValueError) as exc:
            raise PipelineFailure(
                "CANDIDATE_PUSH_FAILED",
                f"candidate remote authority is invalid: {exc}",
            ) from exc

    def _record_review(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, evidence: EvidenceBundle,
    ) -> None:
        store.update(
            status=RunStatus.REVIEWING, review=_review_payload(review),
            review_iterations=number,
        )
        self._cycle_update(
            store, number, status="reviewed",
            checks=_check_payload(evidence), reviewer_conclusion=_review_payload(review),
        )
        self._update_v2_usage(store, ctx.run_dir)

    def _request_human(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, reason: str,
    ) -> RunResult:
        """End the run with the reviewer's request as an operator task."""

        write_repair_task(ctx.run_dir, fields={
            "route": review.route.value,
            "review_summary": review.summary,
            "findings": review.findings,
            "required_fixes": review.required_fixes,
            "missing_tests": review.missing_tests,
            "existing_branch": ctx.info.branch,
            "existing_worktree": str(ctx.info.worktree),
            "run_id": ctx.run_id,
        })
        return self._v2_failed(store, ctx.run_dir, reason, None)

    def _review_repair_exhausted(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, detail: Mapping[str, Any],
    ) -> RunResult:
        """Persist exact review-correction exhaustion without a repair task."""

        return self._v2_failed(
            store, ctx.run_dir, "REVIEW_REPAIR_EXHAUSTED", None,
            {**detail, "review_summary": review.summary, "findings": review.findings},
        )

    def _publish_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        candidate: Mapping[str, Any],
    ) -> RunResult:
        """Publish the reviewed candidate of cycle *number* after its PASS."""

        # Only the exact SHA a durable reviewer PASS names is ever published.
        evidence = candidate_evidence(ctx.run_dir, number)
        review = (
            _accepted_review(review_dir(ctx.run_dir, number), evidence, candidate["commit_sha"])
            if evidence is not None and evidence.staged_tree_sha == candidate["tree_sha"]
            else None
        )
        if (
            review is None
            or review.verdict is not ReviewVerdict.PASS
            or review.route is not ReviewRoute.NONE
        ):
            raise PipelineFailure(
                "REVIEW_AUTHORITY_MISSING",
                "no accepted reviewer PASS names the candidate commit",
            )
        approved_tree = candidate["tree_sha"]
        self._cycle_update(store, number, status="approved")
        store.update(
            status=RunStatus.APPROVED, approved_tree_sha=approved_tree, current_step=None,
        )
        return self._complete_candidate_publication(
            store=store, run_dir=ctx.run_dir, info=ctx.info,
            approved_tree=approved_tree, commit_sha=candidate["commit_sha"],
            repository_reference=ctx.repository_reference, cycle=number,
        )

    def _step_profile(self, profile_id: str) -> tuple[ModelProfile, ExecutionRole]:
        """The approved implementation profile of one plan step."""

        return profile_for_role(self.config, profile_id, ExecutionRole.IMPLEMENTER), ExecutionRole.IMPLEMENTER

    def _execute_step_attempts(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        original_spec: str,
        original_plan_identity: str,
        repo: Path,
        worktree: Path,
        base_sha: str,
        branch_ref: str,
        ownership_before: GitOwnership,
        expected_tree: str,
        step: ImplementationStep,
        contract: str,
        profile_id: str,
        artifact_dir: Path,
        fallback_profile_ids: Sequence[str] = (),
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
    ) -> StepExecutionOutcome:
        """Execute a step, repairing semantic contract mismatches in-place.

        A mismatch is a recoverable transaction: its attempt artifacts are
        archived, all in-scope edits are restored to the pre-attempt tree, a
        bounded planner repair is validated, and the worker receives the
        effective contract.  The approved plan and original contract remain
        immutable evidence throughout.
        """

        effective_step = self._load_repaired_step(artifact_dir, step)
        effective_contract = contract
        repair_count = self._count_contract_repairs(artifact_dir)
        if effective_step is not step:
            effective_contract = self._read_repaired_contract(artifact_dir, effective_step)
        max_repairs = getattr(self._run_options, "max_step_contract_repairs", 0)
        recovery_generation = self._recovery_generation(store)
        active_profile_id = profile_id
        fallback_ids = tuple(fallback_profile_ids)[
            :self._run_options.recovery.max_executor_fallbacks
        ]
        fallback_index = 0
        cycle_key = f"{getattr(self, '_trace_cycle', 1):03d}:{step.id}"
        retry_key = (
            f"agent-step:{cycle_key}" if recovery_generation == 0
            else f"agent-step:r{recovery_generation}:{cycle_key}"
        )
        transient_retry_count = self._recovery_counter(store, retry_key)
        pending_transient: tuple[int, str, str, RecoveryDecision] | None = None
        pending_repair = self._pending_contract_repair(artifact_dir)
        if pending_repair is not None:
            try:
                self._set_contract_repair_state(store, step.id, repair_count)
                effective_step = self._repair_step_contract(
                    repo=repo, worktree=worktree, run_dir=run_dir,
                    artifact_dir=pending_repair, original_spec=original_spec,
                    original_plan_identity=original_plan_identity,
                    original_step=effective_step, current_contract=effective_contract,
                    mismatch="resuming durable step contract repair",
                    tree_before=candidate_tree_sha(worktree),
                    future_ownership=future_ownership,
                )
                effective_contract = self._read_repaired_contract(artifact_dir, effective_step)
            except ScopeApprovalRequired:
                delta = _read_json_artifact(pending_repair / "scope_delta.json", 64 * 1024)
                store.update(
                    status=RunStatus.WAITING_SCOPE_APPROVAL,
                    current_step=step.id,
                    scope_delta=delta if isinstance(delta, dict) else {},
                )
                raise
        while True:
            common = {
                "repo": repo, "worktree": worktree, "base_sha": base_sha,
                "branch_ref": branch_ref, "ownership_before": ownership_before,
                "expected_tree": expected_tree, "step": effective_step,
                "contract": effective_contract, "profile_id": active_profile_id,
                "artifact_dir": artifact_dir,
                "forbidden_env_names": forbidden_env_names,
                "future_ownership": future_ownership,
                "original_spec": original_spec,
            }
            try:
                outcome = self._run_step_attempt(
                    **common, initial_mismatch=None,
                    mismatch_retry_count=repair_count,
                )
                if pending_transient is not None:
                    attempt, reason, tree_before, decision = pending_transient
                    self._trace_recovery(
                        "recovery.completed", reason=reason, decision=decision,
                        attempt=attempt, tree_before=tree_before,
                        tree_after=outcome.tree_after,
                        budget_remaining=max(
                            0, self._run_options.recovery.max_transient_attempts
                            - transient_retry_count,
                        ),
                        phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                        step_id=step.id, recovered=True,
                    )
                return outcome
            except StepExecutionFailure as failure:
                atomic_write_text(artifact_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "reason": failure.reason[:120],
                    "step_id": failure.step_id,
                    "profile_id": failure.profile_id,
                    "tree_before": failure.tree_before,
                    "tree_after": failure.tree_after,
                    "status_before": list(failure.status_before or ()),
                }))
                if pending_transient is not None:
                    attempt, reason, tree_before, decision = pending_transient
                    self._trace_recovery(
                        "recovery.completed", reason=reason, decision=decision,
                        attempt=attempt, tree_before=tree_before,
                        tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                        budget_remaining=max(
                            0, self._run_options.recovery.max_transient_attempts
                            - transient_retry_count,
                        ),
                        phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                        step_id=step.id, recovered=False,
                    )
                    pending_transient = None
                retry = self._try_retry_transient_step_failure(
                    failure=failure, store=store, retry_key=retry_key,
                    retries_used=transient_retry_count, repo=repo, worktree=worktree,
                    branch_ref=branch_ref, base_sha=base_sha,
                    ownership_before=ownership_before, step=effective_step,
                    artifact_dir=artifact_dir,
                )
                if retry is not None:
                    transient_retry_count, reason, tree_before, decision = retry
                    pending_transient = (
                        transient_retry_count, reason, tree_before, decision,
                    )
                    continue
                if failure.reason == AGENT_AUTH_FAILURE:
                    failure.reason = "EXTERNAL_AUTH_REQUIRED"
                    failure.detail = "executor credentials or external authorization are required"
                    failure.step_dir = artifact_dir
                    raise
                if (
                    failure.reason in {
                        AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                        AGENT_PROTOCOL_FAILED,
                    }
                    and transient_retry_count >= self._run_options.recovery.max_transient_attempts
                    and fallback_index < len(fallback_ids)
                ):
                    fallback_index += 1
                    active_profile_id = fallback_ids[fallback_index - 1]
                    retry_key = (
                        f"agent-step:{cycle_key}:fallback:{active_profile_id}"
                        if recovery_generation == 0
                        else f"agent-step:r{recovery_generation}:{cycle_key}:fallback:{active_profile_id}"
                    )
                    transient_retry_count = self._recovery_counter(store, retry_key)
                    fallback_decision = RecoveryDecision(
                        RecoveryDisposition.FALLBACK_EXECUTOR,
                        "primary executor transient retry budget exhausted",
                        True, False,
                    )
                    self._trace_recovery(
                        "recovery.executor_selected", reason=failure.reason,
                        decision=fallback_decision, attempt=fallback_index,
                        tree_before=failure.tree_before,
                        tree_after=_safe_candidate_tree(worktree),
                        budget_remaining=max(
                            0, self._run_options.recovery.max_executor_fallbacks - fallback_index,
                        ), phase="implementation",
                        cycle=getattr(self, "_trace_cycle", 1), step_id=step.id,
                    )
                    _archive_attempt(artifact_dir)
                    atomic_write_text(artifact_dir / "executor.json", _json_text({
                        "profile_id": active_profile_id,
                        "role": "implementer",
                        "selection_source": "frozen execution fallback authority",
                    }))
                    store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
                    continue
                if failure.reason != "AGENT_CONTRACT_MISMATCH":
                    failure.step_dir = artifact_dir
                    raise
                if failure.tree_before is None or failure.mismatch is None:
                    failure.step_dir = artifact_dir
                    raise
                allowed = set(
                    (*effective_step.write_set, *effective_step.create_set, *effective_step.delete_set)
                )
                changed: list[str] = []
                if failure.tree_after is not None:
                    try:
                        changed = changed_paths_between_trees(repo, failure.tree_before, failure.tree_after)
                    except GitError:
                        changed = []
                    unexpected = [path for path in changed if path not in allowed]
                    if unexpected:
                        failure.reason = AGENT_SCOPE_VIOLATION
                        failure.detail = "worker changed paths outside scope: " + _paths_detail(unexpected)
                        failure.step_dir = artifact_dir
                        raise
                drift = self._pre_step_boundary_drift(
                    repo, worktree, ownership_before,
                    branch_ref=branch_ref, base_sha=base_sha,
                    tree_before=failure.tree_before,
                )
                if drift and not self._restore_failed_step_attempt(
                    worktree, failure.tree_before, allowed,
                ):
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; " + drift
                    failure.step_dir = artifact_dir
                    raise
                if not self._restore_failed_step_attempt(worktree, failure.tree_before, allowed):
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; failed to restore attempt tree"
                    failure.step_dir = artifact_dir
                    raise
                recovery_decision = classify_failure(
                    "AGENT_CONTRACT_MISMATCH",
                    tree_changed_in_scope=bool(changed),
                    clean_contract_mismatch=True,
                    rollback_succeeded=True,
                )
                recovery_attempt = repair_count + 1
                self._trace_recovery(
                    "recovery.classified", reason="AGENT_CONTRACT_MISMATCH",
                    decision=recovery_decision, attempt=recovery_attempt,
                    tree_before=failure.tree_before, tree_after=failure.tree_before,
                    budget_remaining=max(0, max_repairs - repair_count),
                    phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                    step_id=step.id,
                )
                if recovery_decision.disposition is not RecoveryDisposition.CONTRACT_REPAIR:
                    failure.step_dir = artifact_dir
                    raise
                _archive_attempt(artifact_dir)
                if repair_count >= max_repairs:
                    exhausted = classify_failure(
                        "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True,
                        budget_exhausted=True,
                    )
                    self._trace_recovery(
                        "recovery.exhausted", reason="AGENT_CONTRACT_MISMATCH",
                        decision=exhausted, attempt=recovery_attempt,
                        tree_before=failure.tree_before, tree_after=failure.tree_before,
                        budget_remaining=0, phase="implementation",
                        cycle=getattr(self, "_trace_cycle", 1), step_id=step.id,
                    )
                    failure.detail = (failure.detail or "worker reported a contract mismatch") + "; contract repair budget exhausted"
                    failure.tree_after = failure.tree_before
                    failure.index_tree_after = failure.tree_before
                    failure.step_dir = artifact_dir
                    raise
                repair_count += 1
                repair_dir = artifact_dir / "contract_repairs" / f"{repair_count:02d}"
                try:
                    self._set_contract_repair_state(store, step.id, repair_count)
                    self._trace_recovery(
                        "recovery.started", reason="AGENT_CONTRACT_MISMATCH",
                        decision=recovery_decision, attempt=recovery_attempt,
                        tree_before=failure.tree_before, tree_after=failure.tree_before,
                        budget_remaining=max(0, max_repairs - repair_count),
                        phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                        step_id=step.id,
                    )
                    effective_step = self._repair_step_contract(
                        repo=repo, worktree=worktree, run_dir=run_dir,
                        artifact_dir=repair_dir, original_spec=original_spec,
                        original_plan_identity=original_plan_identity,
                        original_step=effective_step, current_contract=effective_contract,
                        mismatch=failure.mismatch, tree_before=failure.tree_before,
                        future_ownership=future_ownership,
                    )
                    effective_contract = self._read_repaired_contract(artifact_dir, effective_step)
                except ScopeApprovalRequired:
                    delta = _read_json_artifact(repair_dir / "scope_delta.json", 64 * 1024)
                    store.update(
                        status=RunStatus.WAITING_SCOPE_APPROVAL,
                        current_step=step.id,
                        scope_delta=delta if isinstance(delta, dict) else {},
                    )
                    raise
                except (AgentError, GitError, OSError, V2PlanParseError) as exc:
                    raise StepExecutionFailure(
                        "AGENT_CONTRACT_MISMATCH", step.id,
                        _bounded_v2_report(f"contract repair failed: {exc}"),
                        profile_id=profile_id, tree_before=failure.tree_before,
                        tree_after=failure.tree_before, usage=failure.usage,
                        mismatch=failure.mismatch,
                        mismatch_retry_count=repair_count,
                        step_dir=artifact_dir,
                    ) from exc
                self._trace_emit(
                    "step.contract_repair.completed", phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=step.id,
                    data={"repair": repair_count, "tree_sha": failure.tree_before},
                )
                self._trace_recovery(
                    "recovery.completed", reason="AGENT_CONTRACT_MISMATCH",
                    decision=recovery_decision, attempt=recovery_attempt,
                    tree_before=failure.tree_before, tree_after=candidate_tree_sha(worktree),
                    budget_remaining=max(0, max_repairs - repair_count),
                    phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                    step_id=step.id, recovered=True,
                )
                # The next attempt starts at the exact restored tree and uses
                # no blind retry addendum.

    def _try_retry_transient_step_failure(
        self,
        *,
        failure: StepExecutionFailure,
        store: RunStateStore,
        retry_key: str,
        retries_used: int,
        repo: Path,
        worktree: Path,
        branch_ref: str,
        base_sha: str,
        ownership_before: GitOwnership,
        step: ImplementationStep,
        artifact_dir: Path,
    ) -> tuple[int, str, str, RecoveryDecision] | None:
        """Retry only transient worker failures after proving exact restoration."""

        retryable_codes = {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AGENT_PROTOCOL_FAILED,
            AGENT_AUTH_FAILURE,
        }
        reason = failure.reason
        before = failure.tree_before
        if reason not in retryable_codes:
            decision = classify_failure(
                reason,
                tree_changed_out_of_scope=reason == AGENT_SCOPE_VIOLATION,
            )
            if decision.disposition is RecoveryDisposition.HARD_STOP:
                tree = before or _safe_candidate_tree(worktree)
                self._trace_recovery(
                    "recovery.classified", reason=reason, decision=decision,
                    attempt=max(1, retries_used + 1), tree_before=before,
                    tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                    budget_remaining=0, phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
                )
            return None

        attempt = retries_used + 1
        if before is None:
            failure.reason = "RESUME_REQUIRES_OPERATOR"
            decision = classify_failure(failure.reason, rollback_succeeded=False)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=None,
                tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            failure.detail = (failure.detail or "worker failed") + "; pre-attempt tree identity is unavailable"
            failure.step_dir = artifact_dir
            return None
        if failure.tree_after is None:
            failure.reason = "RESUME_REQUIRES_OPERATOR"
            decision = classify_failure(failure.reason, rollback_succeeded=False)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before, tree_after=None,
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            failure.detail = (failure.detail or "worker failed") + "; post-attempt tree identity is unavailable"
            failure.step_dir = artifact_dir
            return None

        try:
            ownership_violations = _ownership_violations(
                ownership_before, _git_ownership(repo, worktree),
                branch_ref=branch_ref, base_sha=base_sha,
            )
        except GitError as exc:
            ownership_violations = [f"Git ownership could not be read: {exc}"]
        if ownership_violations:
            failure.reason = AGENT_GIT_VIOLATION
            failure.detail = "; ".join(ownership_violations)
            failure.step_dir = artifact_dir
            decision = classify_failure(failure.reason)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before,
                tree_after=failure.tree_after or _safe_candidate_tree(worktree),
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            return None

        allowed = set((*step.write_set, *step.create_set, *step.delete_set))
        tree_after = failure.tree_after
        tree_changed = tree_after is not None and tree_after != before
        status = _safe_status(worktree)
        if status is None:
            failure.reason = "RESUME_REQUIRES_OPERATOR"
            failure.detail = (failure.detail or "worker failed") + "; post-attempt status is unreadable"
            failure.step_dir = artifact_dir
            decision = classify_failure(failure.reason, rollback_succeeded=False)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before,
                tree_after=tree_after, budget_remaining=0,
                phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                step_id=failure.step_id,
            )
            return None
        status_changed = bool(status)
        tree_changed = tree_changed or status_changed
        if tree_changed:
            try:
                stage_all(worktree)
                tree_after = index_tree_sha(worktree)
                failure.tree_after = tree_after
                changed = changed_paths_between_trees(repo, before, tree_after)
            except (GitError, OSError, ValueError) as exc:
                failure.reason = "RESUME_REQUIRES_OPERATOR"
                failure.detail = f"failed attempt tree could not be frozen: {type(exc).__name__}"
                failure.tree_after = _safe_candidate_tree(worktree)
                failure.step_dir = artifact_dir
                decision = classify_failure(failure.reason, rollback_succeeded=False)
                self._trace_recovery(
                    "recovery.classified", reason=failure.reason, decision=decision,
                    attempt=attempt, tree_before=before, tree_after=failure.tree_after,
                    budget_remaining=0, phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
                )
                return None
            unexpected = [path for path in changed if path not in allowed]
            if unexpected:
                failure.reason = AGENT_SCOPE_VIOLATION
                failure.detail = "worker changed paths outside scope: " + _paths_detail(unexpected)
                failure.step_dir = artifact_dir
                decision = classify_failure(failure.reason, tree_changed_out_of_scope=True)
                self._trace_recovery(
                    "recovery.classified", reason=failure.reason, decision=decision,
                    attempt=attempt, tree_before=before, tree_after=tree_after,
                    budget_remaining=0, phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
                )
                return None
            try:
                security_failures = scan_staged_security(
                    worktree, secrets=self._secrets,
                )
            except (GitError, OSError, ValueError) as exc:
                failure.reason = "STAGED_BLOB_SCAN_FAILED"
                failure.detail = f"staged changes could not be scanned: {type(exc).__name__}"
                failure.tree_after = _safe_candidate_tree(worktree)
                failure.step_dir = artifact_dir
                decision = classify_failure(failure.reason)
                self._trace_recovery(
                    "recovery.classified", reason=failure.reason, decision=decision,
                    attempt=attempt, tree_before=before, tree_after=failure.tree_after,
                    budget_remaining=0, phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
                )
                return None
            if security_failures:
                failure.reason = security_failures[0].split(":", 1)[0]
                failure.detail = "; ".join(security_failures)
                failure.step_dir = artifact_dir
                decision = classify_failure(failure.reason)
                self._trace_recovery(
                    "recovery.classified", reason=failure.reason, decision=decision,
                    attempt=attempt, tree_before=before, tree_after=tree_after,
                    budget_remaining=0, phase="implementation",
                    cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
                )
                return None
        if tree_changed and not self._restore_failed_step_attempt(worktree, before, allowed):
            failure.reason = "RESUME_REQUIRES_OPERATOR"
            failure.detail = (failure.detail or "worker failed") + "; rollback did not restore the pre-attempt tree"
            failure.tree_after = _safe_candidate_tree(worktree)
            failure.step_dir = artifact_dir
            decision = classify_failure(failure.reason, rollback_succeeded=False)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before, tree_after=failure.tree_after,
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            return None

        try:
            restored_status = status_porcelain(worktree)
            exact_restoration = (
                candidate_tree_sha(worktree) == before
                and index_tree_sha(worktree) == before
                and restored_status == failure.status_before
                and not _status_has_unstaged_or_untracked(restored_status)
            )
        except (GitError, OSError):
            exact_restoration = False
        if not exact_restoration:
            failure.reason = "RESUME_REQUIRES_OPERATOR"
            failure.detail = (failure.detail or "worker failed") + "; retry boundary is not exact"
            failure.tree_after = _safe_candidate_tree(worktree)
            failure.step_dir = artifact_dir
            decision = classify_failure(failure.reason, rollback_succeeded=False)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before, tree_after=failure.tree_after,
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            return None

        try:
            ownership_after = _ownership_violations(
                ownership_before, _git_ownership(repo, worktree),
                branch_ref=branch_ref, base_sha=base_sha,
            )
        except GitError as exc:
            ownership_after = [f"Git ownership could not be read: {exc}"]
        if ownership_after:
            failure.reason = AGENT_GIT_VIOLATION
            failure.detail = "; ".join(ownership_after)
            failure.step_dir = artifact_dir
            decision = classify_failure(failure.reason)
            self._trace_recovery(
                "recovery.classified", reason=failure.reason, decision=decision,
                attempt=attempt, tree_before=before,
                tree_after=_safe_candidate_tree(worktree), budget_remaining=0,
                phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
                step_id=failure.step_id,
            )
            return None

        failure.tree_after = before
        failure.index_tree_after = before
        decision = classify_failure(
            reason, tree_changed_in_scope=tree_changed,
            rollback_succeeded=True,
        )
        budget = self._run_options.recovery.max_transient_attempts
        self._trace_recovery(
            "recovery.classified", reason=reason, decision=decision,
            attempt=attempt, tree_before=before, tree_after=before,
            budget_remaining=max(0, budget - retries_used),
            phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
            step_id=failure.step_id,
        )
        if decision.disposition is RecoveryDisposition.WAIT_EXTERNAL:
            failure.step_dir = artifact_dir
            return None
        if decision.disposition not in {
            RecoveryDisposition.RETRY_SAME,
            RecoveryDisposition.RETRY_AFTER_ROLLBACK,
        }:
            failure.step_dir = artifact_dir
            return None
        if retries_used >= budget:
            exhausted = classify_failure(reason, budget_exhausted=True)
            self._trace_recovery(
                "recovery.exhausted", reason=reason, decision=exhausted,
                attempt=attempt, tree_before=before, tree_after=before,
                budget_remaining=0, phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1), step_id=failure.step_id,
            )
            failure.detail = (failure.detail or "transient agent failure") + "; transient attempt budget exhausted"
            failure.step_dir = artifact_dir
            return None

        retries_used = self._consume_recovery_counter(store, retry_key)
        store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
        self._trace_recovery(
            "recovery.started", reason=reason, decision=decision,
            attempt=attempt, tree_before=before, tree_after=before,
            budget_remaining=max(0, budget - retries_used),
            phase="implementation", cycle=getattr(self, "_trace_cycle", 1),
            step_id=step.id,
        )
        _archive_attempt(artifact_dir)
        return retries_used, reason, before, decision

    @staticmethod
    def _recovery_generation(store: RunStateStore) -> int:
        resume = store.load().get("resume")
        attempts = resume.get("attempts", 0) if isinstance(resume, dict) else 0
        return attempts if isinstance(attempts, int) and not isinstance(attempts, bool) else 0

    @staticmethod
    def _count_contract_repairs(artifact_dir: Path) -> int:
        root = artifact_dir / "contract_repairs"
        if not root.is_dir():
            return 0
        numbers = []
        for path in root.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            validation = _read_json_artifact(path / "validation.json", 64 * 1024)
            if isinstance(validation, dict) and validation.get("status") in {
                "planner_validated", "validated",
            }:
                numbers.append(int(path.name))
        return max(numbers, default=0)

    @staticmethod
    def _pending_contract_repair(artifact_dir: Path) -> Path | None:
        root = artifact_dir / "contract_repairs"
        if not root.is_dir():
            return None
        candidates = []
        for path in root.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            validation = _read_json_artifact(path / "validation.json", 64 * 1024)
            if isinstance(validation, dict) and validation.get("status") == "planner_validated":
                candidates.append(path)
        return max(candidates, key=lambda path: int(path.name), default=None)

    def _load_repaired_step(self, artifact_dir: Path, original: ImplementationStep) -> ImplementationStep:
        root = artifact_dir / "contract_repairs"
        if not root.is_dir():
            return original
        candidates = sorted(
            (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for directory in reversed(candidates):
            validation = _read_json_artifact(directory / "validation.json", 64 * 1024)
            contract_path = directory / "contract.md"
            if not isinstance(validation, dict) or validation.get("status") != "validated" or not contract_path.is_file():
                continue
            try:
                repaired = parse_step_contract_repair(
                    contract_path.read_text(encoding="utf-8"),
                    max_read_paths_per_step=self.config.planning.max_read_paths_per_step,
                )
            except (OSError, UnicodeError, V2PlanParseError):
                continue
            if repaired.id == original.id:
                return repaired
        return original

    def _read_repaired_contract(self, artifact_dir: Path, step: ImplementationStep) -> str:
        root = artifact_dir / "contract_repairs"
        candidates = sorted(
            (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        ) if root.is_dir() else []
        for directory in reversed(candidates):
            path = directory / "contract.md"
            validation = _read_json_artifact(directory / "validation.json", 64 * 1024)
            if path.is_file() and isinstance(validation, dict) and validation.get("status") == "validated":
                try:
                    parsed = parse_step_contract_repair(
                        path.read_text(encoding="utf-8"),
                        max_read_paths_per_step=self.config.planning.max_read_paths_per_step,
                    )
                except (OSError, UnicodeError, V2PlanParseError):
                    continue
                if parsed.id == step.id:
                    return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _restore_failed_step_attempt(
        worktree: Path, tree_before: str, allowed: set[str],
    ) -> bool:
        try:
            restore_paths_from_tree(worktree, tree_before, sorted(allowed))
            stage_all(worktree)
            return (
                candidate_tree_sha(worktree) == tree_before
                and index_tree_sha(worktree) == tree_before
                and not _status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except (GitError, OSError):
            return False

    def _set_contract_repair_state(
        self, store: RunStateStore, step_id: str, repair_count: int,
    ) -> None:
        store.update(
            status=RunStatus.CONTRACT_REPAIRING,
            current_step=step_id,
            contract_repair={"status": "running", "step_id": step_id, "attempt": repair_count},
        )

    def _repair_step_contract(
        self, *, repo: Path, worktree: Path, run_dir: Path,
        artifact_dir: Path, original_spec: str, original_plan_identity: str,
        original_step: ImplementationStep, current_contract: str,
        mismatch: str, tree_before: str,
        future_ownership: Mapping[str, tuple[str, ...]] | None,
    ) -> ImplementationStep:
        """Run and validate one durable StepContractRepairPlanner transaction."""

        planner_profile = profile_for_role(
            self.config, self._run_options.planner_profile, ExecutionRole.PLANNER
        )
        evidence_parts = [
            "TREE SHA: " + tree_before,
            "STATUS: " + "; ".join(status_porcelain(worktree)[:20]),
        ]
        for path in read_set_paths(original_step.read_set):
            target = worktree / path
            try:
                data = target.read_bytes()[:8192]
                evidence_parts.append(
                    f"PATH {path}\n" + data.decode("utf-8", errors="replace")
                )
            except (OSError, UnicodeError):
                evidence_parts.append(f"PATH {path}\n<unavailable>")
        planner = StepContractRepairPlanner(
            self._planner_client or _chat_client(
                build_llm_endpoint(planner_profile), self._runtime_environment
            ),
            max_read_paths_per_step=self.config.planning.max_read_paths_per_step,
        )
        read_set = "\n".join(f"- {item}" for item in original_step.read_set)
        future = _json_text(future_ownership or {})
        repaired = planner.repair(
            original_spec=original_spec, current_tree_sha=tree_before,
            original_plan_identity=original_plan_identity,
            current_contract=current_contract,
            mismatch_explanation=_bounded_v2_report(mismatch),
            read_set=read_set,
            write_set="\n".join(f"- {item}" for item in original_step.write_set) or "NONE",
            create_set="\n".join(f"- {item}" for item in original_step.create_set) or "NONE",
            delete_set="\n".join(f"- {item}" for item in original_step.delete_set) or "NONE",
            future_ownership=future,
            repository_evidence="\n\n".join(evidence_parts),
            artifacts_dir=artifact_dir,
        )
        if (
            repaired.id != original_step.id
            or repaired.title != original_step.title
            or repaired.execution_class != original_step.execution_class
            or repaired.depends_on != original_step.depends_on
        ):
            raise V2PlanParseError("contract repair changed step identity or dependency")
        original_mutable = set((*original_step.write_set, *original_step.create_set, *original_step.delete_set))
        repaired_mutable = set((*repaired.write_set, *repaired.create_set, *repaired.delete_set))
        removed = original_mutable - repaired_mutable
        added = repaired_mutable - original_mutable
        if removed:
            raise V2PlanParseError("contract repair removed approved mutable paths")
        if added:
            policy = self._effective_repair_scope
            if len(added) > policy.max_added_paths or policy.policy == "deny-expansion":
                raise V2PlanParseError("contract repair requested a disallowed mutable-scope expansion")
            if policy.policy == "require-approval":
                delta = {
                    "schema_version": 1, "step_id": original_step.id,
                    "added_paths": sorted(added), "tree_sha": tree_before,
                    "repair_number": int(artifact_dir.name),
                }
                delta_path = artifact_dir / "scope_delta.json"
                atomic_write_text(delta_path, _json_text(delta))
                approval = read_scope_approval(
                    artifact_dir,
                    expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
                )
                if approval is None:
                    raise ScopeApprovalRequired()
                if approval.decision is not ApprovalDecision.APPROVE:
                    raise PipelineFailure("HUMAN_REQUIRED", "contract repair scope rejected")
        drift = self._step_contract_drift(repo, tree_before, tree_before, repaired)
        if drift:
            raise V2PlanParseError("repaired contract is not executable on current tree: " + drift)
        validation_path = artifact_dir / "validation.json"
        validation = _read_json_artifact(validation_path, 64 * 1024)
        if isinstance(validation, dict):
            validation.update({
                "status": "validated", "added_mutable_paths": sorted(added),
                "removed_mutable_paths": sorted(removed),
                "original_step_contract_sha256": hashlib.sha256(current_contract.encode("utf-8")).hexdigest(),
                "repaired_contract_sha256": hashlib.sha256(
                    (artifact_dir / "contract.md").read_bytes()
                ).hexdigest(),
            })
            atomic_write_text(validation_path, _json_text(validation))
        return repaired

    def _pre_step_boundary_drift(
        self, repo: Path, worktree: Path, ownership_before: GitOwnership, *,
        branch_ref: str, base_sha: str, tree_before: str,
    ) -> str | None:
        """Why the repository is no longer exactly in its pre-step state."""

        try:
            if candidate_tree_sha(worktree) != tree_before:
                return "the candidate tree is no longer the pre-step tree"
            if index_tree_sha(worktree) != tree_before:
                return "the index is no longer the pre-step index"
            if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
                return "the worktree has unstaged or untracked modifications"
        except GitError as exc:
            return f"Git state is unreadable: {exc}"
        violations = _ownership_violations(
            ownership_before, _git_ownership(repo, worktree),
            branch_ref=branch_ref, base_sha=base_sha,
        )
        return "; ".join(violations) or None

    def _run_step_attempt(
        self,
        *,
        repo: Path,
        worktree: Path,
        base_sha: str,
        branch_ref: str,
        ownership_before: GitOwnership,
        expected_tree: str,
        step: ImplementationStep,
        contract: str,
        profile_id: str,
        artifact_dir: Path,
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
        original_spec: str = "",
        initial_mismatch: str | None = None,
        mismatch_retry_count: int = 0,
    ) -> StepExecutionOutcome:
        """The single authoritative execution of one step, in any cycle.

        Gates run in a fixed order and every failure raises
        :class:`StepExecutionFailure`; the caller owns the run status.  The
        worker's final report is data only and never drives a decision.
        """

        step_id = step.id
        # The retry mode of this invocation, recorded with every failure it
        # can raise so the durable step record says which attempt failed.
        retry_mode: dict[str, Any] = (
            {
                "mismatch_retry_count": mismatch_retry_count,
                "initial_mismatch": _bounded_v2_report(initial_mismatch or "") or None,
            }
            if mismatch_retry_count else {}
        )
        # 1-2. The exact tree the worker will receive, and the contract's Git
        # preconditions on it.
        tree_before = candidate_tree_sha(worktree)
        drift = self._step_contract_drift(repo, tree_before, expected_tree, step)
        if drift:
            raise StepExecutionFailure(
                "STEP_CONTRACT_DRIFT", step_id, drift,
                profile_id=profile_id, tree_before=tree_before, **retry_mode,
            )
        # The complete Git boundary a no-op mismatch must leave untouched.
        # Accumulated modifications from the earlier steps are legitimate, so
        # the gate is "unchanged", never "empty".
        index_before = index_tree_sha(worktree)
        status_before = status_porcelain(worktree)
        # 3-4. Approved profile and isolated environment.
        profile, step_role = self._step_profile(profile_id)
        executor = self._executor_for_profile(
            profile.id,
            step_role,
            forbidden_env_names=forbidden_env_names,
        )
        selected_executor = self._trace_selected_profile(profile.id, step_role, step_id=step_id)
        selection_source = "primary execution authority"
        frozen_selection = getattr(self, "_last_selection", None)
        if frozen_selection is not None and step_id is not None:
            for selected_step in frozen_selection.steps:
                if selected_step.step_id == step_id and any(
                    fallback.profile_id == profile.id for fallback in selected_step.fallbacks
                ):
                    selection_source = "frozen execution fallback authority"
                    break
        atomic_write_text(artifact_dir / "executor.json", _json_text({
            "profile_id": profile.id,
            "role": step_role.value,
            "config_sha256": getattr(selected_executor, "config_sha256", None),
            "selection_source": selection_source,
        }))
        # 5. One fresh worker process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
        # The selected adapter owns the single execution call.
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompt_payload = build_implementer_payload(
                original_spec=original_spec,
                step_identity=f"{step.id}\nTITLE\n{step.title}",
                step_title=step.title,
                step_objective=step.objective,
                step_invariants=step.forbidden,
                read_set="\n".join(step.read_set),
                write_set="\n".join(step.write_set) or "NONE",
                create_set="\n".join(step.create_set) or "NONE",
                delete_set="\n".join(step.delete_set) or "NONE",
                mutable_scope=_json_text({
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                }),
                repository_instructions=step.instructions,
                verify_instructions=step.verify,
                instructions=step.instructions,
                verify_contract=step.verify,
                forbidden_contract=step.forbidden,
                budget_bytes=self.config.prompt_budget.implementer_max_bytes,
            )
            request_prompt = prompt_payload.rendered
            write_prompt_diagnostics(artifact_dir, prompt_payload)
            trace_started_at = self._trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = selected_executor
            self._trace_emit(
                "step.started",
                phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1),
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
                    "tree_before": tree_before,
                    "session": self._trace_session(
                        profile=profile,
                        selected=trace_selected,
                        role=step_role,
                        prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before,
                    ),
                },
            )
            result = executor.run(
                AgentRunRequest(
                    role=step_role,
                    profile_id=profile.id,
                    prompt=request_prompt,
                    worktree=worktree,
                    artifact_dir=artifact_dir,
                    mutable_paths=tuple(
                        sorted({*step.write_set, *step.create_set, *step.delete_set})
                    ),
                    prompt_mode="raw",
                    contract=contract,
                    retry_addendum=None,
                )
            )
        except AgentScopeError as exc:
            self._trace_emit(
                "step.agent.completed",
                phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1),
                step_id=step_id,
                data={
                    "attempt": mismatch_retry_count + 1,
                    "status": "failed",
                    "session": self._trace_session(
                        profile=profile,
                        selected=trace_selected,
                        role=step_role,
                        prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before,
                        exit_reason=getattr(exc, "code", type(exc).__name__),
                    ),
                },
            )
            self._redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id, redact(str(exc), self._secrets),
                profile_id=profile.id, tree_before=tree_before, **retry_mode,
            ) from None
        except AgentError as exc:
            reason = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
            tree_after = _safe_candidate_tree(worktree)
            _record_failure_tree(artifact_dir, worktree)
            atomic_write_text(artifact_dir / "failure.json", _json_text({
                "schema_version": 1,
                "reason": str(reason)[:120],
                "profile_id": profile.id,
                "tree_before": tree_before,
                "tree_after": tree_after,
                "mutable_scope": sorted({
                    *step.write_set, *step.create_set, *step.delete_set,
                }),
            }))
            self._redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                reason, step_id, redact(str(exc), self._secrets),
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, status_before=status_before,
                **retry_mode,
            ) from None
        # 6. Complete and redact the durable artifacts.
        self._ensure_step_artifacts(artifact_dir, result)
        self._redact_step_artifacts(artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self._secrets),
            stderr_tail=redact(result.stderr_tail, self._secrets),
        )
        self._trace_emit(
            "step.agent.completed",
            phase="implementation",
            cycle=getattr(self, "_trace_cycle", 1),
            step_id=step_id,
            data={
                "attempt": mismatch_retry_count + 1,
                "status": result.status,
                "session": self._trace_session(
                    profile=profile,
                    selected=trace_selected,
                    role=step_role,
                    prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                    started_at=trace_started_at,
                    started_mono=trace_started_mono,
                    tree_before=tree_before,
                    result=result,
                ),
            },
        )
        usage = normalize_usage(result.usage)
        # Advisory, argument-free context diagnostics (never a gate).
        try:
            write_token_diagnostics(artifact_dir, usage, worktree=worktree)
        except (OSError, ResultArtifactError):
            pass
        failed = {
            "profile_id": profile.id, "tree_before": tree_before, "usage": usage,
            "status_before": status_before,
            **retry_mode,
        }
        # 7. Authentication classification from fixed markers only.
        auth_failure = result.backend_reason == "AGENT_AUTH_FAILURE"
        # Capture ownership before interpreting the worker's structural report.
        # A clean mismatch is allowed to defer only when the complete Git
        # boundary is untouched.
        ownership_after = _git_ownership(repo, worktree)
        ownership_violations = _ownership_violations(
            ownership_before, ownership_after, branch_ref=branch_ref, base_sha=base_sha
        )
        # A structural mismatch is the only worker report that has protocol
        # meaning.  It is checked before any staging and its explanation stays
        # bounded and non-authoritative.
        mismatch = None
        if not result.timed_out and result.exit_code == 0:
            mismatch = contract_mismatch_explanation(result.final_message)
            if mismatch is None:
                # A successful, completely clean no-op has the same semantic
                # meaning as a worker-declared structural mismatch.  Feed it
                # through the existing bounded retry path; any boundary drift
                # remains fail-closed below.
                no_change_tree = _safe_candidate_tree(worktree)
                no_change_index = _safe_index_tree(worktree)
                no_change_status = _safe_status(worktree)
                if (
                    no_change_tree == tree_before
                    and index_before == tree_before
                    and no_change_index == tree_before
                    and not _status_has_unstaged_or_untracked(status_before)
                    and no_change_status == status_before
                    and not ownership_violations
                ):
                    mismatch = (
                        _BOUNDED_NO_CHANGE_MISMATCH
                        if mismatch_retry_count else _SYNTHETIC_NO_CHANGE_MISMATCH
                    )
        if mismatch is not None:
            # Freeze even unstaged worker edits into a durable candidate tree
            # before the repair transaction.  This makes a crash at the
            # mismatch boundary recoverable by the normal resume validator.
            try:
                stage_all(worktree)
            except GitError as exc:
                raise StepExecutionFailure(
                    AGENT_RUNTIME_FAILED, step_id, "could not freeze mismatch tree",
                    **failed, tree_after=_safe_candidate_tree(worktree),
                ) from exc
            tree_after = _safe_candidate_tree(worktree)
            index_after = _safe_index_tree(worktree)
            status_after = _safe_status(worktree)
            if ownership_violations:
                _record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    "AGENT_GIT_VIOLATION", step_id,
                    "; ".join(ownership_violations),
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=tree_after, usage=usage,
                    mismatch=_bounded_v2_report(mismatch),
                    mismatch_retry_count=mismatch_retry_count,
                )
            # Clean means "the worker changed nothing": the candidate tree,
            # the index, the porcelain status and Git ownership are all exactly
            # what this step received.  It is deliberately not "git status is
            # empty": the cumulative modifications of the earlier approved
            # steps are legitimate and untracked-but-ignored files are never a
            # gate here.
            clean = (
                bool(mismatch.strip())
                and tree_after == tree_before
                and index_after == index_before
                and status_after == status_before
                and not ownership_violations
            )
            changed = []
            if tree_after is not None:
                try:
                    changed = changed_paths_between_trees(repo, tree_before, tree_after)
                except GitError:
                    changed = []
            allowed = {*step.write_set, *step.create_set, *step.delete_set}
            unexpected = [path for path in changed if path not in allowed]
            if unexpected:
                _record_failure_tree(artifact_dir, worktree)
                raise StepExecutionFailure(
                    AGENT_SCOPE_VIOLATION, step_id,
                    "worker changed paths outside scope: " + _paths_detail(unexpected),
                    profile_id=profile.id, tree_before=tree_before,
                    tree_after=tree_after, usage=usage,
                    mismatch=_bounded_v2_report(mismatch),
                    index_tree_after=index_after,
                    mismatch_retry_count=mismatch_retry_count,
                )
            atomic_write_text(artifact_dir / "step.json", _json_text({
                "id": step_id, "status": "FAILED", "reason": "AGENT_CONTRACT_MISMATCH",
                "profile_id": profile.id, "tree_before": tree_before,
                "tree_after": tree_after, "index_tree_after": index_after,
                "changed_paths": list(changed),
                "mismatch": _bounded_v2_report(mismatch), "usage": usage,
            }))
            _record_failure_tree(artifact_dir, worktree)
            details = []
            if mismatch:
                details.append(_bounded_v2_report(mismatch))
            if tree_after is None:
                details.append("failure tree could not be read")
            residual = _new_status_lines(status_before, status_after)
            raise StepExecutionFailure(
                "AGENT_CONTRACT_MISMATCH", step_id,
                "; ".join(details) or "worker reported a contract mismatch",
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, usage=usage,
                mismatch=_bounded_v2_report(mismatch),
                clean_contract_mismatch=clean,
                mismatch_retry_count=mismatch_retry_count,
                initial_mismatch=(
                    _bounded_v2_report(initial_mismatch)
                    if mismatch_retry_count and initial_mismatch else None
                ),
                index_tree_after=index_after,
            )
        # 8-9. Git ownership: HEAD, branch, branches and worktrees.
        if ownership_after.head != base_sha:
            raise StepExecutionFailure("AGENT_GIT_VIOLATION", step_id, "worktree HEAD changed", **failed)
        if ownership_violations:
            raise StepExecutionFailure(
                "AGENT_GIT_VIOLATION", step_id, "; ".join(ownership_violations), **failed
            )
        if auth_failure:
            raise StepExecutionFailure(
                AGENT_AUTH_FAILURE, step_id, "worker authentication failed", **failed,
                tree_after=_safe_candidate_tree(worktree),
            )
        # 10-11. Process outcome.  The tree left behind is recorded so that a
        # resume can tell a clean retry from partial worker changes.
        if result.timed_out or result.exit_reason == AGENT_TIMEOUT:
            raise StepExecutionFailure(
                AGENT_TIMEOUT,
                step_id, **failed, tree_after=_safe_candidate_tree(worktree)
            )
        if result.exit_code not in (0, None) or result.exit_reason in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            reason = result.exit_reason or AGENT_RUNTIME_FAILED
            raise StepExecutionFailure(
                reason, step_id, f"exit status {result.exit_code}", **failed,
                tree_after=_safe_candidate_tree(worktree),
            )
        # 12-14. Freeze the candidate; a step must change it.
        stage_all(worktree)
        tree_after = index_tree_sha(worktree)
        if tree_after == tree_before:
            raise StepExecutionFailure("AGENT_NO_CHANGE", step_id, **failed)
        # 15-16. Git, not the prompt, is the scope barrier: every changed path
        # must be authorized by this step's WRITE, CREATE or DELETE set.
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        allowed = {*step.write_set, *step.create_set, *step.delete_set}
        unexpected = [path for path in changed_paths if path not in allowed]
        if unexpected:
            raise StepExecutionFailure(
                AGENT_SCOPE_VIOLATION, step_id,
                f"unexpected={_paths_detail(unexpected)}", **failed, tree_after=tree_after,
            )
        # 17-18. Durable step record, then the outcome.  A deferred verify
        # dependency is recorded as data for the reviser and the reviewer; it never
        # relaxes a deterministic gate.
        deferred_verify = _bounded_v2_report(
            deferred_verify_dependency(result.final_message) or ""
        )
        atomic_write_text(artifact_dir / "step.json", _json_text({
            "id": step_id, "status": "COMPLETED", "profile_id": profile.id,
            "tree_before": tree_before, "tree_after": tree_after,
            "changed_paths": list(changed_paths),
            **({"mismatch_retry_count": mismatch_retry_count}
               if mismatch_retry_count else {}),
            **({"initial_mismatch": _bounded_v2_report(initial_mismatch)}
               if mismatch_retry_count and initial_mismatch else {}),
            **({"deferred_verify": deferred_verify} if deferred_verify else {}),
            "usage": usage,
        }))
        return StepExecutionOutcome(
            step_id=step_id,
            profile_id=profile.id,
            tree_before=tree_before,
            tree_after=tree_after,
            changed_paths=tuple(changed_paths),
            usage=usage,
            final_report=result.final_message,
            deferred_verify=deferred_verify,
            mismatch_retry_count=mismatch_retry_count,
        )

    def _step_failed(
        self, store: RunStateStore, run_dir: Path, failure: StepExecutionFailure,
    ) -> RunResult:
        self._trace_emit(
            "step.failed",
            phase="implementation",
            cycle=getattr(self, "_trace_cycle", 1),
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
        return self._v2_failed(
            store, run_dir, failure.reason, failure.step_id, failure.detail,
            usage=failure.usage, profile_id=failure.profile_id,
            tree_before=failure.tree_before, tree_after=failure.tree_after,
            mismatch=failure.mismatch,
            mismatch_clean=failure.clean_contract_mismatch,
            mismatch_retry_count=failure.mismatch_retry_count,
            initial_mismatch=failure.initial_mismatch,
            index_tree_after=failure.index_tree_after,
            step_dir=failure.step_dir,
        )

    def _accept_v2_step_tree(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: WorktreeInfo,
        step: ImplementationStep,
        outcome: StepExecutionOutcome,
        parent_sha: str,
        future_step_ids: Sequence[str],
        run_id: str,
        step_dir: Path,
    ) -> str | None:
        """Run the reusable safety gate and accept one normal step tree.

        Worker attempts never call this method until their structural and
        scope gates have passed.  A red/failed attempt therefore remains an
        artifact tree only.  The explicit deferred contract is the sole
        exception to a passed verification status.
        """

        if current_head(info.worktree) != parent_sha:
            raise CommitSafetyError("step parent HEAD changed before acceptance")
        reported_verification = getattr(outcome, "verification_status", None)
        if isinstance(reported_verification, str) and reported_verification.casefold() in {
            "failed", "fail", "red",
        }:
            self._trace_emit(
                "step.verification.completed",
                phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1),
                step_id=step.id,
                data={
                    "status": "failed",
                    "tree_before": outcome.tree_before,
                    "tree_after": outcome.tree_after,
                    "deferred": False,
                },
            )
            raise CommitSafetyError("step VERIFY did not pass")
        verify_failed = re.search(
            r"^\s*(?:VERIFY|VERIFICATION)\s*(?::|=)\s*(?:FAIL|FAILED|RED)\b",
            outcome.final_report,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        environment_verify_failure = bool(re.search(
            r"^\s*(?:VERIFY|VERIFICATION)\s*[:=].*\b(?:FAIL|FAILED|RED)\b\s*\(\s*environment\s*:",
            outcome.final_report,
            flags=re.IGNORECASE | re.MULTILINE,
        ))
        if verify_failed and not environment_verify_failure:
            self._trace_emit(
                "step.verification.completed",
                phase="implementation",
                cycle=getattr(self, "_trace_cycle", 1),
                step_id=step.id,
                data={
                    "status": "failed",
                    "tree_before": outcome.tree_before,
                    "tree_after": outcome.tree_after,
                    "deferred": False,
                },
            )
            raise CommitSafetyError("step VERIFY did not pass")
        verification_status = "passed"
        deferred = None
        if getattr(outcome, "deferred_verify", "") or getattr(outcome, "status", "") == "DEFERRED_CONTRACT_MISMATCH":
            deferred = parse_deferred_verification(
                outcome.final_report,
                current_step_id=step.id,
                future_step_ids=future_step_ids,
            )
            if deferred is None:
                raise CommitSafetyError(
                    "a deferred step must provide the explicit DEFERRED VERIFY DEPENDENCY contract"
                )
            verification_status = "deferred"

        self._trace_emit(
            "step.verification.completed",
            phase="implementation",
            cycle=getattr(self, "_trace_cycle", 1),
            step_id=step.id,
            data={
                "status": verification_status,
                "tree_before": outcome.tree_before,
                "tree_after": outcome.tree_after,
                "deferred": deferred is not None,
            },
        )

        # A no-change deferred mismatch is a traceable worker outcome, not a
        # new Git state.  There is no legal empty commit; the later candidate
        # gate still sees the durable mismatch artifact.
        if outcome.tree_after == outcome.tree_before:
            state = store.load()
            deferred_records = list(state.get("deferred_verifications") or [])
            deferred_records.append({
                "step_id": step.id,
                "verification_status": "deferred",
                "tree_before": outcome.tree_before,
                "tree_after": outcome.tree_after,
                "changed_paths": [],
                "deferred_reason": deferred.reason,
                "dependent_step_ids": list(deferred.dependent_step_ids),
                "deferred_verify_command_or_contract": deferred.command_or_contract,
                "commit_sha": None,
            })
            parents = commit_parents(info.worktree, parent_sha)
            expected_parent = parents[0] if parents else parent_sha
            store.update(
                status=RunStatus.IMPLEMENTING,
                deferred_verifications=deferred_records,
                expected_head_sha=parent_sha,
                expected_parent_sha=expected_parent,
                expected_tree_sha=outcome.tree_after,
                next_step_id=(future_step_ids[0] if future_step_ids else None),
            )
            return None

        gate = commit_safety_gate(
            info.worktree,
            tree_sha=outcome.tree_after,
            parent_sha=parent_sha,
            mutable_scope=(*step.write_set, *step.create_set, *step.delete_set),
            verification_status=verification_status,
            deferred_reason=deferred.reason if deferred is not None else None,
            dependent_step_ids=deferred.dependent_step_ids if deferred is not None else (),
            deferred_command_or_contract=(
                deferred.command_or_contract if deferred is not None else None
            ),
            secrets=self._secrets,
            # v2 deliberately treats a large diff as bounded review
            # evidence; the canonical blob-size and binary policies still
            # apply in the shared scanner.
            max_diff_bytes=None,
        )
        diff_path = step_dir / "diff.patch"
        atomic_write_text(diff_path, redact(staged_diff(info.worktree), self._secrets))
        commit_sha = commit_step_tree(
            info.worktree,
            tree_sha=gate.tree_sha,
            parent_sha=gate.parent_sha,
            step_id=step.id,
            step_title=step.title,
            body=f"MetaHarness-Run: {run_id}",
        )
        record = accepted_step_record(
            step_id=step.id,
            verification_status=verification_status,
            parent_sha=parent_sha,
            commit_sha=commit_sha,
            tree_before=outcome.tree_before,
            tree_after=outcome.tree_after,
            changed_paths=gate.changed_paths,
            deferred=deferred,
        )
        step_path = step_dir / "step.json"
        step_payload = _read_json_artifact(step_path)
        if not isinstance(step_payload, dict):
            step_payload = {"id": step.id}
        step_payload.update(record)
        step_payload["verification_status"] = verification_status
        atomic_write_text(step_path, _json_text(step_payload))

        state = store.load()
        accepted_steps = list(state.get("accepted_steps") or [])
        accepted_commits = list(state.get("accepted_commits") or [])
        accepted_steps.append(record)
        accepted_commits.append(record)
        chain = list(accepted_chain_records(run_dir))
        chain.append(record)
        atomic_write_text(run_dir / "accepted-chain.json", _json_text({"commits": chain}))
        following = future_step_ids[0] if future_step_ids else None
        store.update(
            status=RunStatus.IMPLEMENTING,
            accepted_steps=accepted_steps,
            accepted_commits=accepted_commits,
            expected_head_sha=commit_sha,
            expected_parent_sha=parent_sha,
            expected_tree_sha=outcome.tree_after,
            next_step_id=following,
        )
        self._trace_emit(
            "step.committed",
            phase="implementation",
            cycle=getattr(self, "_trace_cycle", 1),
            step_id=step.id,
            data={
                "parent_sha": parent_sha,
                "commit_sha": commit_sha,
                "tree_sha": outcome.tree_after,
                "changed_paths": list(gate.changed_paths),
                **self._trace_diff_reference(diff_path),
            },
        )
        return commit_sha

    def _authorize_candidate_tree(
        self, evidence: EvidenceBundle, worktree: Path, parent_sha: str, branch_ref: str,
    ) -> str:
        """Authorize the immutable candidate tree before semantic review."""

        if not evidence.deterministic_passed or evidence.failures or not evidence.staged_tree_sha:
            raise CommitBoundaryError("deterministic gate did not pass for candidate commit")
        if symbolic_head(worktree) != branch_ref or current_head(worktree) != parent_sha:
            raise CommitBoundaryError("worktree HEAD changed before candidate commit")
        candidate = evidence.staged_tree_sha
        if index_tree_sha(worktree) != candidate or candidate_tree_sha(worktree) != candidate:
            raise CommitBoundaryError("candidate tree changed before candidate commit")
        if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
            raise CommitBoundaryError("worktree has changes before candidate commit")
        return candidate

    def _push_candidate(
        self, *, run_dir: Path, info: WorktreeInfo, cycle: int, candidate: dict[str, Any],
        store: RunStateStore,
    ) -> dict[str, Any]:
        """Push the exact candidate commit, once, with no force capability."""

        staging_remote = self.config.repository.remote
        try:
            previous_candidate_sha = None
            if cycle > 1:
                previous = _read_json_artifact(
                    _candidate_commit_path(run_dir, cycle - 1)
                )
                if isinstance(previous, dict):
                    previous_candidate_sha = previous.get("commit_sha")
            remote_tip = remote_run_branch_tip(
                info.source_repo, remote=staging_remote, branch=info.branch
            )
            if remote_tip not in {
                None, candidate["commit_sha"], previous_candidate_sha
            }:
                raise PipelineFailure(
                    "REMOTE_AUTHORITY_MISMATCH",
                    "remote run branch points to a commit outside this candidate chain",
                )
            if remote_tip != candidate["commit_sha"]:
                push_run_branch(
                    info.worktree, remote=staging_remote,
                    branch=info.branch, commit_sha=candidate["commit_sha"],
                )
            verified_tip = remote_run_branch_tip(
                info.source_repo, remote=staging_remote, branch=info.branch
            )
            if verified_tip != candidate["commit_sha"]:
                raise PipelineFailure(
                    "REMOTE_AUTHORITY_MISMATCH",
                    "remote run branch does not point to the candidate commit",
                )
        except (GitError, OSError, ValueError) as exc:
            raise CandidatePushError("PUSH_FAILED: candidate push did not complete") from exc
        candidate = dict(candidate)
        candidate["remote"] = staging_remote
        candidate["remote_branch"] = info.branch
        candidate["remote_sha"] = candidate["commit_sha"]
        candidate["pushed_at"] = candidate.get("pushed_at") or datetime.now(timezone.utc).isoformat()
        # Persist the exact immutable remote identity alongside the local
        # candidate identity before the reviewer is called.
        atomic_write_text(_candidate_commit_path(run_dir, cycle), _json_text(candidate))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"{cycle:03d}"] = candidate
        store.update(
            status=store.load().get("status", RunStatus.APPROVED),
            candidate=candidate_state,
            remote_branch=info.branch,
            remote_sha=candidate["commit_sha"],
        )
        self._trace_emit(
            "candidate.pushed",
            phase="publication",
            cycle=cycle,
            data={
                "parent_sha": candidate.get("parent_sha"),
                "commit_sha": candidate.get("commit_sha"),
                "tree_sha": candidate.get("tree_sha"),
                "remote": candidate.get("remote"),
                "branch": candidate.get("remote_branch") or info.branch,
                "remote_sha": candidate.get("remote_sha"),
                "pushed_at": candidate.get("pushed_at"),
            },
            once=True,
        )
        return candidate

    def _run_v2_reviewer(
        self,
        *,
        reviewer: Reviewer,
        spec: str,
        run_dir: Path,
        repository_reference: RepositoryReference,
        evidence: EvidenceBundle,
        input: ReviewCycleInput,
        artifacts_dir: Path,
        worktree: Path,
        base_sha: str,
        candidate_commit: Mapping[str, Any],
    ) -> ReviewResult:
        """The single reviewer evidence assembly of every cycle.

        The required-check summary, the parse argument and the later commit
        gate all use the same actual ``evidence.deterministic_passed``; the
        call is always a fresh conversation.
        """

        candidate_sha = candidate_commit.get("commit_sha")
        if not _is_object_id(candidate_sha):
            raise OrchestrationError(
                "candidate commit SHA is missing before reviewer"
            )

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        code_evidence = _review_code_evidence(
            repository_reference=repository_reference,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            evidence=evidence,
            remote_sha=candidate_commit.get("remote_sha"),
            remote_branch=candidate_commit.get("remote_branch"),
            remote_name=candidate_commit.get("remote"),
        )
        try:
            code_evidence_payload = json.loads(code_evidence)
        except (TypeError, ValueError):
            code_evidence_payload = {}
        candidate_identity = _json_text({
            key: candidate_commit.get(key)
            for key in (
                "commit_sha", "tree_sha", "parent_sha", "remote_branch",
                "remote_sha", "candidate_url", "compare_url",
                "immutable_commit_url", "pushed_at",
            )
            if candidate_commit.get(key) is not None
            or key == "immutable_commit_url"
        })
        if isinstance(code_evidence_payload, Mapping):
            candidate_identity = _json_text({
                "candidate": json.loads(candidate_identity),
                "candidate_url": code_evidence_payload.get("candidate_url"),
                "compare_url": code_evidence_payload.get("compare_url"),
                "candidate_tree_sha": code_evidence_payload.get("candidate_tree_sha"),
                "remote_exploration": code_evidence_payload.get("remote_exploration"),
            })
        diff_bytes = evidence.diff.encode("utf-8", errors="replace")
        diff_excerpt = bounded_semantic_diff(evidence.diff, 16 * 1024)[0]
        prompt_payload = build_final_review_payload(
            spec=spec,
            compact_approved_plan=input.plan_text,
            required_checks_summary=_required_checks_summary(evidence),
            immutable_candidate_identity=candidate_identity,
            changed_files="\n".join(evidence.changed_files),
            diff_sha256=hashlib.sha256(diff_bytes).hexdigest(),
            diffstat=_diffstat(evidence.diff, evidence.changed_files),
            bounded_diff_excerpt=diff_excerpt,
            cycle_summary=(
                _compact_cycle_summary(input.cycle_history)
                + "\n"
                + "REVISION REPORTS\n"
                + (input.revision_report or "NONE")
                + "\n"
                + input.step_reports
                + "\nDEFERRED CONTRACT MISMATCHES\n"
                + input.deferred_mismatches
            ),
            repository_reference=_json_text(repository_reference_dict(repository_reference)),
            budget_bytes=self.config.prompt_budget.final_review_max_bytes,
        )
        reviewer_profile_id = getattr(
            getattr(getattr(self, "_last_selection", None), "final_reviewer", None),
            "profile_id", None,
        )
        reviewer_profile = None
        reviewer_selected = None
        if reviewer_profile_id is not None:
            reviewer_selected = self._trace_selected_profile(
                reviewer_profile_id, ExecutionRole.REVIEWER
            )
            try:
                reviewer_profile = profile_for_role(
                    self.config, reviewer_profile_id, ExecutionRole.REVIEWER
                )
            except ProfileError:
                reviewer_profile = None
        review_started_at = self._trace_time()
        review_started_mono = time.perf_counter()
        self._trace_emit(
            "review.started",
            phase="review",
            cycle=getattr(self, "_trace_cycle", 1),
            data={
                "candidate_sha": candidate_sha,
                "tree_sha": evidence.staged_tree_sha,
                "session": self._trace_session(
                    profile=reviewer_profile,
                    selected=reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=len(prompt_payload.rendered.encode("utf-8", errors="replace")),
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    tree_before=evidence.staged_tree_sha,
                ),
            },
        )
        review = reviewer.review(
            prompt_payload,
            deterministic_passed=evidence.deterministic_passed,
            artifacts_dir=artifacts_dir,
        )
        reviewer_usage = getattr(reviewer, "last_usage", None)
        if reviewer_usage is None:
            reviewer_usage = read_usage_artifact(
                artifacts_dir / "reviewer.usage.json"
            )
        self._trace_emit(
            "review.completed",
            phase="review",
            cycle=getattr(self, "_trace_cycle", 1),
            data={
                "candidate_sha": candidate_sha,
                "tree_sha": evidence.staged_tree_sha,
                "verdict": review.verdict.value,
                "route": review.route.value,
                "session": self._trace_finished_model_session(
                    profile=reviewer_profile,
                    selected=reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=len(prompt_payload.rendered.encode("utf-8", errors="replace")),
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    usage=reviewer_usage,
                    tree_before=evidence.staged_tree_sha,
                    tree_after=evidence.staged_tree_sha,
                    final_message=review.raw,
                ),
            },
        )
        # planner_thread != reviewer_thread: a driver that reports the
        # planner's own conversation for a review breaks independence.
        planner_thread = _read_planner_conversation(run_dir)
        reviewer_thread = getattr(reviewer, "last_conversation", None)
        if planner_thread is not None and reviewer_thread == planner_thread:
            raise ReviewParseError("reviewer reused the planner conversation")
        return review

    def _final_evidence(
        self,
        worktree: Path,
        base_sha: str,
        evidence_dir: Path,
        *,
        check_failures_hard: bool,
        reuse: bool,
        expected_head_sha: str | None = None,
        required_check_ids: tuple[str, ...] | None = None,
        enforce_diff_size: bool = False,
        stage: GateStage | None = None,
    ) -> EvidenceBundle:
        """Final checks for the exact current candidate.

        With *reuse*, durable evidence already frozen for exactly this index
        tree is reused: checks are never replayed for a tree whose evidence is
        complete.
        """

        if reuse:
            stored = _load_evidence(evidence_dir)
            if (
                stored is not None
                and stored.base_sha == base_sha
                and current_head(worktree) == (expected_head_sha or base_sha)
                and stored.staged_tree_sha == index_tree_sha(worktree)
                and stored.staged_tree_sha == candidate_tree_sha(worktree)
            ):
                self._trace_emit(
                    "checks.completed",
                    phase="validation",
                    cycle=getattr(self, "_trace_cycle", 1),
                    data={
                        "reused": True,
                        "stage": stage.value if stage is not None else None,
                        "passed": stored.deterministic_passed,
                        "failures": list(stored.failures),
                        "required_check_ids": list(stored.required_check_ids),
                        "tree_sha": stored.staged_tree_sha,
                        "changed_paths": list(stored.changed_files),
                    },
                )
                return stored
        checks_started_at = self._trace_time()
        checks_started_mono = time.perf_counter()
        checks_tree_before = _safe_candidate_tree(worktree)
        self._trace_emit(
            "checks.started",
            phase="validation",
            cycle=getattr(self, "_trace_cycle", 1),
            data={
                "stage": stage.value if stage is not None else None,
                "required_check_ids": list(required_check_ids or ()),
                "tree_before": checks_tree_before,
            },
        )
        check_config, check_ids = config_with_check_authority(
            self.config, evidence_dir, requested_check_ids=required_check_ids,
            expected_sha256=self._approved_check_authority_sha256(evidence_dir),
        )
        try:
            evidence = collect_evidence(
                worktree, base_sha, check_config, evidence_dir=evidence_dir,
                secrets=self._secrets, check_failures_hard=check_failures_hard,
                expected_head_sha=expected_head_sha, required_check_ids=check_ids,
                enforce_diff_size=enforce_diff_size,
                # Accepted step commits make the current HEAD itself the
                # candidate.  There is no staged diff against that HEAD, but the
                # authoritative checks still must run and their tree is exact.
                allow_empty_diff=current_head(worktree) != base_sha,
            )
        except Exception as exc:
            self._trace_emit(
                "checks.completed",
                phase="validation",
                cycle=getattr(self, "_trace_cycle", 1),
                data={
                    "stage": stage.value if stage is not None else None,
                    "passed": False,
                    "failures": [type(exc).__name__],
                    "tree_sha": _safe_candidate_tree(worktree),
                    "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                },
            )
            raise
        self._trace_emit(
            "checks.completed",
            phase="validation",
            cycle=getattr(self, "_trace_cycle", 1),
            data={
                "stage": stage.value if stage is not None else None,
                "passed": evidence.deterministic_passed,
                "failures": list(evidence.failures),
                "required_check_ids": list(evidence.required_check_ids),
                "tree_sha": evidence.staged_tree_sha,
                "changed_paths": list(evidence.changed_files),
                "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                "started_at": checks_started_at,
            },
        )
        return evidence

    def _revision_runner(self) -> RevisionRunner:
        """Build the revision runner with this run's live dependencies."""

        return RevisionRunner(
            config=self.config,
            secrets=self._secrets,
            effective_repair_scope=self._effective_repair_scope,
            approved_check_authority_sha256=self._approved_check_authority_sha256,
            run_revision=self._run_revision,
            ensure_revision_artifacts=self._ensure_revision_artifacts,
            redact_revision_artifacts=self._redact_revision_artifacts,
            reusable_pre_checks=_reusable_pre_checks,
            hard_integrity_failures=_hard_integrity_failures,
            soft_check_failures=_soft_check_failures,
            check_repair_scope_candidates=_check_repair_scope_candidates,
            check_repair_prompt=_check_repair_prompt,
        )

    def _run_v2_revision_cycle(
        self, *, cycle: int, check_repair_attempt: int | None = None, **request: Any,
    ) -> tuple[Any | None, str | None]:
        """Run one revision or check-repair pass -- see ``RevisionRunner.run``."""

        is_check_repair = request.get("check_repair_evidence") is not None
        selection = request["selection"]
        selected = selection.check_repair if is_check_repair else selection.semantic_reviser
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        profile = None
        if selected is not None:
            try:
                profile = profile_for_role(self.config, selected.profile_id, role)
            except ProfileError:
                profile = None
        tree_before = _safe_candidate_tree(request["info"].worktree)
        started_at = self._trace_time()
        started_mono = time.perf_counter()
        phase = "repair" if is_check_repair else "revision"
        prefix = "check_repair" if is_check_repair else "revision"

        def session(**extra: Any) -> dict[str, Any]:
            return self._trace_session(
                profile=profile, selected=selected, role=role,
                started_at=started_at, started_mono=started_mono,
                tree_before=tree_before, **extra,
            )

        self._trace_emit(
            f"{prefix}.started", phase=phase, cycle=cycle,
            data={
                "tree_before": tree_before,
                "attempt": check_repair_attempt,
                "session": session(prompt_bytes=None),
            },
        )
        try:
            result, error = self._revision_runner().run(**request)
        except Exception as exc:
            self._trace_emit(
                f"{prefix}.agent.completed", phase=phase, cycle=cycle,
                data={
                    "status": "failed",
                    "error": type(exc).__name__,
                    "session": session(prompt_bytes=None, exit_reason=type(exc).__name__),
                },
            )
            raise
        artifact_path = Path(request["artifact_dir"])
        prompt_path = artifact_path / "agent.prompt.txt"
        self._trace_emit(
            f"{prefix}.agent.completed", phase=phase, cycle=cycle,
            data={
                "status": "completed" if error is None else "failed",
                "error": error,
                "session": session(
                    prompt_bytes=prompt_path.stat().st_size if prompt_path.is_file() else None,
                    result=result, exit_reason=error,
                ),
            },
        )
        pre_checks = _read_json_artifact(artifact_path / "pre_checks.json")
        if not is_check_repair and isinstance(pre_checks, dict):
            self._trace_emit(
                "revision.checks.completed", phase=phase, cycle=cycle,
                data={
                    "passed": bool(pre_checks.get("deterministic_passed", False)),
                    "failures": [
                        item for item in pre_checks.get("failures", [])
                        if isinstance(item, str)
                    ] if isinstance(pre_checks.get("failures"), list) else [],
                    "tree_sha": pre_checks.get("staged_tree_sha"),
                },
            )
        return result, error

    def _run_revision_with_recovery(
        self,
        *,
        store: RunStateStore,
        request: Mapping[str, Any],
        is_check_repair: bool,
        cycle: int,
        attempt: int | None = None,
    ) -> tuple[Any | None, str | None]:
        """Retry one reviser/repair contract after proving an exact rollback."""

        selection = request["selection"]
        primary = selection.check_repair if is_check_repair else selection.semantic_reviser
        if primary is None:
            return None, "CHECK_REPAIR_PROFILE_MISSING" if is_check_repair else "SEMANTIC_REVISER_PROFILE_MISSING"
        fallback_authority = (
            selection.check_repair_fallbacks if is_check_repair
            else selection.semantic_reviser_fallbacks
        )
        fallbacks = tuple(fallback_authority)[:self._run_options.recovery.max_executor_fallbacks]
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        phase = "check-repair" if is_check_repair else "semantic-revision"
        prefix = "check-repair" if is_check_repair else "semantic-revision"
        stage_name = request.get("gate_stage", "")
        generation = self._recovery_generation(store)
        retry_key = f"{prefix}:{generation}:{cycle:03d}:{stage_name}:{attempt or 0}"
        retries_used = self._recovery_counter(store, retry_key)
        active = primary
        fallback_index = 0
        artifact_dir = Path(request["artifact_dir"])
        repo = Path(request["repo"])
        worktree = Path(request["info"].worktree)
        allowed = set(request["mutable_scope"])

        while True:
            active_selection = dataclasses.replace(
                selection,
                check_repair=active if is_check_repair else selection.check_repair,
                semantic_reviser=active if not is_check_repair else selection.semantic_reviser,
            )
            current_request = {
                key: value for key, value in request.items() if key != "gate_stage"
            }
            current_request["selection"] = active_selection
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(artifact_dir / "executor.json", _json_text({
                "profile_id": active.profile_id,
                "role": role.value,
                "config_sha256": active.config_sha256,
                "selection_source": "primary execution authority" if active == primary else "frozen execution fallback authority",
            }))
            try:
                tree_before = candidate_tree_sha(worktree)
                index_before = index_tree_sha(worktree)
                status_before = status_porcelain(worktree)
                ownership_before = _git_ownership(repo, worktree)
                if index_before != tree_before:
                    raise PipelineFailure(
                        "RESUME_REQUIRES_OPERATOR",
                        "revision worker checkpoint does not have an exact index tree "
                        f"(tree={tree_before}, index={index_before}, status={list(status_before)[:8]})",
                    )
            except GitError as exc:
                raise PipelineFailure("RESUME_REQUIRES_OPERATOR", "revision pre-attempt tree is unreadable") from exc
            try:
                result, error = self._run_v2_revision_cycle(**current_request)
            except AgentError as exc:
                result, error = None, getattr(exc, "code", AGENT_RUNTIME_FAILED)
            except (GitError, OSError) as exc:
                result, error = None, getattr(exc, "code", AGENT_RUNTIME_FAILED)

            if error is None:
                return result, None

            failed_tree_after = _safe_candidate_tree(worktree)
            changed = self._restore_failed_revision_attempt(
                repo=repo, worktree=worktree, tree_before=tree_before,
                ownership_before=ownership_before, branch_ref=request["branch_ref"],
                allowed=allowed, artifact_dir=artifact_dir,
                status_before=status_before,
                allow_requested_paths=(error == _SCOPE_REQUEST_ROUTE),
            )
            if error in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                AGENT_PROTOCOL_FAILED, AGENT_AUTH_FAILURE,
            }:
                atomic_write_text(artifact_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "reason": str(error)[:120],
                    "profile_id": active.profile_id,
                    "tree_before": tree_before,
                    "tree_after": failed_tree_after,
                    "mutable_scope": sorted(allowed),
                    "rollback_tree_sha": candidate_tree_sha(worktree),
                    "rollback_index_tree_sha": index_tree_sha(worktree),
                    "rollback_status": list(status_porcelain(worktree)),
                }))
            if error == AGENT_AUTH_FAILURE:
                raise PipelineFailure(
                    "EXTERNAL_AUTH_REQUIRED",
                    "executor credentials or external authorization are required",
                )
            if error == _SCOPE_REQUEST_ROUTE:
                return result, error
            if error not in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                AGENT_PROTOCOL_FAILED,
            }:
                return result, error

            decision = classify_failure(error, tree_changed_in_scope=changed)
            budget = self._run_options.recovery.max_transient_attempts
            if retries_used < budget:
                retries_used = self._consume_recovery_counter(store, retry_key)
                self._trace_recovery(
                    "recovery.started", reason=error, decision=decision,
                    attempt=retries_used, tree_before=tree_before,
                    tree_after=candidate_tree_sha(worktree),
                    budget_remaining=max(0, budget - retries_used), phase=phase,
                    cycle=cycle,
                )
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                continue

            if fallback_index < len(fallbacks):
                fallback_index += 1
                fallback_decision = RecoveryDecision(
                    RecoveryDisposition.FALLBACK_EXECUTOR,
                    "primary executor transient retry budget exhausted",
                    True, False,
                )
                self._trace_recovery(
                    "recovery.executor_selected", reason=error,
                    decision=fallback_decision, attempt=fallback_index,
                    tree_before=tree_before, tree_after=candidate_tree_sha(worktree),
                    budget_remaining=max(0, len(fallbacks) - fallback_index),
                    phase=phase, cycle=cycle,
                )
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                active = fallbacks[fallback_index - 1]
                retry_key = f"{prefix}:{generation}:{cycle:03d}:{stage_name}:{attempt or 0}:{active.profile_id}"
                retries_used = self._recovery_counter(store, retry_key)
                continue

            self._trace_recovery(
                "recovery.exhausted", reason=error, decision=decision,
                attempt=retries_used + 1, tree_before=tree_before,
                tree_after=candidate_tree_sha(worktree), budget_remaining=0,
                phase=phase, cycle=cycle,
            )
            return result, error

    def _restore_failed_revision_attempt(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_before: str,
        ownership_before: GitOwnership,
        branch_ref: str,
        allowed: set[str],
        artifact_dir: Path,
        status_before: tuple[str, ...],
        allow_requested_paths: bool,
    ) -> bool:
        """Audit worker ownership and restore only a fully in-scope failed tree."""

        try:
            ownership_violations = _ownership_violations(
                ownership_before, _git_ownership(repo, worktree),
                branch_ref=branch_ref, base_sha=ownership_before.head,
            )
        except GitError as exc:
            ownership_violations = [f"Git ownership could not be read: {type(exc).__name__}"]
        if ownership_violations:
            raise PipelineFailure("AGENT_GIT_VIOLATION", "; ".join(ownership_violations))

        try:
            current_tree = candidate_tree_sha(worktree)
            current_index = index_tree_sha(worktree)
            status = status_porcelain(worktree)
            changed = changed_paths_between_trees(repo, tree_before, current_tree)
            if (
                current_tree != tree_before or current_index != tree_before
                or _status_has_unstaged_or_untracked(status)
            ):
                stage_all(worktree)
                current_tree = candidate_tree_sha(worktree)
                current_index = index_tree_sha(worktree)
                status = status_porcelain(worktree)
                changed = changed_paths_between_trees(repo, tree_before, current_tree)
        except PipelineFailure:
            raise
        except (GitError, OSError, ValueError) as exc:
            raise PipelineFailure("RESUME_REQUIRES_OPERATOR", "failed revision tree could not be frozen") from exc

        requested: set[str] = set()
        if allow_requested_paths:
            report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
            scope_request = report.get("scope_request") if isinstance(report, dict) else None
            paths = scope_request.get("paths") if isinstance(scope_request, dict) else None
            if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
                requested = set(paths)
        unsafe_request = [path for path in requested if not self._safe_scope_request_path(path)]
        if unsafe_request:
            raise PipelineFailure("AGENT_SCOPE_VIOLATION", "scope request contains an unsafe path")
        outside = [path for path in changed if path not in allowed and path not in requested]
        if outside:
            raise PipelineFailure(
                AGENT_SCOPE_VIOLATION,
                "worker changed paths outside scope: " + _paths_detail(outside),
            )
        if current_tree != tree_before:
            try:
                security_failures = scan_staged_security(worktree, secrets=self._secrets)
            except (GitError, OSError, ValueError) as exc:
                raise PipelineFailure(
                    "RESUME_REQUIRES_OPERATOR", "failed revision changes could not be security-scanned",
                ) from exc
            if security_failures:
                raise PipelineFailure(
                    security_failures[0].split(":", 1)[0], "; ".join(security_failures),
                )

        try:
            if changed:
                restore_paths_from_tree(worktree, tree_before, sorted(changed))
                stage_all(worktree)
            if (
                candidate_tree_sha(worktree) != tree_before
                or index_tree_sha(worktree) != tree_before
                or status_porcelain(worktree) != status_before
            ):
                raise GitError("failed revision rollback did not restore the exact tree")
            violations = _ownership_violations(
                ownership_before, _git_ownership(repo, worktree),
                branch_ref=branch_ref, base_sha=ownership_before.head,
            )
            if violations:
                raise GitError("Git ownership changed during rollback")
            atomic_write_text(artifact_dir / "tree_after_failure.txt", tree_before + "\n")
        except (GitError, OSError) as exc:
            _record_failure_tree(artifact_dir, worktree)
            raise PipelineFailure("RESUME_REQUIRES_OPERATOR", "failed revision rollback was not exact") from exc
        return bool(changed)

    @staticmethod
    def _safe_scope_request_path(path: str) -> bool:
        return bool(
            isinstance(path, str) and path and path == path.strip()
            and path not in {".", ".."}
            and "\x00" not in path and "\\" not in path
            and not path.startswith("/") and "//" not in path
            and all(part not in {"", ".", "..", ".git"} for part in Path(path).parts)
            and not any(char in path for char in "*?[]{}")
        )

    def _redact_revision_artifacts(self, artifact_dir: Path) -> None:
        for name in _REVISION_ARTIFACTS:
            redact_file(artifact_dir / name, self._secrets)

    def _redact_step_artifacts(self, step_dir: Path) -> None:
        for name in _AGENT_ARTIFACTS:
            redact_file(step_dir / name, self._secrets)

    def _step_contract_drift(
        self, repo: Path, before_tree: str, expected_tree: str, step: ImplementationStep,
    ) -> str | None:
        """Check the step's Git preconditions on the tree Codex will receive.

        Every READ/WRITE/DELETE path must exist in *before_tree* and no CREATE
        path may exist.  The tree must also be exactly the base or the tree
        frozen after the previous step.
        """

        if before_tree != expected_tree:
            return "worktree changed outside a step"
        problems: list[str] = []
        for label, paths, must_exist in (
            ("read_missing", read_set_paths(step.read_set), True),
            ("write_missing", step.write_set, True),
            ("delete_missing", step.delete_set, True),
            ("create_exists", step.create_set, False),
        ):
            wrong = [path for path in paths
                     if path_exists_in_tree(repo, before_tree, path) is not must_exist]
            if wrong:
                problems.append(f"{label}={_paths_detail(wrong)}")
        return " ".join(problems) or None

    def _v2_failed(
        self, store: RunStateStore, run_dir: Path, reason: str,
        step_id: str | None, detail: Any = None, *,
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
    ) -> RunResult:
        if reason == "CHECK_REPAIR_EXHAUSTED":
            self._trace_emit(
                "check_repair.exhausted",
                phase="repair",
                cycle=getattr(self, "_trace_cycle", 1),
                data={"reason": reason, "step_id": step_id},
                once=True,
            )
        try:
            self._update_v2_usage(store, run_dir)
        except (OSError, ValueError):
            pass
        state = store.load()
        fields = _terminal_step_fields(state, step_id)
        cycles = list(state.get("cycles") or [])
        current_cycle = state.get("cycle", 1)
        for index, cycle in enumerate(cycles):
            if isinstance(cycle, dict) and cycle.get("number") == current_cycle:
                cycles[index] = {**cycle, "status": "failed", "failure": reason}
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
                atomic_write_text(step_dir / "step.json", _json_text({
                    "id": step_id, "status": "FAILED", "reason": reason,
                    "profile_id": profile_id,
                    "tree_before": tree_before, "tree_after": tree_after,
                    **({"changed_paths": []} if reason == "AGENT_CONTRACT_MISMATCH" and tree_before == tree_after else {}),
                    **({"mismatch": _bounded_v2_report(mismatch)} if mismatch else {}),
                    **({"mismatch_clean": mismatch_clean}
                       if reason == "AGENT_CONTRACT_MISMATCH" else {}),
                    **({"mismatch_retry_count": mismatch_retry_count}
                       if mismatch_retry_count else {}),
                    **({"initial_mismatch": _bounded_v2_report(initial_mismatch)}
                       if initial_mismatch else {}),
                    **({"index_tree_after": index_tree_after}
                       if index_tree_after else {}),
                    "usage": step_usage,
                }))
        return RunResult(run_dir, RunStatus.FAILED,
                         store.record_failure(
                             reason,
                             redact(detail, self._secrets) if detail is not None else None,
                             **fields,
                         ))

    def _cleanup_published_run_branch(
        self,
        *,
        store: RunStateStore,
        info: WorktreeInfo,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Best-effort cleanup after a successful fast-forward publication."""

        cleanup: dict[str, Any] = {
            "status": "warning",
            "remote": self.config.repository.remote,
            "branch": info.branch,
            "commit_sha": commit_sha,
            "warning": "run branch cleanup did not complete; branch retained",
        }
        try:
            persisted_branch = store.load().get("branch")
            if persisted_branch != info.branch:
                raise GitError("persisted run branch does not match the run branch")
            validate_run_branch(persisted_branch, base_ref=self.config.base_ref)
            result = delete_run_branch(
                info.source_repo,
                remote=self.config.repository.remote,
                branch=persisted_branch,
                expected_commit_sha=commit_sha,
                base_ref=self.config.base_ref,
            )
            cleanup["status"] = result.status
            cleanup.pop("warning")
        except Exception:
            # Cleanup is deliberately not a publication failure.  Keep the
            # warning fixed and secret-free; the branch remains for retry or
            # operator cleanup when its identity is not exact.
            pass
        return cleanup

    def _persist_published_run_branch_cleanup(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: WorktreeInfo,
        commit_sha: str,
        publish_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Complete best-effort cleanup after publication is durable.

        The publication state and completed checkpoint are deliberately
        written by the caller before this method is entered.  If interruption
        happens while cleaning up, the already-published state must not be
        downgraded to INTERRUPTED by the outer run boundary.
        """

        try:
            cleanup = self._cleanup_published_run_branch(
                store=store, info=info, commit_sha=commit_sha,
            )
        except KeyboardInterrupt:
            return store.load()
        publish_payload = {
            **publish_payload,
            "run_branch_cleanup": cleanup,
        }
        atomic_write_text(run_dir / "publish.json", _json_text(publish_payload))
        return store.update(status=RunStatus.PUBLISHED, publish=publish_payload)

    def _complete_candidate_publication(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: Any,
        approved_tree: str,
        commit_sha: str,
        repository_reference: RepositoryReference,
        cycle: int,
    ) -> RunResult:
        """Publish an already pushed candidate, only after reviewer PASS."""

        self._validate_github_publication_mode()
        fields: dict[str, Any] = {"commit_sha": commit_sha, "current_step": None, "cycle": cycle}
        try:
            state = store.load()
            assert_deferred_verifications_resolved([
                *(state.get("accepted_steps") or []),
                *(state.get("deferred_verifications") or []),
            ])
            chain_records = accepted_chain_records(run_dir)
            if not chain_records:
                raise GitError("the accepted commit chain is missing")
            validate_accepted_chain(
                info.worktree,
                run_dir=run_dir,
                base_sha=info.base_sha,
                tip_sha=commit_sha,
                approved_tree_sha=approved_tree,
            )
        except (CommitSafetyError, GitError) as exc:
            state = store.record_failure("COMMIT_TREE_MISMATCH", str(exc), **fields)
            return RunResult(run_dir, RunStatus.FAILED, state)
        try:
            if state.get("approved_tree_sha") != approved_tree:
                raise GitError("durable approved tree differs from candidate tree")
            if current_head(info.worktree) != commit_sha:
                raise GitError("candidate commit is not the run branch tip")
            if resolve_tree(info.worktree, commit_sha) != approved_tree:
                raise GitError("candidate commit tree differs from approved tree")
            candidate_record = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
            if (
                not isinstance(candidate_record, dict)
                or candidate_record.get("commit_sha") != commit_sha
                or candidate_record.get("remote") != self.config.repository.remote
                or candidate_record.get("remote_branch") != info.branch
                or candidate_record.get("remote_sha") != commit_sha
                or not isinstance(candidate_record.get("pushed_at"), str)
                or not candidate_record.get("pushed_at")
                or remote_run_branch_tip(
                    info.source_repo,
                    remote=self.config.repository.remote,
                    branch=info.branch,
                ) != commit_sha
            ):
                raise GitError("candidate remote authority is not exact")
            expected_parent = commit_parents(info.worktree, commit_sha)[0]
            validate_run_branch(info.branch, base_ref=self.config.base_ref)
            if self.config.publish.enabled:
                repository_remote_url(info.worktree, self.config.publish.remote)
        except (GitError, OSError, ValueError):
            state = store.record_failure("COMMIT_TREE_MISMATCH", "candidate identity is not exact", **fields)
            return RunResult(run_dir, RunStatus.FAILED, state)

        self._write_checkpoint(
            run_dir, ResumePhase.PUBLISH, cycle=cycle,
            head=commit_sha, tree=approved_tree,
            expected_parent_sha=expected_parent,
        )
        self._trace_emit(
            "publish.started",
            phase="publication",
            cycle=cycle,
            data={
                "enabled": self.config.publish.enabled,
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "remote": self.config.publish.remote,
                "target": self.config.publish.mode,
            },
            once=True,
        )
        if not self.config.publish.enabled:
            self._ensure_github_pull_request_metadata(
                store=store,
                run_id=str(store.load().get("run_id", "")),
                info=info,
                commit_sha=commit_sha,
                cycle=cycle,
            )
            state = store.update(status=RunStatus.COMMITTED, **fields)
            mark_checkpoint_completed(run_dir)
            self._trace_emit(
                "publish.completed",
                phase="publication",
                cycle=cycle,
                data={
                    "enabled": False,
                    "status": "committed",
                    "commit_sha": commit_sha,
                    "tree_sha": approved_tree,
                },
                once=True,
            )
            return RunResult(run_dir, RunStatus.COMMITTED, state)

        store.update(status=RunStatus.PUBLISHING, **fields)
        base_branch = self.config.base_ref
        try:
            fast_forward = self.config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
            candidate = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
            if not isinstance(candidate, dict) or candidate.get("commit_sha") != commit_sha:
                raise GitError("candidate commit artifact does not match publication")
            if remote_run_branch_tip(
                info.source_repo, remote=self.config.repository.remote, branch=info.branch
            ) != commit_sha:
                raise GitError("candidate run branch is not pushed")
            if fast_forward:
                outcome = publish_fast_forward_base(
                    info.source_repo, remote=self.config.publish.remote,
                    base_branch=base_branch, base_sha=info.base_sha,
                    commit_sha=commit_sha, approved_tree=approved_tree,
                    run_branch=info.branch, expected_parent=expected_parent,
                    accepted_commits=chain_records or None,
                )
                publish_payload = {
                    "mode": PublishMode.FAST_FORWARD_BASE.value,
                    "target": base_branch, "remote": self.config.publish.remote,
                    "branch": base_branch, "run_branch": info.branch,
                    "base_sha": info.base_sha, "commit_sha": commit_sha,
                    "web_url": _commit_web_url(repository_reference, commit_sha),
                    "status": "pushed", "local_base_updated": outcome.local_base_updated,
                    "base_checked_out_in": list(outcome.base_checked_out_in),
                    "run_branch_cleanup": {
                        "status": "pending",
                        "remote": self.config.publish.remote,
                        "branch": info.branch,
                        "commit_sha": commit_sha,
                    },
                }
            else:
                publish_payload = {
                    "mode": PublishMode.RUN_BRANCH.value,
                    "target": info.branch, "remote": self.config.publish.remote,
                    "branch": info.branch, "commit_sha": commit_sha,
                    "web_url": _commit_web_url(repository_reference, commit_sha),
                    "status": "pushed",
                }
        except BaseMovedError as exc:
            decision = classify_failure("BASE_MOVED_SINCE_RUN")
            self._trace_recovery(
                "recovery.classified", reason="BASE_MOVED_SINCE_RUN",
                decision=decision, attempt=1, tree_before=approved_tree,
                tree_after=approved_tree, budget_remaining=0,
                phase="publication", cycle=cycle,
            )
            state = store.record_failure(
                "BASE_MOVED_SINCE_RUN", f"{exc}; candidate remains unpublished",
                publish={"mode": self.config.publish.mode, "target": base_branch,
                         "remote": self.config.publish.remote, "commit_sha": commit_sha,
                         "status": "refused", "local_base_updated": False}, **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        except BasePushError as exc:
            detail = "push did not complete"
            if exc.local_base_updated:
                detail += f"; local {base_branch} already points to {commit_sha}"
            decision = classify_failure(
                "PUSH_FAILED", remote_required=True, remote_unavailable=True,
            )
            self._trace_recovery(
                "recovery.classified", reason="PUSH_FAILED", decision=decision,
                attempt=1, tree_before=approved_tree, tree_after=approved_tree,
                budget_remaining=self._run_options.recovery.max_transient_attempts,
                phase="publication", cycle=cycle,
            )
            state = store.record_failure(
                "PUSH_FAILED", detail,
                publish={"mode": self.config.publish.mode, "target": base_branch,
                         "remote": self.config.publish.remote, "commit_sha": commit_sha,
                         "status": "push-failed", "local_base_updated": exc.local_base_updated},
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        except (GitError, OSError, ValueError):
            decision = classify_failure(
                "PUSH_FAILED", remote_required=True, remote_unavailable=True,
            )
            self._trace_recovery(
                "recovery.classified", reason="PUSH_FAILED", decision=decision,
                attempt=1, tree_before=approved_tree, tree_after=approved_tree,
                budget_remaining=self._run_options.recovery.max_transient_attempts,
                phase="publication", cycle=cycle,
            )
            state = store.record_failure("PUSH_FAILED", "publication did not complete", **fields)
            return RunResult(run_dir, RunStatus.FAILED, state)
        self._ensure_github_pull_request_metadata(
            store=store,
            run_id=str(store.load().get("run_id", "")),
            info=info,
            commit_sha=commit_sha,
            cycle=cycle,
        )
        atomic_write_text(run_dir / "publish.json", _json_text(publish_payload))
        metadata_fields = {"remote_branch": info.branch} if self.config.github.enabled else {}
        state = store.update(
            status=RunStatus.PUBLISHED,
            publish=publish_payload,
            **metadata_fields,
            **fields,
        )
        mark_checkpoint_completed(run_dir)
        self._trace_emit(
            "publish.completed",
            phase="publication",
            cycle=cycle,
            data={
                "enabled": True,
                "status": publish_payload.get("status"),
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "remote": publish_payload.get("remote"),
                "target": publish_payload.get("target"),
            },
            once=True,
        )
        if fast_forward:
            state = self._persist_published_run_branch_cleanup(
                store=store, run_dir=run_dir, info=info, commit_sha=commit_sha,
                publish_payload=publish_payload,
            )
        return RunResult(run_dir, RunStatus.PUBLISHED, state)

    # -- resume ------------------------------------------------------------

    def resume(
        self, run_id: str, *, on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Resume a failed or interrupted run at its durable checkpoint.

        Never replays a successful phase.  Every persisted invariant is
        validated first; a mismatch records ``RESUME_INTEGRITY_FAILURE`` (or
        ``RESUME_REQUIRES_OPERATOR``) without any model call.  A run that is
        not resumable raises :class:`ResumeNotAllowedError` and its state is
        left untouched.
        """

        try:
            selected = _safe_run_id(run_id)
        except OrchestrationError as exc:
            raise ResumeError(str(exc)) from exc
        run_dir = (self.config.runs_root / selected).expanduser().resolve()
        if not run_dir.is_dir():
            raise ResumeError("run directory does not exist")
        if not (run_dir / "state.json").is_file():
            raise ResumeError("run state does not exist")
        store = RunStateStore(run_dir / "state.json")
        try:
            state = store.load()
            pipeline_version_from_state(state)
        except (OSError, ValueError) as exc:
            raise ResumeError(f"run state is unreadable: {exc}") from exc
        try:
            options, _ = read_run_options_for_state(run_dir, state)
            self._run_options = options
            self._effective_repair_scope = effective_repair_scope_policy(options)
            self.config = effective_run_config(self.config, options)
        except RunOptionsError as exc:
            raise ResumeNotAllowedError("run options are missing or invalid") from exc
        self._secrets = config_secret_values(self.config, self._runtime_environment)
        self._begin_trace(run_dir, selected, created=False)
        eligibility = resume_info(run_dir, state)
        if not eligibility.resumable:
            raise ResumeNotAllowedError(eligibility.reason or "run is not resumable")
        try:
            checkpoint = read_checkpoint(run_dir)
        except ResumeCheckpointError as exc:
            raise ResumeNotAllowedError(str(exc)) from exc
        if checkpoint is None:
            raise ResumeNotAllowedError("no resume checkpoint")
        previous = state.get("resume") if isinstance(state.get("resume"), dict) else {}
        attempts = previous.get("attempts") if isinstance(previous.get("attempts"), int) else 0
        record = {
            "phase": checkpoint.phase.value,
            "label": resume_label(checkpoint),
            "attempts": attempts + 1,
            "previous_status": state.get("status"),
            "previous_failure": state.get("failure"),
        }
        if checkpoint.phase in {
            ResumePhase.CONTEXT, ResumePhase.PLANNER,
            ResumePhase.PLAN_APPROVAL, ResumePhase.WORKTREE_SETUP,
        }:
            return self._diagnose_result(self._resume_pre_execution(
                store, run_dir, selected, state, checkpoint, record,
                on_claimed=on_claimed,
            ))
        try:
            resumed = validate_resume(
                config=self.config, repair_scope=self._effective_repair_scope,
                run_dir=run_dir, state=state, checkpoint=checkpoint,
                staging_remote=self.config.repository.remote,
            )
        except (ResumeIntegrityError, ResumeRequiresOperatorError) as exc:
            failed = store.record_failure(
                exc.code, redact(str(exc), self._secrets),
                resume={**record, "status": "refused"}, current_step=None,
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.FAILED, failed))
        claimed = store.transition_if(
            state.get("status", RunStatus.FAILED), state.get("updated_at"),
            status=PHASE_STATUS[checkpoint.phase], failure=None, current_step=None,
            resume={**record, "status": "running",
                    "restored_paths": list(resumed.restore_paths)},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            if resumed.restore_paths:
                self._restore_checkpoint_tree(resumed)
            pipeline = PipelineV2Context(
                run_dir=run_dir, run_id=selected, spec=resumed.spec,
                context=resumed.context, repo=resumed.info.source_repo,
                base_sha=resumed.info.base_sha, base_tree_sha=resumed.base_tree_sha,
                repository_reference=resumed.repository_reference, info=resumed.info,
                plan=resumed.plan, bundle=resumed.bundle, selection=resumed.selection,
                options=self._run_options,
            )
            return self._diagnose_result(
                self._run_pipeline(store, pipeline, checkpoint, resumed=True)
            )
        except (ResumeIntegrityError, ResumeRequiresOperatorError) as exc:
            failed = store.record_failure(
                exc.code, redact(str(exc), self._secrets),
                **self._closing_step_fields(store, "failed"),
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.FAILED, failed))
        except KeyboardInterrupt:
            interrupted = store.update(
                status=RunStatus.INTERRUPTED,
                failure={"reason": "INTERRUPTED"},
                **self._closing_step_fields(store, "interrupted"),
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.INTERRUPTED, interrupted))
        except Exception as exc:
            failed = store.record_failure(
                _failure_reason(exc),
                redact(str(exc), self._secrets),
                **self._closing_step_fields(store, "failed"),
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.FAILED, failed))

    def _restore_checkpoint_tree(self, resumed: ResumedRun) -> None:
        """Undo a failed attempt's in-scope edits, exactly and boundedly."""

        worktree = resumed.info.worktree
        expected = resumed.checkpoint.expected_tree_sha
        try:
            restore_paths_from_tree(worktree, expected, resumed.restore_paths)
            stage_all(worktree)
            restored = (
                index_tree_sha(worktree) == expected
                and candidate_tree_sha(worktree) == expected
                and not _status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except GitError:
            restored = False
        if not restored:
            raise ResumeRequiresOperatorError(
                "the checkpoint tree could not be restored exactly"
            )

    # -- operator plan recovery ----------------------------------------------

    def recover_plan(
        self, run_id: str, replacement_raw: str, *,
        on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Replace a failed planner answer with an operator META PLAN v2.

        No model is called.  The replacement is validated exactly like a
        planner answer, published as the run's plan authority, and the
        checkpoint moves to PLAN_APPROVAL.  The run then continues through the
        normal resume workflow: plan approval, worktree setup, the first step...
        A refusal raises :class:`PlanRecoveryError` and changes nothing.
        """

        self._persist_recovered_plan(run_id, replacement_raw)
        return self.resume(run_id, on_claimed=on_claimed)

    def _persist_recovered_plan(self, run_id: str, replacement_raw: str) -> None:
        def refuse(message: str) -> NoReturn:
            raise PlanRecoveryError(message)

        raw = validate_replacement_text(replacement_raw)
        try:
            selected = _safe_run_id(run_id)
        except OrchestrationError as exc:
            refuse(str(exc))
        run_dir = (self.config.runs_root / selected).expanduser().resolve()
        if not (run_dir / "state.json").is_file():
            refuse("run state does not exist")
        store = RunStateStore(run_dir / "state.json")
        try:
            state = store.load()
        except (OSError, ValueError) as exc:
            refuse(f"run state is unreadable: {exc}")
        eligibility = plan_recovery_info(run_dir, state)
        if not eligibility.eligible:
            refuse(eligibility.reason or "run is not eligible for plan recovery")
        try:
            options, _ = read_run_options_for_state(run_dir, state)
            # The run's frozen options decide the policies and catalogues,
            # never the current defaults.
            config = effective_run_config(self.config, options)
        except RunOptionsError:
            refuse("run options are missing or invalid")
        checkpoint = read_checkpoint(run_dir)
        if checkpoint is None or checkpoint.phase is not ResumePhase.PLANNER:
            refuse("run is not at its PLANNER checkpoint")
        for name in ("spec.md", "context.txt"):
            if not (run_dir / name).is_file():
                refuse(f"{name} is missing")

        # The run is bound to its immutable stored BASE, not to where
        # base_ref points today.
        base_sha = state.get("base_sha")
        if not _is_object_id(base_sha):
            refuse("run base SHA is missing or invalid")
        if checkpoint.expected_head_sha is None or checkpoint.expected_tree_sha is None:
            refuse("PLANNER checkpoint has no BASE identity")
        try:
            repo = git_root(config.repo)
            if state.get("repo") not in {None, str(repo), str(config.repo)}:
                refuse("the configured repository is not the run repository")
            if resolve_commit(repo, base_sha) != base_sha:
                refuse("run base SHA does not resolve to itself")
            base_tree = resolve_tree(repo, base_sha)
            if checkpoint.expected_head_sha != base_sha:
                refuse("PLANNER checkpoint base SHA does not match the run")
            if checkpoint.expected_tree_sha != base_tree:
                refuse("PLANNER checkpoint base tree does not match the run")
            reference = _read_repository_reference(run_dir)
            if reference is None or reference.base_sha != base_sha:
                refuse("repository reference does not match the run base SHA")
            worktree_path = (config.worktrees_root / selected).expanduser().resolve()
            if worktree_path.exists() or str(worktree_path) in registered_worktrees(repo):
                refuse("a run worktree already exists")
            if any(
                ref.startswith("refs/heads/harness/") and ref.endswith(f"/{selected}")
                for ref in local_branches(repo)
            ):
                refuse("a run branch already exists")
        except GitError as exc:
            refuse(f"run Git identity cannot be verified: {exc}")

        try:
            profiles = tuple(profiles_for_config(config).values())
        except ProfileError as exc:
            refuse(f"profile catalogue is unavailable: {exc}")
        try:
            plan = parse_task_plan_v2(
                raw,
                planning=config.planning,
                check_catalog=config.check_catalog,
                default_check_ids=config.default_check_ids,
            )
            if plan.decision is not PlanDecision.READY:
                refuse("replacement plan must be STATUS: READY")
            validate_execution_mode_policy(plan, config.planning)
            validate_decomposition_policy(plan, config.planning)
        except V2PlanParseError as exc:
            refuse(f"replacement plan is invalid: {exc}")
        try:
            # An operator plan is bound by the same repository preconditions.
            validate_plan_repository_topology(repo, base_tree, plan)
        except PlanRepositoryPreconditionError as exc:
            refuse(f"{exc.code}: {exc}")
        except GitError as exc:
            refuse(f"run Git identity cannot be verified: {exc}")

        replacement_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        source_status = recoverable_plan_source_status(state)
        if source_status is None:
            refuse("run is not in an exact recoverable planner state")
        claimed = store.transition_if(
            source_status, state.get("updated_at"), status=source_status,
            plan_recovery={"status": "persisting", "replacement_raw_sha256": replacement_sha},
        )
        if claimed is None:
            refuse("run state changed while the plan recovery was validated")
        try:
            raw_path = run_dir / "planner.raw.md"
            previous_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None
            archived = _archive_attempt(run_dir, names=_RECOVERY_ATTEMPT_ARTIFACTS)
            if (run_dir / "steps").is_dir():
                # Only planning-time contracts can be there (eligibility):
                # retire them with the plan they belong to.
                if archived is None:
                    archived = _archive_attempt_target(run_dir)
                os.replace(run_dir / "steps", archived / "steps")
            persist_recovered_plan_artifacts(run_dir, plan)
            # The recovered plan becomes approval authority here, so the check
            # authority it selects must be frozen here too -- exactly as the
            # planner path does before its own approval gate.  Without it the
            # operator would be offered the gate while ``check_authority.json``
            # does not exist yet, and the write-once decision published in that
            # window could never bind the ``checks_sha256`` the run later
            # expects.  The whole trusted catalogue is frozen, not just this
            # selection, so a correction plan still runs approved argv.
            selected_checks = config.select_checks(plan.required_checks)
            write_check_authority(
                run_dir, tuple(config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            validate_implementation_bundle(run_dir, expected_step_ids=[step.id for step in plan.steps])
            identity = compute_plan_identity_from_run(run_dir)
            if identity.raw_sha256 != replacement_sha or identity.execution_sha256 is not None:
                raise ApprovalError("recovered plan identity does not match the replacement")
            record = write_plan_recovery_record(
                run_dir, previous_raw_sha256=previous_sha,
                replacement_raw_sha256=replacement_sha,
                archived_attempt=archived.relative_to(run_dir).as_posix() if archived else None,
            )
            self._write_checkpoint(
                run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha, tree=base_tree,
                plan_identity=identity,
            )
        except Exception as exc:
            store.update(
                status=source_status,
                plan_recovery={"status": "failed", "replacement_raw_sha256": replacement_sha,
                               "detail": redact(str(exc), self._secrets_or_empty())},
            )
            raise
        planner_state = state.get("planner") if isinstance(state.get("planner"), dict) else {}
        store.update(
            status=RunStatus.FAILED,
            plan_identity=asdict(identity),
            plan_recovery={**record, "status": "awaiting_approval"},
            planner={
                **planner_state,
                "decision": plan.decision.value,
                "title": plan.title,
                "source": PLAN_SOURCE_OPERATOR,
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
                "required_checks": list(plan.required_checks),
                "steps": [
                    {"id": step.id, "title": step.title,
                     "execution_class": step.execution_class.value,
                     "recommended_profile": self.config.routing.profile_for(step.execution_class),
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": self._run_options.final_reviewer_profile,
            },
        )

    def _secrets_or_empty(self) -> tuple[str, ...]:
        return tuple(getattr(self, "_secrets", ()) or ())

    def _resume_pre_execution(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        state: Mapping[str, Any],
        checkpoint: ResumeCheckpoint,
        record: Mapping[str, Any],
        *,
        on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Resume context/planning/approval/setup without requiring later artifacts."""

        def refuse(message: str) -> NoReturn:
            raise ResumeIntegrityError(message)

        claimed = store.transition_if(
            state.get("status", RunStatus.FAILED), state.get("updated_at"),
            status=PHASE_STATUS[checkpoint.phase], failure=None, current_step=None,
            resume={**record, "status": "running"},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            repo = git_root(self.config.repo)
            if state.get("repo") not in {None, str(repo), str(self.config.repo)}:
                refuse("the configured repository is not the run repository")
            spec = (run_dir / "spec.md").read_text(encoding="utf-8")
            base_sha = state.get("base_sha")
            if base_sha is None:
                base_sha = resolve_commit(repo, self.config.base_ref)
                base_tree = resolve_tree(repo, base_sha)
                store.update(status=RunStatus.PLANNING, repo=str(repo), base_sha=base_sha)
            else:
                if not _is_object_id(base_sha):
                    refuse("run base SHA is invalid")
                base_tree = resolve_tree(repo, base_sha)
            if checkpoint.expected_head_sha is not None and checkpoint.expected_head_sha != base_sha:
                refuse("the checkpoint base SHA changed")
            if checkpoint.expected_tree_sha is not None and checkpoint.expected_tree_sha != base_tree:
                refuse("the checkpoint base tree changed")
            reference = _read_repository_reference(run_dir)
            if (run_dir / "repository_reference.json").exists() and reference is None:
                refuse("repository reference artifact is corrupted")
            if reference is None:
                if checkpoint.phase is not ResumePhase.CONTEXT:
                    refuse("repository reference artifact is missing")
                try:
                    reference = build_repository_reference(repo, base_sha=base_sha, config=self.config.repository)
                except GitError:
                    reference = RepositoryReference(self.config.repository.remote, None, base_sha, None)
                atomic_write_text(run_dir / "repository_reference.json", json.dumps(
                    repository_reference_dict(reference), indent=2
                ) + "\n")
            if reference.base_sha != base_sha:
                refuse("the base SHA changed for this run")

            context_path = run_dir / "context.txt"
            if checkpoint.phase is ResumePhase.CONTEXT:
                context_bundle = build_context(repo, base_sha, spec, self.config.context)
                context = render_context(context_bundle)
                atomic_write_text(context_path, context)
                store.update(status=RunStatus.PLANNING, context={
                    "base_sha": context_bundle.base_sha,
                    "locator_used": context_bundle.locator_used,
                    "locator_warning": context_bundle.locator_warning,
                    "omitted": list(context_bundle.omitted),
                    "total_bytes": context_bundle.total_bytes,
                })
                self._write_checkpoint(run_dir, ResumePhase.PLANNER,
                                             head=base_sha, tree=base_tree)
                checkpoint = ResumeCheckpoint(
                    phase=ResumePhase.PLANNER, review_cycle=1,
                    expected_head_sha=base_sha, expected_tree_sha=base_tree,
                )
            context = context_path.read_text(encoding="utf-8")
            planner_profile_id = (
                state.get("execution", {}).get("planner", {}).get("profile_id")
                if isinstance(state.get("execution"), Mapping)
                and isinstance(state.get("execution", {}).get("planner"), Mapping)
                else None
            )
            if not isinstance(planner_profile_id, str):
                refuse("run planner profile is missing")
            planner_profile = profile_for_role(self.config, planner_profile_id, ExecutionRole.PLANNER)
            if checkpoint.phase is ResumePhase.PLANNER:
                planner = PlannerV2(
                    self._planner_client or _chat_client(
                        build_llm_endpoint(planner_profile), self._runtime_environment
                    ),
                    repository_reference=reference, planning=self.config.planning,
                    check_catalog=self.config.check_catalog,
                    default_check_ids=self.config.default_check_ids,
                    prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
                    repository_preconditions=RepositoryPreconditions(repo, base_tree),
                    on_event=lambda name, data: self._trace_emit(name, phase="planning", cycle=1, data=data),
                )
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
                _persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            else:
                raw = (run_dir / "planner.raw.md").read_text(encoding="utf-8")
                plan = parse_task_plan_v2(
                    raw,
                    planning=self.config.planning,
                    check_catalog=self.config.check_catalog,
                    default_check_ids=self.config.default_check_ids,
                )
                if checkpoint.plan_identity is not None:
                    try:
                        if compute_plan_identity_from_run(run_dir) != checkpoint.plan_identity:
                            refuse("plan identity no longer matches")
                    except (ApprovalError, OSError, UnicodeError) as exc:
                        refuse(f"plan artifacts are invalid: {exc}")
            if checkpoint.phase in {ResumePhase.PLAN_APPROVAL, ResumePhase.WORKTREE_SETUP}:
                try:
                    _bundle, bundle_sha = validate_implementation_bundle(
                        run_dir, expected_step_ids=[step.id for step in plan.steps]
                    )
                    identity = compute_plan_identity_from_run(run_dir)
                except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
                    refuse(f"plan artifacts are invalid: {exc}")
                if checkpoint.plan_identity is not None and identity != checkpoint.plan_identity:
                    refuse("plan identity no longer matches")
                recorded_identity = state.get("plan_identity")
                if isinstance(recorded_identity, Mapping):
                    try:
                        if identity != plan_identity_from_mapping(recorded_identity):
                            refuse("plan identity no longer matches state")
                    except ResumeCheckpointError as exc:
                        refuse(str(exc))
                if checkpoint.phase is ResumePhase.WORKTREE_SETUP:
                    try:
                        _selection, selection_sha = read_execution_selection_with_sha256(run_dir)
                        validate_execution_selection(self.config, _selection)
                    except (ExecutionSelectionError, ProfileError, OSError, UnicodeError) as exc:
                        refuse(f"execution selection is invalid: {exc}")
                    if selection_sha != checkpoint.execution_selection_sha256:
                        refuse("execution selection hash changed")
                    if self.config.approval.require_plan_approval:
                        try:
                            approval = read_plan_approval(run_dir, expected_identity=identity)
                        except ApprovalError as exc:
                            refuse(f"approval artifact is invalid: {exc}")
                        if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                            refuse("approval was changed or is no longer APPROVE")
            existing_info = self._existing_setup_worktree(
                repo, run_dir, run_id, plan, base_sha, self.config.worktrees_root,
                self.config.base_ref,
            ) if checkpoint.phase is ResumePhase.WORKTREE_SETUP else None
            if existing_info is not None:
                _archive_attempt_tree(run_dir / "setup")
            prepared = self._prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                reference, planner_profile, existing_plan=plan,
                existing_info=existing_info,
            )
            if isinstance(prepared, RunResult):
                return prepared
            return self._execute_v2(
                store, run_dir, run_id, spec, repo, base_sha, context, reference,
                prepared=prepared,
            )
        except ResumeRequiresOperatorError as exc:
            failed = store.record_failure(exc.code, redact(str(exc), self._secrets),
                                          **self._closing_step_fields(store, "failed"))
            return RunResult(run_dir, RunStatus.FAILED, failed)
        except KeyboardInterrupt:
            interrupted = store.update(status=RunStatus.INTERRUPTED,
                                       failure={"reason": "INTERRUPTED"},
                                       **self._closing_step_fields(store, "interrupted"))
            return RunResult(run_dir, RunStatus.INTERRUPTED, interrupted)
        except Exception as exc:
            failed = store.record_failure(_failure_reason(exc), redact(str(exc), self._secrets),
                                          **self._closing_step_fields(store, "failed"))
            return RunResult(run_dir, RunStatus.FAILED, failed)

    @staticmethod
    def _existing_setup_worktree(
        repo: Path, run_dir: Path, run_id: str, plan: TaskPlanV2, base_sha: str,
        worktrees_root: Path, base_ref: str,
    ) -> WorktreeInfo | None:
        """Return only an exactly recognizable partial setup; never delete/repair it."""

        expected_branch = build_run_branch(plan.title, run_id)
        raw_path = None
        try:
            raw_state = _read_json_artifact(run_dir / "state.json", 256 * 1024)
            raw_path = raw_state.get("worktree") if isinstance(raw_state, dict) else None
            recorded_branch = raw_state.get("branch") if isinstance(raw_state, dict) else None
        except Exception:
            recorded_branch = None
        path = Path(raw_path).expanduser().resolve() if isinstance(raw_path, str) else (
            Path(worktrees_root).expanduser().resolve() / run_id
        )
        if not path.exists():
            if branch_exists(repo, expected_branch):
                raise ResumeIntegrityError("expected run branch exists without its worktree")
            return None
        if recorded_branch not in {None, expected_branch}:
            raise ResumeIntegrityError("partial worktree has an unexpected branch")
        try:
            if not path.is_dir() or str(path) not in registered_worktrees(repo):
                raise ResumeIntegrityError("partial worktree is not registered")
            if not branch_exists(repo, expected_branch) or symbolic_head(path) != f"refs/heads/{expected_branch}":
                raise ResumeIntegrityError("partial worktree branch is not the expected run branch")
            if current_head(path) != base_sha or candidate_tree_sha(path) != resolve_tree(repo, base_sha):
                raise ResumeIntegrityError("partial worktree is not exactly at the run base")
            if status_porcelain(path):
                raise ResumeRequiresOperatorError("partial worktree is not clean")
        except GitError as exc:
            raise ResumeIntegrityError(f"partial worktree is unreadable: {exc}") from exc
        return WorktreeInfo(repo, path, expected_branch, base_ref, base_sha)

def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (ResumeIntegrityError, ResumeRequiresOperatorError)):
        # Their stable ``.code`` is the authority: these reasons must stay
        # recognizable as permanently non-resumable.
        return exc.code
    if isinstance(exc, OrchestrationError) and str(exc).startswith("CHECK_PREFLIGHT_FAILED:"):
        return str(exc).split()[0]
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
        return "LLM_FAILURE"
    if isinstance(exc, ValidationError):
        return "CHECK_SETUP_INVALID"
    if isinstance(exc, WorkspaceSetupError):
        return exc.code
    if isinstance(exc, GitHubIntegrationError):
        return getattr(exc, "code", "GITHUB_WORKSTREAM_FAILURE")
    return exc.__class__.__name__.upper()


def run_orchestrator(
    config: HarnessConfig | str | Path,
    spec: str | Path,
    *,
    run_id: str | None = None,
) -> RunResult:
    """Functional entry point for CLI and embedding callers."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).run(spec, run_id=run_id)


def resume_run(config: HarnessConfig | str | Path, run_id: str) -> RunResult:
    """Resume *run_id* at its durable checkpoint (same run id, same worktree)."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).resume(run_id)


def recover_plan_run(config: HarnessConfig | str | Path, run_id: str, replacement_raw: str) -> RunResult:
    """Recover a failed planner run with an operator META PLAN v2 (no model call)."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).recover_plan(run_id, replacement_raw)


__all__ = [
    "CommitBoundaryError",
    "OrchestrationError",
    "Orchestrator",
    "ResumeError",
    "ResumeNotAllowedError",
    "generate_run_id",
    "recover_plan_run",
    "resume_run",
    "run_orchestrator",
]
