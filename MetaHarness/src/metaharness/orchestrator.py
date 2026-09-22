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
from .agent.codex import (
    build_mismatch_retry_addendum,
    contract_mismatch_explanation,
    deferred_verify_dependency,
)
from .prompt_contracts import (
    build_implementer_payload,
    build_final_review_payload,
    write_prompt_diagnostics,
)
from .agent.execution import (
    ExecutorRuntimeConfig,
    executor_for_profile,
    legacy_codex_agent_factory,
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
    commit_repair_tree,
    commit_revision_tree,
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
from .planning import PlanParseError
from .planning_v2 import (
    PlannerV2,
    RepairPlannerV2,
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
    build_agent_config,
    build_llm_endpoint,
    profile_for_role,
    profile_execution_fingerprint,
    profiles_for_config,
)
from .trace import TraceSink, TraceStream
from .result import RunResult, ResultArtifactError, atomic_write_text, write_repair_task
from .review import Reviewer, ReviewParseError, ReviewResult
from .state import RunStateStore
from .workspace import WorkspaceSetupError, prepare_workspace
from .run_options import (
    EffectiveRepairScopePolicy,
    RunOptions,
    RunOptionsError,
    effective_repair_scope_policy,
    effective_run_config,
    read_run_options_with_sha256,
    write_run_options,
)
from .orchestration.shared import (
    CandidatePushError,
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
    _commit_subject,
    _git_ownership,
    _git_ownership_payload,
    _is_object_id,
    _json_text,
    _new_status_lines,
    _ownership_violations,
    _paths_detail,
    _read_bounded_text,
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
    _read_check_repair_scope,
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
    cycle_record_path,
    gate_dir,
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
    candidate_evidence,
    completed_step_records,
    load_correction_plan,
    read_candidate_record,
    read_cycle_record,
    validate_resume,
    verify_correction_scope,
)

# Compatibility hook for callers/tests that patch the constructor at this
# module path.  Actual execution still goes through the generic resolver
# above; this name is only used to build its injected adapter.
CodexAgent = legacy_codex_agent_factory
_DEFAULT_CODEX_AGENT = CodexAgent


def _chat_client(endpoint: Any, environment: Mapping[str, str]) -> OpenAIChatTextClient:
    """Construct the production client with the runtime mapping.

    A small signature compatibility branch keeps older test doubles and
    embedding adapters working while the real client always receives it.
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

    remote_available = candidate_url is not None and compare_url is not None

    payload: dict[str, Any] = {
        "authority": "immutable_candidate_commit",
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "candidate_tree_sha": evidence.staged_tree_sha,
        "candidate_url": candidate_url,
        "compare_url": compare_url,
        "remote_exploration": "ALLOWED" if remote_available else "UNAVAILABLE",
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
        agent: Any | None = None,
        reviser: Any | None = None,
        github_client: GitHubWorkstreamClient | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self._planner_client = planner_client
        self._reviewer_client = reviewer_client
        self._recommender_client = recommender_client
        self._injected_agent = agent
        self._injected_reviser = reviser
        self._github_client = (
            github_client if github_client is not None else NullGitHubWorkstreamClient()
        )
        # The local JSONL sink is always created per run.  This optional sink
        # is an observation-only extension point (for example Nimbalyst).
        self._trace_sink = trace_sink
        # Programmatic callers that still inject the old agent objects keep
        # their historical failure-name projection.  Production profile
        # resolution always uses the generic reasons from AgentRunResult.
        self._legacy_backend_injection = (
            agent is not None
            or reviser is not None
            or all(
                profile.id.startswith("legacy-")
                for profile in config.model_profiles.values()
            )
            or CodexAgent is not _DEFAULT_CODEX_AGENT
        )
        self._secrets: tuple[str, ...] = ()
        self._effective_repair_scope = EffectiveRepairScopePolicy(
            "deny-expansion", 4, "run-options"
        )
        # Loaded production configs always contain a process-environment
        # mapping.  The fallback only preserves direct construction of the
        # legacy HarnessConfig dataclass by embedding callers/tests.
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
        name = {
            ExecutionRole.PLANNER: "planner",
            ExecutionRole.REPAIR: "check_repair",
            ExecutionRole.REVIEWER: "final_reviewer",
            ExecutionRole.REVISER: "semantic_reviser",
        }.get(role)
        selected = getattr(selection, name, None) if name is not None else None
        return selected if getattr(selected, "profile_id", None) == profile_id else None

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
                remote=self.config.publish.remote,
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
    ) -> dict[str, Any]:
        """Build session metadata without deriving unavailable metrics."""

        fingerprint = getattr(selected, "config_sha256", None)
        if fingerprint is None and profile is not None:
            try:
                fingerprint = profile_execution_fingerprint(
                    profile,
                    agent_env_allowlist=self.config.agent.env_allowlist,
                    codex_home=self.config.codex_runtime.home,
                    claude_config_home=self.config.claude_runtime.home,
                )
            except (TypeError, ValueError, AttributeError):
                fingerprint = None
        raw_result = getattr(result, "raw_result", None) if result is not None else None
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
            legacy_agent_factory=self._agent_for_profile,
        )
        return executor_for_profile(
            profile,
            runtime,
            reviser=self._injected_reviser,
        )

    def _agent_for_profile(self, profile_id: str) -> Any:
        """Compatibility factory feeding the generic Codex adapter.

        Older embedders override this hook to observe or replace the selected
        implementer.  The orchestration path still receives only the generic
        executor returned above.
        """

        if self._injected_agent is not None:
            return self._injected_agent
        try:
            profile = profile_for_role(self.config, profile_id, ExecutionRole.IMPLEMENTER)
        except ProfileError:
            profile = profile_for_role(self.config, profile_id, ExecutionRole.REPAIR)
        try:
            agent_config = dataclasses.replace(
                build_agent_config(profile),
                env_allowlist=self.config.agent.env_allowlist,
            )
        except ProfileError:
            # The compatibility hook is Codex-shaped, but the resolved
            # executor is not.  Other registered drivers receive no legacy
            # worker object and own their process boundary themselves.
            return None
        return CodexAgent(agent_config)

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
        implementers = tuple(
            p for p in profiles_for_config(self.config).values()
            if ExecutionRole.IMPLEMENTER in p.roles
        )
        reviewers = tuple(
            p for p in profiles_for_config(self.config).values()
            if ExecutionRole.REVIEWER in p.roles
        )
        if existing_plan is None:
            planner = PlannerV2(
                self._planner_client or _chat_client(build_llm_endpoint(planner_profile), self._runtime_environment),
                implementer_ids=frozenset(p.id for p in implementers),
                reviewer_ids=frozenset(p.id for p in reviewers),
                implementer_profiles=implementers,
                reviewer_profiles=reviewers,
                repository_reference=repository_reference,
                planning=self.config.planning,
                check_catalog=self.config.check_catalog,
                default_check_ids=self.config.default_check_ids,
                prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
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
                    {"id": step.id, "title": step.title, "recommended_profile": step.implementer_profile,
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": plan.reviewer_profile,
            },
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "profile_id": step.implementer_profile}
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

        try:
            # REQUIRED_CHECKS has already been parsed against the trusted
            # catalogue.  Materialize those exact trusted definitions before
            # the plan can become approval authority.
            selected_checks = self.config.select_checks(plan.required_checks)
            if not plan.required_checks and not self.config.check_catalog:
                # Preserve the historical ``[[checks]]`` selection semantics
                # for v2 configurations that predate the explicit catalogue.
                selected_checks = self.config.select_checks(None)
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
                step_profile_ids={
                    step.id: self._run_options.default_implementer_profile or step.implementer_profile
                    for step in plan.steps
                },
                semantic_reviser_profile_id=(
                    self._run_options.semantic_reviser_profile
                    if revision_enabled or repair_enabled else None
                ),
                check_repair_profile_id=(
                    self._run_options.check_repair_profile
                    if check_repair_enabled else None
                ),
                final_reviewer_profile_id=self._run_options.final_reviewer_profile,
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
        preflight_failures = run_check_preflights(
            info.worktree, check_config, check_ids or plan.required_checks
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
        and keeps its legacy behavior.
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
            publish_remote=self.config.publish.remote,
            authorize_tree=self._authorize_candidate_tree,
            push_tree=self._push_candidate,
            cycle_update=self._cycle_update,
        )
        gate_acceptance = GateAcceptanceService(
            secrets=self._secrets,
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
            accept_gate_state=bind(gate_acceptance.accept, store),
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
        planned_profile_ids = {step.id: step.implementer_profile for step in plan.steps}
        try:
            if creating:
                cycle_selection = ensure_cycle_execution_selection(
                    ctx.run_dir,
                    resolve_cycle_execution_selection(
                        self.config, cycle=cycle.number,
                        step_profile_ids=planned_profile_ids,
                    ),
                )
            else:
                cycle_selection = read_cycle_execution_selection(ctx.run_dir, cycle.number)
                validate_cycle_execution_selection(self.config, cycle_selection)
                if [item.step_id for item in cycle_selection.steps] != [step.id for step in plan.steps]:
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
        return CyclePlan(
            cycle=cycle, plan=plan, bundle=bundle,
            contracts_dir=correction_dir(ctx.run_dir, cycle),
            step_profile_ids=step_profile_ids,
            correction_bundle_sha256=bundle_sha,
        )

    def _review_implementation_correction(
        self, ctx: PipelineV2Context, cycle: RunCycle,
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
        if current_head(ctx.info.worktree) != candidate["commit_sha"]:
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
        ), review

    def _approved_scope_before(self, ctx: PipelineV2Context, number: int) -> list[str]:
        """Every mutable path the plans of cycles ``1..number-1`` approved."""

        scope: set[str] = set()
        for earlier in range(1, number):
            scope |= set(self._cycle_plan(ctx, earlier).mutable_scope)
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
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route not in {
            ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN,
        }:
            raise ResumeIntegrityError(
                f"cycle {previous:03d} review did not route a correction"
            )
        head = current_head(ctx.info.worktree)
        if head != candidate["commit_sha"]:
            raise ResumeIntegrityError(
                f"cycle {cycle.number:03d} does not start from the reviewed candidate"
            )
        tree_before = candidate_tree_sha(ctx.info.worktree)
        reviewer_profile = profile_for_role(
            self.config, ctx.selection.final_reviewer.profile_id, ExecutionRole.REVIEWER
        )
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
        implementers = tuple(
            profile for profile in profiles_for_config(self.config).values()
            if ExecutionRole.IMPLEMENTER in profile.roles
        )
        planner = RepairPlannerV2(
            self._planner_client or _chat_client(
                build_llm_endpoint(planner_profile), self._runtime_environment
            ),
            implementer_ids=frozenset(profile.id for profile in implementers),
            reviewer_ids=frozenset({ctx.selection.final_reviewer.profile_id}),
            implementer_profiles=implementers,
            reviewer_profiles=(reviewer_profile,),
            planning=self.config.planning,
            check_catalog=self.config.check_catalog,
            original_required_check_ids=ctx.plan.required_checks,
        )
        # The reviewed candidate commit is the code authority: the planner
        # gets its immutable candidate/compare URLs instead of an inline diff.
        remote_available = (
            immutable_commit_web_url(ctx.repository_reference, head) is not None
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
        preflight_failures = run_check_preflights(
            ctx.info.worktree, check_config, check_ids or plan.required_checks
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
            repo=ctx.repo, worktree=ctx.info.worktree, base_sha=parent_sha,
            branch_ref=ctx.branch_ref,
            ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
            expected_tree=(
                checkpoint.expected_tree_sha if checkpoint is not None
                else candidate_tree_sha(ctx.info.worktree)
            ),
            step=step, contract=contract,
            profile_id=cycle_plan.step_profile_ids[step.id],
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
        _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
        steps = self._completed_steps(ctx, cycle_plan)
        try:
            result, error = self._run_v2_revision_cycle(
                store=store, cycle=number, run_dir=ctx.run_dir, repo=ctx.repo,
                base_sha=ctx.base_sha, base_tree_sha=ctx.base_tree_sha, spec=ctx.spec,
                plan=cycle_plan.plan, repository_reference=ctx.repository_reference,
                info=ctx.info, branch_ref=ctx.branch_ref,
                ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
                selection=ctx.selection, artifact_dir=artifact_dir,
                mutable_scope=list(cycle_plan.mutable_scope), step_results=steps,
                deferred_mismatches=_deferred_contract_mismatches(cycle_plan.plan, steps),
                deferred_mismatch_present=_has_deferred_contract_mismatches(steps),
            )
        except AgentScopeError as exc:
            self._redact_revision_artifacts(artifact_dir)
            raise PipelineFailure(AGENT_SCOPE_VIOLATION, redact(str(exc), self._secrets)) from exc
        except AgentError as exc:
            self._redact_revision_artifacts(artifact_dir)
            _record_failure_tree(artifact_dir, ctx.info.worktree)
            raise PipelineFailure(
                getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self._secrets),
            ) from exc
        if error is not None:
            if error in {_SCOPE_REQUEST_ROUTE, "REVISION_SCOPE_VIOLATION", AGENT_SCOPE_VIOLATION}:
                raise PipelineFailure(
                    "HUMAN_REQUIRED",
                    "semantic revision requested scope outside its approved authority",
                )
            raise PipelineFailure(error)
        self._cycle_update(
            store, cycle_plan.cycle, status="revised",
            semantic_revision_report=_bounded_report(result.final_message) if result else "",
        )

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
        _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
        try:
            result, error = self._run_v2_revision_cycle(
                store=store, cycle=number, run_dir=ctx.run_dir, repo=ctx.repo,
                base_sha=ctx.base_sha, base_tree_sha=ctx.base_tree_sha, spec=ctx.spec,
                plan=previous_plan.plan, repository_reference=ctx.repository_reference,
                info=ctx.info, branch_ref=ctx.branch_ref,
                ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
                selection=ctx.selection, artifact_dir=artifact_dir,
                mutable_scope=approved_scope,
                step_results=self._completed_steps(ctx, previous_plan),
                deferred_mismatches=_deferred_contract_mismatches(
                    previous_plan.plan, self._completed_steps(ctx, previous_plan)
                ),
                deferred_mismatch_present=_has_deferred_contract_mismatches(
                    self._completed_steps(ctx, previous_plan)
                ),
                reviewer_correction_evidence=reviewer_evidence,
                candidate_identity=candidate_identity,
                bounded_diff_evidence=bounded_semantic_diff(evidence.diff, 16 * 1024)[0],
            )
        except AgentScopeError as exc:
            self._redact_revision_artifacts(artifact_dir)
            raise PipelineFailure("HUMAN_REQUIRED", redact(str(exc), self._secrets)) from exc
        except AgentError as exc:
            self._redact_revision_artifacts(artifact_dir)
            _record_failure_tree(artifact_dir, ctx.info.worktree)
            raise PipelineFailure(
                getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self._secrets),
            ) from exc
        if error is not None:
            if error in {
                "REVISION_SCOPE_VIOLATION", _SCOPE_REQUEST_ROUTE,
                "HUMAN_REQUIRED", AGENT_SCOPE_VIOLATION,
            }:
                raise PipelineFailure("HUMAN_REQUIRED", "semantic correction requested or changed a path outside approved scope")
            raise PipelineFailure(error)
        self._cycle_update(
            store, cycle_plan.cycle, status="revised",
            semantic_revision_report=_bounded_report(result.final_message) if result else "",
        )

    def _run_gate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage,
    ) -> EvidenceBundle:
        """Run the authoritative deterministic checks for the current tree."""

        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        # An earlier episode result (a red gate before a repair, or an
        # interrupted run) stays durable under ``attempts/NN``.
        _archive_attempt(directory, names=_CHECK_ATTEMPT_ARTIFACTS)
        store.update(status=RunStatus.VALIDATING, current_step=None)
        evidence = self._final_evidence(
            ctx.info.worktree, ctx.base_sha, directory,
            check_failures_hard=False, reuse=True, stage=stage,
            expected_head_sha=current_head(ctx.info.worktree),
            required_check_ids=cycle_plan.plan.required_checks or None,
            enforce_diff_size=False,
        )
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
        previous_scope = (
            _read_check_repair_scope(
                check_repair_attempt_dir(ctx.run_dir, number, stage, records[-1].number),
                fallback_base=cycle_plan.mutable_scope,
                policy_config=self._effective_repair_scope,
            )
            if records else None
        )
        soft = _soft_check_failures(evidence)
        failed_ids = tuple(item.split(":", 1)[1] for item in soft if ":" in item)
        scope = CheckRepairCoordinator(
            effective_repair_scope=self._effective_repair_scope,
        ).resolve_scope(
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=evidence.staged_tree_sha,
            run_dir=ctx.run_dir, evidence=evidence,
            base_mutable_scope=cycle_plan.mutable_scope, previous=previous_scope,
        )
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
        try:
            _result, error = self._run_v2_revision_cycle(
                store=store, cycle=number, run_dir=ctx.run_dir, repo=ctx.repo,
                base_sha=ctx.base_sha, base_tree_sha=ctx.base_tree_sha, spec=ctx.spec,
                plan=cycle_plan.plan, repository_reference=ctx.repository_reference,
                info=ctx.info, branch_ref=ctx.branch_ref,
                ownership_before=_git_ownership(ctx.repo, ctx.info.worktree),
                selection=ctx.selection, artifact_dir=attempt_dir,
                mutable_scope=list(scope.effective_paths),
                check_repair_evidence=evidence, check_repair_scope=scope,
                check_repair_attempt=attempt,
            )
        except (AgentError, GitError, OSError) as exc:
            error = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
            _record_failure_tree(attempt_dir, ctx.info.worktree)
        if error is not None:
            reason = "REVISION_SCOPE_VIOLATION" if error == _SCOPE_REQUEST_ROUTE else error
            atomic_write_text(attempt_dir / "failure.json", _json_text({
                "schema_version": 1,
                "number": attempt,
                "failed_check_ids_before": list(failed_ids),
                "tree_before": evidence.staged_tree_sha,
                "tree_after": _safe_candidate_tree(ctx.info.worktree),
                "mutable_scope": list(scope.effective_paths),
                "profile_id": selected.profile_id,
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
            "profile_id": selected.profile_id,
            "profile_fingerprint": selected.config_sha256,
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
    ) -> StepExecutionOutcome:
        """One approved step, with at most one bounded mismatch retry.

        A *clean* structural mismatch — the worker changed nothing at all — is
        retried exactly once with the same contract, the same profile, the same
        mutable scope and the same candidate tree, in a new fresh worker
        process.  The retry only adds a prompt addendum; it never widens
        WRITE/CREATE/DELETE.  There is never a third attempt.  Every failure
        names the durable step directory it belongs to.
        """

        common = {
            "repo": repo, "worktree": worktree, "base_sha": base_sha,
            "branch_ref": branch_ref, "ownership_before": ownership_before,
            "expected_tree": expected_tree, "step": step, "contract": contract,
            "profile_id": profile_id, "artifact_dir": artifact_dir,
            "forbidden_env_names": forbidden_env_names,
            "future_ownership": future_ownership,
        }
        try:
            outcome = self._run_step_attempt(
                **common, initial_mismatch=None, mismatch_retry_count=0,
            )
            if not isinstance(outcome, DeferredStepExecutionOutcome):
                return outcome
            # The boundary must still be exactly the pre-step boundary before a
            # second worker is allowed to run against it.  Drift here is lost
            # authority, never a deferrable outcome: fail closed so that no
            # second worker and no later step runs.
            drift = self._pre_step_boundary_drift(
                repo, worktree, ownership_before,
                branch_ref=branch_ref, base_sha=base_sha, tree_before=outcome.tree_before,
            )
            # Attempt 1 keeps its own artifacts, including its diagnostics; its
            # deferred record is never the step's current record on a drift.
            _archive_attempt(artifact_dir)
            if drift:
                raise StepExecutionFailure(
                    "STEP_CONTRACT_DRIFT", outcome.step_id,
                    _bounded_v2_report(
                        f"the step boundary drifted before the bounded mismatch retry: {drift}"
                    ),
                    profile_id=outcome.profile_id, tree_before=outcome.tree_before,
                    tree_after=_safe_candidate_tree(worktree), usage=outcome.usage,
                    mismatch=outcome.mismatch,
                )
            return self._run_step_attempt(
                **common, initial_mismatch=outcome.mismatch, mismatch_retry_count=1,
            )
        except StepExecutionFailure as failure:
            failure.step_dir = artifact_dir
            raise

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
        # 5. One fresh worker process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
        # The selected adapter owns the single .run_step( compatibility path.
        retry_addendum = (
            build_mismatch_retry_addendum(
                initial_mismatch=initial_mismatch or "",
                future_ownership=future_ownership,
            )
            if mismatch_retry_count else None
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompt_payload = build_implementer_payload(
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
                retry_addendum=retry_addendum or "",
                budget_bytes=self.config.prompt_budget.implementer_max_bytes,
            )
            request_prompt = prompt_payload.rendered
            write_prompt_diagnostics(artifact_dir, prompt_payload)
            trace_started_at = self._trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = self._trace_selected_profile(
                profile.id, step_role, step_id=step_id
            )
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
                    retry_addendum=retry_addendum,
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
            tree_after = _safe_candidate_tree(worktree)
            index_after = _safe_index_tree(worktree)
            status_after = _safe_status(worktree)
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
            if clean:
                deferred_verify = _bounded_v2_report(
                    deferred_verify_dependency(result.final_message) or ""
                )
                atomic_write_text(artifact_dir / "step.json", _json_text({
                    "id": step_id, "status": "DEFERRED_CONTRACT_MISMATCH",
                    "profile_id": profile.id,
                    "tree_before": tree_before, "tree_after": tree_before,
                    "changed_paths": [], "mismatch": _bounded_v2_report(mismatch),
                    **({"initial_mismatch": _bounded_v2_report(initial_mismatch)}
                       if mismatch_retry_count and initial_mismatch else {}),
                    **({"mismatch_retry_count": mismatch_retry_count}
                       if mismatch_retry_count else {}),
                    **({"deferred_verify": deferred_verify} if deferred_verify else {}),
                    "usage": usage,
                }))
                return DeferredStepExecutionOutcome(
                    step_id=step_id, profile_id=profile.id,
                    tree_before=tree_before, tree_after=tree_before,
                    changed_paths=(), usage=usage, final_report=result.final_message,
                    mismatch=_bounded_v2_report(mismatch),
                    initial_mismatch=(
                        _bounded_v2_report(initial_mismatch)
                        if mismatch_retry_count and initial_mismatch else ""
                    ),
                    mismatch_retry_count=mismatch_retry_count,
                    deferred_verify=deferred_verify,
                )
            _record_failure_tree(artifact_dir, worktree)
            details = []
            if mismatch:
                details.append(_bounded_v2_report(mismatch))
            if tree_after is not None and tree_after != tree_before:
                details.append("worker left candidate modifications")
            elif tree_after is None:
                details.append("failure tree could not be read")
            if index_after is None or index_after != index_before:
                details.append("worker left index modifications")
            if ownership_violations:
                details.extend(ownership_violations)
            residual = _new_status_lines(status_before, status_after)
            if residual:
                details.append("worker left residual Git modifications: " + _paths_detail(residual))
            raise StepExecutionFailure(
                "AGENT_CONTRACT_MISMATCH", step_id,
                "; ".join(details) or "worker reported a contract mismatch",
                profile_id=profile.id, tree_before=tree_before,
                tree_after=tree_after, usage=usage,
                mismatch=_bounded_v2_report(mismatch),
                clean_contract_mismatch=False,
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
        # 10-11. Process outcome.  The tree left behind is recorded so that a
        # resume can tell a clean retry from partial worker changes.
        if result.timed_out or result.exit_reason == AGENT_TIMEOUT:
            raise StepExecutionFailure(
                AGENT_TIMEOUT if not self._legacy_backend_injection else "AGENT_TIMEOUT",
                step_id, **failed, tree_after=_safe_candidate_tree(worktree)
            )
        if result.exit_code not in (0, None) or result.exit_reason in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            if auth_failure:
                reason = "AGENT_AUTH_FAILURE"
                raise StepExecutionFailure(
                    reason, step_id, "Codex authentication failed", **failed,
                    tree_after=_safe_candidate_tree(worktree),
                )
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
                "STEP_WRITE_SET_VIOLATION", step_id,
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
        if re.search(
            r"^\s*(?:VERIFY|VERIFICATION)\s*(?::|=)\s*(?:FAIL|FAILED|RED)\b",
            outcome.final_report,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
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

        if self.config.publish.enabled:
            try:
                remote_tip = remote_run_branch_tip(
                    info.source_repo, remote=self.config.publish.remote, branch=info.branch
                )
                if remote_tip != candidate["commit_sha"]:
                    push_run_branch(
                        info.worktree, remote=self.config.publish.remote,
                        branch=info.branch, commit_sha=candidate["commit_sha"],
                    )
                if remote_run_branch_tip(
                    info.source_repo, remote=self.config.publish.remote, branch=info.branch
                ) != candidate["commit_sha"]:
                    raise GitError("remote run branch does not point to the candidate commit")
            except (GitError, OSError, ValueError) as exc:
                raise CandidatePushError("PUSH_FAILED: candidate push did not complete") from exc
            candidate = dict(candidate)
            candidate["pushed_at"] = candidate.get("pushed_at") or datetime.now(timezone.utc).isoformat()
            # Persist the exact immutable remote identity alongside the local
            # candidate identity before the reviewer is called.
            candidate["remote_branch"] = info.branch
            candidate["remote_sha"] = candidate["commit_sha"]
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

        The gate payload, the parse argument and the later commit gate all
        use the same actual ``evidence.deterministic_passed``; the call is
        always a fresh conversation.
        """

        candidate_sha = candidate_commit.get("commit_sha")
        if not _is_object_id(candidate_sha):
            raise OrchestrationError(
                "candidate commit SHA is missing before reviewer"
            )

        gate = _json_text({
            "deterministic_passed": evidence.deterministic_passed,
            "required_check_ids": list(evidence.required_check_ids),
            "failures": list(evidence.failures),
            "staged_tree_sha": evidence.staged_tree_sha,
        })
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        code_evidence = _review_code_evidence(
            repository_reference=repository_reference,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            evidence=evidence,
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
            spec, input.plan_text, "", gate, "", "", "", "NONE",
            deterministic_passed=evidence.deterministic_passed,
            artifacts_dir=artifacts_dir,
            prompt_payload=prompt_payload,
            iteration=input.iteration,
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
            "remote": self.config.publish.remote,
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
                remote=self.config.publish.remote,
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
                info.source_repo, remote=self.config.publish.remote, branch=info.branch
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
            state = store.record_failure(
                "PUSH_FAILED", detail,
                publish={"mode": self.config.publish.mode, "target": base_branch,
                         "remote": self.config.publish.remote, "commit_sha": commit_sha,
                         "status": "push-failed", "local_base_updated": exc.local_base_updated},
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        except (GitError, OSError, ValueError):
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
            options, _ = read_run_options_with_sha256(
                run_dir,
                expected_sha256=state.get("run_options_sha256")
                if isinstance(state.get("run_options_sha256"), str) else None,
            )
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
                publish_remote=(
                    self.config.publish.remote if self.config.publish.enabled else None
                ),
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
            options, _ = read_run_options_with_sha256(
                run_dir,
                expected_sha256=state.get("run_options_sha256")
                if isinstance(state.get("run_options_sha256"), str) else None,
            )
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
                implementer_ids=frozenset(p.id for p in profiles if ExecutionRole.IMPLEMENTER in p.roles),
                reviewer_ids=frozenset(p.id for p in profiles if ExecutionRole.REVIEWER in p.roles),
                check_catalog=config.check_catalog,
                default_check_ids=config.default_check_ids,
            )
            if plan.decision is not PlanDecision.READY:
                refuse("replacement plan must be STATUS: READY")
            validate_execution_mode_policy(plan, config.planning)
            validate_decomposition_policy(plan, config.planning)
        except V2PlanParseError as exc:
            refuse(f"replacement plan is invalid: {exc}")

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
            if not plan.required_checks and not config.check_catalog:
                selected_checks = config.select_checks(None)
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
                     "recommended_profile": step.implementer_profile, "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": plan.reviewer_profile,
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
            profiles = profiles_for_config(self.config).values()
            if checkpoint.phase is ResumePhase.PLANNER:
                _archive_attempt(run_dir, names=_PLANNER_ATTEMPT_ARTIFACTS)
                planner = PlannerV2(
                    self._planner_client or _chat_client(
                        build_llm_endpoint(planner_profile), self._runtime_environment
                    ),
                    implementer_ids=frozenset(p.id for p in profiles if ExecutionRole.IMPLEMENTER in p.roles),
                    reviewer_ids=frozenset(p.id for p in profiles if ExecutionRole.REVIEWER in p.roles),
                    implementer_profiles=tuple(p for p in profiles if ExecutionRole.IMPLEMENTER in p.roles),
                    reviewer_profiles=tuple(p for p in profiles if ExecutionRole.REVIEWER in p.roles),
                    repository_reference=reference, planning=self.config.planning,
                    check_catalog=self.config.check_catalog,
                    default_check_ids=self.config.default_check_ids,
                    prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
                )
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
                _persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            else:
                raw = (run_dir / "planner.raw.md").read_text(encoding="utf-8")
                plan = parse_task_plan_v2(
                    raw,
                    implementer_ids=frozenset(p.id for p in profiles if ExecutionRole.IMPLEMENTER in p.roles),
                    reviewer_ids=frozenset(p.id for p in profiles if ExecutionRole.REVIEWER in p.roles),
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
