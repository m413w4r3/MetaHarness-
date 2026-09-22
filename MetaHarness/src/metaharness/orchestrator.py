"""The single-task, single-agent MetaHarness V0 state machine."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, NoReturn, Sequence

from .agent.base import (
    AGENT_RUNTIME_FAILED,
    AGENT_PROTOCOL_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentRunRequest,
    AgentRunResult,
    AgentScopeError,
    normalized_failure_reason,
)
from .agent.diagnostics import TOKEN_DIAGNOSTICS_NAME, write_token_diagnostics
from .agent.codex import (
    build_implementer_step_prompt,
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
from .agent.protocol import parse_scope_request
from .approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    read_plan_approval,
    read_scope_approval,
    read_check_authority,
    write_check_authority,
    wait_for_plan_approval,
)
from .config import load_config
from .context import build_context, render_context
from .evidence import (
    DIFF_TOO_LARGE,
    SECRET_IN_DIFF,
    SECRET_IN_STAGED_BLOB,
    UNSCANNABLE_STAGED_BLOB,
    UNREVIEWABLE_TEXT_DIFF,
    EvidenceBundle,
    bounded_semantic_diff,
    collect_evidence,
)
from .validation import (
    ValidationError,
    config_with_check_authority,
    resolve_check_cwd,
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
    normalize_github_web_url,
    publish_fast_forward_base,
    restore_paths_from_tree,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_reviewed_tree,
    commit_candidate_tree,
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
    run_branch_web_url,
    tracked_files_in_tree,
    validate_run_branch,
)
from .commit_gate import (
    CommitSafetyError,
    accepted_step_record,
    assert_deferred_verifications_resolved,
    commit_safety_gate,
    parse_deferred_verification,
)
from .llm.chat import LLMConversationHandle, LLMError, OpenAIChatTextClient
from .recommendation import (
    ExecutionRecommender,
    RecommendationError,
    write_recommendation_error,
)
from .execution_selection import (
    ExecutionSelectionError,
    ensure_execution_selection,
    is_profile_aware_run,
    read_execution_selection_with_sha256,
    resolve_execution_selection,
    resolve_execution_selection_v3,
    ensure_execution_selection_v3,
    resolve_execution_selection_v4,
    ensure_execution_selection_v4,
    validate_execution_selection,
    read_execution_selection_v3_with_sha256,
    validate_execution_selection_v3,
    read_execution_selection_v4_with_sha256,
    validate_execution_selection_v4,
    resolve_execution_selection_v5,
    ensure_execution_selection_v5,
    read_execution_selection_v5_with_sha256,
    validate_execution_selection_v5,
)
from .models import (
    ExecutionRole,
    ExecutionSelectionV4,
    ExecutionSelectionV5,
    ModelProfile,
    ExecutionSelectionV3,
    HarnessConfig,
    ImplementationStep,
    PublishMode,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    RunStatus,
)
from .planning import (
    PlanDecision,
    Planner,
    PlanParseError,
    TaskPlan,
    render_implementation_contract,
)
from .planning_v2 import (
    CheckScopeRepairPlannerV2,
    PlannerV2,
    RepairPlannerV2,
    TaskPlanV2,
    V2PlanParseError,
    build_scope_repair_planner_prompt_bundle,
    parse_task_plan_v2,
    persist_recovered_plan_artifacts,
    read_approved_step_contract,
    read_set_paths,
    validate_decomposition_policy,
    validate_execution_mode_policy,
    validate_implementation_bundle,
    validate_repair_decomposition_policy,
    render_repair_plan_summary,
    render_repair_step_index,
)
from .plan_recovery import (
    PLAN_RECOVERY_ARTIFACT,
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
    is_clean_contract_mismatch_artifact,
    is_recoverable_dirty_contract_mismatch_artifact,
    load_resume_checkpoint,
    mark_checkpoint_completed,
    mismatch_retry_spent,
    phase_index,
    plan_identity_from_mapping,
    read_checkpoint,
    read_checkpoint_record,
    pipeline_version_from_state,
    resume_info,
    resume_label,
    write_checkpoint,
)
from . import resume as resume_module
from .usage import add_usage, empty_usage, normalize_usage, phase_usage_summary, read_usage_artifact
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

# Compatibility hook for historical callers/tests that patched the old
# constructor at this module path.  Actual execution still goes through the
# generic resolver above; this name is only used to build its injected adapter.
CodexAgent = legacy_codex_agent_factory
_DEFAULT_CODEX_AGENT = CodexAgent
from .result import RunResult, write_repair_task
from .review import Reviewer, ReviewParseError, ReviewResult, blocking_finding_lines, parse_review
from .state import RunStateStore
from .validation import ValidationError, check_result_json
from .workspace import WorkspaceSetupError, prepare_workspace
from .result import ResultArtifactError, atomic_write_text
from .run_options import (
    EffectiveRepairScopePolicy,
    RunOptions,
    RunOptionsError,
    effective_repair_scope_policy,
    effective_run_config,
    legacy_or_durable_run_options,
    legacy_or_durable_run_options_with_raw,
    read_repair_scope_override,
    write_run_options,
)

from .orchestration.shared import (  # noqa: F401  (facade re-exports)
    CandidatePushError,
    CheckRepairScope,
    CommitBoundaryError,
    DeferredStepExecutionOutcome,
    GitOwnership,
    OrchestrationError,
    ReviewerTransportError,
    ScopeApprovalRequired,
    StepExecutionFailure,
    StepExecutionOutcome,
    _AGENT_ARTIFACTS,
    _ATTEMPT_ARTIFACTS,
    _BOUNDED_NO_CHANGE_MISMATCH,
    _CHECK_ALIAS_ARTIFACTS,
    _CHECK_ATTEMPT_ARTIFACTS,
    _COMMIT_SUBJECT_LIMIT,
    _GIT_OBJECT_ID,
    _GitOwnership,
    _MAX_AGENT_REPORT_BYTES,
    _MAX_REPORTED_PATHS,
    _MAX_STEP_REPORT_BYTES,
    _PLANNER_ATTEMPT_ARTIFACTS,
    _PLANNER_CONVERSATION,
    _PRE_CHECK_ATTEMPT_ARTIFACTS,
    _RECOVERY_ATTEMPT_ARTIFACTS,
    _REVIEW_ATTEMPT_ARTIFACTS,
    _REVISION_ARTIFACTS,
    _REVISION_ATTEMPT_ARTIFACTS,
    _SECOND_CHECK_REPAIR_PHASES,
    _SYNTHETIC_NO_CHANGE_MISMATCH,
    _archive_attempt,
    _archive_attempt_target,
    _archive_attempt_tree,
    _artifact_tail,
    _bounded_report,
    _bounded_v2_report,
    _check_payload,
    _commit_subject,
    _create_file_once,
    _git_ownership,
    _git_ownership_payload,
    _is_object_id,
    _json_text,
    _new_status_lines,
    _ownership_from_payload,
    _ownership_violations,
    _paths_detail,
    _read_bounded_text,
    _read_json_artifact,
    _read_tree_file,
    _record_failure_tree,
    _repair_checks_payload,
    _safe_candidate_tree,
    _safe_index_tree,
    _safe_path_label,
    _safe_status,
    _slug,
    _status_has_unstaged_or_untracked,
    _step_result_record,
)
from .orchestration.revision import (  # noqa: F401  (facade re-exports)
    RevisionRunner,
    _MAX_REPAIR_CLAUDE_REPORT_BYTES,
    _REVISION_CHECK_LOG_BYTES,
    _SCOPE_REQUEST_HEADER,
    _SCOPE_REQUEST_ROUTE,
    _bounded_repair_claude_report,
    _compact_step_history,
    _deferred_contract_mismatches,
    _future_step_ownership,
    _has_deferred_contract_mismatches,
    _persist_revision_tree,
    _render_revision_template,
    _revision_check_context,
    _revision_contract_index,
    _revision_execution_anomalies,
    _revision_plan_summary,
    _revision_prompt,
    _revision_report_text,
    _scope_request_diagnostic,
    _scope_request_evidence,
    _scope_request_from_payload,
    _scope_request_payload,
    _step_reports_text,
)
from .orchestration.check_repair import (  # noqa: F401  (facade re-exports)
    CheckRepairAttempt,
    CheckRepairCoordinator,
    CheckRepairResult,
    _AUTO_BOUNDED_SOURCE,
    _CHECK_SCOPE_PATH_RE,
    _DIRECT_FAILURES,
    _LEGACY_DIRECT_FAILURES,
    _MAX_CHECK_SCOPE_LOG_BYTES,
    _SAME_SCOPE_RETRY_SOURCE,
    _SECOND_SCOPE_SOURCES,
    _check_repair_prompt,
    _check_repair_scope_candidates,
    _check_repair_scope_payload,
    _expanded_scope_is_applicable,
    _failed_check_text,
    _hard_failure_items,
    _hard_integrity_failures,
    _is_auto_expandable_test_path,
    _read_check_repair_scope,
    _read_log_tail,
    _resolve_check_path_candidate,
    _second_check_repair_state,
    _soft_check_failures,
    _validate_expanded_check_repair_scope,
)
from .orchestration.scope_repair import (  # noqa: F401  (facade re-exports)
    _build_scope_delta,
    _build_scope_repair_delta,
    _ensure_scope_delta,
    _repair_mutation_sets,
)
from .orchestration.candidate import (  # noqa: F401  (facade re-exports)
    _candidate_chain_parent,
    accepted_chain_records,
    validate_accepted_chain,
    _candidate_commit_path,
    _candidate_commit_payload,
    _commit_web_url,
    authorize_commit,
)
from .orchestration.resume_validation import (  # noqa: F401  (facade re-exports)
    _PersistedRevision,
    _ResumedRun,
    _SCOPE_REPAIR_RECOVERY_TREE_PHASES,
    _accepted_review,
    _load_accepted_c01_review,
    _load_c01_review,
    _load_completed_step,
    _load_evidence,
    _load_revision,
    _persist_planner_conversation,
    _read_planner_conversation,
    _read_repository_reference,
    _retry_checks_evidence,
    _reusable_pre_checks,
    _scope_repair_checkpoint_tree,
    _state_cycle_value,
    _verify_step_chain,
)


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
_MAX_REVIEW_CONTEXT_BYTES = 24 * 1024


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


# Kept as a compatibility alias for callers that imported the old private
# helper while the public generator is used by the web run manager.
_generated_run_id = generate_run_id


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


def _review_step_reports_text(results: list[dict[str, Any]]) -> str:
    """Compact structural Luna history for the independent reviewer."""

    records: list[dict[str, Any]] = []

    for item in results:
        record: dict[str, Any] = {
            "id": item.get("id"),
            "status": item.get("status", "COMPLETED"),
            "changed_paths": list(item.get("changed_paths") or []),
        }

        if item.get("mismatch"):
            record["mismatch"] = _bounded_v2_report(str(item.get("mismatch") or ""))

        if item.get("initial_mismatch"):
            record["initial_mismatch"] = _bounded_v2_report(
                str(item.get("initial_mismatch") or "")
            )

        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]

        if item.get("deferred_verify"):
            record["deferred_verify"] = _bounded_v2_report(
                str(item.get("deferred_verify") or "")
            )

        records.append(record)

    return _json_text(records)


def _review_plan_payload(plan: TaskPlanV2) -> dict[str, Any]:
    return {
        "title": plan.title,
        "objective": plan.objective,
        "constraints": plan.constraints,
        "required_checks": list(plan.required_checks),
        "acceptance": plan.acceptance,
        "tests": plan.tests,
        "risks": plan.risks,
        "steps": [
            {
                "id": step.id,
                "title": step.title,
                "depends_on": step.depends_on,
                "objective": step.objective,
                "mutation_scope": {
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                },
                "instructions": step.instructions,
                "verify": step.verify,
                "forbidden": step.forbidden,
            }
            for step in plan.steps
        ],
    }


def _review_plan_text(plan: TaskPlanV2) -> str:
    return _json_text(_review_plan_payload(plan))


def _compact_approved_plan_text(plan: TaskPlanV2) -> str:
    """Render the reviewer-visible plan index, never worker prompt bodies."""

    payload = {
        "title": plan.title,
        "objective": plan.objective,
        "constraints": plan.constraints,
        "required_checks": list(plan.required_checks),
        "steps": [
            {
                "id": step.id,
                "title": step.title,
                "depends_on": step.depends_on,
                "objective": step.objective,
                "invariants": step.forbidden,
                "writes": list(step.write_set),
                "creates": list(step.create_set),
                "deletes": list(step.delete_set),
                "verify": step.verify,
            }
            for step in plan.steps
        ],
    }
    return _json_text(payload)


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
                    "final", "final_message", "agent_report", "luna_reports",
                    "claude_revision_report",
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


def _bounded_review_context(text: str) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= _MAX_REVIEW_CONTEXT_BYTES:
        return text

    marker = (
        "\n[... project context truncated for reviewer; "
        "inspect the immutable candidate repository for source details ...]\n"
    ).encode("utf-8")

    head = data[: max(0, _MAX_REVIEW_CONTEXT_BYTES - len(marker))].decode(
        "utf-8", errors="ignore"
    )

    return head + marker.decode("utf-8")


def _review_payload(review: ReviewResult) -> dict[str, Any]:
    payload = asdict(review)
    payload["verdict"] = review.verdict.value
    payload["route"] = review.route.value
    payload.pop("raw", None)
    return payload


def _agent_payload(
    result: Any,
    *,
    auth_failure: bool = False,
    provider: str = "Codex",
    safe_stderr: bool = False,
) -> dict[str, Any]:
    # The full worker report is already persisted by the selected executor.
    # State contains only
    # bounded protocol metadata and never an API key or an authorization value.
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "usage": dict(result.usage or {}),
        "driver": getattr(result, "driver", None),
        "backend_reason": getattr(result, "backend_reason", None),
        # Transport errors may contain provider URLs, request IDs, or other
        # infrastructure identifiers.  The complete bounded artifact remains
        # available to the local UI; state keeps a fixed safe marker instead.
        "stderr_tail": (
            f"{provider} authentication failed" if auth_failure else ("" if safe_stderr else result.stderr_tail)
        ),
    }


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


def _codex_auth_failure(events_path: Path, stderr: str) -> bool:
    """Fixed-marker auth classification from stderr and one events file."""

    haystack = f"{stderr}\n{_artifact_tail(events_path)}".casefold()
    return any(
        marker in haystack
        for marker in (
            "401 unauthorized",
            "missing bearer or basic authentication in header",
            "authentication required",
            "not logged in",
        )
    )


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


@dataclasses.dataclass(frozen=True)
class ReviewCycleInput:
    """Cycle-specific reviewer evidence; the gate is always re-derived."""

    iteration: int
    plan_text: str
    luna_reports: str
    revision_report: str
    cycle_history: str
    scope_delta: str = ""
    deferred_mismatches: str = ""


def _bounded_parse_detail(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


# The two failures that route a check-repair attempt into the bounded
# scope-repair recovery.  Both are produced by the *same* check-repair phase
# and share the same durable evidence.
_SCOPE_VIOLATION_ORIGINS = frozenset({"REVISION_SCOPE_VIOLATION", _SCOPE_REQUEST_ROUTE})
# The check-repair phases that may own a scope violation.  A violation is
# never inferred for any other phase.
_SCOPE_VIOLATION_PHASES = frozenset({
    ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
    ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
})
# A refused resume overwrites ``state.failure`` with its own verdict.  The
# original cause is then only reachable through the durable history below.
_RESUME_REFUSAL_REASONS = frozenset({
    "RESUME_INTEGRITY_FAILURE", "RESUME_REQUIRES_OPERATOR",
})


def _scope_violation_origin(
    state: Mapping[str, Any], checkpoint: ResumeCheckpoint,
) -> str | None:
    """The scope-violation cause this checkpoint was closed by, if any.

    ``state.failure`` is the first authority.  When a previous resume already
    refused this run, its verdict replaced that failure, so the original cause
    is recovered from the resume record it preserved and -- after several
    ``--revalidate-integrity`` attempts have replaced that record too -- from
    the durable cycle the checkpoint names.  The cycle fallback is deliberately
    the weakest evidence: it is only ever consulted for the check-repair phases
    that can own a violation, and it can only ever yield the one reason a cycle
    record actually stores.
    """

    if checkpoint.phase not in _SCOPE_VIOLATION_PHASES:
        return None
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = failure.get("reason")
    if reason in _SCOPE_VIOLATION_ORIGINS:
        return str(reason)
    if reason not in _RESUME_REFUSAL_REASONS:
        return None
    resume = state.get("resume") if isinstance(state.get("resume"), Mapping) else {}
    previous = resume.get("previous_failure")
    if isinstance(previous, Mapping) and previous.get("reason") in _SCOPE_VIOLATION_ORIGINS:
        return str(previous["reason"])
    cycles = state.get("cycles")
    for cycle in cycles if isinstance(cycles, list) else ():
        if isinstance(cycle, Mapping) and cycle.get("number") == checkpoint.cycle:
            if cycle.get("failure") == "REVISION_SCOPE_VIOLATION":
                return "REVISION_SCOPE_VIOLATION"
            return None
    return None


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
    """Execute exactly one planner, one implementation agent and one review."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        planner_client: Any | None = None,
        reviewer_client: Any | None = None,
        recommender_client: Any | None = None,
        agent: Any | None = None,
        reviser: Any | None = None,
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
        self._legacy_run_options = True
        self._effective_repair_scope = EffectiveRepairScopePolicy(
            "deny-expansion", 4, "historical-default"
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
        names = {
            ExecutionRole.PLANNER: ("planner",),
            ExecutionRole.IMPLEMENTER: ("implementer", "repair_implementer"),
            ExecutionRole.REPAIR: ("check_repair", "repair_implementer"),
            ExecutionRole.REVIEWER: ("reviewer", "final_reviewer"),
            ExecutionRole.REVISER: ("reviser", "semantic_reviser"),
        }.get(role, ())
        for name in names:
            selected = getattr(selection, name, None)
            if getattr(selected, "profile_id", None) == profile_id:
                return selected
        return None

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
                    codex_home=(self.config.codex_runtime.home if profile.driver.value == "codex" else None),
                    claude_config_home=(self.config.claude_runtime.home if profile.driver.value == "claude-code" else None),
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
                profile.driver.value if profile is not None
                else getattr(selected, "driver", None)
                or getattr(result, "driver", None)
            ),
            "driver_version": None,
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

    def _planner_for_profile(self, profile_id: str) -> Planner:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.PLANNER)
        client = self._planner_client
        if client is None:
            client = _chat_client(
                build_llm_endpoint(profile), self._runtime_environment
            )
        return Planner(client, allow_format_repair=False)

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

        try:
            profile = profile_for_role(self.config, profile_id, role)
        except ProfileError:
            if role is not ExecutionRole.IMPLEMENTER:
                raise
            profile = profile_for_role(self.config, profile_id, ExecutionRole.REPAIR)
        runtime = ExecutorRuntimeConfig(
            config=self.config,
            environment=self._runtime_environment,
            codex_home=self.config.codex_runtime.home,
            claude_home=self.config.claude_runtime.home,
            forbidden_env_names=forbidden_env_names,
        )
        selected_agent = None
        if profile.driver.value == "codex":
            selected_agent = self._agent_for_profile(profile.id)
        return executor_for_profile(
            profile,
            runtime,
            agent=selected_agent,
            reviser=self._injected_reviser if profile.driver.value == "claude-code" else None,
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
        agent_config = dataclasses.replace(
            build_agent_config(profile),
            env_allowlist=self.config.agent.env_allowlist,
        )
        return CodexAgent(agent_config)

    def _run_revision(
        self,
        selection: Any,
        worktree: Path,
        run_dir: Path,
        prompt: str,
        revision_dir: Path | None = None,
        profile_id: str | None = None,
        role: ExecutionRole = ExecutionRole.REVISER,
        mutable_paths: tuple[str, ...] = (),
    ) -> Any | None:
        is_check_repair = revision_dir is not None and "check-repair" in revision_dir.parts
        selected = (
            getattr(selection, "check_repair", None)
            if is_check_repair else getattr(selection, "semantic_reviser", None)
        )
        if selected is None and not hasattr(selection, "semantic_reviser"):
            selected = getattr(selection, "reviser", None)
        if selected is None and profile_id is None:
            return None
        selected_profile_id = profile_id or selected.profile_id
        profile = profile_for_role(self.config, selected_profile_id, role)
        artifact_path = revision_dir or (run_dir / "revision")
        executor = self._executor_for_profile(profile.id, role)
        return executor.run(
            AgentRunRequest(
                role=role,
                profile_id=profile.id,
                prompt=redact(prompt, self._secrets),
                worktree=Path(worktree),
                artifact_dir=Path(artifact_path),
                mutable_paths=mutable_paths,
                prompt_mode="revision",
            )
        )

    @staticmethod
    def _cycle_update(store: RunStateStore, number: int, **fields: Any) -> None:
        """Merge one cycle record without replacing the other cycle records."""

        state = store.load()
        cycles = list(state.get("cycles") or [])
        index = next(
            (i for i, item in enumerate(cycles)
             if isinstance(item, dict) and item.get("number") == number),
            None,
        )
        record = asdict(RunCycle(number, "initial" if number == 1 else "repair"))
        if index is not None and isinstance(cycles[index], dict):
            record.update(cycles[index])
        record.update(fields)
        if index is None:
            cycles.append(record)
        else:
            cycles[index] = record
        store.update(status=state.get("status", RunStatus.PLANNING), cycles=cycles)

    def _update_v2_usage(self, store: RunStateStore, run_dir: Path) -> None:
        """Publish cycle-specific and backward-compatible token totals.

        Always derived from persisted artifacts (``steps/Sxx/step.json`` and
        ``repair/C02/steps/Sxx/step.json``), never from ``state.steps``.
        """

        store.update(
            status=store.load().get("status", RunStatus.PLANNING),
            usage=phase_usage_summary(run_dir),
        )

    @staticmethod
    def _copy_artifacts(source: Path, destination: Path, names: tuple[str, ...]) -> None:
        for name in names:
            source_path = source / name
            if not source_path.exists():
                continue
            try:
                content = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            atomic_write_text(destination / name, content)

    @classmethod
    def _publish_check_aliases(cls, run_dir: Path, checks_dir: Path) -> None:
        """Republish the canonical C01 evidence under its root aliases.

        Authority only ever flows ``checks/C01 -> run_dir``.  The reverse copy
        would let a pre-repair red evidence overwrite the corrected one.
        """

        cls._copy_artifacts(checks_dir, run_dir, _CHECK_ALIAS_ARTIFACTS)

    @classmethod
    def _snapshot_cycle_artifacts(cls, run_dir: Path) -> None:
        """Publish immutable C01 aliases before any C02 work can start."""

        cls._copy_artifacts(
            run_dir / "revision", run_dir / "revision" / "C01",
            ("agent.prompt.txt", "prompt.diagnostics.json", "agent.events.jsonl", "agent.stderr.log",
             "agent.final.md", "agent.result.json", "pre_checks.json",
             "scope.json", "tree_before.txt", "tree_after.txt",
             "report.json", "usage.json"),
        )
        cls._copy_artifacts(
            run_dir, run_dir / "review" / "C01",
            ("reviewer.request.txt", "prompt.diagnostics.json", "reviewer.raw.md",
             "reviewer.usage.json", "review.json"),
        )
        checks_dir = run_dir / "checks" / "C01"
        if (checks_dir / "evidence.json").is_file():
            # ``checks/C01`` already holds the final C01 evidence (possibly the
            # green retry after an automatic check repair): never overwrite it
            # with the stale root aliases.
            cls._publish_check_aliases(run_dir, checks_dir)
        else:
            # A run created before ``checks/C01`` became canonical kept its
            # only C01 evidence at the run root.
            cls._copy_artifacts(run_dir, checks_dir, _CHECK_ALIAS_ARTIFACTS)

    @staticmethod
    def _ensure_step_artifacts(step_dir: Path, result: Any) -> None:
        """Make the durable C02 step envelope complete for test doubles too."""

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
            legacy_run_options = run_options is None
            if run_options is None:
                overrides = {"planner_profile": planner_profile} if planner_profile is not None else {}
                # Direct callers that predate the durable run-options
                # snapshot retain the historical terminal REPLAN and
                # no-expansion behavior.  New/API callers opt into bounded
                # repair by passing an explicit RunOptions instance.
                overrides["repair_scope_policy"] = "deny-expansion"
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
        if not hasattr(self, "_runtime_environment"):
            self._runtime_environment = (
                self.config.runtime_environment
                if self.config.runtime_environment
                else os.environ
            )
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
            if run_options.protocol == "v2":
                resume_module.write_checkpoint(
                    run_dir, ResumeCheckpoint(ResumePhase.CONTEXT, 1, None, None, None, None, None)
                )
            # All downstream methods use this frozen per-run view.  The
            # caller's HarnessConfig object is never mutated.
            self._run_options = run_options
            self._legacy_run_options = legacy_run_options
            self._effective_repair_scope = effective_repair_scope_policy(
                run_options,
                raw_run_options=run_options.to_dict() if not legacy_run_options else None,
                override=None,
            )
            self.config = effective_run_config(original_config, run_options)
            store.update(
                status=RunStatus.CREATED,
                spec_path="spec.md",
                repo=str(self.config.repo),
                base_ref=self.config.base_ref,
                run_options_sha256=options_sha256,
                run_options=run_options.to_dict(),
                run_options_explicit=not legacy_run_options,
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
        initial_state = store.load()
        # Every run executed here is profile-aware: it can never fall back to
        # current defaults or to a schema-v1 approval once awaiting approval.
        if not is_profile_aware_run(initial_state):
            raise OrchestrationError("run has no planner profile")
        planner_profile_id = initial_state["execution"]["planner"]["profile_id"]
        planner_profile = profile_for_role(
            self.config, planner_profile_id, ExecutionRole.PLANNER
        )
        planner = self._planner_for_profile(planner_profile_id)
        if self.config.require_clean_base:
            assert_clean(repo)
        base_sha = resolve_commit(repo, self.config.base_ref)
        store.update(
            status=RunStatus.CREATED, repo=str(repo), base_sha=base_sha,
            planning_protocol=self.config.planning.protocol,
        )

        try:
            repository_reference = build_repository_reference(
                repo, base_sha=base_sha, config=self.config.repository
            )
        except GitError:
            # ``doctor`` remains the fail-closed preflight for enabled remote
            # exploration.  Direct programmatic callers may omit a remote;
            # the exact local base context is still sufficient to proceed.
            repository_reference = RepositoryReference(
                self.config.repository.remote, None, base_sha, None
            )
        atomic_write_text(
            run_dir / "repository_reference.json",
            json.dumps(repository_reference_dict(repository_reference), indent=2) + "\n",
        )

        # Context is a durable boundary of its own.  The checkpoint is moved
        # before the planner call and remains there when context construction
        # fails or is interrupted.
        base_tree_sha = resolve_tree(repo, base_sha)
        if self.config.planning.protocol == "v2":
            self._write_phase_checkpoint(
                run_dir, ResumePhase.CONTEXT, head=base_sha, tree=base_tree_sha
            )
        try:
            context_bundle = build_context(repo, base_sha, spec, self.config.context)
            context = render_context(context_bundle)
        except Exception:
            if self.config.planning.protocol == "v2":
                self._write_phase_checkpoint(
                    run_dir, ResumePhase.CONTEXT, head=base_sha, tree=base_tree_sha
                )
            raise
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

        if self.config.planning.protocol == "v2":
            self._write_phase_checkpoint(
                run_dir, ResumePhase.PLANNER, head=base_sha, tree=base_tree_sha
            )
        if self.config.planning.protocol == "v2":
            return self._execute_v2(
                store, run_dir, run_id, spec, repo, base_sha, context,
                repository_reference,
            )

        # The planner receives the SPEC. The implementation agent receives only
        # the canonical contract rendered from the parsed READY plan.
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
                    tree_before=base_tree_sha,
                )
            },
        )
        plan = planner.plan(spec, context, artifacts_dir=run_dir)
        plan_usage = getattr(planner, "last_usage", None)
        if plan_usage is None:
            plan_usage = read_usage_artifact(run_dir / "planner.usage.json")
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
                    usage=plan_usage,
                    tree_before=base_tree_sha,
                    tree_after=base_tree_sha,
                ),
            },
        )
        store.update(
            status=RunStatus.PLANNING,
            planner={
                "decision": plan.decision.value,
                "title": plan.title,
                "model": planner_profile.model,
                "profile_id": planner_profile.id,
                "selection_mode": planner_profile.selection_mode.value,
            },
        )
        if plan.decision is PlanDecision.BLOCKED:
            state = store.update(
                status=RunStatus.BLOCKED,
                failure={"reason": "PLANNER_BLOCKED", "detail": plan.blockers},
            )
            return RunResult(run_dir, RunStatus.BLOCKED, state)

        self._maybe_recommend_profiles(store, run_dir, planner_profile.id)

        plan_identity = compute_plan_identity_from_run(run_dir)
        store.update(
            status=RunStatus.PLANNING,
            plan_identity=asdict(plan_identity),
        )
        if self.config.approval.require_plan_approval:
            store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
            approval = wait_for_plan_approval(
                run_dir,
                identity=plan_identity,
                poll_interval_seconds=self.config.approval.poll_interval_seconds,
            )
            if approval.decision is ApprovalDecision.REJECT:
                state = store.update(status=RunStatus.PLAN_REJECTED)
                return RunResult(run_dir, RunStatus.PLAN_REJECTED, state)

            # The human-approved snapshot is the only execution authority:
            # no schema-v1 approval and no materialization of defaults.
            if approval.execution_sha256 is None:
                raise ApprovalError(
                    "profile-aware run requires a schema v2 approval bound to an execution selection"
                )
            try:
                selection, execution_sha256 = read_execution_selection_with_sha256(run_dir)
            except ExecutionSelectionError as exc:
                raise ApprovalError(f"approved execution selection is invalid: {exc}") from exc
            if selection.schema_version != 2:
                raise ApprovalError("profile-aware run requires execution selection schema 2")
            if execution_sha256 != approval.execution_sha256:
                raise ApprovalError("execution selection does not match approval")
            durable_identity = compute_plan_identity_from_run(run_dir)
            if (
                approval.raw_sha256 != durable_identity.raw_sha256
                or approval.contract_sha256 != durable_identity.contract_sha256
                or durable_identity.execution_sha256 != execution_sha256
            ):
                raise ApprovalError("approval does not match durable execution selection")
        else:
            selection = ensure_execution_selection(
                run_dir,
                resolve_execution_selection(
                    self.config,
                    planner_profile_id=planner_profile.id,
                    implementer_profile_id=self.config.ui.default_implementer_profile or "legacy-implementer",
                    reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
                ),
            )
            durable_identity = compute_plan_identity_from_run(run_dir)
            store.update(
                status=RunStatus.PLANNING,
                plan_identity=dataclasses.asdict(durable_identity),
            )

        # Prove, before any worktree, agent or reviewer, that the configured
        # profiles are exactly the snapshot that was selected.
        if selection.planner.profile_id != planner_profile.id:
            raise ExecutionSelectionError("execution selection planner is not the run planner")
        validate_execution_selection(self.config, selection)
        if selection.reviser is not None:
            # Claude revision is a META PLAN v2 capability enabled only by
            # revision.enabled; a v1 run never launches Claude Code.
            raise ExecutionSelectionError(
                "execution selection contains a reviser but revision is disabled"
            )

        self._last_selection = selection
        execution_state = {
            "planner": {
                "profile_id": selection.planner.profile_id,
                "model": selection.planner.model,
                "selection_mode": selection.planner.selection_mode,
            },
            "implementer": {
                "profile_id": selection.implementer.profile_id,
                "model": selection.implementer.model,
                "effort": selection.implementer.effort,
                "selection_mode": selection.implementer.selection_mode,
            },
            "reviewer": {
                "profile_id": selection.reviewer.profile_id,
                "model": selection.reviewer.model,
                "selection_mode": selection.reviewer.selection_mode,
            },
        }
        if selection.reviser is not None:
            execution_state["reviser"] = {
                "profile_id": selection.reviser.profile_id,
                "model": selection.reviser.model,
                "effort": selection.reviser.effort,
                "permission_mode": selection.reviser.permission_mode,
                "selection_mode": selection.reviser.selection_mode,
            }
        store.update(
            status=RunStatus.PLANNING,
            execution=execution_state,
            plan_identity=dataclasses.asdict(durable_identity),
        )
        self._trace_emit(
            "plan.approved",
            phase="planning",
            cycle=1,
            data={
                "plan_identity": dataclasses.asdict(durable_identity),
                "execution_selection_sha256": durable_identity.execution_sha256,
            },
            once=True,
        )

        branch = f"harness/{_slug(plan.title)}/{run_id}"
        worktree_path = self.config.worktrees_root / run_id
        info = create_run_worktree(
            repo,
            base_ref=base_sha,
            branch=branch,
            worktree_path=worktree_path,
            require_clean_base=self.config.require_clean_base,
        )
        branch_ref = f"refs/heads/{info.branch}"
        store.update(
            status=RunStatus.WORKTREE_READY,
            branch=info.branch,
            worktree=str(info.worktree),
            base_sha=info.base_sha,
        )
        self._trace_emit(
            "worktree.created",
            phase="setup",
            cycle=1,
            data={
                "branch": info.branch,
                "worktree": str(info.worktree),
                "base_sha": info.base_sha,
                "tree_sha": resolve_tree(info.worktree, info.base_sha),
            },
            once=True,
        )

        ownership_before = _git_ownership(repo, info.worktree)
        # The durable ownership proof is persisted with the status of the
        # write it belongs to: the state store owns the status of every update.
        store.update(
            status=RunStatus.PREPARING,
            git_ownership=_git_ownership_payload(ownership_before),
        )
        try:
            setup_results = prepare_workspace(
                info.worktree,
                self.config.workspace_setup,
                environment=self._runtime_environment,
                artifacts_dir=run_dir,
                secrets=self._secrets,
            )
        except WorkspaceSetupError as exc:
            if exc.results:
                store.update(
                    status=RunStatus.PREPARING,
                    workspace_setup=[asdict(result) for result in exc.results],
                )
            raise
        store.update(
            status=RunStatus.PREPARING,
            workspace_setup=[asdict(result) for result in setup_results],
        )
        store.update(status=RunStatus.IMPLEMENTING)
        implementation_contract = render_implementation_contract(plan)
        implementation_payload = build_implementer_payload(
            step_title=getattr(plan, "title", ""),
            step_objective=getattr(plan, "objective", ""),
            step_invariants=getattr(plan, "constraints", ""),
            read_set=getattr(plan, "files", "NONE"),
            mutable_scope=getattr(plan, "files", "NONE"),
            repository_instructions=getattr(plan, "implementation", ""),
            verify_instructions=getattr(plan, "tests", ""),
            budget_bytes=self.config.prompt_budget.implementer_max_bytes,
        )
        implementation_contract = implementation_payload.rendered
        implementer_profile = profile_for_role(
            self.config, selection.implementer.profile_id, ExecutionRole.IMPLEMENTER
        )
        v1_step_id = "S01"
        trace_started_at = self._trace_time()
        trace_started_mono = time.perf_counter()
        trace_selected = self._trace_selected_profile(
            implementer_profile.id, ExecutionRole.IMPLEMENTER
        )
        tree_before_agent = _safe_candidate_tree(info.worktree)

        def trace_v1_step_failed(reason: str, detail: Any = None) -> None:
            self._trace_emit(
                "step.failed",
                phase="implementation",
                cycle=1,
                step_id=v1_step_id,
                data={
                    "reason": reason,
                    "detail": detail,
                    "profile_id": implementer_profile.id,
                    "tree_before": tree_before_agent,
                    "tree_after": _safe_candidate_tree(info.worktree),
                    "changed_paths": [],
                    "commit_sha": None,
                },
            )

        try:
            executor = self._executor_for_profile(
                implementer_profile.id,
                ExecutionRole.IMPLEMENTER,
                forbidden_env_names=(
                    planner_profile.api_key_env,
                    profile_for_role(
                        self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER
                    ).api_key_env,
                ),
            )
            tree_before_agent = candidate_tree_sha(info.worktree)
            self._trace_emit(
                "step.started",
                phase="implementation",
                cycle=1,
                step_id=v1_step_id,
                data={
                    "attempt": 1,
                    "tree_before": tree_before_agent,
                    "session": self._trace_session(
                        profile=implementer_profile,
                        selected=trace_selected,
                        role=ExecutionRole.IMPLEMENTER,
                        prompt_bytes=len(implementation_contract.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before_agent,
                    ),
                },
            )
            write_prompt_diagnostics(
                run_dir,
                implementation_payload,
                filename="prompt.diagnostics.implementer.json",
            )
            agent_result = executor.run(
                AgentRunRequest(
                    role=ExecutionRole.IMPLEMENTER,
                    profile_id=implementer_profile.id,
                    prompt=implementation_contract,
                    worktree=info.worktree,
                    artifact_dir=run_dir,
                    mutable_paths=tuple(
                        sorted(
                            {
                                path
                                for path in (
                                    *getattr(plan, "read_set", ()),
                                    *getattr(plan, "write_set", ()),
                                )
                                if isinstance(path, str)
                            }
                        )
                    ),
                    prompt_mode="plan",
                )
            )
        except AgentScopeError as exc:
            trace_v1_step_failed(AGENT_SCOPE_VIOLATION, redact(str(exc), self._secrets))
            self._trace_emit(
                "step.agent.completed",
                phase="implementation",
                cycle=1,
                step_id=v1_step_id,
                data={
                    "attempt": 1,
                    "status": "failed",
                    "session": self._trace_session(
                        profile=implementer_profile,
                        selected=trace_selected,
                        role=ExecutionRole.IMPLEMENTER,
                        prompt_bytes=len(implementation_contract.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before_agent,
                        exit_reason=getattr(exc, "code", type(exc).__name__),
                    ),
                },
            )
            self._redact_agent_artifacts(run_dir)
            state = store.record_failure(
                AGENT_SCOPE_VIOLATION, redact(str(exc), self._secrets),
                agent={"driver": implementer_profile.driver.value},
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        auth_failure = (
            not agent_result.timed_out
            and agent_result.exit_code != 0
            and agent_result.backend_reason == "CODEX_AUTH_FAILURE"
        )
        self._redact_agent_artifacts(run_dir)
        agent_result = dataclasses.replace(
            agent_result,
            final_message=redact(agent_result.final_message, self._secrets),
            stderr_tail=redact(agent_result.stderr_tail, self._secrets),
        )
        self._trace_emit(
            "step.agent.completed",
            phase="implementation",
            cycle=1,
            step_id=v1_step_id,
            data={
                "attempt": 1,
                "status": agent_result.status,
                "session": self._trace_session(
                    profile=implementer_profile,
                    selected=trace_selected,
                    role=ExecutionRole.IMPLEMENTER,
                    prompt_bytes=len(implementation_contract.encode("utf-8", errors="replace")),
                    started_at=trace_started_at,
                    started_mono=trace_started_mono,
                    tree_before=tree_before_agent,
                    result=agent_result,
                ),
            },
        )
        store.update(
            status=RunStatus.IMPLEMENTING,
            agent=_agent_payload(agent_result, auth_failure=auth_failure),
        )
        violations = _ownership_violations(
            ownership_before,
            _git_ownership(repo, info.worktree),
            branch_ref=branch_ref,
            base_sha=base_sha,
        )
        if violations:
            trace_v1_step_failed("AGENT_GIT_VIOLATION", violations)
            state = store.record_failure("AGENT_GIT_VIOLATION", violations)
            return RunResult(run_dir, RunStatus.FAILED, state)
        if agent_result.timed_out:
            trace_v1_step_failed(AGENT_TIMEOUT)
            state = store.record_failure(
                AGENT_TIMEOUT,
                {"driver": agent_result.driver, "backend_reason": agent_result.backend_reason},
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        agent_failure = normalized_failure_reason(agent_result)
        if agent_failure is not None:
            trace_v1_step_failed(agent_failure)
            if auth_failure:
                reason = (
                    "CODEX_AUTH_FAILURE"
                    if self._legacy_backend_injection else AGENT_RUNTIME_FAILED
                )
                state = store.record_failure(
                    reason,
                    "Codex authentication failed"
                    if self._legacy_backend_injection
                    else {
                        "driver": agent_result.driver,
                        "backend_reason": agent_result.backend_reason,
                    },
                )
                return RunResult(run_dir, RunStatus.FAILED, state)
            reason = "AGENT_FAILED" if self._legacy_backend_injection else agent_failure
            state = store.record_failure(
                reason,
                {
                    "exit_code": agent_result.exit_code,
                    "driver": agent_result.driver,
                    "backend_reason": agent_result.backend_reason,
                    "agent_reason": agent_failure,
                },
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

        tree_after_agent = candidate_tree_sha(info.worktree)
        store.update(
            status=RunStatus.IMPLEMENTING,
            agent_candidate_tree_before=tree_before_agent,
            agent_candidate_tree_after=tree_after_agent,
        )
        if tree_after_agent == tree_before_agent:
            trace_v1_step_failed("AGENT_NO_CHANGE")
            state = store.record_failure("AGENT_NO_CHANGE")
            return RunResult(run_dir, RunStatus.FAILED, state)

        store.update(status=RunStatus.VALIDATING)
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
        self._trace_cycle = 1
        self._trace_emit(
            "checks.started",
            phase="validation",
            cycle=1,
            data={"tree_before": _safe_candidate_tree(info.worktree)},
        )
        evidence = collect_evidence(
            info.worktree,
            base_sha,
            self.config,
            required_check_ids=getattr(plan, "required_checks", ()) or None,
            evidence_dir=run_dir,
            secrets=self._secrets,
        )
        self._trace_emit(
            "checks.completed",
            phase="validation",
            cycle=1,
            data={
                "passed": evidence.deterministic_passed,
                "failures": list(evidence.failures),
                "required_check_ids": list(evidence.required_check_ids),
                "tree_sha": evidence.staged_tree_sha,
                "changed_paths": list(evidence.changed_files),
            },
        )
        self._trace_emit(
            "step.verification.completed",
            phase="implementation",
            cycle=1,
            step_id=v1_step_id,
            data={
                "status": "passed" if evidence.deterministic_passed else "failed",
                "tree_before": tree_before_agent,
                "tree_after": evidence.staged_tree_sha,
                "deferred": False,
            },
        )
        store.update(
            status=RunStatus.VALIDATING,
            checks=_check_payload(evidence),
            staged_tree_sha=evidence.staged_tree_sha,
            changed_files=list(evidence.changed_files),
            deterministic_gate={
                "passed": evidence.deterministic_passed,
                "required_check_ids": list(evidence.required_check_ids),
                "failures": list(evidence.failures),
            },
        )

        integrity_failures = [
            item
            for item in evidence.failures
            if (
                item in _LEGACY_DIRECT_FAILURES
                or any(item.startswith(f"{prefix}:") for prefix in _LEGACY_DIRECT_FAILURES)
                or item.startswith("CHECK_MUTATED:")
            )
        ]
        if integrity_failures:
            reason = integrity_failures[0].split(":", 1)[0]
            trace_v1_step_failed(reason, ", ".join(integrity_failures))
            state = store.record_failure(reason, ", ".join(integrity_failures))
            return RunResult(run_dir, RunStatus.FAILED, state)

        gate = _json_text(
            {
                "deterministic_passed": evidence.deterministic_passed,
                "required_check_ids": list(evidence.required_check_ids),
                "failures": list(evidence.failures),
                "staged_tree_sha": evidence.staged_tree_sha,
            }
        )
        checks_text = _json_text(_check_payload(evidence))
        store.update(status=RunStatus.REVIEWING)
        # The reviewer receives SPEC and PLAN so it can route a defect to
        # IMPLEMENTATION or REPLAN.  Diff, checks and report are review data.
        review_started_at = self._trace_time()
        review_started_mono = time.perf_counter()
        v1_reviewer_profile = profile_for_role(
            self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER
        )
        v1_reviewer_selected = self._trace_selected_profile(
            selection.reviewer.profile_id, ExecutionRole.REVIEWER
        )
        self._trace_emit(
            "review.started",
            phase="review",
            cycle=1,
            data={
                "tree_sha": evidence.staged_tree_sha,
                "session": self._trace_session(
                    profile=v1_reviewer_profile,
                    selected=v1_reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=None,
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    tree_before=evidence.staged_tree_sha,
                ),
            },
        )
        review = reviewer.review(
            spec,
            plan.raw,
            context,
            gate,
            "\n".join(evidence.changed_files),
            evidence.diff,
            checks_text,
            _bounded_report(agent_result.final_message),
            # The deterministic gate is evaluated cumulatively below.  Passing
            # it here allows a reviewer PASS to remain a useful diagnostic on
            # a failed check, as required by V0.
            deterministic_passed=True,
            artifacts_dir=run_dir,
            diagnostics_filename="prompt.diagnostics.final-reviewer.json",
        )
        self._trace_emit(
            "review.completed",
            phase="review",
            cycle=1,
            data={
                "tree_sha": evidence.staged_tree_sha,
                "verdict": review.verdict.value,
                "route": review.route.value,
                "session": self._trace_finished_model_session(
                    profile=v1_reviewer_profile,
                    selected=v1_reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=(
                        (run_dir / "reviewer.request.txt").stat().st_size
                        if (run_dir / "reviewer.request.txt").is_file() else None
                    ),
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    usage=(
                        getattr(reviewer, "last_usage", None)
                        if getattr(reviewer, "last_usage", None) is not None
                        else read_usage_artifact(run_dir / "reviewer.usage.json")
                    ),
                    tree_before=evidence.staged_tree_sha,
                    tree_after=evidence.staged_tree_sha,
                ),
            },
        )
        store.update(
            status=RunStatus.REVIEWING,
            review=_review_payload(review),
            review_iterations=1,
        )

        if review.verdict is ReviewVerdict.REVISE:
            # V0 never re-implements automatically: REVISE produces an
            # inspectable repair task and ends the run.
            write_repair_task(
                run_dir,
                fields={
                    "route": review.route.value,
                    "review_summary": review.summary,
                    "findings": review.findings,
                    "required_fixes": review.required_fixes,
                    "missing_tests": review.missing_tests,
                    "existing_branch": info.branch,
                    "existing_worktree": str(info.worktree),
                    "run_id": run_id,
                },
            )
            state = store.record_failure("REVIEW_REVISE")
            return RunResult(run_dir, RunStatus.FAILED, state)
        if review.verdict is ReviewVerdict.FAIL:
            state = store.record_failure("REVIEW_FAIL")
            return RunResult(run_dir, RunStatus.FAILED, state)
        if review.route is not ReviewRoute.NONE or not evidence.deterministic_passed:
            reason = "REVIEW_ROUTE_NOT_NONE" if review.route is not ReviewRoute.NONE else "DETERMINISTIC_GATE_FAILED"
            if not evidence.deterministic_passed:
                trace_v1_step_failed(reason)
            state = store.record_failure(reason)
            return RunResult(run_dir, RunStatus.FAILED, state)

        store.update(
            status=RunStatus.APPROVED,
            approved_tree_sha=evidence.staged_tree_sha,
        )
        approved_tree = authorize_commit(
            plan=plan,
            agent_result=agent_result,
            evidence=evidence,
            review=review,
            worktree=info.worktree,
            base_sha=base_sha,
            branch_ref=branch_ref,
        )
        commit_sha = commit_reviewed_tree(
            info.worktree,
            tree_sha=approved_tree,
            parent_sha=base_sha,
            subject=_commit_subject(plan.title),
            body=f"MetaHarness-Run: {run_id}",
        )
        self._trace_emit(
            "step.committed",
            phase="implementation",
            cycle=1,
            step_id=v1_step_id,
            data={
                "parent_sha": base_sha,
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "changed_paths": list(evidence.changed_files),
                **self._trace_diff_reference(run_dir / "diff.patch"),
            },
        )
        return self._complete_commit(
            store=store,
            run_dir=run_dir,
            info=info,
            approved_tree=approved_tree,
            commit_sha=commit_sha,
            repository_reference=repository_reference,
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
        pipeline_enabled = revision_enabled or repair_enabled or check_repair_enabled
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
                self._write_phase_checkpoint(
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
            # Freeze the whole trusted catalogue, not just the C01 selection:
            # a C02 repair plan may legitimately require another approved
            # check, and it must still run the argv approved at this boundary.
            write_check_authority(
                run_dir, tuple(self.config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            _bundle, _bundle_sha = validate_implementation_bundle(run_dir)
            plan_identity = compute_plan_identity_from_run(run_dir)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update(status=RunStatus.PLANNING, plan_identity=asdict(plan_identity))
        self._write_phase_checkpoint(
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
                if pipeline_enabled:
                    selection, execution_sha = read_execution_selection_v5_with_sha256(run_dir)
                    validate_execution_selection_v5(self.config, selection)
                else:
                    selection, execution_sha = read_execution_selection_v3_with_sha256(run_dir)
                    validate_execution_selection_v3(self.config, selection)
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
            if pipeline_enabled:
                requested = resolve_execution_selection_v5(
                    self.config,
                    planner_profile_id=planner_profile_id,
                    step_profile_ids={
                        step.id: self.config.ui.default_implementer_profile or step.implementer_profile
                        for step in plan.steps
                    },
                    semantic_reviser_profile_id=(
                        self._run_options.semantic_reviser_profile
                        if revision_enabled else None
                    ),
                    check_repair_profile_id=(
                        self._run_options.check_repair_profile
                        if check_repair_enabled or repair_enabled else None
                    ),
                    final_reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
                )
                selection = ensure_execution_selection_v5(run_dir, requested)
            else:
                requested = resolve_execution_selection_v3(
                    self.config,
                    planner_profile_id=planner_profile_id,
                    step_profile_ids={
                        step.id: self.config.ui.default_implementer_profile or step.implementer_profile
                        for step in plan.steps
                    },
                    reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
                )
                selection = ensure_execution_selection_v3(run_dir, requested)
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
        if pipeline_enabled:
            if not isinstance(selection, ExecutionSelectionV5) or selection.schema_version != 5:
                raise ExecutionSelectionError("v2 execution requires execution selection schema 5")
            validate_execution_selection_v5(self.config, selection)
        else:
            validate_execution_selection_v3(self.config, selection)
            if selection.reviser is not None:
                # A historical v3 snapshot carrying a reviser would launch
                # Claude while revision is disabled: fail closed instead.
                raise ExecutionSelectionError(
                    "execution selection contains a reviser but revision is disabled"
                )
        self._last_selection = selection
        execution_state: dict[str, Any] = {
            "planner": asdict(selection.planner),
            "steps": [
                {"step_id": item.step_id, "implementer": asdict(item.implementer)}
                for item in selection.steps
            ],
        }
        if isinstance(selection, ExecutionSelectionV5):
            if selection.semantic_reviser is not None:
                execution_state["semantic_reviser"] = asdict(selection.semantic_reviser)
            if selection.check_repair is not None:
                execution_state["check_repair"] = asdict(selection.check_repair)
            execution_state["final_reviewer"] = asdict(selection.final_reviewer)
        elif isinstance(selection, ExecutionSelectionV4):
            execution_state["reviser"] = asdict(selection.reviser)
            execution_state["repair_implementer"] = asdict(selection.repair_implementer)
            execution_state["reviewer"] = asdict(selection.reviewer)
        else:
            execution_state["reviewer"] = asdict(selection.reviewer)
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
        self._write_phase_checkpoint(
            run_dir, ResumePhase.WORKTREE_SETUP, head=base_sha,
            tree=base_tree_sha, plan_identity=durable_identity,
            execution_selection_sha256=durable_identity.execution_sha256,
        )
        # Plan approval complete: the next operation is the first Luna step.
        checkpoint = self._initial_checkpoint(plan, base_sha, base_tree_sha, durable_identity)
        if checkpoint is not None:
            write_checkpoint(run_dir, checkpoint)

        branch = f"harness/{_slug(plan.title)}/{run_id}"
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
        # Worktree and setup complete: still the first Luna step.
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
            ResumePhase.INITIAL_STEP, 1, plan.steps[0].id, base_sha, base_tree_sha,
            identity.execution_sha256, identity,
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
    def _write_phase_checkpoint(
        run_dir: Path,
        phase: ResumePhase,
        *,
        head: str | None,
        tree: str | None,
        plan_identity: PlanIdentity | None = None,
        execution_selection_sha256: str | None = None,
        step_id: str | None = None,
        cycle: int | None = None,
        repair_bundle_sha256: str | None = None,
        scope_delta_sha256: str | None = None,
        check_repair_attempt: int | None = None,
        expected_parent_sha: str | None = None,
        next_step_id: str | None = None,
    ) -> None:
        """Write a new-schema boundary without changing the old hook API."""

        try:
            record = read_checkpoint_record(run_dir)
        except ResumeCheckpointError:
            # A malformed checkpoint is evidence of lost authority.  Do not
            # silently replace it with a new boundary; the next resume must
            # remain fail-closed and expose the corruption to the operator.
            raise
        if record is None:
            return
        previous = record[0]
        identity = plan_identity if plan_identity is not None else previous.plan_identity
        execution = (
            execution_selection_sha256
            if execution_selection_sha256 is not None else previous.execution_selection_sha256
        )
        resume_module.write_checkpoint(
            run_dir,
            ResumeCheckpoint(
                phase=phase,
                cycle=cycle or (2 if phase_index(phase) >= phase_index(ResumePhase.REPAIR_PLANNER) else 1),
                step_id=step_id,
                expected_head_sha=head,
                expected_tree_sha=tree,
                execution_selection_sha256=execution,
                plan_identity=identity,
                repair_bundle_sha256=repair_bundle_sha256 or previous.repair_bundle_sha256,
                scope_delta_sha256=scope_delta_sha256 or previous.scope_delta_sha256,
                check_repair_attempt=check_repair_attempt,
                expected_parent_sha=expected_parent_sha or previous.expected_parent_sha,
                next_step_id=next_step_id,
            ),
        )

    def _checkpoint(
        self,
        run_dir: Path,
        phase: ResumePhase,
        *,
        head: str | None,
        tree: str | None,
        cycle: int | None = None,
        step_id: str | None = None,
        repair_bundle_sha256: str | None = None,
        scope_delta_sha256: str | None = None,
        check_repair_attempt: int | None = None,
        expected_parent_sha: str | None = None,
        next_step_id: str | None = None,
    ) -> None:
        """Persist the next operation that has not yet succeeded.

        Identity fields are carried over from the run's current checkpoint;
        a run without one (v1, legacy callers) has nothing to resume.
        """

        try:
            record = read_checkpoint_record(run_dir)
        except ResumeCheckpointError:
            raise
        if record is None or record[1] != "pending" or head is None or tree is None:
            return
        previous = record[0]
        scope_repair_phase = phase in {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        }
        scope_repair_planner_phase = phase in {
            ResumePhase.CHECK_SCOPE_PLANNER_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02,
        }
        if scope_repair_planner_phase and repair_bundle_sha256 is None:
            repair = None
        elif phase_index(phase) <= phase_index(ResumePhase.REPAIR_PLANNER) and not scope_repair_phase:
            repair = None
        else:
            repair = repair_bundle_sha256 or previous.repair_bundle_sha256
        if scope_repair_planner_phase and scope_delta_sha256 is None:
            scope_delta = None
        else:
            scope_delta = scope_delta_sha256 or previous.scope_delta_sha256
        if cycle is None:
            cycle = 2 if phase_index(ResumePhase.REPAIR_PLANNER) <= phase_index(phase) < phase_index(ResumePhase.PUBLISH) else 1
        write_checkpoint(run_dir, ResumeCheckpoint(
            phase=phase,
            cycle=cycle,
            step_id=step_id,
            expected_head_sha=head,
            expected_tree_sha=tree,
            execution_selection_sha256=previous.execution_selection_sha256,
            plan_identity=previous.plan_identity,
            repair_bundle_sha256=repair,
            scope_delta_sha256=scope_delta,
            check_repair_attempt=check_repair_attempt,
            expected_parent_sha=expected_parent_sha or previous.expected_parent_sha,
            next_step_id=next_step_id,
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
        resumed: "_ResumedRun | None" = None,
        prepared: "_V2Setup | RunResult | None" = None,
    ) -> RunResult:
        """Execute a v2 bundle: one worktree, fresh Codex process per step.

        With *resumed*, planning, approval and workspace setup are never
        replayed: execution restarts at the checkpoint's next operation and
        every earlier result is read back from its durable artifacts.
        """

        # The configuration is the only authority for P25/P26.  It was fully
        # cross-validated at load time; profiles in the catalogue never
        # enable Claude or C02 implicitly.
        revision_enabled = self._run_options.semantic_revision_enabled
        repair_enabled = self._run_options.max_review_repair_cycles > 0
        check_repair_enabled = self._run_options.max_check_repair_attempts > 0

        planner_profile_id = store.load()["execution"]["planner"]["profile_id"]
        planner_profile = profile_for_role(self.config, planner_profile_id, ExecutionRole.PLANNER)
        if resumed is None:
            prepared = prepared or self._prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                repository_reference, planner_profile,
            )
            if isinstance(prepared, RunResult):
                return prepared
            plan, bundle, selection = prepared.plan, prepared.bundle, prepared.selection
            info, ownership_before = prepared.info, prepared.ownership_before
            base_tree_sha = prepared.base_tree_sha
            if prepared.checkpoint is None:
                # A v2 run always has an execution selection bound to its
                # plan identity once approved; never run without one.
                raise OrchestrationError("v2 run has no execution selection identity")
            start = prepared.checkpoint
            completed_steps: list[dict[str, Any]] = []
        else:
            plan, bundle, selection = resumed.plan, resumed.bundle, resumed.selection
            info = resumed.info
            ownership_before = _git_ownership(repo, info.worktree)
            base_tree_sha = resolve_tree(repo, base_sha)
            start = resumed.checkpoint
            completed_steps = list(resumed.c01_steps)
            self._last_selection = selection
        branch_ref = f"refs/heads/{info.branch}"
        phase = start.phase
        at = phase_index(phase)
        # The selected executor owns preparation of its managed runtime home.
        codex_home = self.config.codex_runtime.home
        forbidden_env_names = (
            planner_profile.api_key_env,
            profile_for_role(self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER).api_key_env,
        )
        step_items = {item.step_id: item for item in selection.steps}
        done_ids = {record["id"] for record in completed_steps}
        state_steps = [
            {"id": step.id, "title": step.title,
             "status": next(
                 (record["status"].lower() for record in completed_steps if record["id"] == step.id),
                 "waiting",
             ),
             "profile_id": step_items[step.id].implementer.profile_id}
            for step in plan.steps
        ]
        self._last_v2_step_results: list[dict[str, Any]] = list(completed_steps)
        self._repair_v2_step_results: list[dict[str, Any]] = []
        self._trace_cycle = 1
        self._v2_usage_rows: list[dict[str, Any]] = [
            {"id": record["id"], **record["usage"]} for record in completed_steps
        ]
        if resumed is not None and resumed.scope_violation_recovery is not None and phase in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
            ResumePhase.CANDIDATE_COMMIT_C02, ResumePhase.CANDIDATE_PUSH_C02,
            ResumePhase.REVIEWER_C02,
        }:
            try:
                return self._execute_scope_repair_cycle(
                    store=store, run_dir=run_dir, run_id=run_id, spec=spec,
                    repo=repo, base_sha=base_sha, repository_reference=repository_reference,
                    info=info, branch_ref=branch_ref,
                    ownership_before=ownership_before, selection=selection,
                    original_plan=plan, original_bundle=bundle, resumed=resumed,
                    cycle=1 if resumed.checkpoint.cycle == 1 else 2,
                )
            except ScopeApprovalRequired:
                return RunResult(run_dir, RunStatus.WAITING_SCOPE_APPROVAL, store.load())
            except StepExecutionFailure as failure:
                return self._step_failed(
                    store, run_dir, failure,
                    run_dir / "scope-repair" / ("C01" if resumed.checkpoint.cycle == 1 else "C02")
                    / "steps" / failure.step_id,
                )
            except LLMError as exc:
                return self._v2_failed(store, run_dir, "LLM_FAILURE", None, _bounded_parse_detail(exc))
            except OrchestrationError as exc:
                reason = str(exc).split(":", 1)[0].strip() or "SCOPE_REPAIR_FAILED"
                return self._v2_failed(store, run_dir, reason, None, _bounded_parse_detail(exc))
        if resumed is not None and at <= phase_index(ResumePhase.REVIEWER_C01):
            store.update(status=store.load().get("status", RunStatus.IMPLEMENTING),
                         steps=state_steps, current_step=None, cycle=1)
        # Tree every step must start from: the base, then each frozen step.
        expected_tree = start.expected_tree_sha
        retry_step = start.step_id if resumed is not None and phase is ResumePhase.INITIAL_STEP else None
        # A clean mismatch persisted by an earlier run: this run performs that
        # step's single bounded retry, with the addendum and no replay.
        pending_retries = dict(resumed.mismatch_retries) if resumed is not None else {}
        for index, step in enumerate(plan.steps if phase is ResumePhase.INITIAL_STEP else ()):
            if step.id in done_ids:
                continue
            selected_step = step_items.get(step.id)
            if selected_step is None:
                raise ExecutionSelectionError(f"missing selection for {step.id}")
            # The approved file is the executed file: its bytes are re-hashed
            # against the validated bundle and never re-rendered or rewritten.
            try:
                contract = read_approved_step_contract(run_dir, bundle, step.id)
            except (V2PlanParseError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
            if step.id == retry_step:
                _archive_attempt(run_dir / "steps" / step.id)
            store.update(status=RunStatus.IMPLEMENTING, current_step=step.id,
                         steps=[{**item, "status": "running" if item["id"] == step.id else item["status"]}
                                for item in state_steps])
            step_parent_sha = current_head(info.worktree)
            step_ownership_before = _git_ownership(repo, info.worktree)
            try:
                outcome = self._execute_codex_step(
                    repo=repo, worktree=info.worktree, base_sha=step_parent_sha,
                    branch_ref=branch_ref, ownership_before=step_ownership_before,
                    expected_tree=expected_tree, step=step, contract=contract,
                    profile_id=selected_step.implementer.profile_id,
                    artifact_dir=run_dir / "steps" / step.id,
                    codex_home=codex_home, forbidden_env_names=forbidden_env_names,
                    future_ownership=_future_step_ownership(plan.steps, index),
                    pending_mismatch_retry=pending_retries.pop(step.id, None),
                )
            except StepExecutionFailure as failure:
                if failure.usage is not None:
                    self._v2_usage_rows.append({"id": step.id, **failure.usage})
                return self._step_failed(store, run_dir, failure, run_dir / "steps" / step.id)
            try:
                accepted_commit_sha = self._accept_v2_step_tree(
                    store=store,
                    run_dir=run_dir,
                    info=info,
                    step=step,
                    outcome=outcome,
                    parent_sha=step_parent_sha,
                    future_step_ids=tuple(item.id for item in plan.steps[index + 1:]),
                    run_id=run_id,
                )
            except (CommitSafetyError, GitError) as exc:
                return self._v2_failed(
                    store, run_dir, "COMMIT_GATE_FAILED", step.id,
                    _bounded_parse_detail(exc),
                )
            self._v2_usage_rows.append({"id": step.id, **outcome.usage})
            expected_tree = outcome.tree_after
            ownership_before = _git_ownership(repo, info.worktree)
            self._last_v2_step_results.append(_step_result_record(outcome))
            state_steps = [
                {**item, "status": "deferred" if getattr(outcome, "status", "COMPLETED") == "DEFERRED_CONTRACT_MISMATCH" else "completed", "usage": outcome.usage,
                 "input_tokens": outcome.usage["input_tokens"],
                 "output_tokens": outcome.usage["output_tokens"]}
                if item["id"] == step.id else item
                for item in state_steps
            ]
            store.update(status=RunStatus.IMPLEMENTING, current_step=None, steps=state_steps,
                         agent_usage=self._v2_agent_usage())
            following = plan.steps[index + 1].id if index + 1 < len(plan.steps) else None
            checkpoint_parent_sha = step_parent_sha
            if accepted_commit_sha is None:
                # A deferred no-change outcome has no legal empty commit.  Its
                # durable HEAD is still the step parent, so the checkpoint
                # records that commit's real parent rather than pretending a
                # new accepted commit exists.
                parents = commit_parents(info.worktree, current_head(info.worktree))
                checkpoint_parent_sha = parents[0] if parents else current_head(info.worktree)
            # Step complete: the next operation is the next step, then the
            # C01 checks (pre-revision checks with Claude, final without).
            self._checkpoint(
                run_dir,
                ResumePhase.INITIAL_STEP if following else ResumePhase.CHECKS_C01,
                step_id=following, head=current_head(info.worktree), tree=outcome.tree_after,
                expected_parent_sha=checkpoint_parent_sha, next_step_id=following,
            )

        deferred_mismatches = _deferred_contract_mismatches(
            plan, self._last_v2_step_results
        )
        # Subsequent gates are still evaluated against the immutable run
        # base for evidence, but their Git authority is the accepted step tip.
        accepted_head_sha = current_head(info.worktree)
        if _has_deferred_contract_mismatches(self._last_v2_step_results) and not revision_enabled:
            return self._v2_failed(
                store, run_dir, "UNRESOLVED_CONTRACT_MISMATCH", None,
                "HUMAN_REQUIRED: Claude revision is disabled while Luna contract mismatches are deferred",
            )

        # With Claude, CHECKS_C01 names the pending pre-revision checks and
        # FINAL_CHECKS_C01 the final checks after a durable Claude revision.
        final_checks_phase = ResumePhase.FINAL_CHECKS_C01 if revision_enabled else ResumePhase.CHECKS_C01
        revision_result = None
        revision_report_c01 = ""
        check_repair_result_c01 = None
        check_repair_scope_c01: CheckRepairScope | None = None
        expanded_check_repair_result_c01 = None
        if revision_enabled:
            if at <= phase_index(ResumePhase.CLAUDE_C01):
                if resumed is not None and phase is ResumePhase.CHECKS_C01:
                    _archive_attempt(run_dir / "revision", names=_PRE_CHECK_ATTEMPT_ARTIFACTS)
                if resumed is not None and phase is ResumePhase.CLAUDE_C01:
                    _archive_attempt(run_dir / "revision", names=_REVISION_ATTEMPT_ARTIFACTS)
                try:
                    revision_result, revision_error = self._run_v2_revision_cycle(
                        store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                        base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                        repository_reference=repository_reference, info=info,
                        branch_ref=branch_ref, ownership_before=ownership_before,
                        selection=selection,
                        deferred_mismatches=deferred_mismatches,
                        deferred_mismatch_present=_has_deferred_contract_mismatches(
                            self._last_v2_step_results
                        ),
                    )
                except AgentScopeError as exc:
                    self._redact_revision_artifacts(run_dir)
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None,
                                           redact(str(exc), self._secrets))
                except AgentError as exc:
                    self._redact_revision_artifacts(run_dir)
                    _record_failure_tree(run_dir / "revision", info.worktree)
                    return self._v2_failed(store, run_dir, "CLAUDE_FAILED", None,
                                           redact(str(exc), self._secrets))
                if revision_error is not None:
                    return self._v2_failed(store, run_dir, revision_error, None)
                self._cycle_update(
                    store, 1, status="completed",
                    claude_revision_report=revision_result.final_message if revision_result else "",
                )
            elif resumed is not None:
                revision_result = resumed.c01_revision
            if revision_result is not None:
                revision_report_c01 = _revision_report_text(revision_result, run_dir / "revision")

        c01_candidate: dict[str, Any]
        # ``FINAL_CHECKS_RETRY_C01`` is its own durable boundary: the normal
        # check repair already succeeded there, and the retry checks it names
        # may legitimately have produced no evidence at all.  The checkpoint,
        # never ``state.failure``, decides; the bridge below owns the phase.
        retry_bridge_c01 = (
            self._legacy_run_options and revision_enabled and resumed is not None
            and phase is ResumePhase.FINAL_CHECKS_RETRY_C01
        )
        if at <= phase_index(ResumePhase.FINAL_CHECKS_C01):
            store.update(
                status=RunStatus.REVALIDATING if revision_enabled else RunStatus.VALIDATING,
                current_step=None,
            )
            # The final checks evaluate exactly the checkpointed tree; a failure
            # keeps that boundary instead of blessing whatever the checks left.
            checks_tree = candidate_tree_sha(info.worktree)
            checks_dir_c01 = run_dir / "checks" / "C01"
            checks_dir_c01.mkdir(parents=True, exist_ok=True)
            try:
                if resumed is not None and phase is final_checks_phase:
                    _archive_attempt(checks_dir_c01, names=_CHECK_ATTEMPT_ARTIFACTS)
                evidence = self._final_evidence(
                    info.worktree, base_sha, checks_dir_c01,
                    check_failures_hard=not revision_enabled,
                    reuse=resumed is not None and phase is ResumePhase.REVIEWER_C01,
                    required_check_ids=plan.required_checks or None,
                    enforce_diff_size=False,
                    expected_head_sha=accepted_head_sha,
                    # A run created before ``checks/C01`` was canonical kept
                    # its only final evidence at the run root.
                    reuse_fallback_dir=run_dir,
                )
                self._publish_check_aliases(run_dir, checks_dir_c01)
            except Exception:
                self._write_phase_checkpoint(
                    run_dir, final_checks_phase, cycle=1, head=accepted_head_sha, tree=checks_tree,
                )
                raise
            store.update(status=RunStatus.VALIDATING, checks=_check_payload(evidence),
                         staged_tree_sha=evidence.staged_tree_sha,
                         changed_files=list(evidence.changed_files),
                         deterministic_gate={"passed": evidence.deterministic_passed,
                                             "required_check_ids": list(evidence.required_check_ids),
                                             "failures": list(evidence.failures)})
            integrity_failures = _hard_integrity_failures(evidence) if revision_enabled else [item for item in evidence.failures if item in _DIRECT_FAILURES or
                                  any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES) or
                                  item.startswith("CHECK_MUTATED:")]
            if integrity_failures:
                self._write_phase_checkpoint(
                    run_dir, final_checks_phase, cycle=1, head=accepted_head_sha, tree=checks_tree,
                )
                return self._v2_failed(store, run_dir, integrity_failures[0].split(":", 1)[0], None,
                                        ", ".join(integrity_failures))
        else:
            evidence = resumed.c01_evidence
            # Phase-dependent, never blanket: only ``FINAL_CHECKS_RETRY_C01``
            # is allowed to carry no current bundle, because its retry checks
            # are exactly the operation that has not succeeded yet.  Every
            # later phase keeps the existing mandatory-evidence validation.
            if evidence is None and not retry_bridge_c01 and not (
                not self._legacy_run_options
                and phase is ResumePhase.FINAL_CHECKS_RETRY_C01
            ):
                raise ResumeIntegrityError("C01 candidate evidence is missing")

        soft_failures_c01 = (
            _soft_check_failures(evidence)
            if revision_enabled and evidence is not None else []
        )
        base_repair_scope_c01 = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        if not self._legacy_run_options and evidence is not None:
            direct_result_c01, evidence = self._run_direct_check_repair_loop(
                store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                repository_reference=repository_reference, info=info,
                branch_ref=branch_ref, ownership_before=ownership_before,
                selection=selection, cycle=1, evidence=evidence,
                base_scope=base_repair_scope_c01,
                required_check_ids=plan.required_checks or None,
                expected_head_sha=accepted_head_sha, resumed=resumed,
            )
            check_repair_result_c01 = direct_result_c01
            self._cycle_update(
                store, 1, status="check_repair_completed",
                automatic_check_repair=self._check_repair_attempt_state(direct_result_c01),
            )
            if direct_result_c01.status == "integrity-failed":
                failures = _hard_integrity_failures(evidence)
                return self._v2_failed(
                    store, run_dir,
                    failures[0].split(":", 1)[0] if failures else "INTEGRITY_FAILED",
                    None, ", ".join(failures or evidence.failures),
                )
            if direct_result_c01.status == "scope-required":
                self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                return self.resume(run_id)
            if direct_result_c01.status == "agent-failed":
                return self._v2_failed(
                    store, run_dir, AGENT_RUNTIME_FAILED, None,
                    "check-repair worker failed",
                )
            if direct_result_c01.status == "exhausted":
                remaining = [
                    item.split(":", 1)[1]
                    for item in _soft_check_failures(evidence) if ":" in item
                ]
                return self._v2_failed(
                    store, run_dir, "CHECK_REPAIR_EXHAUSTED", None,
                    _json_text({
                        "remaining_failed_check_ids": remaining,
                        "attempts": len(direct_result_c01.attempts),
                        "repair_profile_id": (
                            getattr(selection, "check_repair", None)
                            or getattr(selection, "repair_implementer", None)
                        ).profile_id
                        if (
                            getattr(selection, "check_repair", None)
                            or getattr(selection, "repair_implementer", None)
                        ) is not None else None,
                    }),
                )
        if (
            self._legacy_run_options
            and
            not retry_bridge_c01
            and at < phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01)
            and evidence is not None and not evidence.deterministic_passed
            and revision_enabled
        ):
            if not soft_failures_c01:
                return self._v2_failed(
                    store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                    ", ".join(evidence.failures),
                )
            check_repair_phase = ResumePhase.CHECK_REPAIR_C01
            retry_checks_phase = ResumePhase.FINAL_CHECKS_RETRY_C01
            if at <= phase_index(check_repair_phase):
                check_repair_scope_c01 = self._resolve_check_repair_scope(
                    repo=repo, worktree=info.worktree,
                    tree_sha=evidence.staged_tree_sha, run_dir=run_dir,
                    evidence=evidence, base_mutable_scope=base_repair_scope_c01,
                )
            elif resumed is not None:
                check_repair_scope_c01 = _read_check_repair_scope(
                    run_dir / "revision" / "check-repair" / "C01",
                    fallback_base=base_repair_scope_c01,
                    policy_config=self._effective_repair_scope,
                )
            self._cycle_update(
                store, 1, status="check_repair_attempted",
                automatic_check_repair={
                    "attempted": True, "failure_ids": soft_failures_c01,
                    "before": _check_payload(evidence),
                    **(_check_repair_scope_payload(check_repair_scope_c01)
                       if check_repair_scope_c01 is not None else {}),
                },
            )
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True, "failure_ids": soft_failures_c01,
                    **(_check_repair_scope_payload(check_repair_scope_c01)
                       if check_repair_scope_c01 is not None else {}),
                },
            )
            if at <= phase_index(check_repair_phase):
                if resumed is not None and phase is check_repair_phase:
                    # The previous attempt produced the failure tree this
                    # resume already validated and restored; its prompt, events
                    # and logs are archived before new ones are written.
                    _archive_attempt_tree(run_dir / "revision" / "check-repair" / "C01")
                try:
                    check_repair_result_c01, repair_error = self._run_v2_revision_cycle(
                        store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                        base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                        repository_reference=repository_reference, info=info,
                        branch_ref=branch_ref, ownership_before=ownership_before,
                        selection=selection, mutable_scope=list(
                            check_repair_scope_c01.effective_paths
                            if check_repair_scope_c01 is not None else base_repair_scope_c01
                        ), check_repair_evidence=evidence, cycle=1,
                        check_repair_scope=check_repair_scope_c01,
                    )
                except AgentScopeError as exc:
                    self._redact_revision_artifacts(
                        run_dir, revision_dir=run_dir / "revision" / "check-repair" / "C01"
                    )
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None,
                                           redact(str(exc), self._secrets))
                except AgentError as exc:
                    self._redact_revision_artifacts(
                        run_dir, revision_dir=run_dir / "revision" / "check-repair" / "C01"
                    )
                    _record_failure_tree(run_dir / "revision" / "check-repair" / "C01", info.worktree)
                    return self._v2_failed(store, run_dir, "CLAUDE_FAILED", None,
                                           redact(str(exc), self._secrets))
                if repair_error is not None:
                    if repair_error == _SCOPE_REQUEST_ROUTE:
                        self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                        return self.resume(run_id)
                    return self._v2_failed(
                        store, run_dir, repair_error, None, ", ".join(soft_failures_c01)
                    )
            elif resumed is not None:
                check_repair_result_c01 = resumed.c01_check_repair_revision

            if at <= phase_index(retry_checks_phase):
                store.update(status=RunStatus.REVALIDATING, current_step=None)
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    _archive_attempt(run_dir / "checks" / "C01", names=_CHECK_ATTEMPT_ARTIFACTS)
                    evidence = self._final_evidence(
                        info.worktree, base_sha, run_dir / "checks" / "C01",
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=accepted_head_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, run_dir / "checks" / "C01")
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, retry_checks_phase, cycle=1,
                        head=accepted_head_sha, tree=retry_tree,
                    )
                    raise
                store.update(
                    status=RunStatus.REVALIDATING, checks=_check_payload(evidence),
                    staged_tree_sha=evidence.staged_tree_sha,
                    changed_files=list(evidence.changed_files),
                    deterministic_gate={"passed": evidence.deterministic_passed,
                                        "required_check_ids": list(evidence.required_check_ids),
                                        "failures": list(evidence.failures)},
                    check_repair={"attempted": True, "failure_ids": soft_failures_c01,
                                  "after": list(evidence.failures)},
                )
                retry_integrity = _hard_integrity_failures(evidence)
                if retry_integrity:
                    return self._v2_failed(
                        store, run_dir, retry_integrity[0].split(":", 1)[0], None,
                        ", ".join(retry_integrity),
                    )
                if not evidence.deterministic_passed:
                    expanded_phase = ResumePhase.CHECK_REPAIR_EXPANDED_C01
                    expanded_retry_phase = ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01
                    expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
                    expanded_scope = self._second_check_repair_scope(
                        repo=repo,
                        worktree=info.worktree,
                        tree_sha=evidence.staged_tree_sha,
                        run_dir=run_dir,
                        evidence=evidence,
                        normal_scope=self._normal_check_repair_scope(
                            check_repair_scope_c01, base_repair_scope_c01,
                            self._effective_repair_scope,
                        ),
                        expanded_dir=expanded_dir,
                    )
                    if expanded_scope is not None:
                        _archive_attempt(run_dir / "checks" / "C01", names=_CHECK_ATTEMPT_ARTIFACTS)
                        store.update(
                            status=RunStatus.REVISING,
                            check_repair={
                                "attempted": True,
                                "failure_ids": soft_failures_c01,
                                "after": list(evidence.failures),
                                **_second_check_repair_state(expanded_scope),
                            },
                        )
                        self._cycle_update(
                            store, 1, status="expanded_check_repair_attempted",
                            automatic_check_repair={
                                "attempted": True,
                                "after": _check_payload(evidence),
                                **_second_check_repair_state(expanded_scope),
                            },
                        )
                        if at <= phase_index(expanded_phase):
                            expanded_check_repair_result_c01, repair_error = self._run_v2_revision_cycle(
                                store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                                base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                                repository_reference=repository_reference, info=info,
                                branch_ref=branch_ref, ownership_before=ownership_before,
                                selection=selection,
                                mutable_scope=list(expanded_scope.effective_paths),
                                check_repair_evidence=evidence, cycle=1,
                                artifact_dir=expanded_dir,
                                check_repair_scope=expanded_scope,
                                check_repair_phase_override=expanded_phase,
                                check_repair_next_phase_override=expanded_retry_phase,
                            )
                            if repair_error is not None:
                                if repair_error == _SCOPE_REQUEST_ROUTE:
                                    self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                                    return self.resume(run_id)
                                return self._v2_failed(store, run_dir, repair_error, None)
                        elif resumed is not None:
                            expanded_check_repair_result_c01 = resumed.c01_expanded_check_repair_revision
                        if at <= phase_index(expanded_retry_phase):
                            store.update(status=RunStatus.REVALIDATING, current_step=None)
                            retry_tree = candidate_tree_sha(info.worktree)
                            try:
                                evidence = self._final_evidence(
                                    info.worktree, base_sha, run_dir / "checks" / "C01",
                                    check_failures_hard=False, reuse=False,
                                    expected_head_sha=accepted_head_sha,
                                    required_check_ids=plan.required_checks or None,
                                    enforce_diff_size=False,
                                )
                            except Exception:
                                self._write_phase_checkpoint(
                                    run_dir, expanded_retry_phase, cycle=1,
                                    head=accepted_head_sha, tree=retry_tree,
                                )
                                raise
                            store.update(
                                status=RunStatus.REVALIDATING,
                                checks=_check_payload(evidence),
                                staged_tree_sha=evidence.staged_tree_sha,
                                changed_files=list(evidence.changed_files),
                                deterministic_gate={
                                    "passed": evidence.deterministic_passed,
                                    "required_check_ids": list(evidence.required_check_ids),
                                    "failures": list(evidence.failures),
                                },
                                check_repair={
                                    "attempted": True,
                                    "failure_ids": soft_failures_c01,
                                    "after": list(evidence.failures),
                                    **_second_check_repair_state(expanded_scope),
                                },
                            )
                            expanded_integrity = _hard_integrity_failures(evidence)
                            if expanded_integrity:
                                return self._v2_failed(
                                    store, run_dir, expanded_integrity[0].split(":", 1)[0], None,
                                    ", ".join(expanded_integrity),
                                )
                        if not evidence.deterministic_passed:
                            return self._v2_failed(
                                store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                                ", ".join(evidence.failures),
                            )
                    else:
                        return self._v2_failed(
                            store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                            ", ".join(evidence.failures),
                        )
            if check_repair_result_c01 is not None:
                revision_report_c01 = _json_text({
                    "initial_revision": revision_report_c01,
                    "automatic_check_repair": _revision_report_text(
                        check_repair_result_c01,
                        run_dir / "revision" / "check-repair" / "C01",
                    ),
                    **({"expanded_check_repair": _revision_report_text(
                        expanded_check_repair_result_c01,
                        run_dir / "revision" / "check-repair-expanded" / "C01",
                    )} if expanded_check_repair_result_c01 is not None else {}),
                })

        if retry_bridge_c01:
            # Resume exactly at the retry checks of the normal C01 repair.
            # Luna, the initial Claude revision and that repair are durable
            # and are never replayed here; only the retry checks, and then at
            # most the single expanded repair, may still run.
            check_repair_scope_c01 = _read_check_repair_scope(
                run_dir / "revision" / "check-repair" / "C01",
                fallback_base=base_repair_scope_c01,
                policy_config=self._effective_repair_scope,
            )
            check_repair_result_c01 = resumed.c01_check_repair_revision
            expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
            checks_dir_c01 = run_dir / "checks" / "C01"
            # The only bundle that can answer for the checkpointed tree is a
            # current one; the archived first pass answers for the tree the
            # normal repair was given, so it is never read as a retry result.
            retry_evidence = (
                _load_evidence(checks_dir_c01) if checks_dir_c01.is_dir()
                else _load_evidence(run_dir)
            )
            if retry_evidence is None and evidence is not None and (
                evidence.staged_tree_sha == start.expected_tree_sha
            ):
                retry_evidence = evidence
            if retry_evidence is not None and (
                retry_evidence.staged_tree_sha != start.expected_tree_sha
            ):
                raise ResumeIntegrityError(
                    "the C01 retry evidence is not for the checkpoint tree"
                )
            if retry_evidence is None:
                # The repair succeeded and the process died before the retry
                # checks wrote their bundle: run exactly those checks once, on
                # exactly the checkpointed tree.
                if candidate_tree_sha(info.worktree) != start.expected_tree_sha:
                    raise ResumeIntegrityError(
                        "the worktree differs from the C01 retry checkpoint tree"
                    )
                store.update(status=RunStatus.REVALIDATING, current_step=None)
                try:
                    retry_evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir_c01,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=accepted_head_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, checks_dir_c01)
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_C01, cycle=1,
                        head=accepted_head_sha, tree=start.expected_tree_sha,
                    )
                    raise
            evidence = retry_evidence
            soft_failures_c01 = _soft_check_failures(evidence)
            store.update(
                status=RunStatus.REVALIDATING, checks=_check_payload(evidence),
                staged_tree_sha=evidence.staged_tree_sha,
                changed_files=list(evidence.changed_files),
                deterministic_gate={"passed": evidence.deterministic_passed,
                                    "required_check_ids": list(evidence.required_check_ids),
                                    "failures": list(evidence.failures)},
                check_repair={"attempted": True, "failure_ids": soft_failures_c01,
                              "after": list(evidence.failures)},
            )
            retry_integrity = _hard_integrity_failures(evidence)
            if retry_integrity:
                return self._v2_failed(
                    store, run_dir, retry_integrity[0].split(":", 1)[0], None,
                    ", ".join(retry_integrity),
                )
            if not evidence.deterministic_passed:
                expanded_scope, archive_required = (
                    self._durable_second_check_repair_scope(
                        repo=repo, worktree=info.worktree, run_dir=run_dir,
                        evidence=evidence, normal_scope=check_repair_scope_c01,
                        expanded_dir=expanded_dir,
                    )
                )
                if expanded_scope is None:
                    return self._v2_failed(
                        store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                        ", ".join(evidence.failures),
                    )
                if archive_required:
                    _archive_attempt(checks_dir_c01, names=_CHECK_ATTEMPT_ARTIFACTS)
                store.update(
                    status=RunStatus.REVISING,
                    check_repair={
                        "attempted": True,
                        "failure_ids": soft_failures_c01,
                        "after": list(evidence.failures),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                self._cycle_update(
                    store, 1, status="expanded_check_repair_attempted",
                    automatic_check_repair={
                        "attempted": True,
                        "after": _check_payload(evidence),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                expanded_check_repair_result_c01, repair_error = self._run_v2_revision_cycle(
                    store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                    base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                    repository_reference=repository_reference, info=info,
                    branch_ref=branch_ref, ownership_before=ownership_before,
                    selection=selection,
                    mutable_scope=list(expanded_scope.effective_paths),
                    check_repair_evidence=evidence, cycle=1,
                    artifact_dir=expanded_dir,
                    check_repair_scope=expanded_scope,
                    check_repair_phase_override=ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                    check_repair_next_phase_override=(
                        ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01
                    ),
                )
                if repair_error is not None:
                    if repair_error == _SCOPE_REQUEST_ROUTE:
                        self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                        return self.resume(run_id)
                    return self._v2_failed(store, run_dir, repair_error, None)
                store.update(status=RunStatus.REVALIDATING, current_step=None)
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir_c01,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=accepted_head_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, checks_dir_c01)
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
                        cycle=1, head=accepted_head_sha, tree=retry_tree,
                    )
                    raise
                store.update(
                    status=RunStatus.REVALIDATING,
                    checks=_check_payload(evidence),
                    staged_tree_sha=evidence.staged_tree_sha,
                    changed_files=list(evidence.changed_files),
                    deterministic_gate={
                        "passed": evidence.deterministic_passed,
                        "required_check_ids": list(evidence.required_check_ids),
                        "failures": list(evidence.failures),
                    },
                    check_repair={
                        "attempted": True,
                        "failure_ids": soft_failures_c01,
                        "after": list(evidence.failures),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                expanded_integrity = _hard_integrity_failures(evidence)
                if expanded_integrity:
                    return self._v2_failed(
                        store, run_dir, expanded_integrity[0].split(":", 1)[0], None,
                        ", ".join(expanded_integrity),
                    )
                # One expanded pass and no more: a still-red gate is final.
                if not evidence.deterministic_passed:
                    return self._v2_failed(
                        store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                        ", ".join(evidence.failures),
                    )
            revision_report_c01 = _json_text({
                "initial_revision": revision_report_c01,
                "automatic_check_repair": (
                    _revision_report_text(
                        check_repair_result_c01,
                        run_dir / "revision" / "check-repair" / "C01",
                    ) if check_repair_result_c01 is not None else ""
                ),
                **({"expanded_check_repair": _revision_report_text(
                    expanded_check_repair_result_c01, expanded_dir,
                )} if expanded_check_repair_result_c01 is not None else {}),
            })

        if self._legacy_run_options and at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01) and at <= phase_index(
            ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01
        ):
            expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
            expanded_scope = _validate_expanded_check_repair_scope(
                expanded_dir,
                repo=repo,
                tree_sha=candidate_tree_sha(info.worktree),
                normal_scope=self._durable_normal_check_repair_scope(
                    run_dir, cycle=1, base_paths=base_repair_scope_c01,
                    scope=check_repair_scope_c01,
                ),
                policy_config=self._effective_repair_scope,
            )
            if at <= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01):
                expanded_check_repair_result_c01, repair_error = self._run_v2_revision_cycle(
                    store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                    base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                    repository_reference=repository_reference, info=info,
                    branch_ref=branch_ref, ownership_before=ownership_before,
                    selection=selection,
                    mutable_scope=list(expanded_scope.effective_paths),
                    check_repair_evidence=evidence, cycle=1,
                    artifact_dir=expanded_dir, check_repair_scope=expanded_scope,
                    check_repair_phase_override=ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                    check_repair_next_phase_override=ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
                )
                if repair_error is not None:
                    if repair_error == _SCOPE_REQUEST_ROUTE:
                        self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                        return self.resume(run_id)
                    return self._v2_failed(store, run_dir, repair_error, None)
            elif resumed is not None:
                expanded_check_repair_result_c01 = resumed.c01_expanded_check_repair_revision
            if at <= phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01):
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    evidence = self._final_evidence(
                        info.worktree, base_sha, run_dir / "checks" / "C01",
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=accepted_head_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, run_dir / "checks" / "C01")
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
                        cycle=1, head=accepted_head_sha, tree=retry_tree,
                    )
                    raise
                store.update(
                    status=RunStatus.REVALIDATING,
                    checks=_check_payload(evidence),
                    staged_tree_sha=evidence.staged_tree_sha,
                    changed_files=list(evidence.changed_files),
                    deterministic_gate={
                        "passed": evidence.deterministic_passed,
                        "required_check_ids": list(evidence.required_check_ids),
                        "failures": list(evidence.failures),
                    },
                    check_repair={
                        "attempted": True,
                        "after": list(evidence.failures),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                expanded_integrity = _hard_integrity_failures(evidence)
                if expanded_integrity:
                    return self._v2_failed(
                        store, run_dir, expanded_integrity[0].split(":", 1)[0], None,
                        ", ".join(expanded_integrity),
                    )
                if not evidence.deterministic_passed:
                    return self._v2_failed(
                        store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                        ", ".join(evidence.failures),
                    )
            revision_report_c01 = _json_text({
                "initial_revision": revision_report_c01,
                "automatic_check_repair": (
                    _revision_report_text(
                        check_repair_result_c01,
                        run_dir / "revision" / "check-repair" / "C01",
                    ) if check_repair_result_c01 is not None else ""
                ),
                "expanded_check_repair": (
                    _revision_report_text(
                        expanded_check_repair_result_c01, expanded_dir,
                    ) if expanded_check_repair_result_c01 is not None else ""
                ),
            })

        if at <= phase_index(ResumePhase.CANDIDATE_COMMIT_C01):
            # The deterministic gate is the only gate before the immutable
            # candidate commit.  Semantic review deliberately comes later.
            if not evidence.deterministic_passed:
                return self._v2_failed(
                    store, run_dir, "DETERMINISTIC_GATE_FAILED", None,
                    ", ".join(evidence.failures),
                )
            candidate_parent = current_head(info.worktree)
            self._checkpoint(run_dir, ResumePhase.CANDIDATE_COMMIT_C01,
                             head=candidate_parent, tree=evidence.staged_tree_sha,
                             expected_parent_sha=(
                                 commit_parents(info.worktree, candidate_parent)[0]
                                 if candidate_parent != base_sha else None
                             ))
            self._authorize_candidate_tree(
                evidence, info.worktree, candidate_parent, branch_ref
            )
            c01_candidate = self._ensure_candidate_commit(
                run_dir=run_dir, info=info, cycle=1, tree_sha=evidence.staged_tree_sha,
                parent_sha=candidate_parent, title=plan.title,
                repository_reference=repository_reference, store=store, run_id=run_id,
                commit_kind=(
                    "repair" if (
                        (check_repair_result_c01 is not None and getattr(check_repair_result_c01, "attempts", ()))
                        or expanded_check_repair_result_c01 is not None
                    )
                    else "revision" if revision_result is not None
                    else "candidate"
                ),
            )
        else:
            c01_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 1))
            if not isinstance(c01_candidate, dict):
                raise ResumeIntegrityError("C01 candidate commit artifact is missing")

        if at <= phase_index(ResumePhase.CANDIDATE_PUSH_C01):
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_PUSH_C01,
                head=c01_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                expected_parent_sha=c01_candidate["parent_sha"],
            )
            try:
                c01_candidate = self._push_candidate(
                    run_dir=run_dir, info=info, cycle=1, candidate=c01_candidate, store=store,
                )
            except (GitError, OSError, ValueError) as exc:
                return self._v2_failed(store, run_dir, "PUSH_FAILED", None, "candidate push did not complete")

        if at <= phase_index(ResumePhase.REVIEWER_C01):
            # Candidate push complete: the reviewer receives the exact pushed
            # commit, never a moving default branch or a local staged tree.
            self._checkpoint(
                run_dir, ResumePhase.REVIEWER_C01,
                head=c01_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                expected_parent_sha=c01_candidate["parent_sha"],
            )
            reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
            store.update(status=RunStatus.REVIEWING)
            try:
                if resumed is not None and phase is ResumePhase.REVIEWER_C01:
                    _archive_attempt(run_dir, names=_REVIEW_ATTEMPT_ARTIFACTS)
                review = self._run_v2_reviewer(
                    reviewer=reviewer, spec=spec, context=context,
                    repository_reference=repository_reference, evidence=evidence,
                    input=ReviewCycleInput(
                        iteration=1,
                        plan_text=_compact_approved_plan_text(plan),
                        luna_reports=_review_step_reports_text(self._last_v2_step_results),
                        deferred_mismatches=deferred_mismatches,
                        revision_report=revision_report_c01,
                        cycle_history="C01 is the initial implementation cycle.",
                    ),
                    artifacts_dir=run_dir, worktree=info.worktree, base_sha=base_sha,
                    candidate_commit=c01_candidate,
                    reuse_accepted=resumed is not None and phase is ResumePhase.REVIEWER_C01,
                )
            except ReviewParseError as exc:
                # Includes PASS on a red gate: fail closed, never commit, and
                # never turn an invalid PASS into a C02 authorization.
                if revision_enabled:
                    self._snapshot_cycle_artifacts(run_dir)
                return self._v2_failed(store, run_dir, "REVIEWER_OUTPUT_INVALID", None,
                                       _bounded_parse_detail(exc))
            except LLMError as exc:
                # Transport only: no reviewer answer was accepted, so the same
                # exact candidate can be reviewed again on resume.
                return self._v2_failed(store, run_dir, "REVIEWER_TRANSPORT_FAILURE", None,
                                       _bounded_parse_detail(exc))
            store.update(
                status=RunStatus.REVIEWING,
                review=_review_payload(review),
                review_iterations=1,
            )
            self._cycle_update(
                store, 1, status="reviewed",
                luna_steps_summary=self._last_v2_step_results,
                claude_revision_report=revision_result.final_message if revision_result else "",
                checks=_check_payload(evidence), reviewer_conclusion=_review_payload(review),
            )
            self._update_v2_usage(store, run_dir)
            if revision_enabled:
                self._snapshot_cycle_artifacts(run_dir)
        else:
            review = resumed.c01_review
            if review is None:
                raise ResumeIntegrityError("C01 reviewer result is missing")
        if review.verdict is ReviewVerdict.REVISE:
            repair_routes = {ReviewRoute.IMPLEMENTATION}
            if not self._legacy_run_options:
                repair_routes.add(ReviewRoute.REPLAN)
            if repair_enabled and review.route in repair_routes:
                if at <= phase_index(ResumePhase.REVIEWER_C01):
                    store.update(status=RunStatus.IMPLEMENTING, cycle=2)
                    self._cycle_update(
                        store, 2, status="starting",
                        trigger="Reviewer requested one bounded implementation correction loop.",
                        audit_route=review.route.value,
                        audit_request=("plan/scope repair" if review.route is ReviewRoute.REPLAN
                                       else "implementation repair"),
                    )
                    # Reviewer #1 complete: the next operation is the repair
                    # planner, from the committed, pushed and reviewed C01
                    # candidate (never BASE).
                    self._checkpoint(run_dir, ResumePhase.REPAIR_PLANNER, cycle=2,
                                     head=c01_candidate["commit_sha"],
                                     tree=evidence.staged_tree_sha)
                else:
                    store.update(status=store.load().get("status", RunStatus.PLANNING), cycle=2)
                try:
                    repair_plan, _repair_revision, evidence, review = self._execute_v2_repair_cycle(
                        store=store, run_dir=run_dir, run_id=run_id, spec=spec,
                        repo=repo, base_sha=base_sha, context=context,
                        repository_reference=repository_reference, info=info,
                        branch_ref=branch_ref, ownership_before=ownership_before,
                        selection=selection, original_plan=plan,
                        original_bundle=bundle, cycle_1_evidence=evidence,
                        cycle_1_review=review, cycle_1_revision=revision_result,
                        cycle_1_revision_report=revision_report_c01,
                        claude_revision_enabled=revision_enabled,
                        resumed=resumed if at > phase_index(ResumePhase.REVIEWER_C01) else None,
                    )
                except StepExecutionFailure as failure:
                    return self._step_failed(
                        store, run_dir, failure,
                        run_dir / "repair" / "C02" / "steps" / failure.step_id,
                    )
                except ReviewParseError as exc:
                    return self._v2_failed(store, run_dir, "REVIEWER_OUTPUT_INVALID", None,
                                           _bounded_parse_detail(exc))
                except AgentScopeError as exc:
                    self._redact_revision_artifacts(run_dir, revision_dir=run_dir / "revision" / "C02")
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None, redact(str(exc), self._secrets))
                except AgentError as exc:
                    self._redact_revision_artifacts(run_dir, revision_dir=run_dir / "revision" / "C02")
                    _record_failure_tree(run_dir / "revision" / "C02", info.worktree)
                    return self._v2_failed(store, run_dir, "CLAUDE_FAILED", None, redact(str(exc), self._secrets))
                except ReviewerTransportError as exc:
                    return self._v2_failed(store, run_dir, "REVIEWER_TRANSPORT_FAILURE", None,
                                           _bounded_parse_detail(exc))
                except ScopeApprovalRequired:
                    return RunResult(run_dir, RunStatus.WAITING_SCOPE_APPROVAL, store.load())
                except LLMError as exc:
                    return self._v2_failed(store, run_dir, "LLM_FAILURE", None,
                                           _bounded_parse_detail(exc))
                except OrchestrationError as exc:
                    reason = (
                        _failure_reason(exc) if str(exc).startswith("CHECK_PREFLIGHT_FAILED:")
                        else str(exc).split(":", 1)[0].strip()
                    ) or "REPAIR_FAILED"
                    if reason == _SCOPE_REQUEST_ROUTE:
                        self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                        return self.resume(run_id)
                    return self._v2_failed(store, run_dir, reason, None)
                if review.verdict is ReviewVerdict.REVISE and review.route in {
                    ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN,
                }:
                    # Legacy direct callers retain the P40 implementation
                    # reason.  Durable runs uniformly expose a spent repair
                    # budget as an operator-required terminal outcome.
                    reason = (
                        "REVIEW_LOOP_EXHAUSTED"
                        if self._legacy_run_options and review.route is ReviewRoute.IMPLEMENTATION
                        else "REPAIR_EXHAUSTED"
                    )
                    return self._v2_failed(store, run_dir, reason, None,
                                           "HUMAN_REQUIRED")
                if review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
                    return self._v2_failed(store, run_dir, "REVIEW_FAILED", None)
                if not evidence.deterministic_passed:
                    return self._v2_failed(store, run_dir, "DETERMINISTIC_GATE_FAILED", None)
                approved_tree = evidence.staged_tree_sha
                self._cycle_update(store, 2, status="approved")
                store.update(
                    status=RunStatus.APPROVED,
                    approved_tree_sha=approved_tree,
                    current_step=None,
                )
                candidate = _read_json_artifact(_candidate_commit_path(run_dir, 2))
                if not isinstance(candidate, dict) or not _is_object_id(candidate.get("commit_sha")):
                    raise ResumeIntegrityError("C02 candidate commit artifact is missing after PASS")
                return self._complete_candidate_publication(
                    store=store,
                    run_dir=run_dir,
                    info=info,
                    approved_tree=approved_tree,
                    commit_sha=candidate["commit_sha"],
                    repository_reference=repository_reference,
                    cycle=2,
                )
            if not repair_enabled:
                write_repair_task(run_dir, fields={"route": review.route.value,
                    "review_summary": review.summary, "findings": review.findings,
                    "required_fixes": review.required_fixes, "missing_tests": review.missing_tests,
                    "existing_branch": info.branch, "existing_worktree": str(info.worktree), "run_id": run_id})
                # Preserve the historical terminal reason for direct CLI or
                # embedding callers that did not provide a durable snapshot.
                reason = "REVIEW_REVISE" if self._legacy_run_options else "HUMAN_REQUIRED"
                return self._v2_failed(store, run_dir, reason, None)
            if revision_enabled and review.route is ReviewRoute.REPLAN:
                return self._v2_failed(store, run_dir, "REPLAN_REQUIRED", None)
            if revision_enabled and review.route is ReviewRoute.HUMAN:
                return self._v2_failed(store, run_dir, "HUMAN_REQUIRED", None)
            write_repair_task(run_dir, fields={"route": review.route.value,
                "review_summary": review.summary, "findings": review.findings,
                "required_fixes": review.required_fixes, "missing_tests": review.missing_tests,
                "existing_branch": info.branch, "existing_worktree": str(info.worktree), "run_id": run_id})
            return self._v2_failed(store, run_dir, "REVIEW_REVISE", None)
        if review.verdict is ReviewVerdict.FAIL:
            if revision_enabled:
                return self._v2_failed(store, run_dir, "REVIEW_FAILED", None)
            return self._v2_failed(store, run_dir, "REVIEW_FAIL", None)
        if review.route is not ReviewRoute.NONE or not evidence.deterministic_passed:
            return self._v2_failed(store, run_dir, "REVIEW_ROUTE_NOT_NONE" if review.route is not ReviewRoute.NONE else "DETERMINISTIC_GATE_FAILED", None)
        approved_tree = evidence.staged_tree_sha
        if revision_enabled:
            self._cycle_update(store, 1, status="approved")
        store.update(
            status=RunStatus.APPROVED,
            approved_tree_sha=approved_tree,
            current_step=None,
        )
        candidate = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(candidate, dict) or not _is_object_id(candidate.get("commit_sha")):
            raise ResumeIntegrityError("C01 candidate commit artifact is missing after PASS")
        return self._complete_candidate_publication(
            store=store,
            run_dir=run_dir,
            info=info,
            approved_tree=approved_tree,
            commit_sha=candidate["commit_sha"],
            repository_reference=repository_reference,
        )

    def _codex_step_profile(self, profile_id: str) -> ModelProfile:
        """The approved Codex profile of one step (initial or repair role)."""

        try:
            return profile_for_role(self.config, profile_id, ExecutionRole.IMPLEMENTER)
        except ProfileError:
            return profile_for_role(self.config, profile_id, ExecutionRole.REPAIR)

    def _execute_codex_step(
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
        codex_home: Path,
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
        pending_mismatch_retry: str | None = None,
    ) -> StepExecutionOutcome:
        """One approved step, with at most one bounded mismatch retry.

        A *clean* structural mismatch — the worker changed nothing at all — is
        retried exactly once with the same contract, the same profile, the same
        mutable scope and the same candidate tree, in a new fresh Codex
        process.  The retry only adds a prompt addendum; it never widens
        WRITE/CREATE/DELETE.  There is never a third attempt.

        With *pending_mismatch_retry*, the first attempt already happened in an
        earlier run and this call **is** the bounded retry.
        """

        common = {
            "repo": repo, "worktree": worktree, "base_sha": base_sha,
            "branch_ref": branch_ref, "ownership_before": ownership_before,
            "expected_tree": expected_tree, "step": step, "contract": contract,
            "profile_id": profile_id, "artifact_dir": artifact_dir,
            "codex_home": codex_home, "forbidden_env_names": forbidden_env_names,
            "future_ownership": future_ownership,
        }
        if pending_mismatch_retry:
            return self._run_codex_step_attempt(
                **common, initial_mismatch=pending_mismatch_retry,
                mismatch_retry_count=1,
            )
        outcome = self._run_codex_step_attempt(
            **common, initial_mismatch=None, mismatch_retry_count=0,
        )
        if not isinstance(outcome, DeferredStepExecutionOutcome):
            return outcome
        # The boundary must still be exactly the pre-step boundary before a
        # second worker is allowed to run against it.  Drift here is lost
        # authority, never a deferrable outcome: fail closed so that no second
        # worker and no later step runs.
        drift = self._pre_step_boundary_drift(
            repo, worktree, ownership_before,
            branch_ref=branch_ref, base_sha=base_sha, tree_before=outcome.tree_before,
        )
        if drift:
            # Attempt 1 keeps its own artifacts, but its deferred record must
            # not remain the step's current durable record: the step failed.
            _archive_attempt(artifact_dir)
            raise StepExecutionFailure(
                "STEP_CONTRACT_DRIFT", outcome.step_id,
                _bounded_v2_report(
                    f"the step boundary drifted before the bounded mismatch retry: {drift}"
                ),
                profile_id=outcome.profile_id, tree_before=outcome.tree_before,
                tree_after=_safe_candidate_tree(worktree), usage=outcome.usage,
                mismatch=outcome.mismatch,
            )
        # Attempt 1 keeps its own artifacts, including its diagnostics.
        _archive_attempt(artifact_dir)
        return self._run_codex_step_attempt(
            **common, initial_mismatch=outcome.mismatch, mismatch_retry_count=1,
        )

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

    def _run_codex_step_attempt(
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
        codex_home: Path,
        forbidden_env_names: tuple[str | None, ...],
        future_ownership: Mapping[str, tuple[str, ...]] | None = None,
        initial_mismatch: str | None = None,
        mismatch_retry_count: int = 0,
    ) -> StepExecutionOutcome:
        """The single authoritative execution of one Codex step (C01 and C02).

        Gates run in a fixed order and every failure raises
        :class:`StepExecutionFailure`; the caller owns the run status.  The
        worker's final report is data only and never drives a decision.
        """

        step_id = step.id
        # The retry mode of this invocation.  Every failure it can raise
        # carries it, so a resume of a transient failure reruns exactly this
        # semantic operation (same contract, same addendum, same future
        # ownership) instead of a new normal first attempt.
        retry_mode: dict[str, Any] = (
            {
                "mismatch_retry_count": mismatch_retry_count,
                "initial_mismatch": _bounded_v2_report(initial_mismatch or "") or None,
            }
            if mismatch_retry_count else {}
        )
        # 1-2. The exact tree Codex will receive, and the contract's Git
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
        profile = self._codex_step_profile(profile_id)
        executor = self._executor_for_profile(
            profile.id,
            ExecutionRole.IMPLEMENTER,
            forbidden_env_names=forbidden_env_names,
        )
        # 5. One fresh Codex process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
        # The selected adapter owns the single .run_step( compatibility path.
        retry_addendum = (
            build_mismatch_retry_addendum(
                initial_mismatch=initial_mismatch or "",
                future_ownership=future_ownership,
            )
            if mismatch_retry_count else None
        )
        extra = {"retry_addendum": retry_addendum} if retry_addendum else {}
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            # The retry addendum is part of the final prompt, while the
            # approved contract itself remains byte-identical in the artifact.
            # The approved step contract is already the narrow implementer
            # payload.  Keep the exact byte-shaped wrapper used by the
            # contract hash/resume protocol; diagnostics are recorded as a
            # separate role payload and never broaden the request.
            request_prompt = build_implementer_step_prompt(
                contract, retry_addendum=retry_addendum
            )
            prompt_payload = build_implementer_payload(
                step_title=step.title,
                step_objective=step.objective,
                step_invariants=step.forbidden,
                read_set="\n".join(step.read_set),
                mutable_scope=_json_text({
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                }),
                repository_instructions=step.instructions,
                verify_instructions=step.verify,
                retry_addendum=retry_addendum or "",
                budget_bytes=self.config.prompt_budget.implementer_max_bytes,
            )
            prompt_payload = dataclasses.replace(
                prompt_payload,
                rendered=request_prompt,
                total_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                static_prompt_bytes=max(
                    0,
                    len(request_prompt.encode("utf-8", errors="replace"))
                    - prompt_payload.dynamic_payload_bytes,
                ),
                budget_overrun=(
                    bool(self.config.prompt_budget.implementer_max_bytes)
                    and len(request_prompt.encode("utf-8", errors="replace"))
                    > self.config.prompt_budget.implementer_max_bytes
                ),
            )
            write_prompt_diagnostics(artifact_dir, prompt_payload)
            trace_started_at = self._trace_time()
            trace_started_mono = time.perf_counter()
            trace_selected = self._trace_selected_profile(
                profile.id, ExecutionRole.IMPLEMENTER, step_id=step_id
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
                        role=ExecutionRole.IMPLEMENTER,
                        prompt_bytes=len(request_prompt.encode("utf-8", errors="replace")),
                        started_at=trace_started_at,
                        started_mono=trace_started_mono,
                        tree_before=tree_before,
                    ),
                },
            )
            result = executor.run(
                AgentRunRequest(
                    role=ExecutionRole.IMPLEMENTER,
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
                        role=ExecutionRole.IMPLEMENTER,
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
                    role=ExecutionRole.IMPLEMENTER,
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
        auth_failure = result.backend_reason == "CODEX_AUTH_FAILURE"
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
            raise StepExecutionFailure("AGENT_COMMITTED", step_id, "worktree HEAD changed", **failed)
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
                reason = "CODEX_AUTH_FAILURE" if self._legacy_backend_injection else AGENT_RUNTIME_FAILED
                raise StepExecutionFailure(
                    reason, step_id, "Codex authentication failed", **failed,
                    tree_after=_safe_candidate_tree(worktree),
                )
            reason = "AGENT_FAILED" if self._legacy_backend_injection else (
                result.exit_reason or AGENT_RUNTIME_FAILED
            )
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
        # dependency is recorded as data for Claude and the reviewer; it never
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
        step_dir: Path,
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
            step_dir=step_dir,
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
        diff_path = run_dir / "steps" / step.id / "diff.patch"
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
        step_path = run_dir / "steps" / step.id / "step.json"
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

    def _ensure_candidate_commit(
        self, *, run_dir: Path, info: WorktreeInfo, cycle: int, tree_sha: str,
        parent_sha: str, title: str, repository_reference: RepositoryReference,
        store: RunStateStore, run_id: str, commit_kind: str = "candidate",
    ) -> dict[str, Any]:
        """Reconcile or create exactly one candidate commit for a cycle."""

        state = store.load()
        try:
            assert_deferred_verifications_resolved([
                *(state.get("accepted_steps") or []),
                *(state.get("deferred_verifications") or []),
            ])
        except CommitSafetyError as exc:
            raise CommitBoundaryError(str(exc)) from exc
        path = _candidate_commit_path(run_dir, cycle)
        stored = _read_json_artifact(path)
        commit_sha = stored.get("commit_sha") if isinstance(stored, dict) else None
        effective_parent_sha = parent_sha
        if not _is_object_id(commit_sha):
            commit_sha = None
        if commit_sha is None:
            try:
                head = current_head(info.worktree)
                if head == parent_sha and resolve_tree(info.worktree, head) == tree_sha:
                    # The last accepted step is already the candidate tip;
                    # publication must not manufacture an empty metadata-only
                    # commit.  Bind the artifact to its real parent.
                    commit_sha = head
                    parents = commit_parents(info.worktree, head)
                    if len(parents) != 1:
                        raise CommitBoundaryError("candidate tip has no single parent")
                    effective_parent_sha = parents[0]
                elif commit_parents(info.worktree, head) == (parent_sha,) and resolve_tree(info.worktree, head) == tree_sha:
                    commit_sha = head
            except GitError:
                pass
        if commit_sha is None:
            try:
                commit_safety_gate(
                    info.worktree,
                    tree_sha=tree_sha,
                    parent_sha=parent_sha,
                    mutable_scope=(),
                    verification_status="passed",
                    secrets=self._secrets,
                    max_diff_bytes=None,
                )
            except CommitSafetyError as exc:
                raise CommitBoundaryError(str(exc)) from exc
            body = f"MetaHarness-Run: {run_id}"
            if commit_kind == "revision":
                commit_sha = commit_revision_tree(
                    info.worktree, tree_sha=tree_sha, parent_sha=parent_sha, body=body,
                )
            elif commit_kind == "repair":
                commit_sha = commit_repair_tree(
                    info.worktree, tree_sha=tree_sha, parent_sha=parent_sha,
                    cycle=cycle, body=body,
                )
            else:
                commit_sha = commit_candidate_tree(
                    info.worktree, tree_sha=tree_sha, parent_sha=parent_sha,
                    subject=_commit_subject(title), body=body,
                )
        if current_head(info.worktree) != commit_sha:
            raise CommitBoundaryError("candidate commit is not the run branch tip")
        if commit_parents(info.worktree, commit_sha) != (effective_parent_sha,) or resolve_tree(info.worktree, commit_sha) != tree_sha:
            raise CommitBoundaryError("candidate commit identity is not exact")
        payload = _candidate_commit_payload(
            commit_sha=commit_sha, tree_sha=tree_sha, parent_sha=effective_parent_sha,
            branch=info.branch, remote=self.config.publish.remote,
            immutable_url=_commit_web_url(repository_reference, commit_sha),
            pushed_at=(stored.get("pushed_at") if isinstance(stored, dict) else None),
        )
        atomic_write_text(path, _json_text(payload))
        chain = list(accepted_chain_records(run_dir))
        if not any(
            isinstance(item, dict) and item.get("commit_sha") == commit_sha
            for item in chain
        ):
            chain.append({
                "commit_sha": commit_sha,
                "tree_sha": tree_sha,
                "parent_sha": effective_parent_sha,
            })
            atomic_write_text(run_dir / "accepted-chain.json", _json_text({"commits": chain}))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"C{cycle:02d}"] = payload
        store.update(
            status=RunStatus.APPROVED, candidate=candidate_state,
            candidate_commit_sha=commit_sha, approved_tree_sha=tree_sha,
            expected_head_sha=commit_sha, expected_parent_sha=effective_parent_sha,
            expected_tree_sha=tree_sha, next_step_id=None,
        )
        if commit_kind in {"repair", "revision"}:
            event_name = (
                "check_repair.committed" if commit_kind == "repair"
                else "revision.committed"
            )
            diff_path = run_dir / "checks" / f"C{cycle:02d}" / "diff.patch"
            evidence_record = _read_json_artifact(
                run_dir / "checks" / f"C{cycle:02d}" / "evidence.json"
            )
            self._trace_emit(
                event_name,
                phase="repair" if commit_kind == "repair" else "revision",
                cycle=cycle,
                data={
                    "parent_sha": effective_parent_sha,
                    "commit_sha": commit_sha,
                    "tree_sha": tree_sha,
                    "changed_paths": list(
                        evidence_record.get("changed_files", [])
                        if isinstance(evidence_record, dict)
                        else []
                    ),
                    **self._trace_diff_reference(diff_path),
                },
            )
        return payload

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
            candidate_state[f"C{cycle:02d}"] = candidate
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
        context: str,
        repository_reference: RepositoryReference,
        evidence: EvidenceBundle,
        input: ReviewCycleInput,
        artifacts_dir: Path,
        worktree: Path,
        base_sha: str,
        candidate_commit: Mapping[str, Any],
        reuse_accepted: bool = False,
    ) -> ReviewResult:
        """The single reviewer evidence assembly for C01 and C02.

        The gate payload, the parse argument and the later commit gate all
        use the same actual ``evidence.deterministic_passed``.  On a resume,
        a reviewer answer already accepted for this exact candidate tree is
        re-parsed instead of asking again; the call is always a fresh one.
        """

        candidate_sha = candidate_commit.get("commit_sha")
        if not _is_object_id(candidate_sha):
            raise OrchestrationError(
                "candidate commit SHA is missing before reviewer"
            )

        if reuse_accepted:
            accepted = _accepted_review(artifacts_dir, evidence, candidate_sha)
            if accepted is not None:
                return accepted
        gate = _json_text({
            "deterministic_passed": evidence.deterministic_passed,
            "required_check_ids": list(evidence.required_check_ids),
            "failures": list(evidence.failures),
            "staged_tree_sha": evidence.staged_tree_sha,
        })
        repository_state = _json_text({
            "BASE_SHA": base_sha,
            "HEAD_SHA": current_head(worktree),
            "CANDIDATE_TREE_SHA": evidence.staged_tree_sha,
            "CANDIDATE_COMMIT_SHA": candidate_commit.get("commit_sha"),
            "CHANGED_FILES": list(evidence.changed_files),
            "GIT_STATUS": status_porcelain(worktree),
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
                + input.luna_reports
                + "\n"
                + input.revision_report
                + "\nDEFERRED CONTRACT MISMATCHES\n"
                + input.deferred_mismatches
            ),
            repository_reference=_json_text(repository_reference_dict(repository_reference)),
            budget_bytes=self.config.prompt_budget.final_review_max_bytes,
        )
        reviewer_profile_id = getattr(
            getattr(self, "_last_selection", None), "final_reviewer", None
        ) or getattr(getattr(self, "_last_selection", None), "reviewer", None)
        reviewer_profile_id = getattr(reviewer_profile_id, "profile_id", None)
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
        run_root = artifacts_dir.parent.parent if artifacts_dir.parent.name == "review" else artifacts_dir
        planner_thread = _read_planner_conversation(run_root)
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
        reuse_fallback_dir: Path | None = None,
    ) -> EvidenceBundle:
        """Final checks for the exact current candidate.

        On a reviewer resume, durable evidence already frozen for exactly
        this index tree is reused: checks are never replayed for a tree whose
        evidence is complete.  *reuse_fallback_dir* lets a run created before
        ``checks/C01`` became canonical reuse its root evidence.
        """

        if reuse:
            stored = _load_evidence(evidence_dir)
            if stored is None and reuse_fallback_dir is not None:
                stored = _load_evidence(reuse_fallback_dir)
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
                "passed": evidence.deterministic_passed,
                "failures": list(evidence.failures),
                "required_check_ids": list(evidence.required_check_ids),
                "tree_sha": evidence.staged_tree_sha,
                "changed_paths": list(evidence.changed_files),
                "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                "started_at": checks_started_at,
            },
        )
        if getattr(self, "_trace_check_repair_active", False):
            self._trace_emit(
                "check_repair.checks.completed",
                phase="repair",
                cycle=getattr(self, "_trace_cycle", 1),
                data={
                    "passed": evidence.deterministic_passed,
                    "failures": list(evidence.failures),
                    "tree_sha": evidence.staged_tree_sha,
                },
            )
            self._trace_check_repair_active = False
        return evidence

    def _check_repair_coordinator(self) -> CheckRepairCoordinator:
        """Build the check-repair coordinator with the live scope policy."""

        return CheckRepairCoordinator(
            effective_repair_scope=self._effective_repair_scope,
        )

    def _resolve_check_repair_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_sha: str,
        run_dir: Path,
        evidence: EvidenceBundle,
        base_mutable_scope: Sequence[str],
    ) -> CheckRepairScope:
        """See ``CheckRepairCoordinator.resolve_scope``."""

        return self._check_repair_coordinator().resolve_scope(
            repo=repo, worktree=worktree, tree_sha=tree_sha, run_dir=run_dir,
            evidence=evidence, base_mutable_scope=base_mutable_scope,
        )

    def _second_check_repair_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_sha: str,
        run_dir: Path,
        evidence: EvidenceBundle,
        normal_scope: CheckRepairScope,
        expanded_dir: Path,
    ) -> CheckRepairScope | None:
        """See ``CheckRepairCoordinator.second_scope``."""

        return self._check_repair_coordinator().second_scope(
            repo=repo, worktree=worktree, tree_sha=tree_sha, run_dir=run_dir,
            evidence=evidence, normal_scope=normal_scope, expanded_dir=expanded_dir,
        )

    def _durable_normal_check_repair_scope(
        self,
        run_dir: Path,
        *,
        cycle: int,
        base_paths: Sequence[str],
        scope: CheckRepairScope | None = None,
    ) -> CheckRepairScope:
        """See ``CheckRepairCoordinator.durable_normal_scope``."""

        return self._check_repair_coordinator().durable_normal_scope(
            run_dir, cycle=cycle, base_paths=base_paths, scope=scope,
        )

    _normal_check_repair_scope = staticmethod(CheckRepairCoordinator.normal_scope)

    def _durable_second_check_repair_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        run_dir: Path,
        evidence: EvidenceBundle,
        normal_scope: CheckRepairScope,
        expanded_dir: Path,
    ) -> tuple[CheckRepairScope | None, bool]:
        """See ``CheckRepairCoordinator.durable_second_scope``."""

        return self._check_repair_coordinator().durable_second_scope(
            repo=repo, worktree=worktree, run_dir=run_dir, evidence=evidence,
            normal_scope=normal_scope, expanded_dir=expanded_dir,
        )

    def _revision_step_results(self, cycle: int) -> list[dict[str, Any]]:
        """The step history the revision prompt reports, per cycle."""

        return (
            self._repair_v2_step_results if cycle == 2
            else self._last_v2_step_results
        )

    def _revision_runner(self, *, legacy_failure_names: bool | None = None) -> RevisionRunner:
        """Build the revision runner with this run's live dependencies."""

        return RevisionRunner(
            config=self.config,
            secrets=self._secrets,
            effective_repair_scope=self._effective_repair_scope,
            checkpoint=self._checkpoint,
            write_phase_checkpoint=self._write_phase_checkpoint,
            approved_check_authority_sha256=self._approved_check_authority_sha256,
            run_revision=self._run_revision,
            ensure_revision_artifacts=self._ensure_revision_artifacts,
            redact_revision_artifacts=self._redact_revision_artifacts,
            step_results_for_cycle=self._revision_step_results,
            reusable_pre_checks=_reusable_pre_checks,
            hard_integrity_failures=_hard_integrity_failures,
            soft_check_failures=_soft_check_failures,
            check_repair_scope_candidates=_check_repair_scope_candidates,
            check_repair_prompt=_check_repair_prompt,
            legacy_failure_names=(
                self._legacy_backend_injection
                if legacy_failure_names is None else legacy_failure_names
            ),
        )

    def _run_v2_revision_cycle(self, **cycle: Any) -> tuple[Any | None, str | None]:
        """Run one Claude revision cycle -- see ``RevisionRunner.run``."""
        cycle_number = cycle.get("cycle", getattr(self, "_trace_cycle", 1))
        is_check_repair = cycle.get("check_repair_evidence") is not None
        selection = cycle.get("selection")
        selected = (
            getattr(selection, "check_repair", None)
            if is_check_repair else getattr(selection, "semantic_reviser", None)
        )
        if selected is None and selection is not None and not hasattr(selection, "semantic_reviser"):
            selected = getattr(selection, "reviser", None)
        profile = None
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        if selected is not None:
            try:
                profile = profile_for_role(self.config, selected.profile_id, role)
            except ProfileError:
                profile = None
        info = cycle.get("info")
        worktree = getattr(info, "worktree", None)
        tree_before = _safe_candidate_tree(worktree) if worktree is not None else None
        started_at = self._trace_time()
        started_mono = time.perf_counter()
        event_name = "check_repair.started" if is_check_repair else "revision.started"
        self._trace_emit(
            event_name,
            phase="repair" if is_check_repair else "revision",
            cycle=cycle_number,
            data={
                "tree_before": tree_before,
                "attempt": cycle.get("check_repair_attempt"),
                "session": self._trace_session(
                    profile=profile,
                    selected=selected,
                    role=role,
                    prompt_bytes=None,
                    started_at=started_at,
                    started_mono=started_mono,
                    tree_before=tree_before,
                ),
            },
        )
        if is_check_repair:
            self._trace_check_repair_active = True
        legacy_failure_names = cycle.pop("_legacy_failure_names", None)
        try:
            result, error = self._revision_runner(legacy_failure_names=legacy_failure_names).run(**cycle)
        except Exception as exc:
            self._trace_emit(
                "check_repair.agent.completed" if is_check_repair else "revision.agent.completed",
                phase="repair" if is_check_repair else "revision",
                cycle=cycle_number,
                data={
                    "status": "failed",
                    "error": type(exc).__name__,
                    "session": self._trace_session(
                        profile=profile,
                        selected=selected,
                        role=role,
                        prompt_bytes=None,
                        started_at=started_at,
                        started_mono=started_mono,
                        tree_before=tree_before,
                        exit_reason=type(exc).__name__,
                    ),
                },
            )
            raise
        artifact_dir = cycle.get("artifact_dir")
        if artifact_dir is not None:
            artifact_path = Path(artifact_dir)
        elif is_check_repair:
            artifact_path = Path(cycle["run_dir"]) / "revision" / "check-repair" / f"C0{cycle_number}"
        else:
            artifact_path = Path(cycle["run_dir"]) / "revision"
        prompt_path = artifact_path / "agent.prompt.txt"
        prompt_bytes = prompt_path.stat().st_size if prompt_path.is_file() else None
        self._trace_emit(
            "check_repair.agent.completed" if is_check_repair else "revision.agent.completed",
            phase="repair" if is_check_repair else "revision",
            cycle=cycle_number,
            data={
                "status": "completed" if error is None else "failed",
                "error": error,
                "session": self._trace_session(
                    profile=profile,
                    selected=selected,
                    role=role,
                    prompt_bytes=prompt_bytes,
                    started_at=started_at,
                    started_mono=started_mono,
                    tree_before=tree_before,
                    result=result,
                    exit_reason=error,
                ),
            },
        )
        pre_checks = _read_json_artifact(artifact_path / "pre_checks.json")
        if not isinstance(pre_checks, dict):
            pre_checks = {}
        if not is_check_repair or pre_checks:
            check_event = "check_repair.checks.completed" if is_check_repair else "revision.checks.completed"
            self._trace_emit(
                check_event,
                phase="repair" if is_check_repair else "revision",
                cycle=cycle_number,
                data={
                    "passed": bool(pre_checks.get("deterministic_passed", False)),
                    "failures": list(pre_checks.get("failures", [])) if isinstance(pre_checks.get("failures", []), list) else [],
                    "tree_sha": pre_checks.get("staged_tree_sha"),
                },
            )
            if is_check_repair:
                self._trace_check_repair_active = False
        return result, error

    @staticmethod
    def _check_repair_attempt_root(run_dir: Path, cycle: int, number: int) -> Path:
        return (
            run_dir / "revision" / "check-repair" / f"C0{cycle}"
            / "attempts" / f"{number:02d}"
        )

    @staticmethod
    def _check_repair_attempt_records(
        run_dir: Path, cycle: int,
    ) -> tuple[CheckRepairAttempt, ...]:
        root = run_dir / "revision" / "check-repair" / f"C0{cycle}" / "attempts"
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

    def _read_durable_check_repair_scope(
        self, run_dir: Path, cycle: int, number: int, base_scope: Sequence[str],
    ) -> CheckRepairScope:
        return _read_check_repair_scope(
            self._check_repair_attempt_root(run_dir, cycle, number),
            fallback_base=base_scope,
            policy_config=self._effective_repair_scope,
            allowed_added_sources=frozenset({_AUTO_BOUNDED_SOURCE}),
        )

    def _check_repair_scope_union(
        self, run_dir: Path, cycle: int, base_scope: Sequence[str],
    ) -> list[str]:
        records = self._check_repair_attempt_records(run_dir, cycle)
        paths = set(base_scope)
        for record in records:
            scope = self._read_durable_check_repair_scope(
                run_dir, cycle, record.number, base_scope,
            )
            paths.update(scope.effective_paths)
        return sorted(paths)

    @staticmethod
    def _check_repair_chain_tree(
        run_dir: Path, cycle: int, initial_tree: str,
        checkpoint: ResumeCheckpoint,
    ) -> str:
        records = Orchestrator._check_repair_attempt_records(run_dir, cycle)
        if checkpoint.check_repair_attempt is None:
            completed = len(records)
        elif checkpoint.phase in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_C02,
        }:
            completed = checkpoint.check_repair_attempt - 1
        else:
            completed = checkpoint.check_repair_attempt
        expected = initial_tree
        for number in range(1, completed + 1):
            record = next((item for item in records if item.number == number), None)
            if record is None or record.tree_before != expected:
                raise ResumeIntegrityError(
                    f"C0{cycle} check-repair attempt chain is incomplete"
                )
            expected = record.tree_after
        return expected

    @staticmethod
    def _check_repair_attempt_state(result: CheckRepairResult) -> dict[str, Any]:
        return {
            "status": result.status,
            "attempt_count": len(result.attempts),
            "attempts": [asdict(item) for item in result.attempts],
        }

    def _write_check_repair_attempt_alias(
        self, run_dir: Path, cycle: int, attempt_dir: Path,
    ) -> None:
        """Publish compact latest-attempt metadata for old readers.

        Complete agent logs stay in ``attempts/NN``.  Only bounded metadata is
        mirrored at the historical cycle root so old resume/diagnostic readers
        can still inspect a run without duplicating large logs.
        """

        root = attempt_dir.parent.parent
        for name in ("report.json", "scope.json", "tree_before.txt", "tree_after.txt"):
            source = attempt_dir / name
            if not source.is_file():
                continue
            if name.endswith(".txt"):
                value = _read_bounded_text(source, 4096)
                atomic_write_text(root / name, value)
            else:
                payload = _read_json_artifact(source, 1024 * 1024)
                if payload is not None:
                    atomic_write_text(root / name, _json_text(payload))

    def _check_repair_scope_for_attempt(
        self, *, repo: Path, worktree: Path, run_dir: Path,
        evidence: EvidenceBundle, base_scope: Sequence[str],
        previous_scope: CheckRepairScope | None,
    ) -> CheckRepairScope:
        """Resolve scope for this attempt independently of retry budget."""

        policy = self._effective_repair_scope
        if previous_scope is None:
            return self._resolve_check_repair_scope(
                repo=repo, worktree=worktree, tree_sha=evidence.staged_tree_sha,
                run_dir=run_dir, evidence=evidence, base_mutable_scope=base_scope,
            )
        base_paths = tuple(sorted(set(previous_scope.base_paths)))
        added = set(previous_scope.added_paths)
        if policy.policy == "auto-bounded":
            candidates = _check_repair_scope_candidates(
                repo=repo, worktree=worktree, tree_sha=evidence.staged_tree_sha,
                run_dir=run_dir, evidence=evidence,
                base_mutable_scope=previous_scope.effective_paths,
            )
            proposed = added | set(candidates)
            if len(proposed) <= policy.max_added_paths:
                added = proposed
        effective = tuple(sorted(set(base_paths) | added))
        return CheckRepairScope(
            base_paths=base_paths,
            added_paths=tuple(sorted(added)),
            effective_paths=effective,
            policy=policy.policy,
            bound=policy.max_added_paths,
            source=(
                _AUTO_BOUNDED_SOURCE
                if added else "human-approved mutable scope"
            ),
        )

    def _run_direct_check_repair_loop(
        self, *, store: RunStateStore, run_dir: Path, repo: Path,
        base_sha: str, base_tree_sha: str, spec: str, plan: TaskPlanV2,
        repository_reference: RepositoryReference, info: WorktreeInfo,
        branch_ref: str, ownership_before: GitOwnership,
        selection: ExecutionSelectionV4 | ExecutionSelectionV5,
        cycle: int, evidence: EvidenceBundle, base_scope: Sequence[str],
        required_check_ids: Sequence[str] | None,
        expected_head_sha: str, resumed: _ResumedRun | None = None,
    ) -> tuple[CheckRepairResult, EvidenceBundle]:
        """Run the sole v2 mechanical correction policy.

        A successful worker is followed by exactly one authoritative check
        run.  A red result consumes no scope authority; it only advances the
        durable attempt ordinal until the configured budget is exhausted.
        """

        check_dir = run_dir / "checks" / f"C0{cycle}"
        repair_root = run_dir / "revision" / "check-repair" / f"C0{cycle}"
        repair_root.mkdir(parents=True, exist_ok=True)
        max_attempts = self._run_options.max_check_repair_attempts
        records = list(self._check_repair_attempt_records(run_dir, cycle))
        checkpoint = resumed.checkpoint if resumed is not None else None
        direct_phases = {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.FINAL_CHECKS_RETRY_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.FINAL_CHECKS_RETRY_C02,
        }
        phase = checkpoint.phase if checkpoint is not None else None
        attempt_number = (
            checkpoint.check_repair_attempt
            if checkpoint is not None and phase in direct_phases
            and checkpoint.check_repair_attempt is not None
            else len(records) + 1
        )

        # A resume after the worker has completed but before its checks must
        # re-run only the checks.  The current evidence is authoritative only
        # when it belongs to the checkpointed repaired tree.
        if phase in {
            ResumePhase.FINAL_CHECKS_RETRY_C01,
            ResumePhase.FINAL_CHECKS_RETRY_C02,
        }:
            current = _load_evidence(check_dir)
            current_tree = candidate_tree_sha(info.worktree)
            if current is not None and current.staged_tree_sha == current_tree:
                evidence = current

        if evidence.deterministic_passed:
            result = CheckRepairResult("passed", tuple(records))
            return result, evidence

        hard = _hard_integrity_failures(evidence)
        if hard:
            return CheckRepairResult("integrity-failed", tuple(records)), evidence

        if max_attempts <= 0:
            return CheckRepairResult("exhausted", tuple(records)), evidence

        selected_profile = (
            getattr(selection, "check_repair", None)
            or getattr(selection, "repair_implementer", None)
        )
        if selected_profile is None:
            return CheckRepairResult("agent-failed", tuple(records)), evidence

        previous_scope: CheckRepairScope | None = None
        if records:
            previous_scope = self._read_durable_check_repair_scope(
                run_dir, cycle, records[-1].number, base_scope,
            )

        while not evidence.deterministic_passed:
            hard = _hard_integrity_failures(evidence)
            if hard:
                result = CheckRepairResult("integrity-failed", tuple(records))
                return result, evidence
            soft = _soft_check_failures(evidence)
            if not soft:
                result = CheckRepairResult("integrity-failed", tuple(records))
                return result, evidence
            if attempt_number > max_attempts or max_attempts <= 0:
                result = CheckRepairResult("exhausted", tuple(records))
                return result, evidence

            attempt_dir = self._check_repair_attempt_root(run_dir, cycle, attempt_number)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt_record = next(
                (item for item in records if item.number == attempt_number), None
            )
            if attempt_record is not None:
                # A durable attempt already owns the worker result.  This is
                # the resume-after-worker boundary; never call the worker
                # again, even if the check artifact was not written yet.
                tree_after = attempt_record.tree_after
            else:
                if _load_evidence(check_dir) is not None:
                    _archive_attempt(check_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                failed_ids = tuple(
                    item.split(":", 1)[1] for item in soft if ":" in item
                )
                scope = self._check_repair_scope_for_attempt(
                    repo=repo, worktree=info.worktree, run_dir=run_dir,
                    evidence=evidence, base_scope=base_scope,
                    previous_scope=previous_scope,
                )
                previous_scope = scope
                atomic_write_text(
                    attempt_dir / "failed_check_evidence_before.json",
                    _json_text(_check_payload(evidence)),
                )
                atomic_write_text(
                    attempt_dir / "profile.json",
                    _json_text({
                        "profile_id": selected_profile.profile_id,
                        "profile_fingerprint": selected_profile.config_sha256,
                        "selected_profile": asdict(selected_profile),
                    }),
                )
                self._cycle_update(
                    store, cycle, status="check_repair_attempted",
                    automatic_check_repair={
                        "status": "running", "attempt_number": attempt_number,
                        "before": _check_payload(evidence),
                        "repair_profile_id": selected_profile.profile_id,
                        "repair_profile_fingerprint": selected_profile.config_sha256,
                        "mutable_scope": list(scope.effective_paths),
                    },
                )
                store.update(
                    status=RunStatus.REVISING,
                    check_repair={
                        "status": "running", "attempt_count": len(records),
                        "attempt_number": attempt_number,
                        "failure_ids": list(soft),
                        "repair_profile_id": selected_profile.profile_id,
                        "repair_profile_fingerprint": selected_profile.config_sha256,
                        "mutable_scope": list(scope.effective_paths),
                    },
                )
                try:
                    _result, repair_error = self._run_v2_revision_cycle(
                        store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                        base_tree_sha=base_tree_sha, spec=spec, plan=plan,
                        repository_reference=repository_reference, info=info,
                        branch_ref=branch_ref, ownership_before=ownership_before,
                        selection=selection, mutable_scope=list(scope.effective_paths),
                        check_repair_evidence=evidence, cycle=cycle,
                        artifact_dir=attempt_dir,
                        check_repair_scope=scope,
                        check_repair_phase_override=(
                            ResumePhase.CHECK_REPAIR_C01
                            if cycle == 1 else ResumePhase.CHECK_REPAIR_C02
                        ),
                        check_repair_next_phase_override=(
                            ResumePhase.FINAL_CHECKS_RETRY_C01
                            if cycle == 1 else ResumePhase.FINAL_CHECKS_RETRY_C02
                        ),
                        check_repair_attempt=attempt_number,
                        _legacy_failure_names=False,
                    )
                except (AgentError, GitError, OSError) as exc:
                    repair_error = redact(str(exc), self._secrets) or AGENT_RUNTIME_FAILED
                    _result = None
                if repair_error is not None:
                    status = (
                        "scope-required" if repair_error == _SCOPE_REQUEST_ROUTE
                        else "integrity-failed"
                        if repair_error in {"TOCTOU_FAILURE", "AGENT_GIT_VIOLATION", AGENT_SCOPE_VIOLATION}
                        else "agent-failed"
                    )
                    # The worker boundary itself is durable even when the
                    # worker did not produce a successful revision report.
                    # This keeps the attempt ledger useful for diagnostics and
                    # prevents a runtime failure from being confused with a
                    # deterministic check failure on resume.
                    tree_after = _safe_candidate_tree(info.worktree) or evidence.staged_tree_sha or ""
                    failed_record = CheckRepairAttempt(
                        number=attempt_number,
                        failed_check_ids_before=tuple(failed_ids),
                        tree_before=evidence.staged_tree_sha or "",
                        tree_after=tree_after,
                        mutable_scope=tuple(scope.effective_paths),
                    )
                    atomic_write_text(
                        attempt_dir / "failure.json",
                        _json_text({
                            "schema_version": 1,
                            **asdict(failed_record),
                            "profile_id": selected_profile.profile_id,
                            "profile_fingerprint": selected_profile.config_sha256,
                            "status": status,
                            "error": repair_error,
                        }),
                    )
                    # A structured scope request already has an authoritative
                    # RevisionRunner report (including changed/outside paths).
                    # Preserve it for the existing scope-recovery route; a
                    # synthetic summary is only needed for infrastructure
                    # failures that produced no report of their own.
                    report_path = attempt_dir / "report.json"
                    if not report_path.is_file():
                        atomic_write_text(
                            report_path,
                            _json_text({
                                "status": status,
                                "failure_ids": list(soft),
                                "error": repair_error,
                                "profile_id": selected_profile.profile_id,
                                "tree_before": failed_record.tree_before,
                                "tree_after": failed_record.tree_after,
                                "changed_paths": [],
                            }),
                        )
                    self._write_check_repair_attempt_alias(run_dir, cycle, attempt_dir)
                    store.update(
                        status=RunStatus.REVISING,
                        check_repair={
                            "status": status,
                            "attempt_count": attempt_number,
                            "failure_ids": list(soft),
                            "repair_profile_id": selected_profile.profile_id,
                            "repair_profile_fingerprint": selected_profile.config_sha256,
                            "error": repair_error,
                        },
                    )
                    return CheckRepairResult(status, tuple(records)), evidence
                tree_after = candidate_tree_sha(info.worktree)
                record = CheckRepairAttempt(
                    number=attempt_number,
                    failed_check_ids_before=failed_ids,
                    tree_before=evidence.staged_tree_sha or "",
                    tree_after=tree_after,
                    mutable_scope=tuple(scope.effective_paths),
                )
                records.append(record)
                atomic_write_text(
                    attempt_dir / "attempt.json",
                    _json_text({
                        "schema_version": 1,
                        **asdict(record),
                        "profile_id": selected_profile.profile_id,
                        "profile_fingerprint": selected_profile.config_sha256,
                        "status": "completed",
                    }),
                )
                self._write_check_repair_attempt_alias(run_dir, cycle, attempt_dir)

            self._checkpoint(
                run_dir,
                ResumePhase.FINAL_CHECKS_RETRY_C01 if cycle == 1 else ResumePhase.FINAL_CHECKS_RETRY_C02,
                cycle=cycle, head=expected_head_sha, tree=tree_after,
                check_repair_attempt=attempt_number,
            )
            store.update(status=RunStatus.REVALIDATING, current_step=None)
            try:
                evidence = self._final_evidence(
                    info.worktree, base_sha, check_dir, check_failures_hard=False,
                    reuse=False, expected_head_sha=expected_head_sha,
                    required_check_ids=required_check_ids, enforce_diff_size=False,
                )
            except Exception:
                self._checkpoint(
                    run_dir,
                    ResumePhase.FINAL_CHECKS_RETRY_C01 if cycle == 1 else ResumePhase.FINAL_CHECKS_RETRY_C02,
                    cycle=cycle, head=expected_head_sha, tree=tree_after,
                    check_repair_attempt=attempt_number,
                )
                raise
            atomic_write_text(
                attempt_dir / "checks_after.json",
                _json_text(_check_payload(evidence)),
            )
            self._write_check_repair_attempt_alias(run_dir, cycle, attempt_dir)
            store.update(
                status=RunStatus.REVALIDATING,
                checks=_check_payload(evidence),
                staged_tree_sha=evidence.staged_tree_sha,
                changed_files=list(evidence.changed_files),
                deterministic_gate={
                    "passed": evidence.deterministic_passed,
                    "required_check_ids": list(evidence.required_check_ids),
                    "failures": list(evidence.failures),
                },
                check_repair={
                    "status": "passed" if evidence.deterministic_passed else "running",
                    "attempt_count": len(records),
                    "attempts": [asdict(item) for item in records],
                    "remaining_failed_check_ids": [
                        item.split(":", 1)[1] for item in _soft_check_failures(evidence)
                        if ":" in item
                    ],
                    "repair_profile_id": selected_profile.profile_id,
                    "repair_profile_fingerprint": selected_profile.config_sha256,
                },
            )
            hard = _hard_integrity_failures(evidence)
            if hard:
                return CheckRepairResult("integrity-failed", tuple(records)), evidence
            if evidence.deterministic_passed:
                return CheckRepairResult("passed", tuple(records)), evidence
            attempt_number += 1
            self._checkpoint(
                run_dir,
                ResumePhase.CHECK_REPAIR_C01 if cycle == 1 else ResumePhase.CHECK_REPAIR_C02,
                cycle=cycle, head=expected_head_sha,
                tree=evidence.staged_tree_sha,
                check_repair_attempt=attempt_number,
            )

        return CheckRepairResult("passed", tuple(records)), evidence

    def _execute_scope_repair_cycle(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        repository_reference: RepositoryReference,
        info: WorktreeInfo,
        branch_ref: str,
        ownership_before: GitOwnership,
        selection: ExecutionSelectionV4 | ExecutionSelectionV5,
        original_plan: TaskPlanV2,
        original_bundle: Mapping[str, Any],
        resumed: _ResumedRun,
        cycle: int,
    ) -> RunResult:
        """Execute the bounded Luna repair created after scope recovery.

        The method is shared by C01 and C02.  It is deliberately entered only
        after :meth:`_recover_failed_revision_scope_violation` has restored
        the failed candidate exactly; no failed Claude edit is carried into
        this state machine.
        """

        self._trace_cycle = cycle

        del original_bundle
        phase = resumed.checkpoint.phase
        at = phase_index(phase)
        scope_dir = run_dir / "scope-repair" / f"C0{cycle}"
        scope_dir.mkdir(parents=True, exist_ok=True)
        normal_old_dir = run_dir / "revision" / "check-repair" / f"C0{cycle}"
        expanded_old_dir = run_dir / "revision" / "check-repair-expanded" / f"C0{cycle}"
        old_dir = (
            expanded_old_dir
            if phase in {ResumePhase.CHECK_REPAIR_EXPANDED_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C02}
            or (
                not (normal_old_dir / "scope_violation_recovery.json").is_file()
                and (expanded_old_dir / "scope_violation_recovery.json").is_file()
            )
            else normal_old_dir
        )
        recovery = resumed.scope_violation_recovery
        if not isinstance(recovery, Mapping):
            raise ResumeIntegrityError("scope-repair recovery evidence is missing")
        outside_paths = recovery.get("outside_scope_paths")
        if not isinstance(outside_paths, list) or any(not isinstance(path, str) for path in outside_paths):
            raise ResumeIntegrityError("scope-repair outside-scope evidence is malformed")
        scope_request = _scope_request_from_payload(recovery.get("scope_request"))
        if recovery.get("scope_request") is not None and scope_request is None:
            raise ResumeIntegrityError("scope-repair Claude scope request is malformed")
        if scope_request is not None:
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                },
            )

        if cycle == 1:
            failed_plan = original_plan
            failed_evidence = resumed.c01_evidence
        else:
            failed_plan = resumed.repair_plan
            failed_evidence = resumed.c02_evidence
        if failed_plan is None:
            raise ResumeIntegrityError("scope-repair source plan is missing")
        if failed_evidence is None:
            raise ResumeIntegrityError("scope-repair failed-check evidence is missing")

        fallback_scope = sorted({
            path for step in failed_plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        failed_scope = _read_check_repair_scope(
            old_dir, fallback_base=fallback_scope,
            policy_config=self._effective_repair_scope,
            allowed_added_sources=_SECOND_SCOPE_SOURCES,
        )
        original_scope = list(failed_scope.effective_paths)
        planner_profile = profile_for_role(
            self.config, selection.planner.profile_id, ExecutionRole.PLANNER
        )
        repair_profile = profile_for_role(
            self.config, selection.repair_implementer.profile_id, ExecutionRole.REPAIR
        )
        reviewer_profile = profile_for_role(
            self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER
        )
        planner_dir = scope_dir
        current_tree = candidate_tree_sha(info.worktree)
        if current_tree != resumed.checkpoint.expected_tree_sha and phase in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
        }:
            raise ResumeIntegrityError("scope-repair did not start from the rolled-back checkpoint tree")

        current_state = _json_text({
            "BASE_SHA": base_sha,
            "CURRENT_TREE_SHA": current_tree,
            "HEAD_SHA": current_head(info.worktree),
            "GIT_STATUS": status_porcelain(info.worktree),
            "REMOTE_REPOSITORY_IS_EVIDENCE": True,
            "IMMUTABLE_BASE_SHA": base_sha,
            "CURRENT_WORKTREE_DIFF": bounded_semantic_diff(
                failed_evidence.diff, _MAX_REVIEW_FALLBACK_DIFF_BYTES
            )[0],
        })
        report = _read_json_artifact(old_dir / "report.json", 1024 * 1024)
        if not isinstance(report, dict):
            raise ResumeIntegrityError("failed Claude scope report is missing")
        failed_report = _json_text({
            "tree_before": report.get("tree_before"),
            "tree_after": report.get("tree_after"),
            "changed_paths": report.get("changed_paths"),
            "outside_scope_paths": report.get("outside_scope_paths"),
            "failure_ids": report.get("failure_ids", list(failed_evidence.failures)),
            "final": _bounded_report(_read_bounded_text(old_dir / "agent.final.md")),
        })

        def parse_scope_plan() -> tuple[TaskPlanV2, dict[str, Any], str]:
            try:
                raw = (scope_dir / "planner.raw.md").read_text(encoding="utf-8")
                plan_candidate = parse_task_plan_v2(
                    raw,
                    implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                    reviewer_ids=frozenset({selection.reviewer.profile_id}),
                    check_catalog=self.config.check_catalog,
                    inherited_check_ids=failed_plan.required_checks,
                )
                validate_repair_decomposition_policy(plan_candidate, self.config.planning)
                if plan_candidate.decision is not PlanDecision.READY:
                    raise OrchestrationError("SCOPE_REPAIR_PLANNER_BLOCKED")
                bundle_candidate, bundle_sha_candidate = validate_implementation_bundle(
                    scope_dir, expected_step_ids=[step.id for step in plan_candidate.steps]
                )
            except (OSError, UnicodeError, V2PlanParseError, ValueError, AttributeError) as exc:
                raise ResumeIntegrityError(f"scope-repair plan is unreadable: {exc}") from exc
            return plan_candidate, bundle_candidate, bundle_sha_candidate

        repair_plan: TaskPlanV2
        repair_bundle: dict[str, Any]
        repair_bundle_sha: str
        if phase in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
        }:
            # Publish the new operation boundary before the strong planner is
            # invoked.  A transport crash is therefore a planner resume, not
            # a replay of Claude's failed check-repair.
            self._checkpoint(
                run_dir, phase_planner if "phase_planner" in locals() else (
                    ResumePhase.CHECK_SCOPE_PLANNER_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_PLANNER_C02
                ), cycle=cycle, head=current_head(info.worktree), tree=current_tree,
            )
        # A crash after the bridge wrote its bundle but before the phase moved
        # must reuse the exact answer.  Evidence bytes are the binding, so a
        # stale planner response cannot be accidentally adopted.
        planner_bundle = build_scope_repair_planner_prompt_bundle(
            repository_reference=render_repository_reference(repository_reference),
            original_spec=spec,
            original_plan_summary=render_repair_plan_summary(failed_plan),
            original_step_index=render_repair_step_index(failed_plan),
            current_repository_state=current_state,
            failed_checks=_json_text(_check_payload(failed_evidence)),
            current_authorized_mutable_scope=_json_text(original_scope),
            failed_claude_repair_report=failed_report,
            claude_scope_request=_scope_request_evidence(scope_request),
            outside_scope_paths_observed=_json_text(outside_paths),
            implementer_profiles=(repair_profile,),
            reviewer_profiles=(reviewer_profile,),
            check_catalog=self.config.check_catalog,
            original_required_check_ids=failed_plan.required_checks,
            staged_step_max_mutable_paths=self.config.planning.staged_step_max_mutable_paths,
        )
        reusable = False
        if at <= phase_index(ResumePhase.CHECK_SCOPE_PLANNER_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_PLANNER_C02):
            try:
                reusable = (
                    (scope_dir / "planner.evidence.md").read_bytes()
                    == planner_bundle.evidence_text.encode("utf-8")
                    and (scope_dir / "planner.raw.md").is_file()
                    and (scope_dir / "implementation_bundle.json").is_file()
                )
            except OSError:
                reusable = False
        if reusable:
            repair_plan, repair_bundle, repair_bundle_sha = parse_scope_plan()
        elif at <= phase_index(ResumePhase.CHECK_SCOPE_PLANNER_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_PLANNER_C02):
            planner = CheckScopeRepairPlannerV2(
                self._planner_client or _chat_client(
                    build_llm_endpoint(planner_profile), self._runtime_environment
                ),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                implementer_profiles=(repair_profile,), reviewer_profiles=(reviewer_profile,),
                planning=self.config.planning, check_catalog=self.config.check_catalog,
                original_required_check_ids=failed_plan.required_checks,
            )
            scope_plan_started_at = self._trace_time()
            scope_plan_started_mono = time.perf_counter()
            self._trace_emit(
                "plan.started",
                phase="repair",
                cycle=cycle,
                data={
                    "kind": "scope_repair",
                    "tree_before": current_tree,
                    "session": self._trace_session(
                        profile=planner_profile,
                        selected=self._trace_selected_profile(
                            selection.planner.profile_id, ExecutionRole.PLANNER
                        ),
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=None,
                        started_at=scope_plan_started_at,
                        started_mono=scope_plan_started_mono,
                        tree_before=current_tree,
                    ),
                },
            )
            try:
                repair_plan = planner.plan(
                    repository_reference=render_repository_reference(repository_reference),
                    original_spec=spec,
                    original_plan_summary=render_repair_plan_summary(failed_plan),
                    original_step_index=render_repair_step_index(failed_plan),
                    current_repository_state=current_state,
                    failed_checks=_json_text(_check_payload(failed_evidence)),
                    current_authorized_mutable_scope=_json_text(original_scope),
                    failed_claude_repair_report=failed_report,
                    claude_scope_request=_scope_request_evidence(scope_request),
                    outside_scope_paths_observed=_json_text(outside_paths),
                    artifacts_dir=scope_dir,
                    fallback_current_diff=failed_evidence.diff,
                )
                _persist_planner_conversation(
                    scope_dir, getattr(planner, "last_conversation", None)
                )
                self._trace_emit(
                    "plan.completed",
                    phase="repair",
                    cycle=cycle,
                    data={
                        "kind": "scope_repair",
                        "decision": repair_plan.decision.value,
                        "title": repair_plan.title,
                        "session": self._trace_finished_model_session(
                            profile=planner_profile,
                            selected=self._trace_selected_profile(
                                selection.planner.profile_id, ExecutionRole.PLANNER
                            ),
                            role=ExecutionRole.PLANNER,
                            prompt_bytes=(
                                (scope_dir / "planner.request.txt").stat().st_size
                                if (scope_dir / "planner.request.txt").is_file() else None
                            ),
                            started_at=scope_plan_started_at,
                            started_mono=scope_plan_started_mono,
                            usage=getattr(planner, "last_usage", None),
                            tree_before=current_tree,
                            tree_after=current_tree,
                        ),
                    },
                )
            except LLMError as exc:
                raise OrchestrationError("LLM_FAILURE: scope-repair planner") from exc
            if repair_plan.decision is PlanDecision.BLOCKED:
                raise OrchestrationError("SCOPE_REPAIR_PLANNER_BLOCKED")
            repair_bundle, repair_bundle_sha = validate_implementation_bundle(
                scope_dir, expected_step_ids=[step.id for step in repair_plan.steps]
            )
        else:
            repair_plan, repair_bundle, repair_bundle_sha = parse_scope_plan()

        requested_scope = sorted({
            path for step in repair_plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        repair_base_tree = recovery.get("tree_before")
        if not isinstance(repair_base_tree, str) or not _is_object_id(repair_base_tree):
            raise ResumeIntegrityError("scope-repair rollback tree is malformed")
        writes, creates, deletes = _repair_mutation_sets(repair_plan)
        for path in (*writes, *deletes):
            if not path_exists_in_tree(repo, repair_base_tree, path):
                raise OrchestrationError("REPAIR_SCOPE_EXISTING_PATH_MISSING")
        for path in creates:
            if path_exists_in_tree(repo, repair_base_tree, path):
                raise OrchestrationError("REPAIR_SCOPE_CREATE_PATH_EXISTS")
        scope_delta, scope_delta_content = _build_scope_repair_delta(
            scope_dir, original_scope=original_scope, plan=repair_plan,
            failure_ids=failed_evidence.failures,
            observed_outside_scope_paths=outside_paths,
            repair_bundle_sha=repair_bundle_sha,
        )
        scope_delta_expected_sha = resumed.checkpoint.scope_delta_sha256
        if phase in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_PLANNER_C02,
        }:
            # C02's old check-repair checkpoint is bound to the ordinary C02
            # repair delta.  The new scope-repair planner starts a distinct
            # authority chain and must create its own delta hash.
            scope_delta_expected_sha = None
        scope_delta_sha = _ensure_scope_delta(
            scope_dir, scope_delta_content,
            expected_sha256=scope_delta_expected_sha,
        )
        scope_payload = {
            "schema_version": 1,
            "original_mutable_paths": original_scope,
            "requested_mutable_paths": requested_scope,
            "effective_mutable_paths": sorted(set(original_scope) | set(requested_scope)),
            "added_paths": scope_delta["added_paths"],
            "scope_delta_sha256": scope_delta_sha,
            "repair_bundle_sha256": repair_bundle_sha,
            "policy": self._effective_repair_scope.policy,
            "bound": self._effective_repair_scope.max_added_paths,
        }
        scope_text = _json_text(scope_payload)
        scope_path = scope_dir / "scope.json"
        if scope_path.exists():
            if scope_path.read_text(encoding="utf-8") != scope_text:
                raise ResumeIntegrityError("scope-repair scope artifact diverges")
        else:
            atomic_write_text(scope_path, scope_text)

        phase_planner = ResumePhase.CHECK_SCOPE_PLANNER_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_PLANNER_C02
        phase_approval = ResumePhase.CHECK_SCOPE_APPROVAL_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_APPROVAL_C02
        phase_step = ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02
        phase_checks = ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02
        phase_claude = ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02
        phase_final = ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01 if cycle == 1 else ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02
        if scope_delta["added_paths"] and self._effective_repair_scope.policy == "deny-expansion":
            raise OrchestrationError("REPAIR_SCOPE_EXPANSION")
        requires_approval = bool(scope_delta["added_paths"]) and (
            self._effective_repair_scope.policy == "require-approval"
            or (
                self._effective_repair_scope.policy == "auto-bounded"
                and len(scope_delta["added_paths"]) > self._effective_repair_scope.max_added_paths
            )
        )
        if requires_approval:
            approval = read_scope_approval(scope_dir, expected_sha256=scope_delta_sha)
            if approval is None:
                self._checkpoint(
                    run_dir, phase_approval, cycle=cycle, head=current_head(info.worktree),
                    tree=candidate_tree_sha(info.worktree), repair_bundle_sha256=repair_bundle_sha,
                    scope_delta_sha256=scope_delta_sha,
                )
                store.update(
                    status=RunStatus.WAITING_SCOPE_APPROVAL,
                    scope_delta=scope_delta,
                    scope_repair_scope_escalation={
                        "trigger": "REVISION_SCOPE_VIOLATION",
                        "added_paths": scope_delta["added_paths"],
                        "policy": self._effective_repair_scope.policy,
                    },
                )
                raise ScopeApprovalRequired()
            if approval.decision is not ApprovalDecision.APPROVE:
                raise OrchestrationError("HUMAN_REQUIRED: scope-repair scope rejected")
        elif scope_delta["added_paths"]:
            # The auto-bounded authorization is an annotation, not a phase
            # change: this cycle can be resumed from a later phase, so the
            # durable status is carried over instead of being regressed.
            current_status = store.load().get("status", PHASE_STATUS[phase_planner])
            store.update(
                status=current_status,
                scope_repair_scope_escalation={
                    "trigger": "REVISION_SCOPE_VIOLATION",
                    "added_paths": scope_delta["added_paths"],
                    "policy": self._effective_repair_scope.policy,
                    "auto_authorized": True,
                },
            )

        ids = [step.id for step in repair_plan.steps]
        completed: list[dict[str, Any]] = []
        for step_id in ids:
            record = _load_completed_step(scope_dir / "steps" / step_id, step_id)
            if record is not None:
                completed.append(record)
        start_tree = resumed.checkpoint.expected_tree_sha
        chain_end = _verify_step_chain(completed, recovery.get("tree_before", start_tree))
        if chain_end is None:
            raise ResumeIntegrityError("scope-repair Luna step chain is broken")
        pending_steps = [
            step for step in repair_plan.steps
            if step.id not in {record["id"] for record in completed}
        ]
        if pending_steps and (
            at <= phase_index(phase_step)
            or phase in {phase_planner, phase_approval}
        ):
            # Establish the exact step boundary before starting Luna.  A
            # crash in the worker is therefore resumable at Sxx, not at the
            # planner boundary that produced its contract.
            self._checkpoint(
                run_dir, phase_step, cycle=cycle, step_id=pending_steps[0].id,
                head=current_head(info.worktree), tree=candidate_tree_sha(info.worktree),
                repair_bundle_sha256=repair_bundle_sha, scope_delta_sha256=scope_delta_sha,
            )
        # The old check-repair checkpoint and the new planner/approval/step
        # checkpoints all begin from the exact rolled-back candidate.  Only
        # the later checks/Claude phases resume from the completed Luna chain.
        expected_tree = (
            start_tree
            if phase in {phase_step, phase_planner, phase_approval}
            or phase in {
                ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
            }
            else chain_end
        )
        done = {record["id"] for record in completed}
        if at <= phase_index(phase_step) or phase in {phase_planner, phase_approval}:
            repair_codex_home = self.config.codex_runtime.home
            forbidden_env_names = (
                planner_profile.api_key_env,
                reviewer_profile.api_key_env,
            )
            for index, step in enumerate(repair_plan.steps):
                if step.id in done:
                    continue
                contract = read_approved_step_contract(scope_dir, repair_bundle, step.id)
                step_dir = scope_dir / "steps" / step.id
                store.update(status=RunStatus.IMPLEMENTING, current_step=step.id)
                try:
                    outcome = self._execute_codex_step(
                        repo=repo, worktree=info.worktree, base_sha=current_head(info.worktree),
                        branch_ref=branch_ref, ownership_before=ownership_before,
                        expected_tree=expected_tree, step=step, contract=contract,
                        profile_id=selection.repair_implementer.profile_id,
                        artifact_dir=step_dir, codex_home=repair_codex_home,
                        forbidden_env_names=forbidden_env_names,
                        future_ownership=_future_step_ownership(repair_plan.steps, index),
                        pending_mismatch_retry=None,
                    )
                except StepExecutionFailure as failure:
                    return self._step_failed(store, run_dir, failure, step_dir)
                expected_tree = outcome.tree_after
                completed.append(_step_result_record(outcome))
                self._v2_usage_rows.append({"id": step.id, **outcome.usage})
                following = ids[index + 1] if index + 1 < len(ids) else None
                self._checkpoint(
                    run_dir, phase_step if following else phase_checks, cycle=cycle,
                    step_id=following, head=current_head(info.worktree), tree=expected_tree,
                    repair_bundle_sha256=repair_bundle_sha, scope_delta_sha256=scope_delta_sha,
                )
            done = {record["id"] for record in completed}
            chain_end = expected_tree
        if len(done) != len(ids):
            raise ResumeIntegrityError("scope-repair did not complete every Luna step")

        checks_dir = scope_dir / "checks"
        checks_dir.mkdir(parents=True, exist_ok=True)
        evidence = _load_evidence(checks_dir)
        if at <= phase_index(phase_checks) or phase in {phase_planner, phase_approval, phase_step}:
            if evidence is None or evidence.staged_tree_sha != candidate_tree_sha(info.worktree):
                check_tree = candidate_tree_sha(info.worktree)
                self._checkpoint(
                    run_dir, phase_checks, cycle=cycle, head=current_head(info.worktree),
                    tree=check_tree, repair_bundle_sha256=repair_bundle_sha,
                    scope_delta_sha256=scope_delta_sha,
                )
                evidence = self._final_evidence(
                    info.worktree, base_sha, checks_dir, check_failures_hard=False,
                    reuse=False, expected_head_sha=current_head(info.worktree),
                    required_check_ids=failed_plan.required_checks or None,
                    enforce_diff_size=False,
                )
        # A resume at the final-checks boundary legitimately finds no current
        # bundle: the red pre-residual one was archived at the exact moment
        # the residual Claude superseded it.  That archived bundle is the
        # proof this state is the expected one; the final checks below are
        # then re-run for the residual tree.  Every other phase still
        # requires the current bundle.
        archived_red = _load_evidence(checks_dir / "attempts" / "01")
        if evidence is None and not (phase is phase_final and archived_red is not None):
            raise ResumeIntegrityError("scope-repair checks evidence is missing")
        hard = _hard_integrity_failures(evidence) if evidence is not None else []
        if hard:
            raise OrchestrationError(hard[0].split(":", 1)[0])
        effective_scope = sorted(set(original_scope) | set(requested_scope))
        residual_result = None
        if evidence is not None and not evidence.deterministic_passed:
            soft = _soft_check_failures(evidence)
            if not soft:
                raise OrchestrationError("DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures))
            residual_dir = scope_dir / "residual-claude"
            if at <= phase_index(phase_claude) or phase in {phase_planner, phase_approval, phase_step, phase_checks}:
                self._checkpoint(
                    run_dir, phase_claude, cycle=cycle, head=current_head(info.worktree),
                    tree=candidate_tree_sha(info.worktree), repair_bundle_sha256=repair_bundle_sha,
                    scope_delta_sha256=scope_delta_sha,
                )
                persisted = _load_revision(residual_dir)
                if persisted is not None and persisted.tree_before == candidate_tree_sha(info.worktree):
                    residual_result = persisted
                else:
                    residual_scope = CheckRepairScope(
                        base_paths=tuple(original_scope),
                        added_paths=tuple(sorted(set(effective_scope) - set(original_scope))),
                        effective_paths=tuple(effective_scope),
                        policy=self._effective_repair_scope.policy,
                        bound=self._effective_repair_scope.max_added_paths,
                        source="scope-repair planner authorized scope",
                    )
                    try:
                        residual_result, residual_error = self._run_v2_revision_cycle(
                            store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                            base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                            plan=failed_plan, repository_reference=repository_reference,
                            info=info, branch_ref=branch_ref, ownership_before=ownership_before,
                            selection=selection, artifact_dir=residual_dir,
                            mutable_scope=effective_scope, check_repair_evidence=evidence,
                            cycle=cycle, check_repair_scope=residual_scope,
                            check_repair_phase_override=phase_claude,
                            check_repair_next_phase_override=phase_final,
                        )
                    except AgentError as exc:
                        raise OrchestrationError("CLAUDE_FAILED: residual scope repair") from exc
                    if residual_error is not None:
                        raise OrchestrationError(residual_error)
            if residual_result is not None:
                _archive_attempt(checks_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                evidence = None
        if evidence is None or not evidence.deterministic_passed:
            if at <= phase_index(phase_final) or phase in {phase_claude}:
                self._checkpoint(
                    run_dir, phase_final, cycle=cycle, head=current_head(info.worktree),
                    tree=candidate_tree_sha(info.worktree), repair_bundle_sha256=repair_bundle_sha,
                    scope_delta_sha256=scope_delta_sha,
                )
                evidence = self._final_evidence(
                    info.worktree, base_sha, checks_dir, check_failures_hard=False,
                    reuse=False, expected_head_sha=current_head(info.worktree),
                    required_check_ids=failed_plan.required_checks or None,
                    enforce_diff_size=False,
                )
            if evidence is None or not evidence.deterministic_passed:
                raise OrchestrationError(
                    "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures if evidence else ())
                )
        self._update_v2_usage(store, run_dir)
        if cycle == 1:
            resumed_next = dataclasses.replace(resumed, checkpoint=ResumeCheckpoint(
                ResumePhase.CANDIDATE_COMMIT_C01, 1, None, base_sha,
                evidence.staged_tree_sha, resumed.checkpoint.execution_selection_sha256,
                resumed.checkpoint.plan_identity, repair_bundle_sha, scope_delta_sha,
            ), c01_evidence=evidence)
        else:
            c01 = _read_json_artifact(_candidate_commit_path(run_dir, 1))
            if not isinstance(c01, dict) or not _is_object_id(c01.get("commit_sha")):
                raise ResumeIntegrityError("C01 candidate commit is missing for C02 scope repair")
            resumed_next = dataclasses.replace(resumed, checkpoint=ResumeCheckpoint(
                ResumePhase.CANDIDATE_COMMIT_C02, 2, None, c01["commit_sha"],
                evidence.staged_tree_sha, resumed.checkpoint.execution_selection_sha256,
                resumed.checkpoint.plan_identity, repair_bundle_sha, scope_delta_sha,
            ), c02_evidence=evidence)
        write_checkpoint(run_dir, resumed_next.checkpoint)
        if cycle == 2:
            return self._complete_scope_repair_c02_candidate(
                store=store, run_dir=run_dir, run_id=run_id, spec=spec,
                repository_reference=repository_reference, info=info,
                branch_ref=branch_ref, selection=selection,
                original_plan=original_plan, resumed=resumed_next,
                scope_evidence=evidence, scope_delta=scope_delta,
                scope_delta_sha=scope_delta_sha, scope_bundle_sha=repair_bundle_sha,
            )
        return self._execute_v2(
            store, run_dir, run_id, spec, repo, base_sha,
            resumed.context, repository_reference, resumed=resumed_next,
        )

    def _complete_scope_repair_c02_candidate(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repository_reference: RepositoryReference,
        info: WorktreeInfo,
        branch_ref: str,
        selection: ExecutionSelectionV4 | ExecutionSelectionV5,
        original_plan: TaskPlanV2,
        resumed: _ResumedRun,
        scope_evidence: EvidenceBundle,
        scope_delta: Mapping[str, Any],
        scope_delta_sha: str,
        scope_bundle_sha: str,
    ) -> RunResult:
        """Commit/review a green C02 scope-repair candidate directly.

        C02's ordinary repair bundle and this recovery bundle are separate
        authorities.  Once the scope-repair checks are green, this helper
        performs only the candidate/reviewer tail and never re-enters the
        ordinary C02 repair planner.
        """

        if not scope_evidence.deterministic_passed:
            raise OrchestrationError(
                "DETERMINISTIC_GATE_FAILED: " + ", ".join(scope_evidence.failures)
            )
        c01_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(c01_candidate, dict) or not _is_object_id(c01_candidate.get("commit_sha")):
            raise ResumeIntegrityError("C01 candidate commit is missing for C02 scope repair")
        start = resumed.checkpoint
        if phase_index(start.phase) <= phase_index(ResumePhase.CANDIDATE_COMMIT_C02):
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_COMMIT_C02, cycle=2,
                head=c01_candidate["commit_sha"], tree=scope_evidence.staged_tree_sha,
                repair_bundle_sha256=scope_bundle_sha, scope_delta_sha256=scope_delta_sha,
            )
            if current_head(info.worktree) == c01_candidate["commit_sha"]:
                self._authorize_candidate_tree(
                    scope_evidence, info.worktree, c01_candidate["commit_sha"], branch_ref
                )
            elif not (
                resumed.existing_commit_sha is not None
                and commit_parents(info.worktree, resumed.existing_commit_sha) == (c01_candidate["commit_sha"],)
                and resolve_tree(info.worktree, resumed.existing_commit_sha) == scope_evidence.staged_tree_sha
            ):
                raise ResumeIntegrityError("C02 scope candidate commit exists with the wrong identity")
            c02_candidate = self._ensure_candidate_commit(
                run_dir=run_dir, info=info, cycle=2, tree_sha=scope_evidence.staged_tree_sha,
                parent_sha=c01_candidate["commit_sha"], title=resumed.repair_plan.title,
                repository_reference=repository_reference, store=store, run_id=run_id,
            )
        else:
            c02_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 2))
            if not isinstance(c02_candidate, dict):
                raise ResumeIntegrityError("C02 scope candidate commit is missing")
        if phase_index(start.phase) <= phase_index(ResumePhase.CANDIDATE_PUSH_C02):
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_PUSH_C02, cycle=2,
                head=c02_candidate["commit_sha"], tree=scope_evidence.staged_tree_sha,
                repair_bundle_sha256=scope_bundle_sha, scope_delta_sha256=scope_delta_sha,
            )
            c02_candidate = self._push_candidate(
                run_dir=run_dir, info=info, cycle=2, candidate=c02_candidate, store=store,
            )
        self._checkpoint(
            run_dir, ResumePhase.REVIEWER_C02, cycle=2,
            head=c02_candidate["commit_sha"], tree=scope_evidence.staged_tree_sha,
            repair_bundle_sha256=scope_bundle_sha, scope_delta_sha256=scope_delta_sha,
        )
        cycle_1_evidence = resumed.c01_evidence
        cycle_1_review = resumed.c01_review
        if cycle_1_evidence is None or cycle_1_review is None:
            raise ResumeIntegrityError("C01 review evidence is missing for C02 scope repair")
        scope_dir = run_dir / "scope-repair" / "C02"
        scope_revision = _load_revision(scope_dir / "residual-claude")
        normal_repair_plan = resumed.repair_plan
        if normal_repair_plan is None:
            raise ResumeIntegrityError("C02 repair plan is missing for scope candidate review")
        scope_plan = parse_task_plan_v2(
            (scope_dir / "planner.raw.md").read_text(encoding="utf-8"),
            implementer_ids=frozenset({selection.repair_implementer.profile_id}),
            reviewer_ids=frozenset({selection.reviewer.profile_id}),
            check_catalog=self.config.check_catalog,
            inherited_check_ids=normal_repair_plan.required_checks,
        )
        scope_records = [
            record for step in scope_plan.steps
            for record in [_load_completed_step(scope_dir / "steps" / step.id, step.id)]
            if record is not None
        ]
        self._repair_v2_step_results = scope_records
        scope_delta_text = _json_text(dict(scope_delta))
        cycle_history = _json_text({
            "C01": {
                "planner_summary": original_plan.title,
                "checks": _check_payload(cycle_1_evidence),
                "reviewer_1_conclusion": _review_payload(cycle_1_review),
            },
            "C02": {
                "repair_planner_summary": normal_repair_plan.title,
                "scope_repair_planner": "bounded scope-repair planner",
                "final_checks": _check_payload(scope_evidence),
            },
        })
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
        try:
            if start.phase is ResumePhase.REVIEWER_C02:
                _archive_attempt(run_dir / "review" / "C02", names=_REVIEW_ATTEMPT_ARTIFACTS)
            review = self._run_v2_reviewer(
                reviewer=reviewer, spec=spec, context=resumed.context,
                repository_reference=repository_reference, evidence=scope_evidence,
                input=ReviewCycleInput(
                    iteration=2,
                    plan_text=_json_text({
                        "original_approved_plan": json.loads(_compact_approved_plan_text(original_plan)),
                        "repair_plan_c02": json.loads(_compact_approved_plan_text(normal_repair_plan)),
                        "scope_delta": scope_delta_text,
                    }),
                    luna_reports=(
                        "C02 SCOPE REPAIR LUNA REPORTS\n"
                        + _review_step_reports_text(scope_records)
                    ),
                    revision_report=(
                        "C02 RESIDUAL CLAUDE\n"
                        + (_revision_report_text(scope_revision, scope_dir / "residual-claude")
                           if scope_revision is not None else "NONE")
                    ),
                    cycle_history=cycle_history,
                    scope_delta=scope_delta_text,
                ),
                artifacts_dir=run_dir / "review" / "C02",
                worktree=info.worktree, base_sha=resumed.info.base_sha,
                candidate_commit=c02_candidate,
                reuse_accepted=start.phase is ResumePhase.REVIEWER_C02,
            )
        except LLMError as exc:
            raise ReviewerTransportError(
                f"REVIEWER_TRANSPORT_FAILURE: {_bounded_parse_detail(exc)}"
            ) from exc
        store.update(status=RunStatus.REVIEWING, review=_review_payload(review), review_iterations=2)
        if review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
            return self._v2_failed(store, run_dir, "REVIEW_FAILED", None)
        self._cycle_update(store, 2, status="reviewed", reviewer_conclusion=_review_payload(review))
        store.update(status=RunStatus.APPROVED, approved_tree_sha=scope_evidence.staged_tree_sha)
        return self._complete_candidate_publication(
            store=store, run_dir=run_dir, info=info,
            approved_tree=scope_evidence.staged_tree_sha,
            commit_sha=c02_candidate["commit_sha"],
            repository_reference=repository_reference, cycle=2,
        )

    def _execute_v2_repair_cycle(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        context: str,
        repository_reference: RepositoryReference,
        info: Any,
        branch_ref: str,
        ownership_before: Any,
        selection: ExecutionSelectionV4 | ExecutionSelectionV5,
        original_plan: TaskPlanV2,
        original_bundle: Mapping[str, Any],
        cycle_1_evidence: EvidenceBundle,
        cycle_1_review: ReviewResult,
        cycle_1_revision: Any | None,
        claude_revision_enabled: bool,
        cycle_1_revision_report: str = "",
        resumed: "_ResumedRun | None" = None,
    ) -> tuple[TaskPlanV2, Any | None, EvidenceBundle, ReviewResult]:
        """Plan, execute, revise, check and review exactly one repair cycle.

        With *resumed*, every C02 phase before its checkpoint is read back
        from the durable ``repair/C02`` artifacts instead of being replayed.
        """

        self._trace_cycle = 2

        repair_dir = run_dir / "repair" / "C02"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_profile = profile_for_role(
            self.config, selection.repair_implementer.profile_id, ExecutionRole.REPAIR
        )
        reviewer_profile = profile_for_role(
            self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER
        )
        planner_profile = profile_for_role(
            self.config, selection.planner.profile_id, ExecutionRole.PLANNER
        )
        start = resumed.checkpoint if resumed is not None else None
        at = phase_index(start.phase) if start is not None else phase_index(ResumePhase.REPAIR_PLANNER)
        c01_candidate_record = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(c01_candidate_record, dict) or not _is_object_id(c01_candidate_record.get("commit_sha")):
            raise ResumeIntegrityError("C01 candidate commit is missing for C02")
        cycle_parent_sha = c01_candidate_record["commit_sha"]
        # C02 has a different immutable Git boundary from C01: its worker
        # starts on the reviewed C01 candidate commit.  Refresh the durable
        # ownership proof before the first C02 worker so a later resume can
        # compare the current branch/worktree set to the exact C02 boundary.
        ownership_before = _git_ownership(repo, info.worktree)
        store.update(
            status=store.load().get("status", RunStatus.IMPLEMENTING),
            git_ownership=_git_ownership_payload(ownership_before),
        )
        original_scope = sorted({
            path for step in original_plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        tree_before = candidate_tree_sha(info.worktree)
        if start is None and current_head(info.worktree) != cycle_parent_sha:
            raise ResumeIntegrityError("C02 does not start from the C01 candidate commit")
        if at <= phase_index(ResumePhase.REPAIR_PLANNER):
            _archive_attempt_tree(repair_dir)
            current_state = _json_text({
                "BASE_SHA": base_sha,
                "HEAD_SHA": current_head(info.worktree),
                "CANDIDATE_TREE_SHA": tree_before,
                "CHANGED_FILES": changed_paths_between_trees(repo, resolve_tree(repo, base_sha), tree_before),
                "GIT_STATUS": status_porcelain(info.worktree),
            })
            repair_planner = RepairPlannerV2(
                self._planner_client or _chat_client(
                    build_llm_endpoint(planner_profile), self._runtime_environment
                ),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                implementer_profiles=(repair_profile,),
                reviewer_profiles=(reviewer_profile,),
                planning=self.config.planning,
                check_catalog=self.config.check_catalog,
                original_required_check_ids=original_plan.required_checks,
            )
            # The C01 candidate commit is the code authority: the planner gets
            # its immutable candidate/compare URLs instead of an inline diff.
            repair_code_evidence = _review_code_evidence(
                repository_reference=repository_reference,
                base_sha=base_sha,
                candidate_sha=cycle_parent_sha,
                evidence=cycle_1_evidence,
            )
            remote_available = (
                immutable_commit_web_url(repository_reference, cycle_parent_sha)
                is not None
                and compare_commits_web_url(
                    repository_reference, base_sha, cycle_parent_sha
                )
                is not None
            )
            repair_plan_started_at = self._trace_time()
            repair_plan_started_mono = time.perf_counter()
            self._trace_emit(
                "plan.started",
                phase="planning",
                cycle=2,
                data={
                    "kind": "repair",
                    "tree_before": tree_before,
                    "session": self._trace_session(
                        profile=planner_profile,
                        selected=self._trace_selected_profile(
                            selection.planner.profile_id, ExecutionRole.PLANNER
                        ),
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=None,
                        started_at=repair_plan_started_at,
                        started_mono=repair_plan_started_mono,
                        tree_before=tree_before,
                    ),
                },
            )
            repair_plan = repair_planner.plan(
                repository_reference=render_repository_reference(repository_reference),
                original_spec=spec,
                original_plan_summary=render_repair_plan_summary(original_plan),
                original_step_index=render_repair_step_index(original_plan),
                current_repository_state=current_state,
                candidate_code_evidence=repair_code_evidence,
                final_checks_cycle_1=_json_text(
                    _repair_checks_payload(cycle_1_evidence)
                ),
                claude_revision_report_cycle_1=_bounded_repair_claude_report(
                    cycle_1_revision_report or "NONE"
                ),
                original_approved_mutable_scope=_json_text(original_scope),
                reviewer_result=_json_text(_review_payload(cycle_1_review)),
                artifacts_dir=repair_dir,
                fallback_candidate_diff=(
                    "" if remote_available else cycle_1_evidence.diff
                ),
            )
            self._trace_emit(
                "plan.completed",
                phase="planning",
                cycle=2,
                data={
                    "kind": "repair",
                    "decision": repair_plan.decision.value,
                    "title": repair_plan.title,
                    "session": self._trace_finished_model_session(
                        profile=planner_profile,
                        selected=self._trace_selected_profile(
                            selection.planner.profile_id, ExecutionRole.PLANNER
                        ),
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=(
                            (repair_dir / "planner.request.txt").stat().st_size
                            if (repair_dir / "planner.request.txt").is_file() else None
                        ),
                        started_at=repair_plan_started_at,
                        started_mono=repair_plan_started_mono,
                        usage=getattr(repair_planner, "last_usage", None),
                        tree_before=tree_before,
                        tree_after=tree_before,
                    ),
                },
            )
            self._cycle_update(
                store, 2, status="planning", kind="repair",
                plan_summary=repair_plan.title if repair_plan else "",
            )
            store.update(cycle=2, status=RunStatus.PLANNING)
            if repair_plan.decision is PlanDecision.BLOCKED:
                self._cycle_update(store, 2, status="blocked", blockers=repair_plan.blockers)
                raise OrchestrationError("REPAIR_PLANNER_BLOCKED")
            repair_bundle, repair_bundle_sha = validate_implementation_bundle(
                repair_dir, expected_step_ids=[step.id for step in repair_plan.steps]
            )
        else:
            repair_plan = resumed.repair_plan
            repair_bundle, repair_bundle_sha = resumed.repair_bundle, resumed.repair_bundle_sha
        repair_scope = sorted({
            path for step in repair_plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        scope_delta, scope_delta_content = _build_scope_delta(
            repair_dir, original_scope=original_scope, plan=repair_plan,
            candidate_commit_sha=cycle_parent_sha, review=cycle_1_review,
            repair_bundle_sha=repair_bundle_sha,
        )
        # Created once; on every later pass (resume included) the persisted
        # bytes are only compared, never repaired.
        scope_delta_sha = _ensure_scope_delta(
            repair_dir, scope_delta_content,
            expected_sha256=start.scope_delta_sha256 if start is not None else None,
        )
        for path in scope_delta["requested_write_paths"] + scope_delta["requested_delete_paths"]:
            if not path_exists_in_tree(repo, cycle_parent_sha, path):
                raise OrchestrationError("REPAIR_SCOPE_EXISTING_PATH_MISSING")
        for path in scope_delta["requested_create_paths"]:
            if path_exists_in_tree(repo, cycle_parent_sha, path):
                raise OrchestrationError("REPAIR_SCOPE_CREATE_PATH_EXISTS")
        if scope_delta["added_paths"] and not cycle_1_review.required_fixes.strip() and not cycle_1_review.findings.strip():
            raise OrchestrationError("REPAIR_SCOPE_UNJUSTIFIED")
        atomic_write_text(
            repair_dir / "scope.json",
            _json_text({
                "repair_mutable_scope": repair_scope,
                "original_approved_mutable_scope": original_scope,
                "scope_delta_sha256": scope_delta_sha,
            }),
        )
        added_paths = scope_delta["added_paths"]
        scope_policy = self._effective_repair_scope
        policy = scope_policy.policy
        bound = scope_policy.max_added_paths
        if added_paths and policy == "deny-expansion":
            self._cycle_update(store, 2, status="failed", failure="REPAIR_SCOPE_EXPANSION",
                               repair_mutable_scope=repair_scope, scope_delta=scope_delta)
            raise OrchestrationError("REPAIR_SCOPE_EXPANSION")
        requires_scope_approval = bool(added_paths) and (
            policy == "require-approval"
            or (policy == "auto-bounded" and len(added_paths) > bound)
        )
        if requires_scope_approval:
            approval = read_scope_approval(repair_dir, expected_sha256=scope_delta_sha)
            if approval is None:
                self._checkpoint(run_dir, ResumePhase.SCOPE_APPROVAL, cycle=2,
                                 head=cycle_parent_sha, tree=tree_before,
                                 repair_bundle_sha256=repair_bundle_sha,
                                 scope_delta_sha256=scope_delta_sha)
                self._cycle_update(store, 2, status="waiting_scope_approval",
                                   scope_delta=scope_delta)
                store.update(status=RunStatus.WAITING_SCOPE_APPROVAL,
                             scope_delta=scope_delta, current_step=None)
                raise ScopeApprovalRequired()
            if approval.decision is not ApprovalDecision.APPROVE:
                raise OrchestrationError("HUMAN_REQUIRED: repair scope rejected")
        elif added_paths and policy == "auto-bounded":
            self._cycle_update(store, 2, status="scope_auto_approved",
                               scope_delta=scope_delta)
        completed = list(resumed.c02_steps) if resumed is not None else []
        done_ids = {record["id"] for record in completed}
        pending_steps = [step for step in repair_plan.steps if step.id not in done_ids]
        if at <= phase_index(ResumePhase.SCOPE_APPROVAL) and pending_steps:
            # Repair planner (and any scope approval) complete: the durable
            # next operation is the first repair step, before any worker runs.
            self._checkpoint(
                run_dir, ResumePhase.REPAIR_STEP, step_id=pending_steps[0].id,
                head=cycle_parent_sha, tree=tree_before, repair_bundle_sha256=repair_bundle_sha,
                scope_delta_sha256=scope_delta_sha,
            )
        # With Claude, CHECKS_C02 and CLAUDE_C02 both precede the revision;
        # FINAL_CHECKS_C02 means Claude C02 already completed durably.
        run_claude_c02 = claude_revision_enabled and at <= phase_index(ResumePhase.CLAUDE_C02)
        final_checks_phase = (
            ResumePhase.FINAL_CHECKS_C02 if claude_revision_enabled else ResumePhase.CHECKS_C02
        )
        if pending_steps or run_claude_c02:
            # The repair plan may require trusted checks C01 did not; their
            # config-only preflights run before any expensive C02 worker and,
            # on failure, leave the checkpoint at the next worker boundary.
            check_config, check_ids = config_with_check_authority(
                self.config, run_dir, requested_check_ids=repair_plan.required_checks,
                expected_sha256=self._approved_check_authority_sha256(run_dir),
            )
            preflight_failures = run_check_preflights(
                info.worktree, check_config, check_ids or repair_plan.required_checks
            )
            if preflight_failures:
                raise OrchestrationError(preflight_failures[0])
        state_steps = [
            {"id": step.id, "title": step.title,
             "status": next(
                 (record["status"].lower() for record in completed if record["id"] == step.id),
                 "waiting",
             ),
             "profile_id": selection.repair_implementer.profile_id}
            for step in repair_plan.steps
        ]
        store.update(status=RunStatus.IMPLEMENTING, steps=state_steps, current_step=None)
        expected_tree = (
            start.expected_tree_sha
            if start is not None and start.phase is ResumePhase.REPAIR_STEP else tree_before
        )
        retry_step = start.step_id if start is not None and start.phase is ResumePhase.REPAIR_STEP else None
        # A clean mismatch (or a transient failure of its bounded retry)
        # persisted by an earlier run: this run owes that step exactly the same
        # semantic retry, with the addendum and no replay.
        pending_retries = dict(resumed.mismatch_retries) if resumed is not None else {}
        self._repair_v2_step_results = list(completed)
        codex_home = self.config.codex_runtime.home
        forbidden_env_names = (planner_profile.api_key_env, reviewer_profile.api_key_env)
        for index, step in enumerate(repair_plan.steps):
            if step.id in done_ids:
                continue
            contract = read_approved_step_contract(repair_dir, repair_bundle, step.id)
            if step.id == retry_step:
                _archive_attempt(repair_dir / "steps" / step.id)
            store.update(
                status=RunStatus.IMPLEMENTING, current_step=step.id,
                steps=[{**item, "status": "running" if item["id"] == step.id else item["status"]}
                       for item in state_steps],
            )
            # Same primitive, same gates and same failure reasons as C01; a
            # StepExecutionFailure propagates to the caller, which owns state.
            outcome = self._execute_codex_step(
                repo=repo, worktree=info.worktree, base_sha=cycle_parent_sha,
                branch_ref=branch_ref, ownership_before=ownership_before,
                expected_tree=expected_tree, step=step, contract=contract,
                profile_id=selection.repair_implementer.profile_id,
                artifact_dir=repair_dir / "steps" / step.id,
                codex_home=codex_home, forbidden_env_names=forbidden_env_names,
                future_ownership=_future_step_ownership(repair_plan.steps, index),
                pending_mismatch_retry=pending_retries.pop(step.id, None),
            )
            self._repair_v2_step_results.append(_step_result_record(outcome))
            expected_tree = outcome.tree_after
            state_steps = [
                {**item, "status": "deferred" if getattr(outcome, "status", "COMPLETED") == "DEFERRED_CONTRACT_MISMATCH" else "completed", "usage": outcome.usage,
                 "input_tokens": outcome.usage["input_tokens"],
                 "output_tokens": outcome.usage["output_tokens"]}
                if item["id"] == step.id else item
                for item in state_steps
            ]
            store.update(status=RunStatus.IMPLEMENTING, current_step=None, steps=state_steps)
            following = repair_plan.steps[index + 1].id if index + 1 < len(repair_plan.steps) else None
            self._checkpoint(
                run_dir,
                ResumePhase.REPAIR_STEP if following else ResumePhase.CHECKS_C02,
                step_id=following, head=cycle_parent_sha, tree=outcome.tree_after,
            )
        self._update_v2_usage(store, run_dir)
        deferred_mismatches = _deferred_contract_mismatches(
            repair_plan, self._repair_v2_step_results
        )
        if _has_deferred_contract_mismatches(self._repair_v2_step_results) and not claude_revision_enabled:
            raise OrchestrationError(
                "UNRESOLVED_CONTRACT_MISMATCH: HUMAN_REQUIRED: Claude revision is disabled"
            )
        if run_claude_c02:
            if start is not None and start.phase is ResumePhase.CHECKS_C02:
                _archive_attempt(run_dir / "revision" / "C02", names=_PRE_CHECK_ATTEMPT_ARTIFACTS)
            if start is not None and start.phase is ResumePhase.CLAUDE_C02:
                _archive_attempt(run_dir / "revision" / "C02", names=_REVISION_ATTEMPT_ARTIFACTS)
            cycle_2_revision, revision_error = self._run_v2_revision_cycle(
                store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                base_tree_sha=resolve_tree(repo, base_sha), spec=spec, plan=repair_plan,
                repository_reference=repository_reference, info=info,
                branch_ref=branch_ref, ownership_before=ownership_before, selection=selection,
                artifact_dir=run_dir / "revision" / "C02",
                mutable_scope=repair_scope,
                deferred_mismatches=deferred_mismatches,
                deferred_mismatch_present=(
                    _has_deferred_contract_mismatches(self._last_v2_step_results)
                    or _has_deferred_contract_mismatches(self._repair_v2_step_results)
                ),
                cycle=2,
            )
            if revision_error is not None:
                raise OrchestrationError(revision_error)
            self._cycle_update(
                store, 2, status="revised",
                repair_luna_reports=_step_reports_text(self._repair_v2_step_results),
                claude_revision_report=cycle_2_revision.final_message if cycle_2_revision else "",
            )
        else:
            cycle_2_revision = resumed.c02_revision if resumed is not None else None
        revision_report_c02 = (
            _revision_report_text(cycle_2_revision, run_dir / "revision" / "C02")
            if cycle_2_revision is not None else ""
        )
        check_repair_result_c02 = None
        check_repair_scope_c02: CheckRepairScope | None = None
        expanded_check_repair_result_c02 = None
        checks_dir = run_dir / "checks" / "C02"
        checks_dir.mkdir(parents=True, exist_ok=True)
        store.update(status=RunStatus.REVALIDATING, current_step=None)
        # C02 parity with ``FINAL_CHECKS_RETRY_C01``: its own durable boundary,
        # owned by the bridge below and not by the normal repair path.
        retry_bridge_c02 = (
            self._legacy_run_options and claude_revision_enabled and resumed is not None and start is not None
            and start.phase is ResumePhase.FINAL_CHECKS_RETRY_C02
        )
        if start is None or phase_index(start.phase) <= phase_index(ResumePhase.FINAL_CHECKS_C02):
            checks_tree = candidate_tree_sha(info.worktree)
            try:
                if start is not None and start.phase is final_checks_phase:
                    _archive_attempt(checks_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                evidence = self._final_evidence(
                    info.worktree, base_sha, checks_dir, check_failures_hard=False,
                    reuse=False, expected_head_sha=cycle_parent_sha,
                    required_check_ids=repair_plan.required_checks or None,
                    enforce_diff_size=False,
                )
            except Exception:
                self._write_phase_checkpoint(
                    run_dir, final_checks_phase, cycle=2, head=cycle_parent_sha,
                    tree=checks_tree, repair_bundle_sha256=repair_bundle_sha,
                )
                raise
            store.update(
                status=RunStatus.REVALIDATING, checks=_check_payload(evidence),
                staged_tree_sha=evidence.staged_tree_sha,
                changed_files=list(evidence.changed_files),
                deterministic_gate={"passed": evidence.deterministic_passed,
                                    "required_check_ids": list(evidence.required_check_ids),
                                    "failures": list(evidence.failures)},
            )
            integrity_failures = _hard_integrity_failures(evidence)
            if integrity_failures:
                self._write_phase_checkpoint(
                    run_dir, final_checks_phase, cycle=2,
                    head=cycle_parent_sha, tree=checks_tree,
                    repair_bundle_sha256=repair_bundle_sha,
                )
                raise OrchestrationError(integrity_failures[0].split(":", 1)[0])
        else:
            evidence = resumed.c02_evidence
            # Phase-dependent, exactly like C01: only the C02 retry checks may
            # legitimately have produced no bundle before the crash.
            if evidence is None and not retry_bridge_c02 and not (
                not self._legacy_run_options
                and start is not None
                and start.phase is ResumePhase.FINAL_CHECKS_RETRY_C02
            ):
                raise ResumeIntegrityError("C02 candidate evidence is missing")

        soft_failures_c02 = (
            _soft_check_failures(evidence)
            if claude_revision_enabled and evidence is not None else []
        )
        base_repair_scope_c02 = list(repair_scope)
        if not self._legacy_run_options and evidence is not None:
            direct_result_c02, evidence = self._run_direct_check_repair_loop(
                store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                plan=repair_plan, repository_reference=repository_reference,
                info=info, branch_ref=branch_ref,
                ownership_before=ownership_before, selection=selection,
                cycle=2, evidence=evidence, base_scope=base_repair_scope_c02,
                required_check_ids=repair_plan.required_checks or None,
                expected_head_sha=cycle_parent_sha, resumed=resumed,
            )
            check_repair_result_c02 = direct_result_c02
            self._cycle_update(
                store, 2, status="check_repair_completed",
                automatic_check_repair=self._check_repair_attempt_state(direct_result_c02),
            )
            if direct_result_c02.status == "integrity-failed":
                failures = _hard_integrity_failures(evidence)
                raise OrchestrationError(
                    (failures[0].split(":", 1)[0] if failures else "INTEGRITY_FAILED")
                    + ": " + ", ".join(failures or evidence.failures)
                )
            if direct_result_c02.status == "scope-required":
                raise OrchestrationError(_SCOPE_REQUEST_ROUTE)
            if direct_result_c02.status == "agent-failed":
                raise OrchestrationError(AGENT_RUNTIME_FAILED)
            if direct_result_c02.status == "exhausted":
                remaining = [
                    item.split(":", 1)[1]
                    for item in _soft_check_failures(evidence) if ":" in item
                ]
                profile = (
                    getattr(selection, "check_repair", None)
                    or getattr(selection, "repair_implementer", None)
                )
                raise OrchestrationError(
                    "CHECK_REPAIR_EXHAUSTED: " + _json_text({
                        "remaining_failed_check_ids": remaining,
                        "attempts": len(direct_result_c02.attempts),
                        "repair_profile_id": profile.profile_id if profile is not None else None,
                    })
                )
        if (
            self._legacy_run_options
            and
            not retry_bridge_c02
            and at < phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02)
            and evidence is not None and not evidence.deterministic_passed
            and claude_revision_enabled
        ):
            if not soft_failures_c02:
                raise OrchestrationError(
                    "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                )
            check_repair_phase = ResumePhase.CHECK_REPAIR_C02
            retry_checks_phase = ResumePhase.FINAL_CHECKS_RETRY_C02
            if at <= phase_index(check_repair_phase):
                check_repair_scope_c02 = self._resolve_check_repair_scope(
                    repo=repo, worktree=info.worktree,
                    tree_sha=evidence.staged_tree_sha, run_dir=run_dir,
                    evidence=evidence, base_mutable_scope=base_repair_scope_c02,
                )
            elif resumed is not None:
                check_repair_scope_c02 = _read_check_repair_scope(
                    run_dir / "revision" / "check-repair" / "C02",
                    fallback_base=base_repair_scope_c02,
                    policy_config=self._effective_repair_scope,
                )
            self._cycle_update(
                store, 2, status="check_repair_attempted",
                automatic_check_repair={
                    "attempted": True, "failure_ids": soft_failures_c02,
                    "before": _check_payload(evidence),
                    **(_check_repair_scope_payload(check_repair_scope_c02)
                       if check_repair_scope_c02 is not None else {}),
                },
            )
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True, "failure_ids": soft_failures_c02,
                    **(_check_repair_scope_payload(check_repair_scope_c02)
                       if check_repair_scope_c02 is not None else {}),
                },
            )
            if at <= phase_index(check_repair_phase):
                if start is not None and start.phase is check_repair_phase:
                    # Same guarantee as C01: never lose the logs of the attempt
                    # that produced the failure tree this resume restored.
                    _archive_attempt_tree(run_dir / "revision" / "check-repair" / "C02")
                check_repair_result_c02, repair_error = self._run_v2_revision_cycle(
                    store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                    base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                    plan=repair_plan,
                    repository_reference=repository_reference, info=info,
                    branch_ref=branch_ref, ownership_before=ownership_before,
                    selection=selection, artifact_dir=(
                        run_dir / "revision" / "check-repair" / "C02"
                    ), mutable_scope=list(
                        check_repair_scope_c02.effective_paths
                        if check_repair_scope_c02 is not None else repair_scope
                    ),
                    check_repair_evidence=evidence, cycle=2,
                    check_repair_scope=check_repair_scope_c02,
                )
                if repair_error is not None:
                    raise OrchestrationError(
                        f"{repair_error}: " + ", ".join(soft_failures_c02)
                    )
            elif resumed is not None:
                check_repair_result_c02 = resumed.c02_check_repair_revision
            if at <= phase_index(retry_checks_phase):
                store.update(status=RunStatus.REVALIDATING, current_step=None)
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    _archive_attempt(checks_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                    evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=cycle_parent_sha,
                        required_check_ids=repair_plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, retry_checks_phase, cycle=2,
                        head=cycle_parent_sha, tree=retry_tree,
                        repair_bundle_sha256=repair_bundle_sha,
                    )
                    raise
                store.update(
                    status=RunStatus.REVALIDATING, checks=_check_payload(evidence),
                    staged_tree_sha=evidence.staged_tree_sha,
                    changed_files=list(evidence.changed_files),
                    deterministic_gate={"passed": evidence.deterministic_passed,
                                        "required_check_ids": list(evidence.required_check_ids),
                                        "failures": list(evidence.failures)},
                    check_repair={"attempted": True, "failure_ids": soft_failures_c02,
                                  "after": list(evidence.failures)},
                )
                retry_integrity = _hard_integrity_failures(evidence)
                if retry_integrity:
                    raise OrchestrationError(
                        retry_integrity[0].split(":", 1)[0] + ": "
                        + ", ".join(retry_integrity)
                    )
                if not evidence.deterministic_passed:
                    expanded_phase = ResumePhase.CHECK_REPAIR_EXPANDED_C02
                    expanded_retry_phase = ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02
                    expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
                    expanded_scope = self._second_check_repair_scope(
                        repo=repo,
                        worktree=info.worktree,
                        tree_sha=evidence.staged_tree_sha,
                        run_dir=run_dir,
                        evidence=evidence,
                        normal_scope=self._normal_check_repair_scope(
                            check_repair_scope_c02, base_repair_scope_c02,
                            self._effective_repair_scope,
                        ),
                        expanded_dir=expanded_dir,
                    )
                    if expanded_scope is not None:
                        _archive_attempt(checks_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                        store.update(
                            status=RunStatus.REVISING,
                            check_repair={
                                "attempted": True,
                                "failure_ids": soft_failures_c02,
                                "after": list(evidence.failures),
                                **_second_check_repair_state(expanded_scope),
                            },
                        )
                        self._cycle_update(
                            store, 2, status="expanded_check_repair_attempted",
                            automatic_check_repair={
                                "attempted": True,
                                "after": _check_payload(evidence),
                                **_second_check_repair_state(expanded_scope),
                            },
                        )
                        if at <= phase_index(expanded_phase):
                            expanded_check_repair_result_c02, repair_error = self._run_v2_revision_cycle(
                                store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                                base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                                plan=repair_plan,
                                repository_reference=repository_reference, info=info,
                                branch_ref=branch_ref, ownership_before=ownership_before,
                                selection=selection, artifact_dir=expanded_dir,
                                mutable_scope=list(expanded_scope.effective_paths),
                                check_repair_evidence=evidence, cycle=2,
                                check_repair_scope=expanded_scope,
                                check_repair_phase_override=expanded_phase,
                                check_repair_next_phase_override=expanded_retry_phase,
                            )
                            if repair_error is not None:
                                raise OrchestrationError(repair_error)
                        elif resumed is not None:
                            expanded_check_repair_result_c02 = resumed.c02_expanded_check_repair_revision
                        if at <= phase_index(expanded_retry_phase):
                            retry_tree = candidate_tree_sha(info.worktree)
                            try:
                                evidence = self._final_evidence(
                                    info.worktree, base_sha, checks_dir,
                                    check_failures_hard=False, reuse=False,
                                    expected_head_sha=cycle_parent_sha,
                                    required_check_ids=repair_plan.required_checks or None,
                                    enforce_diff_size=False,
                                )
                            except Exception:
                                self._write_phase_checkpoint(
                                    run_dir, expanded_retry_phase, cycle=2,
                                    head=cycle_parent_sha, tree=retry_tree,
                                    repair_bundle_sha256=repair_bundle_sha,
                                )
                                raise
                            expanded_integrity = _hard_integrity_failures(evidence)
                            if expanded_integrity:
                                raise OrchestrationError(
                                    expanded_integrity[0].split(":", 1)[0] + ": "
                                    + ", ".join(expanded_integrity)
                                )
                        if not evidence.deterministic_passed:
                            raise OrchestrationError(
                                "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                            )
                    else:
                        raise OrchestrationError(
                            "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                        )
            if check_repair_result_c02 is not None:
                revision_report_c02 = _json_text({
                    "initial_revision": revision_report_c02,
                    "automatic_check_repair": _revision_report_text(
                        check_repair_result_c02,
                        run_dir / "revision" / "check-repair" / "C02",
                    ),
                    **({"expanded_check_repair": _revision_report_text(
                        expanded_check_repair_result_c02,
                        run_dir / "revision" / "check-repair-expanded" / "C02",
                    )} if expanded_check_repair_result_c02 is not None else {}),
                })

        if retry_bridge_c02:
            # Resume exactly at the retry checks of the normal C02 repair: the
            # C02 Luna steps, the C02 Claude revision and that repair are all
            # durable and are never replayed.
            check_repair_scope_c02 = _read_check_repair_scope(
                run_dir / "revision" / "check-repair" / "C02",
                fallback_base=base_repair_scope_c02,
                policy_config=self._effective_repair_scope,
            )
            check_repair_result_c02 = resumed.c02_check_repair_revision
            expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
            retry_evidence = _load_evidence(checks_dir)
            if retry_evidence is None and evidence is not None and (
                evidence.staged_tree_sha == start.expected_tree_sha
            ):
                retry_evidence = evidence
            if retry_evidence is not None and (
                retry_evidence.staged_tree_sha != start.expected_tree_sha
            ):
                raise ResumeIntegrityError(
                    "the C02 retry evidence is not for the checkpoint tree"
                )
            if retry_evidence is None:
                if candidate_tree_sha(info.worktree) != start.expected_tree_sha:
                    raise ResumeIntegrityError(
                        "the worktree differs from the C02 retry checkpoint tree"
                    )
                store.update(status=RunStatus.REVALIDATING, current_step=None)
                try:
                    retry_evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=cycle_parent_sha,
                        required_check_ids=repair_plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_C02, cycle=2,
                        head=cycle_parent_sha, tree=start.expected_tree_sha,
                        repair_bundle_sha256=repair_bundle_sha,
                        scope_delta_sha256=scope_delta_sha,
                    )
                    raise
            evidence = retry_evidence
            soft_failures_c02 = _soft_check_failures(evidence)
            store.update(
                status=RunStatus.REVALIDATING, checks=_check_payload(evidence),
                staged_tree_sha=evidence.staged_tree_sha,
                changed_files=list(evidence.changed_files),
                deterministic_gate={"passed": evidence.deterministic_passed,
                                    "required_check_ids": list(evidence.required_check_ids),
                                    "failures": list(evidence.failures)},
                check_repair={"attempted": True, "failure_ids": soft_failures_c02,
                              "after": list(evidence.failures)},
            )
            retry_integrity = _hard_integrity_failures(evidence)
            if retry_integrity:
                raise OrchestrationError(
                    retry_integrity[0].split(":", 1)[0] + ": "
                    + ", ".join(retry_integrity)
                )
            if not evidence.deterministic_passed:
                expanded_scope, archive_required = (
                    self._durable_second_check_repair_scope(
                        repo=repo, worktree=info.worktree, run_dir=run_dir,
                        evidence=evidence, normal_scope=check_repair_scope_c02,
                        expanded_dir=expanded_dir,
                    )
                )
                if expanded_scope is None:
                    raise OrchestrationError(
                        "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                    )
                if archive_required:
                    _archive_attempt(checks_dir, names=_CHECK_ATTEMPT_ARTIFACTS)
                store.update(
                    status=RunStatus.REVISING,
                    check_repair={
                        "attempted": True,
                        "failure_ids": soft_failures_c02,
                        "after": list(evidence.failures),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                self._cycle_update(
                    store, 2, status="expanded_check_repair_attempted",
                    automatic_check_repair={
                        "attempted": True,
                        "after": _check_payload(evidence),
                        **_second_check_repair_state(expanded_scope),
                    },
                )
                expanded_check_repair_result_c02, repair_error = self._run_v2_revision_cycle(
                    store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                    base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                    plan=repair_plan,
                    repository_reference=repository_reference, info=info,
                    branch_ref=branch_ref, ownership_before=ownership_before,
                    selection=selection, artifact_dir=expanded_dir,
                    mutable_scope=list(expanded_scope.effective_paths),
                    check_repair_evidence=evidence, cycle=2,
                    check_repair_scope=expanded_scope,
                    check_repair_phase_override=ResumePhase.CHECK_REPAIR_EXPANDED_C02,
                    check_repair_next_phase_override=(
                        ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02
                    ),
                )
                if repair_error is not None:
                    raise OrchestrationError(repair_error)
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=cycle_parent_sha,
                        required_check_ids=repair_plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02,
                        cycle=2, head=cycle_parent_sha, tree=retry_tree,
                        repair_bundle_sha256=repair_bundle_sha,
                        scope_delta_sha256=scope_delta_sha,
                    )
                    raise
                expanded_integrity = _hard_integrity_failures(evidence)
                if expanded_integrity:
                    raise OrchestrationError(
                        expanded_integrity[0].split(":", 1)[0] + ": "
                        + ", ".join(expanded_integrity)
                    )
                if not evidence.deterministic_passed:
                    raise OrchestrationError(
                        "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                    )
            revision_report_c02 = _json_text({
                "initial_revision": revision_report_c02,
                "automatic_check_repair": (
                    _revision_report_text(
                        check_repair_result_c02,
                        run_dir / "revision" / "check-repair" / "C02",
                    ) if check_repair_result_c02 is not None else ""
                ),
                **({"expanded_check_repair": _revision_report_text(
                    expanded_check_repair_result_c02, expanded_dir,
                )} if expanded_check_repair_result_c02 is not None else {}),
            })

        if self._legacy_run_options and at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02) and at <= phase_index(
            ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02
        ):
            expanded_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
            expanded_scope = _validate_expanded_check_repair_scope(
                expanded_dir,
                repo=repo,
                tree_sha=candidate_tree_sha(info.worktree),
                normal_scope=self._durable_normal_check_repair_scope(
                    run_dir, cycle=2, base_paths=base_repair_scope_c02,
                    scope=check_repair_scope_c02,
                ),
                policy_config=self._effective_repair_scope,
            )
            if at <= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02):
                expanded_check_repair_result_c02, repair_error = self._run_v2_revision_cycle(
                    store=store, run_dir=run_dir, repo=repo, base_sha=base_sha,
                    base_tree_sha=resolve_tree(repo, base_sha), spec=spec,
                    plan=repair_plan,
                    repository_reference=repository_reference, info=info,
                    branch_ref=branch_ref, ownership_before=ownership_before,
                    selection=selection, artifact_dir=expanded_dir,
                    mutable_scope=list(expanded_scope.effective_paths),
                    check_repair_evidence=evidence, cycle=2,
                    check_repair_scope=expanded_scope,
                    check_repair_phase_override=ResumePhase.CHECK_REPAIR_EXPANDED_C02,
                    check_repair_next_phase_override=ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02,
                )
                if repair_error is not None:
                    raise OrchestrationError(repair_error)
            elif resumed is not None:
                expanded_check_repair_result_c02 = resumed.c02_expanded_check_repair_revision
            if at <= phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02):
                retry_tree = candidate_tree_sha(info.worktree)
                try:
                    evidence = self._final_evidence(
                        info.worktree, base_sha, checks_dir,
                        check_failures_hard=False, reuse=False,
                        expected_head_sha=cycle_parent_sha,
                        required_check_ids=repair_plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02,
                        cycle=2, head=cycle_parent_sha, tree=retry_tree,
                        repair_bundle_sha256=repair_bundle_sha,
                    )
                    raise
                expanded_integrity = _hard_integrity_failures(evidence)
                if expanded_integrity:
                    raise OrchestrationError(
                        expanded_integrity[0].split(":", 1)[0] + ": "
                        + ", ".join(expanded_integrity)
                    )
                if not evidence.deterministic_passed:
                    raise OrchestrationError(
                        "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                    )
            revision_report_c02 = _json_text({
                "initial_revision": revision_report_c02,
                "automatic_check_repair": (
                    _revision_report_text(
                        check_repair_result_c02,
                        run_dir / "revision" / "check-repair" / "C02",
                    ) if check_repair_result_c02 is not None else ""
                ),
                "expanded_check_repair": (
                    _revision_report_text(expanded_check_repair_result_c02, expanded_dir)
                    if expanded_check_repair_result_c02 is not None else ""
                ),
            })

        c01_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(c01_candidate, dict) or not _is_object_id(c01_candidate.get("commit_sha")):
            raise ResumeIntegrityError("C01 candidate commit is missing for C02 ancestry")
        c02_candidate: dict[str, Any]
        if start is None or phase_index(start.phase) <= phase_index(ResumePhase.CANDIDATE_COMMIT_C02):
            if not evidence.deterministic_passed:
                raise OrchestrationError(
                    "DETERMINISTIC_GATE_FAILED: " + ", ".join(evidence.failures)
                )
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_COMMIT_C02, cycle=2,
                head=c01_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                repair_bundle_sha256=repair_bundle_sha,
            )
            if current_head(info.worktree) == c01_candidate["commit_sha"]:
                self._authorize_candidate_tree(
                    evidence, info.worktree, c01_candidate["commit_sha"], branch_ref
                )
            elif not (
                resumed is not None and resumed.existing_commit_sha is not None
                and commit_parents(info.worktree, resumed.existing_commit_sha) == (c01_candidate["commit_sha"],)
                and resolve_tree(info.worktree, resumed.existing_commit_sha) == evidence.staged_tree_sha
            ):
                raise ResumeIntegrityError("C02 candidate commit exists with the wrong identity")
            c02_candidate = self._ensure_candidate_commit(
                run_dir=run_dir, info=info, cycle=2, tree_sha=evidence.staged_tree_sha,
                parent_sha=c01_candidate["commit_sha"], title=repair_plan.title,
                repository_reference=repository_reference, store=store, run_id=run_id,
                commit_kind=(
                    "repair" if (
                        (check_repair_result_c02 is not None and getattr(check_repair_result_c02, "attempts", ()))
                        or expanded_check_repair_result_c02 is not None
                    ) else "candidate"
                ),
            )
        else:
            c02_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 2))
            if not isinstance(c02_candidate, dict):
                raise ResumeIntegrityError("C02 candidate commit artifact is missing")
        if start is None or phase_index(start.phase) <= phase_index(ResumePhase.CANDIDATE_PUSH_C02):
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_PUSH_C02, cycle=2,
                head=c02_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                repair_bundle_sha256=repair_bundle_sha,
                expected_parent_sha=c02_candidate["parent_sha"],
            )
            c02_candidate = self._push_candidate(
                run_dir=run_dir, info=info, cycle=2, candidate=c02_candidate, store=store,
            )
            self._cycle_update(store, 2, status="candidate_pushed")
        # C02 candidate push complete: the next operation is reviewer #2.
        self._checkpoint(run_dir, ResumePhase.REVIEWER_C02, cycle=2,
                         head=c02_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                         repair_bundle_sha256=repair_bundle_sha,
                         expected_parent_sha=c02_candidate["parent_sha"])
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
        cycle_history = _json_text({
            "C01": {
                "planner_summary": original_plan.title,
                "luna_steps_summary": _compact_step_history(self._last_v2_step_results),
                "claude_revision_report": cycle_1_revision.final_message if cycle_1_revision else "",
                "checks": _check_payload(cycle_1_evidence),
                "reviewer_1_conclusion": _review_payload(cycle_1_review),
            },
            "C02": {
                "repair_planner_summary": repair_plan.title,
                "repair_luna_reports": _compact_step_history(self._repair_v2_step_results),
                "claude_revision_report": cycle_2_revision.final_message if cycle_2_revision else "",
                "final_checks": _check_payload(evidence),
            },
        })
        scope_delta_text = _json_text(scope_delta)
        store.update(status=RunStatus.REVIEWING, current_step=None)
        # Reviewer #2 checks original SPEC <-> original approved architecture
        # <-> bounded repair <-> final cumulative diff, with every report.
        try:
            if start is not None and start.phase is ResumePhase.REVIEWER_C02:
                _archive_attempt(run_dir / "review" / "C02", names=_REVIEW_ATTEMPT_ARTIFACTS)
            review = self._run_v2_reviewer(
                reviewer=reviewer, spec=spec, context=context,
                repository_reference=repository_reference, evidence=evidence,
                input=ReviewCycleInput(
                    iteration=2,
                    plan_text=_json_text(
                        {
                            "original_approved_plan": json.loads(_compact_approved_plan_text(original_plan)),
                            "repair_plan_c02": json.loads(_compact_approved_plan_text(repair_plan)),
                            "scope_delta": scope_delta_text,
                        }
                    ),
                    luna_reports=(
                        "C01 LUNA REPORTS\n"
                        f"{_review_step_reports_text(self._last_v2_step_results)}\n\n"
                        "C02 LUNA REPAIR REPORTS\n"
                        f"{_review_step_reports_text(self._repair_v2_step_results)}"
                    ),
                    revision_report=(
                        "C01 CLAUDE REVISION\n"
                        f"{cycle_1_revision_report or 'NONE'}\n\n"
                        "C02 CLAUDE REVISION\n"
                        f"{revision_report_c02 or 'NONE'}"
                    ),
                    cycle_history=cycle_history,
                    scope_delta=scope_delta_text,
                    deferred_mismatches=(
                        "C01 DEFERRED CONTRACT MISMATCHES\n"
                        f"{_deferred_contract_mismatches(original_plan, self._last_v2_step_results)}\n"
                        "C02 DEFERRED CONTRACT MISMATCHES\n"
                        f"{deferred_mismatches}"
                    ),
                ),
                artifacts_dir=run_dir / "review" / "C02",
                worktree=info.worktree, base_sha=base_sha,
                candidate_commit=c02_candidate,
                reuse_accepted=start is not None and start.phase is ResumePhase.REVIEWER_C02,
            )
        except LLMError as exc:
            raise ReviewerTransportError(
                f"REVIEWER_TRANSPORT_FAILURE: {_bounded_parse_detail(exc)}"
            ) from exc
        store.update(status=RunStatus.REVIEWING, review=_review_payload(review), review_iterations=2)
        self._cycle_update(
            store, 2, status="reviewed", reviewer_conclusion=_review_payload(review),
            final_checks=_check_payload(evidence),
        )
        self._update_v2_usage(store, run_dir)
        return repair_plan, cycle_2_revision, evidence, review

    def _redact_step_artifacts(self, step_dir: Path) -> None:
        for name in _AGENT_ARTIFACTS:
            redact_file(step_dir / name, self._secrets)

    def _v2_agent_usage(self) -> dict[str, Any]:
        rows = getattr(self, "_v2_usage_rows", [])
        total = add_usage(rows)
        return {
            "total_input_tokens": total["input_tokens"],
            "total_output_tokens": total["output_tokens"],
            "total": total,
            "steps": [dict(row) for row in rows],
        }

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
            if hasattr(self, "_v2_usage_rows"):
                fields["agent_usage"] = self._v2_agent_usage()
            step_dir = step_dir or (run_dir / "steps" / step_id)
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

    def _authorize_v2_commit(
        self, plan: TaskPlanV2, review: ReviewResult, evidence: EvidenceBundle,
        worktree: Path, base_sha: str, branch_ref: str,
    ) -> str:
        if not evidence.deterministic_passed or evidence.failures or not evidence.staged_tree_sha:
            raise CommitBoundaryError("v2 deterministic gate did not pass")
        try:
            # Same flag as the gate payload and the reviewer parse.
            reparsed = parse_review(review.raw, deterministic_passed=evidence.deterministic_passed)
        except ReviewParseError as exc:
            raise CommitBoundaryError(f"reviewer answer does not authorize a commit: {exc}") from exc
        if review.verdict is not ReviewVerdict.PASS or reparsed.verdict is not ReviewVerdict.PASS:
            raise CommitBoundaryError("reviewer verdict is not PASS")
        if review.route is not ReviewRoute.NONE or reparsed.route is not ReviewRoute.NONE or blocking_finding_lines(review.raw):
            raise CommitBoundaryError("reviewer did not authorize the exact v2 tree")
        if symbolic_head(worktree) != branch_ref or current_head(worktree) != base_sha:
            raise CommitBoundaryError("worktree HEAD changed before v2 commit")
        approved = evidence.staged_tree_sha
        if index_tree_sha(worktree) != approved or candidate_tree_sha(worktree) != approved:
            raise CommitBoundaryError("v2 reviewed tree changed before commit")
        if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
            raise CommitBoundaryError("worktree has changes after v2 review")
        return approved

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
        cycle: int | None = None,
    ) -> RunResult:
        """Publish an already pushed candidate, only after reviewer PASS."""

        fields: dict[str, Any] = {"commit_sha": commit_sha, "current_step": None}
        if cycle is not None:
            fields["cycle"] = cycle
        chain_records = accepted_chain_records(run_dir)
        if chain_records:
            try:
                state = store.load()
                assert_deferred_verifications_resolved([
                    *(state.get("accepted_steps") or []),
                    *(state.get("deferred_verifications") or []),
                ])
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
            if store.load().get("approved_tree_sha") != approved_tree:
                raise GitError("durable approved tree differs from candidate tree")
            if current_head(info.worktree) != commit_sha:
                raise GitError("candidate commit is not the run branch tip")
            if resolve_tree(info.worktree, commit_sha) != approved_tree:
                raise GitError("candidate commit tree differs from approved tree")
            accepted_steps = store.load().get("accepted_steps") or []
            if not isinstance(accepted_steps, list):
                raise GitError("accepted step metadata is malformed")
            assert_deferred_verifications_resolved([
                *accepted_steps,
                *(store.load().get("deferred_verifications") or []),
            ])
            chain_records = accepted_chain_records(run_dir)
            if chain_records:
                validate_accepted_chain(
                    info.worktree,
                    run_dir=run_dir,
                    base_sha=info.base_sha,
                    tip_sha=commit_sha,
                    approved_tree_sha=approved_tree,
                )
                expected_parent = commit_parents(info.worktree, commit_sha)[0]
            else:
                # Compatibility for historical runs that have no arbitrary
                # chain artifact yet.
                expected_parent = _candidate_chain_parent(
                    info.worktree, run_dir, info.base_sha, cycle or 1, commit_sha,
                )
            validate_run_branch(info.branch, base_ref=self.config.base_ref)
            if self.config.publish.enabled:
                repository_remote_url(info.worktree, self.config.publish.remote)
        except (GitError, OSError, ValueError) as exc:
            state = store.record_failure("COMMIT_TREE_MISMATCH", "candidate identity is not exact", **fields)
            return RunResult(run_dir, RunStatus.FAILED, state)

        self._checkpoint(
            run_dir, ResumePhase.PUBLISH, cycle=cycle or _state_cycle_value(store.load()),
            head=commit_sha, tree=approved_tree,
            expected_parent_sha=expected_parent,
            next_step_id=None,
        )
        publish_cycle = cycle or _state_cycle_value(store.load())
        self._trace_emit(
            "publish.started",
            phase="publication",
            cycle=publish_cycle,
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
            state = store.update(status=RunStatus.COMMITTED, **fields)
            mark_checkpoint_completed(run_dir)
            self._trace_emit(
                "publish.completed",
                phase="publication",
                cycle=publish_cycle,
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
            candidate = _read_json_artifact(_candidate_commit_path(run_dir, cycle or 1))
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
        atomic_write_text(run_dir / "publish.json", _json_text(publish_payload))
        state = store.update(status=RunStatus.PUBLISHED, publish=publish_payload, **fields)
        mark_checkpoint_completed(run_dir)
        self._trace_emit(
            "publish.completed",
            phase="publication",
            cycle=publish_cycle,
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

    def _complete_commit(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: Any,
        approved_tree: str,
        commit_sha: str,
        repository_reference: RepositoryReference,
        cycle: int | None = None,
    ) -> RunResult:
        """Verify the commit tree, then publish it exactly once.

        ``run-branch`` pushes only the run branch.  ``fast-forward-base``
        compare-and-swaps the local base branch from the run base to the
        reviewed commit (no checkout of the user's worktree), then pushes that
        exact commit to the remote base branch.  Agents never worked on the
        base branch: they only ever touched the isolated run worktree.
        """

        fields: dict[str, Any] = {"commit_sha": commit_sha, "current_step": None}
        if cycle is not None:
            fields["cycle"] = cycle
        chain_records = accepted_chain_records(run_dir)
        if chain_records:
            try:
                state = store.load()
                assert_deferred_verifications_resolved([
                    *(state.get("accepted_steps") or []),
                    *(state.get("deferred_verifications") or []),
                ])
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
        if store.load().get("approved_tree_sha") != approved_tree:
            state = store.record_failure(
                "COMMIT_TREE_MISMATCH",
                "durable approved tree differs from the commit candidate",
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        try:
            committed_tree = resolve_tree(info.worktree, commit_sha)
        except GitError as exc:
            state = store.record_failure(
                "COMMIT_TREE_MISMATCH",
                "the committed object has no readable exact tree",
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        if committed_tree != approved_tree:
            state = store.record_failure(
                "COMMIT_TREE_MISMATCH",
                "HEAD tree differs from the approved reviewed tree",
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

        # The exact reviewed commit is durable: the next operation is the
        # publication, which a resume retries alone.
        self._checkpoint(
            run_dir, ResumePhase.PUBLISH, cycle=cycle or _state_cycle_value(store.load()),
            head=commit_sha, tree=approved_tree,
        )
        publish_cycle = cycle or _state_cycle_value(store.load())
        self._trace_emit(
            "publish.started",
            phase="publication",
            cycle=publish_cycle,
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
            state = store.update(status=RunStatus.COMMITTED, **fields)
            mark_checkpoint_completed(run_dir)
            self._trace_emit(
                "publish.completed",
                phase="publication",
                cycle=publish_cycle,
                data={
                    "enabled": False,
                    "status": "committed",
                    "commit_sha": commit_sha,
                    "tree_sha": approved_tree,
                },
                once=True,
            )
            return RunResult(run_dir, RunStatus.COMMITTED, state)

        # PUBLISHING is durable before the first and only push process starts.
        store.update(status=RunStatus.PUBLISHING, **fields)
        fast_forward = self.config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
        base_branch = self.config.base_ref
        outcome = None
        try:
            _archive_attempt(run_dir, names=("publish.json",))
            if status_porcelain(info.worktree):
                raise GitError("worktree is not clean before push")
            if current_head(info.worktree) != commit_sha:
                raise GitError("HEAD does not match the commit to publish")
            expected_ref = f"refs/heads/{info.branch}"
            if symbolic_head(info.worktree) != expected_ref:
                raise GitError("current branch is not the expected run branch")
            validate_run_branch(info.branch, base_ref=self.config.base_ref)
            # This is deliberately a URL existence check, not a transport
            # probe; its return value is never persisted or displayed.
            repository_remote_url(info.worktree, self.config.publish.remote)
            if resolve_tree(info.worktree, "HEAD") != approved_tree:
                raise GitError("HEAD tree differs from the approved reviewed tree")
            if fast_forward:
                outcome = publish_fast_forward_base(
                    info.source_repo,
                    remote=self.config.publish.remote,
                    base_branch=base_branch,
                    base_sha=info.base_sha,
                    commit_sha=commit_sha,
                    approved_tree=approved_tree,
                    run_branch=info.branch,
                    accepted_commits=chain_records or None,
                )
                web_url = _commit_web_url(repository_reference, commit_sha)
            else:
                web_url = run_branch_web_url(repository_reference, info.branch)
                push_run_branch(
                    info.worktree,
                    remote=self.config.publish.remote,
                    branch=info.branch,
                    commit_sha=commit_sha,
                )
        except BaseMovedError as exc:
            state = store.record_failure(
                "BASE_MOVED_SINCE_RUN",
                f"{exc}; nothing was merged, rebased, forced or pushed",
                publish={
                    "mode": self.config.publish.mode, "target": base_branch,
                    "remote": self.config.publish.remote, "commit_sha": commit_sha,
                    "status": "refused", "local_base_updated": False,
                },
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        except BasePushError as exc:
            # Git transport diagnostics can contain a credential-bearing URL:
            # the durable failure is fixed text.
            detail = "push did not complete"
            if exc.local_base_updated:
                detail += (
                    f"; local {base_branch} already points to {commit_sha}"
                    f" but {self.config.publish.remote}/{base_branch} was not updated"
                )
            state = store.record_failure(
                "PUSH_FAILED",
                detail,
                publish={
                    "mode": self.config.publish.mode, "target": base_branch,
                    "remote": self.config.publish.remote, "commit_sha": commit_sha,
                    "status": "push-failed", "local_base_updated": exc.local_base_updated,
                },
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)
        except (GitError, OSError, ValueError) as exc:
            # Git transport diagnostics can contain a credential-bearing URL.
            # Keep the durable failure bounded and secret-free.
            state = store.record_failure(
                "PUSH_FAILED",
                "push did not complete",
                **fields,
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

        if fast_forward and outcome is not None:
            publish_payload = {
                "mode": PublishMode.FAST_FORWARD_BASE.value,
                "target": base_branch,
                "remote": self.config.publish.remote,
                "branch": base_branch,
                "run_branch": info.branch,
                "base_sha": info.base_sha,
                "commit_sha": commit_sha,
                "web_url": web_url,
                "status": "pushed",
                "local_base_updated": outcome.local_base_updated,
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
                "target": info.branch,
                "remote": self.config.publish.remote,
                "branch": info.branch,
                "commit_sha": commit_sha,
                "web_url": web_url,
                "status": "pushed",
            }
        atomic_write_text(
            run_dir / "publish.json",
            _json_text(publish_payload),
        )
        state = store.update(
            status=RunStatus.PUBLISHED,
            publish=publish_payload,
            **fields,
        )
        mark_checkpoint_completed(run_dir)
        self._trace_emit(
            "candidate.pushed",
            phase="publication",
            cycle=publish_cycle,
            data={
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "remote": self.config.publish.remote,
                "branch": info.branch,
                "remote_sha": commit_sha,
            },
            once=True,
        )
        self._trace_emit(
            "publish.completed",
            phase="publication",
            cycle=publish_cycle,
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
        revalidate_integrity: bool = False,
    ) -> RunResult:
        """Dispatch resume using the run's durable pipeline version."""

        try:
            selected = _safe_run_id(run_id)
        except OrchestrationError as exc:
            raise ResumeError(str(exc)) from exc
        run_dir = (self.config.runs_root / selected).expanduser().resolve()
        if not run_dir.is_dir():
            raise ResumeError("run directory does not exist")
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            raise ResumeError("run state does not exist")
        try:
            state = RunStateStore(state_path).load()
            version = pipeline_version_from_state(state)
        except (OSError, ValueError, ResumeCheckpointError) as exc:
            raise ResumeError(f"run state is unreadable: {exc}") from exc
        if version == 1:
            return self.resume_pipeline_v1(
                selected,
                on_claimed=on_claimed,
                revalidate_integrity=revalidate_integrity,
            )
        return self.resume_pipeline_v2(
            selected,
            on_claimed=on_claimed,
            revalidate_integrity=revalidate_integrity,
        )

    def resume_pipeline_v1(
        self, run_id: str, *, on_claimed: Callable[[Path], None] | None = None,
        revalidate_integrity: bool = False,
    ) -> RunResult:
        """Resume an historical run without selecting the v2 machine."""

        self._active_pipeline_version = 1
        return self._resume_impl(
            run_id, on_claimed=on_claimed, revalidate_integrity=revalidate_integrity,
        )

    def resume_pipeline_v2(
        self, run_id: str, *, on_claimed: Callable[[Path], None] | None = None,
        revalidate_integrity: bool = False,
    ) -> RunResult:
        """Resume only a run durably created with pipeline version 2."""

        self._active_pipeline_version = 2
        return self._resume_impl(
            run_id, on_claimed=on_claimed, revalidate_integrity=revalidate_integrity,
        )

    def _resume_impl(
        self, run_id: str, *, on_claimed: Callable[[Path], None] | None = None,
        revalidate_integrity: bool = False,
    ) -> RunResult:
        """Resume a failed run at its durable checkpoint.

        Never replays a successful phase.  Every persisted invariant is
        validated first; a mismatch records ``RESUME_INTEGRITY_FAILURE`` (or
        ``RESUME_REQUIRES_OPERATOR``) without any model call.  A run that is
        not resumable raises :class:`ResumeNotAllowedError` and its state is
        left untouched.

        *revalidate_integrity* is the explicit operator intent behind
        ``metaharness resume --revalidate-integrity``.  It only waives the
        cheap classification of ``RESUME_INTEGRITY_FAILURE`` as permanently
        non-resumable, so that the *complete* validation below runs again on a
        run a since-fixed validation defect had refused.  Every invariant is
        still enforced fail-closed: when the validation refuses a second time
        the run stays ``RESUME_INTEGRITY_FAILURE``, nothing is written to Git
        and no model or check runs.
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
        except (OSError, ValueError) as exc:
            raise ResumeError(f"run state is unreadable: {exc}") from exc
        try:
            durable_pipeline_version = pipeline_version_from_state(state)
        except ResumeCheckpointError as exc:
            raise ResumeError(str(exc)) from exc
        if durable_pipeline_version != getattr(self, "_active_pipeline_version", 2):
            raise ResumeError("resume pipeline dispatch does not match durable pipeline_version")
        try:
            self._legacy_run_options = not bool(state.get("run_options_explicit", False))
            options, _, raw_run_options = legacy_or_durable_run_options_with_raw(
                self.config,
                run_dir,
                expected_sha256=state.get("run_options_sha256")
                if isinstance(state.get("run_options_sha256"), str) else None,
            )
            override = read_repair_scope_override(run_dir)
            self._run_options = options
            self._effective_repair_scope = effective_repair_scope_policy(
                options, raw_run_options=raw_run_options, override=override
            )
            self.config = effective_run_config(self.config, options)
        except RunOptionsError as exc:
            raise ResumeNotAllowedError("run options are missing or invalid") from exc
        if not hasattr(self, "_runtime_environment"):
            self._runtime_environment = (
                self.config.runtime_environment if self.config.runtime_environment else os.environ
            )
        self._secrets = config_secret_values(self.config, self._runtime_environment)
        self._begin_trace(
            run_dir, selected, created=False,
            pipeline_version=durable_pipeline_version,
        )
        eligibility = resume_info(
            run_dir, state, revalidate_integrity=revalidate_integrity
        )
        if not eligibility.resumable:
            raise ResumeNotAllowedError(eligibility.reason or "run is not resumable")
        try:
            checkpoint = load_resume_checkpoint(run_dir, state)
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
            **({"revalidate_integrity": True} if revalidate_integrity else {}),
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
            resumed = self._validate_resume(run_dir, state, checkpoint)
        except (ResumeIntegrityError, ResumeRequiresOperatorError) as exc:
            failed = store.record_failure(
                exc.code, redact(str(exc), self._secrets),
                resume={**record, "status": "refused"}, current_step=None,
            )
            return self._diagnose_result(RunResult(run_dir, RunStatus.FAILED, failed))
        # Validation may reconcile a historical clean mismatch and advance
        # the durable boundary to the following Luna step.
        checkpoint = resumed.checkpoint
        claimed = store.transition_if(
            state.get("status", RunStatus.FAILED), state.get("updated_at"),
            status=PHASE_STATUS[checkpoint.phase], failure=None, current_step=None,
            resume={**record, "status": "running",
                    "restored_paths": list(resumed.restore_paths)},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if read_checkpoint(run_dir) is None:
            # A historical run: persist the inferred checkpoint so every later
            # transition carries the same identity forward.
            write_checkpoint(run_dir, checkpoint)
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            if resumed.restore_paths:
                self._restore_revision_tree(run_dir, resumed)
            if checkpoint.phase is ResumePhase.PUBLISH:
                candidate_path = _candidate_commit_path(run_dir, checkpoint.cycle)
                if _read_json_artifact(candidate_path) is not None:
                    return self._diagnose_result(self._complete_candidate_publication(
                        store=store, run_dir=run_dir, info=resumed.info,
                        approved_tree=checkpoint.expected_tree_sha,
                        commit_sha=checkpoint.expected_head_sha,
                        repository_reference=resumed.repository_reference,
                        cycle=checkpoint.cycle,
                    ))
                return self._diagnose_result(self._complete_commit(
                    store=store, run_dir=run_dir, info=resumed.info,
                    approved_tree=checkpoint.expected_tree_sha,
                    commit_sha=checkpoint.expected_head_sha,
                    repository_reference=resumed.repository_reference,
                    cycle=checkpoint.cycle,
                ))
            if checkpoint.phase is ResumePhase.COMMIT:
                commit_sha = resumed.existing_commit_sha
                if commit_sha is None:
                    plan_for_commit = resumed.repair_plan if checkpoint.cycle == 2 and resumed.repair_plan else resumed.plan
                    commit_fn = commit_reviewed_tree
                    commit_sha = commit_fn(
                        resumed.info.worktree,
                        tree_sha=checkpoint.expected_tree_sha,
                        parent_sha=resumed.info.base_sha,
                        subject=_commit_subject(plan_for_commit.title),
                        body=f"MetaHarness-Run: {selected}",
                    )
                return self._diagnose_result(self._complete_commit(
                    store=store, run_dir=run_dir, info=resumed.info,
                    approved_tree=checkpoint.expected_tree_sha,
                    commit_sha=commit_sha,
                    repository_reference=resumed.repository_reference,
                    cycle=checkpoint.cycle,
                ))
            return self._diagnose_result(self._execute_v2(
                store, run_dir, selected, resumed.spec, resumed.info.source_repo,
                resumed.info.base_sha, resumed.context, resumed.repository_reference,
                resumed=resumed,
            ))
        except ResumeRequiresOperatorError as exc:
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

    # -- operator plan recovery ----------------------------------------------

    def recover_plan(
        self, run_id: str, replacement_raw: str, *,
        on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Replace a failed planner answer with an operator META PLAN v2.

        No model is called.  The replacement is validated exactly like a
        planner answer, published as the run's plan authority, and the
        checkpoint moves to PLAN_APPROVAL.  The run then continues through the
        normal resume workflow: plan approval, worktree setup, Luna S01...
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
            options, _ = legacy_or_durable_run_options(
                self.config, run_dir,
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
            # expects.  The whole trusted catalogue is frozen, not just the C01
            # selection, so a C02 repair plan still runs approved argv.
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
            self._write_phase_checkpoint(
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
                self._write_phase_checkpoint(run_dir, ResumePhase.PLANNER,
                                             head=base_sha, tree=base_tree)
                checkpoint = ResumeCheckpoint(
                    ResumePhase.PLANNER, 1, None, base_sha, base_tree, None, None
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
                        if self._run_options.schema_version >= 2 and (
                            self._run_options.semantic_revision_enabled
                            or self._run_options.max_review_repair_cycles > 0
                            or self._run_options.max_check_repair_attempts > 0
                        ):
                            _selection, selection_sha = read_execution_selection_v5_with_sha256(run_dir)
                            validate_execution_selection_v5(self.config, _selection)
                        elif self._run_options.claude_revision_enabled or self._run_options.repair_cycles == 1:
                            _selection, selection_sha = read_execution_selection_v4_with_sha256(run_dir)
                            validate_execution_selection_v4(self.config, _selection)
                        else:
                            _selection, selection_sha = read_execution_selection_v3_with_sha256(run_dir)
                            validate_execution_selection_v3(self.config, _selection)
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

        expected_branch = f"harness/{_slug(plan.title)}/{run_id}"
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

    def _restore_revision_tree(self, run_dir: Path, resumed: "_ResumedRun") -> None:
        """Undo a failed attempt's in-scope edits, exactly and boundedly."""

        worktree = resumed.info.worktree
        expected = resumed.checkpoint.expected_tree_sha
        try:
            restore_paths_from_tree(worktree, expected, resumed.restore_paths)
            restored = (
                index_tree_sha(worktree) == expected
                and candidate_tree_sha(worktree) == expected
                and not _status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except GitError:
            restored = False
        if not restored:
            raise ResumeRequiresOperatorError(
                "the tree recorded in tree_before.txt could not be restored exactly"
            )
        if resumed.mismatch_recovery is not None:
            self._write_mismatch_recovery_artifact(run_dir, resumed)

    @staticmethod
    def _write_mismatch_recovery_artifact(run_dir: Path, resumed: "_ResumedRun") -> None:
        """Persist the exact, idempotent proof of a dirty mismatch rollback."""

        payload = resumed.mismatch_recovery
        if payload is None:
            return
        step_id = resumed.checkpoint.step_id
        if resumed.mismatch_recovery_path is None and not isinstance(step_id, str):
            raise ResumeIntegrityError("dirty mismatch recovery has no step id")
        path = resumed.mismatch_recovery_path or (
            Orchestrator._step_root(run_dir, resumed.checkpoint)
            / "steps" / step_id / "mismatch_recovery.json"
        )
        expected = _json_text(payload)
        try:
            if path.exists():
                if path.read_text(encoding="utf-8") != expected:
                    raise ResumeIntegrityError(
                        "dirty mismatch recovery artifact is divergent"
                    )
                return
            atomic_write_text(path, expected)
        except (OSError, UnicodeError, ResultArtifactError) as exc:
            raise ResumeIntegrityError(
                "dirty mismatch recovery artifact could not be written"
            ) from exc

    def _validate_resume(
        self, run_dir: Path, state: Mapping[str, Any], checkpoint: ResumeCheckpoint,
    ) -> "_ResumedRun":
        """Fail-closed integrity gate in front of every resume; no model call."""

        def refuse(message: str) -> NoReturn:
            raise ResumeIntegrityError(message)

        original_checkpoint = checkpoint
        revision_enabled = self._run_options.semantic_revision_enabled
        repair_enabled = self._run_options.max_review_repair_cycles > 0
        check_repair_enabled = self._run_options.max_check_repair_attempts > 0
        durable_pipeline_version = pipeline_version_from_state(state)
        if durable_pipeline_version != getattr(self, "_active_pipeline_version", 2):
            refuse("resume pipeline dispatch does not match durable pipeline_version")
        if durable_pipeline_version == 2 and (
            self.config.planning.protocol != "v2" or state.get("planning_protocol") != "v2"
        ):
            refuse("only META PLAN v2 runs can be resumed by pipeline v2")
        if durable_pipeline_version == 1 and state.get("planning_protocol") != "v2":
            refuse("historical pipeline v1 requires its historical META PLAN artifacts")
        if not revision_enabled and not repair_enabled and not check_repair_enabled and checkpoint.phase not in (
            ResumePhase.INITIAL_STEP, ResumePhase.CHECKS_C01,
            ResumePhase.FINAL_CHECKS_C01, ResumePhase.CANDIDATE_COMMIT_C01,
            ResumePhase.CANDIDATE_PUSH_C01, ResumePhase.REVIEWER_C01,
            ResumePhase.COMMIT, ResumePhase.PUBLISH,
        ):
            refuse("this checkpoint requires revision.enabled")
        try:
            repo = git_root(self.config.repo)
        except GitError as exc:
            refuse(f"repository is unavailable: {exc}")
        if str(repo) != str(state.get("repo")):
            refuse("the configured repository is not the run repository")
        base_sha = state.get("base_sha")
        if not isinstance(base_sha, str) or _GIT_OBJECT_ID.fullmatch(base_sha) is None:
            refuse("run base SHA is invalid")
        reference = _read_repository_reference(run_dir)
        if reference is None or reference.base_sha != base_sha:
            refuse("the base SHA changed for this run")
        try:
            spec = (run_dir / "spec.md").read_text(encoding="utf-8")
            context = (run_dir / "context.txt").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            refuse("run SPEC or planner context is unreadable")
        # Plan identity and the human approval bound to it.
        try:
            identity = compute_plan_identity_from_run(run_dir)
            recorded = plan_identity_from_mapping(state.get("plan_identity"))
            authority = read_check_authority(
                run_dir,
                expected_sha256=identity.checks_sha256,
                trusted_check_ids=tuple(check.id for check in self.config.trusted_checks()),
            )
            approval = read_plan_approval(run_dir, expected_identity=identity)
        except (ApprovalError, ResumeCheckpointError) as exc:
            refuse(f"plan artifacts are invalid: {exc}")
        if checkpoint.plan_identity is not None and checkpoint.plan_identity.checks_sha256 is not None and authority is None:
            refuse("check authority is missing for this run")
        if identity != checkpoint.plan_identity or identity != recorded:
            refuse("plan identity no longer matches")
        if self.config.approval.require_plan_approval:
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                refuse("plan approval was not APPROVE")
            if (
                approval.execution_sha256 != checkpoint.execution_selection_sha256
                or approval.bundle_sha256 != identity.bundle_sha256
            ):
                refuse("the approval does not bind the checkpoint execution selection")
        elif approval is not None and approval.decision is not ApprovalDecision.APPROVE:
            refuse("plan approval artifact is not APPROVE")
        # The approved execution selection, against today's configuration.
        try:
            if self._run_options.schema_version >= 2 and (
                revision_enabled or repair_enabled or check_repair_enabled
            ):
                selection, execution_sha = read_execution_selection_v5_with_sha256(run_dir)
                validate_execution_selection_v5(self.config, selection)
            elif revision_enabled or repair_enabled:
                selection, execution_sha = read_execution_selection_v4_with_sha256(run_dir)
                validate_execution_selection_v4(self.config, selection)
            else:
                selection, execution_sha = read_execution_selection_v3_with_sha256(run_dir)
                validate_execution_selection_v3(self.config, selection)
                if selection.reviser is not None:
                    refuse("execution selection contains a reviser but revision is disabled")
        except (ExecutionSelectionError, ProfileError) as exc:
            refuse(f"execution selection is invalid: {exc}")
        if execution_sha != checkpoint.execution_selection_sha256 or execution_sha != identity.execution_sha256:
            refuse("execution selection hash changed")
        execution = state.get("execution") if isinstance(state.get("execution"), Mapping) else {}
        planner_state = execution.get("planner") if isinstance(execution.get("planner"), Mapping) else {}
        if selection.planner.profile_id != planner_state.get("profile_id"):
            refuse("execution selection planner is not the run planner")
        # The approved plan and its exact bundle.
        profiles = profiles_for_config(self.config).values()
        try:
            plan = parse_task_plan_v2(
                (run_dir / "planner.raw.md").read_text(encoding="utf-8"),
                implementer_ids=frozenset(p.id for p in profiles if ExecutionRole.IMPLEMENTER in p.roles),
                reviewer_ids=frozenset(p.id for p in profiles if ExecutionRole.REVIEWER in p.roles),
                check_catalog=self.config.check_catalog,
                default_check_ids=self.config.default_check_ids,
            )
            bundle, bundle_sha = validate_implementation_bundle(
                run_dir, expected_step_ids=[step.id for step in plan.steps]
            )
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            refuse(f"approved plan is unreadable: {exc}")
        if plan.decision is not PlanDecision.READY or bundle_sha != identity.bundle_sha256:
            refuse("approved bundle changed")
        if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
            refuse("execution selection steps do not match the plan")
        # Worktree, branch, HEAD and the exact candidate tree.
        worktree_value, branch = state.get("worktree"), state.get("branch")
        if not isinstance(worktree_value, str) or not isinstance(branch, str):
            refuse("run has no worktree or branch")
        worktree = Path(worktree_value).expanduser().resolve()
        if not worktree.is_dir():
            refuse("run worktree is missing")
        try:
            frozen_check_config, frozen_check_ids = config_with_check_authority(
                self.config, run_dir,
                expected_sha256=checkpoint.plan_identity.checks_sha256
                if checkpoint.plan_identity is not None else None,
            )
            if frozen_check_ids is not None:
                for frozen_check in frozen_check_config.select_checks(frozen_check_ids):
                    resolve_check_cwd(worktree, frozen_check)
        except (ValidationError, ValueError) as exc:
            refuse(f"check authority is invalid: {exc}")
        # The approved authority is strictly cumulative:
        #     original C01 plan scope
        #     UNION durable valid C01 expanded check-repair scope
        #     UNION validated C02 repair scope
        #     UNION durable valid C02 expanded check-repair scope
        # Each term is still validated exactly as before; only the
        # composition changed, so a path outside all of them stays a
        # fail-closed integrity failure below.
        original_plan_scope = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        scope = list(original_plan_scope)
        expanded_c01_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
        if _expanded_scope_is_applicable(
            checkpoint_phase=checkpoint.phase,
            expansion_phase=ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            scope_path=expanded_c01_dir / "scope.json",
        ):
            # The expansion's base is the C01 check-repair scope, exactly as
            # the run recorded it; the added test paths must still exist in
            # the checkpoint tree, which always carries the C01 candidate.
            normal_scope_c01 = _read_check_repair_scope(
                run_dir / "revision" / "check-repair" / "C01",
                fallback_base=original_plan_scope,
                policy_config=self._effective_repair_scope,
            )
            expanded_c01 = _validate_expanded_check_repair_scope(
                expanded_c01_dir,
                repo=repo, tree_sha=checkpoint.expected_tree_sha,
                normal_scope=normal_scope_c01,
                policy_config=self._effective_repair_scope,
            )
            scope = sorted(set(scope) | set(expanded_c01.effective_paths))
        c02_repair_scope: list[str] = []
        c02_scope_recovery_phases = {
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        }
        scope_recovery_artifact_exists = any(
            (run_dir / "revision" / directory / f"C0{checkpoint.cycle}" / "scope_violation_recovery.json").is_file()
            for directory in ("check-repair", "check-repair-expanded")
        )
        if checkpoint.cycle == 2 and not scope_recovery_artifact_exists and checkpoint.phase not in {
            ResumePhase.REPAIR_PLANNER, ResumePhase.SCOPE_APPROVAL,
            *c02_scope_recovery_phases,
        }:
            # The C02 scope delta is bound to the original plan scope, never
            # to an expansion, so it is validated against that exact base.
            c02_repair_scope = self._validated_repair_scope(
                run_dir, checkpoint, plan, selection, original_plan_scope
            )
            scope = sorted(set(scope) | set(c02_repair_scope))
        expanded_c02_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
        if _expanded_scope_is_applicable(
            checkpoint_phase=checkpoint.phase,
            expansion_phase=ResumePhase.CHECK_REPAIR_EXPANDED_C02,
            scope_path=expanded_c02_dir / "scope.json",
        ):
            # Same-scope parity with C01: the second pass is validated against
            # the exact scope the C02 check repair recorded for itself.
            normal_scope_c02 = _read_check_repair_scope(
                run_dir / "revision" / "check-repair" / "C02",
                fallback_base=c02_repair_scope,
                policy_config=self._effective_repair_scope,
            )
            expanded_c02 = _validate_expanded_check_repair_scope(
                expanded_c02_dir,
                repo=repo, tree_sha=checkpoint.expected_tree_sha,
                normal_scope=normal_scope_c02,
                policy_config=self._effective_repair_scope,
            )
            scope = sorted(set(scope) | set(expanded_c02.effective_paths))
        if (
            checkpoint.check_repair_attempt is not None
            or self._check_repair_attempt_records(run_dir, 1)
        ):
            scope = sorted(set(scope) | set(self._check_repair_scope_union(
                run_dir, 1, original_plan_scope,
            )))
        if checkpoint.cycle == 2 and (
            checkpoint.check_repair_attempt is not None
            or self._check_repair_attempt_records(run_dir, 2)
        ):
            scope = sorted(set(scope) | set(self._check_repair_scope_union(
                run_dir, 2, c02_repair_scope or original_plan_scope,
            )))
        scope_repair_phases = {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02, ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        }
        if checkpoint.phase in scope_repair_phases - {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
        }:
            scope = sorted(set(scope) | set(self._validated_scope_repair_scope(
                run_dir, checkpoint, plan, selection
            )))
        if checkpoint.cycle == 2 and (
            checkpoint.phase in c02_scope_recovery_phases or scope_recovery_artifact_exists
        ):
            try:
                c02_plan = parse_task_plan_v2(
                    (run_dir / "repair" / "C02" / "planner.raw.md").read_text(encoding="utf-8"),
                    implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                    reviewer_ids=frozenset({selection.reviewer.profile_id}),
                    check_catalog=self.config.check_catalog,
                    inherited_check_ids=plan.required_checks,
                )
                _bundle, _sha = validate_implementation_bundle(
                    run_dir / "repair" / "C02",
                    expected_step_ids=[step.id for step in c02_plan.steps],
                )
                c02_writes, c02_creates, c02_deletes = _repair_mutation_sets(c02_plan)
                scope = sorted(set(scope) | set(c02_writes) | set(c02_creates) | set(c02_deletes))
            except (OSError, UnicodeError, V2PlanParseError, ValueError, AttributeError) as exc:
                refuse(f"the prior C02 repair scope is unreadable: {exc}")
        scope_repair_dir = run_dir / "scope-repair" / f"C0{checkpoint.cycle}"
        scope_repair_waiting = {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
        }
        if (
            checkpoint.phase not in scope_repair_waiting
            and (scope_repair_dir / "scope_delta.json").is_file()
            and checkpoint.repair_bundle_sha256 is not None
            and checkpoint.scope_delta_sha256 is not None
        ):
            scope = sorted(set(scope) | set(self._validated_scope_repair_scope(
                run_dir, checkpoint, plan, selection
            )))
        failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
        clean_mismatch_shape = (
            failure.get("reason") == "AGENT_CONTRACT_MISMATCH"
            and is_clean_contract_mismatch_artifact(run_dir, state, checkpoint)
        )
        dirty_mismatch_shape = (
            failure.get("reason") == "AGENT_CONTRACT_MISMATCH"
            and is_recoverable_dirty_contract_mismatch_artifact(run_dir, state, checkpoint)
        )
        try:
            if str(worktree) not in registered_worktrees(repo):
                refuse("run worktree is not registered in the repository")
            if not branch_exists(repo, branch):
                refuse("run branch is missing")
            if symbolic_head(worktree) != f"refs/heads/{branch}":
                refuse("worktree HEAD is not the run branch")
            head = current_head(worktree)
            # The checkpoint is an authority record, not a hint for replay.
            # Once an accepted boundary was persisted, any other HEAD/tree is
            # an integrity failure.  In particular, do not let a model or a
            # best-effort reconciliation choose a replacement commit.
            if checkpoint.expected_head_sha is not None and head != checkpoint.expected_head_sha:
                refuse("HEAD moved since the checkpoint")
            immutable_tree_phases = {
                ResumePhase.CHECKS_C01, ResumePhase.FINAL_CHECKS_C01,
                ResumePhase.FINAL_CHECKS_RETRY_C01,
                ResumePhase.CHECKS_C02, ResumePhase.FINAL_CHECKS_C02,
                ResumePhase.FINAL_CHECKS_RETRY_C02,
                ResumePhase.CANDIDATE_COMMIT_C01, ResumePhase.CANDIDATE_PUSH_C01,
                ResumePhase.REVIEWER_C01, ResumePhase.CANDIDATE_COMMIT_C02,
                ResumePhase.CANDIDATE_PUSH_C02, ResumePhase.REVIEWER_C02,
                ResumePhase.COMMIT, ResumePhase.PUBLISH,
            }
            if checkpoint.expected_tree_sha is not None and checkpoint.phase in immutable_tree_phases:
                try:
                    if candidate_tree_sha(worktree) != checkpoint.expected_tree_sha:
                        refuse("candidate tree differs from the checkpoint tree")
                except GitError as exc:
                    refuse(f"checkpoint candidate tree is unreadable: {exc}")
            # ``base_sha`` is the sentinel parent recorded before the first
            # accepted commit.  It is not the actual parent of the BASE
            # commit; compare a parent only after HEAD has advanced beyond
            # BASE.
            if (
                checkpoint.expected_parent_sha is not None
                and checkpoint.expected_head_sha != base_sha
            ):
                try:
                    if commit_parents(repo, head) != (checkpoint.expected_parent_sha,):
                        refuse("HEAD parent differs from the checkpoint parent")
                except GitError as exc:
                    refuse(f"checkpoint HEAD parent is unreadable: {exc}")
            existing_commit: str | None = None
            c02_phases = {
                ResumePhase.REPAIR_PLANNER, ResumePhase.SCOPE_APPROVAL, ResumePhase.REPAIR_STEP,
                ResumePhase.CHECKS_C02, ResumePhase.CLAUDE_C02,
                ResumePhase.FINAL_CHECKS_C02, ResumePhase.CHECK_REPAIR_C02,
                ResumePhase.FINAL_CHECKS_RETRY_C02,
                ResumePhase.CHECK_REPAIR_EXPANDED_C02,
                ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02,
                ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
                ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02,
                ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
                ResumePhase.CANDIDATE_COMMIT_C02,
                ResumePhase.CANDIDATE_PUSH_C02, ResumePhase.REVIEWER_C02,
            }
            phase_expected_head = checkpoint.expected_head_sha or base_sha
            if checkpoint.phase in c02_phases:
                prior = _read_json_artifact(_candidate_commit_path(run_dir, 1))
                if not isinstance(prior, dict) or not _is_object_id(prior.get("commit_sha")):
                    refuse("C01 candidate commit is missing for C02 resume")
                if checkpoint.expected_head_sha is None:
                    phase_expected_head = prior["commit_sha"]
            candidate_phases = {
                ResumePhase.CANDIDATE_COMMIT_C01, ResumePhase.CANDIDATE_PUSH_C01,
                ResumePhase.REVIEWER_C01, ResumePhase.CANDIDATE_COMMIT_C02,
                ResumePhase.CANDIDATE_PUSH_C02, ResumePhase.REVIEWER_C02,
            }
            if checkpoint.phase in candidate_phases:
                cycle = 2 if checkpoint.phase in {
                    ResumePhase.CANDIDATE_COMMIT_C02, ResumePhase.CANDIDATE_PUSH_C02,
                    ResumePhase.REVIEWER_C02,
                } else 1
                candidate_payload = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
                if not isinstance(candidate_payload, dict):
                    # A crash before artifact persistence can still be
                    # reconciled from the exact commit object below.
                    candidate_payload = {}
                expected_parent = (
                    candidate_payload.get("parent_sha")
                    if _is_object_id(candidate_payload.get("parent_sha"))
                    else checkpoint.expected_parent_sha or base_sha
                )
                if cycle == 2 and not _is_object_id(candidate_payload.get("parent_sha")):
                    prior = _read_json_artifact(_candidate_commit_path(run_dir, 1))
                    if not isinstance(prior, dict) or not _is_object_id(prior.get("commit_sha")):
                        refuse("C01 candidate commit is missing for C02 ancestry")
                    expected_parent = prior["commit_sha"]
                expected_tree = checkpoint.expected_tree_sha
                recorded_commit = candidate_payload.get("commit_sha")
                if _is_object_id(recorded_commit):
                    if recorded_commit != head and checkpoint.phase is not ResumePhase.CANDIDATE_COMMIT_C01 and checkpoint.phase is not ResumePhase.CANDIDATE_COMMIT_C02:
                        refuse("run branch does not point to the recorded candidate commit")
                    candidate_commit = recorded_commit
                elif checkpoint.phase in {ResumePhase.CANDIDATE_COMMIT_C01, ResumePhase.CANDIDATE_COMMIT_C02} and head != checkpoint.expected_head_sha:
                    candidate_commit = head
                else:
                    candidate_commit = None
                if candidate_commit is not None:
                    try:
                        if commit_parents(repo, candidate_commit) != (expected_parent,) or resolve_tree(repo, candidate_commit) != expected_tree:
                            refuse("candidate commit is not the exact parent/tree identity")
                    except GitError as exc:
                        refuse(f"candidate commit is unreadable: {exc}")
                    existing_commit = candidate_commit
                elif checkpoint.phase not in {ResumePhase.CANDIDATE_COMMIT_C01, ResumePhase.CANDIDATE_COMMIT_C02}:
                    refuse("candidate commit is missing")
            elif checkpoint.phase is ResumePhase.COMMIT and head != checkpoint.expected_head_sha:
                # A crash can occur after commit-tree/update-ref and before
                # state.json.  Reuse only the single exact commit object.
                try:
                    if (
                        commit_parents(repo, head) == (base_sha,)
                        and resolve_tree(repo, head) == checkpoint.expected_tree_sha
                    ):
                        existing_commit = head
                    else:
                        refuse("commit exists but is not the exact approved commit")
                except GitError as exc:
                    refuse(f"existing commit is unreadable: {exc}")
            elif head != checkpoint.expected_head_sha:
                refuse("HEAD moved since the checkpoint")
            if (
                checkpoint.phase is ResumePhase.REPAIR_PLANNER
                and resolve_tree(repo, head) != checkpoint.expected_tree_sha
            ):
                refuse("the C01 candidate commit is not the reviewed C01 tree")
            if resolve_commit(repo, f"refs/heads/{branch}") != head:
                refuse("run branch does not point to the worktree HEAD")
            if checkpoint.phase is ResumePhase.PUBLISH:
                candidate_path = _candidate_commit_path(run_dir, checkpoint.cycle)
                candidate_record = _read_json_artifact(candidate_path)
                expected_parent = (
                    candidate_record.get("parent_sha")
                    if isinstance(candidate_record, dict)
                    and _is_object_id(candidate_record.get("parent_sha"))
                    else checkpoint.expected_parent_sha or base_sha
                )
                if checkpoint.cycle == 2 and not (
                    isinstance(candidate_record, dict)
                    and _is_object_id(candidate_record.get("parent_sha"))
                ):
                    prior = _read_json_artifact(_candidate_commit_path(run_dir, 1))
                    if not isinstance(prior, dict) or not _is_object_id(prior.get("commit_sha")):
                        refuse("C01 candidate commit is missing for C02 publication")
                    expected_parent = prior["commit_sha"]
                if head != state.get("commit_sha") or commit_parents(repo, head) != (expected_parent,):
                    refuse("the recorded commit is not the exact candidate commit")
                if (
                    resolve_tree(repo, head) != checkpoint.expected_tree_sha
                    or checkpoint.expected_tree_sha != state.get("approved_tree_sha")
                ):
                    refuse("the run commit tree is not the approved tree")
                if self.config.publish.enabled and remote_run_branch_tip(
                    repo, remote=self.config.publish.remote, branch=branch
                ) != head:
                    refuse("the candidate run branch is not pushed")
            elif checkpoint.phase not in {ResumePhase.COMMIT, *candidate_phases} and head != phase_expected_head:
                refuse("an agent-created commit moved the run branch")
            base_tree = resolve_tree(repo, base_sha)
            candidate = candidate_tree_sha(worktree)
            index_tree = index_tree_sha(worktree)
            dirty = _status_has_unstaged_or_untracked(status_porcelain(worktree))
            if checkpoint.phase in {
                ResumePhase.CANDIDATE_PUSH_C01, ResumePhase.REVIEWER_C01,
                ResumePhase.CANDIDATE_PUSH_C02, ResumePhase.REVIEWER_C02,
            } and self.config.publish.enabled:
                remote_tip = remote_run_branch_tip(
                    repo, remote=self.config.publish.remote, branch=branch
                )
                if checkpoint.phase in {ResumePhase.REVIEWER_C01, ResumePhase.REVIEWER_C02} and remote_tip != head:
                    refuse("remote run branch does not point to the candidate commit")
                if checkpoint.phase in {ResumePhase.CANDIDATE_PUSH_C01, ResumePhase.CANDIDATE_PUSH_C02} and remote_tip not in {None, head}:
                    refuse("remote run branch points to a different commit")
            # A crash can happen after a worker's durable step record was
            # written and before the checkpoint named the following step.
            # That exact shape is reconciled here, before any drift verdict,
            # so the step is never executed twice.
            reconciled_retries: dict[str, str] = {}
            if checkpoint.phase in {
                ResumePhase.INITIAL_STEP, ResumePhase.REPAIR_STEP,
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
            }:
                step_ids = (
                    [step.id for step in plan.steps]
                    if checkpoint.phase is ResumePhase.INITIAL_STEP
                    else self._repair_step_ids(run_dir, checkpoint, selection, plan.required_checks)
                )
                reconciled = self._reconcile_durable_step(
                    run_dir, checkpoint, step_ids or [], repo=repo, worktree=worktree,
                    candidate=candidate, index_tree=index_tree, dirty=dirty, scope=scope,
                    expected_head=phase_expected_head, branch_ref=f"refs/heads/{branch}",
                    recorded_ownership=state.get("git_ownership"),
                )
                if reconciled is not None:
                    checkpoint, reconciled_retries = reconciled
            restore: tuple[str, ...] = ()
            mismatch_recovery: dict[str, Any] | None = None
            mismatch_recovery_path: Path | None = None
            scope_violation_recovery: dict[str, Any] | None = None
            scope_violation_origin = _scope_violation_origin(state, checkpoint)
            if scope_violation_origin is not None:
                if checkpoint.check_repair_attempt is not None:
                    fallback = (
                        original_plan_scope if checkpoint.cycle == 1
                        else c02_repair_scope or original_plan_scope
                    )
                    scope_dir = (
                        run_dir / "revision" / "check-repair"
                        / f"C0{checkpoint.cycle}" / "attempts"
                        / f"{checkpoint.check_repair_attempt:02d}"
                    )
                elif checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
                    fallback = original_plan_scope
                    scope_dir = run_dir / "revision" / "check-repair" / "C01"
                elif checkpoint.phase is ResumePhase.CHECK_REPAIR_EXPANDED_C01:
                    fallback = original_plan_scope
                    scope_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
                elif checkpoint.phase is ResumePhase.CHECK_REPAIR_C02:
                    fallback = c02_repair_scope or original_plan_scope
                    scope_dir = run_dir / "revision" / "check-repair" / "C02"
                else:
                    fallback = c02_repair_scope or original_plan_scope
                    scope_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
                failed_scope = _read_check_repair_scope(
                    scope_dir, fallback_base=fallback,
                    policy_config=self._effective_repair_scope,
                    allowed_added_sources=_SECOND_SCOPE_SOURCES,
                )
                self._recover_failed_revision_scope_violation(
                    repo=repo, worktree=worktree, run_dir=run_dir,
                    checkpoint=checkpoint, mutable_scope=failed_scope.effective_paths,
                    origin_reason=scope_violation_origin,
                )
                scope_violation_recovery = _read_json_artifact(
                    scope_dir / "scope_violation_recovery.json", 64 * 1024
                )
                # The rollback above changes the live Git observations used by
                # the generic drift classifier below.  Refresh them before
                # that classifier runs; otherwise it would inspect the
                # already-replaced failed tree and reject its own recovery.
                candidate = candidate_tree_sha(worktree)
                index_tree = index_tree_sha(worktree)
                dirty = _status_has_unstaged_or_untracked(status_porcelain(worktree))
            if checkpoint.phase in scope_repair_phases:
                recovery_cycle = "C01" if checkpoint.cycle == 1 else "C02"
                normal_recovery_dir = run_dir / "revision" / "check-repair" / recovery_cycle
                expanded_recovery_dir = run_dir / "revision" / "check-repair-expanded" / recovery_cycle
                recovery_dir = (
                    normal_recovery_dir
                    if (normal_recovery_dir / "scope_violation_recovery.json").is_file()
                    else expanded_recovery_dir
                )
                scope_violation_recovery = _read_json_artifact(
                    recovery_dir / "scope_violation_recovery.json", 64 * 1024
                )
                if not isinstance(scope_violation_recovery, dict):
                    refuse("scope-repair recovery artifact is missing")
            if dirty_mismatch_shape:
                step_id = checkpoint.step_id
                if step_id is None:
                    refuse("dirty contract mismatch has no step id")
                step_root = self._step_root(run_dir, checkpoint)
                record = _read_json_artifact(
                    step_root / "steps" / step_id / "step.json", 128 * 1024
                )
                if not isinstance(record, dict):
                    refuse("dirty contract mismatch artifact is missing")
                before, after = record.get("tree_before"), record.get("tree_after")
                if before != checkpoint.expected_tree_sha:
                    refuse("dirty contract mismatch tree_before does not match the checkpoint")
                changed = changed_paths_between_trees(repo, before, after)
                if not changed:
                    refuse("dirty contract mismatch has no changed paths")
                step_scope = self._step_mutable_scope(
                    run_dir=run_dir, checkpoint=checkpoint, plan=plan,
                    selection=selection,
                )
                outside = [path for path in changed if path not in step_scope]
                if outside:
                    raise ResumeRequiresOperatorError(
                        "dirty contract mismatch changed paths outside the failed step scope: "
                        + _paths_detail(outside)
                    )
                failure_tree = _read_tree_file(
                    step_root / "steps" / step_id / "tree_after_failure.txt"
                )
                if failure_tree is not None and failure_tree != after:
                    refuse("dirty contract mismatch failure tree does not match step.json")
                recorded_index = record.get("index_tree_after")
                if recorded_index is not None:
                    if not _is_object_id(recorded_index) or recorded_index != index_tree:
                        raise ResumeRequiresOperatorError(
                            "dirty contract mismatch proof no longer matches the current index"
                        )
                recorded_ownership = _ownership_from_payload(state.get("git_ownership"))
                if recorded_ownership is None:
                    raise ResumeRequiresOperatorError(
                        "dirty contract mismatch Git ownership proof is missing or invalid"
                    )
                current_ownership = _git_ownership(repo, worktree)
                if _git_ownership_payload(current_ownership) != _git_ownership_payload(recorded_ownership):
                    raise ResumeRequiresOperatorError(
                        "dirty contract mismatch proof no longer matches Git ownership"
                    )
                mismatch_recovery = {
                    "schema_version": 1,
                    "mode": "rollback_in_scope_dirty_mismatch",
                    "tree_before": before,
                    "tree_after_failure": after,
                    "restored_paths": list(changed),
                    "mismatch_retry_count_before": (
                        record.get("mismatch_retry_count")
                        if isinstance(record.get("mismatch_retry_count"), int)
                        and not isinstance(record.get("mismatch_retry_count"), bool)
                        else 0
                    ),
                }
                mismatch_recovery_path = (
                    step_root / "steps" / step_id / "mismatch_recovery.json"
                )
                if mismatch_recovery_path.is_file():
                    try:
                        if mismatch_recovery_path.read_text(encoding="utf-8") != _json_text(mismatch_recovery):
                            refuse("dirty mismatch recovery artifact is divergent")
                    except (OSError, UnicodeError) as exc:
                        refuse(f"dirty mismatch recovery artifact is unreadable: {exc}")
                already_restored = (
                    candidate == before
                    and index_tree == before
                    and not dirty
                    and mismatch_recovery_path.is_file()
                )
                if already_restored:
                    pass
                else:
                    if candidate != after:
                        raise ResumeRequiresOperatorError(
                            "dirty contract mismatch proof no longer matches the current candidate tree"
                        )
                    restore = tuple(changed)
            elif candidate != checkpoint.expected_tree_sha or index_tree != checkpoint.expected_tree_sha or dirty:
                restore = self._explain_tree_drift(
                    repo, run_dir, checkpoint, candidate, scope,
                    restore_failed_steps=not self._legacy_run_options,
                )
            unapproved = [
                path for path in changed_paths_between_trees(repo, base_tree, checkpoint.expected_tree_sha)
                if path not in scope
            ]
        except GitError as exc:
            refuse(f"Git state is unreadable: {exc}")
        # A run may have failed on a clean structural mismatch, possibly
        # before clean mismatch semantics existed.  Reconcile only the exact
        # no-op shape: the failed artifact, the checkpoint tree and the
        # current candidate/index must all agree, with no unstaged or
        # untracked residue.  The accumulated, staged modifications of the
        # earlier approved steps are legitimate and are never a gate; nor are
        # ignored files.  Any real residual evidence remains an
        # operator-required failure.
        mismatch_retries: dict[str, str] = dict(reconciled_retries)
        if clean_mismatch_shape or dirty_mismatch_shape:
            step_id = checkpoint.step_id
            root = self._step_root(run_dir, checkpoint)
            record = _read_json_artifact(root / "steps" / step_id / "step.json")
            if not isinstance(record, dict):
                raise ResumeIntegrityError("contract mismatch artifact is missing")
            if clean_mismatch_shape and (
                candidate != record.get("tree_before") or index_tree != candidate or dirty
            ):
                raise ResumeRequiresOperatorError(
                    "clean contract mismatch proof no longer matches the current worktree"
                )
            if clean_mismatch_shape:
                recorded_ownership = state.get("git_ownership")
            else:
                recorded_ownership = None
            if clean_mismatch_shape and isinstance(recorded_ownership, Mapping):
                current_ownership = _git_ownership(repo, worktree)
                if _git_ownership_payload(current_ownership) != dict(recorded_ownership):
                    raise ResumeRequiresOperatorError(
                        "clean contract mismatch proof no longer matches Git ownership"
                    )
            mismatch = record.get("mismatch")
            if not isinstance(mismatch, str) or not mismatch.strip():
                detail = failure.get("detail")
                mismatch = re.sub(rf"^step={re.escape(step_id)}\s*", "", detail or "")
            mismatch = _bounded_v2_report(redact(mismatch, self._secrets))
            if not mismatch:
                raise ResumeIntegrityError("clean contract mismatch explanation is missing")
            spent = record.get("mismatch_retry_count")
            if not mismatch_retry_spent(run_dir, step_id):
                # The bounded retry budget of this step is intact: replay
                # nothing, restore nothing, and run exactly one fresh worker
                # on the same step with the mismatch retry addendum.  The
                # checkpoint already names this step and its pre-step tree.
                mismatch_retries[step_id] = mismatch
            else:
                # The retry was already spent: the clean mismatch becomes the
                # deferred outcome and the chain continues.  Never a third
                # attempt.
                atomic_write_text(root / "steps" / step_id / "step.json", _json_text({
                    "id": step_id, "status": "DEFERRED_CONTRACT_MISMATCH",
                    "profile_id": record.get("profile_id"),
                    "tree_before": record["tree_before"], "tree_after": record["tree_before"],
                    "changed_paths": [], "mismatch": mismatch,
                    "mismatch_retry_count": spent,
                    **({"initial_mismatch": _bounded_v2_report(str(record["initial_mismatch"]))}
                       if isinstance(record.get("initial_mismatch"), str)
                       and record["initial_mismatch"].strip() else {}),
                    "usage": normalize_usage(record.get("usage")),
                }))
                ids = (
                    [step.id for step in plan.steps]
                    if checkpoint.phase is ResumePhase.INITIAL_STEP
                    else self._repair_step_ids(
                        run_dir, checkpoint, selection, plan.required_checks
                    ) or []
                )
                if step_id not in ids:
                    raise ResumeIntegrityError(
                        "contract mismatch step is missing from the approved step plan"
                    )
                index = ids.index(step_id)
                following = ids[index + 1] if index + 1 < len(ids) else None
                if following:
                    next_phase = checkpoint.phase
                else:
                    if checkpoint.phase is ResumePhase.INITIAL_STEP:
                        next_phase = ResumePhase.CHECKS_C01
                    elif checkpoint.phase is ResumePhase.REPAIR_STEP:
                        next_phase = ResumePhase.CHECKS_C02
                    else:
                        next_phase = (
                            ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01
                            if checkpoint.phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01
                            else ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02
                        )
                checkpoint = ResumeCheckpoint(
                    next_phase, checkpoint.cycle, following, checkpoint.expected_head_sha,
                    record["tree_before"], checkpoint.execution_selection_sha256,
                    checkpoint.plan_identity, checkpoint.repair_bundle_sha256,
                    checkpoint.scope_delta_sha256,
                )
                resume_module.write_checkpoint(run_dir, checkpoint)
        # A transient failure (timeout, transport, auth) of a step that was
        # already running its single bounded mismatch retry keeps that mode:
        # the same operation is rerun with the same contract, the same retry
        # addendum and the same future ownership.  A transport retry never
        # creates a normal first attempt, and never a second semantic retry.
        if not mismatch_retries and checkpoint.phase in {
            ResumePhase.INITIAL_STEP, ResumePhase.REPAIR_STEP,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
        }:
            pending = self._pending_mismatch_retry_mode(run_dir, checkpoint)
            if pending:
                mismatch_retries[str(checkpoint.step_id)] = pending
        if unapproved:
            refuse("the candidate contains a path outside the approved scope")
        resumed = _ResumedRun(
            checkpoint=checkpoint, plan=plan, bundle=bundle, selection=selection,
            info=WorktreeInfo(
                source_repo=repo, worktree=worktree, branch=branch,
                base_ref=self.config.base_ref, base_sha=base_sha,
            ),
            repository_reference=reference, spec=spec, context=context,
            restore_paths=restore,
            mismatch_recovery=mismatch_recovery,
            mismatch_recovery_path=mismatch_recovery_path,
            mismatch_retries=mismatch_retries,
            scope_violation_recovery=scope_violation_recovery,
            existing_commit_sha=existing_commit,
        )
        if checkpoint.phase is not ResumePhase.PUBLISH:
            self._load_resumed_results(
                run_dir, resumed, base_tree, revision_enabled, repair_enabled
            )
        if checkpoint != original_checkpoint:
            # A reconciled boundary is persisted only once every durable
            # artifact before it has been read back and verified.
            resume_module.write_checkpoint(run_dir, checkpoint)
        return resumed

    @staticmethod
    def _step_root(run_dir: Path, checkpoint: ResumeCheckpoint) -> Path:
        """The durable step directory root of a step checkpoint's cycle."""

        return (
            run_dir if checkpoint.phase is ResumePhase.INITIAL_STEP
            else run_dir / "scope-repair" / (
                "C01" if checkpoint.phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01
                else "C02"
            ) if checkpoint.phase in {
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
            }
            else run_dir / "repair" / "C02"
        )

    @staticmethod
    def _pending_mismatch_retry_mode(
        run_dir: Path, checkpoint: ResumeCheckpoint,
    ) -> str:
        """The initial mismatch a failed step must be retried with, if any.

        A step whose durable record failed *while already in mismatch-retry
        mode* keeps that mode across the transient failure: the resume owes it
        the same semantic retry, not a fresh first attempt.
        """

        record = _read_json_artifact(
            Orchestrator._step_root(run_dir, checkpoint)
            / "steps" / str(checkpoint.step_id) / "step.json",
            128 * 1024,
        )
        if not isinstance(record, dict) or record.get("id") != checkpoint.step_id:
            return ""
        if record.get("status") != "FAILED":
            return ""
        spent = record.get("mismatch_retry_count")
        initial = record.get("initial_mismatch")
        if (
            not isinstance(spent, int) or isinstance(spent, bool) or spent < 1
            or not isinstance(initial, str) or not initial.strip()
        ):
            return ""
        return _bounded_v2_report(initial)

    def _repair_step_ids(
        self, run_dir: Path, checkpoint: ResumeCheckpoint, selection: Any,
        inherited_check_ids: Any,
    ) -> list[str] | None:
        """The C02 step ids of the hash-bound repair bundle, or ``None``."""

        repair_dir = (
            run_dir / "scope-repair" / (
                "C01" if checkpoint.phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01 else "C02"
            )
            if checkpoint.phase in {
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
                ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
            }
            else run_dir / "repair" / "C02"
        )
        try:
            repair_plan = parse_task_plan_v2(
                (repair_dir / "planner.raw.md").read_text(encoding="utf-8"),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                check_catalog=self.config.check_catalog,
                inherited_check_ids=inherited_check_ids,
            )
            _bundle, repair_sha = validate_implementation_bundle(
                repair_dir, expected_step_ids=[step.id for step in repair_plan.steps]
            )
        except (V2PlanParseError, OrchestrationError, OSError, UnicodeError, ValueError,
                AttributeError):
            return None
        if repair_plan.decision is not PlanDecision.READY or repair_sha != checkpoint.repair_bundle_sha256:
            return None
        return [step.id for step in repair_plan.steps]

    def _step_mutable_scope(
        self,
        *,
        run_dir: Path,
        checkpoint: ResumeCheckpoint,
        plan: TaskPlanV2,
        selection: Any,
    ) -> tuple[str, ...]:
        """Return only the mutable paths owned by the checkpointed step.

        A dirty mismatch is rolled back against this exact step contract.  In
        particular, the original plan's union is deliberately not used here:
        a path owned by a later step is still out of scope for the failed
        step, even when it is approved elsewhere in the run.
        """

        candidate_plan = plan
        if checkpoint.phase in {
            ResumePhase.REPAIR_STEP,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
        }:
            repair_dir = (
                run_dir / "scope-repair" / (
                    "C01" if checkpoint.phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01 else "C02"
                )
                if checkpoint.phase in {
                    ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
                    ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
                }
                else run_dir / "repair" / "C02"
            )
            try:
                candidate_plan = parse_task_plan_v2(
                    (repair_dir / "planner.raw.md").read_text(encoding="utf-8"),
                    implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                    reviewer_ids=frozenset({selection.reviewer.profile_id}),
                    check_catalog=self.config.check_catalog,
                    inherited_check_ids=plan.required_checks,
                )
                _bundle, repair_sha = validate_implementation_bundle(
                    repair_dir, expected_step_ids=[step.id for step in candidate_plan.steps]
                )
            except (
                V2PlanParseError, OrchestrationError, OSError, UnicodeError,
                ValueError, AttributeError,
            ):
                return ()
            if (
                candidate_plan.decision is not PlanDecision.READY
                or repair_sha != checkpoint.repair_bundle_sha256
            ):
                return ()
        step = next(
            (item for item in candidate_plan.steps if item.id == checkpoint.step_id),
            None,
        )
        if step is None:
            return ()
        return tuple(sorted({
            *step.write_set,
            *step.create_set,
            *step.delete_set,
        }))

    def _reconcile_durable_step(
        self, run_dir: Path, checkpoint: ResumeCheckpoint, step_ids: list[str], *,
        repo: Path, worktree: Path, candidate: str, index_tree: str,
        dirty: list[str], scope: list[str], expected_head: str, branch_ref: str,
        recorded_ownership: Any,
    ) -> tuple[ResumeCheckpoint, dict[str, str]] | None:
        """Reconcile a step that succeeded durably before the checkpoint moved.

        A crash can happen after ``steps/Sxx/step.json`` is ``COMPLETED`` (or
        a deferred clean mismatch) and before the checkpoint names Sxx+1.
        Only that exact shape is accepted — the record's own pre-step tree is
        the checkpoint tree, the current candidate *and* index are exactly its
        post-step tree, there is no residue, every changed path is in the
        approved mutable scope of that cycle and the Git boundary still holds.
        The boundary then advances without invoking the worker again; anything
        else returns ``None`` and stays subject to the drift verdict.
        """

        step_id = checkpoint.step_id
        if step_id is None or step_id not in step_ids:
            return None
        record = _load_completed_step(
            self._step_root(run_dir, checkpoint) / "steps" / step_id, step_id
        )
        if record is None:
            return None
        if (
            record["tree_before"] != checkpoint.expected_tree_sha
            or candidate != record["tree_after"]
            or index_tree != record["tree_after"]
            or dirty
            or any(path not in scope for path in record["changed_paths"])
        ):
            return None
        try:
            ownership = _git_ownership(repo, worktree)
        except GitError:
            return None
        if ownership.head_ref != branch_ref or ownership.head != expected_head:
            return None
        if isinstance(recorded_ownership, Mapping):
            before = _ownership_from_payload(recorded_ownership)
            if before is None or _ownership_violations(
                before, ownership, branch_ref=branch_ref, base_sha=expected_head
            ):
                return None
        spent = record.get("mismatch_retry_count")
        if record["status"] == "DEFERRED_CONTRACT_MISMATCH" and not (
            isinstance(spent, int) and not isinstance(spent, bool) and spent >= 1
        ):
            # The durable deferral still owns its single bounded retry: keep
            # the same step and schedule exactly that retry, nothing else.
            return checkpoint, {step_id: str(record.get("mismatch") or "")}
        index = step_ids.index(step_id)
        following = step_ids[index + 1] if index + 1 < len(step_ids) else None
        checks_phase = (
            ResumePhase.CHECKS_C01
            if checkpoint.phase is ResumePhase.INITIAL_STEP
            else ResumePhase.CHECKS_C02
            if checkpoint.phase is ResumePhase.REPAIR_STEP
            else ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01
            if checkpoint.phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01
            else ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02
        )
        return ResumeCheckpoint(
            checkpoint.phase if following else checks_phase,
            checkpoint.cycle, following, checkpoint.expected_head_sha,
            record["tree_after"], checkpoint.execution_selection_sha256,
            checkpoint.plan_identity, checkpoint.repair_bundle_sha256,
            checkpoint.scope_delta_sha256,
        ), {}

    def _validated_repair_scope(
        self, run_dir: Path, checkpoint: ResumeCheckpoint, plan: TaskPlanV2,
        selection: Any, original_scope: list[str],
    ) -> list[str]:
        """The C02 repair mutable scope, derived only from hash-bound artifacts.

        The repair bundle and ``scope_delta.json`` must match the checkpoint
        hashes, the delta must be exactly the parsed repair plan's mutation
        sets against the C01 scope, and an expansion must satisfy the run's
        scope policy (including the exact scope approval when required).
        Nothing is taken from ``state.json`` or reviewer prose.
        """

        def refuse(message: str) -> NoReturn:
            raise ResumeIntegrityError(message)

        if checkpoint.scope_delta_sha256 is None:
            # No validated scope delta: no expansion authority at all.
            return []
        if checkpoint.repair_bundle_sha256 is None:
            refuse("the C02 scope delta is not bound to a repair bundle")
        repair_dir = run_dir / "repair" / "C02"
        try:
            repair_plan = parse_task_plan_v2(
                (repair_dir / "planner.raw.md").read_text(encoding="utf-8"),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                check_catalog=self.config.check_catalog,
                inherited_check_ids=plan.required_checks,
            )
            _bundle, repair_sha = validate_implementation_bundle(
                repair_dir, expected_step_ids=[step.id for step in repair_plan.steps]
            )
            writes, creates, deletes = _repair_mutation_sets(repair_plan)
            path = repair_dir / "scope_delta.json"
            if path.stat().st_size > 256 * 1024:
                refuse("the C02 scope delta is too large")
            delta_bytes = path.read_bytes()
            delta = json.loads(delta_bytes.decode("utf-8"))
        except (V2PlanParseError, OrchestrationError, OSError, UnicodeError, ValueError,
                AttributeError) as exc:
            refuse(f"the C02 repair scope is unreadable: {exc}")
        scope_c02_phases = {
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        }
        if repair_plan.decision is not PlanDecision.READY or (
            checkpoint.phase not in scope_c02_phases
            and repair_sha != checkpoint.repair_bundle_sha256
        ):
            refuse("the C02 repair bundle changed")
        if hashlib.sha256(delta_bytes).hexdigest() != checkpoint.scope_delta_sha256:
            refuse("the C02 scope delta changed")
        requested = sorted(set(writes) | set(creates) | set(deletes))
        added = sorted(set(requested) - set(original_scope))
        if (
            not isinstance(delta, dict)
            or delta.get("repair_bundle_sha256") != repair_sha
            or delta.get("original_mutable_paths") != sorted(set(original_scope))
            or delta.get("requested_write_paths") != writes
            or delta.get("requested_create_paths") != creates
            or delta.get("requested_delete_paths") != deletes
            or delta.get("added_paths") != added
        ):
            refuse("the C02 scope delta does not match the repair plan")
        if added:
            scope_policy = self._effective_repair_scope
            if scope_policy.policy == "deny-expansion":
                refuse("the C02 scope delta is not allowed by the run scope policy")
            requires_scope_approval = (
                scope_policy.policy == "require-approval"
                or (
                    scope_policy.policy == "auto-bounded"
                    and len(added) > scope_policy.max_added_paths
                )
            )
            if requires_scope_approval:
                try:
                    approval = read_scope_approval(
                        repair_dir, expected_sha256=checkpoint.scope_delta_sha256
                    )
                except ApprovalError as exc:
                    refuse(f"the C02 scope approval is invalid: {exc}")
                if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                    refuse("the C02 scope expansion was not approved")
        return requested

    def _validated_scope_repair_scope(
        self, run_dir: Path, checkpoint: ResumeCheckpoint, plan: TaskPlanV2,
        selection: Any,
    ) -> list[str]:
        """Validate the hash-bound scope-repair bundle and return its sets."""

        if checkpoint.phase in {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
        }:
            return []
        if checkpoint.repair_bundle_sha256 is None or checkpoint.scope_delta_sha256 is None:
            raise ResumeIntegrityError("scope-repair checkpoint is missing its bundle or delta hash")
        cycle = 1 if checkpoint.cycle == 1 else 2
        directory = run_dir / "scope-repair" / f"C0{cycle}"
        try:
            parsed = parse_task_plan_v2(
                (directory / "planner.raw.md").read_text(encoding="utf-8"),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                check_catalog=self.config.check_catalog,
                inherited_check_ids=plan.required_checks,
            )
            _bundle, bundle_sha = validate_implementation_bundle(
                directory, expected_step_ids=[step.id for step in parsed.steps]
            )
            delta_path = directory / "scope_delta.json"
            delta_bytes = delta_path.read_bytes()
            delta = json.loads(delta_bytes.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, V2PlanParseError, AttributeError) as exc:
            raise ResumeIntegrityError(f"scope-repair artifacts are unreadable: {exc}") from exc
        if parsed.decision is not PlanDecision.READY or bundle_sha != checkpoint.repair_bundle_sha256:
            raise ResumeIntegrityError("scope-repair bundle changed")
        if hashlib.sha256(delta_bytes).hexdigest() != checkpoint.scope_delta_sha256:
            raise ResumeIntegrityError("scope-repair scope delta changed")
        writes, creates, deletes = _repair_mutation_sets(parsed)
        requested = sorted(set(writes) | set(creates) | set(deletes))
        if (
            not isinstance(delta, dict)
            or delta.get("schema_version") != 1
            or delta.get("trigger") != "revision_scope_violation"
            or delta.get("requested_write_paths") != writes
            or delta.get("requested_create_paths") != creates
            or delta.get("requested_delete_paths") != deletes
            or delta.get("repair_bundle_sha256") != bundle_sha
        ):
            raise ResumeIntegrityError("scope-repair scope delta does not match the plan")
        added = sorted(set(requested) - set(delta.get("original_mutable_paths", [])))
        if delta.get("added_paths") != added:
            raise ResumeIntegrityError("scope-repair added paths changed")
        if added:
            policy = self._effective_repair_scope
            if policy.policy == "deny-expansion":
                raise ResumeIntegrityError("scope-repair expansion is denied")
            if policy.policy == "require-approval" or (
                policy.policy == "auto-bounded" and len(added) > policy.max_added_paths
            ):
                approval = read_scope_approval(directory, expected_sha256=checkpoint.scope_delta_sha256)
                if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                    raise ResumeIntegrityError("scope-repair scope approval is missing or not approved")
        return requested

    def _scope_repair_authority_tree(
        self, run_dir: Path, resumed: "_ResumedRun", *, cycle: int,
        inherited_check_ids: Any,
    ) -> tuple[str | None, str | None]:
        """The durable tree a scope-repair checkpoint must carry.

        The authority of a scope-repair phase is its own chain -- the rolled
        back failed-check tree, the bounded Luna steps, then the residual
        Claude pass -- and never the C01/C02 Claude tree the cycle answers
        to.  The bounded plan is re-derived from the hash-bound bundle so a
        checkpoint can never name a step the approved plan does not contain.
        Returns ``(tree, refusal)``.
        """

        checkpoint = resumed.checkpoint
        recovery = resumed.scope_violation_recovery or {}
        recovery_tree = recovery.get("tree_before")
        if not _is_object_id(recovery_tree):
            return None, "scope-repair rollback tree is malformed"
        directory = run_dir / "scope-repair" / f"C0{cycle}"
        step_ids: tuple[str, ...] = ()
        if checkpoint.phase not in _SCOPE_REPAIR_RECOVERY_TREE_PHASES:
            selection = resumed.selection
            try:
                parsed = parse_task_plan_v2(
                    (directory / "planner.raw.md").read_text(encoding="utf-8"),
                    implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                    reviewer_ids=frozenset({selection.reviewer.profile_id}),
                    check_catalog=self.config.check_catalog,
                    inherited_check_ids=inherited_check_ids,
                )
                _bundle, bundle_sha = validate_implementation_bundle(
                    directory, expected_step_ids=[step.id for step in parsed.steps]
                )
            except (OSError, UnicodeError, V2PlanParseError, OrchestrationError,
                    ValueError, AttributeError) as exc:
                return None, f"the scope-repair plan is unreadable: {exc}"
            if parsed.decision is not PlanDecision.READY:
                return None, "the scope-repair plan is not READY"
            if bundle_sha != checkpoint.repair_bundle_sha256:
                return None, "the scope-repair bundle changed"
            step_ids = tuple(step.id for step in parsed.steps)
        return _scope_repair_checkpoint_tree(
            directory=directory, checkpoint=checkpoint,
            recovery_tree=recovery_tree, step_ids=step_ids,
        )

    @staticmethod
    def _explain_tree_drift(
        repo: Path, run_dir: Path, checkpoint: ResumeCheckpoint, candidate: str,
        scope: list[str], *, restore_failed_steps: bool = False,
    ) -> tuple[str, ...]:
        """Classify a candidate that is not the checkpoint tree.

        Only one drift is recoverable: the tree a failed Claude attempt left
        behind (recorded in ``tree_after_failure.txt``), when every changed
        path is in the approved scope.  It is then restored exactly.  Any
        other difference is tampering or an unknown writer.
        """

        phase = checkpoint.phase
        if phase in (ResumePhase.CLAUDE_C01, ResumePhase.CLAUDE_C02,
                     ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_C02,
                     ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                     ResumePhase.CHECK_REPAIR_EXPANDED_C02):
            if checkpoint.check_repair_attempt is not None:
                directory = run_dir / "revision" / "check-repair" / (
                    "C01" if checkpoint.cycle == 1 else "C02"
                ) / "attempts" / f"{checkpoint.check_repair_attempt:02d}"
            elif phase is ResumePhase.CHECK_REPAIR_C01:
                directory = run_dir / "revision" / "check-repair" / "C01"
            elif phase is ResumePhase.CHECK_REPAIR_C02:
                directory = run_dir / "revision" / "check-repair" / "C02"
            elif phase is ResumePhase.CHECK_REPAIR_EXPANDED_C01:
                directory = run_dir / "revision" / "check-repair-expanded" / "C01"
            elif phase is ResumePhase.CHECK_REPAIR_EXPANDED_C02:
                directory = run_dir / "revision" / "check-repair-expanded" / "C02"
            else:
                directory = run_dir / "revision" / ("C02" if phase is ResumePhase.CLAUDE_C02 else "")
            failure_tree = _read_tree_file(directory / "tree_after_failure.txt")
            if failure_tree is not None and candidate == failure_tree:
                changed = changed_paths_between_trees(repo, checkpoint.expected_tree_sha, candidate)
                if not changed or any(path not in scope for path in changed):
                    raise ResumeRequiresOperatorError(
                        "the failed Claude attempt changed paths outside the approved scope"
                    )
                return tuple(changed)
        elif phase in (ResumePhase.INITIAL_STEP, ResumePhase.REPAIR_STEP,
                       ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
                       ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02):
            root = run_dir if phase is ResumePhase.INITIAL_STEP else run_dir / "repair" / "C02"
            if phase in {ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02}:
                root = run_dir / "scope-repair" / ("C01" if phase is ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01 else "C02")
            record = _read_json_artifact(root / "steps" / str(checkpoint.step_id) / "step.json")
            if (
                isinstance(record, dict) and record.get("status") == "FAILED"
                and record.get("tree_after") == candidate
            ):
                changed = changed_paths_between_trees(repo, checkpoint.expected_tree_sha, candidate)
                # The durable bundle is the authority for the mutable scope;
                # this branch is only enabled for P30 snapshots.  Legacy P29
                # runs retain their historical operator-required behavior.
                if restore_failed_steps and changed and all(path in scope for path in changed):
                    return tuple(changed)
                raise ResumeRequiresOperatorError(
                    f"the failed step {checkpoint.step_id} left partial changes; start a new run"
                )
        raise ResumeIntegrityError("the worktree differs from the checkpoint tree")

    def _recover_failed_revision_scope_violation(
        self,
        *,
        repo: Path,
        worktree: Path,
        run_dir: Path,
        checkpoint: ResumeCheckpoint,
        mutable_scope: Sequence[str],
        origin_reason: str,
    ) -> tuple[str, ...]:
        """Validate, roll back, and archive an unsafe check-repair attempt.

        The failed Claude tree is treated as evidence only.  Every changed
        path is restored from the checkpoint tree, including paths that were
        inside the previous scope; keeping a partial attempt would make the
        subsequent strong planner reason from an unauthorised candidate.

        *origin_reason* is the cause :func:`_scope_violation_origin` recovered
        for this checkpoint.  It is the provenance authority: ``state.failure``
        may already carry a later refusal verdict, which says nothing about the
        attempt this directory recorded.
        """

        state = _read_json_artifact(run_dir / "state.json", 256 * 1024)
        if not isinstance(state, Mapping):
            raise ResumeIntegrityError("scope-violation run state is unreadable")
        if origin_reason not in _SCOPE_VIOLATION_ORIGINS:
            raise ResumeIntegrityError("scope-violation recovery requires a Claude scope route")
        if checkpoint.phase not in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
        }:
            raise ResumeIntegrityError("scope-violation recovery is only valid for check-repair phases")
        if checkpoint.expected_head_sha is None or checkpoint.expected_tree_sha is None:
            raise ResumeIntegrityError("scope-violation checkpoint has no Git identity")

        if checkpoint.check_repair_attempt is not None:
            directory = (
                run_dir / "revision" / "check-repair"
                / f"C0{checkpoint.cycle}" / "attempts"
                / f"{checkpoint.check_repair_attempt:02d}"
            )
        elif checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
            directory = run_dir / "revision" / "check-repair" / "C01"
        elif checkpoint.phase is ResumePhase.CHECK_REPAIR_EXPANDED_C01:
            directory = run_dir / "revision" / "check-repair-expanded" / "C01"
        elif checkpoint.phase is ResumePhase.CHECK_REPAIR_C02:
            directory = run_dir / "revision" / "check-repair" / "C02"
        else:
            directory = run_dir / "revision" / "check-repair-expanded" / "C02"
        report_path = directory / "report.json"
        failure_tree_path = directory / "tree_after_failure.txt"
        report = _read_json_artifact(report_path, 1024 * 1024)
        if not isinstance(report, dict):
            raise ResumeIntegrityError("scope-violation report or failure tree is missing")
        if report.get("tree_before") != checkpoint.expected_tree_sha:
            raise ResumeIntegrityError("scope-violation report tree_before does not match the checkpoint")
        failure_tree = _read_tree_file(failure_tree_path)
        # Runs created before ``tree_after_failure.txt`` became durable carry
        # their failed tree only inside ``report.json``.  That record is
        # promoted to the canonical artifact below, but only when Git still
        # holds the exact tree it names: this reconstructs a lost artifact from
        # live proof, it never asserts a past the harness can no longer see.
        synthesized_failure_tree = failure_tree is None
        if synthesized_failure_tree:
            if failure_tree_path.exists():
                raise ResumeIntegrityError("scope-violation failure tree is malformed")
            legacy_tree_after = report.get("tree_after")
            if not _is_object_id(legacy_tree_after):
                raise ResumeIntegrityError("scope-violation report tree_after is not a Git object id")
            if (
                legacy_tree_after == checkpoint.expected_tree_sha
                and report.get("scope_request") is None
            ):
                raise ResumeIntegrityError("scope-violation report tree_after is the checkpoint tree")
            existing_recovery = _read_json_artifact(
                directory / "scope_violation_recovery.json", 64 * 1024
            )
            # A recovery artifact from a previous resume already proved this
            # exact failed tree; the rollback it records is why the live
            # worktree is no longer that tree.
            proven_by_recovery = (
                isinstance(existing_recovery, dict)
                and existing_recovery.get("tree_before") == checkpoint.expected_tree_sha
                and existing_recovery.get("tree_after_failure") == legacy_tree_after
            )
            if not proven_by_recovery:
                try:
                    live_candidate = candidate_tree_sha(worktree)
                    live_index = index_tree_sha(worktree)
                except GitError as exc:
                    raise ResumeIntegrityError(
                        f"scope-violation Git state is unreadable: {exc}"
                    ) from exc
                if live_candidate != legacy_tree_after or live_index != legacy_tree_after:
                    raise ResumeIntegrityError(
                        "scope-violation failure tree is missing and the worktree is no "
                        "longer the tree_after the report recorded"
                    )
            failure_tree = legacy_tree_after
        if report.get("tree_after") != failure_tree:
            raise ResumeIntegrityError("scope-violation report tree_after does not match the failure tree")
        recorded_scope_request = report.get("scope_request")
        if recorded_scope_request is not None:
            if not isinstance(recorded_scope_request, Mapping):
                raise ResumeIntegrityError("scope-violation scope request is malformed")
            parsed_scope_request = parse_scope_request(
                _read_bounded_text(directory / "agent.final.md")
            )
            if parsed_scope_request is None or _scope_request_payload(parsed_scope_request) != dict(recorded_scope_request):
                raise ResumeIntegrityError("scope-violation scope request does not match the final report")

        try:
            ownership_before = _ownership_from_payload(state.get("git_ownership"))
            current_ownership = _git_ownership(repo, worktree)
            if ownership_before is None:
                raise ResumeIntegrityError("scope-violation Git ownership proof is missing")
            if _git_ownership_payload(current_ownership) != _git_ownership_payload(ownership_before):
                raise ResumeRequiresOperatorError("scope-violation Git ownership changed after failure")
            if current_ownership.head != checkpoint.expected_head_sha:
                raise ResumeRequiresOperatorError("scope-violation HEAD changed after failure")
            branch = state.get("branch")
            if not isinstance(branch, str) or current_ownership.head_ref != f"refs/heads/{branch}":
                raise ResumeRequiresOperatorError("scope-violation branch changed after failure")
            candidate = candidate_tree_sha(worktree)
            index_tree = index_tree_sha(worktree)
            if candidate not in {failure_tree, checkpoint.expected_tree_sha}:
                raise ResumeIntegrityError("current candidate is neither the failure tree nor the checkpoint tree")
            if candidate == failure_tree and index_tree != failure_tree:
                raise ResumeRequiresOperatorError("scope-violation index changed after failure")
            if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
                raise ResumeRequiresOperatorError("scope-violation worktree has new unstaged or untracked changes")
        except GitError as exc:
            raise ResumeIntegrityError(f"scope-violation Git state is unreadable: {exc}") from exc

        changed = tuple(changed_paths_between_trees(repo, checkpoint.expected_tree_sha, failure_tree))
        recorded_changed = report.get("changed_paths")
        if recorded_changed != list(changed):
            raise ResumeIntegrityError("scope-violation report changed_paths do not match the Git delta")
        if not changed and recorded_scope_request is None:
            raise ResumeIntegrityError("scope-violation report has no changed paths")
        scope = set(mutable_scope)
        expected_outside = tuple(sorted(set(changed) - scope))
        recorded_outside = report.get("outside_scope_paths")
        if recorded_outside is None:
            outside = expected_outside
        elif isinstance(recorded_outside, list) and all(isinstance(path, str) for path in recorded_outside):
            outside = tuple(recorded_outside)
        else:
            raise ResumeIntegrityError("scope-violation outside_scope_paths is malformed")
        if tuple(outside) != expected_outside:
            raise ResumeIntegrityError("scope-violation outside_scope_paths do not match the mutable scope")
        if not outside and recorded_scope_request is None:
            raise ResumeIntegrityError("scope-violation has no observed outside-scope path")
        if synthesized_failure_tree:
            if origin_reason == "REVISION_SCOPE_VIOLATION" and not outside:
                raise ResumeIntegrityError("scope-violation has no observed outside-scope path")
            # Every invariant above held against the live tree, so the report's
            # ``tree_after`` is now durable proof in its own right.  Write-once:
            # a concurrent writer that disagrees is a divergence, never a merge.
            try:
                _create_file_once(failure_tree_path, (failure_tree + "\n").encode("utf-8"))
            except FileExistsError:
                if _read_tree_file(failure_tree_path) != failure_tree:
                    raise ResumeIntegrityError(
                        "scope-violation failure tree diverges"
                    ) from None
            except OSError as exc:
                raise ResumeIntegrityError(
                    f"scope-violation failure tree cannot be recorded: {exc}"
                ) from exc

        recovery = {
            "schema_version": 1,
            "tree_before": checkpoint.expected_tree_sha,
            "tree_after_failure": failure_tree,
            "restored_paths": list(changed),
            "outside_scope_paths": list(outside),
            **(
                {
                    "scope_request": dict(recorded_scope_request),
                    "scope_request_diagnostic": (
                        "Claude requested scope expansion:\n"
                        f"  paths: {len(recorded_scope_request.get('paths', []))}\n"
                        "  authoritative: NO\n"
                        "  routed to bridge audit: YES"
                    ),
                }
                if isinstance(recorded_scope_request, Mapping) else {}
            ),
        }
        recovery_path = directory / "scope_violation_recovery.json"
        expected_bytes = _json_text(recovery).encode("utf-8")
        try:
            if recovery_path.exists():
                if recovery_path.read_bytes() != expected_bytes:
                    raise ResumeIntegrityError("scope-violation recovery artifact diverges")
            elif candidate_tree_sha(worktree) == failure_tree:
                restore_paths_from_tree(worktree, checkpoint.expected_tree_sha, list(changed))
                _create_file_once(recovery_path, expected_bytes)
            elif candidate_tree_sha(worktree) == checkpoint.expected_tree_sha:
                # A structured Claude request may have been rolled back
                # immediately after the successful attempt.  The recovery
                # proof is still required, even though there is no live dirty
                # tree left for this resume to restore.
                _create_file_once(recovery_path, expected_bytes)
            else:
                raise ResumeIntegrityError("scope-violation recovery artifact is missing after rollback")
        except FileExistsError:
            if recovery_path.read_bytes() != expected_bytes:
                raise ResumeIntegrityError("scope-violation recovery artifact diverges")
        except (OSError, GitError) as exc:
            raise ResumeRequiresOperatorError(f"scope-violation rollback failed: {exc}") from exc
        try:
            if (
                candidate_tree_sha(worktree) != checkpoint.expected_tree_sha
                or index_tree_sha(worktree) != checkpoint.expected_tree_sha
                or _status_has_unstaged_or_untracked(status_porcelain(worktree))
            ):
                raise ResumeRequiresOperatorError("scope-violation rollback did not restore the exact checkpoint tree")
        except GitError as exc:
            raise ResumeRequiresOperatorError(f"scope-violation rollback proof failed: {exc}") from exc
        return changed

    def _load_resumed_results(
        self, run_dir: Path, resumed: "_ResumedRun", base_tree: str,
        revision_enabled: bool, repair_enabled: bool,
    ) -> None:
        """Read back every phase before the checkpoint and verify its chain."""

        def refuse(message: str) -> NoReturn:
            raise ResumeIntegrityError(message)

        checkpoint = resumed.checkpoint
        at = phase_index(checkpoint.phase)
        # A checkpoint taken inside a scope-repair cycle has its own durable
        # authority chain; the historical C01/C02 Claude tree only explains
        # *why* that cycle exists.  The two identities are never mixed.
        scope_c01_phases = {
            ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
        }
        scope_c02_phases = {
            ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02, ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        }
        expected = checkpoint.expected_tree_sha
        ids = [step.id for step in resumed.plan.steps]
        if checkpoint.phase is ResumePhase.INITIAL_STEP:
            if checkpoint.step_id not in ids:
                refuse("checkpoint step is not in the approved plan")
            prior = ids[: ids.index(checkpoint.step_id)]
        else:
            prior = ids
        records = [_load_completed_step(run_dir / "steps" / step_id, step_id) for step_id in prior]
        if any(record is None for record in records):
            refuse("a completed Luna step record is missing or invalid")
        # Two distinct identities per cycle: worker_tree is the last Luna tree;
        # once Claude succeeded, its record's tree_after (revision_tree) is the
        # only authority for every later C01 phase.
        worker_tree = _verify_step_chain(records, base_tree)
        if worker_tree is None:
            refuse("Luna step trees do not form an unbroken chain")
        resumed.c01_steps = records
        c01_tree = worker_tree
        if revision_enabled and at >= phase_index(ResumePhase.FINAL_CHECKS_C01):
            revision = _load_revision(run_dir / "revision")
            if revision is None or revision.tree_before != worker_tree:
                refuse("the Claude C01 record is missing or not based on the Luna tree")
            resumed.c01_revision = revision
            c01_tree = revision.tree_after
        initial_c01_tree = c01_tree
        check_repair_dir_c01 = run_dir / "revision" / "check-repair" / "C01"
        generic_c01_attempts = self._check_repair_attempt_records(run_dir, 1)
        generic_c01 = bool(generic_c01_attempts) or checkpoint.check_repair_attempt is not None
        if generic_c01:
            c01_tree = self._check_repair_chain_tree(
                run_dir, 1, initial_c01_tree, checkpoint,
            )
        if (
            not generic_c01 and revision_enabled
            and resumed.scope_violation_recovery is None
            and at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_C01) and (
            check_repair_dir_c01 / "report.json"
            ).exists()
        ):
            repair_revision = _load_revision(
                check_repair_dir_c01
            )
            if repair_revision is None or repair_revision.tree_before != initial_c01_tree:
                refuse("the C01 check-repair record is missing or not based on the red checks tree")
            resumed.c01_check_repair_revision = repair_revision
            c01_tree = repair_revision.tree_after
        expanded_c01_dir = run_dir / "revision" / "check-repair-expanded" / "C01"
        if revision_enabled and resumed.scope_violation_recovery is None and at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01) and (
            checkpoint.phase in {
                ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
            }
            or expanded_c01_dir.joinpath("scope.json").is_file()
        ):
            normal_scope = _read_check_repair_scope(
                check_repair_dir_c01,
                fallback_base=sorted({
                    path for step in resumed.plan.steps
                    for path in (*step.write_set, *step.create_set, *step.delete_set)
                }),
                policy_config=self._effective_repair_scope,
            )
            _validate_expanded_check_repair_scope(
                expanded_c01_dir,
                repo=resumed.info.source_repo,
                tree_sha=c01_tree,
                normal_scope=normal_scope,
                policy_config=self._effective_repair_scope,
            )
            if at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01):
                expanded_revision = _load_revision(expanded_c01_dir)
                if expanded_revision is None or expanded_revision.tree_before != c01_tree:
                    refuse("the expanded C01 check-repair record is missing or not based on the red checks tree")
                resumed.c01_expanded_check_repair_revision = expanded_revision
                c01_tree = expanded_revision.tree_after
        if resumed.scope_violation_recovery is not None and checkpoint.phase in {
            ResumePhase.CANDIDATE_COMMIT_C01, ResumePhase.CANDIDATE_PUSH_C01,
            ResumePhase.REVIEWER_C01,
        }:
            scope_evidence = _load_evidence(run_dir / "scope-repair" / "C01" / "checks")
            if scope_evidence is None:
                refuse("the C01 scope-repair candidate evidence is missing")
            c01_tree = scope_evidence.staged_tree_sha
        if checkpoint.phase in scope_c01_phases and resumed.scope_violation_recovery is not None:
            authority, refusal = self._scope_repair_authority_tree(
                run_dir, resumed, cycle=1,
                inherited_check_ids=resumed.plan.required_checks,
            )
            if refusal is not None:
                refuse(refusal)
            if expected != authority:
                refuse("the checkpoint tree is not the C01 scope-repair chain tree")
        elif at <= phase_index(ResumePhase.REVIEWER_C01) and expected != c01_tree:
            refuse(
                "the checkpoint tree is not the Claude C01 tree" if resumed.c01_revision is not None
                else "the checkpoint tree is not the last completed Luna tree"
            )
        if checkpoint.phase is ResumePhase.INITIAL_STEP:
            return
        if generic_c01 and checkpoint.phase in {
            ResumePhase.CHECK_REPAIR_C01,
            ResumePhase.FINAL_CHECKS_RETRY_C01,
        }:
            if checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
                evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            else:
                evidence = _load_evidence(run_dir / "checks" / "C01")
                if evidence is None and checkpoint.check_repair_attempt is not None:
                    evidence = _load_evidence(
                        run_dir / "checks" / "C01" / "attempts"
                        / f"{checkpoint.check_repair_attempt:02d}"
                    )
            if checkpoint.phase is ResumePhase.CHECK_REPAIR_C01 and (
                evidence is None or evidence.staged_tree_sha != expected
            ):
                refuse("the durable check-repair evidence is missing or not for the checkpoint tree")
            if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_C01 and evidence is None:
                refuse("the durable check-repair attempt evidence is missing")
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase in {
            ResumePhase.CHECKS_C01,
            ResumePhase.CLAUDE_C01, ResumePhase.FINAL_CHECKS_C01,
        }:
            # These three phases are checkpointed *before* the C01 final checks
            # have produced their evidence, so there is usually none on disk --
            # a Claude crash or a crash inside the checks themselves lands
            # here.  The resume re-runs the final checks from
            # ``FINAL_CHECKS_C01`` and never consumes ``c01_evidence``, so a
            # missing bundle is normal and not an integrity failure.  Evidence
            # that *is* present must still be for the tree being resumed.
            evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            if evidence is not None and evidence.staged_tree_sha != c01_tree:
                refuse("the C01 final evidence is not for the expected checks tree")
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
            # The first final-checks pass completed and the repair Claude has
            # not succeeded yet: the canonical bundle is still the red evidence
            # for the pre-repair tree, which is the authority the repair
            # answers to, so it must exist and be for exactly that tree.
            evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            if evidence is None or evidence.staged_tree_sha != initial_c01_tree:
                refuse("the C01 final evidence is missing or not for the expected checks tree")
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase in {
            ResumePhase.CHECK_SCOPE_PLANNER_C01,
            ResumePhase.CHECK_SCOPE_APPROVAL_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C01,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
        } and resumed.scope_violation_recovery is not None:
            evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            if evidence is None or evidence.staged_tree_sha != initial_c01_tree:
                refuse("the C01 scope-repair evidence is missing or not for the rolled-back tree")
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_C01:
            # The repair Claude did succeed, so ``c01_tree`` is the repaired
            # tree and the canonical bundle is either absent (the first retry
            # crashed) or the retry's own red bundle for that repaired tree.
            evidence, refusal = _retry_checks_evidence(
                run_dir / "checks" / "C01", cycle="C01",
                initial_tree=initial_c01_tree, repaired_tree=c01_tree,
                legacy_dir=run_dir,
            )
            if refusal is not None:
                refuse(refusal)
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.CHECK_REPAIR_EXPANDED_C01:
            evidence = _load_evidence(run_dir / "checks" / "C01")
            if evidence is None or evidence.staged_tree_sha != initial_c01_tree:
                refuse("the C01 expanded repair evidence is missing or not for the red retry tree")
            resumed.c01_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01:
            current = _load_evidence(run_dir / "checks" / "C01")
            if current is not None and current.staged_tree_sha != c01_tree:
                refuse("the C01 expanded retry evidence is not for the repaired tree")
            if current is None:
                current = _load_evidence(run_dir / "checks" / "C01" / "attempts" / "02")
                if current is None or current.staged_tree_sha != initial_c01_tree:
                    refuse("the C01 expanded retry red evidence is missing")
            resumed.c01_evidence = current
            return
        if at <= phase_index(ResumePhase.CANDIDATE_COMMIT_C01):
            scope_evidence = (
                _load_evidence(run_dir / "scope-repair" / "C01" / "checks")
                if resumed.scope_violation_recovery is not None else None
            )
            if scope_evidence is not None:
                if scope_evidence.staged_tree_sha != c01_tree:
                    refuse("the C01 scope-repair evidence is not for the candidate tree")
                resumed.c01_evidence = scope_evidence
                return
            evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            if evidence is None or evidence.staged_tree_sha != c01_tree:
                refuse("the C01 candidate evidence is missing or not for the candidate tree")
            resumed.c01_evidence = evidence
            return
        evidence = (
            _load_evidence(run_dir / "scope-repair" / "C01" / "checks")
            if resumed.scope_violation_recovery is not None else None
        ) or _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
        if evidence is None or evidence.staged_tree_sha != c01_tree:
            refuse("the C01 candidate evidence is missing or not for the candidate tree")
        resumed.c01_evidence = evidence
        if at <= phase_index(ResumePhase.REVIEWER_C01):
            resumed.c01_review = _load_c01_review(run_dir, evidence)
            if checkpoint.phase is ResumePhase.REVIEWER_C01 and resumed.c01_review is None:
                # A transport failure intentionally has no accepted review;
                # the reviewer is retried with the already pushed commit.
                pass
            return
        if checkpoint.phase is ResumePhase.COMMIT and not repair_enabled:
            evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
            if evidence is None or evidence.staged_tree_sha != expected:
                refuse("the C01 commit evidence is missing or not for the approved tree")
            review = _load_c01_review(run_dir, evidence)
            if review is None or review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
                refuse("the C01 reviewer PASS is missing for commit")
            resumed.c01_evidence, resumed.c01_review = evidence, review
            return
        evidence = _load_evidence(run_dir / "checks" / "C01") or _load_evidence(run_dir)
        if evidence is None or evidence.staged_tree_sha != c01_tree:
            refuse("the C01 final evidence is missing or not for the C01 tree")
        if checkpoint.phase is ResumePhase.REPAIR_PLANNER:
            # The repair planner starts from the committed, pushed and
            # reviewed C01 candidate: only reviewer #1's accepted answer for
            # exactly that commit and tree may authorize C02.
            if expected != evidence.staged_tree_sha:
                refuse("the checkpoint tree is not the C01 reviewed tree")
            c01_record = _read_json_artifact(_candidate_commit_path(run_dir, 1))
            c01_sha = c01_record.get("commit_sha") if isinstance(c01_record, dict) else None
            if not _is_object_id(c01_sha) or checkpoint.expected_head_sha != c01_sha:
                refuse("the repair planner checkpoint is not the C01 candidate commit")
            review = _load_accepted_c01_review(run_dir, evidence, c01_sha)
        else:
            review = _load_c01_review(run_dir, evidence)
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route not in {
            ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN,
        }:
            refuse("reviewer #1 did not route an implementation repair")
        resumed.c01_evidence, resumed.c01_review = evidence, review
        if checkpoint.phase is ResumePhase.REPAIR_PLANNER:
            return
        repair_dir = run_dir / "repair" / "C02"
        selection = resumed.selection
        try:
            repair_plan = parse_task_plan_v2(
                (repair_dir / "planner.raw.md").read_text(encoding="utf-8"),
                implementer_ids=frozenset({selection.repair_implementer.profile_id}),
                reviewer_ids=frozenset({selection.reviewer.profile_id}),
                check_catalog=self.config.check_catalog,
                inherited_check_ids=resumed.plan.required_checks,
            )
            repair_bundle, repair_sha = validate_implementation_bundle(
                repair_dir, expected_step_ids=[step.id for step in repair_plan.steps]
            )
        except (V2PlanParseError, OSError, UnicodeError, AttributeError) as exc:
            refuse(f"the C02 repair plan is unreadable: {exc}")
        if repair_plan.decision is not PlanDecision.READY or (
            resumed.scope_violation_recovery is None
            and repair_sha != checkpoint.repair_bundle_sha256
        ):
            refuse("the C02 repair bundle changed")
        resumed.repair_plan, resumed.repair_bundle, resumed.repair_bundle_sha = repair_plan, repair_bundle, repair_sha
        if checkpoint.phase is ResumePhase.SCOPE_APPROVAL:
            delta = _read_json_artifact(repair_dir / "scope_delta.json", 256 * 1024)
            if (not isinstance(delta, dict) or checkpoint.scope_delta_sha256 is None
                    or hashlib.sha256((repair_dir / "scope_delta.json").read_bytes()).hexdigest()
                    != checkpoint.scope_delta_sha256):
                refuse("the C02 scope delta changed")
            return
        repair_ids = [step.id for step in repair_plan.steps]
        if checkpoint.phase is ResumePhase.REPAIR_STEP:
            if checkpoint.step_id not in repair_ids:
                refuse("checkpoint step is not in the repair plan")
            repair_prior = repair_ids[: repair_ids.index(checkpoint.step_id)]
        else:
            repair_prior = repair_ids
        repair_records = [
            _load_completed_step(repair_dir / "steps" / step_id, step_id) for step_id in repair_prior
        ]
        if any(record is None for record in repair_records):
            refuse("a completed C02 step record is missing or invalid")
        # Same two identities for C02: the last repair Luna tree, then the
        # Claude C02 record's tree_after once Claude C02 succeeded.
        repair_end = _verify_step_chain(repair_records, evidence.staged_tree_sha)
        if repair_end is None:
            refuse("C02 step trees do not form an unbroken chain")
        resumed.c02_steps = repair_records
        c02_tree = repair_end
        if revision_enabled and at >= phase_index(ResumePhase.FINAL_CHECKS_C02):
            revision = _load_revision(run_dir / "revision" / "C02")
            if revision is None or revision.tree_before != repair_end:
                refuse("the Claude C02 record is missing or not based on the C02 Luna tree")
            resumed.c02_revision = revision
            c02_tree = revision.tree_after
        initial_c02_tree = c02_tree
        check_repair_dir_c02 = run_dir / "revision" / "check-repair" / "C02"
        generic_c02_attempts = self._check_repair_attempt_records(run_dir, 2)
        generic_c02 = bool(generic_c02_attempts) or checkpoint.check_repair_attempt is not None
        if generic_c02:
            c02_tree = self._check_repair_chain_tree(
                run_dir, 2, initial_c02_tree, checkpoint,
            )
        if (
            not generic_c02 and revision_enabled
            and resumed.scope_violation_recovery is None
            and at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_C02) and (
            check_repair_dir_c02 / "report.json"
            ).exists()
        ):
            repair_revision = _load_revision(
                check_repair_dir_c02
            )
            if repair_revision is None or repair_revision.tree_before != initial_c02_tree:
                refuse("the C02 check-repair record is missing or not based on the red checks tree")
            resumed.c02_check_repair_revision = repair_revision
            c02_tree = repair_revision.tree_after
        expanded_c02_dir = run_dir / "revision" / "check-repair-expanded" / "C02"
        if revision_enabled and resumed.scope_violation_recovery is None and at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02) and (
            checkpoint.phase in {
                ResumePhase.CHECK_REPAIR_EXPANDED_C02,
                ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02,
            }
            or expanded_c02_dir.joinpath("scope.json").is_file()
        ):
            normal_scope = _read_check_repair_scope(
                check_repair_dir_c02,
                fallback_base=sorted({
                    path for step in repair_plan.steps
                    for path in (*step.write_set, *step.create_set, *step.delete_set)
                }),
                policy_config=self._effective_repair_scope,
            )
            _validate_expanded_check_repair_scope(
                expanded_c02_dir,
                repo=resumed.info.source_repo,
                tree_sha=c02_tree,
                normal_scope=normal_scope,
                policy_config=self._effective_repair_scope,
            )
            if at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02):
                expanded_revision = _load_revision(expanded_c02_dir)
                if expanded_revision is None or expanded_revision.tree_before != c02_tree:
                    refuse("the expanded C02 check-repair record is missing or not based on the red checks tree")
                resumed.c02_expanded_check_repair_revision = expanded_revision
                c02_tree = expanded_revision.tree_after
        if resumed.scope_violation_recovery is not None and checkpoint.phase in {
            ResumePhase.CANDIDATE_COMMIT_C02, ResumePhase.CANDIDATE_PUSH_C02,
            ResumePhase.REVIEWER_C02,
        }:
            scope_evidence = _load_evidence(run_dir / "scope-repair" / "C02" / "checks")
            if scope_evidence is None:
                refuse("the C02 scope-repair candidate evidence is missing")
            c02_tree = scope_evidence.staged_tree_sha
        if checkpoint.phase in scope_c02_phases and resumed.scope_violation_recovery is not None:
            authority, refusal = self._scope_repair_authority_tree(
                run_dir, resumed, cycle=2,
                inherited_check_ids=repair_plan.required_checks,
            )
            if refusal is not None:
                refuse(refusal)
            if expected != authority:
                refuse("the checkpoint tree is not the C02 scope-repair chain tree")
        elif expected != c02_tree:
            refuse(
                "the checkpoint tree is not the Claude C02 tree" if resumed.c02_revision is not None
                else "the checkpoint tree is not the last completed C02 tree"
            )
        if checkpoint.phase in scope_c02_phases and resumed.scope_violation_recovery is not None:
            evidence = _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != c02_tree:
                refuse("the C02 scope-repair evidence is missing or not for the rolled-back tree")
            resumed.c02_evidence = evidence
            return
        if generic_c02 and checkpoint.phase in {
            ResumePhase.CHECK_REPAIR_C02,
            ResumePhase.FINAL_CHECKS_RETRY_C02,
        }:
            if checkpoint.phase is ResumePhase.CHECK_REPAIR_C02:
                evidence = _load_evidence(run_dir / "checks" / "C02")
            else:
                evidence = _load_evidence(run_dir / "checks" / "C02")
                if evidence is None and checkpoint.check_repair_attempt is not None:
                    evidence = _load_evidence(
                        run_dir / "checks" / "C02" / "attempts"
                        / f"{checkpoint.check_repair_attempt:02d}"
                    )
            if checkpoint.phase is ResumePhase.CHECK_REPAIR_C02 and (
                evidence is None or evidence.staged_tree_sha != expected
            ):
                refuse("the durable C02 check-repair evidence is missing or not for the checkpoint tree")
            if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_C02 and evidence is None:
                refuse("the durable C02 check-repair attempt evidence is missing")
            resumed.c02_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.CHECK_REPAIR_C02:
            # C02 parity with ``CHECK_REPAIR_C01``: the red pre-repair bundle
            # is mandatory and must be for the pre-repair C02 tree.
            evidence = _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != initial_c02_tree:
                refuse("the C02 final evidence is missing or not for the expected checks tree")
            resumed.c02_evidence = evidence
            return
        if checkpoint.phase in {
            ResumePhase.CHECK_SCOPE_PLANNER_C02,
            ResumePhase.CHECK_SCOPE_APPROVAL_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CHECKS_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_CLAUDE_C02,
            ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
        } and resumed.scope_violation_recovery is not None:
            evidence = _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != initial_c02_tree:
                refuse("the C02 scope-repair evidence is missing or not for the rolled-back tree")
            resumed.c02_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_C02:
            # C02 parity with ``FINAL_CHECKS_RETRY_C01``: the checkpoint tree
            # is the repaired C02 tree and the current bundle, when present,
            # must be the retry's own bundle for it.
            evidence, refusal = _retry_checks_evidence(
                run_dir / "checks" / "C02", cycle="C02",
                initial_tree=initial_c02_tree, repaired_tree=c02_tree,
            )
            if refusal is not None:
                refuse(refusal)
            resumed.c02_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.CHECK_REPAIR_EXPANDED_C02:
            evidence = _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != initial_c02_tree:
                refuse("the C02 expanded repair evidence is missing or not for the red retry tree")
            resumed.c02_evidence = evidence
            return
        if checkpoint.phase is ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02:
            current = _load_evidence(run_dir / "checks" / "C02")
            if current is not None and current.staged_tree_sha != c02_tree:
                refuse("the C02 expanded retry evidence is not for the repaired tree")
            if current is None:
                current = _load_evidence(run_dir / "checks" / "C02" / "attempts" / "02")
                if current is None or current.staged_tree_sha != initial_c02_tree:
                    refuse("the C02 expanded retry red evidence is missing")
            resumed.c02_evidence = current
            return
        if checkpoint.phase in {
            ResumePhase.CANDIDATE_COMMIT_C02, ResumePhase.CANDIDATE_PUSH_C02,
            ResumePhase.REVIEWER_C02,
        }:
            evidence = (
                _load_evidence(run_dir / "scope-repair" / "C02" / "checks")
                if resumed.scope_violation_recovery is not None else None
            ) or _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != expected:
                refuse("the C02 candidate evidence is missing or not for the candidate tree")
            resumed.c02_evidence = evidence
            if checkpoint.phase is ResumePhase.REVIEWER_C02:
                resumed.c02_review = _load_c01_review(run_dir / "review" / "C02", evidence)
            return
        if checkpoint.phase is ResumePhase.COMMIT:
            evidence = _load_evidence(run_dir / "checks" / "C02")
            if evidence is None or evidence.staged_tree_sha != expected:
                refuse("the C02 commit evidence is missing or not for the approved tree")
            review = _load_c01_review(run_dir / "review" / "C02", evidence)
            if review is None or review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
                refuse("the C02 reviewer PASS is missing for commit")
            resumed.c02_evidence, resumed.c02_review = evidence, review

    def _commit_body_v2(self, run_id: str, base_sha: str, tree_sha: str,
                        evidence: EvidenceBundle, selection: ExecutionSelectionV3) -> str:
        return f"MetaHarness-Run: {run_id}"

    def _redact_agent_artifacts(self, run_dir: Path) -> None:
        for name in _AGENT_ARTIFACTS:
            redact_file(run_dir / name, self._secrets)

    def _redact_revision_artifacts(
        self, run_dir: Path, *, revision_dir: Path | None = None
    ) -> None:
        directory = revision_dir or (run_dir / "revision")
        for name in _REVISION_ARTIFACTS:
            redact_file(directory / name, self._secrets)

    def _commit_body(
        self,
        run_id: str,
        base_sha: str,
        tree_sha: str,
        evidence: EvidenceBundle,
    ) -> str:
        return f"MetaHarness-Run: {run_id}"


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


def resume_run(
    config: HarnessConfig | str | Path, run_id: str, *,
    revalidate_integrity: bool = False,
) -> RunResult:
    """Resume *run_id* at its durable checkpoint (same run id, same worktree)."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).resume(run_id, revalidate_integrity=revalidate_integrity)


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
    "authorize_commit",
    "generate_run_id",
    "recover_plan_run",
    "resume_run",
    "run_orchestrator",
]
