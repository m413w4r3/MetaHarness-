"""The single-task, single-agent MetaHarness V0 state machine."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, NoReturn, Sequence

from .agent.base import AgentError, AgentResult
from .agent.diagnostics import TOKEN_DIAGNOSTICS_NAME, write_token_diagnostics
from .agent.codex import (
    AgentCommittedError,
    CodexAgent,
    build_agent_environment,
    build_implementer_step_prompt,
    build_mismatch_retry_addendum,
    classify_codex_failure,
    contract_mismatch_explanation,
    deferred_verify_dependency,
)
from .agent.runtime import prepare_codex_home
from .claude.agent import (
    ClaudeAgentError,
    ClaudeCodeAgent,
    ClaudeCommittedError,
    ScopeRequest,
    build_claude_environment,
    build_revision_prompt,
    classify_claude_failure,
    parse_scope_request,
)
from .claude.runtime import ClaudeRuntimeError, prepare_claude_home
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
    repository_remote_url,
    run_branch_web_url,
    tracked_files_in_tree,
    validate_run_branch,
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
)
from .models import (
    ExecutionRole,
    ExecutionSelectionV4,
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
    build_claude_profile,
    build_llm_endpoint,
    profile_for_role,
    profiles_for_config,
)
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


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """A commit precondition does not hold immediately before the commit."""


class ScopeApprovalRequired(OrchestrationError):
    """A scope expansion is durably paused until its exact delta is approved."""

    code = "WAITING_SCOPE_APPROVAL"


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


# Gate failures for which a semantic review is pointless or unsafe: the
# candidate is empty, unreviewable, not the agent's output, or leaks a secret.
_DIRECT_FAILURES = frozenset(
    {
        "EMPTY_DIFF",
        "HEAD_MISMATCH",
        SECRET_IN_DIFF,
        SECRET_IN_STAGED_BLOB,
        UNSCANNABLE_STAGED_BLOB,
        UNREVIEWABLE_TEXT_DIFF,
    }
)
_LEGACY_DIRECT_FAILURES = _DIRECT_FAILURES | frozenset({DIFF_TOO_LARGE})
_COMMIT_SUBJECT_LIMIT = 72
_MAX_AGENT_REPORT_BYTES = 32_000
_MAX_STEP_REPORT_BYTES = 2_048
_MAX_REVIEW_FALLBACK_DIFF_BYTES = 32 * 1024
_MAX_REVIEW_CONTEXT_BYTES = 24 * 1024
_MAX_REPAIR_CLAUDE_REPORT_BYTES = 16 * 1024
_SCOPE_REQUEST_HEADER = "META SCOPE REQUEST v1"
_SCOPE_REQUEST_ROUTE = "CLAUDE_SCOPE_REQUEST"
_AGENT_ARTIFACTS = (
    "agent.events.jsonl",
    "agent.stderr.log",
    "agent.final.md",
    "agent.result.json",
)
_REVISION_ARTIFACTS = (
    "agent.prompt.txt",
    "agent.events.jsonl",
    "agent.stderr.log",
    "agent.final.md",
    "agent.result.json",
)


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


def _slug(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9]+", "-", value.casefold()).strip("-")
    return (candidate[:60] or "task")


def _commit_subject(title: str) -> str:
    """One Git subject line of at most 72 characters from the plan title."""

    first = next((line for line in title.splitlines() if line.strip()), "")
    subject = " ".join(first.split()).strip("#*_` ") or "MetaHarness change"
    if len(subject) > _COMMIT_SUBJECT_LIMIT:
        subject = subject[: _COMMIT_SUBJECT_LIMIT - 3].rstrip() + "..."
    return subject


def _bounded_report(text: str) -> str:
    """Bound the non-authoritative agent report sent to the reviewer."""

    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_AGENT_REPORT_BYTES:
        return text
    head = encoded[:_MAX_AGENT_REPORT_BYTES].decode("utf-8", errors="ignore")
    omitted = len(encoded) - _MAX_AGENT_REPORT_BYTES
    return f"{head}\n[... {omitted} bytes truncated; full report in agent.final.md ...]"


def _bounded_repair_claude_report(text: str) -> str:
    """Bound the advisory Claude C01 report handed to the repair planner.

    The report is consultative; the immutable candidate commit is the code
    authority, so truncating it can never hide a fact the planner must know.
    """

    data = text.encode("utf-8", errors="replace")

    if len(data) <= _MAX_REPAIR_CLAUDE_REPORT_BYTES:
        return text

    marker = (
        "\n[... Claude report truncated for repair planner; "
        "candidate commit is authoritative ...]\n"
    ).encode("utf-8")

    head = data[
        : max(
            0,
            _MAX_REPAIR_CLAUDE_REPORT_BYTES - len(marker),
        )
    ].decode("utf-8", errors="ignore")

    return head + marker.decode("utf-8")


def _scope_request_payload(request: ScopeRequest) -> dict[str, Any]:
    return {
        "reason": request.reason,
        "paths": list(request.paths),
        "evidence": list(request.evidence),
        "authoritative": False,
        "routed_to_bridge_audit": True,
    }


def _scope_request_evidence(request: ScopeRequest | None) -> str:
    if request is None:
        return "NONE\n"
    return "\n".join([
        "REASON",
        request.reason,
        "",
        "PATHS",
        *(f"- {path}" for path in request.paths),
        "",
        "EVIDENCE",
        *(f"- {item}" for item in request.evidence),
        "",
    ])


def _scope_request_diagnostic(request: ScopeRequest) -> str:
    return "\n".join([
        "Claude requested scope expansion:",
        f"  paths: {len(request.paths)}",
        "  authoritative: NO",
        "  routed to bridge audit: YES",
    ])


def _scope_request_from_payload(payload: Any) -> ScopeRequest | None:
    if not isinstance(payload, Mapping):
        return None
    reason = payload.get("reason")
    paths = payload.get("paths")
    evidence = payload.get("evidence")
    if (
        not isinstance(reason, str)
        or not isinstance(paths, list)
        or not isinstance(evidence, list)
        or any(not isinstance(path, str) for path in paths)
        or any(not isinstance(item, str) for item in evidence)
    ):
        return None
    return ScopeRequest(reason=reason, paths=tuple(paths), evidence=tuple(evidence))


def _repair_checks_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    """Summarize the accepted C01 deterministic gate for the repair planner.

    Reviewer #1 only exists because the C01 gate was accepted, so argv, cwd,
    durations and log tails add no decision value here; they stay in the
    durable check artifacts.
    """

    checks: list[dict[str, Any]] = []

    for check in bundle.checks:
        payload = (
            dict(check)
            if isinstance(check, Mapping)
            else check_result_json(check)
        )

        checks.append(
            {
                "name": payload.get("name"),
                "exit_code": payload.get("exit_code"),
                "timed_out": bool(payload.get("timed_out", False)),
                "workspace_mutated": bool(
                    payload.get("workspace_mutated", False)
                ),
            }
        )

    return {
        "deterministic_passed": bundle.deterministic_passed,
        "failures": list(bundle.failures),
        "checks": checks,
    }


def _bounded_v2_report(text: str) -> str:
    """Bound an individual staged-step report for every semantic prompt."""

    limit = _MAX_STEP_REPORT_BYTES
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    marker = b"\n[... report truncated ...]"
    head = data[: max(0, limit - len(marker))].decode("utf-8", errors="ignore")
    return head + marker.decode()


def _step_reports_text(results: list[dict[str, Any]]) -> str:
    """Render every step with bounded metadata and a bounded report body."""

    chunks: list[str] = []
    for item in results:
        chunk = "\n".join([
            item["id"], f"status: {item.get('status', 'COMPLETED')}",
            f"profile: {item['profile_id']}",
            f"tree_before: {item['tree_before']}", f"tree_after: {item['tree_after']}",
            f"usage: {json.dumps(item['usage'], sort_keys=True)}", "final report:",
            _bounded_v2_report(item.get("final", "")),
        ]) + "\n"
        chunks.append(chunk)
    return "\n".join(chunks)


def _revision_execution_anomalies(results: list[dict[str, Any]]) -> str:
    """Render only exceptional Luna execution facts for Claude.

    The resulting worktree is the authority for successful implementation
    details.  Reports are retained for reviewer/audit flows, but normal Luna
    narration, usage and tree metadata do not belong in Claude's task prompt.
    """

    records: list[dict[str, Any]] = []
    for item in results:
        status = item.get("status", "COMPLETED")
        exceptional = (
            status != "COMPLETED"
            or bool(item.get("mismatch"))
            or bool(item.get("initial_mismatch"))
            or bool(item.get("deferred_verify"))
            or bool(item.get("mismatch_retry_count"))
        )
        if not exceptional:
            continue

        record: dict[str, Any] = {
            "id": item.get("id"),
            "status": status,
            "changed_paths": list(item.get("changed_paths") or []),
        }
        for key in ("mismatch", "initial_mismatch", "deferred_verify"):
            if item.get(key):
                record[key] = _bounded_v2_report(str(item[key]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        records.append(record)

    return _json_text(records) if records else "NONE\n"


def _revision_contract_index(plan: TaskPlanV2) -> str:
    """Render only the approved behavioral and mutation contract index."""

    return _json_text([
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
            "verify": step.verify,
            "forbidden": step.forbidden,
        }
        for step in plan.steps
    ])


def _revision_plan_summary(plan: TaskPlanV2) -> str:
    return _json_text({
        "title": plan.title,
        "objective": plan.objective,
        "constraints": plan.constraints,
        "acceptance": plan.acceptance,
        "tests": plan.tests,
        "risks": plan.risks,
    })


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


def _deferred_contract_mismatches(
    plan: TaskPlanV2, results: list[dict[str, Any]],
) -> str:
    """Render bounded summaries for Claude and the independent reviewer."""

    steps = {step.id: step for step in plan.steps}
    summaries: list[dict[str, Any]] = []
    for item in results:
        deferred = item.get("status") == "DEFERRED_CONTRACT_MISMATCH"
        verify = _bounded_v2_report(str(item.get("deferred_verify") or ""))
        if not deferred and not verify:
            continue
        step = steps.get(item.get("id"))
        if step is None:
            continue
        scope = {
            "write": list(step.write_set),
            "create": list(step.create_set),
            "delete": list(step.delete_set),
        }
        record: dict[str, Any] = {
            "step_id": step.id,
            "step_title": step.title,
            "kind": (
                "DEFERRED_CONTRACT_MISMATCH" if deferred
                else "DEFERRED_VERIFY_DEPENDENCY"
            ),
            "original_scope": scope,
        }
        if deferred:
            record["mismatch"] = _bounded_v2_report(str(item.get("mismatch") or ""))
            record["tree_at_mismatch"] = item.get("tree_before")
        if item.get("initial_mismatch"):
            record["initial_mismatch"] = _bounded_v2_report(str(item["initial_mismatch"]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        if verify:
            # The step completed inside its approved scope but one VERIFY
            # command still fails on a path a later step owns.  Claude and
            # the reviewer must decide; nothing here accepts that failure.
            record["deferred_verify_dependency"] = verify
        summaries.append(record)
    return _json_text(summaries) if summaries else "NONE\n"


def _future_step_ownership(
    steps: Sequence[ImplementationStep], index: int,
) -> dict[str, tuple[str, ...]]:
    """The mutation paths the approved plan assigns to the remaining steps.

    Informative only: a bounded retry uses it to recognize an out-of-scope
    verification dependency.  It never grants write authority.
    """

    ownership: dict[str, tuple[str, ...]] = {}
    for step in steps[index + 1:]:
        paths = tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))
        if paths:
            ownership[step.id] = paths
    return ownership


def _has_deferred_contract_mismatches(results: list[dict[str, Any]]) -> bool:
    return any(item.get("status") == "DEFERRED_CONTRACT_MISMATCH" for item in results)


def _compact_step_history(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep C02 history structural; reports live in ``luna_reports``."""

    compact: list[dict[str, Any]] = []
    for item in results:
        record = {
            "id": item.get("id"),
            "profile_id": item.get("profile_id"),
            "tree_after": item.get("tree_after"),
            "changed_paths": item.get("changed_paths", []),
            "usage": item.get("usage", {}),
        }
        if item.get("status") == "DEFERRED_CONTRACT_MISMATCH":
            record["status"] = item["status"]
            record["mismatch"] = _bounded_v2_report(str(item.get("mismatch") or ""))
        if item.get("deferred_verify"):
            record["deferred_verify"] = _bounded_v2_report(str(item["deferred_verify"]))
        compact.append(record)
    return compact


def _check_payload(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    # A bundle rebuilt from ``evidence.json`` on resume carries the persisted
    # reviewer-safe payloads instead of CheckResult objects.
    payload: list[dict[str, Any]] = []
    for check in bundle.checks:
        item = dict(check) if isinstance(check, Mapping) else check_result_json(check)
        if bundle.required_check_ids:
            item["required"] = item.get("name") in bundle.required_check_ids
        payload.append(item)
    return payload


_REVISION_CHECK_LOG_BYTES = 16 * 1024


def _revision_check_context(payload: Mapping[str, Any]) -> str:
    """Render compact pre-revision check state for Claude.

    Successful checks contribute status only.  Output tails are decision
    evidence only for failed checks and stay bounded for prompt safety; the
    complete logs remain in the durable check artifacts.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("check payload must be a mapping")

    failures = [
        item for item in payload.get("failures", [])
        if isinstance(item, str)
    ]
    failed_names = {
        item.split(":", 1)[1]
        for item in failures
        if item.startswith("CHECK_FAILED:")
    }
    checks: list[dict[str, Any]] = []
    raw_checks = payload.get("checks", [])
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raw_checks = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, Mapping):
            continue
        name = raw_check.get("name")
        failed = name in failed_names
        check: dict[str, Any] = {
            "name": name,
            "required": bool(raw_check.get("required", False)),
            "exit_code": raw_check.get("exit_code"),
            "timed_out": bool(raw_check.get("timed_out", False)),
            "workspace_mutated": bool(raw_check.get("workspace_mutated", False)),
        }
        if failed:
            for key in ("stdout_tail", "stderr_tail"):
                value = raw_check.get(key)
                if isinstance(value, str) and value:
                    data = value.encode("utf-8", errors="replace")
                    if len(data) > _REVISION_CHECK_LOG_BYTES:
                        data = data[-_REVISION_CHECK_LOG_BYTES:]
                    check[key] = data.decode("utf-8", errors="replace")
        checks.append(check)

    return _json_text({
        "deterministic_passed": bool(payload.get("deterministic_passed", False)),
        "failure_ids": failures,
        "checks": checks,
    })


def _hard_integrity_failures(bundle: EvidenceBundle) -> list[str]:
    """Return failures that make semantic review unsafe.

    A normal configured check failure is evidence for the reviewer in P25;
    mutations, timeouts, secrets, ownership and malformed/oversized trees are
    still terminal integrity failures.
    """

    return _hard_failure_items(bundle.failures)


def _soft_check_failures(bundle: EvidenceBundle) -> list[str]:
    """Return ordinary deterministic check failures eligible for one repair.

    This deliberately accepts only the exact ``CHECK_FAILED:<name>`` family.
    Anything else, including a future failure category, fails closed and must
    never be handed to the corrective Claude pass.
    """

    if _hard_integrity_failures(bundle) or bundle.deterministic_passed:
        return []
    failures = list(bundle.failures)
    if not failures or any(
        not isinstance(item, str) or not item.startswith("CHECK_FAILED:")
        for item in failures
    ):
        return []
    return failures


_MAX_CHECK_SCOPE_LOG_BYTES = 256 * 1024
_CHECK_SCOPE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:file://)?"
    r"(?:/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"
    r"\.(?:py|pyi|ts|tsx|js|jsx|java|go|rs|rb|php|c|cc|cpp|h|hpp|json|yaml|yml|toml|ini)"
    r"(?::[0-9]+(?::[0-9]+)?)?(?![A-Za-z0-9_.-])"
)


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_CHECK_SCOPE_LOG_BYTES))
            return stream.read(_MAX_CHECK_SCOPE_LOG_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _failed_check_text(
    *,
    run_dir: Path,
    evidence: EvidenceBundle,
) -> str:
    """Return only bounded output belonging to failed ordinary checks."""

    failed_names = {
        failure.split(":", 1)[1]
        for failure in _soft_check_failures(evidence)
        if ":" in failure
    }
    root = run_dir.resolve()
    chunks: list[str] = []
    for check in evidence.checks:
        payload = dict(check) if isinstance(check, Mapping) else check_result_json(check)
        if payload.get("name") not in failed_names:
            continue
        if not isinstance(check, Mapping):
            for key in ("stdout_log", "stderr_log"):
                value = getattr(check, key, None)
                if isinstance(value, str):
                    chunks.append(value[-_MAX_CHECK_SCOPE_LOG_BYTES:])
        for key in ("stdout_tail", "stderr_tail"):
            value = payload.get(key)
            if isinstance(value, str):
                chunks.append(value)
        for key in ("stdout_log_path", "stderr_log_path"):
            raw_path = payload.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                candidate = (root / raw_path).resolve()
                candidate.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            chunks.append(_read_log_tail(candidate))
    return "\n".join(chunk for chunk in chunks if chunk)


def _resolve_check_path_candidate(
    raw_path: str,
    *,
    worktree: Path,
    tracked_files: frozenset[str],
) -> str | None:
    """Normalize one path-shaped check-output token without guessing."""

    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = raw_path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
        path = path[1:-1]
    if path.startswith("file://"):
        path = path[7:]
    path = re.sub(r":\d+(?::\d+)?$", "", path)
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
        path = path[1:-1]
    if not path or "\x00" in path or "\\" in path:
        return None
    try:
        candidate_path = Path(path)
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve()
            resolved.relative_to(worktree.resolve())
            candidate = resolved.relative_to(worktree.resolve()).as_posix()
        else:
            if ".." in candidate_path.parts:
                return None
            candidate = candidate_path.as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate in tracked_files:
        return candidate
    matches = sorted(
        tracked for tracked in tracked_files
        if tracked.endswith("/" + candidate)
    )
    return matches[0] if len(matches) == 1 else None


def _is_auto_expandable_test_path(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    parts = PurePosixPath(path).parts
    forbidden = {
        ".git", ".venv", "venv", "node_modules", "dist", "build", "coverage",
        "test-results", "playwright-report",
    }
    if any(part in forbidden for part in parts):
        return False
    basename = parts[-1] if parts else ""
    return (
        "tests" in parts
        or "__tests__" in parts
        or (basename.startswith("test_") and basename.endswith((".py", ".pyi")))
        or basename.endswith((
            ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
            ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
        ))
    )


def _check_repair_scope_candidates(
    *,
    repo: Path,
    worktree: Path,
    tree_sha: str,
    run_dir: Path,
    evidence: EvidenceBundle,
    base_mutable_scope: Sequence[str],
) -> list[str]:
    """Find tracked test paths named by failing-check evidence only."""

    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    text = _failed_check_text(run_dir=run_dir, evidence=evidence)
    resolved = [
        _resolve_check_path_candidate(
            match.group(0), worktree=worktree, tracked_files=tracked
        )
        for match in _CHECK_SCOPE_PATH_RE.finditer(text)
    ]
    base = set(base_mutable_scope)
    return sorted({
        path for path in resolved
        if path is not None
        and path not in base
        and _is_auto_expandable_test_path(path)
    })


def _hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if item in _DIRECT_FAILURES
        or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
        or item.startswith("CHECK_MUTATED:")
        or item.startswith("CHECK_TIMEOUT:")
    ]


def _persist_revision_tree(run_dir: Path, name: str, tree: str) -> None:
    atomic_write_text(run_dir / "revision" / name, tree.rstrip() + "\n")


def _render_revision_template(
    template: str, values: Mapping[str, str], *, name: str,
) -> str:
    """Render one Claude template with one non-recursive substitution pass."""

    expected = set(re.findall(r"\{\{[A-Z0-9_]+\}\}", template))
    missing = expected.difference(values)
    if missing:
        raise OrchestrationError(
            f"{name} template contains unresolved placeholders: "
            + ", ".join(sorted(missing))
        )
    return re.sub(
        r"\{\{[A-Z0-9_]+\}\}",
        lambda match: values[match.group(0)],
        template,
    )


def _revision_prompt(
    *,
    repository_reference: RepositoryReference,
    spec: str,
    plan: TaskPlanV2,
    changed_files: str,
    execution_anomalies: str,
    pre_checks: str,
    mutable_scope: str,
    deferred_mismatches: str,
) -> str:
    template = (Path(__file__).with_name("prompts") / "reviser.txt").read_text(encoding="utf-8")
    values: dict[str, str] = {
        "{{REPOSITORY_REFERENCE}}": json.dumps(repository_reference_dict(repository_reference), ensure_ascii=False, indent=2),
        "{{SPEC}}": spec,
        "{{PLAN_SUMMARY}}": _revision_plan_summary(plan),
        "{{APPROVED_CONTRACT_INDEX}}": _revision_contract_index(plan),
        "{{EXECUTION_ANOMALIES}}": execution_anomalies,
        "{{CURRENT_CHANGED_FILES}}": changed_files,
        "{{PRE_REVISION_CHECKS}}": pre_checks,
        "{{APPROVED_MUTABLE_SCOPE}}": mutable_scope,
        "{{DEFERRED_LUNA_CONTRACT_MISMATCHES}}": deferred_mismatches,
    }
    return _render_revision_template(template, values, name="reviser")


def _check_repair_prompt(
    *,
    spec: str,
    plan: TaskPlanV2,
    approved_contract_index: str,
    changed_files: str,
    evidence: EvidenceBundle,
    mutable_scope: list[str],
    previous_report: str,
    added_paths: Sequence[str] = (),
    scope_source: str = "",
) -> str:
    """Build the bounded prompt for one automatic check-repair pass."""

    failed_ids = _soft_check_failures(evidence)
    check_payload = {
        "deterministic_passed": evidence.deterministic_passed,
        "failures": list(evidence.failures),
        "checks": _check_payload(evidence),
    }
    values = {
        "{{SPEC}}": spec,
        "{{PLAN_SUMMARY}}": _revision_plan_summary(plan),
        "{{APPROVED_CONTRACT_INDEX}}": approved_contract_index,
        "{{FAILURE_IDS}}": _json_text(failed_ids),
        "{{CHECK_DETAILS}}": _revision_check_context(check_payload),
        "{{CHANGED_FILES}}": changed_files,
        "{{MUTABLE_SCOPE}}": _json_text(mutable_scope),
        "{{ADDED_PATHS}}": _json_text(list(added_paths) or ["NONE"]),
        "{{SECOND_PASS_NOTE}}": _SAME_SCOPE_RETRY_NOTE
        if scope_source == _SAME_SCOPE_RETRY_SOURCE else "",
        "{{PREVIOUS_REPAIR_REPORT}}": previous_report or "NONE\n",
    }
    template = (Path(__file__).with_name("prompts") / "check_repair.txt").read_text(
        encoding="utf-8"
    )
    return _render_revision_template(template, values, name="check_repair")


_SAME_SCOPE_RETRY_NOTE = """
This is the second and final bounded automatic check-repair pass.

No mutable-scope expansion was required.

Correct the remaining deterministic failures inside the exact existing
mutable scope.

Do not repeat unrelated changes from the previous repair.
"""


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
    # The full report is already persisted by CodexAgent.  State contains only
    # bounded protocol metadata and never an API key or an authorization value.
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "usage": dict(result.usage),
        # Transport errors may contain provider URLs, request IDs, or other
        # infrastructure identifiers.  The complete bounded artifact remains
        # available to the local UI; state keeps a fixed safe marker instead.
        "stderr_tail": (
            f"{provider} authentication failed" if auth_failure else ("" if safe_stderr else result.stderr_tail)
        ),
    }


def _artifact_tail(path: Path, limit: int = 64 * 1024) -> str:
    """Read only the tail needed for deterministic failure classification."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _codex_auth_failure(events_path: Path, stderr: str) -> bool:
    """Fixed-marker auth classification from stderr and one events file."""

    return classify_codex_failure(stderr, _artifact_tail(events_path)) == "CODEX_AUTH_FAILURE"


def _claude_auth_failure(
    run_dir: Path, stderr: str, *, revision_dir: Path | None = None
) -> bool:
    events_path = (revision_dir or (run_dir / "revision")) / "agent.events.jsonl"
    return classify_claude_failure(stderr, _artifact_tail(events_path)) == "CLAUDE_AUTH_FAILURE"


_MAX_REPORTED_PATHS = 20


def _safe_path_label(path: str) -> str:
    """A printable rendering of one repository path for failure details."""

    return "".join(
        character if character.isprintable() else f"\\x{ord(character) & 0xFF:02x}"
        for character in path
    )[:300]


def _paths_detail(paths: list[str]) -> str:
    shown = [_safe_path_label(path) for path in paths[:_MAX_REPORTED_PATHS]]
    extra = len(paths) - len(shown)
    return ",".join(shown) + (f" (+{extra} more)" if extra > 0 else "")


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


def _status_has_unstaged_or_untracked(status: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for line in status:
        if line.startswith("?? "):
            problems.append(f"new untracked file: {line[3:]}")
        elif len(line) >= 2 and line[1] != " ":
            problems.append(f"unstaged change: {line}")
    return problems


@dataclasses.dataclass(frozen=True)
class GitOwnership:
    """Git state the implementation agent is not allowed to change."""

    head_ref: str | None
    head: str
    branches: frozenset[str]
    worktrees: frozenset[str]


# Compatibility alias for the former private name.
_GitOwnership = GitOwnership


def _git_ownership(repo: Path, worktree: Path) -> GitOwnership:
    return GitOwnership(
        head_ref=symbolic_head(worktree),
        head=current_head(worktree),
        branches=local_branches(repo),
        worktrees=registered_worktrees(repo),
    )


def _git_ownership_payload(ownership: GitOwnership) -> dict[str, Any]:
    return {
        "head_ref": ownership.head_ref,
        "head": ownership.head,
        "branches": sorted(ownership.branches),
        "worktrees": sorted(ownership.worktrees),
    }


def _ownership_from_payload(payload: Mapping[str, Any]) -> GitOwnership | None:
    """Rebuild a persisted ownership proof, or ``None`` when it is malformed."""

    head_ref, head = payload.get("head_ref"), payload.get("head")
    branches, worktrees = payload.get("branches"), payload.get("worktrees")
    if (
        (head_ref is not None and not isinstance(head_ref, str))
        or not _is_object_id(head)
        or not isinstance(branches, list) or any(not isinstance(item, str) for item in branches)
        or not isinstance(worktrees, list) or any(not isinstance(item, str) for item in worktrees)
    ):
        return None
    return GitOwnership(
        head_ref=head_ref, head=head,
        branches=frozenset(branches), worktrees=frozenset(worktrees),
    )


def _ownership_violations(
    before: GitOwnership, after: GitOwnership, *, branch_ref: str, base_sha: str
) -> list[str]:
    problems: list[str] = []
    if after.head_ref != branch_ref:
        problems.append(
            f"worktree HEAD switched from {branch_ref} to {after.head_ref or 'a detached HEAD'}"
        )
    if after.head != base_sha:
        problems.append("worktree HEAD commit changed (commit, merge, reset or rewrite)")
    created = sorted(after.branches - before.branches)
    if created:
        problems.append("branch(es) created: " + ", ".join(created))
    deleted = sorted(before.branches - after.branches)
    if deleted:
        problems.append("branch(es) deleted: " + ", ".join(deleted))
    added_worktrees = sorted(after.worktrees - before.worktrees)
    if added_worktrees:
        problems.append("worktree(s) created: " + ", ".join(added_worktrees))
    removed_worktrees = sorted(before.worktrees - after.worktrees)
    if removed_worktrees:
        problems.append("worktree(s) removed: " + ", ".join(removed_worktrees))
    return problems

@dataclasses.dataclass(frozen=True)
class StepExecutionOutcome:
    """The durable result of one successful Codex step (C01 or C02)."""

    step_id: str
    profile_id: str
    tree_before: str
    tree_after: str
    changed_paths: tuple[str, ...]
    usage: dict[str, int]
    final_report: str
    # A verification the worker could not complete because an out-of-scope
    # path owned by a later approved step still fails.  Data only: the
    # deterministic gate and the reviewer remain the authority.
    deferred_verify: str = ""
    mismatch_retry_count: int = 0


class DeferredStepExecutionOutcome:
    """A clean mismatch result without widening the normal outcome schema."""

    status = "DEFERRED_CONTRACT_MISMATCH"

    def __init__(
        self, *, step_id: str, profile_id: str, tree_before: str,
        tree_after: str, changed_paths: tuple[str, ...], usage: dict[str, int],
        final_report: str, mismatch: str, initial_mismatch: str = "",
        mismatch_retry_count: int = 0, deferred_verify: str = "",
    ) -> None:
        self.step_id = step_id
        self.profile_id = profile_id
        self.tree_before = tree_before
        self.tree_after = tree_after
        self.changed_paths = changed_paths
        self.usage = usage
        self.final_report = final_report
        self.mismatch = mismatch
        self.initial_mismatch = initial_mismatch
        self.mismatch_retry_count = mismatch_retry_count
        self.deferred_verify = deferred_verify


_SYNTHETIC_NO_CHANGE_MISMATCH = (
    "Worker completed successfully without producing an in-scope candidate "
    "change. Retry once to distinguish an already-satisfied step from a stale "
    "contract."
)
_BOUNDED_NO_CHANGE_MISMATCH = (
    "No in-scope change remained necessary after bounded retry; the step is "
    "deferred until the contract is revisited."
)


class StepExecutionFailure(OrchestrationError):
    """One Codex step failed a gate; the C01/C02 caller owns the run status."""

    def __init__(
        self,
        reason: str,
        step_id: str,
        detail: str | None = None,
        *,
        profile_id: str | None,
        tree_before: str | None,
        tree_after: str | None = None,
        usage: dict[str, int] | None = None,
        mismatch: str | None = None,
        clean_contract_mismatch: bool = False,
        mismatch_retry_count: int = 0,
        initial_mismatch: str | None = None,
        index_tree_after: str | None = None,
    ) -> None:
        super().__init__(f"{reason}: step={step_id}")
        self.reason = reason
        self.step_id = step_id
        self.detail = detail
        self.profile_id = profile_id
        self.tree_before = tree_before
        self.tree_after = tree_after
        self.usage = usage
        self.mismatch = mismatch
        self.clean_contract_mismatch = clean_contract_mismatch
        self.mismatch_retry_count = mismatch_retry_count
        self.initial_mismatch = initial_mismatch
        self.index_tree_after = index_tree_after


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


@dataclasses.dataclass(frozen=True)
class CheckRepairScope:
    base_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    policy: str
    bound: int
    source: str


# The two provenances a *second* bounded check-repair scope may carry.  They
# are durable artifact values: never rename them, a stored run depends on the
# exact string.
_AUTO_BOUNDED_SOURCE = "auto-bounded failing-test evidence"
_SAME_SCOPE_RETRY_SOURCE = "bounded same-scope retry"
_SECOND_SCOPE_SOURCES = frozenset({_AUTO_BOUNDED_SOURCE, _SAME_SCOPE_RETRY_SOURCE})
# Historically named "expanded": these are the phases of the *second* bounded
# check-repair pass, whether or not it expands the mutable scope.  The names
# are durable and are never renamed.
_SECOND_CHECK_REPAIR_PHASES = frozenset({
    ResumePhase.CHECK_REPAIR_EXPANDED_C01,
    ResumePhase.CHECK_REPAIR_EXPANDED_C02,
})


def _check_repair_scope_payload(scope: CheckRepairScope) -> dict[str, Any]:
    return {
        "base_paths": list(scope.base_paths),
        "added_paths": list(scope.added_paths),
        "effective_paths": list(scope.effective_paths),
        "policy": scope.policy,
        "bound": scope.bound,
        "source": scope.source,
    }


def _second_check_repair_state(scope: CheckRepairScope) -> dict[str, Any]:
    """State/diagnostics facts about the second bounded check-repair pass.

    ``expanded_attempted`` is kept for durable compatibility with runs and
    UIs that predate the same-scope retry; ``scope_expanded`` is the fact
    that actually distinguishes a true expansion from a same-scope retry.
    """

    return {
        "expanded_attempted": True,
        "second_check_repair_attempted": True,
        "scope_expanded": scope.source != _SAME_SCOPE_RETRY_SOURCE,
        **_check_repair_scope_payload(scope),
    }


def _read_check_repair_scope(
    directory: Path,
    *,
    fallback_base: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
    allowed_added_sources: frozenset[str] = frozenset({_AUTO_BOUNDED_SOURCE}),
) -> CheckRepairScope:
    payload = _read_json_artifact(directory / "scope.json", 64 * 1024)
    base = tuple(sorted(set(fallback_base)))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        return CheckRepairScope(
            base_paths=base, added_paths=(), effective_paths=base,
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source="human-approved mutable scope",
        )
    raw_base = payload.get("base_mutable_scope")
    raw_added = payload.get("added_paths")
    raw_effective = payload.get("effective_mutable_scope")
    if not all(isinstance(value, list) for value in (raw_base, raw_added, raw_effective)):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    if any(not isinstance(path, str) for paths in (raw_base, raw_added, raw_effective) for path in paths):
        raise ResumeIntegrityError("check-repair scope artifact contains invalid paths")
    parsed_base = tuple(sorted(set(raw_base)))
    parsed_added = tuple(sorted(set(raw_added)))
    parsed_effective = tuple(sorted(set(raw_effective)))
    if parsed_base != base or parsed_effective != tuple(sorted(set(parsed_base) | set(parsed_added))):
        raise ResumeIntegrityError("check-repair scope artifact does not match its base scope")
    policy = payload.get("policy")
    bound = payload.get("bound")
    source = payload.get("source")
    if policy != policy_config.policy or bound != policy_config.max_added_paths or not isinstance(source, str):
        raise ResumeIntegrityError("check-repair scope policy changed")
    if parsed_added and source not in allowed_added_sources:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    return CheckRepairScope(parsed_base, parsed_added, parsed_effective, policy, bound, source)


def _validate_expanded_check_repair_scope(
    directory: Path,
    *,
    repo: Path,
    tree_sha: str,
    normal_scope: CheckRepairScope,
    policy_config: EffectiveRepairScopePolicy,
) -> CheckRepairScope:
    """Validate the durable scope of the second bounded check-repair pass.

    Two forms are legitimate, and each is validated strictly.

    Form A, a true expansion (``auto-bounded failing-test evidence``): every
    added path must be tracked in the checkpoint tree, must be an
    auto-expandable test path, must not already be in the base scope, and the
    policy and its bound must still hold.

    Form B, a same-scope retry (``bounded same-scope retry``): the scope must
    be *exactly* the one the first repair already held.  One extra path, or
    any other difference, is a resume integrity failure: the second pass
    never creates authority.
    """

    scope = _read_check_repair_scope(
        directory, fallback_base=normal_scope.base_paths,
        policy_config=policy_config, allowed_added_sources=_SECOND_SCOPE_SOURCES,
    )
    if scope.source == _SAME_SCOPE_RETRY_SOURCE:
        if (
            scope.added_paths != tuple(sorted(set(normal_scope.added_paths)))
            or scope.effective_paths != tuple(sorted(set(normal_scope.effective_paths)))
        ):
            raise ResumeIntegrityError(
                "same-scope check-repair retry scope is not the first repair scope"
            )
        return scope
    if not scope.added_paths:
        raise ResumeIntegrityError("expanded check-repair scope has no added paths")
    if scope.policy != "auto-bounded" or len(scope.added_paths) > scope.bound:
        raise ResumeIntegrityError("expanded check-repair scope violates its bound")
    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    if any(
        path in scope.base_paths
        or path not in tracked
        or not _is_auto_expandable_test_path(path)
        for path in scope.added_paths
    ):
        raise ResumeIntegrityError("expanded check-repair scope contains an invalid path")
    return scope


def _expanded_scope_is_applicable(
    *,
    checkpoint_phase: ResumePhase,
    expansion_phase: ResumePhase,
    scope_path: Path,
) -> bool:
    """Whether an expanded check-repair scope still carries authority.

    A validated expanded scope is cumulative: it stays part of the candidate
    tree's authority for every phase at or after the expansion, up to
    publication.  Applicability is decided from the canonical phase order and
    the durable artifact, never from a hand-maintained list of downstream
    phases -- such a list silently forgets every phase added later.
    """

    if phase_index(checkpoint_phase) < phase_index(expansion_phase):
        # The expansion has not happened yet: no additional authority.
        return False
    if checkpoint_phase is expansion_phase:
        # At the expansion phase itself the artifact is mandatory; its absence
        # must surface as a strict validation failure, not as a silent skip.
        return True
    # Downstream, only a durable expansion carries authority.  A run that
    # never expanded gains nothing.
    return scope_path.is_file()


def _step_result_record(outcome: StepExecutionOutcome) -> dict[str, Any]:
    status = getattr(outcome, "status", "COMPLETED")
    return {
        "id": outcome.step_id, "status": status, "profile_id": outcome.profile_id,
        "tree_before": outcome.tree_before, "tree_after": outcome.tree_after,
        "changed_paths": list(outcome.changed_paths),
        "usage": outcome.usage,
        "final": _bounded_v2_report(outcome.final_report),
        **({"mismatch": _bounded_v2_report(getattr(outcome, "mismatch", ""))}
           if getattr(outcome, "mismatch", None) else {}),
        **({"initial_mismatch": _bounded_v2_report(getattr(outcome, "initial_mismatch", ""))}
           if getattr(outcome, "initial_mismatch", None) else {}),
        **({"mismatch_retry_count": getattr(outcome, "mismatch_retry_count", 0)}
           if getattr(outcome, "mismatch_retry_count", 0) else {}),
        **({"deferred_verify": _bounded_v2_report(getattr(outcome, "deferred_verify", ""))}
           if getattr(outcome, "deferred_verify", None) else {}),
    }


def _revision_report_text(result: Any, artifact_dir: Path) -> str:
    """Bounded, reviewer-facing record of one Claude revision."""

    def tree(name: str) -> str | None:
        try:
            return (artifact_dir / name).read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeError):
            return None

    return _json_text({
        "final": _bounded_report(result.final_message),
        "tree_before": tree("tree_before.txt"),
        "tree_after": tree("tree_after.txt"),
        "usage": normalize_usage(result.usage),
    })


def _bounded_parse_detail(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


def authorize_commit(
    *,
    plan: TaskPlan,
    agent_result: AgentResult,
    evidence: EvidenceBundle,
    review: ReviewResult,
    worktree: Path,
    base_sha: str,
    branch_ref: str,
) -> str:
    """Single gate in front of the only commit call; return the tree to commit.

    Every precondition is re-derived here, from primary evidence where
    possible (the reviewer's raw answer is parsed again), immediately before
    committing.  Any failure raises :class:`CommitBoundaryError`.
    """

    if plan.decision is not PlanDecision.READY:
        raise CommitBoundaryError("planner decision is not READY")
    if agent_result.timed_out or agent_result.exit_code != 0:
        raise CommitBoundaryError("implementation agent did not exit successfully")
    if not evidence.deterministic_passed or evidence.failures:
        raise CommitBoundaryError("deterministic gate did not pass")
    approved_tree = evidence.staged_tree_sha
    if not approved_tree:
        raise CommitBoundaryError("no reviewed tree identity was recorded")
    try:
        reparsed = parse_review(review.raw, deterministic_passed=True)
    except ReviewParseError as exc:
        raise CommitBoundaryError(f"reviewer answer does not authorize a commit: {exc}") from exc
    if review.verdict is not ReviewVerdict.PASS or reparsed.verdict is not ReviewVerdict.PASS:
        raise CommitBoundaryError("reviewer verdict is not PASS")
    if review.route is not ReviewRoute.NONE or reparsed.route is not ReviewRoute.NONE:
        raise CommitBoundaryError("reviewer route is not NONE")
    if blocking_finding_lines(review.raw):
        raise CommitBoundaryError("reviewer reported a MAJOR or BLOCKER finding")

    if symbolic_head(worktree) != branch_ref:
        raise CommitBoundaryError("worktree HEAD no longer points to the run branch")
    if current_head(worktree) != base_sha:
        raise CommitBoundaryError("HEAD changed after review")
    if index_tree_sha(worktree) != approved_tree:
        raise CommitBoundaryError("index changed after review")
    problems = _status_has_unstaged_or_untracked(status_porcelain(worktree))
    if problems:
        raise CommitBoundaryError("; ".join(problems))
    if candidate_tree_sha(worktree) != approved_tree:
        raise CommitBoundaryError("working tree differs from the reviewed tree")
    return approved_tree


_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# Artifacts of one failed attempt, moved to ``attempts/NN/`` before the same
# operation is retried so a stale report is never read as the new one.
_ATTEMPT_ARTIFACTS = (
    "agent.prompt.txt", "agent.events.jsonl", "agent.stderr.log", "agent.final.md",
    "agent.result.json", "step.json", TOKEN_DIAGNOSTICS_NAME, "tree_after_failure.txt",
    "usage.json", "results.json",
)
_PLANNER_ATTEMPT_ARTIFACTS = (
    "planner.request.txt", "planner.raw.md", "task_plan_v2.json", "task_plan.json",
    "implementation_bundle.json", "planner.usage.json",
)
_REVIEW_ATTEMPT_ARTIFACTS = (
    "reviewer.request.txt", "reviewer.request.meta.json", "reviewer.raw.md",
    "reviewer.usage.json", "review.json",
)
_CHECK_ATTEMPT_ARTIFACTS = ("checks.json", "changed-files.txt", "diff.patch", "evidence.json")
# The historical root aliases of the C01 evidence.  ``checks/C01`` is the
# canonical directory; these names are only ever republished *from* it.
_CHECK_ALIAS_ARTIFACTS = _CHECK_ATTEMPT_ARTIFACTS
# Archiving ``pre_checks.json`` makes a CHECKS_Cxx resume replay the checks.
_PRE_CHECK_ATTEMPT_ARTIFACTS = _CHECK_ATTEMPT_ARTIFACTS + ("pre_checks.json",)
_REVISION_ATTEMPT_ARTIFACTS = _AGENT_ARTIFACTS + ("tree_after_failure.txt",)
_PLANNER_CONVERSATION = "planner.conversation.json"
# An operator recovery also retires the previous plan summary, the planner
# conversation (the repair planner must never continue a conversation whose
# answer was replaced) and any earlier recovery record.
_RECOVERY_ATTEMPT_ARTIFACTS = _PLANNER_ATTEMPT_ARTIFACTS + (
    "implementation_contract.md", _PLANNER_CONVERSATION, PLAN_RECOVERY_ARTIFACT,
)


class ReviewerTransportError(OrchestrationError):
    """No reviewer answer was obtained: a transport failure, not a verdict."""


class CandidatePushError(OrchestrationError):
    """The immutable candidate could not be pushed to the run branch."""

    code = "PUSH_FAILED"


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


@dataclasses.dataclass(frozen=True)
class _PersistedRevision:
    """A completed Claude revision read back from its durable artifacts."""

    final_message: str
    usage: dict[str, int]
    tree_before: str
    tree_after: str
    exit_code: int = 0
    timed_out: bool = False
    stderr_tail: str = ""


@dataclasses.dataclass
class _ResumedRun:
    """Everything a resume needs, rebuilt from persisted artifacts only."""

    checkpoint: ResumeCheckpoint
    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: Any
    info: WorktreeInfo
    repository_reference: RepositoryReference
    spec: str
    context: str
    restore_paths: tuple[str, ...] = ()
    mismatch_recovery: dict[str, Any] | None = None
    mismatch_recovery_path: Path | None = None
    scope_violation_recovery: dict[str, Any] | None = None
    c01_steps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # step id -> the clean mismatch its first attempt returned, for the single
    # bounded retry this resume owes that step.
    mismatch_retries: dict[str, str] = dataclasses.field(default_factory=dict)
    c01_revision: _PersistedRevision | None = None
    c01_check_repair_revision: _PersistedRevision | None = None
    c01_expanded_check_repair_revision: _PersistedRevision | None = None
    c01_evidence: EvidenceBundle | None = None
    c01_review: ReviewResult | None = None
    repair_plan: TaskPlanV2 | None = None
    repair_bundle: dict[str, Any] | None = None
    repair_bundle_sha: str | None = None
    c02_steps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    c02_revision: _PersistedRevision | None = None
    c02_check_repair_revision: _PersistedRevision | None = None
    c02_expanded_check_repair_revision: _PersistedRevision | None = None
    c02_evidence: EvidenceBundle | None = None
    c02_review: ReviewResult | None = None
    existing_commit_sha: str | None = None


def _read_json_artifact(path: Path, limit: int = 16 * 1024 * 1024) -> Any:
    try:
        if path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def _read_bounded_text(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_tree_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value if _GIT_OBJECT_ID.fullmatch(value) else None


def _is_object_id(value: Any) -> bool:
    return isinstance(value, str) and _GIT_OBJECT_ID.fullmatch(value) is not None


def _safe_candidate_tree(worktree: Path) -> str | None:
    try:
        return candidate_tree_sha(worktree)
    except GitError:
        return None


def _safe_index_tree(worktree: Path) -> str | None:
    try:
        return index_tree_sha(worktree)
    except GitError:
        return None


def _safe_status(worktree: Path) -> tuple[str, ...] | None:
    try:
        return status_porcelain(worktree)
    except GitError:
        return None


def _new_status_lines(
    before: tuple[str, ...], after: tuple[str, ...] | None,
) -> list[str]:
    """The porcelain status lines an attempt added or removed."""

    if after is None:
        return ["the Git status could not be read"]
    kept = set(before)
    seen = set(after)
    return [line for line in after if line not in kept] + [
        line for line in before if line not in seen
    ]


def _record_failure_tree(artifact_dir: Path, worktree: Path) -> None:
    """Record the tree a failed attempt left, so a resume can recognize it."""

    tree = _safe_candidate_tree(worktree)
    if tree is None:
        return
    try:
        atomic_write_text(artifact_dir / "tree_after_failure.txt", tree + "\n")
    except ResultArtifactError:
        pass


def _archive_attempt(directory: Path, *, names: tuple[str, ...] = _ATTEMPT_ARTIFACTS) -> Path | None:
    """Move a failed attempt's artifacts aside before retrying that operation."""

    present = [name for name in names if (directory / name).exists()]
    if not present:
        return None
    target = _archive_attempt_target(directory)
    for name in present:
        os.replace(directory / name, target / name)
    return target


def _archive_attempt_target(directory: Path) -> Path:
    """Create and return the next free ``attempts/NN/`` directory."""

    root = directory / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    target = root / f"{index:02d}"
    target.mkdir(parents=True)
    return target


def _archive_attempt_tree(directory: Path) -> None:
    """Archive every file of a retryable operation, including dynamic logs."""

    if not directory.is_dir():
        return
    files = [path for path in directory.rglob("*") if path.is_file() and "attempts" not in path.parts]
    if not files:
        return
    root = directory / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    target = root / f"{index:02d}"
    for source in files:
        relative = source.relative_to(directory)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def _reusable_pre_checks(artifact_dir: Path, tree: str) -> dict[str, Any] | None:
    """Durable pre-revision evidence frozen for exactly *tree*, if any."""

    payload = _read_json_artifact(artifact_dir / "pre_checks.json")
    if not isinstance(payload, dict) or payload.get("staged_tree_sha") != tree:
        return None
    failures = payload.get("failures")
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        return None
    if _hard_failure_items(failures):
        return None
    evidence = _read_json_artifact(artifact_dir / "evidence.json")
    if not isinstance(evidence, dict) or evidence.get("staged_tree_sha") != tree:
        return None
    try:
        # Keep the durable evidence existence check, but never return its
        # contents to a Claude prompt.
        (artifact_dir / "diff.patch").read_bytes()
    except OSError:
        return None
    return payload


def _load_evidence(directory: Path) -> EvidenceBundle | None:
    """Rebuild a frozen evidence bundle from ``evidence.json``."""

    payload = _read_json_artifact(directory / "evidence.json")
    if not isinstance(payload, dict):
        return None
    changed = payload.get("changed_files")
    checks = payload.get("checks")
    failures = payload.get("failures")
    if (
        not _is_object_id(payload.get("base_sha"))
        or not _is_object_id(payload.get("staged_tree_sha"))
        or not isinstance(payload.get("diff"), str)
        or not isinstance(payload.get("deterministic_passed"), bool)
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
        or not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks)
        or not isinstance(failures, list) or any(not isinstance(item, str) for item in failures)
    ):
        return None
    return EvidenceBundle(
        base_sha=payload["base_sha"],
        staged_tree_sha=payload["staged_tree_sha"],
        changed_files=tuple(changed),
        diff=payload["diff"],
        checks=tuple(checks),
        deterministic_passed=payload["deterministic_passed"],
        failures=tuple(failures),
        required_check_ids=tuple(
            item for item in payload.get("required_check_ids", [])
            if isinstance(item, str)
        ),
    )


def _retry_checks_evidence(
    checks_dir: Path, *, cycle: str, initial_tree: str, repaired_tree: str,
    legacy_dir: Path | None = None,
) -> tuple[EvidenceBundle | None, str | None]:
    """Resolve the evidence authority of a ``FINAL_CHECKS_RETRY`` resume.

    This phase is reached only after the automatic check-repair Claude
    *succeeded*, so the checkpoint tree is the repaired tree, not the red tree
    the repair answered to.  Two durable states are legitimate:

    * the first retry crashed before writing its evidence -- the red
      first-pass bundle has already been moved to ``attempts/01/`` and there
      is no current bundle;
    * a complete retry already ran and stayed red -- the current bundle is for
      the repaired tree.

    The red first-pass evidence stays the authority the repair answers to, so
    it is preferred when present; the retry checks are re-executed in both
    cases, which is why a red current bundle is never read as proof that the
    new checks are green.  Returns ``(evidence, refusal)``.
    """

    prior = _load_evidence(checks_dir / "attempts" / "01")
    if prior is not None and prior.staged_tree_sha != initial_tree:
        return None, (
            f"the archived {cycle} first-pass evidence is not for the pre-repair checks tree"
        )
    # Only a run without the per-cycle directory may fall back to the root
    # aliases: those aliases still name the *first* pass once the retry
    # archived it, so they are never the authority for the repaired tree.
    current = (
        _load_evidence(checks_dir) if checks_dir.is_dir() or legacy_dir is None
        else _load_evidence(legacy_dir)
    )
    if current is not None and current.staged_tree_sha != repaired_tree:
        return None, f"the {cycle} retry evidence is not for the repaired checks tree"
    evidence = prior or current
    if evidence is None:
        return None, f"the {cycle} final evidence is missing or not for the expected checks tree"
    return evidence, None


def _accepted_review(
    directory: Path, evidence: EvidenceBundle, candidate_sha: str | None = None,
) -> ReviewResult | None:
    """A reviewer answer already accepted for exactly this candidate tree."""

    if not (directory / "review.json").is_file():
        return None
    try:
        request = (directory / "reviewer.request.txt").read_text(encoding="utf-8")
        raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if f'"CANDIDATE_TREE_SHA": "{evidence.staged_tree_sha}"' not in request:
        return None
    if candidate_sha is not None and candidate_sha not in request:
        return None
    try:
        return parse_review(raw, deterministic_passed=evidence.deterministic_passed)
    except ReviewParseError:
        return None


def _load_c01_review(run_dir: Path, evidence: EvidenceBundle) -> ReviewResult | None:
    for directory in (run_dir / "review" / "C01", run_dir):
        try:
            raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        try:
            return parse_review(raw, deterministic_passed=evidence.deterministic_passed)
        except ReviewParseError:
            return None
    return None


def _load_accepted_c01_review(
    run_dir: Path, evidence: EvidenceBundle, candidate_sha: str,
) -> ReviewResult | None:
    """Reviewer #1's accepted answer for exactly the pushed C01 candidate."""

    for directory in (run_dir / "review" / "C01", run_dir):
        review = _accepted_review(directory, evidence, candidate_sha)
        if review is not None:
            return review
    return None


def _load_completed_step(step_dir: Path, step_id: str) -> dict[str, Any] | None:
    """One completed or cleanly deferred step record."""

    record = _read_json_artifact(step_dir / "step.json", 128 * 1024)
    if not isinstance(record, dict) or record.get("id") != step_id:
        return None
    status = record.get("status")
    if status not in {"COMPLETED", "DEFERRED_CONTRACT_MISMATCH"}:
        return None
    changed = record.get("changed_paths")
    if (
        not _is_object_id(record.get("tree_before"))
        or not _is_object_id(record.get("tree_after"))
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
    ):
        return None
    if status == "DEFERRED_CONTRACT_MISMATCH" and (
        record["tree_before"] != record["tree_after"]
        or changed
        or not isinstance(record.get("mismatch"), str)
        or not record["mismatch"].strip()
        or len(record["mismatch"].encode("utf-8", errors="replace")) > _MAX_STEP_REPORT_BYTES
    ):
        return None
    return {
        "id": step_id, "status": status, "profile_id": record.get("profile_id"),
        "tree_before": record["tree_before"], "tree_after": record["tree_after"],
        "changed_paths": list(changed),
        "usage": normalize_usage(record.get("usage")),
        "final": _bounded_v2_report(_read_bounded_text(step_dir / "agent.final.md")),
        **({"mismatch": _bounded_v2_report(record["mismatch"])}
           if status == "DEFERRED_CONTRACT_MISMATCH" else {}),
        **({"initial_mismatch": _bounded_v2_report(str(record["initial_mismatch"]))}
           if isinstance(record.get("initial_mismatch"), str) and record["initial_mismatch"].strip()
           else {}),
        **({"mismatch_retry_count": record["mismatch_retry_count"]}
           if isinstance(record.get("mismatch_retry_count"), int) else {}),
        **({"deferred_verify": _bounded_v2_report(str(record["deferred_verify"]))}
           if isinstance(record.get("deferred_verify"), str) and record["deferred_verify"].strip()
           else {}),
    }


def _verify_step_chain(records: list[dict[str, Any] | None], start_tree: str) -> str | None:
    """The last tree of an unbroken step chain starting at *start_tree*."""

    tree = start_tree
    for record in records:
        if record is None or record["tree_before"] != tree:
            return None
        tree = record["tree_after"]
    return tree


def _load_revision(directory: Path) -> _PersistedRevision | None:
    report = _read_json_artifact(directory / "report.json", 1024 * 1024)
    if not isinstance(report, dict) or report.get("status") not in {"COMPLETED", "NO_CHANGE"}:
        return None
    if not _is_object_id(report.get("tree_before")) or not _is_object_id(report.get("tree_after")):
        return None
    final = _read_bounded_text(directory / "agent.final.md", _MAX_AGENT_REPORT_BYTES * 2)
    if not final and isinstance(report.get("final"), str):
        final = report["final"]
    usage = read_usage_artifact(directory / "usage.json") or normalize_usage(report.get("usage"))
    return _PersistedRevision(final, usage, report["tree_before"], report["tree_after"])


def _read_repository_reference(run_dir: Path) -> RepositoryReference | None:
    payload = _read_json_artifact(run_dir / "repository_reference.json", 16 * 1024)
    if not isinstance(payload, dict) or set(payload) != {"remote_name", "web_url", "base_sha", "immutable_url"}:
        return None
    if not isinstance(payload["remote_name"], str) or not _is_object_id(payload["base_sha"]):
        return None
    if any(payload[key] is not None and not isinstance(payload[key], str) for key in ("web_url", "immutable_url")):
        return None
    return RepositoryReference(**payload)


def _persist_planner_conversation(run_dir: Path, handle: Any) -> None:
    """Persist a driver-provided planner conversation handle, never a guess."""

    if isinstance(handle, LLMConversationHandle):
        atomic_write_text(run_dir / _PLANNER_CONVERSATION, _json_text({
            "provider_id": handle.provider_id, "conversation_id": handle.conversation_id,
        }))


def _read_planner_conversation(run_dir: Path) -> LLMConversationHandle | None:
    payload = _read_json_artifact(run_dir / _PLANNER_CONVERSATION, 4096)
    if not isinstance(payload, dict):
        return None
    try:
        return LLMConversationHandle(payload.get("provider_id"), payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


def _commit_web_url(reference: RepositoryReference, commit_sha: str) -> str | None:
    if reference.web_url is None:
        return None
    try:
        normalized = normalize_github_web_url(reference.web_url)
    except ValueError:
        return None
    return f"{normalized}/commit/{commit_sha}" if normalized is not None else None


def _state_cycle_value(state: Mapping[str, Any]) -> int:
    value = state.get("cycle")
    return value if value in (1, 2) and not isinstance(value, bool) else 1


def _candidate_commit_path(run_dir: Path, cycle: int) -> Path:
    return run_dir / "candidate" / f"C{cycle:02d}" / "commit.json"


def _repair_mutation_sets(plan: TaskPlanV2) -> tuple[list[str], list[str], list[str]]:
    """Return canonical C02 mutation sets and reject structural ambiguity."""

    writes = sorted({path for step in plan.steps for path in step.write_set})
    creates = sorted({path for step in plan.steps for path in step.create_set})
    deletes = sorted({path for step in plan.steps for path in step.delete_set})
    if (set(writes) & set(creates)) or (set(writes) & set(deletes)) or (set(creates) & set(deletes)):
        raise OrchestrationError("REPAIR_SCOPE_MUTATION_SETS_OVERLAP")
    for path in (*writes, *creates, *deletes):
        posix = PurePosixPath(path)
        if (not path or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in posix.parts)
                or any(char in path for char in "*?[")):
            raise OrchestrationError("REPAIR_SCOPE_UNSAFE_PATH")
    return writes, creates, deletes


def _build_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, review: ReviewResult, repair_bundle_sha: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta, in memory only: from parsed plan sets, never reviewer prose."""

    writes, creates, deletes = _repair_mutation_sets(plan)
    requested = sorted(set(writes) | set(creates) | set(deletes))
    original = sorted(set(original_scope))
    added = sorted(set(requested) - set(original))
    unchanged = sorted(set(requested) & set(original))
    findings = review.required_fixes.strip() or review.findings.strip()
    reasons: dict[str, Any] = {}
    for path in added:
        steps = [step for step in plan.steps if path in set(step.write_set) | set(step.create_set) | set(step.delete_set)]
        step = steps[0]
        reasons[path] = {
            "reason": f"{step.title}: {step.objective}",
            "source_finding": findings,
        }
    try:
        raw_plan = (repair_dir / "planner.raw.md").read_bytes()
    except OSError as exc:
        raise OrchestrationError("REPAIR_SCOPE_PLAN_UNREADABLE") from exc
    plan_sha = hashlib.sha256(raw_plan).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "original_mutable_paths": original,
        "requested_write_paths": writes,
        "requested_create_paths": creates,
        "requested_delete_paths": deletes,
        "added_paths": added,
        "unchanged_paths": unchanged,
        "added_path_reasons": reasons,
        "source_finding": findings,
        "candidate_commit_sha": candidate_commit_sha,
        "repair_plan_sha256": plan_sha,
        "repair_bundle_sha256": repair_bundle_sha,
    }
    return payload, _json_text(payload)


def _create_file_once(path: Path, data: bytes) -> None:
    """Atomically create *path* with *data*; never replace an existing file.

    Raises :class:`FileExistsError` when *path* already exists.
    """

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _ensure_scope_delta(
    repair_dir: Path, content: str, *, expected_sha256: str | None,
) -> str:
    """Persist ``scope_delta.json`` exactly once, then only verify it.

    The first creation writes the canonical bytes atomically.  An existing
    artifact is never rewritten: its bytes must equal the canonical bytes and,
    when the checkpoint binds one, the checkpoint hash.  Any difference is a
    :class:`ResumeIntegrityError` and the file is left as found.
    """

    expected = content.encode("utf-8")
    digest = hashlib.sha256(expected).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ResumeIntegrityError("the C02 scope delta changed")
    path = repair_dir / "scope_delta.json"
    if expected_sha256 is None:
        try:
            _create_file_once(path, expected)
            return digest
        except FileExistsError:
            pass
        except OSError as exc:
            raise OrchestrationError("REPAIR_SCOPE_DELTA_UNWRITABLE") from exc
    try:
        if path.stat().st_size > 256 * 1024:
            raise ResumeIntegrityError("the C02 scope delta is too large")
        existing = path.read_bytes()
    except OSError as exc:
        raise ResumeIntegrityError(f"the C02 scope delta is unreadable: {exc}") from exc
    if existing != expected:
        raise ResumeIntegrityError("the C02 scope delta changed")
    return digest


def _build_scope_repair_delta(
    repair_dir: Path, *, original_scope: Sequence[str], plan: TaskPlanV2,
    failure_ids: Sequence[str], observed_outside_scope_paths: Sequence[str],
    repair_bundle_sha: str,
) -> tuple[dict[str, Any], str]:
    """Build the scope-repair authority from plan mutation sets only."""

    writes, creates, deletes = _repair_mutation_sets(plan)
    requested = sorted(set(writes) | set(creates) | set(deletes))
    original = sorted(set(original_scope))
    added = sorted(set(requested) - set(original))
    try:
        raw_plan = (repair_dir / "planner.raw.md").read_bytes()
    except OSError as exc:
        raise OrchestrationError("SCOPE_REPAIR_PLAN_UNREADABLE") from exc
    payload = {
        "schema_version": 1,
        "trigger": "revision_scope_violation",
        "original_mutable_paths": original,
        "requested_write_paths": writes,
        "requested_create_paths": creates,
        "requested_delete_paths": deletes,
        "added_paths": added,
        "failure_ids": sorted(set(failure_ids)),
        "observed_outside_scope_paths": sorted(set(observed_outside_scope_paths)),
        "repair_plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
        "repair_bundle_sha256": repair_bundle_sha,
    }
    return payload, _json_text(payload)


def _candidate_chain_parent(
    worktree: Path, run_dir: Path, base_sha: str, cycle: int, commit_sha: str,
) -> str:
    """The exact direct parent the published candidate of *cycle* must have.

    C01: ``BASE``.  C02: the persisted C01 candidate, itself exactly parented
    to ``BASE`` with its recorded tree.  Raises :class:`GitError` otherwise.
    """

    record = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
    if not isinstance(record, dict) or record.get("commit_sha") != commit_sha:
        raise GitError("candidate commit artifact does not match publication")
    expected_parent = base_sha
    if cycle == 2:
        c01 = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(c01, dict) or not _is_object_id(c01.get("commit_sha")):
            raise GitError("C01 candidate commit artifact is missing")
        if (
            c01.get("parent_sha") != base_sha
            or commit_parents(worktree, c01["commit_sha"]) != (base_sha,)
            or resolve_tree(worktree, c01["commit_sha"]) != c01.get("tree_sha")
        ):
            raise GitError("C01 candidate identity is not exact")
        expected_parent = c01["commit_sha"]
    if record.get("parent_sha") != expected_parent or commit_parents(worktree, commit_sha) != (expected_parent,):
        raise GitError("candidate commit parent is not the expected parent")
    return expected_parent


def _candidate_commit_payload(
    *, commit_sha: str, tree_sha: str, parent_sha: str, branch: str,
    remote: str, immutable_url: str | None, pushed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "parent_sha": parent_sha,
        "branch": branch,
        "remote": remote,
        "immutable_commit_url": immutable_url,
        "pushed_at": pushed_at,
    }



class Orchestrator:
    """Execute exactly one planner, one implementation agent and one review."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        planner_client: Any | None = None,
        reviewer_client: Any | None = None,
        recommender_client: Any | None = None,
        agent: CodexAgent | None = None,
        reviser: ClaudeCodeAgent | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self._planner_client = planner_client
        self._reviewer_client = reviewer_client
        self._recommender_client = recommender_client
        self._injected_agent = agent
        self._injected_reviser = reviser
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

    def _agent_for_profile(
        self, profile_id: str, role: ExecutionRole = ExecutionRole.IMPLEMENTER
    ) -> CodexAgent:
        try:
            profile = profile_for_role(self.config, profile_id, role)
        except ProfileError:
            if role is not ExecutionRole.IMPLEMENTER:
                raise
            profile = profile_for_role(self.config, profile_id, ExecutionRole.REPAIR)
        if self._injected_agent is not None:
            return self._injected_agent
        return CodexAgent(
            dataclasses.replace(
                build_agent_config(profile),
                env_allowlist=self.config.agent.env_allowlist,
            )
        )

    def _reviser_for_profile(self, profile_id: str) -> ClaudeCodeAgent:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.REVISER)
        build_claude_profile(profile)
        if self._injected_reviser is not None:
            return self._injected_reviser
        return ClaudeCodeAgent()

    def _run_revision(
        self,
        selection: Any,
        worktree: Path,
        run_dir: Path,
        prompt: str,
        revision_dir: Path | None = None,
    ) -> Any | None:
        selected = getattr(selection, "reviser", None)
        if selected is None:
            return None
        profile = profile_for_role(self.config, selected.profile_id, ExecutionRole.REVISER)
        claude_home = prepare_claude_home(self.config)
        environment = build_claude_environment(
            self._runtime_environment, claude_home=claude_home
        )
        kwargs: dict[str, Any] = {
            "artifacts_dir": run_dir,
            "profile": profile,
            "environment": environment,
        }
        if revision_dir is not None:
            kwargs["revision_dir"] = revision_dir
        return self._reviser_for_profile(profile.id).run_revision(
            redact(prompt, self._secrets), worktree, **kwargs
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
            ("agent.prompt.txt", "agent.events.jsonl", "agent.stderr.log",
             "agent.final.md", "agent.result.json", "pre_checks.json",
             "scope.json", "tree_before.txt", "tree_after.txt",
             "report.json", "usage.json"),
        )
        cls._copy_artifacts(
            run_dir, run_dir / "review" / "C01",
            ("reviewer.request.txt", "reviewer.raw.md", "reviewer.usage.json", "review.json"),
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
            payload = asdict(result) if dataclasses.is_dataclass(result) else {
                "exit_code": getattr(result, "exit_code", None),
                "timed_out": getattr(result, "timed_out", None),
                "usage": getattr(result, "usage", {}),
            }
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
            payload = asdict(result) if dataclasses.is_dataclass(result) else {
                "exit_code": getattr(result, "exit_code", None),
                "timed_out": getattr(result, "timed_out", None),
                "usage": getattr(result, "usage", {}),
            }
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
            store.initialize(selected_run_id)
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
        plan = planner.plan(spec, context, artifacts_dir=run_dir)
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
        codex_home = prepare_codex_home(self.config)
        store.update(status=RunStatus.IMPLEMENTING)
        implementation_contract = render_implementation_contract(plan)
        agent = self._agent_for_profile(selection.implementer.profile_id)
        implementer_profile = profile_for_role(
            self.config, selection.implementer.profile_id, ExecutionRole.IMPLEMENTER
        )
        agent_config = getattr(agent, "config", None)
        if not isinstance(agent_config, type(self.config.agent)):
            agent_config = dataclasses.replace(
                build_agent_config(implementer_profile),
                env_allowlist=self.config.agent.env_allowlist,
            )
        try:
            agent_environment = build_agent_environment(
                agent_config,
                source_environment=self._runtime_environment,
                codex_home=codex_home,
                forbidden_names=(
                    planner_profile.api_key_env,
                    profile_for_role(self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER).api_key_env,
                ),
            )
            tree_before_agent = candidate_tree_sha(info.worktree)
            agent_result = agent.run(
                implementation_contract,
                info.worktree,
                run_dir,
                base_sha=base_sha,
                env=agent_environment,
            )
        except AgentCommittedError as exc:
            # Detected, recorded and preserved: the worktree is never reset.
            self._redact_agent_artifacts(run_dir)
            state = store.record_failure("AGENT_COMMITTED", redact(str(exc), self._secrets))
            return RunResult(run_dir, RunStatus.FAILED, state)
        auth_failure = (
            not agent_result.timed_out
            and agent_result.exit_code != 0
            and _codex_auth_failure(run_dir / "agent.events.jsonl", agent_result.stderr_tail)
        )
        self._redact_agent_artifacts(run_dir)
        agent_result = dataclasses.replace(
            agent_result,
            final_message=redact(agent_result.final_message, self._secrets),
            stderr_tail=redact(agent_result.stderr_tail, self._secrets),
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
            state = store.record_failure("AGENT_GIT_VIOLATION", violations)
            return RunResult(run_dir, RunStatus.FAILED, state)
        if agent_result.timed_out:
            state = store.record_failure("AGENT_TIMEOUT")
            return RunResult(run_dir, RunStatus.FAILED, state)
        if agent_result.exit_code != 0:
            if auth_failure:
                state = store.record_failure(
                    "CODEX_AUTH_FAILURE", "Codex authentication failed"
                )
                return RunResult(run_dir, RunStatus.FAILED, state)
            state = store.record_failure(
                "AGENT_FAILED", f"exit status {agent_result.exit_code}"
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

        tree_after_agent = candidate_tree_sha(info.worktree)
        store.update(
            status=RunStatus.IMPLEMENTING,
            agent_candidate_tree_before=tree_before_agent,
            agent_candidate_tree_after=tree_after_agent,
        )
        if tree_after_agent == tree_before_agent:
            state = store.record_failure("AGENT_NO_CHANGE")
            return RunResult(run_dir, RunStatus.FAILED, state)

        store.update(status=RunStatus.VALIDATING)
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
        evidence = collect_evidence(
            info.worktree,
            base_sha,
            self.config,
            required_check_ids=getattr(plan, "required_checks", ()) or None,
            evidence_dir=run_dir,
            secrets=self._secrets,
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

        revision_enabled = self._run_options.claude_revision_enabled
        pipeline_enabled = revision_enabled or self._run_options.repair_cycles == 1
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
                    # Schema 4 only: a v3 approval is never a fallback.
                    selection, execution_sha = read_execution_selection_v4_with_sha256(run_dir)
                    validate_execution_selection_v4(self.config, selection)
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
                requested = resolve_execution_selection_v4(
                    self.config,
                    planner_profile_id=planner_profile_id,
                    step_profile_ids={
                        step.id: self.config.ui.default_implementer_profile or step.implementer_profile
                        for step in plan.steps
                    },
                    reviser_profile_id=self.config.ui.default_reviser_profile or "",
                    repair_implementer_profile_id=self.config.ui.default_repair_profile or "",
                    reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
                )
                selection = ensure_execution_selection_v4(run_dir, requested)
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
            if not isinstance(selection, ExecutionSelectionV4) or selection.schema_version != 4:
                raise ExecutionSelectionError("revision.enabled requires execution selection schema 4")
            validate_execution_selection_v4(self.config, selection)
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
        if isinstance(selection, ExecutionSelectionV4):
            execution_state["reviser"] = asdict(selection.reviser)
            execution_state["repair_implementer"] = asdict(selection.repair_implementer)
        execution_state["reviewer"] = asdict(selection.reviewer)
        store.update(status=RunStatus.PLANNING, execution=execution_state,
                     plan_identity=asdict(durable_identity))
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
                phase, cycle or (2 if phase_index(phase) >= phase_index(ResumePhase.REPAIR_PLANNER) else 1),
                step_id, head, tree, execution, identity,
                repair_bundle_sha256 or previous.repair_bundle_sha256,
                scope_delta_sha256 or previous.scope_delta_sha256,
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
            phase, cycle, step_id, head, tree,
            previous.execution_selection_sha256, previous.plan_identity, repair,
            scope_delta,
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
        revision_enabled = self._run_options.claude_revision_enabled
        repair_enabled = self._run_options.repair_cycles == 1

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
        codex_home = prepare_codex_home(self.config)
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
            try:
                outcome = self._execute_codex_step(
                    repo=repo, worktree=info.worktree, base_sha=base_sha,
                    branch_ref=branch_ref, ownership_before=ownership_before,
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
            self._v2_usage_rows.append({"id": step.id, **outcome.usage})
            expected_tree = outcome.tree_after
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
            # Step complete: the next operation is the next step, then the
            # C01 checks (pre-revision checks with Claude, final without).
            self._checkpoint(
                run_dir,
                ResumePhase.INITIAL_STEP if following else ResumePhase.CHECKS_C01,
                step_id=following, head=base_sha, tree=outcome.tree_after,
            )

        deferred_mismatches = _deferred_contract_mismatches(
            plan, self._last_v2_step_results
        )
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
                except ClaudeCommittedError as exc:
                    self._redact_revision_artifacts(run_dir)
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None,
                                           redact(str(exc), self._secrets))
                except (ClaudeAgentError, ClaudeRuntimeError) as exc:
                    self._redact_revision_artifacts(run_dir)
                    _record_failure_tree(run_dir / "revision", info.worktree)
                    return self._v2_failed(store, run_dir, "CLAUDE_FAILED", None,
                                           redact(str(exc), self._secrets))
                if revision_error is not None:
                    if revision_error == _SCOPE_REQUEST_ROUTE:
                        self._v2_failed(store, run_dir, "REVISION_SCOPE_VIOLATION", None)
                        return self.resume(run_id)
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
            revision_enabled and resumed is not None
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
                    # A run created before ``checks/C01`` was canonical kept
                    # its only final evidence at the run root.
                    reuse_fallback_dir=run_dir,
                )
                self._publish_check_aliases(run_dir, checks_dir_c01)
            except Exception:
                self._write_phase_checkpoint(
                    run_dir, final_checks_phase, cycle=1, head=base_sha, tree=checks_tree,
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
                    run_dir, final_checks_phase, cycle=1, head=base_sha, tree=checks_tree,
                )
                return self._v2_failed(store, run_dir, integrity_failures[0].split(":", 1)[0], None,
                                        ", ".join(integrity_failures))
        else:
            evidence = resumed.c01_evidence
            # Phase-dependent, never blanket: only ``FINAL_CHECKS_RETRY_C01``
            # is allowed to carry no current bundle, because its retry checks
            # are exactly the operation that has not succeeded yet.  Every
            # later phase keeps the existing mandatory-evidence validation.
            if evidence is None and not retry_bridge_c01:
                raise ResumeIntegrityError("C01 candidate evidence is missing")

        soft_failures_c01 = (
            _soft_check_failures(evidence)
            if revision_enabled and evidence is not None else []
        )
        base_repair_scope_c01 = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        if (
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
                except ClaudeCommittedError as exc:
                    self._redact_revision_artifacts(
                        run_dir, revision_dir=run_dir / "revision" / "check-repair" / "C01"
                    )
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None,
                                           redact(str(exc), self._secrets))
                except (ClaudeAgentError, ClaudeRuntimeError) as exc:
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
                        expected_head_sha=base_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, run_dir / "checks" / "C01")
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, retry_checks_phase, cycle=1,
                        head=base_sha, tree=retry_tree,
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
                                    expected_head_sha=base_sha,
                                    required_check_ids=plan.required_checks or None,
                                    enforce_diff_size=False,
                                )
                            except Exception:
                                self._write_phase_checkpoint(
                                    run_dir, expanded_retry_phase, cycle=1,
                                    head=base_sha, tree=retry_tree,
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
                        expected_head_sha=base_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, checks_dir_c01)
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_C01, cycle=1,
                        head=base_sha, tree=start.expected_tree_sha,
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
                        expected_head_sha=base_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, checks_dir_c01)
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
                        cycle=1, head=base_sha, tree=retry_tree,
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

        if at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01) and at <= phase_index(
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
                        expected_head_sha=base_sha,
                        required_check_ids=plan.required_checks or None,
                        enforce_diff_size=False,
                    )
                    self._publish_check_aliases(run_dir, run_dir / "checks" / "C01")
                except Exception:
                    self._write_phase_checkpoint(
                        run_dir, ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01,
                        cycle=1, head=base_sha, tree=retry_tree,
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
            self._checkpoint(run_dir, ResumePhase.CANDIDATE_COMMIT_C01,
                             head=base_sha, tree=evidence.staged_tree_sha)
            if current_head(info.worktree) == base_sha:
                self._authorize_candidate_tree(
                    evidence, info.worktree, base_sha, branch_ref
                )
            elif not (
                resumed is not None and resumed.existing_commit_sha is not None
                and commit_parents(info.worktree, resumed.existing_commit_sha) == (base_sha,)
                and resolve_tree(info.worktree, resumed.existing_commit_sha) == evidence.staged_tree_sha
            ):
                raise ResumeIntegrityError("C01 candidate commit exists with the wrong identity")
            c01_candidate = self._ensure_candidate_commit(
                run_dir=run_dir, info=info, cycle=1, tree_sha=evidence.staged_tree_sha,
                parent_sha=base_sha, title=plan.title,
                repository_reference=repository_reference, store=store, run_id=run_id,
            )
        else:
            c01_candidate = _read_json_artifact(_candidate_commit_path(run_dir, 1))
            if not isinstance(c01_candidate, dict):
                raise ResumeIntegrityError("C01 candidate commit artifact is missing")

        if at <= phase_index(ResumePhase.CANDIDATE_PUSH_C01):
            self._checkpoint(
                run_dir, ResumePhase.CANDIDATE_PUSH_C01,
                head=c01_candidate["commit_sha"], tree=evidence.staged_tree_sha,
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
                        plan_text=_review_plan_text(plan),
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
                except ClaudeCommittedError as exc:
                    self._redact_revision_artifacts(run_dir, revision_dir=run_dir / "revision" / "C02")
                    return self._v2_failed(store, run_dir, "CLAUDE_COMMITTED", None, redact(str(exc), self._secrets))
                except (ClaudeAgentError, ClaudeRuntimeError) as exc:
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
        agent = self._agent_for_profile(profile.id)
        agent_config = dataclasses.replace(
            build_agent_config(profile), env_allowlist=self.config.agent.env_allowlist
        )
        environment = build_agent_environment(
            agent_config, source_environment=self._runtime_environment,
            codex_home=codex_home, forbidden_names=forbidden_env_names,
        )
        # 5. One fresh Codex process for this step.  On a bounded retry the
        # contract is byte-identical; only the addendum is added.
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
            if hasattr(agent, "run_step"):
                result = agent.run_step(contract, worktree, artifact_dir,
                                        base_sha=base_sha, env=environment, **extra)
            else:
                # Test doubles from the v1 API may only expose run(); the
                # production CodexAgent always takes the step path above.
                result = agent.run(
                    build_implementer_step_prompt(contract, retry_addendum=retry_addendum),
                    worktree, artifact_dir, base_sha=base_sha, env=environment,
                )
        except AgentCommittedError as exc:
            self._redact_step_artifacts(artifact_dir)
            raise StepExecutionFailure(
                "AGENT_COMMITTED", step_id, redact(str(exc), self._secrets),
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
        auth_failure = _codex_auth_failure(artifact_dir / "agent.events.jsonl", result.stderr_tail)
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
        if result.timed_out:
            raise StepExecutionFailure(
                "AGENT_TIMEOUT", step_id, **failed, tree_after=_safe_candidate_tree(worktree)
            )
        if result.exit_code != 0:
            if auth_failure:
                raise StepExecutionFailure(
                    "CODEX_AUTH_FAILURE", step_id, "Codex authentication failed", **failed,
                    tree_after=_safe_candidate_tree(worktree),
                )
            raise StepExecutionFailure(
                "AGENT_FAILED", step_id, f"exit status {result.exit_code}", **failed,
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
        store: RunStateStore, run_id: str,
    ) -> dict[str, Any]:
        """Reconcile or create exactly one candidate commit for a cycle."""

        path = _candidate_commit_path(run_dir, cycle)
        stored = _read_json_artifact(path)
        commit_sha = stored.get("commit_sha") if isinstance(stored, dict) else None
        if not _is_object_id(commit_sha):
            commit_sha = None
        if commit_sha is None:
            try:
                head = current_head(info.worktree)
                if commit_parents(info.worktree, head) == (parent_sha,) and resolve_tree(info.worktree, head) == tree_sha:
                    commit_sha = head
            except GitError:
                pass
        if commit_sha is None:
            commit_sha = commit_candidate_tree(
                info.worktree, tree_sha=tree_sha, parent_sha=parent_sha,
                subject=_commit_subject(title), body=f"MetaHarness-Run: {run_id}",
            )
        if current_head(info.worktree) != commit_sha:
            raise CommitBoundaryError("candidate commit is not the run branch tip")
        if commit_parents(info.worktree, commit_sha) != (parent_sha,) or resolve_tree(info.worktree, commit_sha) != tree_sha:
            raise CommitBoundaryError("candidate commit identity is not exact")
        payload = _candidate_commit_payload(
            commit_sha=commit_sha, tree_sha=tree_sha, parent_sha=parent_sha,
            branch=info.branch, remote=self.config.publish.remote,
            immutable_url=_commit_web_url(repository_reference, commit_sha),
            pushed_at=(stored.get("pushed_at") if isinstance(stored, dict) else None),
        )
        atomic_write_text(path, _json_text(payload))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"C{cycle:02d}"] = payload
        store.update(
            status=RunStatus.APPROVED, candidate=candidate_state,
            candidate_commit_sha=commit_sha, approved_tree_sha=tree_sha,
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
            atomic_write_text(_candidate_commit_path(run_dir, cycle), _json_text(candidate))
            candidate_state = dict(store.load().get("candidate") or {})
            candidate_state[f"C{cycle:02d}"] = candidate
            store.update(status=store.load().get("status", RunStatus.APPROVED), candidate=candidate_state)
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
        review = reviewer.review(
            spec, input.plan_text, _bounded_review_context(context), gate,
            "\n".join(evidence.changed_files),
            "",
            _json_text(_check_payload(evidence)),
            "NONE",
            deterministic_passed=evidence.deterministic_passed,
            artifacts_dir=artifacts_dir,
            repository=_json_text(repository_reference_dict(repository_reference)),
            luna_reports=input.luna_reports,
            revision_report=input.revision_report,
            repository_state=repository_state,
            candidate_commit=_json_text(dict(candidate_commit)),
            iteration=input.iteration,
            cycle_history=input.cycle_history,
            deferred_mismatches=input.deferred_mismatches,
            code_evidence=code_evidence,
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
                return stored
        check_config, check_ids = config_with_check_authority(
            self.config, evidence_dir, requested_check_ids=required_check_ids,
            expected_sha256=self._approved_check_authority_sha256(evidence_dir),
        )
        return collect_evidence(
            worktree, base_sha, check_config, evidence_dir=evidence_dir,
            secrets=self._secrets, check_failures_hard=check_failures_hard,
            expected_head_sha=expected_head_sha, required_check_ids=check_ids,
            enforce_diff_size=enforce_diff_size,
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
        base_paths = tuple(sorted(set(base_mutable_scope)))
        scope_policy = self._effective_repair_scope
        policy = scope_policy.policy
        bound = scope_policy.max_added_paths
        candidates = (
            _check_repair_scope_candidates(
                repo=repo,
                worktree=worktree,
                tree_sha=tree_sha,
                run_dir=run_dir,
                evidence=evidence,
                base_mutable_scope=base_paths,
            )
            if policy == "auto-bounded" else []
        )
        added = tuple(candidates) if len(candidates) <= bound else ()
        return CheckRepairScope(
            base_paths=base_paths,
            added_paths=added,
            effective_paths=tuple(sorted(set(base_paths) | set(added))),
            policy=policy,
            bound=bound,
            source=(
                _AUTO_BOUNDED_SOURCE
                if added else "human-approved mutable scope"
            ),
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
        """The scope of the one second bounded check-repair pass, or ``None``.

        This is the single decision point for "may the corrective Claude pass
        run once more", and the retry budget is deliberately independent from
        the mutable scope.  A scope expansion is one possible *outcome* of
        this decision, never its precondition: a soft deterministic failure
        whose fix already lives inside the authorized paths earns a second
        pass with exactly that scope.

        It stays model-free and fail-closed; every condition is a durable
        fact -- the failure family, the scope the first repair already held,
        the operator's policy and its bound, and whether a second pass
        already produced a report.  ``None`` means the deterministic gate is
        final.
        """

        if not _soft_check_failures(evidence):
            return None
        if _hard_integrity_failures(evidence):
            return None
        # One second pass per cycle, even across a crash: never a third.
        if (expanded_dir / "report.json").exists():
            return None
        policy_config = self._effective_repair_scope
        # No new path, no new authority: exactly what the first repair held.
        same_scope = CheckRepairScope(
            base_paths=normal_scope.base_paths,
            added_paths=normal_scope.added_paths,
            effective_paths=normal_scope.effective_paths,
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source=_SAME_SCOPE_RETRY_SOURCE,
        )
        if policy_config.policy != "auto-bounded":
            # ``deny-expansion`` and ``require-approval`` forbid *growing* the
            # mutable scope.  Neither forbids one more bounded correction
            # inside the scope a human already approved.
            return same_scope
        candidates = _check_repair_scope_candidates(
            repo=repo,
            worktree=worktree,
            tree_sha=tree_sha,
            run_dir=run_dir,
            evidence=evidence,
            # Only genuinely new paths are candidates; the paths the first
            # repair already earned are never re-added, and never lost.
            base_mutable_scope=normal_scope.effective_paths,
        )
        if not candidates:
            return same_scope
        added = tuple(sorted(set(normal_scope.added_paths) | set(candidates)))
        if len(added) > policy_config.max_added_paths:
            # The bound caps the expansion, not the retry.
            return same_scope
        return CheckRepairScope(
            base_paths=normal_scope.base_paths,
            added_paths=added,
            effective_paths=tuple(sorted(
                set(normal_scope.base_paths) | set(added)
            )),
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source=_AUTO_BOUNDED_SOURCE,
        )

    def _durable_normal_check_repair_scope(
        self,
        run_dir: Path,
        *,
        cycle: int,
        base_paths: Sequence[str],
        scope: CheckRepairScope | None = None,
    ) -> CheckRepairScope:
        """The first repair's scope, from memory or from its durable artifact.

        A resume that starts at the second repair has no in-band scope, so the
        artifact the first repair published is the authority for what that
        pass was allowed to touch.
        """

        if scope is not None:
            return scope
        return _read_check_repair_scope(
            run_dir / "revision" / "check-repair" / f"C0{cycle}",
            fallback_base=base_paths,
            policy_config=self._effective_repair_scope,
        )

    @staticmethod
    def _normal_check_repair_scope(
        scope: CheckRepairScope | None, base_paths: Sequence[str],
        policy: EffectiveRepairScopePolicy,
    ) -> CheckRepairScope:
        """The normal check-repair scope, defaulted to the approved base."""

        if scope is not None:
            return scope
        base = tuple(sorted(set(base_paths)))
        return CheckRepairScope(
            base_paths=base, added_paths=(), effective_paths=base,
            policy=policy.policy, bound=policy.max_added_paths,
            source="human-approved mutable scope",
        )

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
        """Reuse a durable second-pass scope, or decide a new one.

        A crash may have landed between ``scope.json`` and the second
        corrective Claude pass while the checkpoint still names the retry
        checks.  The durable scope then outranks any recomputation: the model
        is never recalled with a scope different from the one already
        published for it.  Returns ``(scope, archive_required)``;
        ``archive_required`` is false for a reused scope, whose attempt was
        already archived.
        """

        if (expanded_dir / "scope.json").is_file():
            scope = _validate_expanded_check_repair_scope(
                expanded_dir, repo=repo, tree_sha=evidence.staged_tree_sha,
                normal_scope=normal_scope,
                policy_config=self._effective_repair_scope,
            )
            # One second pass per cycle, even across a crash.
            if (expanded_dir / "report.json").exists():
                return None, False
            return scope, False
        return self._second_check_repair_scope(
            repo=repo, worktree=worktree, tree_sha=evidence.staged_tree_sha,
            run_dir=run_dir, evidence=evidence, normal_scope=normal_scope,
            expanded_dir=expanded_dir,
        ), True

    def _run_v2_revision_cycle(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        repo: Path,
        base_sha: str,
        base_tree_sha: str,
        spec: str,
        plan: TaskPlanV2,
        repository_reference: RepositoryReference,
        info: Any,
        branch_ref: str,
        ownership_before: Any,
        selection: ExecutionSelectionV4,
        artifact_dir: Path | None = None,
        mutable_scope: list[str] | None = None,
        deferred_mismatches: str | None = None,
        deferred_mismatch_present: bool = False,
        check_repair_evidence: EvidenceBundle | None = None,
        cycle: int = 1,
        check_repair_scope: CheckRepairScope | None = None,
        check_repair_phase_override: ResumePhase | None = None,
        check_repair_next_phase_override: ResumePhase | None = None,
    ) -> tuple[Any | None, str | None]:
        """Run one Claude pre-check/revision/scope cycle.

        Pre-revision checks already durable for the exact current tree are
        reused (a resume never replays them); Claude runs once per attempt.
        """

        claude_phase, review_phase = (
            (ResumePhase.CLAUDE_C01, ResumePhase.REVIEWER_C01) if cycle == 1
            else (ResumePhase.CLAUDE_C02, ResumePhase.REVIEWER_C02)
        )
        is_check_repair = check_repair_evidence is not None
        artifact_dir = artifact_dir or (
            run_dir / "revision" / "check-repair" / f"C0{cycle}"
            if is_check_repair else run_dir / "revision"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        expected_head = current_head(info.worktree)
        stage_all(info.worktree)
        tree_before = candidate_tree_sha(info.worktree)
        if is_check_repair and check_repair_evidence.staged_tree_sha != tree_before:
            return None, "TOCTOU_FAILURE"
        mutable_scope = mutable_scope or sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        if is_check_repair:
            check_repair_scope = check_repair_scope or CheckRepairScope(
                base_paths=tuple(mutable_scope), added_paths=(),
                effective_paths=tuple(mutable_scope),
                policy=self._effective_repair_scope.policy,
                bound=self._effective_repair_scope.max_added_paths,
                source="human-approved mutable scope",
            )
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "schema_version": 2,
                "base_mutable_scope": list(check_repair_scope.base_paths),
                "added_paths": list(check_repair_scope.added_paths),
                "effective_mutable_scope": list(check_repair_scope.effective_paths),
                "policy": check_repair_scope.policy,
                "bound": check_repair_scope.bound,
                "source": check_repair_scope.source,
                "bound_exceeded": (
                    check_repair_scope.policy == "auto-bounded"
                    and not check_repair_scope.added_paths
                    and len(_check_repair_scope_candidates(
                        repo=repo, worktree=info.worktree, tree_sha=tree_before,
                        run_dir=run_dir, evidence=check_repair_evidence,
                        base_mutable_scope=check_repair_scope.base_paths,
                    )) > check_repair_scope.bound
                ),
            }))
        else:
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "approved_mutable_scope": mutable_scope,
                "source": "human-approved mutable scope",
            }))
        if is_check_repair:
            # This boundary is deliberately written before invoking Claude so
            # a timeout/transport failure resumes this exact corrective pass.
            repair_phase = check_repair_phase_override or (
                ResumePhase.CHECK_REPAIR_C01 if cycle == 1
                else ResumePhase.CHECK_REPAIR_C02
            )
            self._checkpoint(
                run_dir, repair_phase, cycle=cycle, head=expected_head, tree=tree_before,
            )
            pre_payload = {
                "checks": _check_payload(check_repair_evidence),
                "failures": list(check_repair_evidence.failures),
                "deterministic_passed": check_repair_evidence.deterministic_passed,
                "staged_tree_sha": check_repair_evidence.staged_tree_sha,
            }
        else:
            reused = _reusable_pre_checks(artifact_dir, tree_before)
            if reused is None:
                self._write_phase_checkpoint(
                    run_dir,
                    ResumePhase.CHECKS_C01 if cycle == 1 else ResumePhase.CHECKS_C02,
                    cycle=cycle, head=expected_head, tree=tree_before,
                )
                store.update(status=RunStatus.PRE_REVISION_VALIDATING, current_step=None)
                check_config, check_ids = config_with_check_authority(
                    self.config, run_dir, requested_check_ids=plan.required_checks or None,
                    expected_sha256=self._approved_check_authority_sha256(run_dir),
                )
                pre_evidence = collect_evidence(
                    info.worktree, base_sha, check_config,
                    required_check_ids=check_ids,
                    evidence_dir=artifact_dir, secrets=self._secrets,
                    check_failures_hard=False,
                    expected_head_sha=expected_head,
                    enforce_diff_size=False,
                )
                pre_payload = {
                    "checks": _check_payload(pre_evidence),
                    "failures": list(pre_evidence.failures),
                    "deterministic_passed": pre_evidence.deterministic_passed,
                    "staged_tree_sha": pre_evidence.staged_tree_sha,
                }
                atomic_write_text(artifact_dir / "pre_checks.json", _json_text(pre_payload))
                pre_hard = _hard_integrity_failures(pre_evidence)
                # A clean deferred mismatch intentionally leaves no candidate
                # delta for the pre-revision gate.  Claude is the recovery owner,
                # so EMPTY_DIFF is evidence for Claude here, not a terminal gate.
                # A deferred *verify* dependency is not this case: that step did
                # change the candidate, so the normal gate applies.
                if deferred_mismatch_present:
                    pre_hard = [item for item in pre_hard if item != "EMPTY_DIFF"]
                if pre_hard:
                    return None, pre_hard[0].split(":", 1)[0]
                # Pre-revision checks complete: the next operation is Claude.  The
                # authorized HEAD is the worktree HEAD (base for C01, the C01
                # candidate commit for C02), exactly as _validate_resume expects.
                self._checkpoint(run_dir, claude_phase, cycle=cycle, head=expected_head, tree=tree_before)
            else:
                pre_payload = reused
        if is_check_repair:
            previous_report = ""
            if check_repair_phase_override in _SECOND_CHECK_REPAIR_PHASES:
                # The second pass must read the *first repair's* report, not
                # the initial revision it already superseded.
                previous_dir = run_dir / "revision" / "check-repair" / f"C0{cycle}"
                previous_report = _read_bounded_text(previous_dir / "agent.final.md")
            revision_prompt = _check_repair_prompt(
                spec=spec,
                plan=plan,
                approved_contract_index=_revision_contract_index(plan),
                changed_files="\n".join(check_repair_evidence.changed_files),
                evidence=check_repair_evidence,
                mutable_scope=mutable_scope,
                previous_report=previous_report,
                added_paths=(check_repair_scope.added_paths if check_repair_scope else ()),
                scope_source=(check_repair_scope.source if check_repair_scope else ""),
            )
        else:
            revision_prompt = _revision_prompt(
                repository_reference=repository_reference,
                spec=spec,
                plan=plan,
                changed_files="\n".join(changed_paths_between_trees(repo, base_tree_sha, tree_before)),
                execution_anomalies=_revision_execution_anomalies(
                    self._repair_v2_step_results if cycle == 2
                    else self._last_v2_step_results
                ),
                pre_checks=_revision_check_context(pre_payload),
                mutable_scope=_json_text(mutable_scope),
                deferred_mismatches=deferred_mismatches or "NONE\n",
            )
        store.update(status=RunStatus.REVISING, current_step=None)
        atomic_write_text(artifact_dir / "tree_before.txt", tree_before.rstrip() + "\n")
        result = self._run_revision(
            selection, info.worktree, run_dir, revision_prompt,
            revision_dir=artifact_dir,
        )
        self._ensure_revision_artifacts(artifact_dir, result)
        claude_auth_failure = _claude_auth_failure(
            run_dir, result.stderr_tail, revision_dir=artifact_dir
        )
        self._redact_revision_artifacts(run_dir, revision_dir=artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self._secrets),
            stderr_tail=redact(result.stderr_tail, self._secrets),
        )
        if result.timed_out:
            _record_failure_tree(artifact_dir, info.worktree)
            return result, "CLAUDE_TIMEOUT"
        terminal_is_error = getattr(result, "terminal_is_error", None) is True
        terminal_subtype = getattr(result, "terminal_subtype", None)
        # A structured terminal marked ``is_error`` is a failure on its own.
        # Some CLI versions and wrappers still exit 0 after one, so exit code
        # is the last signal consulted, never the gate for the others.
        if terminal_is_error or result.exit_code != 0:
            _record_failure_tree(artifact_dir, info.worktree)
            if claude_auth_failure:
                return result, "CLAUDE_AUTH_FAILURE"
            # Claude's terminal result is authoritative when available.  The
            # textual fallback above remains for older CLI versions and old
            # artifacts that do not expose terminal metadata.
            if terminal_subtype == "error_max_turns":
                return result, "CLAUDE_MAX_TURNS"
            return result, "CLAUDE_FAILED"
        revision_ownership = _git_ownership(repo, info.worktree)
        if revision_ownership.head != expected_head:
            return result, "CLAUDE_COMMITTED"
        violations = _ownership_violations(
            ownership_before, revision_ownership,
            branch_ref=branch_ref, base_sha=expected_head,
        )
        if violations:
            return result, "AGENT_GIT_VIOLATION"
        stage_all(info.worktree)
        tree_after = candidate_tree_sha(info.worktree)
        atomic_write_text(artifact_dir / "tree_after.txt", tree_after.rstrip() + "\n")
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        outside_scope = [path for path in changed_paths if path not in set(mutable_scope)]
        scope_request = (
            parse_scope_request(result.final_message)
            if is_check_repair else None
        )
        malformed_scope_request = (
            is_check_repair
            and _SCOPE_REQUEST_HEADER in result.final_message
            and scope_request is None
        )
        usage = normalize_usage(result.usage)
        revision_state = {
            "profile_id": selection.reviser.profile_id,
            "status": "NO_CHANGE" if tree_after == tree_before else "COMPLETED",
            "tree_before": tree_before,
            "tree_after": tree_after,
            "usage": usage,
            **(
                {
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                }
                if scope_request is not None else {}
            ),
            **(
                {"scope_request_warning": "malformed scope request ignored as authority"}
                if malformed_scope_request else {}
            ),
        }
        atomic_write_text(artifact_dir / "usage.json", _json_text(usage))
        atomic_write_text(artifact_dir / "report.json", _json_text({
            **revision_state,
            "final": _bounded_report(result.final_message),
            "stderr_tail": result.stderr_tail,
            "changed_paths": list(changed_paths),
            "outside_scope_paths": list(outside_scope),
            **({"failure_ids": _soft_check_failures(check_repair_evidence)}
               if is_check_repair else {}),
        }))
        store.update(status=RunStatus.REVISING, revision=revision_state)
        if scope_request is not None:
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True,
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                },
            )
        elif malformed_scope_request:
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True,
                    "scope_request_warning": "malformed scope request ignored as authority",
                },
            )
        if outside_scope:
            # This is a successful Claude transport with an unsafe candidate,
            # so the exact failed tree must remain durable for the fail-closed
            # rollback proof used by deterministic check-repair recovery.
            _record_failure_tree(artifact_dir, info.worktree)
            return result, "REVISION_SCOPE_VIOLATION"
        if scope_request is not None:
            # A valid request is advisory evidence, never an authorization
            # delta.  Even an in-scope/no-op attempt is rolled back atomically
            # before the durable bridge path is allowed to inspect it.
            _record_failure_tree(artifact_dir, info.worktree)
            try:
                restore_paths_from_tree(info.worktree, tree_before, list(changed_paths))
                stage_all(info.worktree)
                if (
                    candidate_tree_sha(info.worktree) != tree_before
                    or index_tree_sha(info.worktree) != tree_before
                    or _status_has_unstaged_or_untracked(status_porcelain(info.worktree))
                ):
                    raise GitError("scope-request rollback did not restore the exact tree")
            except (GitError, OSError):
                # Keep the existing dirty-violation recovery as the fail-safe
                # owner of a rollback that could not be proven immediately.
                pass
            return result, _SCOPE_REQUEST_ROUTE
        # Claude complete and durable: the next operation is the final checks
        # followed by candidate commit/push and then the reviewer.
        next_revision_phase = (
            check_repair_next_phase_override or (
                ResumePhase.FINAL_CHECKS_RETRY_C01 if cycle == 1
                else ResumePhase.FINAL_CHECKS_RETRY_C02
            )
            if is_check_repair
            else (ResumePhase.FINAL_CHECKS_C01 if cycle == 1 else ResumePhase.FINAL_CHECKS_C02)
        )
        self._checkpoint(
            run_dir, next_revision_phase, cycle=cycle, head=expected_head, tree=tree_after,
        )
        return result, None

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
        selection: ExecutionSelectionV4,
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
            store.update(
                scope_repair_scope_escalation={
                    "trigger": "REVISION_SCOPE_VIOLATION",
                    "added_paths": scope_delta["added_paths"],
                    "policy": self._effective_repair_scope.policy,
                    "auto_authorized": True,
                }
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
            repair_codex_home = prepare_codex_home(self.config)
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
        if evidence is None:
            raise ResumeIntegrityError("scope-repair checks evidence is missing")
        hard = _hard_integrity_failures(evidence)
        if hard:
            raise OrchestrationError(hard[0].split(":", 1)[0])
        effective_scope = sorted(set(original_scope) | set(requested_scope))
        residual_result = None
        if not evidence.deterministic_passed:
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
                    except (ClaudeAgentError, ClaudeRuntimeError, ClaudeCommittedError) as exc:
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
        selection: ExecutionSelectionV4,
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
                        "original_approved_plan": _review_plan_payload(original_plan),
                        "repair_plan_c02": _review_plan_payload(normal_repair_plan),
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
        selection: ExecutionSelectionV4,
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
        codex_home = prepare_codex_home(self.config)
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
            claude_revision_enabled and resumed is not None and start is not None
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
            if evidence is None and not retry_bridge_c02:
                raise ResumeIntegrityError("C02 candidate evidence is missing")

        soft_failures_c02 = (
            _soft_check_failures(evidence)
            if claude_revision_enabled and evidence is not None else []
        )
        base_repair_scope_c02 = list(repair_scope)
        if (
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

        if at >= phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02) and at <= phase_index(
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
            )
            c02_candidate = self._push_candidate(
                run_dir=run_dir, info=info, cycle=2, candidate=c02_candidate, store=store,
            )
            self._cycle_update(store, 2, status="candidate_pushed")
        # C02 candidate push complete: the next operation is reviewer #2.
        self._checkpoint(run_dir, ResumePhase.REVIEWER_C02, cycle=2,
                         head=c02_candidate["commit_sha"], tree=evidence.staged_tree_sha,
                         repair_bundle_sha256=repair_bundle_sha)
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
                            "original_approved_plan": _review_plan_payload(original_plan),
                            "repair_plan_c02": _review_plan_payload(repair_plan),
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
        try:
            if store.load().get("approved_tree_sha") != approved_tree:
                raise GitError("durable approved tree differs from candidate tree")
            if current_head(info.worktree) != commit_sha:
                raise GitError("candidate commit is not the run branch tip")
            if resolve_tree(info.worktree, commit_sha) != approved_tree:
                raise GitError("candidate commit tree differs from approved tree")
            # BASE -> C01 (-> C02): the exact direct parent, never "any descendant".
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
        )
        if not self.config.publish.enabled:
            state = store.update(status=RunStatus.COMMITTED, **fields)
            mark_checkpoint_completed(run_dir)
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
        if not self.config.publish.enabled:
            state = store.update(status=RunStatus.COMMITTED, **fields)
            mark_checkpoint_completed(run_dir)
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
                        if self._run_options.claude_revision_enabled or self._run_options.repair_cycles == 1:
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
        revision_enabled = self._run_options.claude_revision_enabled
        repair_enabled = self._run_options.repair_cycles == 1
        if self.config.planning.protocol != "v2" or state.get("planning_protocol") != "v2":
            refuse("only META PLAN v2 runs can be resumed")
        if not revision_enabled and not repair_enabled and checkpoint.phase not in (
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
            if revision_enabled or repair_enabled:
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
            phase_expected_head = base_sha
            if checkpoint.phase in c02_phases:
                prior = _read_json_artifact(_candidate_commit_path(run_dir, 1))
                if not isinstance(prior, dict) or not _is_object_id(prior.get("commit_sha")):
                    refuse("C01 candidate commit is missing for C02 resume")
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
                expected_parent = base_sha
                if cycle == 2:
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
                expected_parent = base_sha
                if checkpoint.cycle == 2:
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
            if failure.get("reason") in {"REVISION_SCOPE_VIOLATION", _SCOPE_REQUEST_ROUTE} and checkpoint.phase in {
                ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
                ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
            }:
                if checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
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
            if phase is ResumePhase.CHECK_REPAIR_C01:
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
    ) -> tuple[str, ...]:
        """Validate, roll back, and archive an unsafe check-repair attempt.

        The failed Claude tree is treated as evidence only.  Every changed
        path is restored from the checkpoint tree, including paths that were
        inside the previous scope; keeping a partial attempt would make the
        subsequent strong planner reason from an unauthorised candidate.
        """

        state = _read_json_artifact(run_dir / "state.json", 256 * 1024)
        failure = state.get("failure") if isinstance(state, Mapping) else None
        if not isinstance(failure, Mapping) or failure.get("reason") not in {
            "REVISION_SCOPE_VIOLATION", _SCOPE_REQUEST_ROUTE,
        }:
            raise ResumeIntegrityError("scope-violation recovery requires a Claude scope route")
        if checkpoint.phase not in {
            ResumePhase.CHECK_REPAIR_C01, ResumePhase.CHECK_REPAIR_EXPANDED_C01,
            ResumePhase.CHECK_REPAIR_C02, ResumePhase.CHECK_REPAIR_EXPANDED_C02,
        }:
            raise ResumeIntegrityError("scope-violation recovery is only valid for check-repair phases")
        if checkpoint.expected_head_sha is None or checkpoint.expected_tree_sha is None:
            raise ResumeIntegrityError("scope-violation checkpoint has no Git identity")

        if checkpoint.phase is ResumePhase.CHECK_REPAIR_C01:
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
        failure_tree = _read_tree_file(failure_tree_path)
        if not isinstance(report, dict) or failure_tree is None:
            raise ResumeIntegrityError("scope-violation report or failure tree is missing")
        if report.get("tree_before") != checkpoint.expected_tree_sha:
            raise ResumeIntegrityError("scope-violation report tree_before does not match the checkpoint")
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
        if revision_enabled and resumed.scope_violation_recovery is None and at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_C01) and (
            check_repair_dir_c01 / "report.json"
        ).exists():
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
        if at <= phase_index(ResumePhase.REVIEWER_C01) and expected != c01_tree:
            refuse(
                "the checkpoint tree is not the Claude C01 tree" if resumed.c01_revision is not None
                else "the checkpoint tree is not the last completed Luna tree"
            )
        if checkpoint.phase is ResumePhase.INITIAL_STEP:
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
        if revision_enabled and resumed.scope_violation_recovery is None and at >= phase_index(ResumePhase.FINAL_CHECKS_RETRY_C02) and (
            check_repair_dir_c02 / "report.json"
        ).exists():
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
        if expected != c02_tree:
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
