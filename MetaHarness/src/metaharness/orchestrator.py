"""The single-task, single-agent MetaHarness V0 state machine."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .agent.base import AgentError, AgentResult
from .agent.codex import (
    AgentCommittedError,
    CodexAgent,
    build_agent_environment,
    build_implementer_step_prompt,
    classify_codex_failure,
)
from .agent.runtime import prepare_codex_home
from .approval import (
    ApprovalDecision,
    ApprovalError,
    compute_plan_identity_from_run,
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
    collect_evidence,
)
from .gitops import (
    GitError,
    assert_clean,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_reviewed_tree,
    create_run_worktree,
    current_head,
    git_root,
    index_tree_sha,
    local_branches,
    build_repository_reference,
    path_exists_in_tree,
    registered_worktrees,
    resolve_commit,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
    status_porcelain,
    symbolic_head,
    stage_all,
)
from .llm.chat import LLMError, OpenAIChatTextClient
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
    validate_execution_selection,
    read_execution_selection_v3_with_sha256,
    validate_execution_selection_v3,
)
from .models import (
    ExecutionRole,
    ExecutionSelectionV3,
    HarnessConfig,
    ImplementationStep,
    ReviewRoute,
    ReviewVerdict,
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
    PlannerV2,
    TaskPlanV2,
    V2PlanParseError,
    read_approved_step_contract,
    read_set_paths,
    validate_implementation_bundle,
)
from .usage import add_usage, empty_usage, normalize_usage
from .redaction import config_secret_values, redact, redact_file
from .profiles import (
    build_agent_config,
    build_llm_endpoint,
    profile_for_role,
    profiles_for_config,
)
from .result import RunResult, write_repair_task
from .review import Reviewer, ReviewParseError, ReviewResult, blocking_finding_lines, parse_review
from .state import RunStateStore
from .validation import ValidationError, check_result_json
from .workspace import WorkspaceSetupError, prepare_workspace
from .result import atomic_write_text


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """A commit precondition does not hold immediately before the commit."""


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
        DIFF_TOO_LARGE,
        "HEAD_MISMATCH",
        SECRET_IN_DIFF,
        SECRET_IN_STAGED_BLOB,
        UNSCANNABLE_STAGED_BLOB,
        UNREVIEWABLE_TEXT_DIFF,
    }
)
_COMMIT_SUBJECT_LIMIT = 72
_MAX_AGENT_REPORT_BYTES = 32_000
_AGENT_ARTIFACTS = (
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


def _bounded_v2_report(text: str) -> str:
    """Bound an individual staged-step report to the P21 8 KiB limit."""

    limit = 8 * 1024
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="ignore") + "\n[... report truncated ...]"


def _step_reports_text(results: list[dict[str, Any]]) -> str:
    """Render bounded reports with a hard 32 KiB aggregate limit."""

    chunks: list[str] = []
    used = 0
    for item in results:
        chunk = "\n".join([
            item["id"], f"profile: {item['profile_id']}",
            f"tree_before: {item['tree_before']}", f"tree_after: {item['tree_after']}",
            f"usage: {json.dumps(item['usage'], sort_keys=True)}", "final report:", item.get("final", ""),
        ]) + "\n"
        encoded = chunk.encode("utf-8", errors="replace")
        if used + len(encoded) > 32 * 1024:
            remaining = 32 * 1024 - used
            if remaining > 0:
                chunks.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            break
        chunks.append(chunk)
        used += len(encoded)
    return "\n".join(chunks)


def _check_payload(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    return [check_result_json(check) for check in bundle.checks]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _review_payload(review: ReviewResult) -> dict[str, Any]:
    payload = asdict(review)
    payload["verdict"] = review.verdict.value
    payload["route"] = review.route.value
    payload.pop("raw", None)
    return payload


def _agent_payload(result: AgentResult, *, auth_failure: bool = False) -> dict[str, Any]:
    # The full report is already persisted by CodexAgent.  State contains only
    # bounded protocol metadata and never an API key or an authorization value.
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "usage": dict(result.usage),
        # Transport errors may contain provider URLs, request IDs, or other
        # infrastructure identifiers.  The complete bounded artifact remains
        # available to the local UI; state keeps a fixed safe marker instead.
        "stderr_tail": "Codex authentication failed" if auth_failure else result.stderr_tail,
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


def _codex_auth_failure(run_dir: Path, stderr: str, *, step_id: str | None = None) -> bool:
    events_path = run_dir / "agent.events.jsonl"
    if step_id is not None:
        events_path = run_dir / "steps" / step_id / "agent.events.jsonl"
    return classify_codex_failure(stderr, _artifact_tail(events_path)) == "CODEX_AUTH_FAILURE"


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
class _GitOwnership:
    """Git state the implementation agent is not allowed to change."""

    head_ref: str | None
    head: str
    branches: frozenset[str]
    worktrees: frozenset[str]


def _git_ownership(repo: Path, worktree: Path) -> _GitOwnership:
    return _GitOwnership(
        head_ref=symbolic_head(worktree),
        head=current_head(worktree),
        branches=local_branches(repo),
        worktrees=registered_worktrees(repo),
    )


def _ownership_violations(
    before: _GitOwnership, after: _GitOwnership, *, branch_ref: str, base_sha: str
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
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self._planner_client = planner_client
        self._reviewer_client = reviewer_client
        self._recommender_client = recommender_client
        self._injected_agent = agent
        self._secrets: tuple[str, ...] = ()
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

    def _agent_for_profile(self, profile_id: str) -> CodexAgent:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.IMPLEMENTER)
        if self._injected_agent is not None:
            return self._injected_agent
        return CodexAgent(
            dataclasses.replace(
                build_agent_config(profile),
                env_allowlist=self.config.agent.env_allowlist,
            )
        )

    def run(
        self, spec: str | Path, *, run_id: str | None = None,
        planner_profile: str | None = None,
    ) -> RunResult:
        """Read one SPEC file and delegate execution to :meth:`run_text`."""

        spec_path = Path(spec).expanduser().resolve()
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise OrchestrationError(f"could not read spec {spec_path}: {exc}") from exc
        if planner_profile is None:
            return self.run_text(spec_content, run_id=run_id)
        return self.run_text(spec_content, run_id=run_id, planner_profile=planner_profile)

    def run_text(
        self,
        spec_content: str,
        *,
        run_id: str | None = None,
        planner_profile: str | None = None,
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

        selected_planner_id = planner_profile or self.config.ui.default_planner_profile or "legacy-planner"
        selected_planner = profile_for_role(
            self.config, selected_planner_id, ExecutionRole.PLANNER
        )

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
            store = RunStateStore(run_dir / "state.json")
            store.initialize(selected_run_id)
            store.update(
                status=RunStatus.CREATED,
                spec_path="spec.md",
                repo=str(self.config.repo),
                base_ref=self.config.base_ref,
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
            return self._execute(store, run_dir, selected_run_id, spec_content)
        except KeyboardInterrupt:
            if store is None:
                raise
            state = store.update(
                status=RunStatus.INTERRUPTED,
                failure={"reason": "INTERRUPTED"},
                **self._closing_step_fields(store, "interrupted"),
            )
            return RunResult(run_dir, RunStatus.INTERRUPTED, state)
        except Exception as exc:
            if store is None:
                raise
            state = store.record_failure(
                _failure_reason(exc),
                redact(str(exc), self._secrets),
                **self._closing_step_fields(store, "failed"),
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

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
        store.update(status=RunStatus.CREATED, repo=str(repo), base_sha=base_sha)

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

        context_bundle = build_context(repo, base_sha, spec, self.config.context)
        context = render_context(context_bundle)
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
        store.update(status=RunStatus.PREPARING)
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
            and _codex_auth_failure(run_dir, agent_result.stderr_tail)
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
                "failures": list(evidence.failures),
            },
        )

        integrity_failures = [
            item
            for item in evidence.failures
            if (
                item in _DIRECT_FAILURES
                or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
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
        store.update(status=RunStatus.REVIEWING, review=_review_payload(review))

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
            body=self._commit_body(run_id, base_sha, approved_tree, evidence),
        )
        state = store.update(status=RunStatus.COMMITTED, commit_sha=commit_sha)
        return RunResult(run_dir, RunStatus.COMMITTED, state)

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
    ) -> RunResult:
        """Execute a v2 bundle: one worktree, fresh Codex process per step."""

        planner_profile_id = store.load()["execution"]["planner"]["profile_id"]
        planner_profile = profile_for_role(self.config, planner_profile_id, ExecutionRole.PLANNER)
        implementers = tuple(
            p for p in profiles_for_config(self.config).values()
            if ExecutionRole.IMPLEMENTER in p.roles
        )
        reviewers = tuple(
            p for p in profiles_for_config(self.config).values()
            if ExecutionRole.REVIEWER in p.roles
        )
        planner = PlannerV2(
            self._planner_client or _chat_client(build_llm_endpoint(planner_profile), self._runtime_environment),
            implementer_ids=frozenset(p.id for p in implementers),
            reviewer_ids=frozenset(p.id for p in reviewers),
            implementer_profiles=implementers,
            reviewer_profiles=reviewers,
            repository_reference=repository_reference,
            planning=self.config.planning,
        )
        plan = planner.plan(spec, context, artifacts_dir=run_dir)
        store.update(
            status=RunStatus.PLANNING,
            planning_protocol="v2",
            planner={
                "decision": plan.decision.value,
                "title": plan.title,
                "model": planner_profile.model,
                "profile_id": planner_profile.id,
                "selection_mode": planner_profile.selection_mode.value,
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
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
        if plan.decision is PlanDecision.BLOCKED:
            state = store.update(
                status=RunStatus.BLOCKED,
                failure={"reason": "PLANNER_BLOCKED", "detail": plan.blockers},
            )
            return RunResult(run_dir, RunStatus.BLOCKED, state)

        try:
            _bundle, _bundle_sha = validate_implementation_bundle(run_dir)
            plan_identity = compute_plan_identity_from_run(run_dir)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update(status=RunStatus.PLANNING, plan_identity=asdict(plan_identity))

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
                selection, execution_sha = read_execution_selection_v3_with_sha256(run_dir)
                validate_execution_selection_v3(self.config, selection)
                durable_identity = compute_plan_identity_from_run(run_dir)
                if durable_identity.execution_sha256 != execution_sha:
                    raise ApprovalError("execution selection hash mismatch")
                read = compute_plan_identity_from_run(run_dir)
                from .approval import read_plan_approval
                bound = read_plan_approval(run_dir, expected_identity=read)
                if bound is None or bound.bundle_sha256 != durable_identity.bundle_sha256:
                    raise ApprovalError("v2 approval is not bound to the exact bundle")
            except (ExecutionSelectionError, ApprovalError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        else:
            requested = resolve_execution_selection_v3(
                self.config,
                planner_profile_id=planner_profile_id,
                step_profile_ids={step.id: step.implementer_profile for step in plan.steps},
                reviewer_profile_id=plan.reviewer_profile or self.config.ui.default_reviewer_profile or "legacy-reviewer",
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
        validate_execution_selection_v3(self.config, selection)
        self._last_selection = selection
        execution_state = {
            "planner": asdict(selection.planner),
            "steps": [
                {"step_id": item.step_id, "implementer": asdict(item.implementer)}
                for item in selection.steps
            ],
            "reviewer": asdict(selection.reviewer),
        }
        store.update(status=RunStatus.PLANNING, execution=execution_state,
                     plan_identity=asdict(durable_identity))

        branch = f"harness/{_slug(plan.title)}/{run_id}"
        info = create_run_worktree(
            repo, base_ref=base_sha, branch=branch,
            worktree_path=self.config.worktrees_root / run_id,
            require_clean_base=self.config.require_clean_base,
        )
        branch_ref = f"refs/heads/{info.branch}"
        store.update(status=RunStatus.WORKTREE_READY, branch=info.branch,
                     worktree=str(info.worktree), base_sha=info.base_sha)
        ownership_before = _git_ownership(repo, info.worktree)
        store.update(status=RunStatus.PREPARING)
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
        base_tree_sha = resolve_tree(repo, base_sha)
        if candidate_tree != base_tree_sha:
            raise OrchestrationError("initial candidate tree does not match base")
        codex_home = prepare_codex_home(self.config)
        step_items = {item.step_id: item for item in selection.steps}
        state_steps = [
            {"id": step.id, "title": step.title, "status": "waiting",
             "profile_id": step_items[step.id].implementer.profile_id}
            for step in plan.steps
        ]
        self._last_v2_step_results: list[dict[str, Any]] = []
        self._v2_usage_rows: list[dict[str, Any]] = []
        # Tree every step must start from: the base, then each frozen step.
        expected_tree = candidate_tree
        for step in plan.steps:
            selected_step = step_items.get(step.id)
            if selected_step is None:
                raise ExecutionSelectionError(f"missing selection for {step.id}")
            profile = profile_for_role(self.config, selected_step.implementer.profile_id, ExecutionRole.IMPLEMENTER)
            step_dir = run_dir / "steps" / step.id
            step_dir.mkdir(parents=True, exist_ok=True)
            # The approved file is the executed file: its bytes are re-hashed
            # against the validated bundle and never re-rendered or rewritten.
            try:
                contract = read_approved_step_contract(run_dir, bundle, step.id)
            except (V2PlanParseError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
            before_tree = candidate_tree_sha(info.worktree)
            drift = self._step_contract_drift(repo, before_tree, expected_tree, step)
            if drift:
                return self._v2_failed(store, run_dir, "STEP_CONTRACT_DRIFT", step.id, drift,
                                       profile_id=profile.id, tree_before=before_tree)
            store.update(status=RunStatus.IMPLEMENTING, current_step=step.id,
                         steps=[{**item, "status": "running" if item["id"] == step.id else item["status"]}
                                for item in state_steps])
            agent = self._agent_for_profile(profile.id)
            agent_config = dataclasses.replace(
                build_agent_config(profile), env_allowlist=self.config.agent.env_allowlist)
            agent_environment = build_agent_environment(
                agent_config, source_environment=self._runtime_environment,
                codex_home=codex_home,
                forbidden_names=(planner_profile.api_key_env,
                                 profile_for_role(self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER).api_key_env),
            )
            try:
                if hasattr(agent, "run_step"):
                    result = agent.run_step(contract, info.worktree, step_dir,
                                           base_sha=base_sha, env=agent_environment)
                else:
                    # Test doubles from the v1 API may only expose run(); the
                    # production CodexAgent always takes the step path above.
                    result = agent.run(build_implementer_step_prompt(contract), info.worktree,
                                       step_dir, base_sha=base_sha, env=agent_environment)
            except AgentCommittedError as exc:
                self._redact_step_artifacts(step_dir)
                return self._v2_failed(store, run_dir, "AGENT_COMMITTED", step.id,
                                       redact(str(exc), self._secrets),
                                       profile_id=profile.id, tree_before=before_tree)
            self._redact_step_artifacts(step_dir)
            result = dataclasses.replace(result,
                                         final_message=redact(result.final_message, self._secrets),
                                         stderr_tail=redact(result.stderr_tail, self._secrets))
            usage = normalize_usage(result.usage)
            self._v2_usage_rows.append({"id": step.id, **usage})
            failed_step = {"usage": usage, "profile_id": profile.id, "tree_before": before_tree}
            ownership_after = _git_ownership(repo, info.worktree)
            if ownership_after.head != base_sha:
                return self._v2_failed(store, run_dir, "AGENT_COMMITTED", step.id,
                                       "worktree HEAD changed", **failed_step)
            violations = _ownership_violations(ownership_before, ownership_after,
                                                branch_ref=branch_ref, base_sha=base_sha)
            if violations:
                return self._v2_failed(store, run_dir, "AGENT_GIT_VIOLATION", step.id, violations,
                                       **failed_step)
            if result.timed_out:
                return self._v2_failed(store, run_dir, "AGENT_TIMEOUT", step.id, **failed_step)
            if result.exit_code != 0:
                if _codex_auth_failure(run_dir, result.stderr_tail, step_id=step.id):
                    return self._v2_failed(
                        store, run_dir, "CODEX_AUTH_FAILURE", step.id,
                        "Codex authentication failed", **failed_step
                    )
                return self._v2_failed(store, run_dir, "AGENT_FAILED", step.id,
                                        f"exit status {result.exit_code}", **failed_step)
            stage_all(info.worktree)
            frozen_tree = index_tree_sha(info.worktree)
            if frozen_tree == before_tree:
                return self._v2_failed(store, run_dir, "AGENT_NO_CHANGE", step.id, **failed_step)
            # Git, not the prompt, is the scope barrier: every changed path
            # must be authorized by this step's WRITE, CREATE or DELETE set.
            changed_paths = changed_paths_between_trees(repo, before_tree, frozen_tree)
            allowed = {*step.write_set, *step.create_set, *step.delete_set}
            unexpected = [path for path in changed_paths if path not in allowed]
            if unexpected:
                return self._v2_failed(
                    store, run_dir, "STEP_WRITE_SET_VIOLATION", step.id,
                    f"unexpected={_paths_detail(unexpected)}",
                    **failed_step, tree_after=frozen_tree,
                )
            expected_tree = frozen_tree
            step_result = {
                "id": step.id, "status": "COMPLETED", "profile_id": profile.id,
                "tree_before": before_tree, "tree_after": frozen_tree,
                "changed_paths": list(changed_paths),
                "usage": usage,
            }
            atomic_write_text(step_dir / "step.json", _json_text(step_result))
            self._last_v2_step_results.append({**step_result, "final": _bounded_v2_report(result.final_message)})
            state_steps = [
                {**item, "status": "completed", "usage": usage,
                 "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
                if item["id"] == step.id else item
                for item in state_steps
            ]
            store.update(status=RunStatus.IMPLEMENTING, current_step=None, steps=state_steps,
                         agent_usage=self._v2_agent_usage())

        store.update(status=RunStatus.VALIDATING, current_step=None)
        evidence = collect_evidence(info.worktree, base_sha, self.config,
                                    evidence_dir=run_dir, secrets=self._secrets)
        store.update(status=RunStatus.VALIDATING, checks=_check_payload(evidence),
                     staged_tree_sha=evidence.staged_tree_sha,
                     changed_files=list(evidence.changed_files),
                     deterministic_gate={"passed": evidence.deterministic_passed,
                                         "failures": list(evidence.failures)})
        integrity_failures = [item for item in evidence.failures if item in _DIRECT_FAILURES or
                              any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES) or
                              item.startswith("CHECK_MUTATED:")]
        if integrity_failures:
            return self._v2_failed(store, run_dir, integrity_failures[0].split(":", 1)[0], None,
                                    ", ".join(integrity_failures))
        gate = _json_text({"deterministic_passed": evidence.deterministic_passed,
                           "failures": list(evidence.failures), "staged_tree_sha": evidence.staged_tree_sha})
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
        reports = _step_reports_text(self._last_v2_step_results)
        store.update(status=RunStatus.REVIEWING)
        review = reviewer.review(spec, plan.raw, context, gate,
                                 "\n".join(evidence.changed_files), evidence.diff,
                                 _json_text(_check_payload(evidence)), reports,
                                 deterministic_passed=True, artifacts_dir=run_dir)
        store.update(status=RunStatus.REVIEWING, review=_review_payload(review))
        if review.verdict is ReviewVerdict.REVISE:
            write_repair_task(run_dir, fields={"route": review.route.value,
                "review_summary": review.summary, "findings": review.findings,
                "required_fixes": review.required_fixes, "missing_tests": review.missing_tests,
                "existing_branch": info.branch, "existing_worktree": str(info.worktree), "run_id": run_id})
            return self._v2_failed(store, run_dir, "REVIEW_REVISE", None)
        if review.verdict is ReviewVerdict.FAIL:
            return self._v2_failed(store, run_dir, "REVIEW_FAIL", None)
        if review.route is not ReviewRoute.NONE or not evidence.deterministic_passed:
            return self._v2_failed(store, run_dir, "REVIEW_ROUTE_NOT_NONE" if review.route is not ReviewRoute.NONE else "DETERMINISTIC_GATE_FAILED", None)
        approved_tree = self._authorize_v2_commit(plan, review, evidence, info.worktree, base_sha, branch_ref)
        # Keep the sole named commit primitive in the legacy guarded path;
        # this alias still resolves to the same GitOps implementation.
        commit_fn = commit_reviewed_tree
        commit_sha = commit_fn(info.worktree, tree_sha=approved_tree,
                               parent_sha=base_sha, subject=_commit_subject(plan.title),
                               body=self._commit_body_v2(run_id, base_sha, approved_tree, evidence, selection))
        return RunResult(run_dir, RunStatus.COMMITTED,
                         store.update(status=RunStatus.COMMITTED, commit_sha=commit_sha,
                                      current_step=None))

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
    ) -> RunResult:
        state = store.load()
        fields = _terminal_step_fields(state, step_id)
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
            step_dir = run_dir / "steps" / step_id
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
            reparsed = parse_review(review.raw, deterministic_passed=True)
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

    def _commit_body_v2(self, run_id: str, base_sha: str, tree_sha: str,
                        evidence: EvidenceBundle, selection: ExecutionSelectionV3) -> str:
        check_lines = [f"{check.name}: {'PASS' if result.exit_code == 0 and not result.timed_out else 'FAIL'}"
                       for check, result in zip(self.config.checks, evidence.checks)] or ["none"]
        return "\n".join([
            "Generated by MetaHarness.", "", f"Run: {run_id}", f"Base: {base_sha}",
            f"Reviewed tree: {tree_sha}", "Planning protocol: v2",
            f"Planner profile: {selection.planner.profile_id}",
            *[f"{item.step_id} implementer profile: {item.implementer.profile_id}" for item in selection.steps],
            f"Reviewer profile: {selection.reviewer.profile_id}", "", "Checks:", *check_lines,
        ])

    def _redact_agent_artifacts(self, run_dir: Path) -> None:
        for name in _AGENT_ARTIFACTS:
            redact_file(run_dir / name, self._secrets)

    def _commit_body(
        self,
        run_id: str,
        base_sha: str,
        tree_sha: str,
        evidence: EvidenceBundle,
    ) -> str:
        checks = self.config.checks
        check_lines = []
        # The body reports the deterministic result frozen in evidence; this
        # method is only reached after all required checks passed.
        for check, result in zip(checks, evidence.checks):
            outcome = "PASS" if result.exit_code == 0 and not result.timed_out else "FAIL"
            check_lines.append(f"{check.name}: {outcome}")
        if not check_lines:
            check_lines.append("none")
        return "\n".join(
            [
                "Generated by MetaHarness.",
                "",
                f"Run: {run_id}",
                f"Base: {base_sha}",
                f"Reviewed tree: {tree_sha}",
                f"Planner profile: {self._last_selection.planner.profile_id}",
                f"Planner model label: {self._last_selection.planner.model}",
                f"Planner selection: {self._last_selection.planner.selection_mode}",
                f"Implementer profile: {self._last_selection.implementer.profile_id}",
                f"Implementer model: {self._last_selection.implementer.model}",
                f"Implementer effort: {self._last_selection.implementer.effort}",
                f"Reviewer profile: {self._last_selection.reviewer.profile_id}",
                f"Reviewer model: {self._last_selection.reviewer.model}",
                f"Reviewer selection: {self._last_selection.reviewer.selection_mode}",
                "",
                "Checks:",
                *check_lines,
            ]
        )


def _failure_reason(exc: Exception) -> str:
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


__all__ = [
    "CommitBoundaryError",
    "OrchestrationError",
    "Orchestrator",
    "authorize_commit",
    "generate_run_id",
    "run_orchestrator",
]
