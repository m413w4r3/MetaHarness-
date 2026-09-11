"""The single-task, single-agent MetaHarness V0 state machine."""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent.base import AgentError, AgentResult
from .agent.codex import AgentCommittedError, CodexAgent, build_agent_environment
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
    commit_reviewed_tree,
    create_run_worktree,
    current_head,
    git_root,
    index_tree_sha,
    local_branches,
    registered_worktrees,
    resolve_commit,
    status_porcelain,
    symbolic_head,
)
from .llm.chat import LLMError, OpenAIChatTextClient
from .execution_selection import (
    read_execution_selection,
    resolve_execution_selection,
    write_execution_selection,
)
from .models import ExecutionRole, HarnessConfig, ReviewRoute, ReviewVerdict, RunStatus
from .planning import (
    PlanDecision,
    Planner,
    PlanParseError,
    TaskPlan,
    render_implementation_contract,
)
from .redaction import config_secret_values, redact, redact_file
from .profiles import build_agent_config, build_llm_endpoint, profile_for_role
from .result import RunResult, write_repair_task
from .review import Reviewer, ReviewParseError, ReviewResult, blocking_finding_lines, parse_review
from .state import RunStateStore
from .validation import ValidationError, check_result_json


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """A commit precondition does not hold immediately before the commit."""


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


def _agent_payload(result: AgentResult) -> dict[str, Any]:
    # The full report is already persisted by CodexAgent.  State contains only
    # bounded protocol metadata and never an API key or an authorization value.
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "usage": dict(result.usage),
        "stderr_tail": result.stderr_tail,
    }


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
        agent: CodexAgent | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self._planner_client = planner_client
        self._reviewer_client = reviewer_client
        self._injected_agent = agent
        self._secrets: tuple[str, ...] = ()

    def _planner_for_profile(self, profile_id: str) -> Planner:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.PLANNER)
        client = self._planner_client
        if client is None:
            client = OpenAIChatTextClient(build_llm_endpoint(profile))
        return Planner(client, allow_format_repair=False)

    def _reviewer_for_profile(self, profile_id: str) -> Reviewer:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.REVIEWER)
        client = self._reviewer_client
        if client is None:
            client = OpenAIChatTextClient(build_llm_endpoint(profile))
        return Reviewer(client, allow_format_repair=False)

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
        self._secrets = config_secret_values(self.config)

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
            )
            return RunResult(run_dir, RunStatus.INTERRUPTED, state)
        except Exception as exc:
            if store is None:
                raise
            state = store.record_failure(
                _failure_reason(exc), redact(str(exc), self._secrets)
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

    execute = run

    def _execute(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
    ) -> RunResult:
        repo = git_root(self.config.repo)
        planner_selection = store.load().get("execution", {}).get("planner", {})
        planner_profile_id = planner_selection.get("profile_id")
        if not isinstance(planner_profile_id, str):
            raise OrchestrationError("run has no planner profile")
        planner_profile = profile_for_role(
            self.config, planner_profile_id, ExecutionRole.PLANNER
        )
        planner = self._planner_for_profile(planner_profile_id)
        if self.config.require_clean_base:
            assert_clean(repo)
        base_sha = resolve_commit(repo, self.config.base_ref)
        store.update(status=RunStatus.CREATED, repo=str(repo), base_sha=base_sha)

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

            try:
                selection = read_execution_selection(run_dir)
            except ValueError:
                # P15 CLI approvals have no profile snapshot.  Keep those
                # historic approvals readable by materializing current
                # defaults before continuing; all P16 web approvals already
                # contain the durable snapshot and use schema v2.
                if approval.execution_sha256 is not None:
                    raise
                selection = resolve_execution_selection(
                    self.config,
                    planner_profile_id=planner_profile.id,
                    implementer_profile_id=self.config.ui.default_implementer_profile or "legacy-implementer",
                    reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
                )
                write_execution_selection(run_dir, selection)
            durable_identity = compute_plan_identity_from_run(run_dir)
            if approval.raw_sha256 != durable_identity.raw_sha256 or approval.contract_sha256 != durable_identity.contract_sha256 or (approval.execution_sha256 is not None and approval.execution_sha256 != durable_identity.execution_sha256):
                raise ApprovalError("approval does not match durable execution selection")
        else:
            selection = resolve_execution_selection(
                self.config,
                planner_profile_id=planner_profile.id,
                implementer_profile_id=self.config.ui.default_implementer_profile or "legacy-implementer",
                reviewer_profile_id=self.config.ui.default_reviewer_profile or "legacy-reviewer",
            )
            write_execution_selection(run_dir, selection)
            durable_identity = compute_plan_identity_from_run(run_dir)
            store.update(
                status=RunStatus.PLANNING,
                plan_identity=dataclasses.asdict(durable_identity),
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
        store.update(status=RunStatus.IMPLEMENTING)
        implementation_contract = render_implementation_contract(plan)
        agent = self._agent_for_profile(selection.implementer.profile_id)
        reviewer = self._reviewer_for_profile(selection.reviewer.profile_id)
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
                forbidden_names=(
                    planner_profile.api_key_env,
                    profile_for_role(self.config, selection.reviewer.profile_id, ExecutionRole.REVIEWER).api_key_env,
                ),
            )
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
        self._redact_agent_artifacts(run_dir)
        agent_result = dataclasses.replace(
            agent_result,
            final_message=redact(agent_result.final_message, self._secrets),
            stderr_tail=redact(agent_result.stderr_tail, self._secrets),
        )
        store.update(status=RunStatus.IMPLEMENTING, agent=_agent_payload(agent_result))
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
            state = store.record_failure(
                "AGENT_FAILED", f"exit status {agent_result.exit_code}"
            )
            return RunResult(run_dir, RunStatus.FAILED, state)

        store.update(status=RunStatus.VALIDATING)
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
    if isinstance(exc, PlanParseError):
        return "PLANNER_OUTPUT_INVALID"
    if isinstance(exc, ReviewParseError):
        return "REVIEWER_OUTPUT_INVALID"
    if isinstance(exc, LLMError):
        return "LLM_FAILURE"
    if isinstance(exc, ValidationError):
        return "CHECK_SETUP_INVALID"
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
