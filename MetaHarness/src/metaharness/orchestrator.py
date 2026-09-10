"""The single-task, single-agent MetaHarness V0 state machine."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent.base import AgentError, AgentResult
from .agent.codex import CodexAgent
from .config import load_config
from .context import build_context, render_context
from .evidence import DIFF_TOO_LARGE, EvidenceBundle, collect_evidence
from .gitops import (
    GitError,
    assert_clean,
    commit_staged,
    create_run_worktree,
    current_head,
    git_root,
    index_tree_sha,
    resolve_commit,
    status_porcelain,
)
from .llm.chat import OpenAIChatTextClient
from .models import HarnessConfig, ReviewRoute, ReviewVerdict, RunStatus
from .planning import PlanDecision, Planner, TaskPlan
from .result import RunResult, write_repair_task
from .review import Reviewer, ReviewResult
from .state import RunStateStore
from .validation import check_result_json


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """The reviewed index or worktree changed before the commit."""


def _generated_run_id() -> str:
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
    return value


def _slug(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9]+", "-", value.casefold()).strip("-")
    return (candidate[:60] or "task")


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
        self.planner = Planner(
            planner_client if planner_client is not None else OpenAIChatTextClient(config.planner),
            allow_format_repair=False,
        )
        self.reviewer = Reviewer(
            reviewer_client if reviewer_client is not None else OpenAIChatTextClient(config.reviewer),
            allow_format_repair=False,
        )
        self.agent = agent or CodexAgent(config.agent)

    def run(self, spec: str | Path, *, run_id: str | None = None) -> RunResult:
        """Run one SPEC and return its durable final state.

        A worktree is deliberately never removed.  This keeps failed and
        interrupted runs inspectable and makes the run directory the handoff
        point for operators.
        """

        spec_path = Path(spec).expanduser().resolve()
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise OrchestrationError(f"could not read spec {spec_path}: {exc}") from exc
        if not spec_content.strip():
            raise OrchestrationError("spec must not be empty")

        selected_run_id = _safe_run_id(run_id) if run_id is not None else _generated_run_id()
        run_dir = (self.config.runs_root / selected_run_id).expanduser().resolve()
        if run_dir.exists():
            raise OrchestrationError(f"run directory already exists: {run_dir}")

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
            )
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
            state = store.record_failure(_failure_reason(exc), str(exc))
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

        plan = self.planner.plan(spec, context, artifacts_dir=run_dir)
        store.update(
            status=RunStatus.PLANNING,
            planner={
                "decision": plan.decision.value,
                "title": plan.title,
                "model": self.config.planner.model,
            },
        )
        if plan.decision is PlanDecision.BLOCKED:
            state = store.update(
                status=RunStatus.BLOCKED,
                failure={"reason": "PLANNER_BLOCKED", "detail": plan.blockers},
            )
            return RunResult(run_dir, RunStatus.BLOCKED, state)

        branch = f"harness/{_slug(plan.title)}/{run_id}"
        worktree_path = self.config.worktrees_root / run_id
        info = create_run_worktree(
            repo,
            base_ref=base_sha,
            branch=branch,
            worktree_path=worktree_path,
            require_clean_base=self.config.require_clean_base,
        )
        store.update(
            status=RunStatus.WORKTREE_READY,
            branch=info.branch,
            worktree=str(info.worktree),
            base_sha=info.base_sha,
        )

        store.update(status=RunStatus.IMPLEMENTING)
        agent_result = self.agent.run(
            plan.raw,
            info.worktree,
            run_dir,
            base_sha=base_sha,
        )
        store.update(status=RunStatus.IMPLEMENTING, agent=_agent_payload(agent_result))
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

        direct_failures = {"EMPTY_DIFF", DIFF_TOO_LARGE, "HEAD_MISMATCH", "CHECK_MUTATED"}
        integrity_failures = [item for item in evidence.failures if item in direct_failures]
        if integrity_failures:
            state = store.record_failure(
                integrity_failures[0], ", ".join(integrity_failures)
            )
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
        review = self.reviewer.review(
            spec,
            plan.raw,
            context,
            gate,
            "\n".join(evidence.changed_files),
            evidence.diff,
            checks_text,
            agent_result.final_message,
            # The deterministic gate is evaluated cumulatively below.  Passing
            # it here allows a reviewer PASS to remain a useful diagnostic on
            # a failed check, as required by V0.
            deterministic_passed=True,
            artifacts_dir=run_dir,
        )
        store.update(status=RunStatus.REVIEWING, review=_review_payload(review))

        if review.verdict is ReviewVerdict.REVISE:
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
        self._assert_commit_boundary(info.worktree, base_sha, evidence)
        body = self._commit_body(run_id, base_sha, plan, evidence)
        commit_sha = commit_staged(info.worktree, subject=plan.title, body=body)
        state = store.update(status=RunStatus.COMMITTED, commit_sha=commit_sha)
        return RunResult(run_dir, RunStatus.COMMITTED, state)

    @staticmethod
    def _assert_commit_boundary(
        worktree: Path, base_sha: str, evidence: EvidenceBundle
    ) -> None:
        if current_head(worktree) != base_sha:
            raise CommitBoundaryError("HEAD changed after review")
        if index_tree_sha(worktree) != evidence.staged_tree_sha:
            raise CommitBoundaryError("index changed after review")
        problems = _status_has_unstaged_or_untracked(status_porcelain(worktree))
        if problems:
            raise CommitBoundaryError("; ".join(problems))

    def _commit_body(
        self,
        run_id: str,
        base_sha: str,
        plan: TaskPlan,
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
                f"Planner: {self.config.planner.model}",
                f"Implementer: {self.config.agent.model} / {self.config.agent.effort}",
                f"Reviewer: {self.config.reviewer.model}",
                "",
                "Checks:",
                *check_lines,
            ]
        )


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, GitError):
        return "GIT_FAILURE"
    if isinstance(exc, AgentError):
        return getattr(exc, "code", "AGENT_FAILURE")
    if isinstance(exc, CommitBoundaryError):
        return "TOCTOU_FAILURE"
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
    "run_orchestrator",
]
