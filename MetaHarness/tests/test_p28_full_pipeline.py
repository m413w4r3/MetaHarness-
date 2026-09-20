"""P28 full-pipeline coverage: explicit revision, C01/C02 parity, cycle UI.

Every run uses a real temporary Git repository, a real local bare remote,
the real orchestrator state machine and the real Git primitives.  Planner,
Luna (Codex), Claude and reviewers are in-process fakes: no network, no LLM.
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import cli  # noqa: E402
from metaharness.approval import (  # noqa: E402
    read_check_authority,
    write_check_authority,
)
from metaharness import orchestrator as orchestrator_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.claude.agent import (  # noqa: E402
    ClaudeCodeAgent,
    ClaudeResult,
    build_claude_environment,
    parse_scope_request,
)
from metaharness.claude.runtime import prepare_claude_home  # noqa: E402
from metaharness.config import ConfigError, load_config  # noqa: E402
from metaharness.execution_selection import (  # noqa: E402
    ExecutionSelectionError,
    resolve_execution_selection_v4,
)
from metaharness.gitops import (  # noqa: E402
    RepositoryReference,
    build_repository_reference,
    run_branch_web_url,
)
from metaharness.llm.chat import (  # noqa: E402
    LLMConversationHandle,
    TextLLMResult,
)
from metaharness.models import (  # noqa: E402
    AgentConfig,
    ClaudeRuntimeConfig,
    CodexRuntimeConfig,
    ContextConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ProfileDriver,
    RepositoryConfig,
    RevisionConfig,
    RunStatus,
    SelectionMode,
)
from metaharness.orchestrator import Orchestrator  # noqa: E402
from metaharness import planning_v2  # noqa: E402
from metaharness.resume import (  # noqa: E402
    ResumeNotAllowedError,
    ResumePhase,
    read_checkpoint,
    read_checkpoint_record,
    resume_info,
)
from metaharness.run_options import RunOptions  # noqa: E402
from metaharness.web.api import (  # noqa: E402
    WebAPIError,
    approve_run,
    cycle_step_progress_tail,
    get_run,
)
from metaharness.web.pages import render_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "autowork.toml"
SPEC = "Implement the P28 feature end to end.\n"

PASS = """VERDICT: PASS
ROUTE: NONE
SUMMARY: The implementation is acceptable.
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""
REVISE_IMPLEMENTATION = """VERDICT: REVISE
ROUTE: IMPLEMENTATION
SUMMARY: A concrete defect remains.
FINDINGS: MAJOR | behavior | the candidate is incomplete | implement the fix
REQUIRED FIXES: Correct src/a.py.
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""


def git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    ).stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def step_block(
    number: int,
    *,
    read: tuple[str, ...] = ("src/a.py",),
    write_set: tuple[str, ...] = ("src/a.py",),
    create: tuple[str, ...] = (),
    delete: tuple[str, ...] = (),
    operation: str = "Perform",
) -> str:
    step_id = f"S{number:02d}"
    return "\n".join([
        f"BEGIN STEP {step_id}",
        f"TITLE: {operation} step {number}",
        "IMPLEMENTER_PROFILE: luna",
        f"DEPENDS_ON: {'NONE' if number == 1 else f'S{number - 1:02d}'}",
        "",
        "OBJECTIVE",
        f"{operation} objective {number}.",
        "",
        "READ_SET",
        *[f"- {path} :: anchor-{number}" for path in read],
        "",
        "WRITE_SET",
        *([f"- {path}" for path in write_set] or ["NONE"]),
        "",
        "CREATE_SET",
        *([f"- {path}" for path in create] or ["NONE"]),
        "",
        "DELETE_SET",
        *([f"- {path}" for path in delete] or ["NONE"]),
        "",
        "INSTRUCTIONS",
        f"1. {operation} operation {number} exactly.",
        "",
        "VERIFY",
        "- run the configured gate",
        "",
        "FORBIDDEN",
        "- Do not touch any other file.",
        "",
        f"END STEP {step_id}",
    ])


def plan_text(*steps: str, title: str = "P28 feature") -> str:
    return "\n".join([
        "META PLAN v2", "", "STATUS: READY", f"TITLE: {title}", "",
        "OBJECTIVE", "Deliver the P28 feature.", "",
        "CONSTRAINTS", "NONE", "",
        f"EXECUTION_MODE: {'SINGLE' if len(steps) == 1 else 'STAGED'}",
        f"STEP_COUNT: {len(steps)}",
        "REVIEWER_PROFILE: reviewer", "",
        "\n\n".join(steps), "",
        "ACCEPTANCE", "The gate is green.", "",
        "TESTS", "Run the configured gate.", "",
        "RISKS", "NONE", "",
        "BLOCKERS", "NONE", "",
        "END META PLAN", "",
    ])


SINGLE_PLAN = plan_text(step_block(1))
REPAIR_PLAN = plan_text(step_block(1, operation="Repair"), title="P28 repair")
STAGED_PLAN = plan_text(
    step_block(1),
    step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
    step_block(3, read=("src/a.py",), write_set=(), create=("src/c.py",)),
)


class QueueClient:
    """Planner/reviewer double: one queued answer per call, shared event log."""

    def __init__(self, name: str, responses: list[str], log: list[str]):
        self.name = name
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.log = log

    def complete(self, prompt: str) -> TextLLMResult:
        self.prompts.append(prompt)
        text = self.responses.pop(0)
        self.log.append(f"{self.name}:{text.splitlines()[0]}")
        return TextLLMResult(
            text=text, model="offline", usage={"input_tokens": 10, "output_tokens": 2},
            raw_response={},
        )


class FakeLuna:
    """In-process Codex double; the artifact directory identifies the cycle."""

    def __init__(self, behaviors: dict[tuple[int, str], Any]):
        self.behaviors = behaviors
        self.calls: list[dict[str, Any]] = []

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, *,
                 base_sha: str | None = None, env: dict[str, str] | None = None,
                 retry_addendum: str | None = None) -> AgentResult:
        directory = Path(artifacts_dir)
        cycle = 2 if directory.parent.parent.name == "C02" else 1
        step_id = directory.name
        root = Path(worktree)
        attempt = 1 + sum(
            1 for call in self.calls if (call["cycle"], call["step"]) == (cycle, step_id)
        )
        self.calls.append({"cycle": cycle, "step": step_id, "contract": contract,
                           "env": dict(env or {}), "dir": directory,
                           "retry_addendum": retry_addendum, "attempt": attempt})
        behavior = self.behaviors.get((cycle, step_id, attempt), None)
        if behavior is None:
            behavior = self.behaviors.get((cycle, step_id), "nochange")
        usage = {"input_tokens": 100 * cycle + int(step_id[1:]), "output_tokens": 10}
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": f"C0{cycle} {step_id} working"}},
            {"type": "turn.completed", "usage": usage},
        ]
        exit_code, timed_out, stderr = 0, False, ""
        if callable(behavior):
            behavior(root)
        elif behavior == "auth":
            exit_code = 1
            stderr = "ERROR: 401 Unauthorized request-id: req_p28secret https://api.openai.com/v1/responses"
        elif behavior == "commit":
            write(root / "src/a.py", "A = 5\n")
            git(root, "add", "--all")
            git(root, "commit", "-qm", "agent-owned commit")
        elif behavior == "timeout":
            exit_code, timed_out = 124, True
        elif behavior == "unexpected":
            write(root / "src/a.py", "A = 6\n")
            write(root / "src/b.py", "B = 99\n")
        elif behavior == "exit1":
            exit_code, stderr = 1, "boom"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        final = f"C0{cycle} {step_id} report\n"
        (directory / "agent.final.md").write_text(final, encoding="utf-8")
        return AgentResult(exit_code=exit_code, timed_out=timed_out, final_message=final,
                           usage=usage, stderr_tail=stderr)


class CleanMismatchLuna(FakeLuna):
    """Luna double that reports a contract mismatch for selected attempts.

    ``mismatch_steps`` holds a step id (every attempt) or a ``"S01#1"`` key
    naming exactly one attempt of that step.
    """

    def __init__(
        self,
        behaviors: dict[tuple[int, str], Any],
        mismatch_steps: set[str],
        *,
        mismatch_text: str = "local contract is stale",
    ):
        super().__init__(behaviors)
        self.mismatch_steps = mismatch_steps
        self.mismatch_text = mismatch_text

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, **kwargs: Any) -> AgentResult:
        result = super().run_step(contract, worktree, artifacts_dir, **kwargs)
        call = self.calls[-1]
        keys = {call["step"], f"{call['step']}#{call['attempt']}"}
        if keys & self.mismatch_steps:
            message = f"META CONTRACT MISMATCH v1\n{self.mismatch_text}\n"
            Path(artifacts_dir, "agent.final.md").write_text(message, encoding="utf-8")
            return dataclasses.replace(result, final_message=message)
        return result


def writer(path: str, content: str) -> Callable[[Path], None]:
    return lambda root: write(root / path, content)


class FakeClaude:
    """Claude double with distinct initial and automatic-repair actions."""

    def __init__(self, actions: dict[int, Callable[[Path], None]] | None = None,
                 log: list[str] | None = None,
                 stage_actions: dict[tuple[int, str], Callable[[Path], None]] | None = None,
                 failures: dict[tuple[int, str], str] | None = None,
                 reports: dict[tuple[int, str], str] | None = None):
        self.actions = actions or {}
        self.stage_actions = stage_actions or {}
        self.failures = failures or {}
        self.reports = reports or {}
        self.calls: list[dict[str, Any]] = []
        self.log = log

    def run_revision(self, prompt: str, worktree: Path, *, artifacts_dir: Path,
                     profile: ModelProfile, environment: dict[str, str],
                     revision_dir: Path | None = None) -> ClaudeResult:
        target = Path(revision_dir) if revision_dir is not None else Path(artifacts_dir) / "revision"
        cycle = 2 if "C02" in target.parts else 1
        stage = "check-repair" if any(
            part in {"check-repair", "check-repair-expanded"} for part in target.parts
        ) else "initial-revision"
        self.calls.append({"cycle": cycle, "stage": stage, "prompt": prompt,
                           "environment": dict(environment), "revision_dir": target})
        if self.log is not None:
            self.log.append(f"claude:C0{cycle}")
        action = self.stage_actions.get((cycle, stage))
        if action is None and stage == "initial-revision":
            action = self.actions.get(cycle)
        if action is not None:
            action(Path(worktree))
        target.mkdir(parents=True, exist_ok=True)
        (target / "agent.prompt.txt").write_text(prompt, encoding="utf-8")
        failure = self.failures.get((cycle, stage))
        if failure == "timeout":
            (target / "agent.events.jsonl").write_text("", encoding="utf-8")
            (target / "agent.stderr.log").write_text("timeout\n", encoding="utf-8")
            return ClaudeResult(124, True, "", {}, "timeout")
        final = self.reports.get((cycle, stage)) or (
            f"Claude C0{cycle} revision report\n"
            if stage == "initial-revision"
            else f"Claude C0{cycle} check-repair report\n"
        )
        (target / "agent.events.jsonl").write_text(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": f"claude C0{cycle}"}}) + "\n",
            encoding="utf-8",
        )
        (target / "agent.final.md").write_text(final, encoding="utf-8")
        return ClaudeResult(0, False, final, {"input_tokens": 7 * cycle, "output_tokens": 3}, "")


class P28Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "MetaHarness P28")
        git(self.repo, "config", "user.email", "p28@example.invalid")
        write(self.repo / "README.md", "P28 readme\n")
        write(self.repo / "src/a.py", "A = 1\n")
        write(self.repo / "src/b.py", "B = 1\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.bare = self.root / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(self.bare)], check=True)
        git(self.repo, "remote", "add", "origin", str(self.bare))
        self.runs = self.root / "runs"
        self.counter = self.root / "checks-ran.txt"
        self.check = self.root / "gate.py"
        # Red when the candidate contains BUG; never writes inside the worktree.
        self.check.write_text(
            "import pathlib, sys\n"
            "open(sys.argv[1], 'a').write('ran\\n')\n"
            "sys.exit(1 if 'BUG' in pathlib.Path('src/a.py').read_text() else 0)\n",
            encoding="utf-8",
        )
        self.events: list[str] = []

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config_text(self, *, revision: bool = True, publish: bool = True,
                    require_approval: bool = False, web_url: str | None = None) -> str:
        self._web_url_section = (
            f'\n[repository]\nweb_url = "{web_url}"\n' if web_url else ""
        )
        return textwrap.dedent(
            f"""
            repo = {str(self.repo)!r}
            base_ref = "main"
            runs_root = {str(self.runs)!r}
            worktrees_root = {str(self.root / 'worktrees')!r}
            require_clean_base = true

            [planning]
            protocol = "v2"

            [revision]
            enabled = {'true' if revision else 'false'}
            max_cycles = 2

            [publish]
            enabled = {'true' if publish else 'false'}
            remote = "origin"
            mode = "run-branch"

            [codex_runtime]
            home = {str(self.root / 'codex-home')!r}

            [claude_runtime]
            home = {str(self.root / 'claude-home')!r}

            [approval]
            require_plan_approval = {'true' if require_approval else 'false'}
            poll_interval_seconds = 0.01

            [ui]
            enable_profile_recommendation = false
            default_planner_profile = "planner"
            default_implementer_profile = "luna"
            default_reviewer_profile = "reviewer"
            default_reviser_profile = "claude"
            default_repair_profile = "luna"

            [context]
            always_files = ["README.md"]

            [model_profiles.planner]
            display_name = "Planner"
            roles = ["planner"]
            driver = "openai-chat"
            model = "fake-planner"
            selection_mode = "request"
            base_url = "http://127.0.0.1:9"
            endpoint_path = "/v1/chat/completions"
            retries = 0

            [model_profiles.reviewer]
            display_name = "Reviewer"
            roles = ["reviewer"]
            driver = "openai-chat"
            model = "fake-reviewer"
            selection_mode = "request"
            base_url = "http://127.0.0.1:9"
            endpoint_path = "/v1/chat/completions"
            retries = 0

            [model_profiles.luna]
            display_name = "Luna"
            roles = ["implementer", "repair"]
            driver = "codex"
            model = "luna-offline"
            effort = "high"
            sandbox = "workspace-write"
            selection_mode = "cli"
            timeout_seconds = 30

            [model_profiles.claude]
            display_name = "Claude"
            roles = ["reviser", "repair"]
            driver = "claude-code"
            model = "claude-offline"
            effort = "medium"
            permission_mode = "acceptEdits"
            selection_mode = "cli"
            timeout_seconds = 30

            [[checks]]
            name = "gate"
            argv = [{sys.executable!r}, {str(self.check)!r}, {str(self.counter)!r}]
            timeout_seconds = 30
            """
        ) + self._web_url_section

    def load(self, **kwargs: Any) -> HarnessConfig:
        path = self.root / "p28.toml"
        path.write_text(self.config_text(**kwargs), encoding="utf-8")
        return load_config(path)

    def run_pipeline(
        self, *, luna: FakeLuna, reviews: list[str], plan: str = SINGLE_PLAN,
        repair_plan: str | None = None, claude: FakeClaude | None = None,
        run_id: str = "p28", revision: bool = True, web_url: str | None = None,
        planner: Any = None,
        run_options: RunOptions | None = None,
    ):
        config = self.load(revision=revision, web_url=web_url)
        planner = planner or QueueClient("planner", [plan] + ([repair_plan] if repair_plan else []), self.events)
        reviewer = QueueClient("reviewer", reviews, self.events)
        claude = claude or FakeClaude(log=self.events)
        claude.log = self.events
        orchestrator = Orchestrator(config, planner_client=planner, reviewer_client=reviewer,
                                    agent=luna, reviser=claude)
        real_push = orchestrator_module.push_run_branch

        def push(*args: Any, **kwargs: Any) -> Any:
            self.events.append("push")
            return real_push(*args, **kwargs)

        with mock.patch.object(orchestrator_module, "push_run_branch", side_effect=push) as pushed:
            result = orchestrator.run_text(SPEC, run_id=run_id, run_options=run_options)
        self.config_value = config
        return result, planner, reviewer, claude, pushed

    # -- assertions -------------------------------------------------------
    def worktree(self, run_id: str = "p28") -> Path:
        return self.root / "worktrees" / run_id

    def remote_refs(self) -> str:
        return subprocess.run(["git", "--git-dir", str(self.bare), "show-ref"],
                              capture_output=True, text=True).stdout.strip()

    def assert_no_commit_no_push(self, result: Any, pushed: Any, run_id: str = "p28") -> None:
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertIsNone(result.state.get("commit_sha"))
        self.assertEqual(git(self.worktree(run_id), "rev-parse", "HEAD"), self.base_sha)
        self.assertEqual(pushed.call_count, 0)
        self.assertEqual(self.remote_refs(), "")
        self.assertNotIn("push", self.events)

    def assert_published_once(self, result: Any, pushed: Any, run_id: str = "p28", expected_pushes: int = 1, expected_commits: int = 1) -> None:
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(pushed.call_count, expected_pushes)
        worktree = self.worktree(run_id)
        self.assertEqual(git(worktree, "rev-list", "--count", f"{self.base_sha}..HEAD"), str(expected_commits))
        self.assertEqual(git(worktree, "rev-parse", "HEAD"), result.state["commit_sha"])
        self.assertEqual(git(worktree, "rev-parse", "HEAD^{tree}"), result.state["approved_tree_sha"])
        branch = result.state["branch"]
        self.assertTrue(branch.startswith("harness/"))
        remote = subprocess.run(
            ["git", "--git-dir", str(self.bare), "rev-parse", f"refs/heads/{branch}"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(remote, result.state["commit_sha"])
        # Never main / base_ref, never a tag.
        refs = self.remote_refs().splitlines()
        self.assertEqual([line.split()[1] for line in refs], [f"refs/heads/{branch}"])
        # The candidate push is before the final reviewer PASS; final
        # publication is recorded only after PASS.
        last_pass = max(i for i, event in enumerate(self.events) if event == "reviewer:VERDICT: PASS")
        self.assertLess(self.events.index("push"), last_pass)


class FullPipelineTests(P28Harness):
    def test_clean_contract_mismatch_is_deferred_and_recovered_by_claude(self) -> None:
        luna = CleanMismatchLuna(
            {
                (1, "S02"): writer("src/b.py", "B = 2\n"),
                (1, "S03"): writer("src/c.py", "C = 3\n"),
            },
            {"S01"},
        )
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=luna,
            reviews=[PASS],
            plan=STAGED_PLAN,
            claude=FakeClaude({1: writer("src/a.py", "A = 2\n")}),
        )
        self.assert_published_once(result, pushed)
        # The clean mismatch is retried exactly once, then deferred.
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S01", "S02", "S03"])
        self.assertIsNone(luna.calls[0]["retry_addendum"])
        self.assertIn("MISMATCH RETRY ADDENDUM", luna.calls[1]["retry_addendum"])
        step = json.loads((result.run_dir / "steps/S01/step.json").read_text())
        self.assertEqual(step["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertEqual(step["changed_paths"], [])
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertIn("local contract is stale", step["mismatch"])
        self.assertIn("local contract is stale", step["initial_mismatch"])
        # Attempt 1 keeps its own diagnostics, never overwritten.
        archived = json.loads(
            (result.run_dir / "steps/S01/attempts/01/step.json").read_text()
        )
        self.assertEqual(archived["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertNotIn("mismatch_retry_count", archived)
        self.assertIn("<DEFERRED CONTRACT MISMATCHES>", claude.calls[0]["prompt"])
        self.assertIn("S01", claude.calls[0]["prompt"])
        self.assertIn("local contract is stale", claude.calls[0]["prompt"])
        self.assertIn("DEFERRED CONTRACT MISMATCHES", reviewer.prompts[0])
        self.assertIn("S01", reviewer.prompts[0])

    def test_clean_contract_mismatch_without_claude_never_creates_candidate(self) -> None:
        luna = CleanMismatchLuna({}, {"S01"})
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], revision=False,
        )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "UNRESOLVED_CONTRACT_MISMATCH")
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 2])
        self.assertFalse((result.run_dir / "candidate/C01/commit.json").exists())
        self.assertEqual(pushed.call_count, 0)
        self.assertEqual(reviewer.prompts, [])
        self.assertEqual(claude.calls, [])

    def test_v2_large_diff_uses_excerpt_but_still_revises_commits_pushes_and_reviews(self) -> None:
        config = dataclasses.replace(self.load(), max_diff_bytes=400)
        large = "A = " + ("1" * 300_000) + "\n# CLAUDE_DIFF_MUST_NOT_BE_PROMPTED\n"
        luna = FakeLuna({(1, "S01"): writer("src/a.py", large)})
        planner = QueueClient("planner", [SINGLE_PLAN], self.events)
        reviewer = QueueClient("reviewer", [PASS], self.events)
        claude = FakeClaude(log=self.events)
        orchestrator = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
            agent=luna, reviser=claude,
        )

        real_push = orchestrator_module.push_run_branch

        def push(*args: Any, **kwargs: Any) -> Any:
            self.events.append("push")
            return real_push(*args, **kwargs)

        with mock.patch.object(orchestrator_module, "push_run_branch", side_effect=push) as pushed:
            result = orchestrator.run_text(SPEC, run_id="large-v2")

        self.assert_published_once(result, pushed, run_id="large-v2")
        self.assertEqual(len(claude.calls), 1)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertNotIn("TRUNCATED: true", claude.calls[0]["prompt"])
        self.assertNotIn("1" * 10_000, claude.calls[0]["prompt"])
        self.assertNotIn("CLAUDE_DIFF_MUST_NOT_BE_PROMPTED", claude.calls[0]["prompt"])
        self.assertLess(len(claude.calls[0]["prompt"].encode("utf-8")), 100_000)
        self.assertIn("TRUNCATED: true", reviewer.prompts[0])
        self.assertNotIn("DIFF_TOO_LARGE", result.state["deterministic_gate"]["failures"])
        candidate = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        self.assertIn(candidate["commit_sha"], reviewer.prompts[0])
        if candidate["immutable_commit_url"]:
            self.assertIn(candidate["immutable_commit_url"], reviewer.prompts[0])
        else:
            self.assertIn('"immutable_commit_url": null', reviewer.prompts[0])
        self.assertIn("1" * 100_000, (result.run_dir / "diff.patch").read_text())
        self.assertIn("CLAUDE_DIFF_MUST_NOT_BE_PROMPTED", (result.run_dir / "diff.patch").read_text())

    def test_a_staged_c01_pass_commits_and_pushes_once(self) -> None:
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (1, "S02"): writer("src/b.py", "B = 2\n"),
            (1, "S03"): writer("src/c.py", "C = 3\n"),
        })
        result, planner, reviewer, claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], plan=STAGED_PLAN,
        )
        self.assert_published_once(result, pushed)
        self.assertEqual(len(planner.prompts), 1)
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])
        self.assertEqual(len({str(call["dir"]) for call in luna.calls}), 3)
        self.assertEqual([call["cycle"] for call in claude.calls], [1])
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertFalse((result.run_dir / "repair" / "C02").exists())
        self.assertEqual(self.events, ["planner:META PLAN v2", "claude:C01",
                                       "push", "reviewer:VERDICT: PASS"])
        selection = json.loads((result.run_dir / "execution_selection.json").read_text())
        self.assertEqual(selection["schema_version"], 4)
        self.assertEqual(
            list(result.state["execution"]),
            ["planner", "steps", "reviser", "repair_implementer", "reviewer"],
        )
        self.assertEqual(result.state["execution"]["repair_implementer"]["profile_id"], "luna")
        # The Claude process environment is the managed one.
        environment = claude.calls[0]["environment"]
        home = (self.root / "claude-home").resolve()
        self.assertEqual(environment["HOME"], str(home / "home"))
        self.assertEqual(environment["TMPDIR"], str(home / "tmp"))
        publish = json.loads((result.run_dir / "publish.json").read_text())
        self.assertEqual(publish["commit_sha"], result.state["commit_sha"])
        run = get_run(self.runs, "p28")
        self.assertEqual(len(run["cycle_artifacts"]), 1)
        page = render_run(run, None, config=self.config_value)
        self.assertIn("CYCLE 1 — INITIAL", page)
        self.assertNotIn("CYCLE 2", page)

    def test_b_c01_red_precheck_fixed_by_claude_then_pass(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude({1: writer("src/a.py", "A = 3\n")})
        result, _planner, _reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude,
        )
        pre = json.loads((result.run_dir / "revision" / "C01" / "pre_checks.json").read_text())
        self.assertFalse(pre["deterministic_passed"])
        self.assertTrue(result.state["deterministic_gate"]["passed"])
        self.assert_published_once(result, pushed)
        self.assertEqual(git(self.worktree(), "show", "HEAD:src/a.py"), "A = 3")

    def test_c_c01_red_gate_with_reviewer_pass_is_invalid(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        result, planner, reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS],
        )
        self.assertEqual(result.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assert_no_commit_no_push(result, pushed)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (1, 0))
        self.assertFalse((result.run_dir / "repair" / "C02").exists())
        self.assertIsNone(result.state.get("approved_tree_sha"))
        self.assertFalse(reviewer.prompts)

    def test_i_automatic_check_repair_c01_fixes_red_final_check(self) -> None:
        luna = FakeLuna({
            (1, "S01"): writer(
                "src/a.py",
                "A = BUG\nCLAUDE_DIFF_MUST_NOT_BE_PROMPTED\n",
            )
        })
        claude = FakeClaude(stage_actions={(1, "check-repair"): writer("src/a.py", "A = 3\n")})
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude, run_id="check-repair-c01",
        )
        self.assert_published_once(result, pushed, run_id="check-repair-c01")
        self.assertEqual([call["stage"] for call in claude.calls], ["initial-revision", "check-repair"])
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(claude.calls[1]["revision_dir"].relative_to(result.run_dir).as_posix(), "revision/check-repair/C01")
        self.assertTrue((result.run_dir / "revision/check-repair/C01/agent.prompt.txt").exists())
        repair_prompt = claude.calls[1]["prompt"]
        self.assertNotIn("CLAUDE_DIFF_MUST_NOT_BE_PROMPTED", claude.calls[0]["prompt"])
        self.assertNotIn("CLAUDE_DIFF_MUST_NOT_BE_PROMPTED", repair_prompt)
        self.assertIn("CHECK_FAILED:gate", repair_prompt)
        self.assertIn("src/a.py", repair_prompt)
        self.assertIn(
            "CLAUDE_DIFF_MUST_NOT_BE_PROMPTED",
            (result.run_dir / "revision/diff.patch").read_text(),
        )
        self.assertIn("<PREVIOUS REPAIR REPORT>\nNONE", repair_prompt)
        self.assertNotIn("Claude C01 revision report", repair_prompt)

    def test_bounded_scope_expands_for_a_tracked_failing_test(self) -> None:
        write(self.repo / "tests/test_service.py", "def test_fake_uow():  # stale\n    pass\n")
        git(self.repo, "add", "tests/test_service.py")
        git(self.repo, "commit", "-qm", "add stale fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.check.write_text(
            "import pathlib, sys\n"
            "print('FAILED tests/test_service.py::test_fake_uow')\n"
            "sys.exit(1 if 'stale' in pathlib.Path('tests/test_service.py').read_text() else 0)\n",
            encoding="utf-8",
        )
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        claude = FakeClaude(
            stage_actions={(1, "check-repair"): writer(
                "tests/test_service.py", "def test_fake_uow():\n    pass\n"
            )}
        )
        result, _planner, reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude,
            run_options=RunOptions.from_config(
                self.config_value if hasattr(self, "config_value") else self.load(),
                repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
            ),
        )
        self.assert_published_once(result, pushed)
        self.assertEqual(len(claude.calls), 2)
        scope = json.loads(
            (result.run_dir / "revision/check-repair/C01/scope.json").read_text()
        )
        self.assertEqual(scope["base_mutable_scope"], ["src/a.py"])
        self.assertEqual(scope["added_paths"], ["tests/test_service.py"])
        self.assertEqual(scope["effective_mutable_scope"], ["src/a.py", "tests/test_service.py"])
        self.assertIsNone(result.state.get("failure"))
        self.assertEqual(len(reviewer.prompts), 1)

    def test_one_expanded_repair_is_used_when_the_test_appears_on_retry(self) -> None:
        write(self.repo / "tests/test_service.py", "def test_fake_uow():  # stale\n    pass\n")
        git(self.repo, "add", "tests/test_service.py")
        git(self.repo, "commit", "-qm", "add stale fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.check.write_text(
            "import pathlib, sys\n"
            "source = pathlib.Path('src/a.py').read_text()\n"
            "fixture = pathlib.Path('tests/test_service.py').read_text()\n"
            "if 'BUG' in source:\n"
            "    print('FAILED src/other_production.py')\n"
            "    sys.exit(1)\n"
            "print('FAILED tests/test_service.py::test_fake_uow')\n"
            "sys.exit(1 if 'stale' in fixture else 0)\n",
            encoding="utf-8",
        )
        repair_calls = 0
        claude: FakeClaude

        def repair_action(root: Path) -> None:
            nonlocal repair_calls
            repair_calls += 1
            if repair_calls == 1:
                write(root / "src/a.py", "A = 3\n")
            else:
                write(root / "tests/test_service.py", "def test_fake_uow():\n    pass\n")

        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={(1, "check-repair"): repair_action})
        result, _planner, _reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude,
            run_options=RunOptions.from_config(
                self.load(), repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
            ),
        )
        self.assert_published_once(result, pushed)
        self.assertEqual(repair_calls, 2)
        self.assertTrue((result.run_dir / "revision/check-repair-expanded/C01/report.json").exists())
        self.assertTrue((result.run_dir / "checks/C01/attempts/02/evidence.json").exists())
        expanded_scope = json.loads(
            (result.run_dir / "revision/check-repair-expanded/C01/scope.json").read_text()
        )
        self.assertEqual(expanded_scope["added_paths"], ["tests/test_service.py"])

    def test_j_automatic_check_repair_c01_is_bounded_and_stays_red(self) -> None:
        """The repair budget is bounded at two passes, and stops there.

        A soft ``CHECK_FAILED`` earns the second bounded pass even when no
        mutable-scope expansion is needed, so the budget is the initial
        revision plus two corrective passes -- and never a third.
        """

        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude()
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude, run_id="check-repair-red",
        )
        self.assertEqual(result.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assert_no_commit_no_push(result, pushed, run_id="check-repair-red")
        self.assertEqual([call["stage"] for call in claude.calls],
                         ["initial-revision", "check-repair", "check-repair"])
        self.assertIn("<PREVIOUS REPAIR REPORT>", claude.calls[2]["prompt"])
        self.assertIn("Claude C01 check-repair report", claude.calls[2]["prompt"])
        # The second pass ran inside the exact scope the first one held.
        scope = json.loads(
            (result.run_dir / "revision/check-repair-expanded/C01/scope.json").read_text()
        )
        self.assertEqual(scope["source"], "bounded same-scope retry")
        self.assertEqual(scope["added_paths"], [])
        self.assertEqual(scope["effective_mutable_scope"], ["src/a.py"])
        self.assertEqual(reviewer.prompts, [])
        self.assertIn("CHECK_FAILED:gate", result.state["failure"]["detail"])

    def test_k_automatic_check_repair_c02_has_full_parity(self) -> None:
        claude = FakeClaude(stage_actions={(2, "check-repair"): writer("src/a.py", "A = 5\n")})
        _luna, result, _planner, reviewer, claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = BUG\n"), [REVISE_IMPLEMENTATION, PASS], claude=claude,
            run_id="check-repair-c02",
        )
        self.assert_published_once(result, pushed, run_id="check-repair-c02", expected_pushes=2, expected_commits=2)
        self.assertEqual([(call["cycle"], call["stage"]) for call in claude.calls], [
            (1, "initial-revision"), (2, "initial-revision"), (2, "check-repair"),
        ])
        self.assertEqual(len(reviewer.prompts), 2)
        self.assertTrue((result.run_dir / "revision/check-repair/C02/report.json").exists())

    def test_l_check_repair_timeout_resumes_without_replaying_luna_or_initial_revision(self) -> None:
        first_claude = FakeClaude(failures={(1, "check-repair"): "timeout"})
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        first, _planner, _reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=first_claude, run_id="check-repair-resume",
        )
        self.assertEqual(first.state["failure"]["reason"], "CLAUDE_TIMEOUT")
        self.assertEqual(read_checkpoint(first.run_dir).phase, ResumePhase.CHECK_REPAIR_C01)
        self.assertEqual(len(first_claude.calls), 2)
        fresh_claude = FakeClaude(stage_actions={(1, "check-repair"): writer("src/a.py", "A = 3\n")})
        fresh = Orchestrator(
            self.config_value,
            planner_client=QueueClient("planner", [], self.events),
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=FakeLuna({}), reviser=fresh_claude,
        ).resume("check-repair-resume")
        self.assertEqual(fresh.status, RunStatus.PUBLISHED, fresh.state.get("failure"))
        self.assertEqual(git(self.worktree("check-repair-resume"), "show", "HEAD:src/a.py"), "A = 3")
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["check-repair"])
        self.assertEqual(fresh_claude.calls[0]["cycle"], 1)

    def test_m_check_timeout_never_starts_automatic_repair(self) -> None:
        self.check.write_text(
            "import pathlib, sys, time\n"
            "if 'TRIGGER' in pathlib.Path('src/a.py').read_text(): time.sleep(6)\n"
            "sys.exit(0)\n", encoding="utf-8",
        )
        config_path = self.root / "timeout.toml"
        config_path.write_text(self.config_text().replace("timeout_seconds = 30", "timeout_seconds = 1"), encoding="utf-8")
        config = load_config(config_path)
        planner = QueueClient("planner", [SINGLE_PLAN], self.events)
        reviewer = QueueClient("reviewer", [PASS], self.events)
        claude = FakeClaude({1: writer("src/a.py", "A = TRIGGER\n")})
        result = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
            agent=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}), reviser=claude,
        ).run_text(SPEC, run_id="check-timeout")
        self.assertEqual(result.state["failure"]["reason"], "CHECK_TIMEOUT")
        self.assertEqual(len(claude.calls), 1)
        self.assertEqual(reviewer.prompts, [])
        self.assertFalse((result.run_dir / "revision/check-repair/C01").exists())

    def test_n_check_mutation_never_starts_automatic_repair(self) -> None:
        self.check.write_text(
            "import pathlib\npathlib.Path('src/a.py').write_text('MUTATED\\n')\n", encoding="utf-8",
        )
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviews=[PASS], run_id="check-mutated",
        )
        self.assertEqual(result.state["failure"]["reason"], "CHECK_MUTATED")
        self.assert_no_commit_no_push(result, pushed, run_id="check-mutated")
        self.assertEqual(claude.calls, [])
        self.assertEqual(reviewer.prompts, [])

    def test_o_check_repair_scope_violation_is_terminal(self) -> None:
        claude = FakeClaude(stage_actions={(1, "check-repair"): writer("README.md", "outside\n")})
        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_id="check-repair-scope",
        )
        self.assertEqual(result.state["failure"]["reason"], "REVISION_SCOPE_VIOLATION")
        self.assert_no_commit_no_push(result, pushed, run_id="check-repair-scope")
        self.assertEqual(len(claude.calls), 2)
        self.assertEqual(reviewer.prompts, [])

    def test_p_scope_violation_rolls_back_and_replans_bounded_repair(self) -> None:
        config = self.load()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
        )
        claude = FakeClaude(
            stage_actions={(1, "check-repair"): writer("README.md", "outside\n")}
        )
        first, planner, _reviewer, _claude, pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], repair_plan=REPAIR_PLAN, claude=claude,
            run_id="check-repair-scope-recovery", run_options=options,
        )
        self.assertEqual(first.state["failure"]["reason"], "REVISION_SCOPE_VIOLATION")
        self.assertEqual(pushed.call_count, 0)
        self.assertTrue((self.worktree("check-repair-scope-recovery") / "README.md").exists())
        historical_report = first.run_dir / "revision/check-repair/C01/report.json"
        report = json.loads(historical_report.read_text())
        report.pop("outside_scope_paths", None)
        historical_report.write_text(json.dumps(report) + "\n")

        repair_luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 3\n")})
        real_push = orchestrator_module.push_run_branch

        def push(*args: Any, **kwargs: Any) -> Any:
            self.events.append("push")
            return real_push(*args, **kwargs)

        with mock.patch.object(orchestrator_module, "push_run_branch", side_effect=push) as resumed_pushed:
            resumed = Orchestrator(
                self.config_value,
                planner_client=planner,
                reviewer_client=QueueClient("reviewer", [PASS], self.events),
                agent=repair_luna,
                reviser=FakeClaude(log=self.events),
            ).resume("check-repair-scope-recovery")

        self.assert_published_once(
            resumed, resumed_pushed, run_id="check-repair-scope-recovery",
        )
        self.assertEqual(
            (self.worktree("check-repair-scope-recovery") / "README.md").read_text(),
            "P28 readme\n",
        )
        recovery = json.loads(
            (resumed.run_dir / "revision/check-repair/C01/scope_violation_recovery.json").read_text()
        )
        self.assertEqual(recovery["restored_paths"], ["README.md"])
        self.assertEqual(recovery["outside_scope_paths"], ["README.md"])
        self.assertEqual([(call["cycle"], call["step"]) for call in repair_luna.calls], [(1, "S01")])
        self.assertEqual(len(planner.prompts), 2)
        self.assertIn("strong bounded scope-repair planner", planner.prompts[1])
        scope_delta = json.loads(
            (resumed.run_dir / "scope-repair/C01/scope_delta.json").read_text()
        )
        self.assertEqual(scope_delta["observed_outside_scope_paths"], ["README.md"])
        self.assertEqual(scope_delta["added_paths"], [])

    def test_q_scope_violation_has_c02_parity_without_replaying_c02_luna(self) -> None:
        config = self.load()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
        )
        planner = QueueClient(
            "planner", [SINGLE_PLAN, REPAIR_PLAN, REPAIR_PLAN], self.events,
        )
        first, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (2, "S01"): writer("src/a.py", "A = BUG\n"),
            }),
            reviews=[REVISE_IMPLEMENTATION, PASS],
            claude=FakeClaude(
                stage_actions={(2, "check-repair"): writer("README.md", "outside\n")}
            ),
            run_id="check-repair-scope-c02-recovery",
            planner=planner,
            run_options=options,
        )
        self.assertEqual(first.state["failure"]["reason"], "REVISION_SCOPE_VIOLATION")
        self.assertEqual(read_checkpoint(first.run_dir).cycle, 2)

        repair_luna = FakeLuna({(2, "S01"): writer("src/a.py", "A = 5\n")})
        real_push = orchestrator_module.push_run_branch

        def push(*args: Any, **kwargs: Any) -> Any:
            self.events.append("push")
            return real_push(*args, **kwargs)

        with mock.patch.object(orchestrator_module, "push_run_branch", side_effect=push) as resumed_pushed:
            resumed = Orchestrator(
                self.config_value,
                planner_client=planner,
                reviewer_client=QueueClient("reviewer", [PASS], self.events),
                agent=repair_luna,
                reviser=FakeClaude(log=self.events),
            ).resume("check-repair-scope-c02-recovery")

        self.assert_published_once(
            resumed, resumed_pushed,
            run_id="check-repair-scope-c02-recovery", expected_commits=2,
        )
        self.assertEqual([(call["cycle"], call["step"]) for call in repair_luna.calls], [(2, "S01")])
        self.assertEqual(len(planner.prompts), 3)
        self.assertTrue((resumed.run_dir / "scope-repair/C02/scope_delta.json").exists())

    def _c01_then_repair(self, c02_behavior: Any, reviews: list[str], *,
                         claude: FakeClaude | None = None, run_id: str = "p28"):
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): c02_behavior})
        return luna, *self.run_pipeline(
            luna=luna, reviews=reviews, repair_plan=REPAIR_PLAN, claude=claude, run_id=run_id,
        )

    def test_d_revise_implementation_then_c02_pass_publishes(self) -> None:
        luna, result, planner, reviewer, claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = 4\n"), [REVISE_IMPLEMENTATION, PASS],
        )
        self.assert_published_once(result, pushed, expected_pushes=2, expected_commits=2)
        self.assertEqual(self.events, [
            "planner:META PLAN v2", "claude:C01", "push", "reviewer:VERDICT: REVISE",
            "planner:META PLAN v2", "claude:C02", "push", "reviewer:VERDICT: PASS",
        ])
        self.assertEqual([(call["cycle"], call["step"]) for call in luna.calls], [(1, "S01"), (2, "S01")])
        self.assertEqual([call["cycle"] for call in claude.calls], [1, 2])
        self.assertNotIn("C01 S01 report", claude.calls[0]["prompt"])
        self.assertNotIn("C02 S01 report", claude.calls[1]["prompt"])
        self.assertEqual(result.state["cycle"], 2)
        self.assertEqual(result.state["review_iterations"], 2)
        self.assertEqual(git(self.worktree(), "show", "HEAD:src/a.py"), "A = 4")
        # C01 v2 passes only structural worker history through LUNA REPORTS;
        # the legacy AGENT_REPORT section is not part of the default template.
        first = reviewer.prompts[0]
        self.assertNotIn("<NON-AUTHORITATIVE IMPLEMENTER REPORT>", first)
        self.assertNotIn("C01 S01 report", first)
        self.assertEqual(first.count("Claude C01 revision report"), 1)
        repair_prompt = planner.prompts[1]
        self.assertNotIn("REVIEWER #1 RAW", repair_prompt)
        self.assertNotIn("META REVIEW v1", repair_prompt)
        self.assertIn('"required_fixes": "Correct src/a.py."', repair_prompt)
        # Reviewer #2 evidence: both plans, all worker reports, both revisions.
        second = reviewer.prompts[1]
        original = (result.run_dir / "planner.raw.md").read_text()
        repair = (result.run_dir / "repair" / "C02" / "planner.raw.md").read_text()
        self.assertIn('"original_approved_plan"', second)
        self.assertIn('"repair_plan_c02"', second)
        self.assertIn('"scope_delta"', second)
        self.assertNotIn(original, second)
        self.assertNotIn(repair, second)
        for marker in ("C01 LUNA REPORTS", "C02 LUNA REPAIR REPORTS",
                       "C02 S01 report", "C01 CLAUDE REVISION", "C02 CLAUDE REVISION",
                       "Claude C01 revision report", "Claude C02 revision report",
                       '"deterministic_passed": true'):
            if marker == "C02 S01 report":
                self.assertNotIn(marker, second)
            else:
                self.assertIn(marker, second)
        usage = result.state["usage"]
        self.assertEqual(usage["luna_c01"]["input_tokens"], 101)
        self.assertEqual(usage["luna_c02"]["input_tokens"], 201)
        self.assertEqual(usage["grand_total"]["input_tokens"], 10 + 101 + 7 + 10 + 10 + 201 + 14 + 10)

    def _staged_repair_run(self, *, run_id: str, web_url: str | None, planner: Any = None):
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = C01_DIFF_SENTINEL\n"),
            (1, "S02"): writer("src/b.py", "B = C01_DIFF_SENTINEL\n"),
            (1, "S03"): writer("src/c.py", "C = C01_DIFF_SENTINEL\n"),
            (2, "S01"): writer("src/a.py", "A = 4\n"),
        })
        return self.run_pipeline(
            luna=luna, reviews=[REVISE_IMPLEMENTATION, PASS], plan=STAGED_PLAN,
            repair_plan=REPAIR_PLAN, run_id=run_id, web_url=web_url, planner=planner,
        )

    def test_d2_repair_request_is_compact_around_the_immutable_candidate(self) -> None:
        result, planner, _r, claude, _pushed = self._staged_repair_run(
            run_id="p28-compact", web_url="https://github.com/example/p28",
        )
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        c02_revision_call = next(
            call for call in claude.calls
            if call["cycle"] == 2 and call["stage"] == "initial-revision"
        )
        self.assertNotIn("C01_DIFF_SENTINEL", c02_revision_call["prompt"])

        repair_dir = result.run_dir / "repair" / "C02"
        request = (repair_dir / "planner.request.txt").read_text(encoding="utf-8")
        fallback = (repair_dir / "planner.request.fallback.txt").read_text(encoding="utf-8")
        evidence = (repair_dir / "planner.evidence.md").read_text(encoding="utf-8")
        meta = json.loads((repair_dir / "planner.request.meta.json").read_text())

        self.assertEqual(request, planner.prompts[1])
        candidate_sha = json.loads(
            (result.run_dir / "candidate" / "C01" / "commit.json").read_text()
        )["commit_sha"]

        # Present: SPEC, compact plan summary, compact step index, the exact
        # candidate identity and the structured reviewer answer.
        for present in (
            SPEC.strip(),
            "TITLE: P28 feature",
            '"id": "S01"',
            '"mutation_scope"',
            candidate_sha,
            f'"candidate_url": "https://github.com/example/p28/tree/{candidate_sha}"',
            f'"compare_url": "https://github.com/example/p28/compare/'
            f'{self.base_sha}...{candidate_sha}"',
            "src/a.py",
            '"verdict": "REVISE"',
            '"required_fixes": "Correct src/a.py."',
            '"deterministic_passed": true',
        ):
            self.assertIn(present, request, present)

        # Absent: the full C01 diff, the raw step contracts and token counters.
        for absent in (
            "C01_DIFF_SENTINEL",
            "Perform operation 1 exactly.",
            "Perform operation 3 exactly.",
            "META IMPLEMENTATION STEP v1",
            "C01 S01 report",
            "LUNA REPORTS",
            '"argv"',
            '"stdout_tail"',
            '"duration_seconds"',
        ):
            self.assertNotIn(absent, request, absent)

        # The full diff stays durable in the normal Git evidence artifacts.
        self.assertIn("C01_DIFF_SENTINEL", (result.run_dir / "diff.patch").read_text())

        self.assertIn("repair-evidence.md", fallback)
        self.assertNotIn(evidence, fallback)
        self.assertIn(evidence, request)
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["inline_bytes"], len(request.encode("utf-8")))
        self.assertEqual(meta["evidence_bytes"], len(evidence.encode("utf-8")))
        self.assertEqual(meta["file_fallback_attempt"], 3)
        # Remote exploration is available, so no candidate.diff is ever offered.
        self.assertEqual(meta["candidate_diff_attachment_bytes"], 0)

    def test_d3_without_a_remote_the_candidate_diff_is_attachment_only(self) -> None:
        result, _p, _r, _c, _pushed = self._staged_repair_run(
            run_id="p28-no-remote", web_url=None,
        )
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        repair_dir = result.run_dir / "repair" / "C02"
        meta = json.loads((repair_dir / "planner.request.meta.json").read_text())
        request = (repair_dir / "planner.request.txt").read_text(encoding="utf-8")

        self.assertGreater(meta["candidate_diff_attachment_bytes"], 0)
        # Even without a remote the full diff is never inlined in the request.
        self.assertNotIn("C01_DIFF_SENTINEL\n+", request)

    def test_d4_the_repair_planner_never_continues_the_planner_conversation(self) -> None:
        class ConversationPlanner(QueueClient):
            def __init__(self, responses, log):
                super().__init__("planner", responses, log)
                self.conversation_calls = 0

            def complete(self, prompt: str) -> TextLLMResult:
                result = super().complete(prompt)
                return TextLLMResult(
                    result.text, result.model, result.usage, {},
                    conversation=LLMConversationHandle("bridge", "conv-planner-1"),
                )

            def complete_in_conversation(self, handle, prompt: str) -> TextLLMResult:
                self.conversation_calls += 1
                return super().complete(prompt)

        planner = ConversationPlanner([STAGED_PLAN, REPAIR_PLAN], self.events)
        result, planner, _r, _c, _pushed = self._staged_repair_run(
            run_id="p28-fresh-c02", web_url=None, planner=planner,
        )
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        # The historical artifact still exists; the repair planner ignores it.
        self.assertTrue((result.run_dir / "planner.conversation.json").exists())
        self.assertEqual(planner.conversation_calls, 0)
        self.assertEqual(len(planner.prompts), 2)

    def test_e_c02_red_gate_with_reviewer_pass_is_invalid(self) -> None:
        _luna, result, planner, reviewer, _claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = BUG\n"), [REVISE_IMPLEMENTATION, PASS],
        )
        self.assertEqual(result.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(pushed.call_count, 1)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (2, 1))
        self.assertFalse((result.run_dir / "repair" / "C03").exists())
        self.assertFalse(result.state["deterministic_gate"]["passed"])

    def test_f_c02_revise_exhausts_the_loop(self) -> None:
        _luna, result, planner, reviewer, _claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = 4\n"), [REVISE_IMPLEMENTATION, REVISE_IMPLEMENTATION],
        )
        self.assertEqual(result.state["failure"]["reason"], "REVIEW_LOOP_EXHAUSTED")
        self.assertEqual(pushed.call_count, 2)
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD^^"), self.base_sha)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (2, 2))
        self.assertFalse((result.run_dir / "repair" / "C03").exists())

    def test_g_codex_failures_have_c01_c02_parity(self) -> None:
        # "nochange" is deliberately absent: since P4 a clean no-change is no
        # longer a terminal Codex failure.  It is handled by bounded retry and
        # may end as DEFERRED_CONTRACT_MISMATCH, so it has no C01/C02 terminal
        # parity to assert here.  The dedicated P49 tests own that coverage.
        expected = {
            "auth": "CODEX_AUTH_FAILURE",
            "commit": "AGENT_COMMITTED",
            "timeout": "AGENT_TIMEOUT",
            "unexpected": "STEP_WRITE_SET_VIOLATION",
            "exit1": "AGENT_FAILED",
        }
        for behavior, reason in expected.items():
            for cycle in (1, 2):
                run_id = f"g-{behavior}-c{cycle}"
                with self.subTest(behavior=behavior, cycle=cycle):
                    self.events.clear()
                    if cycle == 1:
                        luna = FakeLuna({(1, "S01"): behavior})
                        result, _p, reviewer, _c, pushed = self.run_pipeline(
                            luna=luna, reviews=[], run_id=run_id,
                        )
                        step_dir = result.run_dir / "steps" / "S01"
                        self.assertEqual(reviewer.prompts, [])
                    else:
                        _l, result, _p, reviewer, _c, pushed = self._c01_then_repair(
                            behavior, [REVISE_IMPLEMENTATION], run_id=run_id,
                        )
                        step_dir = result.run_dir / "repair" / "C02" / "steps" / "S01"
                        self.assertEqual(len(reviewer.prompts), 1)
                    failure = result.state["failure"]
                    self.assertEqual(failure["reason"], reason)
                    self.assertTrue(failure["detail"].startswith("step=S01"), failure)
                    serialized = json.dumps(result.state)
                    self.assertNotIn("req_p28secret", serialized)
                    self.assertNotIn("api.openai.com", serialized)
                    if reason == "CODEX_AUTH_FAILURE":
                        self.assertEqual(failure["detail"], "step=S01 Codex authentication failed")
                    self.assertEqual(json.loads((step_dir / "step.json").read_text())["reason"], reason)
                    self.assertEqual(result.state["steps"][0]["status"], "failed")
                    self.assertEqual(pushed.call_count, 0 if cycle == 1 else 1)
                    self.assertIsNone(result.state.get("commit_sha"))
                    self.assertFalse((result.run_dir / "repair" / "C03").exists())

    def test_h_ui_api_cycle_artifacts_are_exact_and_unmixed(self) -> None:
        _luna, result, _p, _r, _c, _pushed = self._c01_then_repair(
            writer("src/a.py", "A = 4\n"), [REVISE_IMPLEMENTATION, PASS],
        )
        run_dir = result.run_dir
        run = get_run(self.runs, "p28")
        c1, c2 = run["cycle_artifacts"]
        self.assertEqual((c1["number"], c1["kind"], c2["number"], c2["kind"]), (1, "initial", 2, "repair"))
        s1, s2 = c1["steps"][0], c2["steps"][0]
        self.assertEqual(s1["contract"], (run_dir / "steps/S01/contract.md").read_text())
        self.assertEqual(s2["contract"], (run_dir / "repair/C02/steps/S01/contract.md").read_text())
        self.assertTrue(s1["contract_matches_bundle"] and s2["contract_matches_bundle"])
        self.assertNotEqual(s1["contract"], s2["contract"])
        self.assertEqual((s1["final"], s2["final"]), ("C01 S01 report\n", "C02 S01 report\n"))
        self.assertEqual((s1["status"], s2["status"]), ("completed", "completed"))
        self.assertIn("message: C01 S01 working", s1["events"])
        self.assertIn("message: C02 S01 working", s2["events"])
        self.assertFalse(any("C02" in event for event in s1["events"]))
        self.assertEqual(c1["review"]["raw"], (run_dir / "review/C01/reviewer.raw.md").read_text())
        self.assertEqual(c2["review"]["raw"], (run_dir / "review/C02/reviewer.raw.md").read_text())
        self.assertNotEqual(c1["review"]["raw"], c2["review"]["raw"])
        self.assertEqual(c1["review"]["review"]["verdict"], "REVISE")
        self.assertEqual(c2["review"]["review"]["verdict"], "PASS")
        self.assertEqual(c1["revision"]["final"], "Claude C01 revision report\n")
        self.assertEqual(c2["revision"]["final"], "Claude C02 revision report\n")
        self.assertEqual(c1["checks"]["checks"], json.loads((run_dir / "checks/C01/checks.json").read_text()))
        self.assertEqual(c2["checks"]["checks"], json.loads((run_dir / "checks/C02/checks.json").read_text()))
        # Top-level aliases describe the final cycle (C02).
        self.assertEqual(run["checks"], json.loads((run_dir / "checks/C02/checks.json").read_text()))
        self.assertEqual(run["review"], json.loads((run_dir / "review/C02/review.json").read_text()))
        self.assertEqual(run["reviewer_raw"], (run_dir / "review/C02/reviewer.raw.md").read_text())
        self.assertEqual(run["revision"], json.loads((run_dir / "revision/C02/report.json").read_text()))
        self.assertEqual(run["candidate"]["changed_files"],
                         (run_dir / "checks/C02/changed-files.txt").read_text().splitlines())
        self.assertEqual(run["candidate"]["diff_tail"], (run_dir / "checks/C02/diff.patch").read_text())
        # Historical step_artifacts stays C01 even though state.steps is C02.
        self.assertEqual(result.state["cycle"], 2)
        self.assertEqual(run["step_artifacts"][0]["contract"], s1["contract"])
        # Live events are cycle-aware and bounded to cycles 1 and 2.
        self.assertEqual(cycle_step_progress_tail(run_dir, 2, "S01"), s2["events"])
        with self.assertRaises(WebAPIError):
            cycle_step_progress_tail(run_dir, 3, "S01")
        # Usage is read from persisted step.json of both cycles.
        self.assertEqual(run["usage"]["luna_c01"]["input_tokens"], 101)
        self.assertEqual(run["usage"]["luna_c02"]["input_tokens"], 201)
        self.assertEqual([(row["id"], row["cycle"]) for row in run["usage"]["implementer"]["steps"]],
                         [("S01", 1), ("S01", 2)])
        page = render_run(run, None, config=self.config_value)
        self.assertLess(page.index("CYCLE 1 — INITIAL"), page.index("CYCLE 2 — REPAIR"))
        self.assertIn("Reviewer #1", page)
        self.assertIn("Reviewer #2", page)
        self.assertIn("Final cycle C02 (repair) · checks PASS · reviewer #2 PASS / NONE", page)

    def test_revision_disabled_never_launches_claude_or_c02(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        result, planner, reviewer, claude, _pushed = self.run_pipeline(
            luna=luna, reviews=[REVISE_IMPLEMENTATION], revision=False,
        )
        self.assertEqual(result.state["failure"]["reason"], "REVIEW_REVISE")
        self.assertEqual(claude.calls, [])
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (1, 1))
        self.assertFalse((result.run_dir / "repair").exists())
        selection = json.loads((result.run_dir / "execution_selection.json").read_text())
        self.assertEqual(selection["schema_version"], 3)
        self.assertNotIn("reviser", selection)


def required_checks_section(plan: str, *check_ids: str) -> str:
    section = "REQUIRED_CHECKS\n" + "".join(f"- {check_id}\n" for check_id in check_ids)
    return plan.replace("CONSTRAINTS\nNONE\n", f"CONSTRAINTS\nNONE\n\n{section}", 1)


class StructuredScopeRequestPipelineTests(P28Harness):
    REQUEST = """Claude completed the bounded attempt.

META SCOPE REQUEST v1

REASON
The failing correction needs the existing source file.

PATHS
- src/a.py

EVIDENCE
- CHECK_FAILED:gate | src/a.py still contains BUG

END META SCOPE REQUEST
"""

    def test_valid_request_rolls_back_and_routes_advisory_evidence_to_bridge(self) -> None:
        config = self.load()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
        )
        claude = FakeClaude(
            stage_actions={(1, "check-repair"): writer("src/a.py", "A = REQUESTED\n")},
            reports={(1, "check-repair"): self.REQUEST},
        )
        first, planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({
                (1, "S01", 1): writer("src/a.py", "A = BUG\n"),
                (1, "S01", 2): writer("src/a.py", "A = 3\n"),
            }),
            reviews=[PASS], repair_plan=REPAIR_PLAN, claude=claude,
            run_id="structured-scope-request", run_options=options,
        )

        self.assertEqual(first.status, RunStatus.PUBLISHED, first.state.get("failure"))
        self.assertEqual(git(self.worktree("structured-scope-request"), "show", "HEAD:src/a.py"), "A = 3")
        self.assertEqual(
            first.state["check_repair"]["scope_request_diagnostic"],
            "Claude requested scope expansion:\n"
            "  paths: 1\n"
            "  authoritative: NO\n"
            "  routed to bridge audit: YES",
        )
        self.assertEqual(len(planner.prompts), 2)
        bridge_prompt = planner.prompts[1]
        self.assertIn("<CLAUDE SCOPE REQUEST>", bridge_prompt)
        self.assertIn("<OBSERVED OUTSIDE SCOPE PATHS>", bridge_prompt)
        self.assertIn("src/a.py", bridge_prompt)
        recovery = json.loads(
            (first.run_dir / "revision/check-repair/C01/scope_violation_recovery.json").read_text()
        )
        self.assertEqual(recovery["restored_paths"], ["src/a.py"])
        self.assertEqual(recovery["outside_scope_paths"], [])
        scope_delta = json.loads(
            (first.run_dir / "scope-repair/C01/scope_delta.json").read_text()
        )
        self.assertEqual(scope_delta["added_paths"], [])

    def test_scope_request_parser_rejects_duplicates_globs_and_more_than_32_paths(self) -> None:
        base = self.REQUEST.replace("- src/a.py", "- src/a.py\n- src/a.py")
        self.assertIsNone(parse_scope_request(base))
        self.assertIsNone(parse_scope_request(self.REQUEST.replace("src/a.py", "src/*.py")))
        paths = "\n".join(f"- src/{index}.py" for index in range(50))
        self.assertIsNone(parse_scope_request(self.REQUEST.replace("- src/a.py", paths)))


class CheckAuthorityPipelineTests(P28Harness):
    """``checks/C01`` is canonical, and the frozen catalogue is the only argv."""

    def evidence(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_p_check_repair_c01_keeps_the_red_attempt_and_publishes_the_green(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={(1, "check-repair"): writer("src/a.py", "A = 3\n")})
        result, _planner, _reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[PASS], claude=claude, run_id="c01-layout",
        )
        self.assert_published_once(result, pushed, run_id="c01-layout")
        run_dir = result.run_dir
        checks_dir = run_dir / "checks" / "C01"

        red = self.evidence(checks_dir / "attempts" / "01" / "evidence.json")
        self.assertFalse(red["deterministic_passed"])
        self.assertIn("CHECK_FAILED:gate", red["failures"])
        for name in ("checks.json", "changed-files.txt", "diff.patch"):
            self.assertTrue((checks_dir / "attempts" / "01" / name).exists(), name)

        green = self.evidence(checks_dir / "evidence.json")
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual(green["failures"], [])
        candidate = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(candidate["tree_sha"], green["staged_tree_sha"])
        # The historical root aliases exist and are the corrected evidence.
        self.assertEqual(self.evidence(run_dir / "evidence.json"), green)
        self.assertEqual((run_dir / "diff.patch").read_text(),
                         (checks_dir / "diff.patch").read_text())

    def test_q_the_green_c01_evidence_survives_the_c02_snapshot(self) -> None:
        claude = FakeClaude(stage_actions={(1, "check-repair"): writer("src/a.py", "A = 3\n")})
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n"),
                         (2, "S01"): writer("src/a.py", "A = 4\n")})
        result, _planner, reviewer, _claude, pushed = self.run_pipeline(
            luna=luna, reviews=[REVISE_IMPLEMENTATION, PASS], repair_plan=REPAIR_PLAN,
            claude=claude, run_id="c01-snapshot",
        )
        self.assert_published_once(result, pushed, run_id="c01-snapshot",
                                   expected_pushes=2, expected_commits=2)
        self.assertEqual(len(reviewer.prompts), 2)
        run_dir = result.run_dir
        green = self.evidence(run_dir / "checks" / "C01" / "evidence.json")
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual(
            green["staged_tree_sha"],
            json.loads((run_dir / "candidate/C01/commit.json").read_text())["tree_sha"],
        )
        # The pre-repair red evidence stays archived and never returns.
        red = self.evidence(run_dir / "checks/C01/attempts/01/evidence.json")
        self.assertFalse(red["deterministic_passed"])
        self.assertNotEqual(red["staged_tree_sha"], green["staged_tree_sha"])
        self.assertTrue(self.evidence(run_dir / "checks/C02/evidence.json")["deterministic_passed"])

    # -- frozen catalogue -------------------------------------------------
    def catalogue_config(self) -> tuple[Any, Path]:
        """A three-check catalogue; C01 selects only ``lint`` and ``gate``."""

        self.integration_marker = self.root / "integration-ran.txt"
        integration = self.root / "integration.py"
        integration.write_text(
            "import sys\nopen(sys.argv[1], 'a').write('frozen\\n')\nsys.exit(0)\n",
            encoding="utf-8",
        )
        text = self.config_text()
        text = text.replace("require_clean_base = true",
                            'require_clean_base = true\ndefault_check_ids = ["gate"]', 1)
        text = text.replace('[[checks]]\nname = "gate"', '[[check_catalog]]\nid = "gate"', 1)
        text += textwrap.dedent(f"""
            [[check_catalog]]
            id = "lint"
            argv = [{sys.executable!r}, "-c", "pass"]
            timeout_seconds = 30

            [[check_catalog]]
            id = "integration"
            argv = [{sys.executable!r}, {str(integration)!r}, {str(self.integration_marker)!r}]
            timeout_seconds = 30
            """)
        path = self.root / "catalogue.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path), path

    def test_r_c02_may_require_a_frozen_check_the_current_toml_has_changed(self) -> None:
        config, path = self.catalogue_config()
        changed_marker = self.root / "changed-ran.txt"
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"),
                         (2, "S01"): writer("src/a.py", "A = 3\n")})
        planner = QueueClient("planner", [
            required_checks_section(SINGLE_PLAN, "lint", "gate"),
            required_checks_section(REPAIR_PLAN, "lint", "gate", "integration"),
        ], self.events)
        reviewer = QueueClient("reviewer", [REVISE_IMPLEMENTATION, PASS], self.events)
        claude = FakeClaude(log=self.events)
        orchestrator = Orchestrator(config, planner_client=planner, reviewer_client=reviewer,
                                    agent=luna, reviser=claude)

        # After C01 the operator rewrites ``integration`` in the TOML.  The
        # replacement command must never run: C02 uses the approved catalogue.
        changed = self.root / "changed.py"
        changed.write_text(
            "import sys\nopen(sys.argv[1], 'a').write('changed\\n')\n", encoding="utf-8",
        )
        real_repair = Orchestrator._execute_v2_repair_cycle

        def rewrite_then_repair(self_, **kwargs: Any):
            text = path.read_text(encoding="utf-8").replace(
                f"{str(self.root / 'integration.py')!r}, {str(self.integration_marker)!r}",
                f"{str(changed)!r}, {str(changed_marker)!r}",
            )
            path.write_text(text, encoding="utf-8")
            self_.config = load_config(path)
            return real_repair(self_, **kwargs)

        with mock.patch.object(Orchestrator, "_execute_v2_repair_cycle", rewrite_then_repair):
            result = orchestrator.run_text(SPEC, run_id="frozen-catalogue")

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(result.state["deterministic_gate"]["required_check_ids"],
                         ["gate", "lint", "integration"])
        # The frozen integration command ran; the rewritten one never did.
        self.assertTrue(self.integration_marker.exists())
        self.assertFalse(changed_marker.exists())
        authority = json.loads((result.run_dir / "check_authority.json").read_text())
        self.assertEqual(authority["schema_version"], 2)
        self.assertEqual(authority["required_check_ids"], ["gate", "lint"])
        self.assertEqual(sorted(entry["id"] for entry in authority["checks"]),
                         ["gate", "integration", "lint"])

    def test_s_a_repair_check_absent_from_the_authority_fails_closed(self) -> None:
        config, path = self.catalogue_config()
        late_marker = self.root / "late-ran.txt"
        late = self.root / "late.py"
        late.write_text("import sys\nopen(sys.argv[1], 'a').write('late\\n')\n", encoding="utf-8")
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        planner = QueueClient("planner", [
            required_checks_section(SINGLE_PLAN, "gate"),
            required_checks_section(REPAIR_PLAN, "gate", "late"),
        ], self.events)
        orchestrator = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [REVISE_IMPLEMENTATION, PASS], self.events),
            agent=luna, reviser=FakeClaude(log=self.events),
        )
        real_repair = Orchestrator._execute_v2_repair_cycle

        def add_late_check(self_, **kwargs: Any):
            path.write_text(path.read_text(encoding="utf-8") + textwrap.dedent(f"""
                [[check_catalog]]
                id = "late"
                argv = [{sys.executable!r}, {str(late)!r}, {str(late_marker)!r}]
                timeout_seconds = 30
                """), encoding="utf-8")
            self_.config = load_config(path)
            return real_repair(self_, **kwargs)

        with mock.patch.object(Orchestrator, "_execute_v2_repair_cycle", add_late_check):
            result = orchestrator.run_text(SPEC, run_id="late-check")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "CHECK_SETUP_INVALID")
        self.assertIn("late", result.state["failure"]["detail"])
        self.assertFalse(late_marker.exists())
        self.assertFalse((result.run_dir / "checks" / "C02" / "evidence.json").exists())

    def test_t_a_rewritten_check_authority_is_rejected_before_any_check_runs(self) -> None:
        """A live TOCTOU: canonical, same IDs, attacker-controlled command."""

        for field, mutate in (
            ("argv", lambda check, marker: dataclasses.replace(
                check, argv=(sys.executable, str(marker[0]), str(marker[1])))),
            ("cwd", lambda check, marker: dataclasses.replace(check, cwd="src")),
            ("timeout_seconds", lambda check, marker: dataclasses.replace(
                check, timeout_seconds=1)),
            ("preflight_argv", lambda check, marker: dataclasses.replace(
                check, preflight_argv=(sys.executable, str(marker[0]), str(marker[1])))),
        ):
            with self.subTest(field=field):
                self.setUp()
                self.assert_rewritten_authority_fails(field, mutate)

    def assert_rewritten_authority_fails(self, field: str, mutate: Any) -> None:
        marker_script = self.root / "attacker.py"
        marker_script.write_text(
            "import sys\nopen(sys.argv[1], 'a').write('attacker\\n')\n", encoding="utf-8",
        )
        marker = self.root / f"attacker-{field}.txt"
        config = self.load()

        def rewrite(root: Path) -> None:
            run_dir = self.runs / "toctou"
            frozen = read_check_authority(run_dir)
            (run_dir / "check_authority.json").unlink()
            write_check_authority(
                run_dir,
                tuple(mutate(check, (marker_script, marker)) for check in frozen[1]),
                required_check_ids=frozen[0],
            )
            write(root / "src/a.py", "A = 2\n")

        result, _planner, reviewer, claude, pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): rewrite}), reviews=[PASS], run_id="toctou",
        )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "CHECK_SETUP_INVALID")
        self.assertFalse(marker.exists(), f"the rewritten {field} was executed")
        self.assertEqual(reviewer.prompts, [])
        self.assert_no_commit_no_push(result, pushed, run_id="toctou")

    def test_u_a_structured_error_terminal_fails_even_with_exit_code_zero(self) -> None:
        cases = (
            ("error_max_turns", "CLAUDE_MAX_TURNS"),
            ("error_during_execution", "CLAUDE_FAILED"),
        )
        for subtype, reason in cases:
            with self.subTest(subtype=subtype):
                self.setUp()
                self.assert_error_terminal_fails(subtype, reason)

    def assert_error_terminal_fails(self, subtype: str, reason: str) -> None:
        class ErrorTerminalClaude(FakeClaude):
            def run_revision(self_, prompt: str, worktree: Path, **kwargs: Any) -> ClaudeResult:
                result = super().run_revision(prompt, worktree, **kwargs)
                # A CLI that exits 0 after an explicitly errored terminal.
                return dataclasses.replace(
                    result, exit_code=0, timed_out=False,
                    terminal_type="result", terminal_subtype=subtype,
                    terminal_is_error=True,
                )

        claude = ErrorTerminalClaude()
        run_id = f"terminal-{subtype.replace('_', '-')}"
        result, _planner, reviewer, _claude, pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviews=[PASS], claude=claude, run_id=run_id,
        )
        self.assertEqual(result.state["failure"]["reason"], reason)
        self.assert_no_commit_no_push(result, pushed, run_id=run_id)
        # Nothing downstream of the failed revision happened.
        self.assertEqual(len(claude.calls), 1)
        self.assertEqual(reviewer.prompts, [])
        self.assertFalse((result.run_dir / "checks" / "C01" / "evidence.json").exists())
        self.assertFalse((result.run_dir / "candidate").exists())


THIRD_TREE = "c" * 40


class CheckRepairRetryResumeTests(P28Harness):
    """``FINAL_CHECKS_RETRY`` resumes against the repaired tree.

    ``CHECK_REPAIR_C0x`` and ``FINAL_CHECKS_RETRY_C0x`` have opposite evidence
    invariants.  At ``CHECK_REPAIR`` the repair Claude has not succeeded yet,
    so the canonical bundle is the red first pass for the pre-repair tree.  At
    ``FINAL_CHECKS_RETRY`` that Claude did succeed, the checkpoint tree is the
    *repaired* tree, the red first pass has been archived under
    ``attempts/01`` and the canonical bundle is either absent (the first retry
    crashed) or the retry's own bundle for the repaired tree.
    """

    def evidence(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def checks_ran(self) -> int:
        """How many times the real check argv has executed so far."""

        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def retarget(self, path: Path, tree_sha: str) -> None:
        payload = self.evidence(path)
        payload["staged_tree_sha"] = tree_sha
        path.write_text(json.dumps(payload), encoding="utf-8")

    def resume(self, run_id: str, *, reviews: list[str] | None = None,
               revalidate_integrity: bool = False):
        """Resume with fresh doubles so any model call is observable."""

        planner = QueueClient("planner", [], self.events)
        reviewer = QueueClient("reviewer", list(reviews or []), self.events)
        claude, luna = FakeClaude(), FakeLuna({})
        result = Orchestrator(
            self.config_value, planner_client=planner, reviewer_client=reviewer,
            agent=luna, reviser=claude,
        ).resume(run_id, revalidate_integrity=revalidate_integrity)
        return result, planner, reviewer, claude, luna

    def assert_no_agent_ran(self, planner, reviewer, claude, luna) -> None:
        self.assertEqual(planner.prompts, [])
        self.assertEqual(reviewer.prompts, [])
        self.assertEqual(claude.calls, [])
        self.assertEqual(luna.calls, [])

    def assert_only_the_second_repair_ran(self, planner, reviewer, claude, luna) -> None:
        """The retry bridge spends exactly the one remaining bounded repair."""

        self.assertEqual(planner.prompts, [])
        self.assertEqual(reviewer.prompts, [])
        self.assertEqual([call["stage"] for call in claude.calls], ["check-repair"])
        self.assertEqual(luna.calls, [])

    def crash_before_the_second_repair(self, cycle: str):
        """Stop exactly between the red retry evidence and the second repair.

        The archive of the red retry bundle is the first write the second
        bounded pass performs, so failing it parks the run on the
        ``FINAL_CHECKS_RETRY_C0x`` boundary with that bundle durable and the
        second pass entirely unspent -- the state these resumes are about.
        """

        real = orchestrator_module._archive_attempt
        seen: list[int] = []

        def archive(directory: Any, **kwargs: Any) -> Any:
            path = Path(directory)
            if path.name == cycle and path.parent.name == "checks":
                seen.append(1)
                if len(seen) == 2:
                    raise RuntimeError("simulated crash before the second repair")
            return real(directory, **kwargs)

        return mock.patch.object(
            orchestrator_module, "_archive_attempt", side_effect=archive
        )

    # -- C01 ---------------------------------------------------------------

    def _red_retry_c01(self, run_id: str) -> Any:
        """A run whose repaired C01 tree is a *different*, still red tree."""

        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={
            (1, "check-repair"): writer("src/a.py", "A = STILL BUG\n"),
        })
        with self.crash_before_the_second_repair("C01"):
            result, _planner, reviewer, claude, pushed = self.run_pipeline(
                luna=luna, reviews=[PASS], claude=claude, run_id=run_id,
            )
        self.assertEqual(result.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual([call["stage"] for call in claude.calls],
                         ["initial-revision", "check-repair"])
        self.assertEqual(read_checkpoint(result.run_dir).phase,
                         ResumePhase.FINAL_CHECKS_RETRY_C01)
        self.assert_no_commit_no_push(result, pushed, run_id=run_id)
        self.assertEqual(reviewer.prompts, [])
        return result

    def repair_trees(self, run_dir: Path, cycle: str) -> tuple[str, str]:
        report = json.loads(
            (run_dir / "revision" / "check-repair" / cycle / "report.json").read_text()
        )
        self.assertNotEqual(report["tree_before"], report["tree_after"])
        return report["tree_before"], report["tree_after"]

    def test_r_a_red_c01_retry_resumes_and_replays_only_the_retry_checks(self) -> None:
        first = self._red_retry_c01("retry-red-c01")
        run_dir = first.run_dir
        checks = run_dir / "checks" / "C01"
        initial_tree, repaired_tree = self.repair_trees(run_dir, "C01")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C01)
        self.assertEqual(checkpoint.expected_tree_sha, repaired_tree)
        self.assertEqual(
            self.evidence(checks / "attempts" / "01" / "evidence.json")["staged_tree_sha"],
            initial_tree,
        )
        current = self.evidence(checks / "evidence.json")
        self.assertEqual(current["staged_tree_sha"], repaired_tree)
        self.assertIn("CHECK_FAILED:gate", current["failures"])

        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-red-c01")
        # The refusal is gone.  The retry's own red bundle for the repaired
        # tree is already the durable result of a complete retry, so it is
        # reused instead of paying for the same checks twice -- it *is* the
        # authority that earns the one remaining bounded repair.  Its output
        # names no tracked test path, so nothing is added to the scope; the
        # second pass runs inside the same scope and the gate is then final.
        self.assertEqual(second.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assert_only_the_second_repair_ran(planner, reviewer, claude, luna)
        scope = json.loads(
            (run_dir / "revision/check-repair-expanded/C01/scope.json").read_text()
        )
        self.assertEqual(scope["source"], "bounded same-scope retry")
        self.assertEqual(scope["added_paths"], [])
        # The red retry bundle is archived as attempt #2, and the only checks
        # this resume pays for are the second pass's own retry.
        self.assertEqual(self.checks_ran() - before, 1)
        self.assertEqual(
            self.evidence(checks / "attempts" / "02" / "evidence.json")["staged_tree_sha"],
            repaired_tree,
        )
        self.assertFalse((checks / "attempts" / "03").exists())

    def test_s_a_c01_retry_crash_before_its_evidence_still_resumes(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={
            (1, "check-repair"): writer("src/a.py", "A = STILL BUG\n"),
        })
        real = Orchestrator._final_evidence
        calls: list[int] = []

        def crash_inside_the_retry(self_: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("simulated crash inside the retry checks")
            return real(self_, *args, **kwargs)

        with mock.patch.object(Orchestrator, "_final_evidence", crash_inside_the_retry):
            first, _planner, reviewer, _claude, pushed = self.run_pipeline(
                luna=luna, reviews=[PASS], claude=claude, run_id="retry-crash-c01",
            )
        self.assertEqual(first.state["failure"]["reason"], "RUNTIMEERROR")
        self.assert_no_commit_no_push(first, pushed, run_id="retry-crash-c01")
        self.assertEqual(reviewer.prompts, [])
        run_dir = first.run_dir
        checks = run_dir / "checks" / "C01"
        initial_tree, repaired_tree = self.repair_trees(run_dir, "C01")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.expected_tree_sha),
                         (ResumePhase.FINAL_CHECKS_RETRY_C01, repaired_tree))
        # The red first pass is archived and there is no current bundle.
        self.assertFalse((checks / "evidence.json").exists())
        self.assertEqual(
            self.evidence(checks / "attempts" / "01" / "evidence.json")["staged_tree_sha"],
            initial_tree,
        )
        # The root aliases still name the first pass, which is exactly why
        # they are never the authority for the repaired tree.
        self.assertEqual(self.evidence(run_dir / "evidence.json")["staged_tree_sha"],
                         initial_tree)

        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-crash-c01")
        self.assertEqual(second.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        # The retry checks this crash never finished, then the one remaining
        # bounded repair and its own retry.
        self.assertEqual(self.checks_ran() - before, 2)
        self.assert_only_the_second_repair_ran(planner, reviewer, claude, luna)
        self.assertEqual(self.evidence(checks / "evidence.json")["staged_tree_sha"],
                         repaired_tree)

    def test_t_a_c01_retry_bundle_for_a_third_tree_fails_closed(self) -> None:
        first = self._red_retry_c01("retry-third-c01")
        checks = first.run_dir / "checks" / "C01"
        self.retarget(checks / "evidence.json", THIRD_TREE)
        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-third-c01")
        self.assertEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.checks_ran(), before)
        self.assert_no_agent_ran(planner, reviewer, claude, luna)

    def test_u_a_c01_archived_first_pass_for_a_wrong_tree_fails_closed(self) -> None:
        first = self._red_retry_c01("retry-archive-c01")
        checks = first.run_dir / "checks" / "C01"
        self.retarget(checks / "attempts" / "01" / "evidence.json", THIRD_TREE)
        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-archive-c01")
        self.assertEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.checks_ran(), before)
        self.assert_no_agent_ran(planner, reviewer, claude, luna)

    # -- C02 ---------------------------------------------------------------

    def _red_retry_c02(self, run_id: str) -> Any:
        """The same four states, one cycle later."""

        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"),
                         (2, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={
            (2, "check-repair"): writer("src/a.py", "A = STILL BUG\n"),
        })
        with self.crash_before_the_second_repair("C02"):
            result, _planner, reviewer, claude, pushed = self.run_pipeline(
                luna=luna, reviews=[REVISE_IMPLEMENTATION, PASS], repair_plan=REPAIR_PLAN,
                claude=claude, run_id=run_id,
            )
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual([(call["cycle"], call["stage"]) for call in claude.calls], [
            (1, "initial-revision"), (2, "initial-revision"), (2, "check-repair"),
        ])
        self.assertEqual(read_checkpoint(result.run_dir).phase,
                         ResumePhase.FINAL_CHECKS_RETRY_C02)
        # Only reviewer #1 ran: C02 never reached its own reviewer.
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(pushed.call_count, 1)
        return result

    def test_v_a_red_c02_retry_resumes_and_replays_only_the_retry_checks(self) -> None:
        first = self._red_retry_c02("retry-red-c02")
        run_dir = first.run_dir
        checks = run_dir / "checks" / "C02"
        initial_tree, repaired_tree = self.repair_trees(run_dir, "C02")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C02)
        self.assertEqual(checkpoint.expected_tree_sha, repaired_tree)
        self.assertEqual(
            self.evidence(checks / "attempts" / "01" / "evidence.json")["staged_tree_sha"],
            initial_tree,
        )
        self.assertEqual(self.evidence(checks / "evidence.json")["staged_tree_sha"],
                         repaired_tree)

        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-red-c02")
        self.assertEqual(second.status, RunStatus.FAILED)
        self.assertNotEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        # Same reuse as C01: the durable red retry bundle is the authority,
        # and it earns exactly one same-scope second repair -- never a third.
        self.assertEqual(self.checks_ran() - before, 1)
        self.assert_only_the_second_repair_ran(planner, reviewer, claude, luna)
        scope = json.loads(
            (run_dir / "revision/check-repair-expanded/C02/scope.json").read_text()
        )
        self.assertEqual(scope["source"], "bounded same-scope retry")
        self.assertEqual(scope["added_paths"], [])
        self.assertEqual(
            self.evidence(checks / "attempts" / "02" / "evidence.json")["staged_tree_sha"],
            repaired_tree,
        )
        self.assertFalse((checks / "attempts" / "03").exists())

    def test_w_a_c02_retry_crash_before_its_evidence_still_resumes(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"),
                         (2, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(stage_actions={
            (2, "check-repair"): writer("src/a.py", "A = STILL BUG\n"),
        })
        real = Orchestrator._final_evidence
        calls: list[int] = []

        def crash_inside_the_c02_retry(self_: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            # C01 final checks, C02 final checks, then the C02 retry.
            if len(calls) == 3:
                raise RuntimeError("simulated crash inside the C02 retry checks")
            return real(self_, *args, **kwargs)

        with mock.patch.object(Orchestrator, "_final_evidence", crash_inside_the_c02_retry):
            first, _planner, reviewer, _claude, _pushed = self.run_pipeline(
                luna=luna, reviews=[REVISE_IMPLEMENTATION, PASS], repair_plan=REPAIR_PLAN,
                claude=claude, run_id="retry-crash-c02",
            )
        self.assertEqual(first.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual(len(reviewer.prompts), 1)
        run_dir = first.run_dir
        checks = run_dir / "checks" / "C02"
        initial_tree, repaired_tree = self.repair_trees(run_dir, "C02")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.expected_tree_sha),
                         (ResumePhase.FINAL_CHECKS_RETRY_C02, repaired_tree))
        self.assertFalse((checks / "evidence.json").exists())
        self.assertEqual(
            self.evidence(checks / "attempts" / "01" / "evidence.json")["staged_tree_sha"],
            initial_tree,
        )

        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-crash-c02")
        self.assertEqual(second.status, RunStatus.FAILED)
        self.assertNotEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        # The retry checks this crash never finished, then the one remaining
        # bounded repair and its own retry.
        self.assertEqual(self.checks_ran() - before, 2)
        self.assert_only_the_second_repair_ran(planner, reviewer, claude, luna)
        self.assertEqual(self.evidence(checks / "evidence.json")["staged_tree_sha"],
                         repaired_tree)

    def test_x_a_c02_retry_bundle_for_a_third_tree_fails_closed(self) -> None:
        first = self._red_retry_c02("retry-third-c02")
        self.retarget(first.run_dir / "checks" / "C02" / "evidence.json", THIRD_TREE)
        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-third-c02")
        self.assertEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.checks_ran(), before)
        self.assert_no_agent_ran(planner, reviewer, claude, luna)

    def test_y_a_c02_archived_first_pass_for_a_wrong_tree_fails_closed(self) -> None:
        first = self._red_retry_c02("retry-archive-c02")
        self.retarget(
            first.run_dir / "checks" / "C02" / "attempts" / "01" / "evidence.json",
            THIRD_TREE,
        )
        before = self.checks_ran()
        second, planner, reviewer, claude, luna = self.resume("retry-archive-c02")
        self.assertEqual(second.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.checks_ran(), before)
        self.assert_no_agent_ran(planner, reviewer, claude, luna)

    # -- operator revalidation --------------------------------------------

    def test_z_revalidate_integrity_reopens_the_validation_without_bypassing_it(self) -> None:
        """``--revalidate-integrity`` re-opens the validation, nothing else."""

        first = self._red_retry_c01("revalidate-c01")
        run_dir = first.run_dir
        checks = run_dir / "checks" / "C01"
        _initial_tree, repaired_tree = self.repair_trees(run_dir, "C01")
        intact = self.evidence(checks / "evidence.json")
        self.retarget(checks / "evidence.json", THIRD_TREE)

        # 1. A broken invariant closes the run and leaves it non-resumable.
        refused, planner, reviewer, claude, luna = self.resume("revalidate-c01")
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assert_no_agent_ran(planner, reviewer, claude, luna)
        state = json.loads((run_dir / "state.json").read_text())
        self.assertFalse(resume_info(run_dir, state).resumable)
        self.assertEqual(read_checkpoint_record(run_dir)[1], "pending")

        # 2. Without the flag the run stays refused before any validation.
        with self.assertRaises(ResumeNotAllowedError):
            self.resume("revalidate-c01")

        # 3. With the flag the *complete* validation runs again -- and still
        #    refuses, without a model call, a check or a Git write.
        before = self.checks_ran()
        head = git(self.worktree("revalidate-c01"), "rev-parse", "HEAD")
        again, planner, reviewer, claude, luna = self.resume(
            "revalidate-c01", revalidate_integrity=True,
        )
        self.assertEqual(again.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.checks_ran(), before)
        self.assert_no_agent_ran(planner, reviewer, claude, luna)
        self.assertEqual(git(self.worktree("revalidate-c01"), "rev-parse", "HEAD"), head)
        self.assertIsNone(again.state.get("commit_sha"))
        self.assertEqual(read_checkpoint_record(run_dir)[1], "pending")
        self.assertTrue(
            resume_info(
                run_dir, json.loads((run_dir / "state.json").read_text()),
                revalidate_integrity=True,
            ).resumable
        )

        # 4. Once the invariant holds again, the same checkpoint is resumed
        #    and its restored durable bundle -- not a fresh check run -- is
        #    the authority that spends the one remaining bounded repair
        #    before the deterministic gate closes.
        (checks / "evidence.json").write_text(json.dumps(intact), encoding="utf-8")
        resumed, planner, reviewer, claude, luna = self.resume(
            "revalidate-c01", revalidate_integrity=True,
        )
        self.assertEqual(resumed.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(self.checks_ran() - before, 1)
        self.assert_only_the_second_repair_ran(planner, reviewer, claude, luna)
        self.assertEqual(
            self.evidence(checks / "attempts" / "02" / "evidence.json")["staged_tree_sha"],
            repaired_tree,
        )

    def test_z_revalidate_integrity_is_rejected_for_any_other_failure(self) -> None:
        first = self._red_retry_c01("revalidate-other")
        state = json.loads((first.run_dir / "state.json").read_text())
        self.assertEqual(state["failure"]["reason"], "RUNTIMEERROR")
        info = resume_info(first.run_dir, state, revalidate_integrity=True)
        self.assertFalse(info.resumable)
        self.assertEqual(info.reason, "this run is not eligible for an integrity revalidation")
        with self.assertRaises(ResumeNotAllowedError):
            self.resume("revalidate-other", revalidate_integrity=True)
        # The ordinary resume of that same run is unaffected.
        self.assertTrue(resume_info(first.run_dir, state).resumable)


class ApprovalV4Tests(P28Harness):
    def _await(self, config: HarnessConfig, luna: FakeLuna, run_id: str):
        planner = QueueClient("planner", [SINGLE_PLAN], self.events)
        reviewer = QueueClient("reviewer", [PASS], self.events)
        orchestrator = Orchestrator(config, planner_client=planner, reviewer_client=reviewer,
                                    agent=luna, reviser=FakeClaude())
        holder: dict[str, Any] = {}
        thread = threading.Thread(target=lambda: holder.setdefault(
            "result", orchestrator.run_text(SPEC, run_id=run_id)))
        thread.start()
        state_path = self.runs / run_id / "state.json"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if state_path.exists() and json.loads(state_path.read_text())["status"] == "awaiting_plan_approval":
                break
            time.sleep(0.01)
        return thread, holder

    def test_web_approval_uses_v4_and_shows_four_families(self) -> None:
        config = load_config(self._write(require_approval=True))
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        thread, holder = self._await(config, luna, "approve")
        page = render_run(get_run(self.runs, "approve"), "token", config=config)
        approve_at = page.index("APPROVE PLAN")
        for heading in ("<h3>Planner</h3>", "<h3>Initial implementation</h3>", "<h3>Reviser</h3>",
                        "<h3>Repair implementer</h3>", "<h3>Reviewer</h3>"):
            self.assertLess(page.index(heading), approve_at, heading)
        self.assertIn('name="repair_profile"', page)
        repair_select = page[page.index('name="repair_profile"'):]
        repair_select = repair_select[:repair_select.index("</select>")]
        self.assertIn('value="luna" selected', repair_select)
        self.assertNotIn('value="claude"', repair_select)
        with self.assertRaises(WebAPIError) as refused:
            approve_run(config.runs_root, "approve", "APPROVE", config=config,
                        reviewer_profile="reviewer", step_profiles={"S01": "luna"},
                        repair_profile="claude")
        self.assertEqual(refused.exception.status, 400)
        approve_run(config.runs_root, "approve", "APPROVE", config=config,
                    reviewer_profile="reviewer", step_profiles={"S01": "luna"})
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        result = holder["result"]
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        selection = json.loads((self.runs / "approve" / "execution_selection.json").read_text())
        self.assertEqual(selection["schema_version"], 4)
        self.assertEqual(selection["repair_implementer"]["profile_id"], "luna")
        self.assertEqual(selection["reviser"]["profile_id"], "claude")
        self.assertIn("repair_implementer", result.state["execution"])

    def test_disabled_revision_rejects_reviser_fields(self) -> None:
        config = load_config(self._write(require_approval=True, revision=False))
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        thread, holder = self._await(config, luna, "disabled")
        page = render_run(get_run(self.runs, "disabled"), "token", config=config)
        self.assertNotIn('name="reviser_profile"', page)
        self.assertNotIn('name="repair_profile"', page)
        with self.assertRaises(WebAPIError):
            approve_run(config.runs_root, "disabled", "APPROVE", config=config,
                        reviewer_profile="reviewer", step_profiles={"S01": "luna"},
                        reviser_profile="claude")
        approve_run(config.runs_root, "disabled", "APPROVE", config=config,
                    reviewer_profile="reviewer", step_profiles={"S01": "luna"})
        thread.join(timeout=30)
        self.assertEqual(holder["result"].status, RunStatus.PUBLISHED)
        selection = json.loads((self.runs / "disabled" / "execution_selection.json").read_text())
        self.assertEqual(selection["schema_version"], 3)

    def _write(self, **kwargs: Any) -> Path:
        path = self.root / "approval.toml"
        path.write_text(self.config_text(**kwargs), encoding="utf-8")
        return path


class ConfigActivationTests(P28Harness):
    def _load_variant(self, old: str, new: str) -> HarnessConfig:
        text = self.config_text()
        self.assertIn(old, text)
        path = self.root / "variant.toml"
        path.write_text(text.replace(old, new), encoding="utf-8")
        return load_config(path)

    def test_enabled_requires_the_complete_architecture_at_load_time(self) -> None:
        cases = (
            ('protocol = "v2"', 'protocol = "v1"', "planning.protocol"),
            ('default_repair_profile = "luna"\n', "", "default_repair_profile"),
            ('default_reviser_profile = "claude"\n', "", "default_reviser_profile"),
            ('default_repair_profile = "luna"', 'default_repair_profile = "claude"',
             "revision repair profile must use codex driver"),
            ("max_cycles = 2", "max_cycles = 3", "max_cycles must be exactly 2"),
            ("enabled = true\nmax_cycles", 'enabled = "yes"\nmax_cycles', "revision.enabled"),
        )
        for old, new, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ConfigError, re.escape(message)):
                    self._load_variant(old, new)

    def test_catalogue_profiles_never_enable_revision(self) -> None:
        config = self._load_variant("enabled = true\nmax_cycles", "enabled = false\nmax_cycles")
        self.assertFalse(config.revision.enabled)
        # Defaults are explicit only; a lone catalogue reviser is not a default.
        text = self.config_text(revision=False).replace('default_reviser_profile = "claude"\n', "")
        path = self.root / "implicit.toml"
        path.write_text(text, encoding="utf-8")
        self.assertIsNone(load_config(path).ui.default_reviser_profile)
        self.assertEqual(RevisionConfig().enabled, False)

    def test_v4_selection_requires_claude_reviser_and_codex_repair(self) -> None:
        config = self.load()
        with self.assertRaisesRegex(ExecutionSelectionError, "codex driver"):
            resolve_execution_selection_v4(
                config, planner_profile_id="planner", step_profile_ids={"S01": "luna"},
                reviser_profile_id="claude", repair_implementer_profile_id="claude",
                reviewer_profile_id="reviewer",
            )

    def test_i_production_example_is_the_target_workflow(self) -> None:
        raw = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(raw["planning"]["protocol"], "v2")
        self.assertEqual(raw["planning"]["decomposition"], "aggressive")
        self.assertIs(raw["revision"]["enabled"], True)
        self.assertEqual(raw["revision"]["max_cycles"], 2)
        self.assertEqual(raw["ui"]["default_reviser_profile"], "claude-opus-medium")
        self.assertEqual(raw["ui"]["default_repair_profile"], "codex-luna-high")
        self.assertIs(raw["publish"]["enabled"], True)
        # P29: publish the final reviewed commit to main by safe fast-forward,
        # and require the planner to decompose (STAGED) for real work.
        self.assertEqual(raw["publish"]["mode"], "fast-forward-base")
        self.assertEqual(raw["planning"]["execution_mode_policy"], "require-staged")
        self.assertEqual(raw["publish"]["remote"], "origin")
        # Validate the real profiles through load_config with local paths.
        text = EXAMPLE.read_text(encoding="utf-8")
        replacements = {
            r'^repo = .*$': f"repo = {str(self.repo)!r}",
            r'^runs_root = .*$': f"runs_root = {str(self.runs)!r}",
            r'^worktrees_root = .*$': f"worktrees_root = {str(self.root / 'wt')!r}",
            r'^files = \[.*\]$': "files = []",
            r'^home = "~/.local/share/metaharness/codex"$': f"home = {str(self.root / 'codex')!r}",
            r'^home = "~/.local/share/metaharness/claude"$': f"home = {str(self.root / 'claude')!r}",
        }
        for pattern, value in replacements.items():
            text, count = re.subn(pattern, lambda _match, value=value: value, text, flags=re.M)
            self.assertEqual(count, 1, pattern)
        path = self.root / "autowork-local.toml"
        path.write_text(text, encoding="utf-8")
        config = load_config(path)
        self.assertTrue(config.revision.enabled)
        self.assertTrue(config.publish.enabled)
        self.assertEqual(config.planning.decomposition, "aggressive")
        reviser = config.model_profiles[config.ui.default_reviser_profile]
        repair = config.model_profiles[config.ui.default_repair_profile]
        self.assertEqual((reviser.driver, repair.driver), (ProfileDriver.CLAUDE_CODE, ProfileDriver.CODEX))
        self.assertIn(ExecutionRole.REVISER, reviser.roles)
        self.assertIn(ExecutionRole.REPAIR, repair.roles)
        selection = resolve_execution_selection_v4(
            config, planner_profile_id="planner-chatgpt",
            step_profile_ids={"S01": "codex-luna-high"},
            reviser_profile_id=config.ui.default_reviser_profile,
            repair_implementer_profile_id=config.ui.default_repair_profile,
            reviewer_profile_id="planner-chatgpt",
        )
        self.assertEqual(selection.schema_version, 4)


class PublicationUrlTests(unittest.TestCase):
    def test_branch_url_keeps_slashes(self) -> None:
        reference = RepositoryReference("origin", "https://github.com/OWNER/REPO", "a" * 40, None)
        url = run_branch_web_url(reference, "harness/my-plan/run-123")
        self.assertEqual(url, "https://github.com/OWNER/REPO/tree/harness/my-plan/run-123")
        self.assertNotIn("%2F", url)

    def test_web_url_must_match_a_normalizable_remote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            git(repo, "remote", "add", "origin", "git@github.com:OWNER/REPO.git")
            with self.assertRaisesRegex(ValueError, "repository.web_url does not match configured Git remote"):
                build_repository_reference(
                    repo, base_sha="b" * 40,
                    config=RepositoryConfig(web_url="https://github.com/OTHER/REPO"),
                )
            reference = build_repository_reference(
                repo, base_sha="b" * 40,
                config=RepositoryConfig(web_url="https://github.com/OWNER/REPO"),
            )
            self.assertEqual(reference.web_url, "https://github.com/OWNER/REPO")


class DoctorTests(unittest.TestCase):
    """J: Codex and Claude diagnostics are independent; no model is called."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        repo = self.root / "repo"
        repo.mkdir()
        for args in (("init", "-q"), ("config", "user.name", "D"), ("config", "user.email", "d@example.invalid")):
            git(repo, *args)
        write(repo / "README.md", "base\n")
        git(repo, "add", "--all")
        git(repo, "commit", "-qm", "base")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # PATH contains only the fakes and git: a real claude is never found.
        self.gitbin = self.root / "gitbin"
        self.gitbin.mkdir()
        (self.gitbin / "git").symlink_to(subprocess.run(
            ["which", "git"], capture_output=True, text=True, check=True).stdout.strip())
        self.codex_mode = self.root / "codex-mode"
        self.claude_mode = self.root / "claude-mode"
        self.claude_record = self.root / "claude-env.json"
        self._script("codex", """
            import os, sys
            args = sys.argv[1:]
            mode = open(MODE).read().strip() if os.path.exists(MODE) else "available"
            if args == ["login", "--help"]:
                print("Commands:\\n  status  Show login status"); sys.exit(0)
            if args == ["login", "status"]:
                if mode == "available":
                    print("Logged in"); sys.exit(0)
                if mode == "unavailable":
                    sys.stderr.write("Not logged in\\n"); sys.exit(1)
                sys.stderr.write("unexpected internal error\\n"); sys.exit(2)
            sys.exit(0)
        """, self.codex_mode)
        self.claude_script = """
            import json, os, sys
            args = sys.argv[1:]
            mode = open(MODE).read().strip() if os.path.exists(MODE) else "pass"
            with open(RECORD, "w") as stream:
                json.dump({key: os.environ.get(key) for key in ("HOME", "CLAUDE_CONFIG_DIR", "TMPDIR", "XDG_CACHE_HOME", "CODEX_HOME")}, stream)
            if args == ["auth", "--help"]:
                print("Commands:\\n  status  Show authentication status"); sys.exit(0)
            if args and args[-1] == "--help":
                flags = ["--print", "--verbose", "--output-format", "--model", "--effort", "--permission-mode", "--mcp-config", "--strict-mcp-config"]
                if mode == "missing_capability":
                    sys.exit(2)
                print(" ".join(flags))
                sys.exit(3 if mode == "help_nonzero" else 0)
            if args == ["auth", "status"]:
                if mode == "auth_failure":
                    print("Not logged in"); sys.exit(1)
                print("Logged in"); sys.exit(0)
            sys.exit(0)
        """

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _script(self, name: str, body: str, mode: Path, record: Path | None = None) -> None:
        path = self.bin / name
        source = textwrap.dedent(body).replace("MODE", repr(str(mode)))
        if record is not None:
            source = source.replace("RECORD", repr(str(record)))
        path.write_text(f"#!{sys.executable}\n" + source, encoding="utf-8")
        path.chmod(0o755)

    def _config(self, *, claude_home: Path | None = None) -> Path:
        codex_home = self.root / "codex-home"
        path = self.root / "doctor.toml"
        path.write_text(textwrap.dedent(f"""
            repo = "repo"
            base_ref = "HEAD"
            runs_root = "runs"
            worktrees_root = "worktrees"

            [codex_runtime]
            home = {str(codex_home)!r}

            [claude_runtime]
            home = {str(claude_home or (self.root / 'claude-home'))!r}

            [ui]
            default_planner_profile = "bridge"
            default_implementer_profile = "impl"
            default_reviewer_profile = "bridge"
            default_reviser_profile = "claude"

            [model_profiles.bridge]
            display_name = "Bridge"
            roles = ["planner", "reviewer"]
            driver = "openai-chat"
            model = "remote"
            selection_mode = "request"
            base_url = "https://bridge.example.invalid"
            endpoint_path = "/v1/chat/completions"

            [model_profiles.impl]
            display_name = "Impl"
            roles = ["implementer"]
            driver = "codex"
            model = "luna"
            effort = "high"
            sandbox = "workspace-write"
            selection_mode = "cli"

            [model_profiles.claude]
            display_name = "Claude"
            roles = ["reviser"]
            driver = "claude-code"
            model = "opus"
            effort = "medium"
            permission_mode = "acceptEdits"
            selection_mode = "cli"

            [[checks]]
            name = "test"
            argv = [{sys.executable!r}, "-c", "pass"]
        """), encoding="utf-8")
        return path

    def doctor(self, *, codex: str = "available", claude: str | None = "pass",
               claude_home: Path | None = None) -> tuple[int, str, list[str]]:
        self.codex_mode.write_text(codex, encoding="utf-8")
        claude_path = self.bin / "claude"
        if claude is None:
            claude_path.unlink(missing_ok=True)
        else:
            self._script("claude", self.claude_script, self.claude_mode, self.claude_record)
            self.claude_mode.write_text(claude, encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        environment = {"PATH": f"{self.bin}{os.pathsep}{self.gitbin}", "HOME": str(self.root / "personal")}
        with mock.patch.dict(os.environ, environment, clear=False):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main(["doctor", "--config", str(self._config(claude_home=claude_home))])
        errors = [line[len("error: "):] for line in stderr.getvalue().splitlines() if line.startswith("error: ")]
        return code, stdout.getvalue(), errors

    def test_codex_auth_branches(self) -> None:
        code, out, errors = self.doctor(codex="available")
        self.assertEqual((code, errors), (0, []), out)
        self.assertIn("OK codex authentication: available", out)
        self.assertIn("doctor: PASS", out)
        code, _out, errors = self.doctor(codex="unavailable")
        self.assertEqual((code, errors), (1, ["codex authentication is unavailable for managed CODEX_HOME"]))
        code, out, errors = self.doctor(codex="unverifiable")
        self.assertEqual((code, errors), (1, ["codex authentication could not be verified for managed CODEX_HOME"]))
        self.assertIn("OK claude authentication: available", out)

    def test_claude_diagnostics_are_single_and_never_codex(self) -> None:
        not_a_directory = self.root / "claude-file"
        not_a_directory.write_text("x", encoding="utf-8")
        cases = (
            ({"claude": None}, "claude binary is not resolvable"),
            ({"claude_home": not_a_directory}, "could not prepare managed Claude config home"),
            ({"claude": "help_nonzero"}, "unsupported Claude Code CLI for MetaHarness reviser"),
            ({"claude": "missing_capability"}, "unsupported Claude Code CLI for MetaHarness reviser"),
            ({"claude": "auth_failure"}, "Claude Code authentication unavailable"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message, kwargs=kwargs):
                code, out, errors = self.doctor(**kwargs)
                self.assertEqual((code, errors), (1, [message]))
                self.assertIn("OK codex authentication: available", out)
                self.assertFalse(any(error.startswith("codex") for error in errors))

    def test_help_probe_nonzero_is_unsupported_even_with_every_flag(self) -> None:
        self._script("claude", self.claude_script, self.claude_mode, self.claude_record)
        self.claude_mode.write_text("help_nonzero", encoding="utf-8")
        home = self.root / "probe-home"
        home.mkdir()
        supported, _detail = cli._probe_claude_capabilities(str(self.bin / "claude"), {"PATH": str(self.bin)}, home)
        self.assertFalse(supported)

    def test_doctor_probes_claude_with_the_managed_environment(self) -> None:
        code, _out, errors = self.doctor()
        self.assertEqual((code, errors), (0, []))
        recorded = json.loads(self.claude_record.read_text())
        home = (self.root / "claude-home").resolve()
        self.assertEqual(recorded, {
            "HOME": str(home / "home"), "CLAUDE_CONFIG_DIR": str(home),
            "TMPDIR": str(home / "tmp"), "XDG_CACHE_HOME": str(home / "cache"),
            "CODEX_HOME": None,
        })


class ClaudeIsolationTests(unittest.TestCase):
    """K: the Claude environment and managed home never reuse personal state."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        git(self.repo, "config", "user.email", "k@example.invalid")
        git(self.repo, "config", "user.name", "k")
        write(self.repo / "README.md", "base\n")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-qm", "base")
        self.personal = self.root / "personal"
        write(self.personal / ".claude" / "settings.json", '{"hooks": {"personal": true}}\n')
        write(self.personal / ".claude.json", '{"mcpServers": {"personal": {"command": "x"}}}\n')
        self.config = HarnessConfig(
            repo=self.repo, base_ref="HEAD", runs_root=self.root / "runs",
            worktrees_root=self.root / "worktrees", require_clean_base=True,
            planner=LLMEndpointConfig("https://planner.invalid", "/v1", "planner"),
            reviewer=LLMEndpointConfig("https://reviewer.invalid", "/v1", "reviewer"),
            context=ContextConfig(always_files=()), agent=AgentConfig(), checks=(),
            allow_no_required_checks=True,
            codex_runtime=CodexRuntimeConfig(self.root / "codex-home"),
            claude_runtime=ClaudeRuntimeConfig(self.root / "claude-home"),
        )
        self.source = {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "xterm",
            "HOME": str(self.personal), "TMPDIR": str(self.personal / "tmp"),
            "XDG_CONFIG_HOME": str(self.personal / ".config"),
            "XDG_CACHE_HOME": str(self.personal / ".cache"),
            "CODEX_HOME": str(self.personal / ".codex"),
            "BRIDGE_API_KEY": "bridge-secret", "OPENAI_API_KEY": "openai-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_environment_is_exactly_the_managed_one(self) -> None:
        with mock.patch.dict(os.environ, {"HOME": str(self.personal)}):
            home = prepare_claude_home(self.config)
        environment = build_claude_environment(self.source, claude_home=home)
        self.assertEqual(environment, {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "xterm",
            "HOME": str(home / "home"), "CLAUDE_CONFIG_DIR": str(home),
            "XDG_CACHE_HOME": str(home / "cache"), "TMPDIR": str(home / "tmp"),
        })
        for value in ("bridge-secret", "openai-secret", "anthropic-secret", str(self.personal)):
            self.assertNotIn(value, json.dumps(environment))
        # Nothing personal is copied: only the managed skeleton exists.
        self.assertEqual(sorted(path.name for path in home.iterdir()),
                         ["cache", "empty-mcp.json", "home", "settings.json", "tmp"])
        self.assertEqual([path for path in (home / "home").rglob("*")], [])
        self.assertEqual((home / "empty-mcp.json").read_text(), '{\n  "mcpServers": {}\n}\n')

    def test_claude_process_cannot_see_personal_configuration(self) -> None:
        capture = self.root / "capture.json"
        executable = self.root / "claude"
        executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""
            import json, os, pathlib, sys
            home = pathlib.Path(os.environ["HOME"])
            pathlib.Path({str(capture)!r}).write_text(json.dumps({{
                "argv": sys.argv[1:],
                "env": dict(os.environ),
                "personal_settings_visible": (home / ".claude" / "settings.json").exists(),
                "personal_mcp_visible": (home / ".claude.json").exists(),
            }}))
            print(json.dumps({{"type": "result", "result": "done"}}))
        """), encoding="utf-8")
        executable.chmod(0o755)
        profile = ModelProfile(
            id="claude", display_name="Claude", roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE, model="opus", selection_mode=SelectionMode.CLI,
            effort="medium", permission_mode="acceptEdits", timeout_seconds=10, retries=0,
        )
        home = prepare_claude_home(self.config)
        ClaudeCodeAgent(executable=str(executable)).run_revision(
            "inspect", self.repo, artifacts_dir=self.root / "run", profile=profile,
            environment=build_claude_environment(self.source, claude_home=home),
        )
        recorded = json.loads(capture.read_text())
        self.assertFalse(recorded["personal_settings_visible"])
        self.assertFalse(recorded["personal_mcp_visible"])
        self.assertEqual(recorded["env"]["HOME"], str(home / "home"))
        self.assertEqual(recorded["env"]["TMPDIR"], str(home / "tmp"))
        for name in ("CODEX_HOME", "BRIDGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XDG_CONFIG_HOME"):
            self.assertNotIn(name, recorded["env"])
        self.assertIn("--strict-mcp-config", recorded["argv"])
        self.assertEqual(recorded["argv"][recorded["argv"].index("--mcp-config") + 1],
                         str(home / "empty-mcp.json"))
        self.assertEqual(sorted(path.name for path in home.iterdir()),
                         ["cache", "empty-mcp.json", "home", "settings.json", "tmp"])


class RepairDecompositionPolicyPipelineTests(P28Harness):
    """The real C02 request states the one limit that actually binds it."""

    def test_the_repair_request_carries_the_staged_per_worker_limit(self) -> None:
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 4\n"),
        })
        result, planner, _r, _c, _pushed = self.run_pipeline(
            luna=luna, reviews=[REVISE_IMPLEMENTATION, PASS],
            repair_plan=REPAIR_PLAN, run_id="p28-repair-policy",
        )
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))

        initial, repair = planner.prompts
        self.assertNotIn("REPAIR DECOMPOSITION POLICY", initial)
        self.assertIn("REPAIR DECOMPOSITION POLICY", repair)
        self.assertIn(
            "Every repair implementation step, including a SINGLE S01", repair
        )
        # The run's staged per-step maximum, not its initial SINGLE threshold.
        self.assertEqual(self.config_value.planning.staged_step_max_mutable_paths, 6)
        self.assertIn("at most 6 distinct mutable\npaths", repair)
        self.assertEqual(repair.count("at most 6"), 1)
        self.assertLess(
            repair.index("REPAIR DECOMPOSITION POLICY"),
            repair.index(
                "The answer must use exactly the existing META PLAN v2 wire protocol."
            ),
        )
        # An authoritative instruction, never part of the evidence packet.
        evidence = (
            result.run_dir / "repair" / "C02" / "planner.evidence.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn("REPAIR DECOMPOSITION POLICY", evidence)
        self.assertNotIn("{{REPAIR_DECOMPOSITION_POLICY}}", repair)

    def test_each_planner_keeps_its_own_decomposition_authority(self) -> None:
        initial = inspect.getsource(planning_v2.PlannerV2.plan)
        repair = inspect.getsource(planning_v2.RepairPlannerV2.plan)
        self.assertIn("validate_decomposition_policy(plan, self.planning)", initial)
        self.assertNotIn("validate_repair_decomposition_policy", initial)
        self.assertIn("validate_repair_decomposition_policy(plan, self.planning)", repair)
        # The repair planner is never told the initial SINGLE threshold.
        self.assertNotIn("single_step_max_mutable_paths", repair)


class StructuralTests(unittest.TestCase):
    """Source guards; the E2E tests above remain the behavioral authority."""

    def test_no_hardcoded_deterministic_pass_in_v2_reviewer_paths(self) -> None:
        for name in ("_run_v2_reviewer", "_execute_v2", "_execute_v2_repair_cycle", "_authorize_v2_commit"):
            source = inspect.getsource(getattr(Orchestrator, name))
            self.assertNotIn("deterministic_passed=True", source, name)
        self.assertIn("deterministic_passed=evidence.deterministic_passed",
                      inspect.getsource(Orchestrator._run_v2_reviewer))
        for name in ("_execute_v2", "_execute_v2_repair_cycle"):
            source = inspect.getsource(getattr(Orchestrator, name))
            self.assertEqual(source.count("self._run_v2_reviewer("), 1, name)
            self.assertNotIn("reviewer.review(", source, name)

    def test_one_authoritative_codex_step_executor(self) -> None:
        module = inspect.getsource(orchestrator_module)
        attempt = inspect.getsource(Orchestrator._run_codex_step_attempt)
        self.assertEqual(module.count(".run_step("), 1)
        self.assertIn(".run_step(", attempt)
        # The bounded mismatch retry is the only reason a step runs twice, and
        # `_execute_codex_step` is the only caller of the attempt executor.
        self.assertEqual(module.count("self._run_codex_step_attempt("), 3)
        self.assertEqual(
            inspect.getsource(Orchestrator._execute_codex_step).count(
                "self._run_codex_step_attempt("
            ),
            3,
        )
        for name in ("_execute_v2", "_execute_v2_repair_cycle"):
            source = inspect.getsource(getattr(Orchestrator, name))
            self.assertEqual(source.count("self._execute_codex_step("), 1, name)
            for forbidden in ("run_step", "build_agent_environment(", "stage_all(", "classify_codex_failure"):
                if forbidden == "stage_all(" and name == "_execute_v2":
                    continue  # the base tree is staged once before any step
                self.assertNotIn(forbidden, source, (name, forbidden))
        self.assertEqual(
            [field.name for field in dataclasses.fields(orchestrator_module.StepExecutionOutcome)],
            ["step_id", "profile_id", "tree_before", "tree_after", "changed_paths", "usage",
             "final_report", "deferred_verify", "mismatch_retry_count"],
        )

    def test_only_gitops_names_history_changing_git_primitives(self) -> None:
        pattern = re.compile(r"[\"'](push|commit|commit-tree|update-ref|merge|rebase|tag|reset)[\"']")
        for path in sorted((ROOT / "src" / "metaharness").rglob("*.py")):
            if path.name == "gitops.py":
                continue
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path)
        gitops = (ROOT / "src" / "metaharness" / "gitops.py").read_text(encoding="utf-8")
        self.assertIn('"push",', gitops)
        self.assertNotIn("--force", gitops)
        self.assertNotIn("--tags", gitops)
        self.assertIn("--delete", gitops)

    def test_push_remains_after_final_review_authorization(self) -> None:
        v2 = inspect.getsource(Orchestrator._execute_v2)
        self.assertLess(v2.index("_push_candidate("), v2.index("_run_v2_reviewer("))
        self.assertIn("_complete_candidate_publication", v2)
        complete = inspect.getsource(Orchestrator._complete_candidate_publication)
        self.assertIn("publish_fast_forward_base", complete)
        self.assertNotIn("commit_candidate_tree", complete)


if __name__ == "__main__":
    unittest.main()
