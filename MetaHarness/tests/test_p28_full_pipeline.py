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
from metaharness import orchestrator as orchestrator_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.claude.agent import (  # noqa: E402
    ClaudeCodeAgent,
    ClaudeResult,
    build_claude_environment,
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
from metaharness.llm.chat import TextLLMResult  # noqa: E402
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
        "NONE",
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
                 base_sha: str | None = None, env: dict[str, str] | None = None) -> AgentResult:
        directory = Path(artifacts_dir)
        cycle = 2 if directory.parent.parent.name == "C02" else 1
        step_id = directory.name
        root = Path(worktree)
        self.calls.append({"cycle": cycle, "step": step_id, "contract": contract,
                           "env": dict(env or {}), "dir": directory})
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


def writer(path: str, content: str) -> Callable[[Path], None]:
    return lambda root: write(root / path, content)


class FakeClaude:
    """Claude Code double: optional in-scope edit per cycle, exact artifacts."""

    def __init__(self, actions: dict[int, Callable[[Path], None]] | None = None, log: list[str] | None = None):
        self.actions = actions or {}
        self.calls: list[dict[str, Any]] = []
        self.log = log

    def run_revision(self, prompt: str, worktree: Path, *, artifacts_dir: Path,
                     profile: ModelProfile, environment: dict[str, str],
                     revision_dir: Path | None = None) -> ClaudeResult:
        target = Path(revision_dir) if revision_dir is not None else Path(artifacts_dir) / "revision"
        cycle = 2 if target.name == "C02" else 1
        self.calls.append({"cycle": cycle, "prompt": prompt, "environment": dict(environment)})
        if self.log is not None:
            self.log.append(f"claude:C0{cycle}")
        if cycle in self.actions:
            self.actions[cycle](Path(worktree))
        target.mkdir(parents=True, exist_ok=True)
        final = f"Claude C0{cycle} revision report\n"
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
                    require_approval: bool = False) -> str:
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
        )

    def load(self, **kwargs: Any) -> HarnessConfig:
        path = self.root / "p28.toml"
        path.write_text(self.config_text(**kwargs), encoding="utf-8")
        return load_config(path)

    def run_pipeline(
        self, *, luna: FakeLuna, reviews: list[str], plan: str = SINGLE_PLAN,
        repair_plan: str | None = None, claude: FakeClaude | None = None,
        run_id: str = "p28", revision: bool = True,
    ):
        config = self.load(revision=revision)
        planner = QueueClient("planner", [plan] + ([repair_plan] if repair_plan else []), self.events)
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
            result = orchestrator.run_text(SPEC, run_id=run_id)
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

    def assert_published_once(self, result: Any, pushed: Any, run_id: str = "p28") -> None:
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(pushed.call_count, 1)
        worktree = self.worktree(run_id)
        self.assertEqual(git(worktree, "rev-list", "--count", f"{self.base_sha}..HEAD"), "1")
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
        # The single push is after the final reviewer PASS.
        last_pass = max(i for i, event in enumerate(self.events) if event == "reviewer:VERDICT: PASS")
        self.assertEqual(self.events.index("push"), len(self.events) - 1)
        self.assertGreater(self.events.index("push"), last_pass)


class FullPipelineTests(P28Harness):
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
                                       "reviewer:VERDICT: PASS", "push"])
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
        self.assertEqual(result.state["failure"]["reason"], "REVIEWER_OUTPUT_INVALID")
        self.assertIn("PASS is forbidden", result.state["failure"]["detail"])
        self.assert_no_commit_no_push(result, pushed)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (1, 1))
        self.assertFalse((result.run_dir / "repair" / "C02").exists())
        self.assertIsNone(result.state.get("approved_tree_sha"))
        # The gate payload the reviewer saw carried the same red flag.
        self.assertIn('"deterministic_passed": false', reviewer.prompts[0])

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
        self.assert_published_once(result, pushed)
        self.assertEqual(self.events, [
            "planner:META PLAN v2", "claude:C01", "reviewer:VERDICT: REVISE",
            "planner:META PLAN v2", "claude:C02", "reviewer:VERDICT: PASS", "push",
        ])
        self.assertEqual([(call["cycle"], call["step"]) for call in luna.calls], [(1, "S01"), (2, "S01")])
        self.assertEqual([call["cycle"] for call in claude.calls], [1, 2])
        self.assertEqual(result.state["cycle"], 2)
        self.assertEqual(result.state["review_iterations"], 2)
        self.assertEqual(git(self.worktree(), "show", "HEAD:src/a.py"), "A = 4")
        # Reviewer #2 evidence: both plans, all worker reports, both revisions.
        second = reviewer.prompts[1]
        original = (result.run_dir / "planner.raw.md").read_text()
        repair = (result.run_dir / "repair" / "C02" / "planner.raw.md").read_text()
        self.assertIn(f"ORIGINAL APPROVED PLAN\n{original}\n\nREPAIR PLAN C02\n{repair}", second)
        for marker in ("C01 LUNA REPORTS", "C02 LUNA REPAIR REPORTS", "C01 S01 report",
                       "C02 S01 report", "C01 CLAUDE REVISION", "C02 CLAUDE REVISION",
                       "Claude C01 revision report", "Claude C02 revision report",
                       '"deterministic_passed": true'):
            self.assertIn(marker, second)
        usage = result.state["usage"]
        self.assertEqual(usage["luna_c01"]["input_tokens"], 101)
        self.assertEqual(usage["luna_c02"]["input_tokens"], 201)
        self.assertEqual(usage["grand_total"]["input_tokens"], 10 + 101 + 7 + 10 + 10 + 201 + 14 + 10)

    def test_e_c02_red_gate_with_reviewer_pass_is_invalid(self) -> None:
        _luna, result, planner, reviewer, _claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = BUG\n"), [REVISE_IMPLEMENTATION, PASS],
        )
        self.assertEqual(result.state["failure"]["reason"], "REVIEWER_OUTPUT_INVALID")
        self.assert_no_commit_no_push(result, pushed)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (2, 2))
        self.assertFalse((result.run_dir / "repair" / "C03").exists())
        self.assertFalse(result.state["deterministic_gate"]["passed"])

    def test_f_c02_revise_exhausts_the_loop(self) -> None:
        _luna, result, planner, reviewer, _claude, pushed = self._c01_then_repair(
            writer("src/a.py", "A = 4\n"), [REVISE_IMPLEMENTATION, REVISE_IMPLEMENTATION],
        )
        self.assertEqual(result.state["failure"]["reason"], "REVIEW_LOOP_EXHAUSTED")
        self.assert_no_commit_no_push(result, pushed)
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (2, 2))
        self.assertFalse((result.run_dir / "repair" / "C03").exists())

    def test_g_codex_failures_have_c01_c02_parity(self) -> None:
        expected = {
            "auth": "CODEX_AUTH_FAILURE",
            "commit": "AGENT_COMMITTED",
            "timeout": "AGENT_TIMEOUT",
            "unexpected": "STEP_WRITE_SET_VIOLATION",
            "nochange": "AGENT_NO_CHANGE",
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
                    self.assertEqual(pushed.call_count, 0)
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
            if args == ["--help"]:
                flags = ["--print", "--verbose", "--output-format", "--model", "--effort", "--permission-mode", "--mcp-config", "--strict-mcp-config"]
                if mode == "missing_capability":
                    flags.remove("--effort")
                print(" ".join(flags))
                sys.exit(3 if mode == "help_nonzero" else 0)
            if args == ["auth", "--help"]:
                print("Commands:\\n  status  Show authentication status"); sys.exit(0)
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
                         ["cache", "empty-mcp.json", "home", "tmp"])
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
                         ["cache", "empty-mcp.json", "home", "tmp"])


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
        executor = inspect.getsource(Orchestrator._execute_codex_step)
        self.assertEqual(module.count(".run_step("), 1)
        self.assertIn(".run_step(", executor)
        for name in ("_execute_v2", "_execute_v2_repair_cycle"):
            source = inspect.getsource(getattr(Orchestrator, name))
            self.assertEqual(source.count("self._execute_codex_step("), 1, name)
            for forbidden in ("run_step", "build_agent_environment(", "stage_all(", "classify_codex_failure"):
                if forbidden == "stage_all(" and name == "_execute_v2":
                    continue  # the base tree is staged once before any step
                self.assertNotIn(forbidden, source, (name, forbidden))
        self.assertEqual(
            [field.name for field in dataclasses.fields(orchestrator_module.StepExecutionOutcome)],
            ["step_id", "profile_id", "tree_before", "tree_after", "changed_paths", "usage", "final_report"],
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
        self.assertNotIn("--delete", gitops)

    def test_push_remains_after_final_review_authorization(self) -> None:
        v2 = inspect.getsource(Orchestrator._execute_v2)
        c02_gate = v2.index("repair_plan, review, evidence, info.worktree, base_sha, branch_ref")
        c01_gate = v2.index("self._authorize_v2_commit(plan, review, evidence")
        self.assertLess(v2.index("REVIEW_LOOP_EXHAUSTED"), c02_gate)
        self.assertLess(c02_gate, v2.index("commit_reviewed_tree"))
        self.assertLess(c02_gate, v2.index("cycle=2,"))
        self.assertLess(c01_gate, v2.rindex("return self._complete_commit("))
        self.assertEqual(v2.count("self._complete_commit("), 2)
        complete = inspect.getsource(Orchestrator._complete_commit)
        self.assertLess(complete.index("COMMIT_TREE_MISMATCH"), complete.index("push_run_branch("))
        self.assertLess(complete.index("validate_run_branch("), complete.index("push_run_branch("))
        self.assertEqual(complete.count("push_run_branch("), 1)


if __name__ == "__main__":
    unittest.main()
