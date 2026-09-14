"""P29: Claude CLI fix, durable resume, live UI, STAGED policy, main publication.

Every pipeline test uses a real temporary Git repository, a real local bare
remote, the real orchestrator state machine and the real Git primitives.
Planner, Luna, Claude and reviewers are in-process fakes: no network, no LLM.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import cli  # noqa: E402
from metaharness import orchestrator as orchestrator_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.agent.diagnostics import token_diagnostics  # noqa: E402
from metaharness.approval import PlanIdentity  # noqa: E402
from metaharness.claude.agent import ClaudeCodeAgent, ClaudeResult, build_claude_environment  # noqa: E402
from metaharness.claude.runtime import prepare_claude_home  # noqa: E402
from metaharness.config import ConfigError, load_config  # noqa: E402
from metaharness.llm.chat import LLMConversationHandle, LLMHTTPError, TextLLMResult  # noqa: E402
from metaharness.models import (  # noqa: E402
    ExecutionMode,
    ExecutionRole,
    ModelProfile,
    PlanningConfig,
    ProfileDriver,
    RunStatus,
    SelectionMode,
)
from metaharness.orchestrator import Orchestrator  # noqa: E402
from metaharness.planning_v2 import (  # noqa: E402
    REQUIRE_STAGED_POLICY_TEXT,
    PlannerV2,
    V2PlanParseError,
    build_planner_prompt_v2,
    parse_task_plan_v2,
    validate_execution_mode_policy,
)
from metaharness.resume import (  # noqa: E402
    CHECKPOINT_NAME,
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeNotAllowedError,
    ResumePhase,
    mark_checkpoint_completed,
    read_checkpoint,
    read_checkpoint_record,
    resume_info,
    write_checkpoint,
)
from metaharness.state import RunStateStore  # noqa: E402
from metaharness.web.api import approve_run, get_run, live_status  # noqa: E402
from metaharness.web.pages import render_new_run, render_run  # noqa: E402
from metaharness.web.server import create_server  # noqa: E402
from tests.test_p28_full_pipeline import (  # noqa: E402
    PASS,
    REPAIR_PLAN,
    REVISE_IMPLEMENTATION,
    SINGLE_PLAN,
    SPEC,
    STAGED_PLAN,
    FakeClaude,
    FakeLuna,
    P28Harness,
    QueueClient,
    git,
    plan_text,
    step_block,
    write,
    writer,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_JS = ROOT / "src" / "metaharness" / "web" / "static" / "run.js"
VERBOSE_ERROR = "Error: When using --print, --output-format=stream-json requires --verbose"
BLOCKED_PLAN = "\n".join([
    "META PLAN v2", "", "STATUS: BLOCKED", "TITLE: Blocked", "",
    "OBJECTIVE", "Nothing can be decided.", "", "BLOCKERS", "The SPEC is ambiguous.", "",
    "END META PLAN", "",
])


class FailingClaude(FakeClaude):
    """Claude double that fails like the real P28 run (optionally after edits)."""

    def __init__(self, *, message: str = VERBOSE_ERROR,
                 modify: Callable[[Path], None] | None = None, log: list[str] | None = None):
        super().__init__(log=log)
        self.message = message
        self.modify = modify

    def run_revision(self, prompt: str, worktree: Path, *, artifacts_dir: Path,
                     profile: ModelProfile, environment: dict[str, str],
                     revision_dir: Path | None = None) -> ClaudeResult:
        target = Path(revision_dir) if revision_dir is not None else Path(artifacts_dir) / "revision"
        self.calls.append({"cycle": 2 if target.name == "C02" else 1, "prompt": prompt})
        if self.modify is not None:
            self.modify(Path(worktree))
        target.mkdir(parents=True, exist_ok=True)
        (target / "agent.stderr.log").write_text(self.message + "\n", encoding="utf-8")
        (target / "agent.events.jsonl").write_text("", encoding="utf-8")
        return ClaudeResult(1, False, "", {}, self.message)


class PartialFailingLuna(FakeLuna):
    """Codex double that edits the tree and then exits non-zero."""

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, *,
                 base_sha: str | None = None, env: dict[str, str] | None = None) -> AgentResult:
        directory = Path(artifacts_dir)
        self.calls.append({"cycle": 1, "step": directory.name, "dir": directory})
        write(Path(worktree) / "src/a.py", "A = PARTIAL\n")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.events.jsonl").write_text("", encoding="utf-8")
        return AgentResult(exit_code=1, timed_out=False, final_message="", usage={}, stderr_tail="boom")


class TransportFailingClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> TextLLMResult:
        self.prompts.append(prompt)
        raise LLMHTTPError("HTTP 503 from reviewer bridge")


class P29Harness(P28Harness):
    def config_path(self, *, mode: str = "run-branch", publish: bool = True,
                    policy: str | None = None, revision: bool = True) -> Path:
        text = self.config_text(require_approval=True, publish=publish, revision=revision)
        text = text.replace('mode = "run-branch"', f'mode = "{mode}"')
        if policy is not None:
            text = text.replace('protocol = "v2"', f'protocol = "v2"\nexecution_mode_policy = "{policy}"')
        path = self.root / "p29.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def make_config(self, **kwargs: Any):
        return load_config(self.config_path(**kwargs))

    def orchestrator(self, config: Any, *, plans: list[str] | None = None,
                     reviews: list[str] | None = None, luna: FakeLuna | None = None,
                     claude: FakeClaude | None = None, reviewer: Any = None,
                     planner: Any = None) -> tuple[Orchestrator, Any, Any, FakeLuna, FakeClaude]:
        planner = planner or QueueClient("planner", list(plans or []), self.events)
        reviewer = reviewer or QueueClient("reviewer", list(reviews or []), self.events)
        luna = luna or FakeLuna({})
        claude = claude or FakeClaude(log=self.events)
        orchestrator = Orchestrator(config, planner_client=planner, reviewer_client=reviewer,
                                    agent=luna, reviser=claude)
        return orchestrator, planner, reviewer, luna, claude

    def run_approved(self, config: Any, orchestrator: Orchestrator, run_id: str,
                     steps: tuple[str, ...] = ("S01",)) -> Any:
        holder: dict[str, Any] = {}
        thread = threading.Thread(
            target=lambda: holder.setdefault("result", orchestrator.run_text(SPEC, run_id=run_id)))
        thread.start()
        state_path = config.runs_root / run_id / "state.json"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if state_path.exists():
                status = json.loads(state_path.read_text())["status"]
                if status == "awaiting_plan_approval":
                    break
                if status in {"failed", "blocked"}:
                    thread.join(30)
                    return holder["result"]
            time.sleep(0.01)
        approve_run(config.runs_root, run_id, "APPROVE", config=config,
                    reviewer_profile="reviewer", step_profiles={step: "luna" for step in steps})
        thread.join(timeout=60)
        self.assertFalse(thread.is_alive())
        return holder["result"]

    def count_pushes(self):
        real = orchestrator_module.push_run_branch

        def push(*args: Any, **kwargs: Any) -> Any:
            self.events.append("push")
            return real(*args, **kwargs)

        return mock.patch.object(orchestrator_module, "push_run_branch", side_effect=push)

    def checks_ran(self) -> int:
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def commits_on_run_branch(self, run_id: str) -> str:
        return git(self.worktree(run_id), "rev-list", "--count", f"{self.base_sha}..HEAD")


class ClaudeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        git(self.repo, "config", "user.email", "c@example.invalid")
        git(self.repo, "config", "user.name", "c")
        write(self.repo / "README.md", "base\n")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-qm", "base")
        self.fake = self.root / "claude"
        # Reproduces the installed CLI: --print + stream-json without --verbose
        # exits 1 with the exact error seen in run 20260914T124017Z-7b74467062.
        self.fake.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""
            import json, sys
            args = sys.argv[1:]
            if "--print" in args and "stream-json" in args and "--verbose" not in args:
                sys.stderr.write({VERBOSE_ERROR!r} + "\\n")
                sys.exit(1)
            sys.stdin.read()
            print(json.dumps({{"type": "result", "subtype": "success", "result": "ok"}}))
        """), encoding="utf-8")
        self.fake.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_verbose_is_authoritative_and_placed_after_print(self) -> None:
        profile = ModelProfile(
            id="claude", display_name="Claude", roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE, model="opus", selection_mode=SelectionMode.CLI,
            effort="medium", permission_mode="acceptEdits", timeout_seconds=10, retries=0,
        )
        argv = ClaudeCodeAgent(executable="claude").build_argv(self.repo, profile=profile, claude_home=self.root / "home")
        self.assertEqual(argv[:5], ["claude", "--print", "--verbose", "--output-format", "stream-json"])
        self.assertEqual(argv.count("--verbose"), 1)
        self.assertEqual(argv[-3:], ["--strict-mcp-config", "--mcp-config", str((self.root / "home" / "empty-mcp.json").resolve())])

    def test_fake_cli_rejects_old_argv_and_accepts_metaharness_argv(self) -> None:
        old = subprocess.run(
            [str(self.fake), "--print", "--output-format", "stream-json"],
            input="x", capture_output=True, text=True,
        )
        self.assertEqual(old.returncode, 1)
        self.assertEqual(old.stderr.strip(), VERBOSE_ERROR)
        from metaharness.models import ClaudeRuntimeConfig, CodexRuntimeConfig, HarnessConfig
        from tests.test_p28_full_pipeline import AgentConfig, ContextConfig, LLMEndpointConfig
        config = HarnessConfig(
            repo=self.repo, base_ref="HEAD", runs_root=self.root / "runs",
            worktrees_root=self.root / "worktrees", require_clean_base=True,
            planner=LLMEndpointConfig("https://p.invalid", "/v1", "p"),
            reviewer=LLMEndpointConfig("https://r.invalid", "/v1", "r"),
            context=ContextConfig(always_files=()), agent=AgentConfig(), checks=(),
            allow_no_required_checks=True,
            codex_runtime=CodexRuntimeConfig(self.root / "codex-home"),
            claude_runtime=ClaudeRuntimeConfig(self.root / "claude-home"),
        )
        home = prepare_claude_home(config)
        profile = ModelProfile(
            id="claude", display_name="Claude", roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE, model="opus", selection_mode=SelectionMode.CLI,
            effort="medium", permission_mode="acceptEdits", timeout_seconds=10, retries=0,
        )
        result = ClaudeCodeAgent(executable=str(self.fake)).run_revision(
            "revise", self.repo, artifacts_dir=self.root / "run", profile=profile,
            environment=build_claude_environment({"PATH": "/usr/bin:/bin"}, claude_home=home),
        )
        self.assertEqual((result.exit_code, result.final_message), (0, "ok"))

    def test_doctor_uses_parser_probe_not_help_text(self) -> None:
        help_fake = self.root / "claude-help"
        help_fake.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "print('Usage: claude')\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        help_fake.chmod(0o755)
        ok, _detail = cli._probe_claude_capabilities(
            str(help_fake), {"PATH": "/usr/bin:/bin"}, self.root
        )
        self.assertTrue(ok)

        help_fake.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.exit(2 if '--no-chrome' in sys.argv else 0)\n",
            encoding="utf-8",
        )
        ok, _detail = cli._probe_claude_capabilities(
            str(help_fake), {"PATH": "/usr/bin:/bin"}, self.root
        )
        self.assertFalse(ok)


class ResumeClaudeTests(P29Harness):
    def _failed_claude_run(self, run_id: str = "claude-bug", **claude_kwargs: Any):
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        broken = FailingClaude(**claude_kwargs)
        orchestrator, planner, reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna, claude=broken,
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, run_id)
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "CLAUDE_FAILED")
        self.assertEqual(pushed.call_count, 0)
        return config, result, planner, reviewer, luna, broken

    def test_real_claude_bug_resumes_at_claude_after_restart(self) -> None:
        config, result, planner, reviewer, luna, broken = self._failed_claude_run()
        run_dir = result.run_dir
        self.assertEqual((run_dir / "revision" / "agent.stderr.log").read_text().strip(), VERBOSE_ERROR)
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.CLAUDE_C01)
        s01 = json.loads((run_dir / "steps/S01/step.json").read_text())
        self.assertEqual(checkpoint.expected_tree_sha, s01["tree_after"])
        self.assertEqual(checkpoint.expected_head_sha, self.base_sha)
        info = resume_info(run_dir, result.state)
        self.assertEqual((info.resumable, info.phase, info.label),
                         (True, "claude_c01", "Reprendre à partir de Claude"))
        page = render_run(get_run(self.runs, "claude-bug", config=config), "tok", config=config)
        self.assertIn("Reprendre à partir de Claude", page)
        self.assertIn('action="/runs/claude-bug/resume"', page)
        self.assertIn("Claude invocation failed", page)
        self.assertIn("↻", page)

    def test_resume_after_restart_never_replays_a_successful_phase(self) -> None:
        config, result, planner, reviewer, luna, broken = self._failed_claude_run()
        checks_before = self.checks_ran()
        fresh_config = load_config(self.config_path())
        fixed = FakeClaude(log=self.events)
        no_planner = QueueClient("planner", [], self.events)
        second = Orchestrator(fresh_config, planner_client=no_planner, reviewer_client=reviewer,
                              agent=luna, reviser=fixed)
        with self.count_pushes() as pushed:
            resumed = second.resume("claude-bug")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(len(planner.prompts) + len(no_planner.prompts), 1)   # planner
        self.assertEqual(len(luna.calls), 1)                                    # Luna S01
        self.assertEqual(len(broken.calls) + len(fixed.calls), 2)               # Claude
        self.assertEqual(len(reviewer.prompts), 1)                              # reviewer
        self.assertEqual(self.commits_on_run_branch("claude-bug"), "1")         # commit
        self.assertEqual(pushed.call_count, 1)                                  # publish
        # Pre-revision checks were durable for this exact tree: only the
        # final checks ran on resume.
        self.assertEqual(self.checks_ran(), checks_before + 1)
        run_dir = resumed.run_dir
        self.assertIsNone(read_checkpoint(run_dir))
        self.assertEqual(read_checkpoint_record(run_dir)[1], "completed")
        self.assertEqual(resumed.state["resume"]["previous_failure"]["reason"], "CLAUDE_FAILED")
        self.assertEqual(resumed.state["resume"]["attempts"], 1)
        archived = run_dir / "revision" / "attempts" / "01" / "agent.stderr.log"
        self.assertEqual(archived.read_text().strip(), VERBOSE_ERROR)
        self.assertEqual((run_dir / "revision" / "agent.final.md").read_text(), "Claude C01 revision report\n")
        with self.assertRaises(ResumeNotAllowedError):
            second.resume("claude-bug")

    def test_historical_run_without_checkpoint_is_inferred(self) -> None:
        config, result, planner, reviewer, luna, _broken = self._failed_claude_run("legacy")
        (result.run_dir / CHECKPOINT_NAME).unlink()
        info = resume_info(result.run_dir, result.state)
        self.assertEqual((info.resumable, info.phase), (True, "claude_c01"))
        second, _p, _r, _l, fixed = self.orchestrator(config, reviewer=reviewer, luna=luna)
        with self.count_pushes():
            resumed = second.resume("legacy")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(len(luna.calls), 1)
        self.assertEqual(len(fixed.calls), 1)

    def test_tampered_worktree_fails_integrity_with_zero_llm_calls(self) -> None:
        config, result, _planner, _reviewer, _luna, _broken = self._failed_claude_run("tamper")
        write(self.worktree("tamper") / "src/a.py", "A = 99\n")
        second, planner, reviewer, luna, claude = self.orchestrator(config)
        refused = second.resume("tamper")
        self.assertEqual(refused.status, RunStatus.FAILED)
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual((planner.prompts, reviewer.prompts, luna.calls, claude.calls), ([], [], [], []))
        self.assertEqual(refused.state["resume"]["previous_failure"]["reason"], "CLAUDE_FAILED")
        self.assertEqual((self.worktree("tamper") / "src/a.py").read_text(), "A = 99\n")
        self.assertFalse(resume_info(refused.run_dir, refused.state).resumable)
        page = render_run(get_run(self.runs, "tamper", config=config), "tok", config=config)
        self.assertNotIn("Reprendre à partir de Claude", page)
        self.assertNotIn('/resume"', page)
        with self.assertRaises(ResumeNotAllowedError):
            second.resume("tamper")

    def test_partial_claude_edit_in_scope_is_restored_exactly(self) -> None:
        config, result, _planner, reviewer, luna, _broken = self._failed_claude_run(
            "partial", modify=writer("src/a.py", "A = HALF\n"))
        run_dir = result.run_dir
        self.assertTrue((run_dir / "revision" / "tree_after_failure.txt").exists())
        second, _p, _r, _l, fixed = self.orchestrator(config, reviewer=reviewer, luna=luna)
        with self.count_pushes():
            resumed = second.resume("partial")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(resumed.state["resume"]["restored_paths"], ["src/a.py"])
        self.assertEqual(git(self.worktree("partial"), "show", "HEAD:src/a.py"), "A = 2")
        self.assertEqual(len(fixed.calls), 1)

    def test_partial_claude_edit_outside_scope_requires_operator(self) -> None:
        config, result, *_rest = self._failed_claude_run(
            "outside", modify=writer("src/b.py", "B = CLAUDE\n"))
        second, planner, reviewer, luna, claude = self.orchestrator(config)
        refused = second.resume("outside")
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_REQUIRES_OPERATOR")
        self.assertEqual((planner.prompts, reviewer.prompts, luna.calls, claude.calls), ([], [], [], []))
        self.assertEqual((self.worktree("outside") / "src/b.py").read_text(), "B = CLAUDE\n")

    def test_checkpoints_always_name_the_next_operation(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (1, "S02"): writer("src/b.py", "B = 2\n"),
                         (1, "S03"): writer("src/c.py", "C = 3\n")})
        orchestrator, *_rest = self.orchestrator(config, plans=[STAGED_PLAN], reviews=[PASS], luna=luna)
        recorded: list[ResumeCheckpoint] = []
        real = orchestrator_module.write_checkpoint

        def record(run_dir: Any, checkpoint: ResumeCheckpoint) -> None:
            recorded.append(checkpoint)
            real(run_dir, checkpoint)

        with mock.patch.object(orchestrator_module, "write_checkpoint", side_effect=record), self.count_pushes():
            result = self.run_approved(config, orchestrator, "sequence", ("S01", "S02", "S03"))
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        phases = [(item.phase.value, item.step_id) for item in recorded]
        self.assertEqual(phases, [
            ("initial_step", "S01"), ("initial_step", "S01"),            # approval, setup
            ("initial_step", "S02"), ("initial_step", "S03"),            # each Luna step
            ("claude_c01", None), ("claude_c01", None),                  # S03 done, pre-checks done
            ("reviewer_c01", None), ("reviewer_c01", None),              # Claude done, final checks done
            ("publish", None),                                           # exact commit done
        ])
        run_dir = result.run_dir
        s03 = json.loads((run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(recorded[4].expected_tree_sha, s03["tree_after"])
        revision = json.loads((run_dir / "revision/report.json").read_text())
        self.assertEqual(recorded[6].expected_tree_sha, revision["tree_after"])
        self.assertEqual(recorded[-1].expected_head_sha, result.state["commit_sha"])
        self.assertEqual(recorded[-1].expected_tree_sha, result.state["approved_tree_sha"])
        self.assertEqual(read_checkpoint_record(run_dir)[1], "completed")


class ResumeCodexReviewerPublishTests(P29Harness):
    def test_codex_failure_retries_only_that_step(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (1, "S02"): "exit1"})
        orchestrator, planner, reviewer, *_rest = self.orchestrator(config, plans=[STAGED_PLAN], reviews=[PASS], luna=luna)
        result = self.run_approved(config, orchestrator, "codex", ("S01", "S02", "S03"))
        self.assertEqual(result.state["failure"]["reason"], "AGENT_FAILED")
        info = resume_info(result.run_dir, result.state)
        self.assertEqual((info.resumable, info.label), (True, "Retry S02"))
        retry = FakeLuna({(1, "S02"): writer("src/b.py", "B = 2\n"), (1, "S03"): writer("src/c.py", "C = 3\n")})
        second, no_planner, _r, _l, claude = self.orchestrator(config, reviewer=reviewer, luna=retry)
        with self.count_pushes():
            resumed = second.resume("codex")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02"])
        self.assertEqual([call["step"] for call in retry.calls], ["S02", "S03"])
        self.assertEqual((len(planner.prompts), len(no_planner.prompts)), (1, 0))
        self.assertEqual(json.loads((resumed.run_dir / "steps/S02/attempts/01/step.json").read_text())["reason"], "AGENT_FAILED")

    def test_partial_codex_failure_requires_operator(self) -> None:
        config = self.make_config()
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], luna=PartialFailingLuna({}))
        result = self.run_approved(config, orchestrator, "partial-codex")
        self.assertEqual(result.state["failure"]["reason"], "AGENT_FAILED")
        second, planner, reviewer, luna, claude = self.orchestrator(config)
        refused = second.resume("partial-codex")
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_REQUIRES_OPERATOR")
        self.assertEqual((planner.prompts, reviewer.prompts, luna.calls, claude.calls), ([], [], [], []))

    def test_non_resumable_failures_are_refused_without_state_change(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): "unexpected"})
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], luna=luna)
        result = self.run_approved(config, orchestrator, "violation")
        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        before = (result.run_dir / "state.json").read_text()
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(config)[0].resume("violation")
        self.assertEqual((result.run_dir / "state.json").read_text(), before)

    def test_reviewer_transport_failure_resumes_reviewer_only(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        failing = TransportFailingClient()
        orchestrator, _planner, _r, _l, claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], luna=luna, reviewer=failing)
        result = self.run_approved(config, orchestrator, "transport")
        self.assertEqual(result.state["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        self.assertFalse((result.run_dir / "review.json").exists())
        checks_before = self.checks_ran()
        info = resume_info(result.run_dir, result.state)
        self.assertEqual((info.phase, info.label), ("reviewer_c01", "Retry reviewer #1"))
        second, planner2, reviewer2, luna2, claude2 = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushed:
            resumed = second.resume("transport")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((len(failing.prompts), len(reviewer2.prompts)), (1, 1))
        self.assertEqual((planner2.prompts, luna2.calls, claude2.calls), ([], [], []))
        self.assertEqual(len(claude.calls), 1)
        self.assertEqual(self.checks_ran(), checks_before)
        self.assertEqual(pushed.call_count, 1)


class FastForwardMainTests(P29Harness):
    def setUp(self) -> None:
        super().setUp()
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "fetch", "-q", "origin")

    def origin_main(self) -> str:
        return subprocess.run(["git", "--git-dir", str(self.bare), "rev-parse", "refs/heads/main"],
                              capture_output=True, text=True, check=True).stdout.strip()

    def test_final_pass_fast_forwards_local_and_origin_main(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna)
        checkout_file = (self.repo / "src/a.py").read_text()
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "ffmain")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        commit = result.state["commit_sha"]
        self.assertEqual(git(self.repo, "rev-parse", "refs/heads/main"), commit)
        self.assertEqual(git(self.repo, "rev-parse", f"{commit}^"), self.base_sha)
        self.assertEqual(self.origin_main(), commit)
        self.assertEqual(git(self.repo, "rev-parse", f"{commit}^{{tree}}"), result.state["approved_tree_sha"])
        # No checkout: the user's files were not touched; the run branch was
        # never pushed and remains local.
        self.assertEqual((self.repo / "src/a.py").read_text(), checkout_file)
        self.assertEqual(pushed.call_count, 0)
        self.assertEqual([line.split()[1] for line in self.remote_refs().splitlines()], ["refs/heads/main"])
        self.assertEqual(git(self.repo, "rev-parse", f"refs/heads/{result.state['branch']}"), commit)
        publish = json.loads((result.run_dir / "publish.json").read_text())
        self.assertEqual((publish["mode"], publish["target"], publish["commit_sha"]),
                         ("fast-forward-base", "main", commit))
        self.assertIn(str(self.repo.resolve()), publish["base_checked_out_in"])
        page = render_run(get_run(self.runs, "ffmain", config=config), None, config=config)
        self.assertIn("Published to origin/main", page)
        self.assertIn(commit, page)

    def test_moved_main_is_refused_without_merge_rebase_or_force(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        repo = self.repo

        class MovingReviewer(QueueClient):
            def complete(self, prompt: str) -> TextLLMResult:
                write(repo / "README.md", "moved main\n")
                git(repo, "commit", "-qam", "C on main")
                git(repo, "push", "-q", "origin", "main")
                return super().complete(prompt)

        reviewer = MovingReviewer("reviewer", [PASS], self.events)
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], luna=luna, reviewer=reviewer)
        result = self.run_approved(config, orchestrator, "moved")
        self.assertEqual(result.state["failure"]["reason"], "BASE_MOVED_SINCE_RUN")
        moved = git(self.repo, "rev-parse", "refs/heads/main")
        self.assertNotEqual(moved, self.base_sha)
        self.assertEqual(git(self.repo, "rev-parse", "main^"), self.base_sha)
        self.assertEqual(self.origin_main(), moved)
        self.assertEqual(git(self.repo, "log", "--format=%s", "-1", "main"), "C on main")
        self.assertFalse(resume_info(result.run_dir, result.state).resumable)

    def test_failed_push_is_resumed_as_publication_only(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        git(self.repo, "remote", "set-url", "origin", str(self.root / "missing.git"))
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, planner, reviewer, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna)
        result = self.run_approved(config, orchestrator, "push-retry")
        self.assertEqual(result.state["failure"]["reason"], "PUSH_FAILED")
        commit = result.state["commit_sha"]
        self.assertIn(f"local main already points to {commit}", result.state["failure"]["detail"])
        self.assertTrue(result.state["publish"]["local_base_updated"])
        self.assertEqual(git(self.repo, "rev-parse", "refs/heads/main"), commit)
        info = resume_info(result.run_dir, result.state)
        self.assertEqual((info.phase, info.label), ("publish", "Retry publish"))
        git(self.repo, "remote", "set-url", "origin", str(self.bare))
        second, planner2, reviewer2, luna2, claude2 = self.orchestrator(config)
        resumed = second.resume("push-retry")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(self.origin_main(), commit)
        self.assertEqual(resumed.state["commit_sha"], commit)
        self.assertEqual((planner2.prompts, reviewer2.prompts, luna2.calls, claude2.calls), ([], [], [], []))
        self.assertEqual(self.commits_on_run_branch("push-retry"), "1")


class UiTests(P29Harness):
    def _server(self, config: Any):
        server = create_server(config, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def request(self, server: Any, method: str, path: str, body: str | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], str]:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=30)
        sent = {"Host": f"127.0.0.1:{server.server_port}", **(headers or {})}
        if body is not None:
            sent.setdefault("Content-Type", "application/x-www-form-urlencoded")
        connection.request(method, path, body=body, headers=sent)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        result = (response.status, {k.lower(): v for k, v in response.getheaders()}, content)
        connection.close()
        return result

    def running_run(self, run_id: str = "live") -> Path:
        run_dir = self.runs / run_id
        store = RunStateStore(run_dir / "state.json")
        store.initialize(run_id, base_sha="b" * 40)
        store.update(status="implementing", planning_protocol="v2", current_step="S01",
                     steps=[{"id": "S01", "title": "one", "status": "running"}])
        step_dir = run_dir / "steps" / "S01"
        step_dir.mkdir(parents=True)
        events = [
            {"type": "item.started", "item": {"id": "c1", "type": "command_execution",
                                              "command": "bash -lc 'cat src/a.py --secret-arg'"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "working on S01"}},
        ]
        (step_dir / "agent.events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (run_dir / "diff.patch").write_text("+SECRET_DIFF_CONTENT\n")
        (run_dir / "planner.request.txt").write_text("RAW PROMPT CONTENT")
        return run_dir

    def test_run_page_has_no_refresh_and_loads_static_script(self) -> None:
        config = self.make_config()
        self.running_run()
        server = self._server(config)
        status, headers, page = self.request(server, "GET", "/runs/live")
        self.assertEqual(status, 200)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn('<script src="/static/run.js" defer></script>', page)
        self.assertNotIn("<script>", page)
        self.assertIn('data-run-id="live"', page)
        csp = headers["content-security-policy"]
        for directive in ("default-src 'none'", "script-src 'self'", "connect-src 'self'",
                          "frame-ancestors 'none'", "base-uri 'none'", "object-src 'none'"):
            self.assertIn(directive, csp)
        # Section order: status/next action, pipeline, execution, current
        # cycle, checks, token usage, plan, files, raw diagnostics.
        markers = ['class="sticky run-card"', "EXECUTION PIPELINE", "<h2>EXECUTION</h2>",
                   "CURRENT CYCLE", "<h2>CHECKS</h2>", "TOKEN USAGE", "<h2>PLAN</h2>",
                   "DIFF / FILES", "RAW ARTIFACTS / DIAGNOSTICS"]
        positions = [page.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        status, headers, script = self.request(server, "GET", "/static/run.js")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/javascript; charset=utf-8")
        self.assertEqual(script, RUN_JS.read_text(encoding="utf-8"))

    def test_run_js_polls_live_stops_when_terminal_and_never_rewrites_html(self) -> None:
        script = RUN_JS.read_text(encoding="utf-8")
        self.assertIn('"/live"', script)
        self.assertIn("setInterval(poll, POLL_MS)", script)
        self.assertIn("clearInterval", script)
        self.assertIn("POLL_MS = 2000", script)
        for status in ("committed", "published", "failed", "blocked", "plan_rejected", "interrupted"):
            self.assertIn(f"{status}: true", script)
        for forbidden in ("innerHTML", "outerHTML", "eval(", "new Function", "Function(",
                          "document.write", "insertAdjacentHTML", ".open =", "removeAttribute", "{{"):
            self.assertNotIn(forbidden, script)
        for used in ("textContent", "classList", "hidden"):
            self.assertIn(used, script)

    def test_live_endpoint_is_small_and_leaks_nothing(self) -> None:
        config = self.make_config()
        self.running_run()
        payload = live_status(self.runs, "live", config)
        for key in ("status", "updated_at", "cycle", "phase", "current_step", "failure",
                    "resumable", "token_totals", "progress_events"):
            self.assertIn(key, payload)
        self.assertEqual((payload["status"], payload["current_step"], payload["resumable"]),
                         ("implementing", "S01", False))
        self.assertIn("message: working on S01", payload["progress_events"])
        serialized = json.dumps(payload)
        for leaked in ("--secret-arg", "SECRET_DIFF_CONTENT", "RAW PROMPT CONTENT", "diff", "prompt"):
            self.assertNotIn(leaked, serialized)
        self.assertLess(len(serialized), 8 * 1024)
        server = self._server(config)
        status, _headers, body = self.request(server, "GET", "/api/runs/live/live")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "implementing")

    def test_resume_post_is_protected_and_refuses_non_resumable_runs(self) -> None:
        config = self.make_config()
        self.running_run()
        server = self._server(config)
        body = f"_token={server.token}"
        self.assertEqual(self.request(server, "POST", "/runs/live/resume", "_token=wrong")[0], 403)
        self.assertEqual(self.request(server, "POST", "/runs/live/resume", body,
                                      {"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.request(server, "POST", "/runs/live/resume", body,
                                      {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.request(server, "POST", "/runs/live/resume", body)[0], 409)

    def test_web_resume_action_resumes_the_same_run(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, _planner, reviewer, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna, claude=FailingClaude())
        result = self.run_approved(config, orchestrator, "web-resume")
        self.assertEqual(result.state["failure"]["reason"], "CLAUDE_FAILED")
        server = self._server(config)
        server.run_manager._orchestrator_factory = lambda cfg: Orchestrator(
            cfg, planner_client=QueueClient("planner", [], []), reviewer_client=reviewer,
            agent=luna, reviser=FakeClaude())
        status, _headers, page = self.request(server, "GET", "/runs/web-resume")
        self.assertIn("Reprendre à partir de Claude", page)
        self.assertIn(server.token, page)
        with self.count_pushes():
            status, headers, _body = self.request(
                server, "POST", "/runs/web-resume/resume", f"_token={server.token}",
                {"Origin": "null"})
            self.assertEqual((status, headers["location"]), (303, "/runs/web-resume"))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = json.loads((result.run_dir / "state.json").read_text())
                if state["status"] in {"published", "failed"}:
                    break
                time.sleep(0.05)
        self.assertEqual(state["status"], "published", state.get("failure"))
        self.assertEqual(state["run_id"], "web-resume")

    def test_token_diagnostics_severe_usage_is_flagged_not_failed(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna)
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "tokens")
        self.assertEqual(result.status, RunStatus.PUBLISHED)
        diagnostics = json.loads((result.run_dir / "steps/S01/token_diagnostics.json").read_text())
        self.assertEqual(sorted(diagnostics), sorted([
            "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens",
            "event_count", "tool_call_count", "files_read_observed", "commands_observed"]))
        self.assertEqual((diagnostics["input_tokens"], diagnostics["event_count"]), (101, 2))
        step_path = result.run_dir / "steps/S01/step.json"
        record = json.loads(step_path.read_text())
        record["usage"].update({"input_tokens": 401702, "cached_input_tokens": 344576, "output_tokens": 8892})
        step_path.write_text(json.dumps(record))
        page = render_run(get_run(self.runs, "tokens", config=config), None, config=config)
        self.assertIn("Luna S01 · 401k input · 344k cached · <strong>SEVERE CONTEXT USAGE</strong>", page)
        self.assertIn('href="#diag-c1-S01">Voir diagnostic</a>', page)
        self.assertIn('id="diag-c1-S01"', page)
        self.assertEqual(result.state["status"], "published")

    def test_token_diagnostics_are_bounded_and_argument_free(self) -> None:
        path = self.root / "events.jsonl"
        events = [{"type": "item.started", "item": {"id": f"c{i}", "type": "command_execution",
                                                    "command": f"bash -lc 'tool{i} --secret-arg x'"}}
                  for i in range(150)]
        events.append({"type": "item.completed", "item": {"id": "c0", "type": "command_execution",
                                                          "command": "bash -lc 'tool0 --secret-arg x'"}})
        events.append({"msg": {"type": "exec_command_begin", "command": ["sed", "-n", "1p", "src/a.py"],
                               "parsed_cmd": [{"type": "read", "cmd": "sed", "path": "src/a.py"},
                                              {"type": "read", "path": "/etc/passwd"},
                                              {"type": "read", "path": "../escape.py"}]}})
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        record = token_diagnostics(path, {"input_tokens": 5, "cached_input_tokens": 2}, worktree=self.root)
        self.assertEqual(len(record["commands_observed"]), 100)
        self.assertEqual(record["commands_observed"][:2], ["tool0", "tool1"])
        self.assertEqual(record["files_read_observed"], ["src/a.py"])
        self.assertEqual(record["tool_call_count"], 151)
        self.assertEqual(record["event_count"], 152)
        self.assertNotIn("--secret-arg", json.dumps(record))


class ExecutionPolicyTests(P29Harness):
    def ids(self) -> dict[str, frozenset[str]]:
        return {"implementer_ids": frozenset({"luna"}), "reviewer_ids": frozenset({"reviewer"})}

    def test_require_staged_rejects_ready_single_only(self) -> None:
        planning = PlanningConfig(protocol="v2", execution_mode_policy="require-staged")
        single = parse_task_plan_v2(SINGLE_PLAN, **self.ids())
        with self.assertRaisesRegex(V2PlanParseError, "execution policy requires STAGED"):
            validate_execution_mode_policy(single, planning)
        validate_execution_mode_policy(parse_task_plan_v2(STAGED_PLAN, **self.ids()), planning)
        validate_execution_mode_policy(parse_task_plan_v2(BLOCKED_PLAN, **self.ids()), planning)
        validate_execution_mode_policy(single, PlanningConfig(protocol="v2"))
        with self.assertRaises(ValueError):
            PlanningConfig(protocol="v2", execution_mode_policy="staged")

    def test_prompt_carries_the_policy_only_when_required(self) -> None:
        required = build_planner_prompt_v2("SPEC", "CTX", execution_mode_policy="require-staged")
        self.assertIn("This run REQUIRES STAGED execution.", required)
        self.assertIn("You must return between 2 and 6 coherent implementation steps.", required)
        self.assertLess(required.index(REQUIRE_STAGED_POLICY_TEXT), required.index("The answer must use exactly this protocol."))
        self.assertNotIn("REQUIRES STAGED", build_planner_prompt_v2("SPEC", "CTX"))

    def test_planner_single_answer_fails_before_any_bundle(self) -> None:
        planner = PlannerV2(
            QueueClient("planner", [SINGLE_PLAN], []), **self.ids(),
            planning=PlanningConfig(protocol="v2", execution_mode_policy="require-staged"),
        )
        with self.assertRaisesRegex(V2PlanParseError, "execution policy requires STAGED"):
            planner.plan("SPEC", "CTX", artifacts_dir=self.root / "plan")
        self.assertFalse((self.root / "plan" / "implementation_bundle.json").exists())
        self.assertIn("REQUIRES STAGED", (self.root / "plan" / "planner.request.txt").read_text())

    def test_pipeline_with_require_staged(self) -> None:
        config = self.make_config(policy="require-staged")
        self.assertEqual(config.planning.execution_mode_policy, "require-staged")
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN])
        result = self.run_approved(config, orchestrator, "single-refused")
        self.assertEqual(result.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        self.assertIsNone(result.state.get("worktree"))
        page = render_new_run(config, "tok")
        self.assertIn("STAGED required", page)
        self.assertIn("run branch harness/&lt;plan&gt;/&lt;run-id&gt; on origin after final PASS", page)
        ff = self.make_config(policy="require-staged", mode="fast-forward-base")
        self.assertIn("main via safe fast-forward after final PASS", render_new_run(ff, "tok"))

    def test_config_rejects_unknown_policy_and_example_requires_staged(self) -> None:
        path = self.config_path(policy="always-staged")
        with self.assertRaisesRegex(ConfigError, "execution_mode_policy"):
            load_config(path)
        example = (ROOT / "examples" / "autowork.toml").read_text(encoding="utf-8")
        self.assertIn('execution_mode_policy = "require-staged"', example)
        self.assertIn('mode = "fast-forward-base"', example)


class RecordingPlanner(QueueClient):
    def __init__(self, responses: list[str], log: list[str], handle: LLMConversationHandle | None):
        super().__init__("planner", responses, log)
        self.handle = handle
        self.resumed: list[LLMConversationHandle] = []

    def complete(self, prompt: str) -> TextLLMResult:
        result = super().complete(prompt)
        return TextLLMResult(result.text, result.model, result.usage, {}, conversation=self.handle)

    def complete_in_conversation(self, handle: LLMConversationHandle, prompt: str) -> TextLLMResult:
        self.resumed.append(handle)
        return super().complete(prompt)


class ReusingReviewer(QueueClient):
    def __init__(self, responses: list[str], log: list[str], handle: LLMConversationHandle):
        super().__init__("reviewer", responses, log)
        self.handle = handle

    def complete(self, prompt: str) -> TextLLMResult:
        result = super().complete(prompt)
        return TextLLMResult(result.text, result.model, result.usage, {}, conversation=self.handle)

    def complete_in_conversation(self, handle: LLMConversationHandle, prompt: str) -> TextLLMResult:
        raise AssertionError("a reviewer never continues a conversation")


class ConversationPolicyTests(P29Harness):
    HANDLE = LLMConversationHandle("chatgpt-bridge", "conv-planner-1")

    def test_only_the_repair_planner_reuses_an_official_planner_handle(self) -> None:
        config = self.make_config()
        planner = RecordingPlanner([SINGLE_PLAN, REPAIR_PLAN], self.events, self.HANDLE)
        reviewer = ReusingReviewer([REVISE_IMPLEMENTATION, PASS], self.events,
                                   LLMConversationHandle("chatgpt-bridge", "conv-review-fresh"))
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): writer("src/a.py", "A = 4\n")})
        orchestrator, *_rest = self.orchestrator(config, planner=planner, reviewer=reviewer, luna=luna)
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "conversation")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(len(planner.prompts), 2)
        self.assertEqual(planner.resumed, [self.HANDLE])       # repair planner only
        self.assertEqual(len(reviewer.prompts), 2)              # two fresh reviews
        persisted = json.loads((result.run_dir / "planner.conversation.json").read_text())
        self.assertEqual(persisted, {"provider_id": "chatgpt-bridge", "conversation_id": "conv-planner-1"})

    def test_a_reviewer_in_the_planner_thread_is_rejected(self) -> None:
        config = self.make_config()
        planner = RecordingPlanner([SINGLE_PLAN], self.events, self.HANDLE)
        reviewer = ReusingReviewer([PASS], self.events, self.HANDLE)
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, *_rest = self.orchestrator(config, planner=planner, reviewer=reviewer, luna=luna)
        result = self.run_approved(config, orchestrator, "same-thread")
        self.assertEqual(result.state["failure"]["reason"], "REVIEWER_OUTPUT_INVALID")
        self.assertIn("planner conversation", result.state["failure"]["detail"])
        self.assertIsNone(result.state.get("commit_sha"))

    def test_without_an_official_handle_nothing_is_simulated(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, *_rest = self.orchestrator(config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna)
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "no-handle")
        self.assertEqual(result.status, RunStatus.PUBLISHED)
        self.assertFalse((result.run_dir / "planner.conversation.json").exists())
        with self.assertRaises(ValueError):
            LLMConversationHandle("bridge", " ")


class CheckpointModelTests(unittest.TestCase):
    IDENTITY = PlanIdentity("a" * 64, "b" * 64, bundle_sha256="c" * 64, execution_sha256="d" * 64)

    def test_roundtrip_completion_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertIsNone(read_checkpoint(run_dir))
            checkpoint = ResumeCheckpoint(ResumePhase.CLAUDE_C01, 1, None, "1" * 40, "2" * 40,
                                          "d" * 64, self.IDENTITY)
            write_checkpoint(run_dir, checkpoint)
            self.assertEqual(read_checkpoint(run_dir), checkpoint)
            mark_checkpoint_completed(run_dir)
            self.assertIsNone(read_checkpoint(run_dir))
            self.assertEqual(read_checkpoint_record(run_dir), (checkpoint, "completed"))
            (run_dir / CHECKPOINT_NAME).write_text("{not json")
            with self.assertRaises(ResumeCheckpointError):
                read_checkpoint(run_dir)
        cases = (
            dict(phase=ResumePhase.REVIEWER_C02, cycle=1),
            dict(phase=ResumePhase.INITIAL_STEP, step_id=None),
            dict(phase=ResumePhase.CLAUDE_C01, step_id="S01"),
            dict(phase=ResumePhase.CLAUDE_C02, cycle=2),  # no repair bundle hash
            dict(expected_tree_sha="not-a-sha"),
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                values = dict(phase=ResumePhase.CLAUDE_C01, cycle=1, step_id=None,
                              expected_head_sha="1" * 40, expected_tree_sha="2" * 40,
                              execution_selection_sha256="d" * 64, plan_identity=self.IDENTITY)
                values.update(overrides)
                with self.assertRaises(ResumeCheckpointError):
                    ResumeCheckpoint(**values)

    def test_cli_resume_command_refuses_unknown_and_published_runs(self) -> None:
        harness = P29Harness("run")
        harness.setUp()
        self.addCleanup(harness.tearDown)
        path = harness.config_path()
        (harness.runs / "done").mkdir(parents=True)
        store = RunStateStore(harness.runs / "done" / "state.json")
        store.initialize("done")
        store.update(status="published", planning_protocol="v2")
        for run_id, expected in (("missing", "run directory does not exist"), ("done", "not failed or interrupted")):
            with self.subTest(run_id=run_id):
                stderr = StringIO()
                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    code = cli.main(["resume", "--config", str(path), "--run-id", run_id])
                self.assertEqual(code, 2)
                self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
