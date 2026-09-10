"""Local end-to-end tests for the complete V0 orchestration state machine."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.cli import main  # noqa: E402
from metaharness.config import load_config  # noqa: E402
from metaharness.models import RunStatus  # noqa: E402
from metaharness.orchestrator import Orchestrator  # noqa: E402


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return result.stdout.strip()


PLAN = """STATUS: READY
TITLE: Add the feature
OBJECTIVE: Implement the requested feature.
CONSTRAINTS: Keep the change local.
FILES: feature.txt
IMPLEMENTATION: Create feature.txt with the requested content.
ACCEPTANCE: The feature file exists.
TESTS: Run the configured test.
RISKS: NONE
BLOCKERS: NONE
"""

PASS_REVIEW = """VERDICT: PASS
ROUTE: NONE
SUMMARY: The implementation is acceptable.
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""

REVISE_REVIEW = """VERDICT: REVISE
ROUTE: IMPLEMENTATION
SUMMARY: One correction is required.
FINDINGS: MINOR | The content needs a correction.
REQUIRED FIXES: Fix feature.txt.
MISSING TESTS: Add a regression test.
RESIDUAL RISKS: NONE
"""

FAIL_REVIEW = """VERDICT: FAIL
ROUTE: HUMAN
SUMMARY: The implementation cannot be accepted.
FINDINGS: BLOCKER | The implementation is unsafe.
REQUIRED FIXES: Rework the change.
MISSING TESTS: Add safety coverage.
RESIDUAL RISKS: High.
"""


class FakeLLM:
    """One local HTTP server standing in for planner and reviewer endpoints."""

    def __init__(self, *, planner: str = PLAN, review: str = PASS_REVIEW, mutate: str | None = None, worktree: Path | None = None):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                owner.requests.append((self.path, self.rfile.read(length).decode()))
                if self.path.endswith("/planner"):
                    owner.planner_calls += 1
                    response = owner.planner
                elif self.path.endswith("/reviewer"):
                    owner.reviewer_calls += 1
                    response = owner.review
                else:
                    self.send_error(404)
                    return
                payload = json.dumps({"model": "fake", "choices": [{"message": {"content": response}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                if self.path.endswith("/reviewer"):
                    owner._mutate_after_review()

            def log_message(self, *_args: Any) -> None:
                return

        self.planner = planner
        self.review = review
        self.mutate = mutate
        self.worktree = worktree
        self.requests: list[tuple[str, str]] = []
        self.planner_calls = 0
        self.reviewer_calls = 0
        self._mutated = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def _mutate_after_review(self) -> None:
        if self._mutated or self.mutate is None or self.worktree is None:
            return
        self._mutated = True
        target = self.worktree / "feature.txt"
        if self.mutate == "index":
            target.write_text("tampered\n", encoding="utf-8")
            git(self.worktree, "add", "feature.txt")
        elif self.mutate == "working":
            target.write_text("tampered\n", encoding="utf-8")
        elif self.mutate == "untracked":
            (self.worktree / "late.txt").write_text("late\n", encoding="utf-8")

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class InterruptingAgent:
    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt


class OrchestratorE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "MetaHarness E2E")
        git(self.repo, "config", "user.email", "e2e@example.invalid")
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.spec = self.root / "SPEC.md"
        self.spec.write_text("Implement the feature without exposing the original request.\n", encoding="utf-8")
        # CodexAgent invokes the configured V0 executable by the stable name
        # ``codex``; putting this test double first on PATH keeps the test local.
        self.codex = self.root / "codex"
        self.codex.write_text(
            "#!/usr/bin/env python3\n" + textwrap.dedent(
                """
                import os, pathlib, subprocess, sys, time
                prompt = sys.stdin.read()
                pathlib.Path(os.environ['FAKE_PROMPT']).write_text(prompt)
                behavior = os.environ.get('FAKE_CODEX_BEHAVIOR', 'change')
                worktree = pathlib.Path(os.environ['FAKE_WORKTREE'])
                target = worktree / 'feature.txt'
                if behavior == 'change':
                    target.write_text('implemented\\n')
                elif behavior == 'large':
                    target.write_text('x' * 1000)
                elif behavior == 'commit':
                    target.write_text('implemented\\n')
                    subprocess.run(['git', '-C', str(worktree), 'add', '-A'], check=True)
                    subprocess.run(['git', '-C', str(worktree), 'commit', '-qm', 'agent-owned'], check=True)
                elif behavior == 'sleep':
                    time.sleep(30)
                final = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])
                final.write_text('fake codex completed\\n')
                if behavior == 'fail':
                    raise SystemExit(1)
                """
            ),
            encoding="utf-8",
        )
        self.codex.chmod(self.codex.stat().st_mode | stat.S_IXUSR)
        self.check = self.root / "check.py"
        self.check.write_text(
            "import os, sys\nif os.environ.get('FAKE_CHECK') == 'fail': sys.exit(1)\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config_file(self, llm: FakeLLM, *, run_id: str = "run-1", max_diff: int = 400000) -> Path:
        config = self.root / "config.toml"
        config.write_text(
            "\n".join([
                f"repo = {str(self.repo)!r}",
                'base_ref = "HEAD"',
                f"runs_root = {str(self.root / 'runs')!r}",
                f"worktrees_root = {str(self.root / 'worktrees')!r}",
                "require_clean_base = true",
                f"max_diff_bytes = {max_diff}",
                "",
                f"[planner]\nbase_url = {llm.base_url!r}\nendpoint_path = \"/planner\"\nmodel = \"fake-planner\"\nretries = 0",
                f"[reviewer]\nbase_url = {llm.base_url!r}\nendpoint_path = \"/reviewer\"\nmodel = \"fake-reviewer\"\nretries = 0",
                "[context]\nalways_files = []",
                "[agent]\nmodel = \"gpt-5.6-luna\"\neffort = \"high\"\ntimeout_seconds = 3",
                f"[[checks]]\nname = \"test\"\nargv = [{str(sys.executable)!r}, {str(self.check)!r}]\ntimeout_seconds = 3",
            ]) + "\n",
            encoding="utf-8",
        )
        return config

    def run_case(
        self,
        *,
        planner: str = PLAN,
        review: str = PASS_REVIEW,
        check_fail: bool = False,
        codex_behavior: str = "change",
        mutate: str | None = None,
        max_diff: int = 400000,
        run_id: str = "run-1",
    ) -> tuple[Any, FakeLLM, Path]:
        worktree = self.root / "worktrees" / run_id
        llm = FakeLLM(planner=planner, review=review, mutate=mutate, worktree=worktree)
        config = self.config_file(llm, run_id=run_id, max_diff=max_diff)
        old_path = os.environ.get("PATH")
        old_behavior = os.environ.get("FAKE_CODEX_BEHAVIOR")
        old_check = os.environ.get("FAKE_CHECK")
        os.environ["PATH"] = str(self.root) + os.pathsep + (old_path or "")
        os.environ["FAKE_CODEX_BEHAVIOR"] = codex_behavior
        os.environ["FAKE_WORKTREE"] = str(worktree)
        os.environ["FAKE_CHECK"] = "fail" if check_fail else "pass"
        os.environ["FAKE_PROMPT"] = str(self.root / "prompt.txt")
        try:
            exit_code = main(["run", "--config", str(config), "--spec", str(self.spec), "--run-id", run_id])
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
            for name, old in (("FAKE_CODEX_BEHAVIOR", old_behavior), ("FAKE_CHECK", old_check)):
                if old is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
        state = json.loads((self.root / "runs" / run_id / "state.json").read_text())
        llm.close()
        return exit_code, llm, state

    def test_ready_changes_green_pass_commits_exactly_once(self) -> None:
        exit_code, llm, state = self.run_case()
        worktree = self.root / "worktrees" / "run-1"
        self.assertEqual(exit_code, 0)
        self.assertEqual(state["status"], RunStatus.COMMITTED.value)
        self.assertEqual(git(worktree, "rev-list", "--count", "HEAD"), "2")
        self.assertIn("Generated by MetaHarness.", git(worktree, "log", "-1", "--format=%B"))
        self.assertEqual(llm.planner_calls, 1)
        self.assertEqual(llm.reviewer_calls, 1)
        self.assertNotIn(self.spec.read_text(), (self.root / "prompt.txt").read_text())

    def test_planner_blocked_creates_no_worktree(self) -> None:
        blocked = "STATUS: BLOCKED\nBLOCKERS: missing information\n"
        exit_code, _, state = self.run_case(planner=blocked, run_id="blocked")
        self.assertEqual(exit_code, 1)
        self.assertEqual(state["status"], RunStatus.BLOCKED.value)
        self.assertFalse((self.root / "worktrees" / "blocked").exists())

    def test_codex_exit_one_creates_no_harness_commit(self) -> None:
        _, _, state = self.run_case(codex_behavior="fail")
        self.assertEqual(state["failure"]["reason"], "AGENT_FAILED")
        self.assertEqual(git(self.root / "worktrees" / "run-1", "rev-list", "--count", "HEAD"), "1")

    def test_codex_commit_itself_is_rejected(self) -> None:
        _, _, state = self.run_case(codex_behavior="commit")
        worktree = self.root / "worktrees" / "run-1"
        self.assertEqual(state["status"], RunStatus.FAILED.value)
        self.assertNotIn("Generated by MetaHarness.", git(worktree, "log", "-1", "--format=%B"))

    def test_commit_is_impossible_without_both_gates(self) -> None:
        _, llm, state = self.run_case(check_fail=True, review=PASS_REVIEW)
        self.assertEqual(state["status"], RunStatus.FAILED.value)
        self.assertEqual(state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(llm.reviewer_calls, 1)
        self.assertEqual(git(self.root / "worktrees" / "run-1", "rev-list", "--count", "HEAD"), "1")

    def test_revise_fail_and_empty_or_large_diff_never_commit(self) -> None:
        _, _, revise = self.run_case(review=REVISE_REVIEW, run_id="revise")
        revise_dir = self.root / "runs" / "revise"
        self.assertEqual(revise["failure"]["reason"], "REVIEW_REVISE")
        self.assertTrue((revise_dir / "repair_task.md").exists())
        self.assertTrue((revise_dir / "repair_task.json").exists())

        _, _, failed = self.run_case(review=FAIL_REVIEW, run_id="review-fail")
        self.assertEqual(failed["failure"]["reason"], "REVIEW_FAIL")

        _, empty_llm, empty = self.run_case(codex_behavior="none", run_id="empty")
        self.assertEqual(empty["failure"]["reason"], "EMPTY_DIFF")
        self.assertEqual(empty_llm.reviewer_calls, 0)

        _, large_llm, large = self.run_case(max_diff=10, run_id="large", codex_behavior="large")
        self.assertEqual(large["failure"]["reason"], "DIFF_TOO_LARGE")
        self.assertEqual(large_llm.reviewer_calls, 0)

    def test_index_worktree_and_untracked_changes_after_review_are_rejected(self) -> None:
        for mode in ("index", "working", "untracked"):
            _, _, state = self.run_case(mutate=mode, run_id=mode)
            self.assertEqual(state["failure"]["reason"], "TOCTOU_FAILURE")
            self.assertEqual(git(self.root / "worktrees" / mode, "rev-list", "--count", "HEAD"), "1")

    def test_ctrl_c_is_persisted_as_interrupted_without_commit(self) -> None:
        llm = FakeLLM()
        config = load_config(self.config_file(llm, run_id="interrupt"))
        result = Orchestrator(config, agent=InterruptingAgent()).run(self.spec, run_id="interrupt")
        llm.close()
        self.assertEqual(result.status, RunStatus.INTERRUPTED)
        self.assertEqual(result.state["failure"]["reason"], "INTERRUPTED")
        self.assertTrue((self.root / "worktrees" / "interrupt").exists())


if __name__ == "__main__":
    unittest.main()
