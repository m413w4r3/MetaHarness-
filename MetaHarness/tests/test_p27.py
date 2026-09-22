"""P27 final commit and safe run-branch publication coverage."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.cli import main  # noqa: E402
from metaharness.config import ConfigError, load_config  # noqa: E402
from metaharness.gitops import GitError, validate_run_branch  # noqa: E402
from metaharness.models import RunStatus  # noqa: E402
from tests import test_orchestrator_e2e as e2e_fixture  # noqa: E402

FakeLLM = e2e_fixture.FakeLLM
PASS_REVIEW = e2e_fixture.PASS_REVIEW
REVISE_REVIEW = e2e_fixture.REVISE_REVIEW


class P27Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = e2e_fixture.OrchestratorE2ETests("run_case")
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _run(self, *, review: str = PASS_REVIEW, run_id: str = "p27") -> tuple[int, dict]:
        root = self.fixture.root
        bare = root / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        subprocess.run(
            ["git", "-C", str(self.fixture.repo), "remote", "add", "origin", str(bare)],
            check=True,
        )
        worktree = root / "worktrees" / run_id
        llm = FakeLLM(review=review, worktree=worktree)
        config = self.fixture.config_file(llm, run_id=run_id)
        with config.open("a", encoding="utf-8") as stream:
            stream.write(
                '\n[publish]\nenabled = true\nremote = "origin"\nmode = "run-branch"\n'
            )
        old_environment = dict(os.environ)
        os.environ.update(
            {
                "PATH": str(root) + os.pathsep + old_environment.get("PATH", ""),
                "FAKE_CODEX_BEHAVIOR": "change",
                "FAKE_WORKTREE": str(worktree),
                "FAKE_CHECK": "pass",
                "FAKE_PROMPT": str(root / "prompt.txt"),
            }
        )
        try:
            exit_code = main(
                [
                    "run",
                    "--config",
                    str(config),
                    "--spec",
                    str(self.fixture.spec),
                    "--run-id",
                    run_id,
                ]
            )
        finally:
            os.environ.clear()
            os.environ.update(old_environment)
            llm.close()
        state = json.loads((root / "runs" / run_id / "state.json").read_text())
        return exit_code, state

    def test_publish_passes_exact_commit_to_bare_remote(self) -> None:
        exit_code, state = self._run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(state["status"], RunStatus.PUBLISHED.value)
        self.assertEqual(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.fixture.root / "origin.git"),
                    "rev-parse",
                    f"refs/heads/{state['branch']}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            state["commit_sha"],
        )
        self.assertEqual(json.loads((self.fixture.root / "runs/p27/publish.json").read_text())["status"], "pushed")

    def test_revise_pushes_the_reviewed_candidate_but_never_publishes(self) -> None:
        _exit_code, state = self._run(review=REVISE_REVIEW, run_id="revise")
        self.assertEqual(state["failure"]["reason"], "REVIEW_REVISE")
        # The reviewer is given the exact pushed candidate; a REVISE verdict
        # never turns it into a publication.
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(self.fixture.root / "origin.git"), "rev-parse", f"refs/heads/{state['branch']}"],
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
            state["candidate_commit_sha"],
        )
        self.assertFalse(state.get("publish"))
        self.assertFalse((self.fixture.root / "runs/revise/publish.json").exists())

    def test_push_rejection_preserves_local_commit_and_reason(self) -> None:
        root = self.fixture.root
        bare = root / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        hook = bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
        subprocess.run(
            ["git", "-C", str(self.fixture.repo), "remote", "add", "origin", str(bare)],
            check=True,
        )
        worktree = root / "worktrees" / "rejected"
        llm = FakeLLM(worktree=worktree)
        config = self.fixture.config_file(llm, run_id="rejected")
        with config.open("a", encoding="utf-8") as stream:
            stream.write('\n[publish]\nenabled = true\nremote = "origin"\n')
        old_environment = dict(os.environ)
        os.environ.update(
            {
                "PATH": str(root) + os.pathsep + old_environment.get("PATH", ""),
                "FAKE_CODEX_BEHAVIOR": "change",
                "FAKE_WORKTREE": str(worktree),
                "FAKE_CHECK": "pass",
                "FAKE_PROMPT": str(root / "prompt.txt"),
            }
        )
        try:
            main(["run", "--config", str(config), "--spec", str(self.fixture.spec), "--run-id", "rejected"])
        finally:
            os.environ.clear()
            os.environ.update(old_environment)
            llm.close()
        state = json.loads((root / "runs/rejected/state.json").read_text())
        self.assertEqual(state["failure"]["reason"], "PUSH_FAILED")
        # The candidate commit exists locally; the rejected push published nothing.
        self.assertIsInstance(state["candidate_commit_sha"], str)
        self.assertIsNone(state.get("commit_sha"))
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(bare), "show-ref"],
                capture_output=True,
                text=True,
            ).stdout,
            "",
        )

    def test_publish_mode_and_branch_namespace_are_strict(self) -> None:
        llm = FakeLLM()
        contents = self.fixture.config_file(llm, run_id="config").read_text()
        llm.close()
        invalid = contents + '\n[publish]\nmode = "main"\n'
        path = self.fixture.root / "invalid-publish.toml"
        path.write_text(invalid, encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "publish.mode"):
            load_config(path)
        for branch in ("main", "master", "tag", "feature/run", "harness//run"):
            with self.subTest(branch=branch):
                with self.assertRaises(GitError):
                    validate_run_branch(branch)


if __name__ == "__main__":
    unittest.main()
