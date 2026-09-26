"""Direct publication-boundary checks kept under the requested test name."""

from __future__ import annotations

import inspect
import json
import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.cli import main  # noqa: E402
from metaharness.gitops import commit_candidate_tree, push_run_branch  # noqa: E402
from metaharness.orchestration.publication import PublicationService
from tests import test_orchestrator_e2e as e2e_fixture  # noqa: E402


class PublicationBoundaryTests(unittest.TestCase):
    def test_candidate_primitives_and_final_publication_are_separate(self) -> None:
        self.assertTrue(callable(commit_candidate_tree))
        self.assertTrue(callable(push_run_branch))
        source = inspect.getsource(PublicationService._complete_candidate_publication)
        self.assertIn("publish_fast_forward_base", source)
        self.assertNotIn("commit_candidate_tree", source)


class RejectedPublicationTests(unittest.TestCase):
    """A refused publication publishes nothing and loses no local candidate."""

    def setUp(self) -> None:
        self.fixture = e2e_fixture.OrchestratorE2ETests("run_case")
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def test_push_rejection_preserves_local_commit_and_reason(self) -> None:
        root = self.fixture.root
        bare = root / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        hook = bare / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
        subprocess.run(
            ["git", "-C", str(self.fixture.repo), "remote", "set-url", "origin", str(bare)],
            check=True,
        )
        worktree = root / "worktrees" / "rejected"
        llm = e2e_fixture.FakeLLM(worktree=worktree)
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
                ["git", "-C", str(bare), "rev-parse", "refs/heads/main"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            self.fixture.base_sha,
        )
        self.assertNotEqual(
            subprocess.run(
                ["git", "-C", str(bare), "rev-parse", f"refs/heads/{state['branch']}"],
                capture_output=True,
                text=True,
            ).returncode,
            0,
        )


if __name__ == "__main__":
    unittest.main()
