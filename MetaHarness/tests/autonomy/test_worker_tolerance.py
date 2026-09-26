"""Scenarios 03 and 04 — harmless worker mistakes must not stop the run.

03: the worker edits its declared path plus one innocent neighbour (a test
file, neither sensitive nor on the hard-deny list).  The step completes and the
neighbour is recorded as out of scope, never as a hard stop.

04: the worker commits on the run branch.  The harness takes the Git state back,
keeps the change exploitable and never simulates a push.
"""

from __future__ import annotations

import unittest

from metaharness.agent import AgentRunRequest
from metaharness.models import ExecutionRole

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import Script, git, review

NEIGHBOUR = "def test_a():\n    assert True\n"
NEIGHBOUR_TOUCHED = "def test_a():\n    assert True\n# touched by the worker\n"
PARASITE_COMMIT = "worker parasite commit"


def edit_two_files() -> Script:
    """A scripted worker that edits its declared path and a neighbour."""

    def action(request: AgentRunRequest) -> str:
        (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
        (request.worktree / "tests/test_a.py").write_text(NEIGHBOUR_TOUCHED, encoding="utf-8")
        return "done\n"

    return action


def edit_and_commit() -> Script:
    """A scripted worker that edits its declared path and commits it."""

    def action(request: AgentRunRequest) -> str:
        (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
        git(request.worktree, "add", "src/a.py")
        git(request.worktree, "commit", "-qm", PARASITE_COMMIT)
        return "done\n"

    return action


class WorkerToleranceTests(AutonomyHarness):
    def change_a_plan(self) -> str:
        return meta_plan(Step(id="S01", title="Change a", read=("src/a.py",), write=("src/a.py",)))

    @unittest.expectedFailure
    def test_a_neighbour_file_edit_completes_the_step_as_out_of_scope(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n", "tests/test_a.py": NEIGHBOUR})
        self.green_check()
        self.workers.on(ExecutionRole.IMPLEMENTER, edit_two_files())

        result = self.orchestrator(
            self.config(), planner=[self.change_a_plan()], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
        step = self.step_record()
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["out_of_scope_paths"], ["tests/test_a.py"])
        self.assertEqual(
            (self.worktree() / "tests/test_a.py").read_text(encoding="utf-8"),
            NEIGHBOUR_TOUCHED,
        )

    @unittest.expectedFailure
    def test_a_worker_commit_leaves_the_harness_in_control_of_git(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n"})
        self.green_check()
        self.workers.on(ExecutionRole.IMPLEMENTER, edit_and_commit())

        result = self.orchestrator(
            self.config(), planner=[self.change_a_plan()], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
        # The worker's change stays exploitable and its commit is gone: the
        # accepted step commit is the harness's, on top of the base commit.
        self.assertEqual(
            (self.worktree() / "src/a.py").read_text(encoding="utf-8"), "a = 2\n",
        )
        self.assertNotIn(PARASITE_COMMIT, git(self.worktree(), "log", "--format=%s").splitlines())
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD^"), self.base_sha)
        self.assertEqual(self.state()["commit_sha"], git(self.worktree(), "rev-parse", "HEAD"))
