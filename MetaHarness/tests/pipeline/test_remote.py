"""The remote run branch: candidate staging, outages and publication authority."""

from __future__ import annotations

import json
import unittest

from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    ScriptedChat,
    break_remote,
    crash_on_review,
    divergent_run_branch,
    git,
    initial_plan,
    move_run_branch,
    reject_pushes,
    restore_remote,
    review,
    run_branch,
    write,
)


class RemoteCandidateTests(PipelineHarness):
    def test_publish_disabled_still_pushes_exact_candidate_before_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(candidate["remote"], "origin")
        self.assertEqual(candidate["remote_branch"], self.state()["branch"])
        self.assertEqual(candidate["remote_sha"], candidate["commit_sha"])
        self.assertTrue(candidate["pushed_at"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])
        self.assertIn(candidate["commit_sha"], self.reviewer.requests[0])
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_optional_candidate_push_failure_uses_local_review_and_commits(self) -> None:
        """The staging remote is a real repository that is unreachable."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 1)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        request = self.reviewer.requests[0]
        self.assertIn('"remote_exploration": "UNAVAILABLE"', request)
        self.assertIn("<BOUNDED DIFF EXCERPT>", request)
        self.assertIn("candidate.push_unavailable", self.trace_names())

    def test_different_remote_tip_does_not_block_local_candidate_review(self) -> None:
        """An unrelated tip already on the run branch is not a local blocker."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        branch = run_branch()
        divergent = divergent_run_branch(self, branch)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(self.state()["branch"], branch)
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 1)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        self.assertEqual(self.remote_tip(branch), divergent)

    def test_remote_push_without_web_url_uses_bounded_diff_fallback(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        request = self.reviewer.requests[0]
        self.assertIn('"remote_exploration": "UNAVAILABLE"', request)
        self.assertIn("<BOUNDED DIFF EXCERPT>", request)

    def test_required_candidate_push_waits_and_resumes_at_candidate_push(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        waiting = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")

        restore_remote(self)
        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])

    def test_pr_creation_requires_remote_candidate_before_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        waiting = self.orchestrator(
            self.config(publish=True, github_pr=True),
            planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertEqual(self.reviewer.requests, [])

    def test_publication_rejects_remote_tip_change_after_local_review_pass(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        branch = run_branch()
        harness = self

        class ReviewerThatMovesTheRunBranch:
            """A reviewer that answers after someone else moved the branch."""

            def __init__(self) -> None:
                self.requests: list[str] = []
                self.moved = False

            def complete(self, prompt: str) -> str:
                self.requests.append(prompt)
                if not self.moved:
                    self.moved = True
                    move_run_branch(harness, branch)
                return review()

        reviewer = ReviewerThatMovesTheRunBranch()
        result = Orchestrator(
            self.config(publish=True),
            planner_client=ScriptedChat([initial_plan(STEP)], name="planner", events=self.events),
            reviewer_client=reviewer,
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "COMMIT_TREE_MISMATCH")
        self.assertFalse((self.run_dir() / "publish.json").exists())
        self.assertEqual(len(reviewer.requests), 1)

    def test_push_then_crash_resumes_review_with_the_same_remote_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_on_review(original, 1):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text(encoding="utf-8")
        )
        branch = self.state()["branch"]
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])

        # The remote now refuses every push while staying readable: the resume
        # must publish the candidate it already staged there, not repush it.
        reject_pushes(self)
        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)

if __name__ == "__main__":
    unittest.main()
