"""The remote run branch: candidate staging, outages and publication authority."""

from __future__ import annotations

import json
import unittest

from metaharness.models import ExecutionRole, RunStatus
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    break_remote,
    crash_at_checkpoint,
    divergent_run_branch,
    git,
    initial_plan,
    move_run_branch,
    reject_pushes,
    restore_remote,
    run_branch,
    write,
)


class RemoteCandidateTests(PipelineHarness):
    def test_publish_disabled_still_pushes_the_exact_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(candidate["remote"], "origin")
        self.assertEqual(candidate["remote_branch"], self.state()["branch"])
        self.assertEqual(candidate["remote_sha"], candidate["commit_sha"])
        self.assertTrue(candidate["pushed_at"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_optional_candidate_push_failure_still_publishes_the_local_candidate(self) -> None:
        """The staging remote is a real repository that is unreachable."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        self.assertIn("candidate.push_unavailable", self.trace_names())

    def test_different_remote_tip_does_not_block_local_publication(self) -> None:
        """An unrelated tip already on the run branch is not a local blocker."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        branch = run_branch()
        divergent = divergent_run_branch(self, branch)
        result = self.orchestrator(self.config(), planner=[initial_plan(STEP)]).run_text(
            SPEC, run_id="run",
        )

        self.assertEqual(self.state()["branch"], branch)
        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        self.assertEqual(self.remote_tip(branch), divergent)

    def test_audit_prompt_carries_the_local_evidence_not_a_remote_url(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        self.orchestrator(self.config(publish=False), planner=[initial_plan(STEP)]).run_text(
            SPEC, run_id="run",
        )

        audit = next(
            request for request in self.workers.calls
            if request.role is ExecutionRole.AUDITOR
        )
        prompt = audit.prompt
        self.assertIn('"gate"', prompt)
        self.assertIn('"baseline"', prompt)
        self.assertIn('"changed_paths"', prompt)
        self.assertIn("+good", prompt)
        self.assertNotIn("remote_exploration", prompt)

    def test_required_candidate_push_waits_and_resumes_at_candidate_push(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        waiting = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])

        restore_remote(self)
        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])

    def test_pr_creation_requires_remote_candidate_before_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        break_remote(self)
        waiting = self.orchestrator(
            self.config(publish=True, github_pr=True),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertFalse((self.run_dir() / "publish.json").exists())
        self.assertFalse((self.run_dir() / "github").exists())

    def test_publication_rejects_remote_tip_change_after_local_acceptance(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        branch = run_branch()
        original = self.orchestrator(self.config(publish=True), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "publish", occurrence=1):
            crashed = original.run_text(SPEC, run_id="run")
        self.assertEqual(crashed.status, RunStatus.WAITING_EXTERNAL)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])

        # Someone else force-moved the run branch after the local acceptance:
        # the publication authority must refuse, never publish another tree.
        moved = move_run_branch(self, branch)
        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "COMMIT_TREE_MISMATCH")
        self.assertFalse((self.run_dir() / "publish.json").exists())
        self.assertEqual(self.remote_tip(branch), moved)

    def test_push_then_crash_resumes_publication_with_the_same_remote_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(publish=True), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "publish", occurrence=1):
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
            self.config(publish=True), planner=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])


if __name__ == "__main__":
    unittest.main()
