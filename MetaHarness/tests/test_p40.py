"""P40: immutable candidate commit/push precedes semantic review."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from metaharness.gitops import commit_parents
from metaharness.models import RunStatus
from metaharness.resume import ResumePhase, read_checkpoint
from tests.test_p29 import (
    P29Harness, QueueClient, FakeLuna, SINGLE_PLAN, REPAIR_PLAN, PASS,
    REVISE_IMPLEMENTATION, SPEC, writer, git, TransportFailingClient,
)


class P40CandidatePipelineTests(P29Harness):
    def test_candidate_commit_and_push_precede_reviewer_and_main_is_unchanged(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "fetch", "-q", "origin")
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        harness = self
        class ObservingReviewer(QueueClient):
            def complete(self, prompt: str):
                self.main_before_pass = git(harness.repo, "rev-parse", "refs/heads/main")
                return super().complete(prompt)
        reviewer = ObservingReviewer("reviewer", [PASS], self.events)
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviewer=reviewer, luna=luna
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "candidate-order")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        candidate = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(candidate["commit_sha"], result.state["commit_sha"])
        self.assertEqual(candidate["parent_sha"], self.base_sha)
        self.assertTrue(candidate["pushed_at"])
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(self.bare), "rev-parse", f"refs/heads/{result.state['branch']}"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            candidate["commit_sha"],
        )
        self.assertEqual(reviewer.main_before_pass, self.base_sha)
        self.assertLess(self.events.index("push"), next(
            index for index, value in enumerate(self.events) if value.startswith("reviewer:")
        ))

    def test_c02_candidate_is_parented_to_c01(self) -> None:
        config = self.make_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 3\n"),
        })
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_PLAN],
            reviews=[REVISE_IMPLEMENTATION, PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "candidate-ancestry")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        c01 = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        c02 = json.loads((result.run_dir / "candidate/C02/commit.json").read_text())
        self.assertEqual(c02["parent_sha"], c01["commit_sha"])
        self.assertEqual(commit_parents(self.repo, c02["commit_sha"]), (c01["commit_sha"],))

    def test_reviewer_transport_resumes_without_replaying_candidate_push(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        first_reviewer = TransportFailingClient()
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviewer=first_reviewer, luna=luna
        )
        failed = self.run_approved(config, orchestrator, "candidate-transport")
        self.assertEqual(failed.state["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        self.assertEqual(read_checkpoint(failed.run_dir).phase, ResumePhase.REVIEWER_C01)
        candidate = json.loads((failed.run_dir / "candidate/C01/commit.json").read_text())
        self.assertTrue(candidate["pushed_at"])
        second, *_rest = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushes:
            resumed = second.resume("candidate-transport")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(pushes.call_count, 0)


if __name__ == "__main__":
    unittest.main()
