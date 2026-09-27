"""The approved plan step stays the only worker and commit authority."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.commit_gate import commit_safety_gate
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestration.step_acceptance import StepAcceptanceService
from metaharness.orchestration.step_authority import read_step_candidate
from metaharness.orchestrator import Orchestrator
from metaharness.resume import resume_info
from tests.pipeline.support import PipelineHarness, git, initial_plan, write

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")


class NoCall:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def complete(self, request: str) -> str:
        self.requests.append(request)
        raise AssertionError("resume must not call the planner")


class StepAuthorityTests(PipelineHarness):
    def step_dir(self) -> Path:
        return self.run_dir() / "cycles/001/implementation/steps/S01"

    def json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def run_pipeline(self):
        return self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

    def resume_without_planner(self):
        planner = NoCall()
        return Orchestrator(self.config(), planner_client=planner).resume("run"), planner

    def test_approved_authority_is_bound_through_candidate_and_commit(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        result = self.run_pipeline()

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        step = self.json(self.step_dir() / "step.json")
        candidate = read_step_candidate(self.step_dir())
        self.assertEqual(step["changed_paths"], ["feature.txt"])
        self.assertEqual(
            candidate["approved_contract_sha256"], candidate["effective_contract_sha256"],
        )
        (record,) = self.json(self.run_dir() / "accepted-chain.json")["commits"]
        self.assertEqual(
            record["effective_authority_sha256"], candidate["effective_authority_sha256"],
        )
        self.assertNotIn("repair_slot", candidate)

    def test_interrupted_acceptance_replays_the_implementation_step(self) -> None:
        def mismatch(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            return CONTRACT_MISMATCH_HEADER + "\nThe step instructions were not met."

        self.workers.on(
            ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"),
            write("feature.txt", "good\n"),
        )
        with mock.patch(
            "metaharness.orchestration.step_acceptance.commit_safety_gate",
            side_effect=KeyboardInterrupt(),
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.INTERRUPTED, self.state().get("failure"))
        self.assertTrue((self.step_dir() / "attempts/01/step.json").exists())
        self.assertFalse((self.step_dir() / "contract_repairs").exists())
        info = resume_info(self.run_dir(), self.state())
        self.assertIsNone(info.operation)
        self.assertEqual(info.phase, "implement_step")

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(planner.requests, [])
        self.assertEqual(len([c for c in self.workers.calls if c.role is ExecutionRole.IMPLEMENTER]), 3)

    def test_uncheckpointed_candidate_is_rebuilt_by_step_replay(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
        )
        with mock.patch(
            "metaharness.orchestration.step_acceptance.commit_safety_gate",
            side_effect=KeyboardInterrupt(),
        ):
            self.run_pipeline()
        path = self.step_dir() / "step_candidate.json"
        payload = self.json(path)
        payload["changed_paths"] = ["other.txt"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(read_step_candidate(self.step_dir())["changed_paths"], ["feature.txt"])
        self.assertEqual(planner.requests, [])
        self.assertEqual(
            len([call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER]), 2,
        )

    def test_git_ownership_drift_fails_closed(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch.object(
            StepAcceptanceService, "_finalize_accepted_step", side_effect=KeyboardInterrupt(),
        ):
            self.run_pipeline()
        git(self.worktree(), "checkout", "-q", "-b", "operator-branch")

        resumed, planner = self.resume_without_planner()

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(planner.requests, [])
