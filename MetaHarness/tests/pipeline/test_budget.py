"""C10: one autonomous budget bounds every expensive autonomous operation.

These tests drive the real pipeline; the worker calls, the audit reports and
the durable state they assert on are the production artifacts.  The budget is
the one object ``[budget]`` freezes into ``run_options.json``: nothing here
patches an internal counter.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from metaharness.agent import AgentRunResult
from metaharness.config import ConfigError, load_config
from metaharness.gitops import candidate_tree_sha
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.planning.continue_request import planner_continue_dir

from tests.pipeline.support import (
    SPEC, STEP, PipelineHarness, audit, continuation_answer, crash_at_checkpoint,
    initial_plan, write,
)
from tests.pipeline_support import ScriptedChat, ScriptedPlannerMux
from tests.test_plan_repository_validation import impossible_plan_message


def milestone(raw: str, number: int) -> str:
    return raw.replace("MILESTONE_ID: M01", f"MILESTONE_ID: M{number:02d}")


def backdated_hours() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()


class BudgetTests(PipelineHarness):
    def mismatch(self):
        def action(_request):
            from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
            return CONTRACT_MISMATCH_HEADER + "\nThe approved step instructions were not met."
        return action

    def timeout(self):
        def action(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=candidate_tree_sha(request.worktree), usage=None,
                external_session_id=None, report_path=None, timed_out=True,
            )
        return action

    def implementer_calls(self) -> list:
        return [call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER]

    def auditor_calls(self) -> list:
        return [call for call in self.workers.calls if call.role is ExecutionRole.AUDITOR]

    def step_record(self, iteration: int = 1) -> dict:
        path = self.run_dir() / f"cycles/{iteration:03d}/implementation/steps/S01/step.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def config_with_fallbacks(self, *, budget, profiles):
        config = self.config(budget=budget)
        declared = ", ".join(f'"{profile}"' for profile in profiles)
        extra = f'\n[recovery.execution_fallbacks]\nmechanical = [{declared}]\n'
        for profile in profiles:
            extra += (
                f'\n[model_profiles.{profile}]\ndisplay_name = "{profile}"\n'
                f'roles = ["implementer"]\ndriver = "fake-worker"\nprovider = "test"\n'
                f'model = "fake-{profile}"\nselection_mode = "cli"\n'
            )
        self.config_path.write_text(
            self.config_path.read_text(encoding="utf-8") + extra, encoding="utf-8",
        )
        return load_config(self.config_path)

    def test_a_step_never_spends_more_than_its_attempt_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, *[self.mismatch() for _ in range(4)])
        result = self.orchestrator(
            self.config(budget={"step_attempts": 3}), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(len(self.implementer_calls()), 3)
        record = self.step_record()
        self.assertEqual(
            (record["status"], record["reason"]), ("FAILED_CONTINUED", "AGENT_CONTRACT_MISMATCH"),
        )
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)

    def test_a_fallback_rung_is_paid_from_the_same_attempt_budget(self) -> None:
        config = self.config_with_fallbacks(
            budget={"step_attempts": 3}, profiles=("rescue-one", "rescue-two"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, *[self.mismatch() for _ in range(4)])
        self.orchestrator(config, planner=[initial_plan(STEP)]).run_text(SPEC, run_id="run")

        calls = self.implementer_calls()
        self.assertEqual(
            [call.profile_id for call in calls], ["worker", "rescue-one", "rescue-two"],
        )
        self.assertEqual(len(calls), 3)

    def test_an_exhausted_transient_step_waits_for_the_external_provider(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, *[self.timeout() for _ in range(4)])
        result = self.orchestrator(
            self.config(budget={"step_attempts": 3}), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(len(self.implementer_calls()), 3)
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL)
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)

    def test_the_planner_shares_the_step_attempt_budget(self) -> None:
        invalid = impossible_plan_message()
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        accepted = self.orchestrator(
            self.config(budget={"step_attempts": 3}),
            planner=[invalid, invalid, initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(accepted.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 3)

        exhausted = self.orchestrator(
            self.config(budget={"step_attempts": 2}),
            planner=[invalid, invalid, initial_plan(STEP)],
        ).run_text(SPEC, run_id="run-two")
        self.assertEqual(len(self.planner.requests), 2)
        self.assertNotEqual(exhausted.status, RunStatus.PUBLISHED)
        self.assertNotEqual(exhausted.status, RunStatus.WAITING_HUMAN)

    def test_planner_continue_corrections_stay_inside_the_attempt_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.AUDITOR,
            audit("NEEDS_WORK", remaining="still broken"),
            audit("NEEDS_WORK", remaining="still broken"),
        )
        result = self.orchestrator(
            self.config(),
            planner=[initial_plan(STEP)],
            continuation=[
                continuation_answer("COMPLETE"), continuation_answer("COMPLETE"),
                continuation_answer("COMPLETE"),
                continuation_answer("NEXT", milestone="M02", plan_text=milestone(initial_plan(STEP), 2)),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(len(self.continuation.requests), 3)
        self.assertEqual(result.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        corrections = planner_continue_dir(self.run_dir() / "iterations", 1) / "corrections"
        self.assertTrue((corrections / "01").is_dir())
        self.assertTrue((corrections / "02").is_dir())
        self.assertFalse((corrections / "03").exists())

    def test_audit_repairs_bounds_the_writable_audits_of_one_iteration(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(
            ExecutionRole.AUDITOR,
            audit("NEEDS_WORK", remaining="still broken"),
            audit("NEEDS_WORK", remaining="still broken"),
            audit("DONE"),
        )
        m02 = milestone(initial_plan(STEP), 2)
        result = self.orchestrator(
            self.config(budget={"audit_repairs": 2}),
            planner=[initial_plan(STEP)],
            continuation=[
                continuation_answer("NEXT", milestone="M02", remaining="- still broken", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")

        first = self.run_dir() / "cycles/001/audit"
        self.assertEqual(sorted(item.name for item in first.iterdir()), ["001", "002"])
        self.assertEqual(len(self.auditor_calls()), 3)
        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))

    def test_zero_audit_repairs_reaches_the_continuation_without_an_audit(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.AUDITOR, audit())
        m02 = milestone(initial_plan(STEP), 2)
        result = self.orchestrator(
            self.config(budget={"audit_repairs": 0}),
            planner=[initial_plan(STEP)],
            continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(self.auditor_calls(), [])
        self.assertFalse((self.run_dir() / "cycles/001/audit").exists())
        request = json.loads(
            (planner_continue_dir(self.run_dir() / "iterations", 1) / "request.json").read_text()
        )
        self.assertIn("CHECK_FAILED:test", request["facts"]["evidence"])
        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))

    def test_max_iterations_counts_the_first_milestone_as_one(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"), write("other.txt", "second\n"),
        )
        m02 = milestone(initial_plan(("S01", "other.txt", "Write the companion")), 2)
        m03 = milestone(initial_plan(STEP), 3)
        result = self.orchestrator(
            self.config(budget={"max_iterations": 2}),
            planner=[initial_plan(STEP)],
            continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("NEXT", milestone="M03", plan_text=m03),
                continuation_answer("COMPLETE"),
            ],
            auditor=[audit(), audit(), audit()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PARTIAL)
        self.assertEqual(result.state["partial"]["reason"], "max_iterations")
        self.assertEqual(result.state["current_iteration"], 2)
        self.assertEqual(len(self.continuation.requests), 2)
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)

    def test_the_wall_clock_is_read_from_the_durable_creation_timestamp(self) -> None:
        class Backdating:
            """First planner answer: backdate the durable run creation time."""

            def __init__(self, run_dir: Path, answer: str) -> None:
                self.run_dir = run_dir
                self.answer = answer

            def complete(self, _request: str) -> str:
                path = self.run_dir / "state.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                state["started_at"] = backdated_hours()
                path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return self.answer

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.continuation = ScriptedChat([continuation_answer("COMPLETE")], name="planner_continue")
        orchestrator = Orchestrator(
            self.config(),
            planner_client=ScriptedPlannerMux(
                Backdating(self.run_dir(), initial_plan(STEP)), self.continuation,
            ),
        )
        result = orchestrator.run_text(SPEC, run_id="run")

        self.assertEqual(self.implementer_calls(), [])
        self.assertEqual(result.status, RunStatus.PARTIAL)
        self.assertEqual(result.state["partial"]["reason"], "wall_clock")
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)

    def test_a_resume_never_restarts_the_wall_clock(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "implement_step"):
            crashed = original.run_text(SPEC, run_id="run")
        self.assertEqual(crashed.status, RunStatus.WAITING_EXTERNAL)

        path = self.run_dir() / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["started_at"] = backdated_hours()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(self.implementer_calls(), [])
        self.assertEqual(resumed.status, RunStatus.PARTIAL)
        self.assertEqual(resumed.state["partial"]["reason"], "wall_clock")

    def test_the_state_exposes_the_budget_without_fabricating_a_cost(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(budget={"max_cost": 0}), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        budget = self.state()["budget"]
        self.assertEqual(
            set(budget), {"configured", "iterations", "elapsed_seconds", "cost_usd"},
        )
        self.assertEqual(budget["configured"]["max_cost"], 0)
        self.assertEqual(budget["iterations"], 1)
        self.assertIsInstance(budget["elapsed_seconds"], int)
        self.assertIsNone(budget["cost_usd"])

    def test_a_positive_cost_cap_is_refused_without_a_cost_authority(self) -> None:
        with self.assertRaisesRegex(ConfigError, "max_cost > 0 is not supported"):
            self.config(budget={"max_cost": 1.5})


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
