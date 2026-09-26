"""A3b — a temporary transport exhaustion is a resumable ``WAIT_EXTERNAL``.

The run must never read a provider outage as a verdict on the SPEC, the plan
or the implementation: the planner transport exhausts its horizon, the run
waits externally at the planner checkpoint, and ``resume`` pays for a fresh
planner call only because no valid answer was ever persisted.
"""

from __future__ import annotations

import time
import unittest
from typing import Any
from unittest import mock

from metaharness.cli import main as cli_main
from metaharness.llm.chat import LLMTransportExhaustedError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.planning.protocol import PlanParseError
from metaharness.resume import resume_info

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import ScriptedChat, review, write


class PlannerTransportExhaustionTests(AutonomyHarness):
    """The planner checkpoint is durable before the planner is ever called."""

    def plan(self) -> str:
        return meta_plan(
            Step(id="S01", title="Write the feature", write=("feature.txt",)),
        )

    def test_planner_transport_exhaustion_is_resumable(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        planner = ScriptedChat(
            [LLMTransportExhaustedError("LLM transport horizon exhausted"), self.plan()],
            name="planner",
        )
        reviewer = ScriptedChat([review()], name="reviewer")
        config = self.config()

        failed = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
        ).run_text(SPEC, run_id="run")

        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.failure_reason(), "LLM_TRANSPORT_EXHAUSTED")
        # No fake plan is persisted and the planner is not marked done: the
        # durable checkpoint is the planner itself, taken before the call.
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertFalse((self.run_dir() / "planner.raw.md").exists())
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(len(planner.requests), 1)

        resumed = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The exhausted call bought nothing durable: the resume pays for one
        # fresh planner answer and the pipeline continues past the planner.
        self.assertEqual(len(planner.requests), 2)


class _CliClock:
    """A fake sleeper for the ``--auto-resume`` loop, and a real clock elsewhere.

    The loop's two time calls are faked so an interval costs no wall time and
    the ceiling is deterministic; every other attribute stays the real module,
    so the rest of the process keeps its own timing.
    """

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self._now = time.monotonic()

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += max(0.0, seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class AutoResumeCommandTests(AutonomyHarness):
    """``--auto-resume`` retries WAIT_EXTERNAL runs in the current process."""

    def plan(self) -> str:
        return meta_plan(
            Step(id="S01", title="Write the feature", write=("feature.txt",)),
        )

    def setUp(self) -> None:
        super().setUp()
        self.spec_path = self.root / "spec.txt"
        self.spec_path.write_text(SPEC, encoding="utf-8")

    def scripted_clients(self, planner: ScriptedChat, reviewer: ScriptedChat) -> Any:
        """Replace the production chat client with the scripted double.

        The replacement keeps the constructor surface ``chat_client`` probes,
        so every profile builds the scripted client instead of a real socket.
        """

        def factory(endpoint, environment=None, on_transport=None):
            if endpoint.model == "fake-planner":
                return planner
            if endpoint.model == "fake-reviewer":
                return reviewer
            raise AssertionError(f"unexpected chat profile: {endpoint.model}")

        return mock.patch(
            "metaharness.orchestration.shared.OpenAIChatTextClient", factory
        )

    def run_with_auto_resume(
        self, planner: ScriptedChat, reviewer: ScriptedChat, *,
        interval: str = "0.01", run_id: str = "run",
    ) -> tuple[int, list[float]]:
        clock = _CliClock()
        argv = [
            "run", "--config", str(self.config_path), "--spec", str(self.spec_path),
            "--run-id", run_id, "--auto-resume", "--auto-resume-interval", interval,
        ]
        with (
            mock.patch("metaharness.cli.time", clock),
            self.scripted_clients(planner, reviewer),
        ):
            code = cli_main(argv)
        return code, clock.sleeps

    def test_auto_resume_retries_wait_external_run(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        planner = ScriptedChat(
            [LLMTransportExhaustedError("LLM transport horizon exhausted"), self.plan()],
            name="planner",
        )
        self.config()

        code, sleeps = self.run_with_auto_resume(planner, ScriptedChat([review()]))

        self.assertEqual(code, 0, self.state().get("failure"))
        self.assertEqual(self.state()["status"], RunStatus.COMMITTED.value)
        self.assertEqual(sleeps, [0.01])
        # One exhausted attempt, then the resumed attempt that planned the run.
        self.assertEqual(len(planner.requests), 2)

    def test_auto_resume_stops_when_run_is_not_wait_external(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        planner = ScriptedChat([self.plan()], name="planner")
        self.config()

        code, sleeps = self.run_with_auto_resume(planner, ScriptedChat([review()]))

        self.assertEqual(code, 0, self.state().get("failure"))
        # A run that never waits externally is never slept on, never resumed.
        self.assertEqual(sleeps, [])
        self.assertEqual(len(planner.requests), 1)

    def test_nonretryable_planner_error_is_not_auto_resumed(self) -> None:
        planner = ScriptedChat([PlanParseError("not a plan")], name="planner")
        self.config()

        code, sleeps = self.run_with_auto_resume(planner, ScriptedChat([review()]))

        self.assertEqual(code, 1)
        self.assertEqual(self.state()["status"], RunStatus.WAITING_HUMAN.value)
        # An invalid planner answer is an operator decision, not a temporary
        # outage: the transport failure would have been retried, this is not.
        self.assertEqual(sleeps, [])
        self.assertIsNone((self.state().get("resume") or {}).get("attempts"))


if __name__ == "__main__":
    unittest.main()
