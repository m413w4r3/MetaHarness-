"""A3b — a temporary transport exhaustion is a resumable ``WAIT_EXTERNAL``.

The run must never read a provider outage as a verdict on the SPEC, the plan
or the implementation: the planner transport exhausts its horizon, the run
waits externally at the planner checkpoint, and ``resume`` pays for a fresh
planner call only because no valid answer was ever persisted.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from metaharness.cli import _auto_resume, main as cli_main
from metaharness.llm.chat import LLMTransportExhaustedError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.planning.protocol import PlanParseError
from metaharness.result import RunResult
from metaharness.resume import resume_info

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import ScriptedChat, continuation_answer, write

AUDIT_DONE = (
    "META AUDIT v1\n\nSTATUS\nDONE\n\nFIXED\n- none\n\n"
    "REFACTORED\n- none\n\nREMAINING\n- none\n\nRISKS\n- none\nEND META AUDIT\n"
)


class PlannerTransportExhaustionTests(AutonomyHarness):
    """The planner checkpoint is durable before the planner is ever called."""

    def plan(self) -> str:
        return meta_plan(
            Step(id="S01", title="Write the feature", write=("feature.txt",)),
        )

    def test_planner_transport_exhaustion_is_resumable(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: AUDIT_DONE)
        planner = ScriptedChat(
            [
                LLMTransportExhaustedError("LLM transport horizon exhausted"), self.plan(),
                continuation_answer("COMPLETE"),
            ],
            name="planner",
        )
        config = self.config()

        failed = Orchestrator(
            config, planner_client=planner,
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
            config, planner_client=planner,
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        # The exhausted call bought nothing durable: the resume pays for one
        # fresh planner answer and the pipeline continues past the planner.
        self.assertEqual(len(planner.requests), 3)


class _CliClock:
    """A fake sleeper for the ``--auto-resume`` loop."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

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

    def scripted_clients(self, planner: ScriptedChat) -> Any:
        """Replace the production chat client with the scripted double.

        The replacement keeps the constructor surface ``chat_client`` probes,
        so the planner profile builds the scripted client instead of a real
        socket.
        """

        def factory(endpoint, environment=None, on_transport=None):
            if endpoint.model == "fake-planner":
                return planner
            raise AssertionError(f"unexpected chat profile: {endpoint.model}")

        return mock.patch(
            "metaharness.orchestration.shared.OpenAIChatTextClient", factory
        )

    def run_with_auto_resume(
        self, planner: ScriptedChat, *,
        interval: str = "0.01", run_id: str = "run",
    ) -> tuple[int, list[float]]:
        clock = _CliClock()
        argv = [
            "run", "--config", str(self.config_path), "--spec", str(self.spec_path),
            "--run-id", run_id, "--auto-resume", "--auto-resume-interval", interval,
        ]
        with (
            mock.patch("metaharness.cli.time", clock),
            self.scripted_clients(planner),
        ):
            code = cli_main(argv)
        return code, clock.sleeps

    def test_auto_resume_retries_wait_external_run(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: AUDIT_DONE)
        planner = ScriptedChat(
            [
                LLMTransportExhaustedError("LLM transport horizon exhausted"), self.plan(),
                continuation_answer("COMPLETE"),
            ],
            name="planner",
        )
        self.config()

        code, sleeps = self.run_with_auto_resume(planner)

        self.assertEqual(code, 0, self.state().get("failure"))
        self.assertEqual(self.state()["status"], RunStatus.PUBLISHED.value)
        self.assertEqual(sleeps, [0.01])
        # One exhausted attempt, then the resumed attempt that planned the run.
        self.assertEqual(len(planner.requests), 3)

    def test_auto_resume_stops_when_run_is_not_wait_external(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: AUDIT_DONE)
        planner = ScriptedChat([self.plan(), continuation_answer("COMPLETE")], name="planner")
        self.config()

        code, sleeps = self.run_with_auto_resume(planner)

        self.assertEqual(code, 0, self.state().get("failure"))
        # A run that never waits externally is never slept on, never resumed.
        self.assertEqual(sleeps, [])
        self.assertEqual(len(planner.requests), 2)

    def test_auto_resume_ignores_transport_horizon(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: AUDIT_DONE)
        planner = ScriptedChat(
            [
                LLMTransportExhaustedError("first transport horizon"),
                LLMTransportExhaustedError("second transport horizon"),
                self.plan(),
                continuation_answer("COMPLETE"),
            ],
            name="planner",
        )
        self.config()

        code, sleeps = self.run_with_auto_resume(planner, interval="1801")

        self.assertEqual(code, 0, self.state().get("failure"))
        self.assertEqual(self.state()["status"], RunStatus.PUBLISHED.value)
        self.assertEqual(sleeps, [1801.0, 1801.0])

    def test_auto_resume_stops_on_runtime_wall_clock_partial(self) -> None:
        config = self.config()
        waiting = RunResult(Path("/runs/run"), RunStatus.WAITING_EXTERNAL, {})
        partial = RunResult(
            Path("/runs/run"), RunStatus.PARTIAL, {"failure": {"reason": "wall_clock"}},
        )
        with (
            mock.patch("metaharness.cli.time.sleep") as sleep,
            mock.patch("metaharness.cli.resume_run", return_value=partial) as resume,
        ):
            code = _auto_resume(config, waiting, 600.0)

        self.assertEqual(code, 0)
        sleep.assert_called_once_with(600.0)
        resume.assert_called_once_with(config, "run")

    def test_auto_resume_never_retries_human_failure_or_terminal_statuses(self) -> None:
        config = self.config()
        statuses = (
            RunStatus.WAITING_HUMAN,
            RunStatus.FAILED,
            RunStatus.PUBLISHED,
            RunStatus.COMMITTED,
            RunStatus.PARTIAL,
        )
        with (
            mock.patch("metaharness.cli.time.sleep") as sleep,
            mock.patch("metaharness.cli.resume_run") as resume,
        ):
            for status in statuses:
                with self.subTest(status=status):
                    result = RunResult(Path("/runs/run"), status, {})
                    _auto_resume(config, result, 600.0)

        sleep.assert_not_called()
        resume.assert_not_called()

    def test_planner_invalid_output_is_fixable_not_human(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: AUDIT_DONE)
        planner = ScriptedChat([
            PlanParseError("not a plan"), self.plan(), continuation_answer("COMPLETE"),
        ], name="planner")
        self.config()

        code, sleeps = self.run_with_auto_resume(planner)

        # An invalid planner answer is an ordinary model error: the run is
        # never handed to an operator, it is resumed and planned again.
        self.assertEqual(code, 0, self.state().get("failure"))
        self.assertEqual(self.state()["status"], RunStatus.PUBLISHED.value)
        self.assertEqual(sleeps, [0.01])
        self.assertEqual(len(planner.requests), 3)

if __name__ == "__main__":
    unittest.main()
