"""Scenario 01 — a transient planner transport outage must not end the run.

The fake transport answers HTTP 503 for the whole outage window and a valid
planner answer afterwards; the injected clock makes a 90-second outage cost no
wall time.  No socket is opened: the configured endpoint is only a label.
"""

from __future__ import annotations

import unittest

from metaharness.models import ExecutionRole
from metaharness.orchestrator import Orchestrator

from tests.autonomy.support import (
    SPEC,
    AutonomyHarness,
    CompletionOverTransport,
    FakeClock,
    FlakyHTTPTransport,
    Step,
    chat_endpoint,
    meta_plan,
)
from tests.pipeline_support import ScriptedChat, review, write


class PlannerTransportOutageTests(AutonomyHarness):
    """The planner's transport absorbs a bounded outage and the run progresses."""

    OUTAGE_SECONDS = 90.0

    def plan(self) -> str:
        return meta_plan(
            Step(id="S01", title="Write the feature", write=("feature.txt",)),
        )

    def test_control_a_healthy_transport_completes_the_run(self) -> None:
        """Control: the transport double itself is sound, with no outage at all."""

        clock = FakeClock()
        transport = FlakyHTTPTransport(clock, outage_seconds=0.0, answer=self.plan())
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        result = Orchestrator(
            self.config(),
            planner_client=CompletionOverTransport(chat_endpoint(), transport, clock),
            reviewer_client=ScriptedChat([review()]),
        ).run_text(SPEC, run_id="run")

        self.assert_run_completed(result)
        self.assertEqual(transport.attempts, 1)

    @unittest.expectedFailure
    def test_the_transport_outlasts_a_ninety_second_outage(self) -> None:
        """The C4 horizon: retries continue while the injected clock runs."""

        clock = FakeClock()
        transport = FlakyHTTPTransport(
            clock, outage_seconds=self.OUTAGE_SECONDS, answer=self.plan(),
        )
        chat = CompletionOverTransport(chat_endpoint(), transport, clock)

        completion = chat.complete(SPEC)

        self.assertEqual(completion.text, self.plan())
        self.assertGreaterEqual(transport.attempts, 2)
        self.assertGreaterEqual(clock.elapsed, self.OUTAGE_SECONDS)

    @unittest.expectedFailure
    def test_a_503_burst_on_the_planner_does_not_end_the_run(self) -> None:
        """The target behaviour: a planner outage is absorbed, the run completes.

        The fixture profiles cap retries at zero, exactly as the live profiles
        do; only a time-based transport horizon can survive this outage, which
        is what the C4 transport policy adds.
        """

        clock = FakeClock()
        transport = FlakyHTTPTransport(
            clock, outage_seconds=self.OUTAGE_SECONDS, answer=self.plan(),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        result = Orchestrator(
            self.config(),
            planner_client=CompletionOverTransport(chat_endpoint(), transport, clock),
            reviewer_client=ScriptedChat([review()]),
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
