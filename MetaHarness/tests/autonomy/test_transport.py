"""Scenario 01 — a transient planner transport outage must not end the run.

The fake transport answers HTTP 503 for the whole outage window and a valid
planner answer afterwards; the injected clock makes a 90-second outage cost no
wall time.  No socket is opened: the configured endpoint is only a label.
"""

from __future__ import annotations

import dataclasses
import unittest

from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.resume import resume_info

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

    def test_an_outage_past_the_horizon_waits_externally_and_resumes(self) -> None:
        """An outage longer than the horizon is a resumable external wait.

        The transport itself cannot outlast this outage, so the run must stop
        at the durable planner checkpoint as ``WAIT_EXTERNAL``, and a later
        resume -- with the provider back -- must buy the plan the first phase
        never carried and drive the pipeline to its delivered candidate.
        """

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        horizon = 30
        endpoint = dataclasses.replace(chat_endpoint(), max_wait_seconds=horizon)
        clock = FakeClock()
        config = self.config()

        failed = Orchestrator(
            config,
            planner_client=CompletionOverTransport(
                endpoint,
                FlakyHTTPTransport(clock, outage_seconds=horizon + 60.0, answer=self.plan()),
                clock,
            ),
            reviewer_client=ScriptedChat([review()]),
        ).run_text(SPEC, run_id="run")

        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.failure_reason(), "LLM_TRANSPORT_EXHAUSTED")
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)

        recovered = FlakyHTTPTransport(clock, outage_seconds=0.0, answer=self.plan())
        resumed = Orchestrator(
            config,
            planner_client=CompletionOverTransport(endpoint, recovered, clock),
            reviewer_client=ScriptedChat([review()]),
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The plan came from the provider once it was reachable again: the
        # exhausted phase never persisted an answer to replay.
        self.assertGreaterEqual(recovered.attempts, 1)

    def test_a_503_burst_on_the_planner_does_not_end_the_run(self) -> None:
        """The target behaviour: a planner outage is absorbed, the run completes.

        The transport double is the planner client itself: only the
        transport's own time horizon can survive this outage, and the run
        never needs a resume, because the completion never fails.
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
