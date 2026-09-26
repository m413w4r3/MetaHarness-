"""The autonomy contract of the recovery ladder, proven table-driven.

Each row of :data:`AUTONOMY_ROUTES` states one route: the failure code as the
pipeline observes it, the deterministic class it belongs to, the strategy the
ladder exposes *at that moment*, and the durable status the exhausted bounded
loop lands on.  The same table drives the pure policy and the projections the
runtime uses, so a route that silently collapses onto an operator wait fails
here instead of only in a long journey test.
"""

from __future__ import annotations

import unittest

from dataclasses import dataclass
from typing import Mapping

from metaharness.models import ExecutionRole, RunDisposition, RunPhase, RunStatus
from metaharness.orchestration.recovery import project_exit, terminal_state_for
from metaharness.recovery_policy import (
    FailureClass,
    RecoveryFacts,
    RecoveryProgression,
    RecoveryStrategy,
    admitted_strategies,
    classify_failure,
    failure_class_for,
    recovery_ladder,
)

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    initial_plan,
    ladder_strategies,
    review,
    write,
)


@dataclass(frozen=True)
class Route:
    """One row of the autonomy contract."""

    label: str
    code: str
    failure_class: FailureClass
    # The ladder strategy the policy exposes the moment the failure occurs.
    strategy: RecoveryStrategy
    # The durable reason the bounded loop records once its budget is exhausted.
    consumed_reason: str
    # The terminal strategy that exhausted loop projects onto.
    exhausted: RecoveryStrategy
    exhausted_status: RunStatus
    phase: RunPhase
    resumable: bool


# The only postures an exhausted bounded loop may leave behind.
EXHAUSTED_RUN_DISPOSITIONS: Mapping[RecoveryStrategy, RunDisposition] = {
    RecoveryStrategy.WAIT_HUMAN: RunDisposition.WAIT_HUMAN,
    RecoveryStrategy.WAIT_EXTERNAL: RunDisposition.WAIT_EXTERNAL,
    RecoveryStrategy.HARD_STOP: RunDisposition.FAILED,
}


AUTONOMY_ROUTES: tuple[Route, ...] = (
    Route(
        "correctness failure", "CHECK_FAILED:unit", FailureClass.CORRECTNESS,
        RecoveryStrategy.REPAIR_TARGETED,
        "CHECK_REPAIR_EXHAUSTED", RecoveryStrategy.WAIT_HUMAN,
        RunStatus.WAITING_CHECK_REPAIR, RunPhase.DETERMINISTIC_GATE, True,
    ),
    Route(
        "contract failure", "AGENT_CONTRACT_MISMATCH", FailureClass.CONTRACT,
        RecoveryStrategy.REPLAN_STEP,
        "AGENT_CONTRACT_MISMATCH", RecoveryStrategy.WAIT_HUMAN,
        RunStatus.WAITING_HUMAN, RunPhase.IMPLEMENT_STEP, False,
    ),
    Route(
        "model protocol failure", "PLANNER_FORMAT_INVALID", FailureClass.MODEL_PROTOCOL,
        RecoveryStrategy.RETRY_TARGETED,
        "PLANNER_FORMAT_INVALID", RecoveryStrategy.WAIT_HUMAN,
        RunStatus.WAITING_HUMAN, RunPhase.PLANNER, False,
    ),
    Route(
        "model protocol failure (review answer)", "REVIEW_FORMAT_INVALID",
        FailureClass.MODEL_PROTOCOL, RecoveryStrategy.REPAIR_TARGETED,
        "REVIEW_FORMAT_INVALID", RecoveryStrategy.WAIT_HUMAN,
        RunStatus.WAITING_HUMAN, RunPhase.FINAL_REVIEW, False,
    ),
    Route(
        "external unavailability", "AGENT_TIMEOUT", FailureClass.EXTERNAL,
        RecoveryStrategy.RETRY_TARGETED,
        "AGENT_TIMEOUT", RecoveryStrategy.WAIT_EXTERNAL,
        RunStatus.WAITING_EXTERNAL, RunPhase.IMPLEMENT_STEP, True,
    ),
    Route(
        "external unavailability (check infrastructure)", "DOCKER_DAEMON_UNAVAILABLE",
        FailureClass.EXTERNAL, RecoveryStrategy.RETRY_TARGETED,
        "CHECK_INFRA_RETRIES_EXHAUSTED", RecoveryStrategy.WAIT_EXTERNAL,
        RunStatus.WAITING_CHECK_INFRASTRUCTURE, RunPhase.DETERMINISTIC_GATE, True,
    ),
    Route(
        "spec ambiguity", "SPEC_DECISION_REQUIRED", FailureClass.SPEC_DECISION,
        RecoveryStrategy.WAIT_HUMAN,
        "SPEC_DECISION_REQUIRED", RecoveryStrategy.WAIT_HUMAN,
        RunStatus.WAITING_HUMAN, RunPhase.PLANNER, False,
    ),
)

# Entry codes: the code a live loop emits when the failure occurs, never a
# `*_EXHAUSTED` sentinel that already names the end of a bounded loop.
AUTONOMOUS_ENTRY_CODES: Mapping[FailureClass, tuple[str, ...]] = {
    FailureClass.CORRECTNESS: (
        "CHECK_FAILED:unit", "CHECK_FAILED:integration", "REVIEW_IMPLEMENTATION",
        "BOUNDED_SCOPE_REQUEST",
    ),
    FailureClass.CONTRACT: (
        "AGENT_CONTRACT_MISMATCH", "CONTRACT_INSUFFICIENT", "CONTRACT_INSUFFICIENCY",
        "PLAN_REPOSITORY_PRECONDITION_INVALID",
    ),
    FailureClass.MODEL_PROTOCOL: (
        "PLANNER_FORMAT_INVALID", "PLANNER_OUTPUT_INVALID", "PLANNER_PROTOCOL_FAILED",
        "REVIEW_FORMAT_INVALID", "REVIEWER_OUTPUT_INVALID",
        "STEP_CONTRACT_REPAIR_OUTPUT_INVALID",
    ),
}

EXTERNAL_ENTRY_CODES = (
    "AGENT_TIMEOUT", "AGENT_RUNTIME_FAILED", "LLM_429", "LLM_5XX", "LLM_TIMEOUT",
    "DOCKER_DAEMON_UNAVAILABLE", "CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED",
    "REVIEWER_TRANSPORT_FAILURE", "WORKSPACE_SETUP_TIMEOUT",
)

BOUNDARY_CODES = (
    "SECRET_IN_DIFF", "SECURITY_VIOLATION", "CHECK_AUTHORITY_TAMPERING",
    "TREE_MISMATCH", "DURABLE_ARTIFACT_CORRUPTED", "AGENT_SCOPE_VIOLATION",
)

CANDIDATE_TREE = "a" * 40
OTHER_TREE = "b" * 40
FAILED_CHECK_FACTS = (("failed_checks", "unit|integration"),)


class AutonomyRouteTests(unittest.TestCase):
    """One table, both the pure policy and the durable projections."""

    def test_every_row_keeps_its_immediate_route(self) -> None:
        for route in AUTONOMY_ROUTES:
            with self.subTest(route=route.label):
                self.assertIs(failure_class_for(route.code), route.failure_class)
                decision = classify_failure(route.code)
                self.assertIs(decision.strategy, route.strategy)
                if route.strategy.terminal:
                    # Only an operator decision waits the moment it occurs.
                    self.assertIs(route.strategy, RecoveryStrategy.WAIT_HUMAN)
                else:
                    self.assertFalse(
                        decision.strategy.terminal,
                        f"{route.label} stops at {decision.strategy.value} the moment it "
                        f"occurs instead of consuming an autonomous step",
                    )
                    self.assertIn(decision.strategy, recovery_ladder(route.failure_class))

    def test_every_row_lands_on_its_durable_status_once_exhausted(self) -> None:
        for route in AUTONOMY_ROUTES:
            with self.subTest(route=route.label):
                decision = classify_failure(route.consumed_reason, budget_exhausted=True)
                projected, terminal = project_exit(route.consumed_reason, phase=route.phase)
                self.assertTrue(projected.strategy.terminal)
                self.assertIs(
                    projected.strategy, route.exhausted,
                    f"{route.label}: {decision.strategy.value} -> {projected.strategy.value}",
                )
                self.assertIs(
                    terminal.disposition, EXHAUSTED_RUN_DISPOSITIONS[route.exhausted],
                )
                self.assertEqual(terminal.status, route.exhausted_status)
                self.assertEqual(terminal.resumable, route.resumable)

    def test_no_autonomous_route_waits_for_a_human_while_a_rung_remains(self) -> None:
        """The contract metric: zero immediate human routes."""

        immediate_human: list[tuple[str, list[str]]] = []
        for failure_class, codes in AUTONOMOUS_ENTRY_CODES.items():
            for code in codes:
                self.assertIs(failure_class_for(code), failure_class)
                decision = classify_failure(code)
                facts = RecoveryFacts.for_failure(code)
                remaining = [
                    step.value
                    for step in admitted_strategies(failure_class, facts)
                    if not step.terminal
                ]
                if decision.strategy is RecoveryStrategy.WAIT_HUMAN:
                    immediate_human.append((code, remaining))
                else:
                    self.assertIn(decision.strategy, recovery_ladder(failure_class))
        self.assertEqual(
            immediate_human, [],
            "a correctness, contract or model-protocol failure waited for a human while "
            "the ladder still had an autonomous rung",
        )

    def test_an_unavailable_external_waits_externally_after_its_retries(self) -> None:
        for code in EXTERNAL_ENTRY_CODES:
            with self.subTest(code=code):
                self.assertIs(failure_class_for(code), FailureClass.EXTERNAL)
                immediate = classify_failure(code)
                self.assertFalse(immediate.strategy.terminal, immediate.reason)
                exhausted = classify_failure(code, budget_exhausted=True)
                self.assertIs(exhausted.strategy, RecoveryStrategy.WAIT_EXTERNAL)
                for phase in (RunPhase.IMPLEMENT_STEP, RunPhase.DETERMINISTIC_GATE):
                    terminal = terminal_state_for(exhausted, failure_code=code, phase=phase)
                    self.assertIs(terminal.disposition, RunDisposition.WAIT_EXTERNAL)
                    self.assertIn(
                        terminal.status,
                        {RunStatus.WAITING_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE},
                    )
                    self.assertTrue(terminal.resumable)

    def test_an_operator_decision_is_the_only_immediate_human_wait(self) -> None:
        decision = classify_failure("SPEC_DECISION_REQUIRED")
        self.assertIs(decision.failure_class, FailureClass.SPEC_DECISION)
        self.assertIs(decision.strategy, RecoveryStrategy.WAIT_HUMAN)
        self.assertEqual(
            recovery_ladder(FailureClass.SPEC_DECISION), (RecoveryStrategy.WAIT_HUMAN,),
        )
        terminal = terminal_state_for(
            decision, failure_code="SPEC_DECISION_REQUIRED", phase=RunPhase.PLANNER,
        )
        self.assertEqual(
            (terminal.disposition, terminal.status, terminal.resumable),
            (RunDisposition.WAIT_HUMAN, RunStatus.WAITING_HUMAN, False),
        )

    def test_a_security_or_integrity_boundary_fails_closed(self) -> None:
        for code in BOUNDARY_CODES:
            with self.subTest(code=code):
                self.assertIn(
                    failure_class_for(code),
                    {FailureClass.SECURITY, FailureClass.INTEGRITY, FailureClass.AUTHORITY},
                )
                decision = classify_failure(code)
                self.assertIs(decision.strategy, RecoveryStrategy.HARD_STOP, decision.reason)
                self.assertTrue(all(step.terminal for step in recovery_ladder(decision.failure_class)))
                terminal = terminal_state_for(
                    decision, failure_code=code, phase=RunPhase.DETERMINISTIC_GATE,
                )
                self.assertIs(terminal.disposition, RunDisposition.FAILED)
                self.assertEqual(terminal.status, RunStatus.FAILED)
                self.assertFalse(terminal.resumable)


class AntiLoopProgressionTests(unittest.TestCase):
    """The same fingerprint is never consumed twice; a new tree reopens it."""

    def _facts(
        self, tree: str = CANDIDATE_TREE, observed: tuple[tuple[str, str], ...] = (),
    ) -> RecoveryFacts:
        return RecoveryFacts(candidate_tree=tree, observed_facts=observed)

    def test_the_same_fingerprint_and_strategy_are_never_consumed_twice(self) -> None:
        progression = RecoveryProgression()
        facts = self._facts()
        fingerprint = progression.consume(
            candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
            facts=facts, strategy=RecoveryStrategy.REPAIR_TARGETED,
        )
        self.assertTrue(progression.is_consumed(fingerprint))
        with self.assertRaises(ValueError):
            progression.consume(
                candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=RecoveryStrategy.REPAIR_TARGETED,
            )
        self.assertIs(
            progression.next_strategy(
                candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
                facts=facts, code="CHECK_FAILED:unit",
            ),
            RecoveryStrategy.REPLAN_STEP,
            "the next unconsumed rung of the correctness ladder",
        )

    def test_a_new_candidate_tree_opens_a_new_progression(self) -> None:
        progression = RecoveryProgression()
        progression.consume(
            candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
            facts=self._facts(), strategy=RecoveryStrategy.REPAIR_TARGETED,
        )
        self.assertIs(
            progression.next_strategy(
                candidate_tree=OTHER_TREE, failure_class=FailureClass.CORRECTNESS,
                facts=self._facts(tree=OTHER_TREE), code="CHECK_FAILED:unit",
            ),
            RecoveryStrategy.REPAIR_TARGETED,
        )

    def test_new_stable_failure_facts_open_a_new_progression(self) -> None:
        progression = RecoveryProgression()
        progression.consume(
            candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
            facts=self._facts(), strategy=RecoveryStrategy.REPAIR_TARGETED,
        )
        self.assertIs(
            progression.next_strategy(
                candidate_tree=CANDIDATE_TREE, failure_class=FailureClass.CORRECTNESS,
                facts=self._facts(observed=FAILED_CHECK_FACTS), code="CHECK_FAILED:unit",
            ),
            RecoveryStrategy.REPAIR_TARGETED,
        )


class CorrectnessRouteIsAutonomousTests(PipelineHarness):
    """The durable route: a red gate consumes its ladder before any human wait."""

    def test_a_red_gate_is_repaired_autonomously_without_a_human_wait(self) -> None:
        counter = self.root / "gate-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "raise SystemExit(1 if count < 1 else 0)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))

        result = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertGreaterEqual(int(counter.read_text()), 2, "the gate was never red")
        self.assertEqual(
            ladder_strategies(self), ["repair_targeted"],
            "the red gate consumed its autonomous rung before anything else",
        )
        self.assertEqual(
            human_terminals(self), [],
            "a recovery loop routed this run to a human wait",
        )


def human_terminals(harness: PipelineHarness) -> list[tuple[str, str]]:
    """(reason, terminal status) of every exhausted recovery loop, from the trace."""

    routes: list[tuple[str, str]] = []
    for event in harness.trace_events():
        if event.get("event") != "recovery.exhausted":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if str(data.get("terminal_status")) == RunStatus.WAITING_HUMAN.value:
            routes.append((str(data.get("reason")), str(data.get("terminal_status"))))
    return routes


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
