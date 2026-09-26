"""Scenario 11 — an unknown failure code is an ordinary fixable failure.

C1 inverts the stop policy: ``HARD_STOP`` becomes a closed allowlist of
``FATAL`` codes, and every other code, including one this version of the
harness has never seen, gets an autonomous recovery instead.  The scenario
injects ``SOME_FUTURE_FIXABLE_FAILURE`` into the fake pipeline.
"""

from __future__ import annotations

import unittest

from metaharness.agent import AgentRunRequest, AgentRunResult
from metaharness.gitops import candidate_tree_sha
from metaharness.models import ExecutionRole
from metaharness.recovery_policy import classify_failure

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import DRIVER, Script, review, write

UNKNOWN_CODE = "SOME_FUTURE_FIXABLE_FAILURE"


def failed_attempt(code: str) -> Script:
    """A scripted worker whose attempt fails with a stable, unknown code."""

    def action(request: AgentRunRequest) -> AgentRunResult:
        tree = candidate_tree_sha(request.worktree)
        return AgentRunResult(
            status="failed", exit_reason=code, tree_before=tree, tree_after=tree,
            usage=None, external_session_id=None, report_path=None, exit_code=1,
            final_message=f"{code}: scripted worker failure", driver=DRIVER,
        )

    return action


class UnknownFailureCodeTests(AutonomyHarness):
    @unittest.expectedFailure
    def test_an_unknown_worker_failure_does_not_hard_stop_the_run(self) -> None:
        self.green_check()
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            failed_attempt(UNKNOWN_CODE), write("feature.txt", "good\n"),
        )
        plan = meta_plan(Step(id="S01", title="Write the feature", write=("feature.txt",)))

        result = self.orchestrator(
            self.config(), planner=[plan], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_not_false_human_stop(result)
        self.assert_run_completed(result)
        # The run retried the step on its own instead of asking an operator.
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    @unittest.expectedFailure
    def test_an_unknown_failure_code_maps_to_a_non_terminal_recovery(self) -> None:
        decision = classify_failure(UNKNOWN_CODE)

        self.assertFalse(
            decision.strategy.terminal,
            f"{UNKNOWN_CODE} was classified {decision.failure_class}/{decision.strategy}",
        )
