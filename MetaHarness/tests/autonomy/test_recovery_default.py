"""Scenario 11 — an unknown failure code is an ordinary fixable failure.

C1 inverts the stop policy: ``HARD_STOP`` becomes a closed allowlist of
``FATAL`` codes, and every other code, including one this version of the
harness has never seen, gets an autonomous recovery instead.  The scenario
injects ``SOME_FUTURE_FIXABLE_FAILURE`` into the fake pipeline.  A step whose
whole ladder is spent is rolled back and marked failed; its dependents are
skipped and every independent step still runs.
"""

from __future__ import annotations

from unittest import mock

from metaharness.agent import AgentRunRequest, AgentRunResult
from metaharness.gitops import candidate_tree_sha
from metaharness.models import ExecutionRole, RunStatus
from metaharness.recovery_policy import classify_failure

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import DRIVER, Script, git, review, write

UNKNOWN_CODE = "SOME_FUTURE_FIXABLE_FAILURE"


def failed_attempt(code: str, *, dirty: str | None = None) -> Script:
    """A scripted worker whose attempt fails with a stable, unknown code.

    With ``dirty`` the worker first leaves a partial edit of that path.
    """

    def action(request: AgentRunRequest) -> AgentRunResult:
        before = candidate_tree_sha(request.worktree)
        if dirty is not None:
            (request.worktree / dirty).write_text("partial\n", encoding="utf-8")
        return AgentRunResult(
            status="failed", exit_reason=code, tree_before=before,
            tree_after=candidate_tree_sha(request.worktree),
            usage=None, external_session_id=None, report_path=None, exit_code=1,
            final_message=f"{code}: scripted worker failure", driver=DRIVER,
        )

    return action


def feature_step(**overrides: object) -> Step:
    return Step(**{"id": "S01", "title": "Write the feature", "write": ("feature.txt",), **overrides})


def other_step(step_id: str, **overrides: object) -> Step:
    return Step(**{
        "id": step_id, "title": f"Write other {step_id}", "read": ("other.txt",),
        "write": ("other.txt",), **overrides,
    })


class UnknownFailureCodeTests(AutonomyHarness):
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

    def test_an_unknown_failure_code_maps_to_a_non_terminal_recovery(self) -> None:
        decision = classify_failure(UNKNOWN_CODE)

        self.assertFalse(
            decision.strategy.terminal,
            f"{UNKNOWN_CODE} was classified {decision.failure_class}/{decision.strategy}",
        )


class FailedStepContinuationTests(AutonomyHarness):
    """MARK_FAILED_CONTINUE: roll back, settle the step, keep independent work."""

    def base_content(self, path: str) -> str:
        return git(self.repo, "show", f"main:{path}") + "\n"

    def spent_ladder(self) -> tuple[Script, ...]:
        # The first attempt and both targeted retries fail the same way.
        return tuple(failed_attempt(UNKNOWN_CODE, dirty="feature.txt") for _ in range(3))

    def test_failed_step_rolls_back_and_independent_step_continues(self) -> None:
        self.green_check()
        self.workers.on(
            ExecutionRole.IMPLEMENTER, *self.spent_ladder(), write("other.txt", "done\n"),
        )
        plan = meta_plan(feature_step(), other_step("S02"))

        result = self.orchestrator(
            self.config(max_step_contract_repairs=0), planner=[plan], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_unrecoverable_hard_stop(result)
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)
        failed = self.step_record("S01")
        self.assertEqual(failed["status"], "FAILED_CONTINUED")
        self.assertEqual(failed["reason"], UNKNOWN_CODE)
        self.assertEqual(failed["tree_restored"], failed["tree_before"])
        self.assertIn("retry_targeted", failed["strategies"])
        self.assertEqual(failed["strategies"][-1], "mark_failed_continue")
        self.assertEqual(self.step_record("S02")["status"], "COMPLETED")
        # The partial edit is gone; the independent step's work is kept.
        worktree = self.worktree()
        self.assertEqual(
            (worktree / "feature.txt").read_text(encoding="utf-8"), self.base_content("feature.txt"),
        )
        self.assertEqual((worktree / "other.txt").read_text(encoding="utf-8"), "done\n")
        self.assertEqual(self.workers.roles(), ["implementer"] * 4)
        # The incomplete result goes through the existing review, deficit shown.
        self.assertIn('"status": "FAILED_CONTINUED"', self.reviewer.requests[-1])

    def test_failed_step_skips_transitive_dependents(self) -> None:
        self.green_check()
        self.workers.on(
            ExecutionRole.IMPLEMENTER, *self.spent_ladder(), write("other.txt", "done\n"),
        )
        plan = meta_plan(
            feature_step(),
            Step(id="S02", title="Extend the feature", write=("feature.txt",), depends_on="S01"),
            Step(id="S03", title="Finish the feature", write=("feature.txt",), depends_on="S02"),
            other_step("S04"),
        )

        result = self.orchestrator(
            self.config(max_step_contract_repairs=0), planner=[plan], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_unrecoverable_hard_stop(result)
        self.assertEqual(self.step_record("S01")["status"], "FAILED_CONTINUED")
        self.assertEqual(
            (self.step_record("S02")["status"], self.step_record("S02")["depends_on"]),
            ("SKIPPED_DEPENDENCY", "S01"),
        )
        self.assertEqual(
            (self.step_record("S03")["status"], self.step_record("S03")["depends_on"]),
            ("SKIPPED_DEPENDENCY", "S02"),
        )
        self.assertEqual(self.step_record("S04")["status"], "COMPLETED")
        # No dependent was ever handed to a worker.
        self.assertEqual(self.workers.roles(), ["implementer"] * 4)

    def test_rollback_failure_is_fatal(self) -> None:
        self.green_check()
        # An out-of-scope partial edit is left for MARK_FAILED_CONTINUE to undo.
        self.workers.on(
            ExecutionRole.IMPLEMENTER, failed_attempt(UNKNOWN_CODE, dirty="other.txt"),
        )
        plan = meta_plan(feature_step(), other_step("S02"))

        with mock.patch(
            "metaharness.attempt_transaction.restore_paths_from_tree", lambda *_args: None,
        ):
            result = self.orchestrator(
                self.config(max_step_contract_repairs=0), planner=[plan], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.failure_reason(), "ROLLBACK_FAILED")
        self.assertEqual(self.workers.roles(), ["implementer"])

