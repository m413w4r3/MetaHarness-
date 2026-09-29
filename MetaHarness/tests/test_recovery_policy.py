"""The v4 recovery policy: progress by default, stop only on a proven boundary."""

from __future__ import annotations

import ast
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import metaharness
from metaharness.models import RunDisposition, RunStatus
from metaharness.orchestration.recovery import RecoveryCoordinator, project_exit
from metaharness.recovery_policy import (
    FAILURE_CLASSES,
    RECOVERY_LADDERS,
    AutonomyBudget,
    FailureClass,
    RecoveryDecision,
    RecoveryStrategy,
    classify_failure,
    elapsed_hours,
    recovery_ladder,
    stable_code,
    wall_clock_exhausted,
)
from metaharness.resume import ResumePhase
from metaharness.state import RunStateStore

# Constructors whose first argument is a stable failure code.
_CODE_CONSTRUCTORS = frozenset({
    "PipelineFailure", "StepExecutionFailure", "AttemptViolation", "record_failure",
})
_SOURCE = Path(metaharness.__file__).parent


def literal_failure_codes(source: Path = _SOURCE) -> dict[str, str]:
    """Every literal code the source raises, mapped to one place it is raised.

    A module-level string constant passed to a code constructor counts as the
    literal it names; a dynamic expression is the runtime's business.
    """

    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in source.rglob("*.py")}
    constants: dict[str, set[str]] = {}
    for tree in trees.values():
        for node in tree.body:
            if (
                isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            ):
                constants.setdefault(node.targets[0].id, set()).add(node.value.value)
    codes: dict[str, str] = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name not in _CODE_CONSTRUCTORS:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                values = {arg.value}
            elif isinstance(arg, ast.Name):
                values = constants.get(arg.id, set())
            else:
                continue
            for value in values:
                codes.setdefault(stable_code(value), f"{path.relative_to(source)}:{node.lineno}")
    return codes


class FailureClassificationTests(unittest.TestCase):
    def test_unknown_codes_do_not_inherit_a_namespace_policy(self) -> None:
        for code in (
            "RESUME_TEMPORARY_FAILURE", "ROLLBACK_RETRY_PENDING",
            "STAGED_BLOB_RETRY_PENDING", "WORKER_SECURITY_FAILURE",
            "CUSTOM_OUTSIDE_AUTHORITY", "CUSTOM_AUTHORITY_MISMATCH",
            "AGENT_FUTURE_ERROR", "CHECK_FUTURE_ERROR",
        ):
            with self.subTest(code=code):
                decision = classify_failure(code, exhausted=True)
                self.assertFalse(decision.known)
                self.assertIs(decision.failure_class, FailureClass.FIXABLE)
                self.assertIs(decision.strategy, RecoveryStrategy.MARK_FAILED_CONTINUE)

    def test_recorded_v4_fatal_boundaries_remain_fatal(self) -> None:
        for code in (
            "UNSCANNABLE_STAGED_BLOB", "STAGED_BLOB_SCAN_FAILED", "UNREVIEWABLE_TEXT_DIFF",
            "HEAD_MODIFIED_OUTSIDE_AUTHORITY", "BRANCH_MODIFIED_OUTSIDE_AUTHORITY",
            "REMOTE_AUTHORITY_MISMATCH", "ROLLBACK_TREE_MISMATCH",
        ):
            with self.subTest(code=code):
                self.assertIs(classify_failure(code).strategy, RecoveryStrategy.HARD_STOP)

    def test_unknown_failure_defaults_to_fixable(self) -> None:
        decision = classify_failure("SOME_FUTURE_FAILURE")

        self.assertIs(decision.failure_class, FailureClass.FIXABLE)
        self.assertIs(decision.strategy, RecoveryStrategy.RETRY_TARGETED)
        self.assertFalse(decision.strategy.terminal)
        self.assertFalse(decision.known)
        # A suffix never changes the class of its stable code.
        self.assertIs(classify_failure("CHECK_FAILED:unit").failure_class, FailureClass.FIXABLE)

    def test_only_fatal_class_can_hard_stop(self) -> None:
        for failure_class in FailureClass:
            with self.subTest(failure_class=failure_class):
                self.assertEqual(
                    RecoveryStrategy.HARD_STOP in recovery_ladder(failure_class),
                    failure_class is FailureClass.FATAL,
                )
        for code in ("COMMIT_SECURITY_FAILURE:secret_in_diff", "SOURCE_STAGED_BLOB_NOT_REVIEWABLE", "ROLLBACK_FAILED"):
            with self.subTest(code=code):
                self.assertIs(classify_failure(code).strategy, RecoveryStrategy.HARD_STOP)
        # An ordinary model mistake is never fatal.
        for code in (
            "AGENT_SCOPE_VIOLATION", "AGENT_GIT_VIOLATION", "CONTRACT_INSUFFICIENT",
            "PLANNER_OUTPUT_INVALID", "CHECK_FAILED", "AUDIT_REMAINING",
        ):
            with self.subTest(code=code):
                self.assertIsNot(
                    classify_failure(code, exhausted=True).strategy, RecoveryStrategy.HARD_STOP,
                )
        with self.assertRaises(ValueError):
            RecoveryDecision(FailureClass.FIXABLE, RecoveryStrategy.HARD_STOP, "refused")

    def test_wait_human_only_for_spec_decision(self) -> None:
        for code, failure_class in FAILURE_CLASSES.items():
            for exhausted in (False, True):
                decision = classify_failure(code.replace("*", "X"), exhausted=exhausted)
                with self.subTest(code=code, exhausted=exhausted):
                    self.assertIs(decision.failure_class, failure_class)
                    self.assertEqual(
                        decision.strategy is RecoveryStrategy.WAIT_HUMAN,
                        failure_class is FailureClass.SPEC_DECISION,
                    )
        # A vague operator code is not a product decision.
        self.assertIs(classify_failure("HUMAN_REQUIRED").failure_class, FailureClass.FIXABLE)
        with self.assertRaises(ValueError):
            RecoveryDecision(FailureClass.TRANSIENT, RecoveryStrategy.WAIT_HUMAN, "refused")

    def test_fixable_ladder_ends_in_failed_continue(self) -> None:
        self.assertEqual(recovery_ladder(FailureClass.FIXABLE), (
            RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
            RecoveryStrategy.MARK_FAILED_CONTINUE,
        ))
        for code in ("AUDIT_REMAINING", "WAITING_REPAIR_EXHAUSTED", "TOTALLY_NEW"):
            with self.subTest(code=code):
                decision, terminal = project_exit(code, phase=ResumePhase.IMPLEMENT_STEP)
                self.assertIs(decision.strategy, RecoveryStrategy.MARK_FAILED_CONTINUE)
                self.assertIs(terminal.disposition, RunDisposition.WAIT_EXTERNAL)
                self.assertTrue(terminal.resumable)

    def test_transient_ladder_ends_wait_external(self) -> None:
        self.assertEqual(recovery_ladder(FailureClass.TRANSIENT), (
            RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
            RecoveryStrategy.WAIT_EXTERNAL,
        ))
        for code in ("LLM_TRANSPORT_EXHAUSTED", "AGENT_TIMEOUT", "CHECK_INFRASTRUCTURE_UNAVAILABLE"):
            with self.subTest(code=code):
                decision, terminal = project_exit(code, phase=ResumePhase.AUDIT)
                self.assertIs(decision.strategy, RecoveryStrategy.WAIT_EXTERNAL)
                self.assertTrue(terminal.resumable)
        _decision, terminal = project_exit("SPEC_DECISION_REQUIRED", phase=ResumePhase.PLANNER)
        self.assertIs(terminal.status, RunStatus.WAITING_HUMAN)
        _decision, terminal = project_exit("COMMIT_SECURITY_FAILURE:secret_in_diff", phase=ResumePhase.DETERMINISTIC_GATE)
        self.assertIs(terminal.status, RunStatus.FAILED)

    def test_all_literal_pipeline_failure_codes_are_explicitly_classified(self) -> None:
        codes = literal_failure_codes()
        self.assertIn("RESUME_INTEGRITY_FAILURE", codes)
        unclassified = {code: where for code, where in codes.items() if not classify_failure(code).known}
        self.assertEqual(unclassified, {}, "classify these codes in FAILURE_CLASSES")
        # The guard disciplines the source; the runtime stays fail-open.
        self.assertIs(classify_failure("DYNAMIC_UNKNOWN").failure_class, FailureClass.FIXABLE)

    def test_the_removed_pipeline_families_stay_out_of_the_policy(self) -> None:
        """C7.1: a family no runtime code emits is never classified here."""

        removed_families = (
            "SEMANTIC_REVISER_", "REVIEW_", "REVIEWER_", "CHECK_REPAIR_",
            "REPLAN_CYCLE", "EXPAND_SCOPE", "REPAIR_TARGETED",
        )
        for key in FAILURE_CLASSES:
            for family in removed_families:
                self.assertFalse(
                    key.startswith(family) or key == family,
                    f"FAILURE_CLASSES still classifies the removed family {family!r}",
                )
        for family in removed_families:
            self.assertNotIn(family, RecoveryStrategy.__members__)
        for ladder in RECOVERY_LADDERS.values():
            names = {strategy.value for strategy in ladder}
            self.assertFalse(names & {"replan_cycle", "expand_scope", "repair_targeted"})

    def test_codes_without_a_runtime_producer_stay_out_of_the_policy(self) -> None:
        """A code no runtime code emits is never classified here."""

        for code in (
            "ATOMIC_SCOPE_POLICY_LIMIT",
            "SECURITY_POLICY_DECISION_REQUIRED",
            "HUMAN_REQUIRED",
        ):
            with self.subTest(code=code):
                self.assertNotIn(code, FAILURE_CLASSES)
                self.assertFalse(classify_failure(code).known)

    def test_the_one_budget_has_exactly_five_bounded_fields(self) -> None:
        self.assertEqual(tuple(AutonomyBudget.__dataclass_fields__), (
            "step_attempts", "audit_repairs", "max_iterations",
            "max_wall_clock_hours", "max_cost",
        ))
        self.assertEqual(AutonomyBudget(), AutonomyBudget(
            step_attempts=3, audit_repairs=2, max_iterations=8,
            max_wall_clock_hours=12, max_cost=0,
        ))
        for overrides in (
            {"step_attempts": 0}, {"step_attempts": 11}, {"audit_repairs": -1},
            {"max_iterations": 0}, {"max_wall_clock_hours": 0},
            {"max_wall_clock_hours": 721}, {"max_cost": -1},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    AutonomyBudget(**overrides)

    def test_wall_clock_is_measured_from_the_durable_start(self) -> None:
        self.assertFalse(wall_clock_exhausted(None, max_wall_clock_hours=12))
        self.assertFalse(wall_clock_exhausted("not-a-date", max_wall_clock_hours=12))
        self.assertFalse(wall_clock_exhausted(
            "2024-01-01T00:00:00Z", max_wall_clock_hours=12,
            now=datetime(2024, 1, 1, 11, 59, tzinfo=timezone.utc),
        ))
        self.assertTrue(wall_clock_exhausted(
            "2024-01-01T00:00:00Z", max_wall_clock_hours=12,
            now=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        ))
        self.assertEqual(
            elapsed_hours(
                "2024-01-01T00:00:00+00:00",
                now=datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
            ),
            6.0,
        )


class UnclassifiedObservabilityTests(unittest.TestCase):
    def test_unclassified_failure_emits_observability_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = RunStateStore(Path(temp) / "state.json")
            store.initialize("run")
            events: list[tuple[str, dict]] = []
            coordinator = RecoveryCoordinator(
                store, emit=lambda event, **kwargs: events.append((event, kwargs)),
            )

            admission = coordinator.admit(
                "agent-step:001:S01", reason="SOME_FUTURE_FAILURE: detail", budget=2,
                phase="implementation", cycle=1, step_id="S01",
            )
            coordinator.admit(
                "agent-step:001:S01", reason="AGENT_TIMEOUT", budget=2,
                phase="implementation", cycle=1, step_id="S01",
            )

        # The unknown code still walks the ordinary ladder.
        self.assertTrue(admission.admitted)
        unclassified = [kwargs for event, kwargs in events if event == "recovery.unclassified_code"]
        # Emitted once per run by the trace stream, for the unknown code only.
        self.assertTrue(unclassified)
        self.assertEqual(
            {(item["data"]["code"], item["once"]) for item in unclassified},
            {("SOME_FUTURE_FAILURE", True)},
        )


if __name__ == "__main__":
    unittest.main()
