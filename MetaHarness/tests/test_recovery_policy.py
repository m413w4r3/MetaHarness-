"""The v4 recovery policy: progress by default, stop only on a proven boundary."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import metaharness
from metaharness.models import RunDisposition, RunStatus
from metaharness.orchestration.recovery import RecoveryCoordinator, project_exit
from metaharness.recovery_policy import (
    FAILURE_CLASSES,
    RECOVERY_LADDERS,
    FailureClass,
    RecoveryBudgets,
    RecoveryDecision,
    RecoveryStrategy,
    classify_failure,
    recovery_ladder,
    stable_code,
)
from metaharness.resume import ResumePhase
from metaharness.state import RunStateStore

# Constructors whose first argument is a stable failure code.
_CODE_CONSTRUCTORS = frozenset({
    "PipelineFailure", "StepExecutionFailure", "AttemptViolation", "record_failure",
})
_SOURCE = Path(metaharness.__file__).parent


def _literal_failure_codes() -> dict[str, str]:
    """Every literal code the source raises, mapped to one place it is raised.

    A module-level string constant passed to a code constructor counts as the
    literal it names; a dynamic expression is the runtime's business.
    """

    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in _SOURCE.rglob("*.py")}
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
                codes.setdefault(stable_code(value), f"{path.relative_to(_SOURCE)}:{node.lineno}")
    return codes


class FailureClassificationTests(unittest.TestCase):
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
        for code in ("SECRET_IN_DIFF", "SOURCE_STAGED_BLOB_NOT_REVIEWABLE", "ROLLBACK_FAILED"):
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

    def test_no_fixable_ladder_ends_in_wait_human(self) -> None:
        self.assertEqual(recovery_ladder(FailureClass.FIXABLE), (
            RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
            RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.MARK_FAILED_CONTINUE,
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
        _decision, terminal = project_exit("SECRET_IN_DIFF", phase=ResumePhase.DETERMINISTIC_GATE)
        self.assertIs(terminal.status, RunStatus.FAILED)

    def test_all_literal_pipeline_failure_codes_are_explicitly_classified(self) -> None:
        codes = _literal_failure_codes()
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

    def test_budgets_have_bounded_durable_defaults(self) -> None:
        self.assertEqual(RecoveryBudgets(), RecoveryBudgets(
            max_transient_attempts=2,
            max_executor_fallbacks=1,
            max_check_infra_retries=2,
            max_workspace_setup_retries=2,
        ))
        with self.assertRaises(ValueError):
            RecoveryBudgets(max_transient_attempts=11)


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
