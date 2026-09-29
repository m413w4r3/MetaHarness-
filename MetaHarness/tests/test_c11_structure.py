"""Runtime-measured C11 structural and autonomy invariants."""

from __future__ import annotations

import ast
import tomllib
import unittest
from dataclasses import fields
from pathlib import Path

from metaharness.models import (
    PARTIAL_REASONS,
    RunDisposition,
    RunEvent,
    RunMachineState,
    RunPhase,
    RunStatus,
    RunTransitionError,
    project_run_outcome,
    transition,
)
from metaharness.orchestration.recovery import normalize_exit_reason, project_exit, terminal_state_for
from metaharness.recovery_policy import (
    AutonomyBudget,
    RecoveryStrategy,
    classify_failure,
    stable_code,
)
from tests.test_recovery_policy import literal_failure_codes


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "metaharness"


def produced_failure_codes(source: Path = SRC) -> set[str]:
    """Canonical durable vocabulary from error, state and evidence producers.

    Provider aliases are normalized at their persistence boundary. Policy
    patterns and consumers (including UI labels and tests) are not producers.
    """
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in source.rglob("*.py")]
    constants: dict[str, set[str]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants.setdefault(target.id, set()).add(node.value.value)

    def values(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return constants.get(node.id, set())
        if isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            return values(first.value) if isinstance(first, ast.FormattedValue) else values(first)
        if isinstance(node, ast.IfExp):
            return values(node.body) | values(node.orelse)
        if isinstance(node, ast.BoolOp):
            return set().union(*(values(part) for part in node.values))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) == 3:
            return values(node.args[2])
        return set()

    def durable_code(code: str) -> str:
        normalized = normalize_exit_reason(code)
        return normalized if normalized in PARTIAL_REASONS else stable_code(normalized)

    codes = {durable_code(code) for code in literal_failure_codes(source)}
    for tree in trees:
        for node in ast.walk(tree):
            candidates: list[ast.AST] = []
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if name in {"ScopeViolation", "WorkspaceSetupError", "CandidatePushError", "GitMutationAudit", "ValidationError", "OrchestrationError"} and node.args:
                    candidates.append(node.args[0])
                if isinstance(node.func, ast.Attribute) and node.func.attr == "append" and node.args:
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "failures":
                        candidates.append(node.args[0])
                candidates.extend(kw.value for kw in node.keywords if kw.arg in {"reason", "exit_reason", "fatal_code"} or (kw.arg == "code" and name and name.endswith("Error")))
            elif isinstance(node, ast.ClassDef):
                for field in node.body:
                    if isinstance(field, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "code" for t in field.targets):
                        candidates.append(field.value)
            elif isinstance(node, ast.FunctionDef) and node.name in {"_failure_reason", "normalize_exit_reason"}:
                candidates.extend(n.value for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None)
            elif isinstance(node, ast.Assign) and any(
                (isinstance(t, ast.Name) and t.id in {"code", "exit_reason"})
                or (isinstance(t, ast.Attribute) and t.attr in {"code", "exit_reason", "fatal_code"})
                for t in node.targets
            ):
                candidates.append(node.value)
            if isinstance(node, ast.FunctionDef):
                candidates.extend(default for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if arg.arg == "code" and default is not None)
            for candidate in candidates:
                for value in values(candidate):
                    code = stable_code(value)
                    if code.isupper() and code.replace("_", "").isalnum():
                        codes.add(durable_code(code))
    return codes | set(PARTIAL_REASONS)


class C11StructureTests(unittest.TestCase):
    def test_autowork_configuration_restores_the_intended_capacity(self) -> None:
        with (ROOT / "examples" / "autowork.toml").open("rb") as stream:
            config = tomllib.load(stream)
        self.assertEqual(config["planning"]["max_steps_per_plan"], 21)
        self.assertFalse(config["approval"]["require_plan_approval"])
        self.assertEqual(config["planning"]["single_step_max_mutable_paths"], 6)
        self.assertEqual(config["planning"]["staged_step_max_mutable_paths"], 12)
        self.assertEqual(config["planning"]["max_read_paths_per_step"], 18)
        self.assertEqual(config["planning"]["max_step_contract_chars"], 8500)

    def test_runtime_budgets_are_closed(self) -> None:
        self.assertEqual(tuple(AutonomyBudget.__dataclass_fields__), (
            "step_attempts", "audit_repairs", "max_iterations",
            "max_wall_clock_hours", "max_cost",
        ))
        self.assertEqual(len(fields(AutonomyBudget)), 5)
        self.assertLessEqual(len(RunPhase), 10)
        self.assertLessEqual(len(RunStatus), 10)
        self.assertLessEqual(len(list((SRC / "orchestration").glob("*.py"))), 14)

    def test_durable_failure_code_budget(self) -> None:
        codes = produced_failure_codes()
        self.assertIn("COMMIT_SECURITY_FAILURE", codes)
        self.assertIn("COMMIT_GATE_FAILED", codes)
        self.assertLessEqual(len(codes), 40, "runtime producers exceed C11; policy patterns do not merge durable codes")

    def test_failure_budget_counts_operator_and_partial_reasons(self) -> None:
        codes = produced_failure_codes()
        self.assertTrue(set(PARTIAL_REASONS) <= codes)
        self.assertTrue({"PAUSED", "INTERRUPTED", "PLAN_REJECTED", "EXTERNAL_AUTH_REQUIRED"} <= codes)
        self.assertNotIn("AGENT_AUTH_FAILURE", codes)
        self.assertEqual(normalize_exit_reason("AGENT_AUTH_FAILURE"), "EXTERNAL_AUTH_REQUIRED")

    def test_failure_inventory_detects_a_new_runtime_producer(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "producer.py").write_text(
                'def fail(store):\n    store.record_failure("NEW_DURABLE_CODE")\n    store.record_failure("AGENT_AUTH_FAILURE")\n',
                encoding="utf-8",
            )
            codes = produced_failure_codes(source)
            self.assertIn("NEW_DURABLE_CODE", codes)
            self.assertIn("EXTERNAL_AUTH_REQUIRED", codes)
            self.assertNotIn("AGENT_AUTH_FAILURE", codes)

    def test_source_line_budget(self) -> None:
        self.assertLessEqual(
            sum(len(path.read_text(encoding="utf-8").splitlines()) for path in SRC.rglob("*.py")),
            32_000,
        )

    def test_wait_human_is_exclusive_to_spec_decision(self) -> None:
        valid = RunMachineState(
            RunPhase.PLANNER, RunDisposition.WAIT_HUMAN, "SPEC_DECISION_REQUIRED",
        )
        self.assertEqual(project_run_outcome(valid).status, RunStatus.WAITING_HUMAN)
        with self.assertRaises(RunTransitionError):
            transition(
                RunMachineState(RunPhase.AUDIT),
                RunEvent.wait(RunDisposition.WAIT_HUMAN, reason="CHECK_INFRASTRUCTURE_UNAVAILABLE"),
            )

    def test_external_wait_resumes_and_fixable_has_no_run_terminal_projection(self) -> None:
        waiting = RunMachineState(
            RunPhase.IMPLEMENT_STEP, RunDisposition.WAIT_EXTERNAL, "AGENT_TIMEOUT",
        )
        self.assertTrue(project_run_outcome(waiting).resumable)
        self.assertEqual(transition(waiting, RunEvent.resume()).disposition, RunDisposition.RUNNING)
        decision = classify_failure("AGENT_CONTRACT_MISMATCH", exhausted=True)
        self.assertIs(decision.strategy, RecoveryStrategy.MARK_FAILED_CONTINUE)
        with self.assertRaisesRegex(ValueError, "not a terminal run disposition"):
            terminal_state_for(
                decision, failure_code="AGENT_CONTRACT_MISMATCH",
                phase=RunPhase.IMPLEMENT_STEP,
            )
        with self.assertRaisesRegex(ValueError, "not a terminal run disposition"):
            project_exit("AGENT_CONTRACT_MISMATCH", phase=RunPhase.IMPLEMENT_STEP)

    def test_partial_is_autonomous_and_terminal(self) -> None:
        outcome = project_run_outcome(
            RunMachineState(RunPhase.PLANNER, RunDisposition.COMPLETED),
        )
        self.assertEqual(outcome.status, RunStatus.PARTIAL)
        self.assertTrue(outcome.disposition.terminal)

    def test_public_statuses_are_one_projection_of_machine_facts(self) -> None:
        cases = (
            (RunMachineState(RunPhase.IMPLEMENT_STEP), RunStatus.RUNNING),
            (
                RunMachineState(
                    RunPhase.DETERMINISTIC_GATE, RunDisposition.WAIT_EXTERNAL,
                    "CHECK_INFRASTRUCTURE_UNAVAILABLE",
                ),
                RunStatus.WAITING_EXTERNAL,
            ),
            (
                RunMachineState(
                    RunPhase.PLANNER, RunDisposition.WAIT_HUMAN, "SPEC_DECISION_REQUIRED",
                ),
                RunStatus.WAITING_HUMAN,
            ),
            (
                RunMachineState(RunPhase.CANDIDATE_PUSH, RunDisposition.COMPLETED),
                RunStatus.COMMITTED,
            ),
            (
                RunMachineState(RunPhase.PUBLISH, RunDisposition.COMPLETED),
                RunStatus.PUBLISHED,
            ),
            (
                RunMachineState(RunPhase.PLANNER, RunDisposition.COMPLETED),
                RunStatus.PARTIAL,
            ),
            (
                RunMachineState(
                    RunPhase.IMPLEMENT_STEP, RunDisposition.FAILED, "AGENT_SCOPE_VIOLATION",
                ),
                RunStatus.FAILED,
            ),
        )
        for machine, expected in cases:
            with self.subTest(expected=expected.value):
                self.assertIs(project_run_outcome(machine).status, expected)


if __name__ == "__main__":
    unittest.main()
