import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.agent.codex import build_implementer_step_prompt  # noqa: E402
from metaharness.models import (  # noqa: E402
    ExecutionMode,
    ExecutionRole,
    ModelProfile,
    PlanDecision,
    ProfileDriver,
    SelectionMode,
)
from metaharness.planning_v2 import (  # noqa: E402
    MAX_STEP_CONTRACT_CHARS,
    MAX_TOTAL_STEP_CONTRACT_CHARS,
    V2PlanParseError,
    build_planner_prompt_v2,
    parse_task_plan_v2,
    render_plan_summary_v2,
    render_safe_profile_catalogue,
    render_step_contract,
    write_implementation_bundle,
)


def _step(number: int, *, instruction: str = "1. edit the named symbol") -> str:
    step_id = f"S{number:02d}"
    dependency = "NONE" if number == 1 else "S01"
    return f"""BEGIN STEP {step_id}
TITLE: Step {number}
IMPLEMENTER_PROFILE: impl-a
DEPENDS_ON: {dependency}

OBJECTIVE
Implement step {number}.

READ_SET
- src/example.py :: function example()

WRITE_SET
- src/example.py

INSTRUCTIONS
{instruction}

VERIFY
- python -m unittest tests.test_planning_v2

FORBIDDEN
- Do not change files outside WRITE_SET.

END STEP {step_id}"""


def _plan(mode: str = "SINGLE", count: int = 1, *, steps: str | None = None) -> str:
    if steps is None:
        steps = _step(1)
    return f"""META PLAN v2

STATUS: READY
TITLE: v2 task

OBJECTIVE
Implement the task.

CONSTRAINTS
NONE

EXECUTION_MODE: {mode}
STEP_COUNT: {count}
REVIEWER_PROFILE: review-a

{steps}

ACCEPTANCE
The requested behavior is observable.

TESTS
Run the narrow tests listed in each step.

RISKS
Existing v1 behavior remains unchanged.

BLOCKERS
NONE

END META PLAN
"""


def _parse(raw: str):
    return parse_task_plan_v2(
        raw,
        implementer_ids=frozenset({"impl-a"}),
        reviewer_ids=frozenset({"review-a"}),
    )


class PlanningV2Tests(unittest.TestCase):
    def test_single_and_staged_boundaries(self):
        single = _parse(_plan())
        self.assertEqual(single.execution_mode, ExecutionMode.SINGLE)
        self.assertEqual(single.steps[0].id, "S01")
        staged = _parse(_plan("STAGED", 2, steps=_step(1) + "\n\n" + _step(2)))
        self.assertEqual(staged.execution_mode, ExecutionMode.STAGED)
        self.assertEqual([step.id for step in staged.steps], ["S01", "S02"])
        six = "\n\n".join(_step(number) for number in range(1, 7))
        self.assertEqual(len(_parse(_plan("STAGED", 6, steps=six)).steps), 6)

    def test_blocked_has_no_steps(self):
        plan = _parse("""META PLAN v2
STATUS: BLOCKED
TITLE: Waiting

OBJECTIVE
Cannot safely proceed.

BLOCKERS
The required API contract is absent.
END META PLAN
""")
        self.assertEqual(plan.decision, PlanDecision.BLOCKED)
        self.assertEqual(plan.steps, ())

    def test_strict_structure_rejects_requested_invalid_cases(self):
        cases = {
            "bad step count": _plan("SINGLE", 2),
            "gap": _plan("STAGED", 2, steps=_step(1) + "\n\n" + _step(3)),
            "future dependency": _plan("STAGED", 2, steps=_step(1).replace("DEPENDS_ON: NONE", "DEPENDS_ON: S02") + "\n\n" + _step(2)),
            "unknown implementer": _plan().replace("IMPLEMENTER_PROFILE: impl-a", "IMPLEMENTER_PROFILE: other"),
            "unknown reviewer": _plan().replace("REVIEWER_PROFILE: review-a", "REVIEWER_PROFILE: other"),
            "unsafe path": _plan().replace("src/example.py ::", "../example.py ::"),
            "write not read": _plan().replace("- src/example.py\n\nINSTRUCTIONS", "- src/other.py\n\nINSTRUCTIONS"),
            "duplicate path": _plan().replace("- src/example.py :: function example()", "- src/example.py :: one\n- src/example.py :: two"),
            "missing VERIFY": _plan().replace("VERIFY\n- python -m unittest tests.test_planning_v2", "VERIFY\n"),
            "fuzzy profile": _plan().replace("REVIEWER_PROFILE: review-a", "REVIEWER_PROFILE: REVIEW-A"),
        }
        for name, raw in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(V2PlanParseError):
                    _parse(raw)

    def test_contract_and_summary_exclude_original_inputs(self):
        plan = _parse(_plan())
        contract = render_step_contract(plan, plan.steps[0])
        self.assertIn("S01 / 01", contract)
        self.assertNotIn("META PLAN v2", contract)
        self.assertNotIn("SPEC", contract)
        self.assertNotIn("CONTEXT", contract)
        summary = render_plan_summary_v2(plan)
        self.assertIn("v2 task", summary)
        self.assertIn("Acceptance", summary)

    def test_bounds_and_bundle_hashes(self):
        huge = _plan().replace("1. edit the named symbol", "x" * MAX_STEP_CONTRACT_CHARS)
        with self.assertRaises(V2PlanParseError):
            _parse(huge)
        six = "\n\n".join(_step(number, instruction="x" * 5100) for number in range(1, 7))
        with self.assertRaises(V2PlanParseError):
            _parse(_plan("STAGED", 6, steps=six))
        plan = _parse(_plan())
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_implementation_bundle(directory, plan)
            contract_path = Path(directory) / "steps/S01/contract.md"
            self.assertFalse((Path(directory) / "steps/S01.contract.md").exists())
            self.assertEqual(bundle["steps"][0]["contract_sha256"], hashlib.sha256(contract_path.read_bytes()).hexdigest())
            self.assertEqual(json.loads((Path(directory) / "implementation_bundle.json").read_text())["schema_version"], 1)
            self.assertIn("ordered steps", (Path(directory) / "implementation_contract.md").read_text().lower())

    def test_prompt_and_safe_catalogue(self):
        profile = ModelProfile(
            id="impl-a", display_name="Implementer", roles=(ExecutionRole.IMPLEMENTER,),
            driver=ProfileDriver.CODEX, model="luna", selection_mode=SelectionMode.CLI,
            api_key_env="DO_NOT_RENDER", base_url="https://secret.invalid",
            effort="high", description="precise", strengths=("tests",),
        )
        catalogue = render_safe_profile_catalogue((profile,))
        self.assertIn("ID: impl-a", catalogue)
        self.assertNotIn("DO_NOT_RENDER", catalogue)
        self.assertNotIn("secret.invalid", catalogue)
        prompt = build_planner_prompt_v2("spec", "context", implementer_profiles=(profile,), reviewer_profiles=(profile,))
        self.assertIn("Do not output code fences.", prompt)
        self.assertIn("exact symbol/variable names when supplied context proves them;", prompt)
        worker = build_implementer_step_prompt("contract")
        self.assertIn("contract", worker)
        self.assertNotIn("{{STEP_CONTRACT}}", worker)


def _change_step(sets: str) -> str:
    return _step(1).replace(
        "READ_SET\n- src/example.py :: function example()\n\nWRITE_SET\n- src/example.py\n", sets
    )


class ChangeSetTests(unittest.TestCase):
    def test_create_and_delete_sets_are_parsed_and_rendered(self) -> None:
        sets = (
            "READ_SET\n- src/example.py :: function example()\n- src/old.py :: module\n\n"
            "WRITE_SET\n- src/example.py\n\n"
            "CREATE_SET\n- src/new.py\n\n"
            "DELETE_SET\n- src/old.py\n"
        )
        plan = _parse(_plan(steps=_change_step(sets)))
        step = plan.steps[0]
        self.assertEqual(step.write_set, ("src/example.py",))
        self.assertEqual(step.create_set, ("src/new.py",))
        self.assertEqual(step.delete_set, ("src/old.py",))
        contract = render_step_contract(plan, step)
        order = [contract.index(name) for name in (
            "OBJECTIVE", "READ SET", "WRITE SET", "CREATE SET", "DELETE SET",
            "INSTRUCTIONS", "VERIFY", "FORBIDDEN", "END META IMPLEMENTATION STEP",
        )]
        self.assertEqual(order, sorted(order))
        self.assertIn("CREATE SET\n- src/new.py", contract)
        self.assertIn("DELETE SET\n- src/old.py", contract)

    def test_legacy_plan_without_sections_has_empty_sets(self) -> None:
        plan = _parse(_plan())
        self.assertEqual((plan.steps[0].create_set, plan.steps[0].delete_set), ((), ()))
        contract = render_step_contract(plan, plan.steps[0])
        self.assertIn("CREATE SET\nNONE", contract)
        self.assertIn("DELETE SET\nNONE", contract)

    def test_create_only_step_may_have_no_write_set(self) -> None:
        sets = (
            "READ_SET\n- src/example.py :: function example()\n\n"
            "WRITE_SET\nNONE\n\nCREATE_SET\n- src/new.py\n\nDELETE_SET\nNONE\n"
        )
        step = _parse(_plan(steps=_change_step(sets))).steps[0]
        self.assertEqual((step.write_set, step.create_set, step.delete_set), ((), ("src/new.py",), ()))

    def test_invalid_change_sets_are_rejected(self) -> None:
        read = "READ_SET\n- src/example.py :: function example()\n\n"
        cases = {
            "delete not read": read + "WRITE_SET\nNONE\n\nCREATE_SET\nNONE\n\nDELETE_SET\n- src/other.py\n",
            "create in read": read + "WRITE_SET\nNONE\n\nCREATE_SET\n- src/example.py\n\nDELETE_SET\nNONE\n",
            "write and delete overlap": read + "WRITE_SET\n- src/example.py\n\nCREATE_SET\nNONE\n\nDELETE_SET\n- src/example.py\n",
            "create twice": read + "WRITE_SET\nNONE\n\nCREATE_SET\n- a.py\n- a.py\n\nDELETE_SET\nNONE\n",
            "empty union": read + "WRITE_SET\nNONE\n\nCREATE_SET\nNONE\n\nDELETE_SET\nNONE\n",
            "blank create": read + "WRITE_SET\n- src/example.py\n\nCREATE_SET\n\nDELETE_SET\nNONE\n",
            "create wildcard": read + "WRITE_SET\nNONE\n\nCREATE_SET\n- src/*.py\n\nDELETE_SET\nNONE\n",
            "create traversal": read + "WRITE_SET\nNONE\n\nCREATE_SET\n- ../escape.py\n\nDELETE_SET\nNONE\n",
            "create anchor": read + "WRITE_SET\nNONE\n\nCREATE_SET\n- src/new.py :: anchor\n\nDELETE_SET\nNONE\n",
            "too many creates": read + "WRITE_SET\nNONE\n\nCREATE_SET\n"
            + "".join(f"- src/n{index}.py\n" for index in range(7)) + "\nDELETE_SET\nNONE\n",
        }
        for name, sets in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(V2PlanParseError):
                    _parse(_plan(steps=_change_step(sets)))

    def test_bundle_validation_binds_step_ids_and_contract_bytes(self) -> None:
        from metaharness.planning_v2 import read_approved_step_contract, validate_implementation_bundle

        plan = _parse(_plan("STAGED", 2, steps=_step(1) + "\n\n" + _step(2)))
        with tempfile.TemporaryDirectory() as directory:
            write_implementation_bundle(directory, plan)
            bundle, _sha = validate_implementation_bundle(directory, expected_step_ids=["S01", "S02"])
            self.assertEqual(
                read_approved_step_contract(directory, bundle, "S02"),
                (Path(directory) / "steps/S02/contract.md").read_text(),
            )
            with self.assertRaises(V2PlanParseError):
                validate_implementation_bundle(directory, expected_step_ids=["S01"])
            path = Path(directory) / "steps/S02/contract.md"
            path.write_bytes(path.read_bytes().replace(b"Step 2", b"Step X", 1))
            with self.assertRaises(V2PlanParseError):
                read_approved_step_contract(directory, bundle, "S02")
            with self.assertRaises(V2PlanParseError):
                validate_implementation_bundle(directory)

    def test_prompts_carry_the_worker_restrictions(self) -> None:
        planner = build_planner_prompt_v2("spec", "context")
        for sentence in (
            "The implementation worker is not a discovery agent.",
            "If the supplied context is insufficient to name the required file, symbol, or architectural operation precisely, return BLOCKED instead of delegating discovery to the worker.",
            "Normal target: keep each step contract under approximately 4000 characters.",
            "Hard parser limit remains 8000 characters.",
            "CREATE_SET\nNONE",
            "DELETE_SET\nNONE",
        ):
            self.assertIn(sentence, planner)
        worker = build_implementer_step_prompt("CONTRACT")
        for sentence in (
            "Do not run repository-wide discovery commands.",
            "- repo-wide grep/git-grep/rg.",
            "Do not inspect other step contracts.",
            "All design decisions are already final.",
        ):
            self.assertIn(sentence, worker)


if __name__ == "__main__":
    unittest.main()
