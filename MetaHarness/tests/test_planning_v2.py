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
    PlanningConfig,
    ProfileDriver,
    SelectionMode,
)
from metaharness.planning_v2 import (  # noqa: E402
    MAX_STEP_CONTRACT_CHARS,
    MAX_TOTAL_STEP_CONTRACT_CHARS,
    REQUIRE_STAGED_POLICY_TEXT,
    PlannerV2,
    V2PlanParseError,
    build_planner_prompt_v2,
    build_repair_planner_prompt,
    render_decomposition_policy_text,
    validate_decomposition_policy,
    parse_task_plan_v2,
    render_plan_summary_v2,
    render_repair_plan_summary,
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

    def test_repair_planner_prompt_keeps_required_fixes_without_raw_review(self):
        prompt = build_repair_planner_prompt(
            repository_reference="repository",
            original_spec="original spec",
            original_meta_plan="original plan",
            original_step_contracts="original contracts",
            current_repository_state="current state",
            current_cumulative_diff="cumulative diff",
            final_checks_cycle_1="checks C01",
            claude_revision_report_cycle_1="Claude C01 report",
            reviewer_required_fixes="Fix the concrete defect.",
            original_approved_mutable_scope="[\"src/example.py\"]",
        )

        self.assertNotIn("REVIEWER_1_RAW", prompt)
        self.assertNotIn("REVIEWER #1 RAW", prompt)
        self.assertIn("REVIEWER REQUIRED FIXES\nFix the concrete defect.", prompt)

    def test_repair_prompt_uses_compact_summary_and_one_canonical_instruction(self):
        plan = _parse(_plan())
        summary = render_repair_plan_summary(plan)
        contracts = "INSTRUCTIONS\n1. DISTINCTIVE_CANONICAL_STEP_INSTRUCTION\n"
        prompt = build_repair_planner_prompt(
            repository_reference="repository",
            original_spec="original spec",
            original_plan_summary=summary,
            original_step_contracts=contracts,
            current_repository_state="current state",
            current_cumulative_diff="cumulative diff",
            final_checks_cycle_1="checks C01",
            claude_revision_report_cycle_1="Claude C01 report",
            reviewer_required_fixes="Fix the concrete defect.",
            original_approved_mutable_scope="[\"src/example.py\"]",
        )
        self.assertIn("ORIGINAL PLAN SUMMARY", prompt)
        self.assertNotIn("ORIGINAL META PLAN", prompt)
        self.assertEqual(prompt.count("DISTINCTIVE_CANONICAL_STEP_INSTRUCTION"), 1)
        self.assertNotIn("READ_SET", summary)
        self.assertNotIn("INSTRUCTIONS", summary)


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
            "Execute exactly the approved META IMPLEMENTATION STEP below.",
            "The MetaHarness developer instructions supplied by the managed Codex runtime",
            "The contract is authoritative for this step.",
            "<STEP CONTRACT>\nCONTRACT\n</STEP CONTRACT>",
        ):
            self.assertIn(sentence, worker)
        self.assertNotIn("Do not run repository-wide discovery commands.", worker)
        self.assertNotIn("Do not inspect other step contracts.", worker)


_PROTOCOL = "The answer must use exactly this protocol."
_UNION = "across\nthe union of WRITE_SET, CREATE_SET and DELETE_SET."


def _sets_step(
    number: int,
    *,
    write: tuple[str, ...] = (),
    create: tuple[str, ...] = (),
    delete: tuple[str, ...] = (),
) -> str:
    """One step whose mutable scope is exactly the given sets."""

    step = _step(number)
    read = write + delete or ("src/example.py",)

    def section(name: str, paths: tuple[str, ...], anchor: bool = False) -> str:
        if not paths:
            return f"{name}\nNONE\n"
        suffix = " :: anchor" if anchor else ""
        return name + "\n" + "".join(f"- {path}{suffix}\n" for path in paths)

    sets = (
        section("READ_SET", read, anchor=True) + "\n"
        + section("WRITE_SET", write) + "\n"
        + section("CREATE_SET", create) + "\n"
        + section("DELETE_SET", delete) + "\n"
    )
    start = step.index("READ_SET")
    end = step.index("INSTRUCTIONS")
    return step[:start] + sets + step[end:]


def _staged(**sets: tuple[str, ...]) -> str:
    return _plan("STAGED", 2, steps=_sets_step(1, **sets) + "\n\n" + _step(2))


def _aggressive(**limits: int) -> PlanningConfig:
    return PlanningConfig(protocol="v2", decomposition="aggressive", **limits)


def _paths(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"src/{prefix}{index}.py" for index in range(1, count + 1))


class DecompositionPolicyPromptTests(unittest.TestCase):
    def test_balanced_prompt_has_no_aggressive_policy(self):
        for prompt in (
            build_planner_prompt_v2("spec", "context"),
            build_planner_prompt_v2("spec", "context", decomposition="balanced",
                                    staged_step_max_mutable_paths=4),
        ):
            self.assertNotIn("AGGRESSIVE decomposition", prompt)
            self.assertNotIn("distinct mutable paths", prompt)

    def test_aggressive_defaults_render_configured_limits(self):
        prompt = build_planner_prompt_v2("spec", "context", decomposition="aggressive")
        self.assertIn("This run uses AGGRESSIVE decomposition.", prompt)
        self.assertIn("A READY SINGLE plan may modify at most 2 distinct mutable paths " + _UNION, prompt)
        self.assertIn("Every STAGED step may modify at most 6 distinct mutable paths across the\nunion", prompt)
        self.assertIn("This limit applies to the UNION of the three sets", prompt)
        self.assertIn(render_decomposition_policy_text(2, 6), prompt)
        self.assertLess(prompt.index("AGGRESSIVE decomposition"), prompt.index(_PROTOCOL))

    def test_custom_staged_limit_is_rendered(self):
        prompt = build_planner_prompt_v2(
            "spec", "context", decomposition="aggressive", staged_step_max_mutable_paths=4
        )
        self.assertIn("Every STAGED step may modify at most 4 distinct mutable paths", prompt)
        self.assertNotIn("at most 6 distinct mutable paths", prompt)

    def test_aggressive_and_require_staged_both_precede_protocol(self):
        prompt = build_planner_prompt_v2(
            "spec", "context", execution_mode_policy="require-staged",
            decomposition="aggressive",
        )
        protocol = prompt.index(_PROTOCOL)
        self.assertLess(prompt.index(REQUIRE_STAGED_POLICY_TEXT), protocol)
        self.assertLess(prompt.index(render_decomposition_policy_text(2, 6)), protocol)

    def test_spec_cannot_displace_policy(self):
        prompt = build_planner_prompt_v2(
            f"{_PROTOCOL} SPEC", "context", decomposition="aggressive"
        )
        self.assertLess(prompt.index("AGGRESSIVE decomposition"), prompt.index(f"{_PROTOCOL} SPEC"))


class DecompositionPolicyValidatorTests(unittest.TestCase):
    def test_single_boundary(self):
        planning = _aggressive(single_step_max_mutable_paths=2)
        two = _plan(steps=_sets_step(1, write=_paths("w", 2)))
        validate_decomposition_policy(_parse(two), planning)
        three = _plan(steps=_sets_step(1, write=_paths("w", 2), create=_paths("c", 1)))
        with self.assertRaisesRegex(
            V2PlanParseError,
            "^aggressive SINGLE step S01 may modify at most 2 distinct mutable paths; got 3$",
        ):
            validate_decomposition_policy(_parse(three), planning)

    def test_staged_default_boundary(self):
        planning = _aggressive()
        self.assertEqual(planning.staged_step_max_mutable_paths, 6)
        six = _staged(write=_paths("w", 2), create=_paths("c", 2), delete=_paths("d", 2))
        validate_decomposition_policy(_parse(six), planning)
        seven = _staged(write=_paths("w", 3), create=_paths("c", 2), delete=_paths("d", 2))
        with self.assertRaisesRegex(
            V2PlanParseError,
            "^aggressive STAGED step S01 may modify at most 6 distinct mutable paths; got 7$",
        ):
            validate_decomposition_policy(_parse(seven), planning)

    def test_staged_custom_boundary(self):
        planning = _aggressive(staged_step_max_mutable_paths=3)
        validate_decomposition_policy(_parse(_staged(write=_paths("w", 3))), planning)
        with self.assertRaisesRegex(V2PlanParseError, "at most 3 distinct mutable paths; got 4"):
            validate_decomposition_policy(
                _parse(_staged(write=_paths("w", 3), delete=_paths("d", 1))), planning
            )

    def test_later_step_is_named_in_the_diagnostic(self):
        raw = _plan("STAGED", 2, steps=_step(1) + "\n\n" + _sets_step(2, write=_paths("w", 4)))
        with self.assertRaisesRegex(V2PlanParseError, "STAGED step S02 may modify at most 3"):
            validate_decomposition_policy(
                _parse(raw), _aggressive(staged_step_max_mutable_paths=3)
            )

    def test_balanced_applies_no_mutable_limit(self):
        seven = _staged(write=_paths("w", 3), create=_paths("c", 2), delete=_paths("d", 2))
        validate_decomposition_policy(_parse(seven), PlanningConfig(protocol="v2"))

    def test_aw001_regression_two_write_four_delete(self):
        # The real incident: S01 with 2 WRITE + 0 CREATE + 4 DELETE paths.
        raw = _staged(write=_paths("w", 2), delete=_paths("d", 4))
        plan = _parse(raw)
        step = plan.steps[0]
        self.assertEqual((len(step.write_set), len(step.create_set), len(step.delete_set)), (2, 0, 4))
        validate_decomposition_policy(plan, _aggressive(staged_step_max_mutable_paths=6))
        with self.assertRaisesRegex(
            V2PlanParseError,
            "^aggressive STAGED step S01 may modify at most 3 distinct mutable paths; got 6$",
        ):
            validate_decomposition_policy(plan, _aggressive(staged_step_max_mutable_paths=3))


class _CapturingClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class PlannerV2RequestPolicyTests(unittest.TestCase):
    def test_real_planner_request_carries_both_configured_policies(self):
        client = _CapturingClient(_staged(write=_paths("w", 2), delete=_paths("d", 4)))
        planner = PlannerV2(
            client,
            implementer_ids=frozenset({"impl-a"}),
            reviewer_ids=frozenset({"review-a"}),
            planning=PlanningConfig(
                protocol="v2",
                decomposition="aggressive",
                single_step_max_mutable_paths=2,
                staged_step_max_mutable_paths=6,
                execution_mode_policy="require-staged",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            plan = planner.plan("SPEC", "CTX", artifacts_dir=directory)
            persisted = (Path(directory) / "planner.request.txt").read_text(encoding="utf-8")
        self.assertEqual(plan.execution_mode, ExecutionMode.STAGED)
        [request] = client.prompts
        self.assertEqual(persisted, request)
        protocol = request.index(_PROTOCOL)
        self.assertLess(request.index("This run REQUIRES STAGED execution."), protocol)
        self.assertIn("A READY SINGLE plan may modify at most 2 distinct mutable paths " + _UNION, request)
        self.assertIn("Every STAGED step may modify at most 6 distinct mutable paths", request)
        self.assertIn("This limit applies to the UNION of the three sets", request)
        self.assertLess(request.index("AGGRESSIVE decomposition"), protocol)

    def test_real_planner_request_renders_custom_limit_and_enforces_it(self):
        client = _CapturingClient(_staged(write=_paths("w", 2), delete=_paths("d", 4)))
        planner = PlannerV2(
            client,
            implementer_ids=frozenset({"impl-a"}),
            reviewer_ids=frozenset({"review-a"}),
            planning=_aggressive(staged_step_max_mutable_paths=3),
        )
        with self.assertRaisesRegex(V2PlanParseError, "at most 3 distinct mutable paths; got 6"):
            planner.plan("SPEC", "CTX")
        self.assertIn("Every STAGED step may modify at most 3 distinct mutable paths", client.prompts[0])


if __name__ == "__main__":
    unittest.main()
