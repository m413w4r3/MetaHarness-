import hashlib
import json
import re
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
    MAX_STEPS,
    MAX_STEP_CONTRACT_CHARS,
    REQUIRE_STAGED_POLICY_TEXT,
    PlannerV2,
    V2PlanParseError,
    build_planner_prompt_v2,
    REPAIR_PLANNER_INLINE_TARGET_BYTES,
    build_repair_planner_prompt,
    build_repair_planner_prompt_bundle,
    render_decomposition_policy_text,
    render_repair_step_index,
    validate_decomposition_policy,
    parse_task_plan_v2,
    render_plan_summary_v2,
    render_repair_plan_summary,
    read_approved_step_contract,
    render_safe_profile_catalogue,
    render_step_contract,
    validate_implementation_bundle,
    write_implementation_bundle,
)
from metaharness import step_ids as step_id_authority  # noqa: E402


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
        self.assertEqual(MAX_STEPS, 99)
        single = _parse(_plan())
        self.assertEqual(single.execution_mode, ExecutionMode.SINGLE)
        self.assertEqual(single.steps[0].id, "S01")
        staged = _parse(_plan("STAGED", 2, steps=_step(1) + "\n\n" + _step(2)))
        self.assertEqual(staged.execution_mode, ExecutionMode.STAGED)
        self.assertEqual([step.id for step in staged.steps], ["S01", "S02"])
        six = "\n\n".join(_step(number) for number in range(1, 7))
        self.assertEqual(len(_parse(_plan("STAGED", 6, steps=six)).steps), 6)
        eight = "\n\n".join(_step(number) for number in range(1, 8 + 1))
        self.assertEqual([step.id for step in _parse(_plan("STAGED", 8, steps=eight)).steps],
                         [f"S{number:02d}" for number in range(1, 8 + 1)])

    def test_blocked_execution_metadata_is_rejected(self):
        observed = """META PLAN v2

STATUS: BLOCKED
TITLE: Cannot safely plan

OBJECTIVE
The requested change cannot be planned safely.

CONSTRAINTS
The repository context is insufficient.

EXECUTION_MODE: STAGED
STEP_COUNT: 7
REVIEWER_PROFILE: review-a

BLOCKERS
The required architectural information is missing.

END META PLAN
"""
        with self.assertRaises(V2PlanParseError):
            _parse(observed)

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

    def test_minimal_blocked_wire_format_is_accepted(self):
        plan = _parse("""META PLAN v2

STATUS: BLOCKED
TITLE: Waiting for an API contract

OBJECTIVE
The implementation cannot be specified safely yet.

BLOCKERS
The required API contract is absent.

END META PLAN
""")
        self.assertEqual(plan.decision, PlanDecision.BLOCKED)
        self.assertEqual(plan.title, "Waiting for an API contract")
        self.assertEqual(plan.blockers, "The required API contract is absent.")

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
        # Six ~5500-character contracts remain below the individual limit.
        six = "\n\n".join(_step(number, instruction="x" * 5100) for number in range(1, 7))
        self.assertEqual(len(_parse(_plan("STAGED", 6, steps=six)).steps), 6)
        plan = _parse(_plan())
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_implementation_bundle(directory, plan)
            contract_path = Path(directory) / "steps/S01/contract.md"
            self.assertFalse((Path(directory) / "steps/S01.contract.md").exists())
            self.assertEqual(bundle["steps"][0]["contract_sha256"], hashlib.sha256(contract_path.read_bytes()).hexdigest())
            self.assertEqual(json.loads((Path(directory) / "implementation_bundle.json").read_text())["schema_version"], 1)
            self.assertIn("ordered steps", (Path(directory) / "implementation_contract.md").read_text().lower())

    def test_aw002_synthetic_seven_steps_parse_and_write_bundle(self):
        steps = "\n\n".join(
            _sets_step(number, write=_paths(f"aw002_{number}_", 6))
            for number in range(1, 8)
        )
        plan = _parse(_plan("STAGED", 7, steps=steps))
        self.assertEqual([step.id for step in plan.steps],
                         [f"S{number:02d}" for number in range(1, 8)])
        self.assertTrue(all(
            len(set(step.write_set) | set(step.create_set) | set(step.delete_set)) <= 6
            for step in plan.steps
        ))
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_implementation_bundle(directory, plan)
            self.assertEqual([entry["id"] for entry in bundle["steps"]],
                             [f"S{number:02d}" for number in range(1, 8)])
            self.assertTrue((Path(directory) / "implementation_bundle.json").exists())

    def test_nine_steps_and_s09_are_accepted_but_s100_is_rejected(self):
        nine = "\n\n".join(_step(number) for number in range(1, 10))
        self.assertEqual(len(_parse(_plan("STAGED", 9, steps=nine)).steps), 9)
        self.assertTrue(step_id_authority.is_step_id("S09"))
        with self.assertRaises(V2PlanParseError):
            _parse(_plan("STAGED", 100, steps="\n\n".join(_step(number) for number in range(1, 100))))

    def test_9_16_and_32_steps_parse_and_write_bundles(self) -> None:
        for count in (9, 16, 32):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                plan = _parse(_plan(
                    "STAGED", count,
                    steps="\n\n".join(_step(number) for number in range(1, count + 1)),
                ))
                bundle = write_implementation_bundle(directory, plan)
                payload, _digest = validate_implementation_bundle(
                    directory, expected_step_ids=[f"S{number:02d}" for number in range(1, count + 1)]
                )
                self.assertEqual(payload, bundle)

    def test_noncontiguous_eighth_step_is_rejected(self):
        steps = "\n\n".join(_step(number) for number in range(1, 7)) + "\n\n" + _step(8)
        with self.assertRaises(V2PlanParseError):
            _parse(_plan("STAGED", 7, steps=steps))

    def test_eight_step_bundle_is_written_read_back_and_hash_validated(self):
        ids = [f"S{number:02d}" for number in range(1, 8 + 1)]
        plan = _parse(_plan("STAGED", 8, steps="\n\n".join(_step(number) for number in range(1, 8 + 1))))
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_implementation_bundle(directory, plan)
            self.assertEqual([entry["id"] for entry in bundle["steps"]], ids)
            payload, digest = validate_implementation_bundle(directory, expected_step_ids=ids)
            self.assertEqual(payload, bundle)
            self.assertEqual(
                digest,
                hashlib.sha256((Path(directory) / "implementation_bundle.json").read_bytes()).hexdigest(),
            )
            for step in plan.steps:
                self.assertEqual(
                    read_approved_step_contract(directory, payload, step.id),
                    render_step_contract(plan, step),
                )
            (Path(directory) / "steps/S08/contract.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(V2PlanParseError):
                validate_implementation_bundle(directory, expected_step_ids=ids)

    def test_contract_budget_is_per_step_only(self):
        self.assertEqual(MAX_STEP_CONTRACT_CHARS, 16_000)
        steps = "\n\n".join(_step(number, instruction="x" * 6000) for number in range(1, 8 + 1))
        plan = _parse(_plan("STAGED", 8, steps=steps))
        self.assertGreater(sum(len(render_step_contract(plan, step)) for step in plan.steps), 48_000)
        too_large = _plan().replace("1. edit the named symbol", "x" * MAX_STEP_CONTRACT_CHARS)
        with self.assertRaises(V2PlanParseError):
            _parse(too_large)

    def test_step_capacity_has_one_authority(self):
        self.assertIs(MAX_STEPS, step_id_authority.MAX_STEPS)
        self.assertEqual(step_id_authority.step_ids(8), tuple(f"S{number:02d}" for number in range(1, 8 + 1)))
        for value in ("S01", "S07", "S08"):
            self.assertTrue(step_id_authority.is_step_id(value), value)
        for value in ("S00", "S100", "s01", "S1", "S001", " S01", 7, None):
            self.assertFalse(step_id_authority.is_step_id(value), value)
        for count in (0, 100, True, "8"):
            with self.assertRaises(ValueError):
                step_id_authority.step_ids(count)
        self.assertIn(f"between 2 and {MAX_STEPS} coherent", REQUIRE_STAGED_POLICY_TEXT)
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        for name in ("planner_v2.txt", "repair_planner_v2.txt"):
            text = (prompts / name).read_text(encoding="utf-8")
            with self.subTest(prompt=name):
                self.assertIn("STAGED has 2 to {{MAX_STEPS}} steps", text)
                self.assertIn("S01 through {{LAST_STEP_ID}}", text)
                self.assertIn("{{MAX_STEP_CONTRACT_CHARS}}", text)

    def test_no_hidden_step_bound_remains_in_sources(self):
        repository = Path(__file__).resolve().parents[1]
        roots = (repository / "src" / "metaharness", repository / "docs", repository / "examples")
        hidden = re.compile("|".join((
            "2 to " + "8", r"2\.\." + "8", "S01 through S" + "08", r"S01\.\.S" + "08",
            "48" + "000", r"MAX_STEPS\s*=\s*" + "8", r"range\(1,\s*" + r"9\)",
            "at most " + "8 steps", "eight " + "steps", r"S0\[1-",
            r"range\(1, ?" + r"7\)", "two and " + "six", "at most " + "six steps",
            "S01 through S" + "06",
        )))
        offenders = [
            str(path.relative_to(repository)) for root in roots for path in sorted(root.rglob("*"))
            if path.suffix in {".py", ".txt", ".js"} and hidden.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_blocked_protocol_is_documented_in_both_planner_prompts(self):
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        template = "META PLAN v2\n\nSTATUS: BLOCKED\nTITLE: <title>\n\nOBJECTIVE\n<text>\n\nBLOCKERS\n<real concrete blockers>\n\nEND META PLAN"
        for name in ("planner_v2.txt", "repair_planner_v2.txt"):
            text = (prompts / name).read_text(encoding="utf-8")
            with self.subTest(prompt=name):
                self.assertIn(template, text)
                self.assertIn(
                    "For BLOCKED, do not emit CONSTRAINTS, REQUIRED_CHECKS, EXECUTION_MODE,\n"
                    "STEP_COUNT, REVIEWER_PROFILE, BEGIN/END STEP, ACCEPTANCE, TESTS, or RISKS.",
                    text,
                )
        # The documented template itself is accepted by the unchanged parser.
        filled = template.replace("<title>", "Blocked").replace("<text>", "Cannot plan.").replace(
            "<real concrete blockers>", "The API contract is absent."
        )
        self.assertEqual(_parse(filled).decision, PlanDecision.BLOCKED)

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

    def test_repair_planner_prompt_keeps_the_structured_review_without_raw_review(self):
        prompt = build_repair_planner_prompt(**_repair_inputs())

        self.assertNotIn("REVIEWER_1_RAW", prompt)
        self.assertNotIn("REVIEWER #1 RAW", prompt)
        self.assertIn("<REVIEWER #1 STRUCTURED RESULT>\nREVIEW\n", prompt)
        self.assertIn("STATUS: BLOCKED", prompt)
        self.assertIn("For BLOCKED, do not emit CONSTRAINTS, REQUIRED_CHECKS, EXECUTION_MODE,", prompt)

    def test_repair_prompt_uses_compact_summary_and_no_step_contracts(self):
        plan = _parse(_plan())
        summary = render_repair_plan_summary(plan)
        prompt = build_repair_planner_prompt(
            **{**_repair_inputs(), "original_plan_summary": summary}
        )
        self.assertIn("ORIGINAL PLAN SUMMARY", prompt)
        self.assertNotIn("ORIGINAL META PLAN", prompt)
        self.assertNotIn("READ_SET", summary)
        self.assertNotIn("INSTRUCTIONS", summary)


def _repair_inputs(**overrides):
    values = {
        "repository_reference": "REPOSITORY",
        "original_spec": "SPEC",
        "original_plan_summary": "PLAN_SUMMARY",
        "original_step_index": "STEP_INDEX",
        "current_repository_state": "STATE",
        "candidate_code_evidence": "CODE_EVIDENCE",
        "final_checks_cycle_1": "CHECKS",
        "claude_revision_report_cycle_1": "CLAUDE",
        "original_approved_mutable_scope": "SCOPE",
        "reviewer_result": "REVIEW",
    }
    values.update(overrides)
    return values


class RepairPlannerPromptBundleTests(unittest.TestCase):
    SENTINELS = (
        "SPEC", "PLAN_SUMMARY", "STEP_INDEX", "STATE", "CODE_EVIDENCE",
        "CHECKS", "CLAUDE", "SCOPE", "REVIEW",
    )

    def test_inline_prompt_carries_every_datum_and_no_removed_section(self):
        bundle = build_repair_planner_prompt_bundle(**_repair_inputs())

        for sentinel in self.SENTINELS:
            self.assertIn(sentinel, bundle.inline_prompt, sentinel)
        for removed in ("ORIGINAL STEP CONTRACTS", "CURRENT CUMULATIVE DIFF",
                        "REVIEWER REQUIRED FIXES", "REVIEWER #1 MISSING TESTS"):
            self.assertNotIn(removed, bundle.inline_prompt, removed)
        self.assertIn("The bounded repair evidence follows inline below.",
                      bundle.inline_prompt)
        self.assertIn(bundle.evidence_text, bundle.inline_prompt)

    def test_fallback_prompt_points_at_the_attachment_and_carries_no_evidence(self):
        bundle = build_repair_planner_prompt_bundle(**_repair_inputs())

        self.assertIn("repair-evidence.md", bundle.fallback_prompt)
        self.assertIn("[repair evidence intentionally moved to attachment]",
                      bundle.fallback_prompt)
        for sentinel in ("SPEC", "STEP_INDEX", "CODE_EVIDENCE"):
            self.assertNotIn(sentinel, bundle.fallback_prompt, sentinel)
        # "REVIEW" alone appears in the control prompt's REVIEWER PROFILES
        # heading, so the reviewer datum is checked through its evidence tag.
        self.assertNotIn("<REVIEWER #1 STRUCTURED RESULT>", bundle.fallback_prompt)
        self.assertNotIn(bundle.evidence_text, bundle.fallback_prompt)

    def test_evidence_text_has_the_exact_envelope_and_every_section(self):
        bundle = build_repair_planner_prompt_bundle(**_repair_inputs())
        evidence = bundle.evidence_text

        self.assertTrue(evidence.startswith("REPAIR PLANNER EVIDENCE v1\n"))
        self.assertTrue(evidence.endswith("END REPAIR PLANNER EVIDENCE\n"))
        for name, value in (
            ("REPOSITORY REFERENCE", "REPOSITORY"),
            ("ORIGINAL SPEC", "SPEC"),
            ("ORIGINAL PLAN SUMMARY", "PLAN_SUMMARY"),
            ("ORIGINAL STEP INDEX", "STEP_INDEX"),
            ("CURRENT REPOSITORY STATE", "STATE"),
            ("CANDIDATE CODE EVIDENCE", "CODE_EVIDENCE"),
            ("FINAL CHECKS CYCLE 1", "CHECKS"),
            ("CLAUDE REVISION REPORT CYCLE 1", "CLAUDE"),
            ("ORIGINAL APPROVED MUTABLE SCOPE", "SCOPE"),
            ("REVIEWER #1 STRUCTURED RESULT", "REVIEW"),
        ):
            self.assertIn(f"<{name}>\n{value}\n</{name}>", evidence, name)

    def test_step_index_is_compact_and_omits_worker_detail(self):
        steps = "\n\n".join(_step(number) for number in range(1, 4))
        plan = _parse(_plan("STAGED", 3, steps=steps))
        index = render_repair_step_index(plan)
        payload = json.loads(index)

        self.assertEqual([entry["id"] for entry in payload], ["S01", "S02", "S03"])
        for key in ("id", "title", "depends_on", "objective", "mutation_scope",
                    "verify", "forbidden"):
            self.assertIn(key, payload[0], key)
        self.assertEqual(payload[0]["mutation_scope"],
                         {"write": ["src/example.py"], "create": [], "delete": []})
        for absent in ("read_set", "instructions", "implementer_profile"):
            self.assertNotIn(absent, payload[0], absent)
        self.assertNotIn("edit the named symbol", index)
        self.assertNotIn("impl-a", index)

    def test_an_aw_002_sized_request_stays_under_the_inline_target(self):
        steps = "\n\n".join(_step(number) for number in range(1, 19))
        plan = _parse(_plan("STAGED", 18, steps=steps))
        # A 500 KB candidate diff exists in the run artifacts and is
        # deliberately never handed to the prompt builder.
        huge_diff = "D" * 500_000

        bundle = build_repair_planner_prompt_bundle(
            repository_reference="REPOSITORY",
            original_spec="S" * 30_000,
            original_plan_summary=render_repair_plan_summary(plan),
            original_step_index=render_repair_step_index(plan),
            current_repository_state=json.dumps(
                {"CHANGED_FILES": [f"src/module_{n:03d}.py" for n in range(100)]}
            ),
            candidate_code_evidence=json.dumps({
                "authority": "immutable_candidate_commit",
                "base_sha": "a" * 40,
                "candidate_sha": "b" * 40,
                "candidate_url": "https://example.invalid/commit/" + "b" * 40,
                "compare_url": "https://example.invalid/compare/a...b",
                "full_diff_bytes": len(huge_diff),
                "full_diff_sha256": "c" * 64,
                "inline_full_diff": False,
            }),
            final_checks_cycle_1=json.dumps({"deterministic_passed": True}),
            claude_revision_report_cycle_1="C" * (16 * 1024),
            original_approved_mutable_scope=json.dumps(
                [f"src/module_{n:03d}.py" for n in range(100)]
            ),
            reviewer_result="R" * 8_192,
        )

        self.assertLess(
            len(bundle.inline_prompt.encode("utf-8")),
            REPAIR_PLANNER_INLINE_TARGET_BYTES,
        )
        self.assertNotIn("D" * 10_000, bundle.inline_prompt)
        self.assertLess(len(bundle.fallback_prompt.encode("utf-8")), 24 * 1024)


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
        }
        for name, sets in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(V2PlanParseError):
                    _parse(_plan(steps=_change_step(sets)))

    def test_read_set_and_structural_sets_have_no_hidden_count_caps(self) -> None:
        paths = tuple(f"src/read{index}.py" for index in range(1, 10))
        read = "READ_SET\n" + "".join(f"- {path} :: anchor\n" for path in paths)
        writes = "WRITE_SET\n" + "".join(f"- {path}\n" for path in paths)
        raw = _plan(steps=_change_step(read + "\n" + writes + "\nCREATE_SET\nNONE\n\nDELETE_SET\nNONE\n"))
        self.assertEqual(len(_parse(raw).steps[0].read_set), 9)
        self.assertEqual(len(_parse(raw).steps[0].write_set), 9)

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
            "Normal target: keep each step contract concise, approximately 4000-6000 chars.",
            "Hard per-step parser limit: 16000 characters.",
            "CREATE_SET\nNONE",
            "DELETE_SET\nNONE",
            "STAGED has 2 to 99 steps. SINGLE has exactly 1 step.",
            "Step IDs are contiguous S01 through S99",
            "There is no aggregate contract-size limit.",
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

    def test_configured_eight_path_boundary_is_checked_only_by_policy(self):
        planning = _aggressive(staged_step_max_mutable_paths=8)
        eight = _parse(_staged(write=_paths("w", 8)))
        validate_decomposition_policy(eight, planning)
        nine = _parse(_staged(write=_paths("w", 9)))
        with self.assertRaisesRegex(V2PlanParseError, "at most 8 distinct mutable paths; got 9"):
            validate_decomposition_policy(nine, planning)

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
