import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from metaharness.prompt_contracts import (
    PromptPayload,
    build_check_repair_payload,
    build_final_review_payload,
    build_implementer_payload,
    build_planner_payload,
    build_semantic_revision_payload,
    write_prompt_diagnostics,
)
from metaharness.orchestration.revision import EffectivePlanView


class PromptContractTests(unittest.TestCase):
    def test_role_prompt_static_size_limits(self) -> None:
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        limits = {
            "check_repair.txt": 3 * 1024,
            "reviser.txt": 4 * 1024,
            "reviewer.txt": 6 * 1024,
            "implementer.txt": 4 * 1024,
        }
        for name, limit in limits.items():
            with self.subTest(prompt=name):
                self.assertLess((prompts / name).stat().st_size, limit)

    def test_reviewer_is_independent_of_runtime_cycle_and_agent_names(self) -> None:
        template = (
            Path(__file__).resolve().parents[1]
            / "src" / "metaharness" / "prompts" / "reviewer.txt"
        ).read_text(encoding="utf-8").casefold()
        v2_prompt = build_final_review_payload(
            spec="SPEC",
            compact_approved_plan="PLAN",
            required_checks_summary="CHECKS",
            cycle_summary="compact cycle summary",
        ).rendered.casefold()
        for forbidden in ("iteration 1", "iteration 2", "c02", "claude", "luna"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, template)
                self.assertNotIn(forbidden, v2_prompt)

    def test_reviser_delegates_the_authoritative_gate_to_metaharness(self) -> None:
        payload = build_semantic_revision_payload(
            spec="SPEC",
            compact_approved_contract_index="S01",
            candidate_identity="TREE",
            changed_files="src/a.py",
            required_checks_summary="unit: failed",
            mutable_scope="src/a.py",
            bounded_diff_evidence="DIFF",
        )
        self.assertIn("MetaHarness owns and reruns the authoritative deterministic gate.", payload.rendered)
        self.assertNotIn("make test-all", payload.rendered)

    def test_implementer_is_executor_only(self) -> None:
        payload = build_implementer_payload(
            step_title="S01",
            step_objective="implement one step",
            step_invariants="preserve API",
            read_set="src/a.py :: symbol",
            mutable_scope="src/a.py",
            repository_instructions="follow AGENTS.md",
            verify_instructions="python -m unittest",
        )
        self.assertIn("You are the implementation executor", payload.rendered)
        self.assertIn("Do not redesign the plan or broaden the task.", payload.rendered)
        self.assertIn("Implement the supplied contract exactly.", payload.rendered)

    def test_normal_check_repair_prompt_does_not_delegate_to_repair_planner(self) -> None:
        payload = build_check_repair_payload(
            spec="SPEC",
            failed_check_ids="CHECK_FAILED:unit",
            failed_check_evidence="FAILED",
            compact_contract_invariants="INVARIANT",
            changed_files="src/a.py",
            mutable_scope="src/a.py",
        )
        self.assertNotIn("repair planner", payload.rendered.casefold())

    def test_sections_are_deterministic_and_hash_injected_bytes(self) -> None:
        kwargs = dict(
            spec="SPEC\né",
            failed_check_ids="CHECK_FAILED:unit",
            failed_check_evidence="stdout that may be shortened",
            compact_contract_invariants="S01 | writes=src/a.py",
            changed_files="src/a.py",
            mutable_scope='["src/a.py"]',
            candidate_identity="tree=" + "a" * 40,
            budget_bytes=40_000,
        )
        first = build_check_repair_payload(**kwargs)
        second = build_check_repair_payload(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first.total_bytes, len(first.rendered.encode("utf-8")))
        section = next(item for item in first.sections if item.name == "spec")
        self.assertEqual(section.byte_count, len(section.text.encode("utf-8")))
        self.assertEqual(section.sha256, hashlib.sha256(section.text.encode("utf-8")).hexdigest())

    def test_authority_is_never_truncated_and_omissions_are_diagnostic(self) -> None:
        payload = build_final_review_payload(
            spec="SPEC" * 100,
            compact_approved_plan="PLAN" * 100,
            required_checks_summary="CHECKS",
            immutable_candidate_identity="SHA",
            changed_files="changed.py",
            diff_sha256="diff-hash",
            diffstat="STAT",
            bounded_diff_excerpt="secondary evidence" * 1000,
            cycle_summary="old cycle" * 1000,
            budget_bytes=100,
        )
        self.assertIsInstance(payload, PromptPayload)
        self.assertTrue(payload.budget_overrun)
        self.assertTrue(payload.omitted_sections)
        for section in payload.sections:
            if section.authority:
                self.assertFalse(section.truncated)
        with tempfile.TemporaryDirectory() as directory:
            path = write_prompt_diagnostics(directory, payload)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["prompt_bytes"], len(payload.rendered.encode("utf-8")))
        self.assertEqual(data["omitted_sections"], list(payload.omitted_sections))
        self.assertNotIn("old cycle", json.dumps(data))

    def test_role_builders_expose_only_bounded_role_contracts(self) -> None:
        planner = build_planner_payload(
            spec="SPEC",
            repository_identity="BASE SHA",
            discovery_context="indexed files",
            trusted_check_catalogue="unit",
            available_profile_catalogue="luna, claude",
            planning_constraints="NONE",
        )
        implementer = build_implementer_payload(
            step_title="S01",
            step_objective="implement one step",
            step_invariants="preserve API",
            read_set="src/a.py :: symbol",
            mutable_scope="src/a.py",
            repository_instructions="follow AGENTS.md",
            verify_instructions="python -m unittest",
        )
        repair = build_check_repair_payload(
            spec="SPEC",
            failed_check_ids="CHECK_FAILED:unit",
            failed_check_evidence="only failed evidence",
            compact_contract_invariants="S01 | writes=src/a.py",
            changed_files="src/a.py",
            mutable_scope="src/a.py",
            candidate_identity="tree=" + "b" * 40,
        )
        reviser = build_semantic_revision_payload(
            spec="SPEC",
            compact_approved_contract_index="S01 | writes=src/a.py",
            candidate_identity="tree=" + "c" * 40,
            changed_files="src/a.py",
            required_checks_summary="unit: passed",
            mutable_scope="src/a.py",
            bounded_diff_evidence="bounded excerpt",
        )
        reviewer = build_final_review_payload(
            spec="SPEC",
            compact_approved_plan="S01 | writes=src/a.py",
            required_checks_summary="unit: passed",
            immutable_candidate_identity="sha=" + "d" * 40,
            candidate_remote_reference="https://example.invalid/commit/d",
            changed_files="src/a.py",
            diff_sha256="hash",
            diffstat="1 file",
            bounded_diff_excerpt="bounded excerpt",
            cycle_summary="cycle 001 passed",
        )
        self.assertEqual(
            [planner.role, implementer.role, repair.role, reviser.role, reviewer.role],
            ["planner", "implementer", "check-repair", "semantic-reviser", "final-reviewer"],
        )
        self.assertNotIn("worker transcript", reviser.rendered)
        self.assertNotIn("stdout that may be shortened", reviewer.rendered)
        self.assertIn("SPEC", repair.rendered)
        self.assertIn("SPEC", reviser.rendered)
        self.assertIn("SPEC", reviewer.rendered)

    def test_final_review_authority_stays_effective_after_ten_replans(self) -> None:
        def cycle_plan(number: int, kind: str, path: str) -> SimpleNamespace:
            plan = SimpleNamespace(
                title=f"Plan {number}",
                objective=(
                    "ORIGINAL OBJECTIVE" if number == 1
                    else f"FULL CORRECTION PLAN TEXT {number}"
                ),
                constraints="ORIGINAL CONSTRAINTS",
                required_checks=("unit", "integration"),
                steps=(SimpleNamespace(
                    id="S01",
                    title=f"Step {number}",
                    depends_on=None,
                    objective=f"step objective {number}",
                    write_set=(path,),
                    create_set=(),
                    delete_set=(),
                    verify="run checks",
                    forbidden="do not widen scope",
                ),),
            )
            cycle = SimpleNamespace(
                number=number,
                kind=SimpleNamespace(value=kind),
            )
            return SimpleNamespace(
                cycle=cycle,
                plan=plan,
                correction_bundle_sha256=(None if number == 1 else f"{number:064x}"),
            )

        cycles = [cycle_plan(1, "initial", "src/initial.py")]
        cycles.extend(
            cycle_plan(number, "review-replan", f"src/correction-{number:03d}.py")
            for number in range(2, 11)
        )
        view = EffectivePlanView.from_cycle_plans(cycles[0].plan, cycles)
        effective = view.render()
        payloads = []
        for count in (1, 10):
            history = "\n".join(
                f'{{"cycle": {number}, "route": "REPLAN"}}'
                for number in range(1, count + 1)
            )
            payloads.append(build_final_review_payload(
                spec="EXACT SPEC",
                compact_approved_plan=effective,
                required_checks_summary='{"required_check_ids":["unit","integration"]}',
                immutable_candidate_identity="candidate_sha=" + "a" * 40,
                changed_files="src/initial.py",
                diff_sha256="b" * 64,
                diffstat='{"files":1,"insertions":1,"deletions":0}',
                bounded_diff_excerpt="bounded diff",
                cycle_summary=history,
                budget_bytes=120_000,
            ))

        self.assertNotIn("FULL CORRECTION PLAN TEXT 2", effective)
        self.assertNotIn("worker transcript", payloads[-1].rendered)
        self.assertNotIn("previous full prompt", payloads[-1].rendered)
        self.assertEqual(json.loads(effective)["original_objective"], "ORIGINAL OBJECTIVE")
        self.assertEqual(
            json.loads(effective)["required_deterministic_check_ids"],
            ["unit", "integration"],
        )
        self.assertEqual(
            json.loads(effective)["current_cumulative_approved_mutable_scope"],
            ["src/correction-002.py", "src/correction-003.py", "src/correction-004.py",
             "src/correction-005.py", "src/correction-006.py", "src/correction-007.py",
             "src/correction-008.py", "src/correction-009.py", "src/correction-010.py",
             "src/initial.py"],
        )
        self.assertIn("candidate_sha=" + "a" * 40, payloads[-1].rendered)
        self.assertIn("EXACT SPEC", payloads[-1].rendered)
        self.assertEqual(payloads[-1].budget_overrun, False)
        self.assertLess(payloads[-1].total_bytes, 120_000)
        self.assertLess(payloads[-1].total_bytes - payloads[0].total_bytes, 2_000)
        self.assertEqual(
            payloads[-1].sections[1].truncated,
            False,
        )


if __name__ == "__main__":
    unittest.main()
