import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metaharness.prompt_contracts import (
    PromptPayload,
    build_check_repair_payload,
    build_final_review_payload,
    build_implementer_payload,
    build_planner_payload,
    build_semantic_revision_payload,
    write_prompt_diagnostics,
)


class PromptContractTests(unittest.TestCase):
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
            cycle_summary="C01 passed",
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


if __name__ == "__main__":
    unittest.main()
