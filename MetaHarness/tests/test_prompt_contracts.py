import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metaharness.prompt_contracts import (
    PromptPayload,
    build_correction_payload,
    build_implementer_payload,
    build_planner_payload,
    write_prompt_diagnostics,
)


class PromptContractTests(unittest.TestCase):
    def test_worker_references_complete_spec_without_repeating_it(self) -> None:
        spec = "semantic authority " * 20_000
        payload = build_implementer_payload(
            original_spec=spec, spec_path="/run/authority/spec.md",
            instructions="preserve the exact invariant", budget_bytes=8_000,
        )
        self.assertNotIn(spec, payload.rendered)
        self.assertIn("/run/authority/spec.md", payload.rendered)
        self.assertIn(hashlib.sha256(spec.encode()).hexdigest(), payload.rendered)
        self.assertLess(payload.total_bytes, 8_000)

    def test_corrections_bound_utf8_output_without_losing_authority(self) -> None:
        payload = build_correction_payload(
            "original SPEC and contract", "fix missing path", "é" * 50_000,
            role="planner-correction", budget_bytes=2_000,
        )
        self.assertLessEqual(payload.total_bytes, 2_000)
        self.assertIn("original SPEC and contract", payload.rendered)
        self.assertIn("fix missing path", payload.rendered)
        self.assertFalse(payload.budget_overrun)

    def test_oversized_authority_is_preserved_with_soft_overrun(self) -> None:
        payload = build_implementer_payload(instructions="x" * 10_000, budget_bytes=1_000)
        self.assertTrue(payload.budget_overrun)
        self.assertIn("x" * 10_000, payload.rendered)

    def test_role_prompt_static_size_limits(self) -> None:
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        self.assertLess((prompts / "implementer.txt").stat().st_size, 4 * 1024)

    def test_implementer_is_executor_only(self) -> None:
        payload = build_implementer_payload(
            step_title="S01",
            context="src/a.py exposes the client contract",
            read_set="src/a.py :: symbol",
            mutable_scope="src/a.py",
            instructions="follow AGENTS.md",
            verify_contract="python -m unittest",
            pitfalls="Do not broaden the retry policy.",
        )
        self.assertIn("You are the implementation executor", payload.rendered)
        self.assertIn("Do not redesign the plan or broaden the task.", payload.rendered)
        self.assertIn("Implement the supplied contract exactly.", payload.rendered)

    def test_implementer_limits_repository_exploration(self) -> None:
        payload = build_implementer_payload(step_title="S01")
        rendered = " ".join(payload.rendered.split())
        self.assertIn("Do not chase transitive imports", rendered)
        self.assertIn("return BLOCKED with the exact contract gap", rendered)

    def test_planner_resolves_dependencies_before_worker_contract(self) -> None:
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        planner = " ".join((prompts / "planner_v2.txt").read_text(encoding="utf-8").split())
        self.assertIn("confirm the named contract supports", planner)
        self.assertIn("Never ask the worker to", planner)
        self.assertIn("the planner must declare it", planner)
        self.assertIn("Sharing one source file or test file", planner)
        self.assertIn("reject noncanonical input forms", planner)
        self.assertIn("Do not repeat full lint, typecheck", planner)

    def test_pitfalls_are_single_nontruncatable_authority(self) -> None:
        pitfalls = "DO NOT CREATE migration 0002"
        payload = build_implementer_payload(
            step_identity="S04", context="remove legacy state",
            read_set="NONE", mutable_scope="NONE", pitfalls=pitfalls,
            budget_bytes=1,
        )
        self.assertIn(f"<PITFALLS>\n{pitfalls}\n</PITFALLS>", payload.rendered)
        self.assertEqual(payload.rendered.count(pitfalls), 1)
        section = next(item for item in payload.sections if item.name == "pitfalls")
        self.assertTrue(section.authority)
        self.assertFalse(section.truncated)
        self.assertEqual(section.sha256, hashlib.sha256(pitfalls.encode()).hexdigest())

    def test_planner_authority_and_diagnostics_are_bounded(self) -> None:
        payload = build_planner_payload(
            spec="SPEC" * 100, repository_identity="BASE SHA",
            discovery_context="indexed files" * 1000,
            trusted_check_catalogue="unit", planning_constraints="NONE",
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


if __name__ == "__main__":
    unittest.main()
