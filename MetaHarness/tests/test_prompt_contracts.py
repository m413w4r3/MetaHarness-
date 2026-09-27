import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metaharness.prompt_contracts import (
    PromptPayload,
    build_implementer_payload,
    build_planner_payload,
    write_prompt_diagnostics,
)


class PromptContractTests(unittest.TestCase):
    def test_role_prompt_static_size_limits(self) -> None:
        prompts = Path(__file__).resolve().parents[1] / "src" / "metaharness" / "prompts"
        self.assertLess((prompts / "implementer.txt").stat().st_size, 4 * 1024)

    def test_implementer_is_executor_only(self) -> None:
        payload = build_implementer_payload(
            step_title="S01", step_objective="implement one step",
            forbidden_contract="preserve API", read_set="src/a.py :: symbol",
            mutable_scope="src/a.py", repository_instructions="follow AGENTS.md",
            verify_instructions="python -m unittest",
        )
        self.assertIn("You are the implementation executor", payload.rendered)
        self.assertIn("Do not redesign the plan or broaden the task.", payload.rendered)
        self.assertIn("Implement the supplied contract exactly.", payload.rendered)

    def test_forbidden_is_single_nontruncatable_authority(self) -> None:
        forbidden = "DO NOT CREATE migration 0002"
        payload = build_implementer_payload(
            step_identity="S04", step_objective="remove legacy state",
            read_set="NONE", mutable_scope="NONE", forbidden_contract=forbidden,
            budget_bytes=1,
        )
        self.assertIn(f"<FORBIDDEN CONTRACT>\n{forbidden}\n</FORBIDDEN CONTRACT>", payload.rendered)
        self.assertEqual(payload.rendered.count(forbidden), 1)
        section = next(item for item in payload.sections if item.name == "forbidden_contract")
        self.assertTrue(section.authority)
        self.assertFalse(section.truncated)
        self.assertEqual(section.sha256, hashlib.sha256(forbidden.encode()).hexdigest())

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
