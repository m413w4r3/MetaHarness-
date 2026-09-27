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
