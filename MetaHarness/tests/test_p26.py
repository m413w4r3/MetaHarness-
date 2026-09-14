"""P26 bounded second-loop offline end-to-end coverage."""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.claude.agent import ClaudeResult  # noqa: E402
from metaharness.llm.chat import TextLLMResult  # noqa: E402
from metaharness.models import (  # noqa: E402
    ExecutionRole,
    ModelProfile,
    ProfileDriver,
    SelectionMode,
)
from metaharness.orchestrator import Orchestrator  # noqa: E402
from tests.test_p21_multistep import (  # noqa: E402
    FakeAgent,
    MultiStepHarness,
    PASS_REVIEW,
    plan_text,
    step_block,
)


REVISE_IMPLEMENTATION = """VERDICT: REVISE
ROUTE: IMPLEMENTATION
SUMMARY: A concrete defect remains.
FINDINGS: MAJOR | behavior | the candidate is incomplete | implement the fix
REQUIRED FIXES: Correct src/a.py mechanically.
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""
REVISE_REPLAN = REVISE_IMPLEMENTATION.replace("IMPLEMENTATION", "REPLAN")
REVISE_HUMAN = REVISE_IMPLEMENTATION.replace("IMPLEMENTATION", "HUMAN")
BLOCKED = """META PLAN v2

STATUS: BLOCKED
TITLE: Repair is architectural

OBJECTIVE
The requested repair cannot be bounded.

BLOCKERS
The required fix needs an architectural redesign.

END META PLAN
"""


class QueueClient:
    def __init__(self, responses: list[str], usage: dict[str, int] | None = None):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.usage = usage or {"input_tokens": 10, "output_tokens": 2}

    def complete(self, prompt: str) -> TextLLMResult:
        self.prompts.append(prompt)
        return TextLLMResult(
            text=self.responses.pop(0), model="offline", usage=dict(self.usage), raw_response={}
        )


class FakeReviser:
    def run_revision(self, prompt: str, worktree: Path, *, artifacts_dir: Path,
                     profile: ModelProfile, environment: dict[str, str],
                     revision_dir: Path | None = None) -> ClaudeResult:
        target = revision_dir or (Path(artifacts_dir) / "revision")
        target.mkdir(parents=True, exist_ok=True)
        (target / "agent.events.jsonl").write_text("", encoding="utf-8")
        (target / "agent.final.md").write_text("offline Claude report\n", encoding="utf-8")
        return ClaudeResult(0, False, "offline Claude report\n", {"input_tokens": 7, "output_tokens": 3}, "")


class P26Harness(MultiStepHarness):
    def p26_config(self):
        base = self.config(require_approval=False)
        profiles = dict(base.model_profiles)
        profiles["reviser"] = ModelProfile(
            id="reviser", display_name="Reviser", roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE, model="claude-offline",
            selection_mode=SelectionMode.CLI, retries=0, effort="high",
            permission_mode="default",
        )
        profiles["repair"] = ModelProfile(
            id="repair", display_name="Repair", roles=(ExecutionRole.REPAIR,),
            driver=ProfileDriver.CODEX, model="luna-repair",
            selection_mode=SelectionMode.CLI, effort="high", sandbox="workspace-write",
        )
        return dataclasses.replace(
            base,
            model_profiles=profiles,
            ui=dataclasses.replace(
                base.ui, default_reviser_profile="reviser", default_repair_profile="repair"
            ),
        )

    def run_p26(self, repair_plan: str, reviews: list[str], run_id: str = "p26"):
        config = self.p26_config()
        planner = QueueClient([plan_text(step_block(1)), repair_plan])
        reviewer = QueueClient(reviews, usage={"input_tokens": 11, "output_tokens": 4})
        count = {"value": 0}

        def implement(root: Path):
            count["value"] += 1
            (root / "src/a.py").write_text(f"A = {count['value'] + 1}\n", encoding="utf-8")

        orchestrator = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
            agent=FakeAgent(actions={"S01": implement}), reviser=FakeReviser(),
        )
        result = orchestrator.run_text("Implement the feature.", run_id=run_id)
        return result, planner, reviewer

    def test_pass_never_creates_c02(self):
        result, planner, reviewer = self.run_p26(plan_text(step_block(1, profile="repair")), [PASS_REVIEW])
        self.assertEqual(result.status.value, "committed")
        self.assertEqual(len(planner.prompts), 1)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertFalse((result.run_dir / "repair" / "C02").exists())

    def test_revise_implementation_runs_one_repair_cycle(self):
        result, planner, reviewer = self.run_p26(plan_text(step_block(1, profile="repair")), [REVISE_IMPLEMENTATION, PASS_REVIEW])
        self.assertEqual(result.status.value, "committed", result.state.get("failure"))
        self.assertEqual((len(planner.prompts), len(reviewer.prompts)), (2, 2))
        self.assertEqual(result.state["cycle"], 2)
        self.assertEqual(result.state["review_iterations"], 2)
        self.assertTrue((result.run_dir / "repair/C02/steps/S01/agent.final.md").exists())
        self.assertTrue((result.run_dir / "revision/C02/report.json").exists())
        self.assertTrue((result.run_dir / "review/C02/reviewer.raw.md").exists())

    def test_replan_and_human_never_execute_c02(self):
        for index, review in enumerate((REVISE_REPLAN, REVISE_HUMAN), 1):
            with self.subTest(review=review):
                result, planner, reviewer = self.run_p26(plan_text(step_block(1, profile="repair")), [review], run_id=f"p26-{index}")
                expected = "REPLAN_REQUIRED" if "REPLAN" in review else "HUMAN_REQUIRED"
                self.assertEqual(result.failure_reason, expected)
                self.assertEqual(len(planner.prompts), 1)
                self.assertEqual(len(reviewer.prompts), 1)
                self.assertFalse((result.run_dir / "repair/C02").exists())

    def test_scope_expansion_prevents_repair_agent(self):
        repair = plan_text(step_block(1, profile="repair", read=("src/a.py", "src/b.py"), write_set=("src/b.py",)))
        result, planner, reviewer = self.run_p26(repair, [REVISE_IMPLEMENTATION])
        self.assertEqual(result.failure_reason, "REPAIR_SCOPE_EXPANSION")
        self.assertEqual(len(planner.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertFalse((result.run_dir / "repair/C02/steps/S01/agent.result.json").exists())

    def test_blocked_repair_stops(self):
        result, planner, reviewer = self.run_p26(BLOCKED, [REVISE_IMPLEMENTATION])
        self.assertEqual(result.failure_reason, "REPAIR_PLANNER_BLOCKED")
        self.assertEqual(len(planner.prompts), 2)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertFalse((result.run_dir / "repair/C02/implementation_bundle.json").exists())

    def test_second_revise_exhausts_without_c03(self):
        result, _planner, _reviewer = self.run_p26(plan_text(step_block(1, profile="repair")), [REVISE_IMPLEMENTATION, REVISE_IMPLEMENTATION])
        self.assertEqual(result.failure_reason, "REVIEW_LOOP_EXHAUSTED")
        self.assertFalse((result.run_dir / "repair/C03").exists())

    def test_c01_artifacts_are_preserved_and_usage_is_exact(self):
        result, _planner, _reviewer = self.run_p26(plan_text(step_block(1, profile="repair")), [REVISE_IMPLEMENTATION, PASS_REVIEW])
        root_raw = (result.run_dir / "reviewer.raw.md").read_bytes()
        self.assertEqual(root_raw, (result.run_dir / "review/C01/reviewer.raw.md").read_bytes())
        self.assertTrue((result.run_dir / "revision/C01/usage.json").exists())
        usage = json.loads((result.run_dir / "state.json").read_text(encoding="utf-8"))["usage"]
        self.assertEqual(usage["grand_total"]["input_tokens"], 10 + 1000 + 7 + 11 + 10 + 1000 + 7 + 11)
        self.assertIn("luna_c01", usage)
        self.assertIn("reviewer_c02", usage)


if __name__ == "__main__":
    unittest.main()
