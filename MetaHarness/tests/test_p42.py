"""P42 trusted check catalogue and planner-selected gate regressions."""

import sys
import tempfile
import unittest
from pathlib import Path

from metaharness.models import AgentConfig, CheckConfig, ContextConfig, HarnessConfig, LLMEndpointConfig
from metaharness.planning_v2 import (
    V2PlanParseError,
    build_planner_prompt_v2,
    parse_task_plan_v2,
)
from metaharness.validation import run_check_preflights


ROOT = Path(__file__).parent / "fixtures"


class P42RequiredCheckTests(unittest.TestCase):
    def catalogue(self) -> tuple[CheckConfig, ...]:
        return (
            CheckConfig("lint", ("make", "lint"), description="Lint"),
            CheckConfig("typecheck", ("make", "typecheck"), description="Types"),
            CheckConfig("test", ("make", "test")),
            CheckConfig("test-integration", ("make", "test-integration"), preflight_argv=("docker", "info")),
            CheckConfig("alembic-heads", ("uv", "run", "alembic", "heads"), cwd="backend"),
        )

    def parse(self, text: str):
        return parse_task_plan_v2(
            text,
            implementer_ids=frozenset({"impl-a"}),
            reviewer_ids=frozenset({"review-a"}),
            check_catalog=self.catalogue(),
            default_check_ids=("lint", "typecheck", "test"),
        )

    def test_aw001_selects_all_explicit_validation_gates_in_catalogue_order(self):
        plan = self.parse((ROOT / "aw001_required_checks_plan.md").read_text())
        self.assertEqual(
            plan.required_checks,
            ("lint", "typecheck", "test", "test-integration", "alembic-heads"),
        )

    def test_unknown_and_missing_default_ids_are_rejected(self):
        raw = (ROOT / "aw001_required_checks_plan.md").read_text()
        with self.assertRaises(V2PlanParseError):
            self.parse(raw.replace("- alembic-heads", "- not-trusted"))
        with self.assertRaises(V2PlanParseError):
            self.parse(raw.replace("- lint\n", "", 1))

    def test_prompt_exposes_only_safe_catalogue_metadata(self):
        prompt = build_planner_prompt_v2(
            (ROOT / "aw001_required_checks_spec.md").read_text(),
            "context",
            check_catalog=self.catalogue(),
            default_check_ids=("lint", "typecheck", "test"),
        )
        self.assertIn("ID: test-integration", prompt)
        self.assertIn("DESCRIPTION: Lint", prompt)
        self.assertNotIn('"make", "lint"', prompt)
        self.assertNotIn('"docker", "info"', prompt)
        self.assertNotIn('"uv", "run"', prompt)

    def test_selected_preflight_fails_with_stable_reason_before_checks(self):
        with self.subTest("docker-style unavailable preflight"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
                config = HarnessConfig(
                    repo=root,
                    base_ref="HEAD",
                    runs_root=root / "runs",
                    worktrees_root=root / "worktrees",
                    require_clean_base=True,
                    planner=endpoint,
                    reviewer=endpoint,
                    context=ContextConfig(),
                    agent=AgentConfig(),
                    checks=(),
                    check_catalog=(CheckConfig(
                        "test-integration", ("true",),
                        preflight_argv=(sys.executable, "-c", "raise SystemExit(7)"),
                    ),),
                    default_check_ids=("test-integration",),
                )
                self.assertEqual(
                    run_check_preflights(root, config, ("test-integration",)),
                    ("CHECK_PREFLIGHT_FAILED:test-integration",),
                )


if __name__ == "__main__":
    unittest.main()
