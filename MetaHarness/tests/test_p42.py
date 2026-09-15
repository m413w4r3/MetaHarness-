"""P42 trusted check catalogue and planner-selected gate regressions."""

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from metaharness.config import load_config
from metaharness.models import AgentConfig, CheckConfig, ContextConfig, HarnessConfig, LLMEndpointConfig, RunStatus
from metaharness.planning_v2 import (
    V2PlanParseError,
    build_planner_prompt_v2,
    parse_task_plan_v2,
)
from metaharness.resume import ResumePhase, read_checkpoint
from metaharness.validation import run_check_preflights
from tests.test_p29 import (
    P29Harness, FakeLuna, SINGLE_PLAN, REPAIR_PLAN, PASS, REVISE_IMPLEMENTATION, writer,
)


ROOT = Path(__file__).parent / "fixtures"


def with_required_checks(plan: str, *check_ids: str) -> str:
    section = "REQUIRED_CHECKS\n" + "".join(f"- {check_id}\n" for check_id in check_ids)
    return plan.replace("CONSTRAINTS\nNONE\n", f"CONSTRAINTS\nNONE\n\n{section}", 1)


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


class P42RepairPreflightTests(P29Harness):
    def integration_config(self):
        self.marker = self.root / "integration-ready"
        probe = self.root / "integration_probe.py"
        probe.write_text("import os, sys\nsys.exit(0 if os.path.exists(sys.argv[1]) else 3)\n",
                         encoding="utf-8")
        text = self.config_text(require_approval=True)
        text = text.replace("require_clean_base = true",
                            'require_clean_base = true\ndefault_check_ids = ["gate"]', 1)
        text = text.replace('[[checks]]\nname = "gate"', '[[check_catalog]]\nid = "gate"', 1)
        text += textwrap.dedent(f"""
            [[check_catalog]]
            id = "integration"
            argv = [{sys.executable!r}, "-c", "pass"]
            preflight_argv = [{sys.executable!r}, {str(probe)!r}, {str(self.marker)!r}]
            timeout_seconds = 30
            """)
        path = self.root / "p42.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_repair_plan_preflight_runs_before_c02_workers_and_resumes_same_c02(self) -> None:
        config = self.integration_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 3\n"),
        })
        orchestrator, planner, _reviewer, _luna, claude = self.orchestrator(
            config,
            plans=[with_required_checks(SINGLE_PLAN, "gate"),
                   with_required_checks(REPAIR_PLAN, "gate", "integration")],
            reviews=[REVISE_IMPLEMENTATION], luna=luna,
        )
        failed = self.run_approved(config, orchestrator, "repair-preflight")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(failed.state["failure"]["reason"], "CHECK_PREFLIGHT_FAILED:integration")
        self.assertEqual(failed.state["deterministic_gate"]["required_check_ids"], ["gate"])
        self.assertEqual(len(planner.prompts), 2)
        self.assertEqual([call["cycle"] for call in luna.calls], [1])
        self.assertEqual([call["cycle"] for call in claude.calls], [1])
        run_dir = failed.run_dir
        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.step_id), (ResumePhase.REPAIR_STEP, "S01"))
        self.assertEqual(checkpoint.expected_head_sha, c01["commit_sha"])

        # Environment still broken: the cheap preflight fails again before
        # any worker, and the repair planner is never replayed.
        again, again_planner, _r, again_luna, again_claude = self.orchestrator(config)
        still = again.resume("repair-preflight")
        self.assertEqual(still.state["failure"]["reason"], "CHECK_PREFLIGHT_FAILED:integration")
        self.assertEqual((again_planner.prompts, again_luna.calls, again_claude.calls), ([], [], []))
        self.assertEqual(read_checkpoint(run_dir), checkpoint)

        self.marker.write_text("ready\n", encoding="utf-8")
        healthy, healthy_planner, _r, healthy_luna, healthy_claude = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({(2, "S01"): writer("src/a.py", "A = 3\n")}),
        )
        resumed = healthy.resume("repair-preflight")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(healthy_planner.prompts, [])
        self.assertEqual([(call["cycle"], call["step"]) for call in healthy_luna.calls], [(2, "S01")])
        self.assertEqual([call["cycle"] for call in healthy_claude.calls], [2])
        self.assertEqual(resumed.state["deterministic_gate"]["required_check_ids"], ["gate", "integration"])
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())
        self.assertEqual(c02["parent_sha"], c01["commit_sha"])


if __name__ == "__main__":
    unittest.main()
