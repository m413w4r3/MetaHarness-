"""P30 durable per-run option snapshots."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.run_options import (  # noqa: E402
    RunOptions,
    RunOptionsConflict,
    RunOptionsError,
    effective_run_config,
    legacy_or_durable_run_options,
    read_run_options_with_sha256,
    write_run_options,
)
from metaharness.web.api import WebAPIError, create_run  # noqa: E402
from metaharness.web.pages import render_new_run, render_run  # noqa: E402
from tests.test_p29 import (  # noqa: E402
    PASS,
    REPAIR_PLAN,
    REVISE_IMPLEMENTATION,
    SINGLE_PLAN,
    SPEC,
    P29Harness,
)
from tests.test_p28_full_pipeline import FakeClaude, FakeLuna, QueueClient, writer  # noqa: E402
from metaharness.orchestrator import Orchestrator  # noqa: E402
from metaharness.web.api import approve_run  # noqa: E402


class _Manager:
    def __init__(self, config):
        self._config = config
        self.options = None

    def start_run(self, spec, *, run_id=None, run_options=None):
        self.options = run_options
        return run_id or "run"


class RunOptionsTests(unittest.TestCase):
    def setUp(self):
        self.harness = P29Harness()
        self.harness.setUp()
        self.config = self.harness.make_config(revision=True)

    def tearDown(self):
        self.harness.tearDown()

    def test_default_round_trip_hash_and_immutable(self):
        options = RunOptions.from_config(self.config)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            digest = write_run_options(run_dir, options)
            loaded, loaded_digest = read_run_options_with_sha256(run_dir, digest)
            self.assertEqual(loaded, options)
            self.assertEqual(digest, loaded_digest)
            changed = RunOptions.from_config(self.config, staged_step_max_mutable_paths=9)
            with self.assertRaises(RunOptionsConflict):
                write_run_options(run_dir, changed)

    def test_malformed_and_invalid_profile_snapshots_are_rejected(self):
        payload = RunOptions.from_config(self.config).to_dict()
        payload["pipeline"]["repair_cycles"] = 2
        with self.assertRaises(RunOptionsError):
            RunOptions.from_mapping(payload)
        with self.assertRaises(RunOptionsError):
            RunOptions.from_config(self.config, reviewer_profile="luna")

    def test_effective_config_isolated_and_legacy_fallback(self):
        options = RunOptions.from_config(
            self.config, decomposition="aggressive", execution_mode_policy="require-staged",
            staged_step_max_mutable_paths=9,
        )
        effective = effective_run_config(self.config, options)
        self.assertEqual(effective.planning.staged_step_max_mutable_paths, 9)
        self.assertNotEqual(self.config.planning.staged_step_max_mutable_paths, 9)
        with tempfile.TemporaryDirectory() as directory:
            fallback, digest = legacy_or_durable_run_options(self.config, directory)
        self.assertEqual(
            fallback,
            RunOptions.from_config(self.config, repair_scope_policy="deny-expansion"),
        )
        self.assertIsNone(digest)

    def test_new_run_html_contains_safe_controls(self):
        html = render_new_run(self.config, "token")
        for name in (
            "RUN OPTIONS", "default_implementer_profile", "reviewer_profile",
            "claude_revision_enabled", "reviser_profile", "repair_cycles",
            "repair_profile", "decomposition", "execution_mode_policy",
            "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
        ):
            self.assertIn(name, html)
        self.assertNotIn("api_key_env", html)
        self.assertNotIn("base_url", html)

    def test_form_and_api_shapes_build_identical_options(self):
        expected = RunOptions.from_config(
            self.config, claude_revision_enabled=False, repair_cycles=0,
            decomposition="aggressive", execution_mode_policy="require-staged",
            single_step_max_mutable_paths=3, staged_step_max_mutable_paths=9,
        )
        common = dict(
            spec="do it", run_id="x", planner_profile=expected.planner_profile,
            default_implementer_profile=expected.default_implementer_profile,
            reviewer_profile=expected.reviewer_profile, reviser_profile=expected.reviser_profile,
            repair_profile=expected.repair_profile, decomposition="aggressive",
            execution_mode_policy="require-staged", single_step_max_mutable_paths=3,
            staged_step_max_mutable_paths=9,
        )
        api_manager = _Manager(self.config)
        create_run(api_manager, **common, claude_revision_enabled=False, repair_cycles=0)
        form_manager = _Manager(self.config)
        create_run(form_manager, **common, claude_revision_enabled="disabled", repair_cycles="0")
        self.assertEqual(api_manager.options, expected)
        self.assertEqual(form_manager.options, expected)

    def test_unknown_create_field_is_not_accepted_by_core_signature(self):
        with self.assertRaises(TypeError):
            create_run(_Manager(self.config), spec="do it", unknown=True)

    def test_run_page_shows_requested_configuration(self):
        options = RunOptions.from_config(self.config)
        state = {
            "run_id": "r", "status": "created", "run_options": options.to_dict(),
            "planner": {}, "execution": {}, "failure": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            run = {"run_id": "r", "state": state, "overview": {}}
            html = render_run(run, config=self.config)
        self.assertIn("RUN CONFIGURATION", html)
        self.assertIn("requested planner", html)

    def test_all_claude_and_repair_combinations_are_independent(self):
        cases = {
            (False, 0): ([PASS], [], "published"),
            (True, 0): ([REVISE_IMPLEMENTATION], [], "failed"),
            (False, 1): ([REVISE_IMPLEMENTATION, PASS], [REPAIR_PLAN], "published"),
            (True, 1): ([REVISE_IMPLEMENTATION, PASS], [REPAIR_PLAN], "published"),
        }
        for index, ((claude, repair), (reviews, plans, expected)) in enumerate(cases.items()):
            with self.subTest(claude=claude, repair=repair):
                run_id = f"combo-{index}"
                options = RunOptions.from_config(
                    self.config,
                    claude_revision_enabled=claude,
                    repair_cycles=repair,
                )
                orchestrator = Orchestrator(
                    self.config,
                    planner_client=QueueClient("planner", [SINGLE_PLAN, *plans], self.harness.events),
                    reviewer_client=QueueClient("reviewer", reviews, self.harness.events),
                    agent=FakeLuna({(1, "S01"): writer("src/a.py", f"A = {index + 2}\n"), (2, "S01"): writer("src/a.py", f"A = {index + 3}\n")}),
                    reviser=FakeClaude(log=self.harness.events),
                )
                result_holder = {}
                thread = threading.Thread(
                    target=lambda: result_holder.setdefault(
                        "result", orchestrator.run_text(SPEC, run_id=run_id, run_options=options)
                    ),
                    daemon=True,
                )
                thread.start()
                run_dir = self.config.runs_root / run_id
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if run_dir.joinpath("state.json").exists():
                        state = json.loads(run_dir.joinpath("state.json").read_text())
                        if state.get("status") == "awaiting_plan_approval":
                            break
                    time.sleep(0.005)
                approval = {
                    "config": self.config,
                    "reviewer_profile": options.reviewer_profile,
                    "step_profiles": {"S01": options.default_implementer_profile},
                }
                if claude or repair:
                    approval.update(reviser_profile=options.reviser_profile, repair_profile=options.repair_profile)
                approve_run(self.config.runs_root, run_id, "APPROVE", **approval)
                thread.join(20)
                self.assertFalse(thread.is_alive())
                self.assertEqual(result_holder["result"].state["status"], expected)
                self.assertEqual(len(orchestrator._injected_reviser.calls), int(claude) * (1 + repair))


if __name__ == "__main__":
    unittest.main()
