"""Contracts for configurable execution roles, budgets and fingerprints."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.execution_selection import (  # noqa: E402
    ExecutionSelectionError,
    resolve_execution_selection_v5,
)
from metaharness.models import (  # noqa: E402
    ExecutionRole,
    ExecutionSelectionV5,
    ProfileDriver,
    RevisionConfig,
    RunCycle,
)
from metaharness.profiles import profile_execution_fingerprint  # noqa: E402
from metaharness.profiles import ProfileError  # noqa: E402
from metaharness.run_options import RunOptions  # noqa: E402
from tests.test_execution_profiles import (  # noqa: E402
    codex_profile,
    make_config,
    openai_profile,
)


class ConfigurableExecutionTests(unittest.TestCase):
    def test_revision_budgets_accept_independent_values(self) -> None:
        for attempts in (0, 1, 3):
            with self.subTest(max_check_repair_attempts=attempts):
                self.assertEqual(
                    RevisionConfig(max_check_repair_attempts=attempts).max_check_repair_attempts,
                    attempts,
                )
        for cycles in (0, 1, 4):
            with self.subTest(max_review_repair_cycles=cycles):
                self.assertEqual(
                    RevisionConfig(max_review_repair_cycles=cycles).max_review_repair_cycles,
                    cycles,
                )

    def test_revision_budget_bounds_are_rejected(self) -> None:
        for field in ("max_check_repair_attempts", "max_review_repair_cycles"):
            for value in (-1, 11):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        RevisionConfig(**{field: value})

    def test_run_options_v2_round_trip_is_stable(self) -> None:
        options = RunOptions(
            schema_version=2,
            pipeline_version=2,
            protocol="v2",
            decomposition="balanced",
            execution_mode_policy="auto",
            single_step_max_mutable_paths=2,
            staged_step_max_mutable_paths=6,
            semantic_revision_enabled=True,
            max_check_repair_attempts=3,
            max_review_repair_cycles=4,
            planner_profile="planner",
            default_implementer_profile="implementer",
            check_repair_profile="repair",
            semantic_reviser_profile="reviser",
            final_reviewer_profile="reviewer",
        )
        raw = options.to_dict()
        self.assertEqual(RunOptions.from_mapping(raw), options)
        self.assertEqual(RunOptions.from_mapping(raw).to_dict(), raw)
        self.assertNotIn("max_cycles", str(raw))

    def test_historical_run_options_v1_maps_aliases_without_rewrite(self) -> None:
        raw = {
            "schema_version": 1,
            "planning": {
                "protocol": "v2",
                "decomposition": "balanced",
                "execution_mode_policy": "auto",
                "single_step_max_mutable_paths": 2,
                "staged_step_max_mutable_paths": 6,
            },
            "pipeline": {
                "claude_revision_enabled": True,
                "repair_cycles": 1,
            },
            "profiles": {
                "planner_profile": "planner",
                "default_implementer_profile": "implementer",
                "reviewer_profile": "reviewer",
                "reviser_profile": "reviser",
                "repair_profile": "repair",
            },
        }
        options = RunOptions.from_mapping(raw)
        self.assertTrue(options.semantic_revision_enabled)
        self.assertEqual(options.max_review_repair_cycles, 1)
        self.assertEqual(options.check_repair_profile, "repair")
        self.assertEqual(options.final_reviewer_profile, "reviewer")
        self.assertEqual(options.to_dict()["schema_version"], 1)
        self.assertIn("claude_revision_enabled", options.to_dict()["pipeline"])

    def test_v5_uses_declared_roles_without_driver_coupling(self) -> None:
        profiles = {
            "planner": openai_profile("planner", (ExecutionRole.PLANNER,)),
            "implementer": codex_profile("implementer"),
            "reviewer": openai_profile("reviewer", (ExecutionRole.REVIEWER,)),
            # A Codex backend is allowed to be the semantic reviser when the
            # profile explicitly declares that business role.
            "reviser": codex_profile(
                "reviser", roles=(ExecutionRole.REVISER,), effort="medium"
            ),
            # A Claude backend is allowed to be check-repair by declaration.
            "repair": dataclasses.replace(
                codex_profile("repair"),
                roles=(ExecutionRole.REPAIR,),
                driver=ProfileDriver.CLAUDE_CODE,
                sandbox=None,
                permission_mode="acceptEdits",
                retries=0,
            ),
        }
        config = make_config(Path(self.enterContext(tempfile.TemporaryDirectory())), profiles)
        selection = resolve_execution_selection_v5(
            config,
            planner_profile_id="planner",
            step_profile_ids={"S01": "implementer"},
            check_repair_profile_id="repair",
            semantic_reviser_profile_id="reviser",
            final_reviewer_profile_id="reviewer",
        )
        self.assertIsInstance(selection, ExecutionSelectionV5)
        self.assertEqual(selection.check_repair.driver, ProfileDriver.CLAUDE_CODE.value)
        self.assertEqual(selection.semantic_reviser.driver, ProfileDriver.CODEX.value)

    def test_v5_rejects_profiles_missing_declared_roles(self) -> None:
        config = make_config(Path(self.enterContext(tempfile.TemporaryDirectory())))
        with self.assertRaises((ExecutionSelectionError, ProfileError)):
            resolve_execution_selection_v5(
                config,
                planner_profile_id="planner",
                step_profile_ids={"S01": "impl-a"},
                check_repair_profile_id="impl-a",
                semantic_reviser_profile_id=None,
                final_reviewer_profile_id="review-a",
            )
        with self.assertRaises((ExecutionSelectionError, ProfileError)):
            resolve_execution_selection_v5(
                config,
                planner_profile_id="planner",
                step_profile_ids={"S01": "impl-a"},
                check_repair_profile_id=None,
                semantic_reviser_profile_id="impl-a",
                final_reviewer_profile_id="review-a",
            )

    def test_fingerprint_changes_for_driver_provider_model_and_effort(self) -> None:
        base = codex_profile("fingerprint")
        original = profile_execution_fingerprint(base)
        for label, changed in (
            ("effort", dataclasses.replace(base, effort="low")),
            ("provider", dataclasses.replace(base, provider="deepseek")),
            ("model", dataclasses.replace(base, model="another-model")),
            (
                "driver",
                dataclasses.replace(
                    base,
                    driver=ProfileDriver.CLAUDE_CODE,
                    roles=(ExecutionRole.REVISER,),
                    sandbox=None,
                    permission_mode="acceptEdits",
                    retries=0,
                ),
            ),
        ):
            with self.subTest(field=label):
                self.assertNotEqual(profile_execution_fingerprint(changed), original)

    def test_run_cycles_are_not_limited_to_initial_and_repair(self) -> None:
        self.assertEqual(RunCycle(3, "repair").number, 3)

    def test_execution_selection_v5_is_immutable(self) -> None:
        config = make_config(Path(self.enterContext(tempfile.TemporaryDirectory())))
        selection = resolve_execution_selection_v5(
            config,
            planner_profile_id="planner",
            step_profile_ids={"S01": "impl-a"},
            check_repair_profile_id=None,
            semantic_reviser_profile_id=None,
            final_reviewer_profile_id="review-a",
        )
        with self.assertRaises(FrozenInstanceError):
            selection.schema_version = 6  # type: ignore[misc]
        self.assertIsInstance(selection.steps, tuple)


if __name__ == "__main__":
    unittest.main()
