"""Focused contracts for the generic pipeline-v2 state machine."""

import tempfile
import unittest
from pathlib import Path

from metaharness.approval import PlanIdentity
from metaharness.execution_selection import (
    SCHEMA_VERSION,
    ensure_execution_selection,
    read_execution_selection,
)
from metaharness.models import ExecutionSelection, RunCycle, SelectedProfile, StepExecutionSelection
from metaharness.orchestration.pipeline_v2 import (
    check_repair_attempt_dir,
    cycle_dir,
    gate_dir,
)
from metaharness.run_options import RunOptions
from metaharness.run_options import SCHEMA_VERSION as RUN_OPTIONS_SCHEMA_VERSION
from metaharness.resume import ResumeCheckpoint, ResumeCheckpointError, ResumePhase


class GenericCheckpointTests(unittest.TestCase):
    def test_review_cycle_is_unbounded_but_positive(self) -> None:
        for cycle in (1, 2, 7):
            checkpoint = ResumeCheckpoint(
                phase=ResumePhase.CONTEXT,
                review_cycle=cycle,
            )
            self.assertEqual(checkpoint.review_cycle, cycle)
        with self.assertRaises(ResumeCheckpointError):
            ResumeCheckpoint(phase=ResumePhase.CONTEXT, review_cycle=0)

    def test_gate_stage_is_part_of_the_generic_gate_identity(self) -> None:
        checkpoint = ResumeCheckpoint(
            phase=ResumePhase.DETERMINISTIC_GATE,
            review_cycle=7,
            stage="POST_REVIEW_REPLAN",
            expected_head_sha="a" * 40,
            expected_tree_sha="b" * 40,
            execution_selection_sha256="c" * 64,
            plan_identity=PlanIdentity("d" * 64, "e" * 64),
        )
        self.assertEqual((checkpoint.review_cycle, checkpoint.stage), (7, "POST_REVIEW_REPLAN"))


class GenericArtifactPathTests(unittest.TestCase):
    def test_paths_accept_arbitrary_positive_cycles(self) -> None:
        root = Path(tempfile.gettempdir()) / "metaharness-generic-test"
        self.assertEqual(cycle_dir(root, 1), root / "cycles/001")
        self.assertEqual(cycle_dir(root, 2), root / "cycles/002")
        self.assertEqual(cycle_dir(root, RunCycle(10, "review-replan")), root / "cycles/010")
        self.assertEqual(
            gate_dir(root, 10, "POST_SEMANTIC_REVISION"),
            root / "cycles/010/checks/post-semantic-revision",
        )
        self.assertEqual(
            check_repair_attempt_dir(root, 10, "POST_IMPLEMENTATION", 2),
            root / "cycles/010/check-repair/post-implementation/attempts/002",
        )


class GenericSnapshotTests(unittest.TestCase):
    def test_run_options_and_selection_use_only_current_role_names(self) -> None:
        options = RunOptions(
            schema_version=RUN_OPTIONS_SCHEMA_VERSION, pipeline_version=2, protocol="v2",
            decomposition="balanced", execution_mode_policy="auto",
            single_step_max_mutable_paths=2, staged_step_max_mutable_paths=6,
            semantic_revision_enabled=False, max_check_repair_attempts=0,
            max_correction_cycles=0, planner_profile="planner",
            mechanical_profile="implementer", reasoning_profile="implementer",
            agentic_profile="implementer", check_repair_profile=None,
            semantic_reviser_profile=None, final_reviewer_profile="reviewer",
        )
        encoded = options.to_dict()
        self.assertNotIn("claude_revision_enabled", encoded)
        self.assertNotIn("repair_cycles", encoded)

        def selected(profile_id: str) -> SelectedProfile:
            return SelectedProfile(
                profile_id=profile_id, driver="codex", provider="openai",
                model="model", effort="high", selection_mode="explicit",
                config_sha256="a" * 64,
            )

        selection = ExecutionSelection(
            schema_version=SCHEMA_VERSION, planner=selected("planner"),
            steps=(StepExecutionSelection("S01", selected("implementer")),),
            check_repair=None, semantic_reviser=None,
            final_reviewer=selected("final-reviewer"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ensure_execution_selection(path, selection)
            durable = read_execution_selection(path)
            payload = (path / "execution_selection.json").read_text(encoding="utf-8")
        self.assertEqual(durable, selection)
        self.assertNotIn('"reviser"', payload)
        self.assertNotIn('"repair_implementer"', payload)
        self.assertNotIn('"reviewer"', payload)


if __name__ == "__main__":
    unittest.main()
