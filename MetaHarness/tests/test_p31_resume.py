"""P31: resume boundaries that exist before the approved worktree."""

from __future__ import annotations

import json
import unittest
from unittest import mock
from pathlib import Path

from metaharness.config import load_config
from metaharness.approval import PlanIdentity
from metaharness.orchestrator import Orchestrator
import metaharness.orchestrator as orchestrator_module
from metaharness.resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumePhase,
    checkpoint_payload,
    read_checkpoint,
    resume_info,
    write_checkpoint,
)
from metaharness.state import RunStateStore

from tests.test_p29 import FakeClaude, FakeLuna, P29Harness, PASS, QueueClient, SINGLE_PLAN, SPEC, writer


class P31ResumeTests(P29Harness):
    def test_pre_approval_checkpoint_does_not_require_approval_or_worktree(self) -> None:
        run_dir = self.runs / "pre-approval"
        run_dir.mkdir(parents=True)
        state = RunStateStore(run_dir / "state.json")
        state.initialize("pre-approval")
        state.update(status="failed", planning_protocol="v2", failure={"reason": "PLANNER_OUTPUT_INVALID"})
        write_checkpoint(run_dir, ResumeCheckpoint(ResumePhase.PLANNER, 1, None, None, None, None, None))

        info = resume_info(run_dir, state.load())
        self.assertTrue(info.resumable)
        self.assertEqual((info.phase, info.label), ("planner", "Retry planner"))

    def test_planner_transport_failure_retries_planner_only(self) -> None:
        config_path = self.root / "p31-no-approval.toml"
        config_path.write_text(self.config_text(require_approval=False, revision=False), encoding="utf-8")
        config = load_config(config_path)
        first_planner = QueueClient("planner", [], self.events)
        first = Orchestrator(
            config,
            planner_client=first_planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviser=FakeClaude(log=self.events),
        )
        failed = first.run_text(SPEC, run_id="planner-retry")
        self.assertEqual(failed.state["failure"]["reason"], "INDEXERROR")
        checkpoint = read_checkpoint(failed.run_dir)
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.phase, ResumePhase.PLANNER)
        self.assertTrue(resume_info(failed.run_dir, failed.state).resumable)

        retry_planner = QueueClient("planner", [SINGLE_PLAN], self.events)
        resumed = Orchestrator(
            config,
            planner_client=retry_planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviser=FakeClaude(log=self.events),
        ).resume("planner-retry")
        self.assertEqual(resumed.status.value, "published", resumed.state.get("failure"))
        self.assertEqual(len(first_planner.prompts), 1)
        self.assertEqual(len(retry_planner.prompts), 1)
        self.assertTrue((failed.run_dir / "attempts" / "01" / "planner.request.txt").exists())

    def test_candidate_commit_is_reconciled_without_a_duplicate(self) -> None:
        config_path = self.root / "p31-no-publish.toml"
        config_path.write_text(
            self.config_text(require_approval=False, revision=False, publish=False),
            encoding="utf-8",
        )
        config = load_config(config_path)
        planner = QueueClient("planner", [SINGLE_PLAN], self.events)
        original = orchestrator_module.commit_candidate_tree
        crashed = False

        def commit_then_crash(*args: object, **kwargs: object) -> str:
            nonlocal crashed
            value = original(*args, **kwargs)
            if not crashed:
                crashed = True
                raise RuntimeError("simulated state update crash")
            return value

        with mock.patch.object(orchestrator_module, "commit_candidate_tree", side_effect=commit_then_crash):
            failed = Orchestrator(
                config,
                planner_client=planner,
                reviewer_client=QueueClient("reviewer", [PASS], self.events),
                agent=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
                reviser=FakeClaude(log=self.events),
            ).run_text(SPEC, run_id="commit-reconcile")
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual(read_checkpoint(failed.run_dir).phase, ResumePhase.CANDIDATE_COMMIT_C01)

        resumed = Orchestrator(
            config,
            planner_client=QueueClient("planner", [], self.events),
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=FakeLuna({}),
            reviser=FakeClaude(log=self.events),
        ).resume("commit-reconcile")
        self.assertEqual(resumed.status.value, "committed", resumed.state.get("failure"))
        self.assertEqual(
            __import__("subprocess").run(
                ["git", "-C", str(self.worktree("commit-reconcile")), "rev-list", "--count", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "2",
        )


class P31CheckpointCompatibilityTests(unittest.TestCase):
    def test_schema_one_checkpoint_remains_readable(self) -> None:
        with self.subTest("legacy payload"):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                identity = PlanIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64)
                checkpoint = ResumeCheckpoint(
                    ResumePhase.CLAUDE_C01, 1, None, "1" * 40, "2" * 40,
                    "d" * 64, identity,
                )
                payload = checkpoint_payload(checkpoint)
                payload["schema_version"] = 1
                (path / "resume_checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(read_checkpoint(path), checkpoint)

    def test_step_checkpoints_accept_s07_s08_s37_and_reject_s100(self) -> None:
        import tempfile

        identity = PlanIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64)
        phases = (
            (ResumePhase.INITIAL_STEP, 1, {}),
            (ResumePhase.REPAIR_STEP, 2, {"repair_bundle_sha256": "e" * 64}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for phase, cycle, extra in phases:
                for step_id in ("S07", "S08", "S37"):
                    with self.subTest(phase=phase.value, step=step_id):
                        checkpoint = ResumeCheckpoint(
                            phase, cycle, step_id, "1" * 40, "2" * 40, "d" * 64, identity, **extra
                        )
                        write_checkpoint(path, checkpoint)
                        self.assertEqual(read_checkpoint(path), checkpoint)
                for step_id in ("S00", "S100", "s37"):
                    with self.subTest(phase=phase.value, step=step_id):
                        with self.assertRaises(ResumeCheckpointError):
                            ResumeCheckpoint(
                                phase, cycle, step_id, "1" * 40, "2" * 40, "d" * 64, identity, **extra
                            )
                payload = checkpoint_payload(ResumeCheckpoint(
                    phase, cycle, "S08", "1" * 40, "2" * 40, "d" * 64, identity, **extra
                ))
                payload["step_id"] = "S100"
                (path / "resume_checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ResumeCheckpointError):
                    read_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
