"""P31: resume boundaries that exist before the approved worktree."""

from __future__ import annotations

import json
import hashlib
import unittest
from unittest import mock
from pathlib import Path

from metaharness.config import load_config
from metaharness.approval import PlanIdentity, write_scope_approval
from metaharness.models import RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.run_options import RunOptions
import metaharness.orchestrator as orchestrator_module
from metaharness.resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeIntegrityError,
    ResumePhase,
    checkpoint_payload,
    read_checkpoint,
    resume_info,
    write_checkpoint,
)
from metaharness.state import RunStateStore

from tests.test_p29 import (
    FakeClaude, FakeLuna, P29Harness, PASS, QueueClient, REVISE_IMPLEMENTATION,
    SINGLE_PLAN, SPEC, plan_text, step_block, write, writer,
)


TWO_STEP_PLAN = plan_text(
    step_block(1),
    step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
    title="P31 two-step plan",
)


class P31ResumeTests(P29Harness):
    def test_a_completed_step_is_reconciled_instead_of_being_replayed(self) -> None:
        """A crash between a step's durable record and its checkpoint.

        The worker of ``S01`` already succeeded and its record is durable, so
        the resume advances the boundary to ``S02`` without invoking that
        worker a second time.
        """

        config = self.make_config()
        actions = {
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (1, "S02"): writer("src/b.py", "B = 2\n"),
        }
        luna = FakeLuna(dict(actions))
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[TWO_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        real = Orchestrator._checkpoint
        crashed: list[str] = []

        def crash_before_the_next_step(self_, run_dir, phase, **kwargs):
            if not crashed and kwargs.get("step_id") == "S02":
                crashed.append("S02")
                raise RuntimeError("simulated crash before the checkpoint advanced")
            return real(self_, run_dir, phase, **kwargs)

        with mock.patch.object(Orchestrator, "_checkpoint", crash_before_the_next_step):
            failed = self.run_approved(
                config, orchestrator, "crash-after-step", ("S01", "S02")
            )
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual([call["step"] for call in luna.calls], ["S01"])
        # The durable record is COMPLETED while the checkpoint still names S01.
        record = json.loads((failed.run_dir / "steps/S01/step.json").read_text())
        self.assertEqual((record["status"], record["changed_paths"]),
                         ("COMPLETED", ["src/a.py"]))
        self.assertEqual(read_checkpoint(failed.run_dir).step_id, "S01")

        second_luna = FakeLuna({(1, "S02"): actions[(1, "S02")]})
        second, planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=second_luna,
        )
        resumed = second.resume("crash-after-step")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # S01's worker call count is unchanged; the resume starts at S02.
        self.assertEqual([call["step"] for call in luna.calls], ["S01"])
        self.assertEqual([call["step"] for call in second_luna.calls], ["S02"])
        self.assertEqual(planner.prompts, [])
        self.assertEqual(
            json.loads((resumed.run_dir / "steps/S01/step.json").read_text())["tree_after"],
            record["tree_after"],
        )

    def test_auto_bounded_scope_waiting_restarts_without_replanning(self) -> None:
        config = self.make_config()
        paths = tuple(f"repair/restart-{number:02d}.txt" for number in range(1, 8))
        repair_plan = plan_text(
            step_block(1, create=paths, operation="Expand repair scope"),
            title="Restarted oversized repair scope",
        )
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=6,
        )

        def repair(root: Path) -> None:
            write(root / "src/a.py", "A = 4\n")
            for number in range(1, 8):
                write(root / f"repair/restart-{number:02d}.txt", f"repair {number}\n")

        planner = QueueClient("planner", [SINGLE_PLAN, repair_plan], self.events)
        reviewer = QueueClient("reviewer", [REVISE_IMPLEMENTATION, PASS], self.events)
        first = Orchestrator(
            config, planner_client=planner, reviewer_client=reviewer,
            agent=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): repair}),
            reviser=FakeClaude(log=self.events),
        )
        waiting = self.run_approved(config, first, "scope-restart", run_options=options)
        self.assertEqual(waiting.status, RunStatus.WAITING_SCOPE_APPROVAL)
        delta_path = waiting.run_dir / "repair/C02/scope_delta.json"
        before = delta_path.read_bytes()
        delta_sha = hashlib.sha256(before).hexdigest()
        paused = read_checkpoint(waiting.run_dir)
        self.assertEqual(paused.phase, ResumePhase.SCOPE_APPROVAL)
        write_scope_approval(
            waiting.run_dir / "repair/C02", decision="APPROVE",
            scope_delta_sha256=delta_sha, source="test",
        )

        fresh_config = load_config(self.config_path())
        fresh_planner = QueueClient("planner", [], self.events)
        fresh_luna = FakeLuna({(2, "S01"): repair})
        resumed = Orchestrator(
            fresh_config, planner_client=fresh_planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=fresh_luna, reviser=FakeClaude(log=self.events),
        ).resume("scope-restart")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(fresh_planner.prompts, [])
        self.assertEqual([(call["cycle"], call["step"]) for call in fresh_luna.calls], [(2, "S01")])
        self.assertEqual(delta_path.read_bytes(), before)

    def test_resume_uses_approved_check_argv_after_config_changes(self) -> None:
        config = self.make_config()
        actions = {
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (1, "S02"): writer("src/b.py", "B = 2\n"),
        }
        first_luna = FakeLuna(dict(actions))
        first, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[TWO_STEP_PLAN], reviews=[PASS], luna=first_luna,
        )
        real_checkpoint = Orchestrator._checkpoint
        crashed: list[str] = []

        def crash_before_s02(self_, run_dir, phase, **kwargs):
            if not crashed and kwargs.get("step_id") == "S02":
                crashed.append("S02")
                raise RuntimeError("simulated stop")
            return real_checkpoint(self_, run_dir, phase, **kwargs)

        with mock.patch.object(Orchestrator, "_checkpoint", crash_before_s02):
            failed = self.run_approved(config, first, "argv-authority", ("S01", "S02"))
        changed_config = self.root / "p29.toml"
        replacement = self.root / "replacement-check-ran.txt"
        changed_config.write_text(
            changed_config.read_text(encoding="utf-8").replace(
                str(self.counter), str(replacement)
            ),
            encoding="utf-8",
        )
        fresh = load_config(changed_config)
        second_luna = FakeLuna({(1, "S02"): actions[(1, "S02")]})
        second, planner, _reviewer, _luna, _claude = self.orchestrator(
            fresh, reviews=[PASS], luna=second_luna,
        )
        resumed = second.resume("argv-authority")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertTrue((failed.run_dir / "check_authority.json").is_file())
        self.assertFalse(replacement.exists())
        self.assertEqual(len(self.counter.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(planner.prompts, [])

    def test_changed_check_authority_refuses_resume_before_model_or_check(self) -> None:
        config = self.make_config()
        first_luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        first, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=first_luna,
        )
        real_checkpoint = Orchestrator._checkpoint
        crashed: list[str] = []

        def crash_at_checks(self_, run_dir, phase, **kwargs):
            if not crashed and phase is ResumePhase.CHECKS_C01:
                crashed.append("checks_c01")
                raise RuntimeError("simulated stop")
            return real_checkpoint(self_, run_dir, phase, **kwargs)

        with mock.patch.object(Orchestrator, "_checkpoint", crash_at_checks):
            failed = self.run_approved(config, first, "authority-tamper")
        authority = failed.run_dir / "check_authority.json"
        authority.write_bytes(authority.read_bytes() + b"\n")
        planner = QueueClient("planner", [], self.events)
        resumed = Orchestrator(
            load_config(self.config_path()), planner_client=planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=FakeLuna({}), reviser=FakeClaude(log=self.events),
        ).resume("authority-tamper")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(resumed.state["failure"]["reason"], ResumeIntegrityError.code)
        self.assertEqual(planner.prompts, [])

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
