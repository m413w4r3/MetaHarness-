"""P31: resume boundaries that exist before the approved worktree."""

from __future__ import annotations

import io
import json
import hashlib
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from pathlib import Path
from typing import Any

from metaharness import cli
from metaharness.config import load_config
from metaharness.llm.chat import LLMHTTPError
from metaharness.approval import PlanIdentity, write_scope_approval
from metaharness.models import RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.run_options import RunOptions
import metaharness.orchestrator as orchestrator_module
import metaharness.planning_v2 as planning_v2
from metaharness.resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeIntegrityError,
    ResumePhase,
    checkpoint_payload,
    integrity_revalidation_allowed,
    mark_checkpoint_completed,
    phase_index,
    read_checkpoint,
    read_checkpoint_record,
    resume_info,
    write_checkpoint,
)
from metaharness.state import RunStateStore

from tests.test_p29 import (
    FakeClaude, FakeLuna, P29Harness, PASS, QueueClient, REVISE_IMPLEMENTATION,
    SINGLE_PLAN, SPEC, plan_text, step_block, write, writer,
)
from tests.test_p28_full_pipeline import REPAIR_PLAN, git


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


class C01EvidenceAuthorityResumeTests(P29Harness):
    """``checks/C01`` stays the authority across a crash and a resume."""

    def red_then_repaired(self, run_id: str, reviews: list[str], plans: list[str]):
        """A C01 whose first final checks are red and whose repair is green."""

        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n"),
                         (2, "S01"): writer("src/a.py", "A = 4\n")})
        claude = FakeClaude(log=self.events)
        claude.stage_actions = {(1, "check-repair"): writer("src/a.py", "A = 3\n")}
        orchestrator, planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=plans, reviews=reviews, luna=luna, claude=claude,
        )
        return config, orchestrator, luna, claude, planner

    def test_resume_after_reviewer_c01_reloads_the_green_evidence(self) -> None:
        """Crash entering the C02 repair planner, just after reviewer #1."""

        config, orchestrator, _luna, claude, _planner = self.red_then_repaired(
            "c01-green-resume", [REVISE_IMPLEMENTATION, PASS], [SINGLE_PLAN, REPAIR_PLAN],
        )
        with mock.patch.object(Orchestrator, "_execute_v2_repair_cycle",
                               side_effect=KeyboardInterrupt):
            failed = self.run_approved(config, orchestrator, "c01-green-resume")
        run_dir = failed.run_dir
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.REPAIR_PLANNER)
        green = json.loads((run_dir / "checks/C01/evidence.json").read_text())
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual([call["stage"] for call in claude.calls],
                         ["initial-revision", "check-repair"])

        fresh, planner, _reviewer, fresh_luna, fresh_claude = self.orchestrator(
            config, plans=[REPAIR_PLAN], reviews=[PASS],
            luna=FakeLuna({(2, "S01"): writer("src/a.py", "A = 4\n")}),
        )
        resumed = fresh.resume("c01-green-resume")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # No RESUME_INTEGRITY_FAILURE, and the C01 check repair is not replayed.
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["initial-revision"])
        self.assertEqual([call["cycle"] for call in fresh_claude.calls], [2])
        self.assertEqual([call["cycle"] for call in fresh_luna.calls], [2])
        self.assertEqual(len(planner.prompts), 1)
        # The evidence the resume loaded is still the green one, and it is the
        # exact tree of the C01 candidate commit.
        after = json.loads((run_dir / "checks/C01/evidence.json").read_text())
        self.assertEqual(after, green)
        candidate = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(candidate["tree_sha"], green["staged_tree_sha"])
        self.assertFalse(
            json.loads((run_dir / "checks/C01/attempts/01/evidence.json").read_text())
            ["deterministic_passed"]
        )

    def test_a_historical_root_only_c01_evidence_still_resumes(self) -> None:
        """A run created before ``checks/C01`` became the canonical location."""

        config = self.make_config()
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS],
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
        )
        # Stop right after the final checks froze their evidence.
        with mock.patch.object(orchestrator_module, "commit_candidate_tree",
                               side_effect=KeyboardInterrupt):
            failed = self.run_approved(config, orchestrator, "root-only")
        run_dir = failed.run_dir
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.CANDIDATE_COMMIT_C01)
        # Rewrite the run exactly as a pre-canonical one: the only C01
        # evidence lives at the run root and the boundary is FINAL_CHECKS_C01.
        checks_dir = run_dir / "checks" / "C01"
        for item in checks_dir.iterdir():
            if item.is_file():
                item.replace(run_dir / item.name)
        shutil.rmtree(checks_dir)
        self.assertFalse(checks_dir.exists())
        self.assertTrue((run_dir / "evidence.json").is_file())
        write_checkpoint(run_dir, ResumeCheckpoint(
            ResumePhase.FINAL_CHECKS_C01, 1, None,
            checkpoint.expected_head_sha, checkpoint.expected_tree_sha,
            checkpoint.execution_selection_sha256, checkpoint.plan_identity,
        ))

        fresh, planner, _reviewer, fresh_luna, fresh_claude = self.orchestrator(
            config, reviews=[PASS],
        )
        resumed = fresh.resume("root-only")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((planner.prompts, fresh_luna.calls, fresh_claude.calls), ([], [], []))
        # The rerun published its evidence in the canonical directory.
        green = json.loads((run_dir / "checks/C01/evidence.json").read_text())
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual(
            json.loads((run_dir / "candidate/C01/commit.json").read_text())["tree_sha"],
            green["staged_tree_sha"],
        )

    def test_a_failed_check_repair_attempt_is_archived_before_the_retry(self) -> None:
        config = self.make_config()
        first_claude = FakeClaude(log=self.events, failures={(1, "check-repair"): "timeout"})
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS],
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}), claude=first_claude,
        )
        failed = self.run_approved(config, orchestrator, "repair-archive")
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "CLAUDE_TIMEOUT")
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.CHECK_REPAIR_C01)
        repair_dir = run_dir / "revision" / "check-repair" / "C01"
        first_prompt = (repair_dir / "agent.prompt.txt").read_text(encoding="utf-8")
        self.assertTrue((repair_dir / "tree_after_failure.txt").exists())
        self.assertFalse((repair_dir / "attempts").exists())

        fresh_claude = FakeClaude(log=self.events)
        fresh_claude.stage_actions = {(1, "check-repair"): writer("src/a.py", "A = 3\n")}
        fresh, planner, _reviewer, fresh_luna, _claude = self.orchestrator(
            config, reviews=[PASS], claude=fresh_claude,
        )
        resumed = fresh.resume("repair-archive")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # Luna and the first (successful) Claude revision are not replayed.
        self.assertEqual((planner.prompts, fresh_luna.calls), ([], []))
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["check-repair"])
        # Attempt #1 keeps its own artifacts; attempt #2 owns the current ones.
        archived = repair_dir / "attempts" / "01"
        self.assertEqual(archived.joinpath("agent.prompt.txt").read_text(encoding="utf-8"),
                         first_prompt)
        for name in ("agent.events.jsonl", "agent.stderr.log", "tree_after_failure.txt"):
            self.assertTrue((archived / name).exists(), name)
        self.assertTrue((repair_dir / "report.json").exists())
        self.assertNotEqual((repair_dir / "agent.final.md").read_text(encoding="utf-8"), "")
        self.assertFalse((repair_dir / "attempts" / "02").exists())


class IntegrityRevalidationEligibilityTests(unittest.TestCase):
    """``--revalidate-integrity`` gates on an explicit, narrow durable shape.

    It is an operator intent, never a bypass: it only stops ``resume_info``
    from rejecting ``RESUME_INTEGRITY_FAILURE`` up front, so that the
    orchestrator's complete fail-closed validation can run a second time.
    """

    CHECKPOINT = ResumeCheckpoint(
        phase=ResumePhase.FINAL_CHECKS_RETRY_C01, cycle=1, step_id=None,
        expected_head_sha="1" * 40, expected_tree_sha="2" * 40,
        execution_selection_sha256="d" * 64,
        plan_identity=PlanIdentity("a" * 64, "b" * 64, "c" * 64, "d" * 64),
    )

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "run"
        self.worktree = Path(self.temp.name) / "worktree"
        self.worktree.mkdir(parents=True)
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "plan_approval.json").write_text(
            json.dumps({"decision": "APPROVE"}), encoding="utf-8"
        )
        write_checkpoint(self.run_dir, self.CHECKPOINT)

    def state(self, reason: str = "RESUME_INTEGRITY_FAILURE") -> dict:
        return {
            "status": "failed", "planning_protocol": "v2",
            "worktree": str(self.worktree), "failure": {"reason": reason},
        }

    def test_the_flag_is_required_to_revalidate_an_integrity_failure(self) -> None:
        state = self.state()
        self.assertTrue(integrity_revalidation_allowed(self.run_dir, state))
        plain = resume_info(self.run_dir, state)
        self.assertFalse(plain.resumable)
        self.assertEqual(plain.reason, "failure requires operator intervention")
        opened = resume_info(self.run_dir, state, revalidate_integrity=True)
        self.assertTrue(opened.resumable)
        self.assertEqual(opened.phase, ResumePhase.FINAL_CHECKS_RETRY_C01.value)
        self.assertEqual(opened.expected_tree, self.CHECKPOINT.expected_tree_sha)

    def test_the_flag_never_waives_another_failure_or_a_closed_checkpoint(self) -> None:
        for label, state in (
            ("other failure", self.state("AGENT_COMMITTED")),
            ("not failed", {**self.state(), "status": "interrupted"}),
            ("no failure", {**self.state(), "failure": None}),
        ):
            with self.subTest(label):
                self.assertFalse(integrity_revalidation_allowed(self.run_dir, state))
                info = resume_info(self.run_dir, state, revalidate_integrity=True)
                self.assertFalse(info.resumable)
                self.assertEqual(
                    info.reason, "this run is not eligible for an integrity revalidation"
                )
        mark_checkpoint_completed(self.run_dir)
        self.assertFalse(integrity_revalidation_allowed(self.run_dir, self.state()))

    def test_the_flag_waives_nothing_else_resume_info_checks(self) -> None:
        """Every other cheap requirement still rejects the run."""

        shutil.rmtree(self.worktree)
        self.assertFalse(
            resume_info(self.run_dir, self.state(), revalidate_integrity=True).resumable
        )
        self.worktree.mkdir(parents=True)
        (self.run_dir / "plan_approval.json").write_text(
            json.dumps({"decision": "REJECT"}), encoding="utf-8"
        )
        info = resume_info(self.run_dir, self.state(), revalidate_integrity=True)
        self.assertFalse(info.resumable)
        self.assertEqual(info.reason, "plan approval was not APPROVE")

    def test_the_cli_flag_reaches_the_orchestrator_and_defaults_to_off(self) -> None:
        for argv, expected in (
            (["resume", "--config", "c.toml", "--run-id", "r"], False),
            (["resume", "--config", "c.toml", "--run-id", "r", "--revalidate-integrity"], True),
        ):
            with self.subTest(argv=argv[-1]):
                args = cli.build_parser().parse_args(argv)
                self.assertEqual(args.revalidate_integrity, expected)
                with mock.patch.object(cli, "resume_run") as resume_run:
                    resume_run.return_value = mock.Mock(
                        status=RunStatus.FAILED, run_dir=self.run_dir, state=self.state(),
                    )
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        cli._resume(Path(args.config), args.run_id, args.revalidate_integrity)
                self.assertEqual(
                    resume_run.call_args.kwargs, {"revalidate_integrity": expected}
                )

    def test_bounded_scope_cli_options_are_parsed(self) -> None:
        args = cli.build_parser().parse_args([
            "resume", "--config", "c.toml", "--run-id", "r",
            "--allow-bounded-test-scope-expansion",
            "--bounded-test-scope-max-paths", "9",
        ])
        self.assertTrue(args.allow_bounded_test_scope_expansion)
        self.assertEqual(args.bounded_test_scope_max_paths, 9)


class RepairPlannerRequestArtifactsTests(P29Harness):
    """The C02 request is durable before any transport is attempted."""

    class TransportFailsOnRepair(QueueClient):
        def complete(self, prompt: str):
            if self.prompts:
                self.prompts.append(prompt)
                raise LLMHTTPError("LLM endpoint returned HTTP 502 after 3 attempt(s)")
            return super().complete(prompt)

    def test_a_repair_transport_failure_leaves_every_request_artifact(self) -> None:
        config = self.make_config()
        planner = self.TransportFailsOnRepair("planner", [SINGLE_PLAN], self.events)
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        orchestrator, _p, reviewer, _l, claude = self.orchestrator(
            config, planner=planner, reviews=[REVISE_IMPLEMENTATION], luna=luna,
        )
        failed = self.run_approved(config, orchestrator, "repair-artifacts")
        self.assertEqual(failed.state["failure"]["reason"], "LLM_FAILURE")

        run_dir = failed.run_dir
        repair_dir = run_dir / "repair" / "C02"
        for name in ("planner.request.txt", "planner.request.fallback.txt",
                     "planner.evidence.md", "planner.request.meta.json"):
            self.assertTrue((repair_dir / name).exists(), name)
        meta = json.loads((repair_dir / "planner.request.meta.json").read_text())
        self.assertEqual(
            meta["inline_sha256"],
            hashlib.sha256(
                (repair_dir / "planner.request.txt").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            meta["evidence_sha256"],
            hashlib.sha256(
                (repair_dir / "planner.evidence.md").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(meta["file_fallback_attempt"], 3)

        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.REPAIR_PLANNER)
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual([call["cycle"] for call in luna.calls], [1])
        self.assertEqual([call["cycle"] for call in claude.calls], [1])

        fresh, planner2, reviewer2, luna2, claude2 = self.orchestrator(
            config, plans=[REPAIR_PLAN], reviews=[PASS],
            luna=FakeLuna({(2, "S01"): writer("src/a.py", "A = 4\n")}),
        )
        resumed = fresh.resume("repair-artifacts")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # Only the repair planner is replayed; C01 keeps its exact candidate.
        self.assertEqual(len(planner2.prompts), 1)
        self.assertEqual([call["cycle"] for call in luna2.calls], [2])
        self.assertEqual([call["cycle"] for call in claude2.calls], [2])
        self.assertEqual(len(reviewer2.prompts), 1)
        self.assertEqual(
            json.loads((run_dir / "candidate/C01/commit.json").read_text()), c01
        )


SERVICE_PLAN = plan_text(
    step_block(1, read=("src/service.py",), write_set=("src/service.py",)),
    title="P31 service plan",
)
EXPANDED_C01 = "revision/check-repair-expanded/C01"
EXPANDED_C02 = "revision/check-repair-expanded/C02"


class RetryCheckpointPhaseOrderTests(unittest.TestCase):
    """``FINAL_CHECKS_RETRY`` sits *before* its expanded repair, not after."""

    def test_the_expanded_repair_phases_follow_the_retry_checks(self) -> None:
        self.assertLess(
            phase_index(ResumePhase.FINAL_CHECKS_RETRY_C01),
            phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01),
        )
        self.assertLess(
            phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C01),
            phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C01),
        )
        self.assertLess(
            phase_index(ResumePhase.FINAL_CHECKS_RETRY_C02),
            phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02),
        )
        self.assertLess(
            phase_index(ResumePhase.CHECK_REPAIR_EXPANDED_C02),
            phase_index(ResumePhase.FINAL_CHECKS_RETRY_EXPANDED_C02),
        )


class ExpandedCheckRepairResumeHarness(P29Harness):
    """A repository whose failing check names a tracked test file."""

    def stage_service_repo(self) -> None:
        write(self.repo / "src/service.py", "SERVICE = 1\n")
        write(self.repo / "tests/test_service.py",
              "def test_fake_uow():  # stale\n    pass\n")
        # A second tracked test the gate never names: it can only ever enter a
        # mutable scope through a durable artifact, never through the detector.
        write(self.repo / "tests/test_other.py", "def test_other():\n    pass\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add the service and its stale fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        # Red on a production path while the service is broken; once it is
        # repaired the same gate names the stale *tracked test* instead, which
        # is the only failure family a bounded expansion may ever answer to.
        self.check.write_text(
            "import pathlib, sys\n"
            "open(sys.argv[1], 'a').write('ran\\n')\n"
            "source = pathlib.Path('src/service.py').read_text()\n"
            "fixture = pathlib.Path('tests/test_service.py').read_text()\n"
            "if 'BUG' in source:\n"
            "    print('FAILED src/other_production.py')\n"
            "    sys.exit(1)\n"
            "print('FAILED tests/test_service.py::test_fake_uow')\n"
            "print('AttributeError: FakeUnitOfWork has no attribute commit')\n"
            "sys.exit(1 if 'stale' in fixture else 0)\n",
            encoding="utf-8",
        )

    def bounded_options(self, config: Any, *, max_added_paths: int = 4) -> RunOptions:
        return RunOptions.from_config(
            config, repair_scope_policy="auto-bounded",
            repair_scope_max_added_paths=max_added_paths,
        )

    def fix_the_fixture(self) -> FakeClaude:
        """A fresh Claude double whose only action repairs the fixture."""

        return FakeClaude(log=self.events, stage_actions={
            (1, "check-repair"): writer(
                "tests/test_service.py", "def test_fake_uow():\n    pass\n"
            ),
            (2, "check-repair"): writer(
                "tests/test_service.py", "def test_fake_uow():\n    pass\n"
            ),
        })

    def crash_before_the_expansion(self, cycle: str):
        """Stop exactly between the red retry evidence and the expansion."""

        real = orchestrator_module._archive_attempt
        seen: list[int] = []

        def archive(directory: Any, **kwargs: Any) -> Any:
            path = Path(directory)
            if path.name == cycle and path.parent.name == "checks":
                seen.append(1)
                if len(seen) == 2:
                    raise RuntimeError("simulated crash before the expansion")
            return real(directory, **kwargs)

        return mock.patch.object(
            orchestrator_module, "_archive_attempt", side_effect=archive
        )

    def crash_inside_the_nth_checks(self, ordinal: int):
        real = Orchestrator._final_evidence
        calls: list[int] = []

        def evidence(self_: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) == ordinal:
                raise RuntimeError("simulated crash inside the checks")
            return real(self_, *args, **kwargs)

        return mock.patch.object(Orchestrator, "_final_evidence", evidence)

    def crash_before_the_phase(self, phase: ResumePhase):
        real = Orchestrator._checkpoint
        crashed: list[int] = []

        def checkpoint(self_: Any, run_dir: Any, written: Any, **kwargs: Any) -> Any:
            if written is phase and not crashed:
                crashed.append(1)
                raise RuntimeError("simulated crash before the boundary moved")
            return real(self_, run_dir, written, **kwargs)

        return mock.patch.object(Orchestrator, "_checkpoint", checkpoint)

    def scope_of(self, run_dir: Path, relative: str) -> dict:
        return json.loads((run_dir / relative / "scope.json").read_text())


class C01RetryCheckpointExpansionTests(ExpandedCheckRepairResumeHarness):
    """``FINAL_CHECKS_RETRY_C01`` is resumable, and is not a dead end.

    The normal check repair already succeeded at this boundary, so Luna, the
    initial Claude revision and that repair are durable and are never
    replayed.  Three durable states are legitimate and each has exactly one
    correct continuation: no retry evidence yet (run those checks once), red
    retry evidence (decide the single bounded expansion), and green retry
    evidence (reconcile straight into the candidate commit).
    """

    def red_retry_c01(self, run_id: str):
        """A crash between the red retry evidence and the expansion."""

        self.stage_service_repo()
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/service.py", "SERVICE = BUG\n")})
        claude = FakeClaude(log=self.events, stage_actions={
            (1, "check-repair"): writer("src/service.py", "SERVICE = 3\n"),
        })
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SERVICE_PLAN], reviews=[PASS], luna=luna, claude=claude,
        )
        with self.crash_before_the_expansion("C01"):
            failed = self.run_approved(
                config, orchestrator, run_id,
                run_options=self.bounded_options(config),
            )
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(failed.run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C01)
        return config, failed, checkpoint, claude

    def test_a_red_retry_checkpoint_resumes_into_the_expanded_repair(self) -> None:
        """The exact AW-003 state: red retry evidence naming a tracked test."""

        config, failed, checkpoint, first_claude = self.red_retry_c01("aw003-red")
        run_dir = failed.run_dir
        # The durable facts the resume must work from, and nothing else.
        self.assertTrue((run_dir / "revision/check-repair/C01/report.json").is_file())
        normal_scope = self.scope_of(run_dir, "revision/check-repair/C01")
        self.assertEqual(normal_scope["base_mutable_scope"], ["src/service.py"])
        self.assertEqual(normal_scope["added_paths"], [])
        evidence = json.loads((run_dir / "checks/C01/evidence.json").read_text())
        self.assertFalse(evidence["deterministic_passed"])
        self.assertEqual(evidence["failures"], ["CHECK_FAILED:gate"])
        self.assertEqual(evidence["staged_tree_sha"], checkpoint.expected_tree_sha)
        self.assertFalse((run_dir / EXPANDED_C01).exists())
        self.assertEqual([call["stage"] for call in first_claude.calls],
                         ["initial-revision", "check-repair"])

        before = self.checks_ran()
        fresh_luna = FakeLuna({})
        fresh_claude = self.fix_the_fixture()
        fresh, planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=fresh_luna, claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-red")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # Nothing before the boundary is replayed: no Luna, no normal Claude
        # revision, no normal check repair.  Exactly one expanded pass.
        self.assertEqual(fresh_luna.calls, [])
        self.assertEqual(planner.prompts, [])
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["check-repair"])
        self.assertEqual(
            fresh_claude.calls[0]["revision_dir"].relative_to(run_dir).as_posix(),
            EXPANDED_C01,
        )
        # The already durable red bundle is the authority, so the only checks
        # this resume pays for are the expanded retry checks.
        self.assertEqual(self.checks_ran() - before, 1)
        # The red retry bundle is archived as attempt #2; attempt #1 stays the
        # first, pre-normal-repair red evidence.
        self.assertTrue((run_dir / "checks/C01/attempts/02/evidence.json").is_file())
        self.assertEqual(
            json.loads(
                (run_dir / "checks/C01/attempts/02/evidence.json").read_text()
            )["staged_tree_sha"],
            checkpoint.expected_tree_sha,
        )
        self.assertFalse(
            json.loads(
                (run_dir / "checks/C01/attempts/01/evidence.json").read_text()
            )["deterministic_passed"]
        )
        expanded = self.scope_of(run_dir, EXPANDED_C01)
        self.assertEqual(expanded["added_paths"], ["tests/test_service.py"])
        self.assertEqual(expanded["base_mutable_scope"], ["src/service.py"])
        self.assertEqual(expanded["effective_mutable_scope"],
                         ["src/service.py", "tests/test_service.py"])
        self.assertTrue((run_dir / EXPANDED_C01 / "report.json").is_file())
        # Green expanded retry, immutable candidate, then reviewer #1.
        self.assertTrue(json.loads(
            (run_dir / "checks/C01/evidence.json").read_text()
        )["deterministic_passed"])
        candidate = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(
            candidate["tree_sha"],
            json.loads((run_dir / "checks/C01/evidence.json").read_text())
            ["staged_tree_sha"],
        )
        self.assertEqual(len(reviewer.prompts), 1)

    def test_b_a_red_retry_checkpoint_is_not_an_implicit_terminal(self) -> None:
        """The phase hole itself: the boundary continues, without duplicates.

        On the broken build this same durable state fell through to the
        candidate gate, so the retry checks were paid for twice before any
        expansion could be decided.
        """

        config, failed, _checkpoint, _claude = self.red_retry_c01("aw003-hole")
        before = self.checks_ran()
        fresh, _planner, _reviewer, fresh_luna, _c = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({}), claude=self.fix_the_fixture(),
        )
        resumed = fresh.resume("aw003-hole")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertNotEqual(resumed.state.get("failure"), "DETERMINISTIC_GATE_FAILED")
        # One check execution only: the expanded retry.  The durable red retry
        # bundle is never re-earned.
        self.assertEqual(self.checks_ran() - before, 1)
        self.assertTrue((failed.run_dir / EXPANDED_C01 / "scope.json").is_file())

    def test_c_a_durable_expanded_scope_is_reused_not_recomputed(self) -> None:
        """A crash between ``scope.json`` and the expanded Claude pass."""

        config, failed, _checkpoint, _claude = self.red_retry_c01("aw003-durable")
        run_dir = failed.run_dir
        # Publish an expanded scope the detector would never produce, then
        # leave the checkpoint where it is.  A recomputation would pick
        # ``tests/test_service.py``; only the durable artifact names the other.
        expanded_dir = run_dir / EXPANDED_C01
        expanded_dir.mkdir(parents=True, exist_ok=True)
        (expanded_dir / "scope.json").write_text(json.dumps({
            "schema_version": 2,
            "base_mutable_scope": ["src/service.py"],
            "added_paths": ["tests/test_other.py"],
            "effective_mutable_scope": ["src/service.py", "tests/test_other.py"],
            "policy": "auto-bounded",
            "bound": 4,
            "source": "auto-bounded failing-test evidence",
        }), encoding="utf-8")

        # The durable scope cannot repair the stale fixture, so this pass is
        # deliberately a no-op edit-wise.
        fresh_claude = FakeClaude(log=self.events)
        fresh, _planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({}), claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-durable")

        # The model is recalled with exactly the durable scope, never a
        # recomputed one, and the already archived attempt is not archived a
        # second time.  That scope cannot repair the gate, so it stays red --
        # and there is still no third repair.
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["check-repair"])
        prompt = fresh_claude.calls[0]["prompt"]
        self.assertIn("tests/test_other.py", prompt)
        self.assertNotIn("tests/test_service.py\"", prompt.split(
            "EFFECTIVE REPAIR MUTABLE SCOPE:", 1
        )[1].split("Previous Claude report", 1)[0])
        self.assertEqual(
            self.scope_of(run_dir, EXPANDED_C01)["added_paths"],
            ["tests/test_other.py"],
        )
        self.assertFalse((run_dir / "checks/C01/attempts/02").exists())
        self.assertEqual(resumed.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(reviewer.prompts, [])

    def test_d_a_malformed_durable_expanded_scope_fails_closed(self) -> None:
        config, failed, _checkpoint, _claude = self.red_retry_c01("aw003-malformed")
        expanded_dir = failed.run_dir / EXPANDED_C01
        expanded_dir.mkdir(parents=True, exist_ok=True)
        (expanded_dir / "scope.json").write_text(json.dumps({
            "schema_version": 2,
            "base_mutable_scope": ["src/service.py"],
            "added_paths": ["src/other_production.py"],
            "effective_mutable_scope": ["src/other_production.py", "src/service.py"],
            "policy": "auto-bounded",
            "bound": 4,
            "source": "auto-bounded failing-test evidence",
        }), encoding="utf-8")
        fresh_claude = self.fix_the_fixture()
        fresh, _planner, _reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({}), claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-malformed")
        self.assertEqual(resumed.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(fresh_claude.calls, [])

    def test_e_a_crash_before_the_retry_evidence_runs_those_checks_once(self) -> None:
        """No current bundle is legitimate here, and never an integrity failure."""

        self.stage_service_repo()
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/service.py", "SERVICE = BUG\n")})
        claude = FakeClaude(log=self.events, stage_actions={
            (1, "check-repair"): writer("src/service.py", "SERVICE = 3\n"),
        })
        orchestrator, _planner, _reviewer, _l, _c = self.orchestrator(
            config, plans=[SERVICE_PLAN], reviews=[PASS], luna=luna, claude=claude,
        )
        # The C01 final checks, then the retry checks that never finish.
        with self.crash_inside_the_nth_checks(2):
            failed = self.run_approved(
                config, orchestrator, "aw003-missing",
                run_options=self.bounded_options(config),
            )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C01)
        self.assertTrue((run_dir / "revision/check-repair/C01/report.json").is_file())
        self.assertFalse((run_dir / "checks/C01/evidence.json").exists())
        self.assertFalse(json.loads(
            (run_dir / "checks/C01/attempts/01/evidence.json").read_text()
        )["deterministic_passed"])

        before = self.checks_ran()
        fresh_luna = FakeLuna({})
        fresh_claude = self.fix_the_fixture()
        fresh, planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=fresh_luna, claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-missing")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # The normal Claude repair is not replayed; the retry checks run once
        # and stay red on the tracked test, which earns the single expansion.
        self.assertEqual((fresh_luna.calls, planner.prompts), ([], []))
        self.assertEqual([call["stage"] for call in fresh_claude.calls], ["check-repair"])
        self.assertEqual(
            fresh_claude.calls[0]["revision_dir"].relative_to(run_dir).as_posix(),
            EXPANDED_C01,
        )
        # Exactly the retry checks plus the expanded retry checks.
        self.assertEqual(self.checks_ran() - before, 2)
        self.assertEqual(
            self.scope_of(run_dir, EXPANDED_C01)["added_paths"],
            ["tests/test_service.py"],
        )
        self.assertTrue((run_dir / "candidate/C01/commit.json").is_file())
        self.assertEqual(len(reviewer.prompts), 1)

    def test_f_a_green_retry_evidence_is_reconciled_not_repeated(self) -> None:
        """The crash landed after a green retry and before the boundary moved."""

        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(log=self.events, stage_actions={
            (1, "check-repair"): writer("src/a.py", "A = 3\n"),
        })
        orchestrator, _planner, _reviewer, _l, _c = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna, claude=claude,
        )
        with self.crash_before_the_phase(ResumePhase.CANDIDATE_COMMIT_C01):
            failed = self.run_approved(
                config, orchestrator, "aw003-green",
                run_options=self.bounded_options(config),
            )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C01)
        green = json.loads((run_dir / "checks/C01/evidence.json").read_text())
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual(green["staged_tree_sha"], checkpoint.expected_tree_sha)
        self.assertFalse((run_dir / "candidate/C01/commit.json").exists())

        before = self.checks_ran()
        fresh_luna, fresh_claude = FakeLuna({}), FakeClaude(log=self.events)
        fresh, planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=fresh_luna, claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-green")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # A durable green retry is a reconciliation, not a bypass: no check,
        # no model call and no expansion -- straight to the candidate commit.
        self.assertEqual(self.checks_ran(), before)
        self.assertEqual((fresh_luna.calls, fresh_claude.calls, planner.prompts),
                         ([], [], []))
        self.assertFalse((run_dir / EXPANDED_C01).exists())
        self.assertEqual(
            json.loads((run_dir / "candidate/C01/commit.json").read_text())["tree_sha"],
            green["staged_tree_sha"],
        )
        self.assertEqual(json.loads(
            (run_dir / "checks/C01/evidence.json").read_text()
        ), green)
        self.assertEqual(len(reviewer.prompts), 1)

    def test_g_a_red_retry_without_any_candidate_stays_red(self) -> None:
        """Nothing to expand to: the deterministic gate is final."""

        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(log=self.events, stage_actions={
            (1, "check-repair"): writer("src/a.py", "A = STILL BUG\n"),
        })
        orchestrator, _planner, _reviewer, _l, _c = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna, claude=claude,
        )
        failed = self.run_approved(
            config, orchestrator, "aw003-nocandidate",
            run_options=self.bounded_options(config),
        )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(read_checkpoint(run_dir).phase,
                         ResumePhase.FINAL_CHECKS_RETRY_C01)

        before = self.checks_ran()
        fresh_claude = FakeClaude(log=self.events)
        fresh, planner, reviewer, fresh_luna, _c = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({}), claude=fresh_claude,
        )
        resumed = fresh.resume("aw003-nocandidate")

        self.assertEqual(resumed.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(self.checks_ran(), before)
        self.assertEqual((fresh_claude.calls, fresh_luna.calls, planner.prompts,
                          reviewer.prompts), ([], [], [], []))
        self.assertFalse((run_dir / EXPANDED_C01).exists())
        self.assertFalse((run_dir / "candidate/C01/commit.json").exists())


class C02RetryCheckpointExpansionTests(ExpandedCheckRepairResumeHarness):
    """Exact C02 parity for the same three durable retry states."""

    C02_PLAN = plan_text(
        step_block(1, operation="Repair"), title="P31 C02 repair",
    )

    def stage_c02_repo(self) -> None:
        write(self.repo / "tests/test_service.py",
              "def test_fake_uow():  # stale\n    pass\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add the stale fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        # C01 is green while ``src/a.py`` is neither BUG nor FIXED; C02 turns
        # it red on a production path, and only the C02 normal repair makes
        # the same gate name the stale tracked test.
        self.check.write_text(
            "import pathlib, sys\n"
            "open(sys.argv[1], 'a').write('ran\\n')\n"
            "source = pathlib.Path('src/a.py').read_text()\n"
            "fixture = pathlib.Path('tests/test_service.py').read_text()\n"
            "if 'BUG' in source:\n"
            "    print('FAILED src/other_production.py')\n"
            "    sys.exit(1)\n"
            "if 'FIXED' in source:\n"
            "    print('FAILED tests/test_service.py::test_fake_uow')\n"
            "    print('AttributeError: FakeUnitOfWork has no attribute commit')\n"
            "    sys.exit(1 if 'stale' in fixture else 0)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )

    def start_c02(self, run_id: str, *, c02_repair: str):
        """Run up to the C02 retry checks with *c02_repair* in ``src/a.py``."""

        self.stage_c02_repo()
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"),
                         (2, "S01"): writer("src/a.py", "A = BUG\n")})
        claude = FakeClaude(log=self.events, stage_actions={
            (2, "check-repair"): writer("src/a.py", c02_repair),
        })
        orchestrator, _planner, _reviewer, _l, _c = self.orchestrator(
            config, plans=[SINGLE_PLAN, self.C02_PLAN],
            reviews=[REVISE_IMPLEMENTATION, PASS], luna=luna, claude=claude,
        )
        return config, orchestrator, luna, claude

    def assert_c02_identity_preserved(self, run_dir: Path, before) -> None:
        """Every C02 boundary the resume wrote keeps the cycle's identity."""

        after = read_checkpoint_record(run_dir)[0]
        self.assertGreaterEqual(
            phase_index(after.phase), phase_index(ResumePhase.CANDIDATE_COMMIT_C02)
        )
        self.assertEqual(after.repair_bundle_sha256, before.repair_bundle_sha256)
        self.assertEqual(after.scope_delta_sha256, before.scope_delta_sha256)
        self.assertEqual(
            before.expected_head_sha,
            json.loads((run_dir / "candidate/C01/commit.json").read_text())["commit_sha"],
        )

    def test_a_red_c02_retry_checkpoint_resumes_into_the_expanded_repair(self) -> None:
        config, orchestrator, _luna, claude = self.start_c02(
            "c02-red", c02_repair="A = FIXED\n",
        )
        with self.crash_before_the_expansion("C02"):
            failed = self.run_approved(
                config, orchestrator, "c02-red",
                run_options=self.bounded_options(config),
            )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C02)
        self.assertIsNotNone(checkpoint.repair_bundle_sha256)
        self.assertIsNotNone(checkpoint.scope_delta_sha256)
        evidence = json.loads((run_dir / "checks/C02/evidence.json").read_text())
        self.assertFalse(evidence["deterministic_passed"])
        self.assertEqual(evidence["staged_tree_sha"], checkpoint.expected_tree_sha)
        self.assertFalse((run_dir / EXPANDED_C02).exists())
        self.assertEqual([(call["cycle"], call["stage"]) for call in claude.calls],
                         [(1, "initial-revision"), (2, "initial-revision"),
                          (2, "check-repair")])

        before = self.checks_ran()
        fresh_luna = FakeLuna({})
        fresh_claude = self.fix_the_fixture()
        fresh, planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=fresh_luna, claude=fresh_claude,
        )
        resumed = fresh.resume("c02-red")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((fresh_luna.calls, planner.prompts), ([], []))
        self.assertEqual([(call["cycle"], call["stage"]) for call in fresh_claude.calls],
                         [(2, "check-repair")])
        self.assertEqual(
            fresh_claude.calls[0]["revision_dir"].relative_to(run_dir).as_posix(),
            EXPANDED_C02,
        )
        self.assertEqual(self.checks_ran() - before, 1)
        self.assertTrue((run_dir / "checks/C02/attempts/02/evidence.json").is_file())
        self.assertEqual(
            self.scope_of(run_dir, EXPANDED_C02)["added_paths"],
            ["tests/test_service.py"],
        )
        self.assertTrue((run_dir / "candidate/C02/commit.json").is_file())
        # Only reviewer #2 ran on this resume; reviewer #1 stays accepted.
        self.assertEqual(len(reviewer.prompts), 1)
        self.assert_c02_identity_preserved(run_dir, checkpoint)

    def test_b_a_crash_before_the_c02_retry_evidence_runs_those_checks_once(self) -> None:
        config, orchestrator, _luna, _claude = self.start_c02(
            "c02-missing", c02_repair="A = FIXED\n",
        )
        # C01 final checks, C02 final checks, then the C02 retry checks.
        with self.crash_inside_the_nth_checks(3):
            failed = self.run_approved(
                config, orchestrator, "c02-missing",
                run_options=self.bounded_options(config),
            )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C02)
        self.assertFalse((run_dir / "checks/C02/evidence.json").exists())
        self.assertTrue((run_dir / "revision/check-repair/C02/report.json").is_file())

        before = self.checks_ran()
        fresh_claude = self.fix_the_fixture()
        fresh, planner, reviewer, fresh_luna, _c = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({}), claude=fresh_claude,
        )
        resumed = fresh.resume("c02-missing")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((fresh_luna.calls, planner.prompts), ([], []))
        self.assertEqual([(call["cycle"], call["stage"]) for call in fresh_claude.calls],
                         [(2, "check-repair")])
        self.assertEqual(self.checks_ran() - before, 2)
        self.assertEqual(
            self.scope_of(run_dir, EXPANDED_C02)["added_paths"],
            ["tests/test_service.py"],
        )
        self.assertEqual(len(reviewer.prompts), 1)
        self.assert_c02_identity_preserved(run_dir, checkpoint)

    def test_c_a_green_c02_retry_evidence_is_reconciled_not_repeated(self) -> None:
        config, orchestrator, _luna, _claude = self.start_c02(
            "c02-green", c02_repair="A = 4\n",
        )
        with self.crash_before_the_phase(ResumePhase.CANDIDATE_COMMIT_C02):
            failed = self.run_approved(
                config, orchestrator, "c02-green",
                run_options=self.bounded_options(config),
            )
        run_dir = failed.run_dir
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.FINAL_CHECKS_RETRY_C02)
        green = json.loads((run_dir / "checks/C02/evidence.json").read_text())
        self.assertTrue(green["deterministic_passed"])
        self.assertEqual(green["staged_tree_sha"], checkpoint.expected_tree_sha)
        self.assertFalse((run_dir / "candidate/C02/commit.json").exists())

        before = self.checks_ran()
        fresh_claude, fresh_luna = FakeClaude(log=self.events), FakeLuna({})
        fresh, planner, reviewer, _l, _c = self.orchestrator(
            config, reviews=[PASS], luna=fresh_luna, claude=fresh_claude,
        )
        resumed = fresh.resume("c02-green")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(self.checks_ran(), before)
        self.assertEqual((fresh_claude.calls, fresh_luna.calls, planner.prompts),
                         ([], [], []))
        self.assertFalse((run_dir / EXPANDED_C02).exists())
        self.assertEqual(
            json.loads((run_dir / "candidate/C02/commit.json").read_text())["tree_sha"],
            green["staged_tree_sha"],
        )
        self.assertEqual(len(reviewer.prompts), 1)
        self.assert_c02_identity_preserved(run_dir, checkpoint)


class AggressiveRepairRecoveryTests(P29Harness):
    """AW-003: a C02 answer rejected only locally is revalidated on resume.

    The run's initial SINGLE threshold is 4 and its STAGED per-step maximum is
    6.  The repair planner answered with a SINGLE ``S01`` touching exactly 6
    mutable paths: refused by the initial planner's limit, allowed for one
    bounded repair worker.  The resume must revalidate that already paid
    ``planner.raw.md`` instead of calling the model again.
    """

    REPAIR_PATHS = tuple(f"repair/aw003-{number:02d}.txt" for number in range(1, 6))
    REPAIR_PLAN_SIX_PATHS = plan_text(
        step_block(1, create=REPAIR_PATHS, operation="Repair"),
        title="AW-003 bounded six-path repair",
    )

    class RefusingPlanner:
        """A planner transport that fails the test if it is ever called."""

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt: str):
            self.prompts.append(prompt)
            raise AssertionError("the repair planner LLM MUST NOT BE CALLED")

    def aggressive_config(self):
        text = self.config_text(require_approval=True).replace(
            'protocol = "v2"',
            'protocol = "v2"\ndecomposition = "aggressive"\n'
            "single_step_max_mutable_paths = 4\nstaged_step_max_mutable_paths = 6",
        )
        path = self.root / "aw003.toml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    @staticmethod
    def repair_writer(root: Path) -> None:
        write(root / "src/a.py", "A = 4\n")
        for path in AggressiveRepairRecoveryTests.REPAIR_PATHS:
            write(root / path, "repair\n")

    def test_a_locally_rejected_repair_plan_resumes_without_replanning(self) -> None:
        config = self.aggressive_config()
        self.assertEqual(config.planning.single_step_max_mutable_paths, 4)
        self.assertEqual(config.planning.staged_step_max_mutable_paths, 6)
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=6,
        )
        planner = QueueClient(
            "planner", [SINGLE_PLAN, self.REPAIR_PLAN_SIX_PATHS], self.events
        )
        first, _p, _r, luna, claude = self.orchestrator(
            config, planner=planner, reviews=[REVISE_IMPLEMENTATION],
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
        )
        # Exactly the durable state AW-003 reached: the repair planner answer
        # is durable and was rejected by the *initial* SINGLE limit.
        with mock.patch.object(
            planning_v2, "validate_repair_decomposition_policy",
            planning_v2.validate_decomposition_policy,
        ):
            failed = self.run_approved(
                config, first, "aw003-recovery", run_options=options,
            )

        self.assertEqual(failed.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        run_dir = failed.run_dir
        repair_dir = run_dir / "repair" / "C02"
        raw_before = (repair_dir / "planner.raw.md").read_bytes()
        self.assertTrue((repair_dir / "planner.evidence.md").is_file())
        self.assertFalse((repair_dir / "implementation_bundle.json").exists())
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.REPAIR_PLANNER)
        self.assertEqual(len(planner.prompts), 2)
        self.assertEqual([call["cycle"] for call in luna.calls], [1])
        self.assertEqual([call["cycle"] for call in claude.calls], [1])

        refusing = self.RefusingPlanner()
        fresh_luna = FakeLuna({(2, "S01"): self.repair_writer})
        fresh, _p2, reviewer2, _l2, claude2 = self.orchestrator(
            load_config(self.root / "aw003.toml"), planner=refusing,
            reviews=[PASS], luna=fresh_luna,
        )
        resumed = fresh.resume("aw003-recovery")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(refusing.prompts, [])
        # The recovered plan is the exact answer that was already paid for.
        self.assertEqual((repair_dir / "planner.raw.md").read_bytes(), raw_before)
        self.assertEqual(
            (repair_dir / "attempts" / "01" / "planner.raw.md").read_bytes(), raw_before
        )
        bundle = json.loads((repair_dir / "implementation_bundle.json").read_text())
        self.assertEqual(bundle["execution_mode"], "SINGLE")
        self.assertTrue((repair_dir / "steps" / "S01" / "contract.md").is_file())
        delta = json.loads((repair_dir / "scope_delta.json").read_text())
        self.assertEqual(len(delta["added_paths"]), len(self.REPAIR_PATHS))
        # C02 really ran: Luna repaired, Claude revised, reviewer #2 passed.
        self.assertEqual([call["cycle"] for call in fresh_luna.calls], [2])
        self.assertEqual([call["cycle"] for call in claude2.calls], [2])
        self.assertEqual(len(reviewer2.prompts), 1)

    def test_a_missing_durable_answer_still_calls_the_repair_planner(self) -> None:
        """No local recovery is invented when nothing durable exists."""

        config = self.aggressive_config()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=6,
        )
        planner = QueueClient(
            "planner", [SINGLE_PLAN, self.REPAIR_PLAN_SIX_PATHS], self.events
        )
        orchestrator, _p, reviewer, _l, _c = self.orchestrator(
            config, planner=planner, reviews=[REVISE_IMPLEMENTATION, PASS],
            luna=FakeLuna({
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (2, "S01"): self.repair_writer,
            }),
        )
        result = self.run_approved(
            config, orchestrator, "aw003-direct", run_options=options,
        )

        # The same 6-path SINGLE repair is now accepted on the first pass.
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(len(planner.prompts), 2)
        self.assertIn(
            "Every repair implementation step, including a SINGLE S01",
            planner.prompts[1],
        )
        self.assertIn("at most 6 distinct mutable\npaths", planner.prompts[1])
        self.assertEqual(len(reviewer.prompts), 2)


if __name__ == "__main__":
    unittest.main()
