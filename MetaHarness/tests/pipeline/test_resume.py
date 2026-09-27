"""Git is the durable authority; the checkpoint names the next operation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from unittest import mock

from metaharness.config import load_config
from metaharness.models import ExecutionRole, RunMachineState, RunPhase, RunStatus
from metaharness.orchestration.gates import GateService
from metaharness.resume import resume_info
from metaharness.state import RunCheckpointError, RunStateStore
from metaharness.planning.artifacts import read_effective_plan
from tests.pipeline.support import (
    SPEC, STEP, PipelineHarness, audit_report, crash_at_checkpoint, git, initial_plan, write,
)


def raw_normalization_plan() -> str:
    return (
        initial_plan(STEP)
        .replace("READ_SET\n- feature.txt :: current content\n", "READ_SET\nNONE\n")
        .replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- missing.txt\n")
        .replace("CREATE_SET\nNONE\n", "CREATE_SET\n- feature.txt\n")
        .replace("DELETE_SET\nNONE\n", "DELETE_SET\n- absent.txt\n")
    )


class ResumeTests(PipelineHarness):
    def test_dirty_interrupted_worker_is_reset_and_step_is_replayed(self) -> None:
        def verify_clean_then_write(request):
            self.assertEqual((request.worktree / "feature.txt").read_text(), "base\n")
            self.assertEqual((request.worktree / "other.txt").read_text(), "base\n")
            self.assertFalse((request.worktree / "attempt-new.txt").exists())
            (request.worktree / "feature.txt").write_text("good\n")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, verify_clean_then_write)
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "implement_step"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL, failed.state)
        worktree = self.worktree()
        (worktree / "feature.txt").write_text("partial feature\n")
        (worktree / "other.txt").write_text("partial other\n")
        (worktree / "attempt-new.txt").write_text("uncommitted\n")
        (worktree / ".attempt-ignore").write_text("attempt-cache.*\n")
        (worktree / "attempt-cache.tmp").write_text("ignored residue\n")

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        self.assertFalse((worktree / "attempt-new.txt").exists())
        self.assertFalse((worktree / ".attempt-ignore").exists())
        self.assertFalse((worktree / "attempt-cache.tmp").exists())
        checkpoint = json.loads((self.run_dir() / "resume_checkpoint.json").read_text())
        self.assertEqual(set(checkpoint), {
            "schema_version", "iteration", "phase", "step_index",
            "last_green_commit", "plan_sha256",
        })

    def test_step_commit_before_next_checkpoint_is_adopted_once(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "implement_step")
        self.assertEqual(self.checkpoint()["step_index"], 0)
        step_commit = self.worktree()

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        self.assertEqual(subprocess.run(
            ["git", "-C", str(step_commit), "rev-list", "--count", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(), "2")

    def test_crash_in_gate_reruns_gate_from_last_green_commit(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        real = GateService.run_gate
        calls = 0

        def crash_once(service, *args, **kwargs):
            nonlocal calls
            calls += 1
            result = real(service, *args, **kwargs)
            if calls == 1:
                raise RuntimeError("crash during deterministic gate")
            return result

        orchestrator = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with mock.patch.object(GateService, "run_gate", crash_once):
            failed = orchestrator.run_text(SPEC, run_id="run")
            self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
            self.assertEqual(self.checkpoint()["phase"], "deterministic_gate")
            resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(self.workers.roles().count("implementer"), 1)

    def test_audit_mutation_before_commit_is_discarded_and_audit_replayed(self) -> None:
        import metaharness.orchestration.audit as audit_module

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.AUDITOR,
            write("feature.txt", "good\n", audit_report()),
            write("feature.txt", "good\n", audit_report()),
        )
        real_commit = audit_module.commit_tree
        commits = 0

        def crash_before_commit(*args, **kwargs):
            nonlocal commits
            commits += 1
            if commits == 1:
                raise RuntimeError("crash before audit commit")
            return real_commit(*args, **kwargs)

        orchestrator = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with mock.patch.object(audit_module, "commit_tree", crash_before_commit):
            failed = orchestrator.run_text(SPEC, run_id="run")
            self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
            self.assertEqual(self.checkpoint()["phase"], "audit")
            self.assertEqual((self.worktree() / "feature.txt").read_text(), "good\n")
            resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("auditor"), 2)
        self.assertEqual(self.worktree().joinpath("feature.txt").read_text(), "good\n")
        subjects = git(self.worktree(), "log", "--format=%s")
        self.assertEqual(subjects.count("metaharness(audit): cycle 1"), 1)

    def test_audit_commit_before_next_checkpoint_is_adopted_once(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.AUDITOR, write("feature.txt", "good\n", audit_report()))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate", occurrence=2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "audit")

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("auditor"), 1)
        self.assertEqual(git(self.worktree(), "log", "--format=%s").count(
            "metaharness(audit): cycle 1"), 1)

    def test_candidate_push_boundary_resumes_from_git_without_agents(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "candidate_push"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "candidate_ready")
        chain = self.run_dir() / "accepted-chain.json"
        chain.write_text("{corrupt report only", encoding="utf-8")
        roles = self.workers.roles()

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), roles)

    def test_missing_accepted_chain_report_does_not_block_resume(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "candidate_push"):
            original.run_text(SPEC, run_id="run")
        (self.run_dir() / "accepted-chain.json").unlink()

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))

    def test_foreign_branch_is_fatal(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        git(self.worktree(), "checkout", "-qb", "foreign-branch")

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles().count("implementer"), 1)

    def test_resume_does_not_clean_a_hard_denied_mutation(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "implement_step"):
            original.run_text(SPEC, run_id="run")
        denied = self.worktree() / ".env"
        denied.write_text("attempt residue\n")

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "HARD_DENY_PATH_MUTATION")
        self.assertTrue(denied.exists())

    def test_missing_last_green_commit_is_fatal(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        checkpoint = self.checkpoint()
        checkpoint["last_green_commit"] = "0" * 40
        (self.run_dir() / "resume_checkpoint.json").write_text(json.dumps(checkpoint))

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")

    def test_old_checkpoint_schema_is_unsupported(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        checkpoint = self.checkpoint()
        checkpoint["schema_version"] = 5
        (self.run_dir() / "resume_checkpoint.json").write_text(json.dumps(checkpoint))

        info = resume_info(self.run_dir(), self.state())

        self.assertFalse(info.resumable)
        self.assertEqual(info.operation, "RUN_SCHEMA_UNSUPPORTED")

    def test_frozen_selection_survives_live_default_changes(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        def change_live_defaults(_run_dir):
            text = self.config_path.read_text(encoding="utf-8")
            for old, new in {
                'default_planner_profile = "planner"': 'default_planner_profile = "live_planner"',
                'default_audit_profile = "auditor"': 'default_audit_profile = "live_auditor"',
                'mechanical_profile = "worker"': 'mechanical_profile = "live_worker"',
                'reasoning_profile = "worker"': 'reasoning_profile = "live_worker"',
                'agentic_profile = "worker"': 'agentic_profile = "live_worker"',
            }.items():
                text = text.replace(old, new)
            self.config_path.write_text(text, encoding="utf-8")

        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run", on_created=change_live_defaults)

        resumed = self.orchestrator(load_config(self.config_path), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual([call.profile_id for call in self.workers.calls], ["worker", "auditor"])

    def test_effective_plan_hash_survives_resume_after_all_normalizations(self) -> None:
        def implement(request):
            (request.worktree / "feature.txt").write_text("good\n")
            (request.worktree / "missing.txt").write_text("created\n")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, implement)
        original = self.orchestrator(self.config(), planner=[raw_normalization_plan()])
        with crash_at_checkpoint(original, "candidate_push"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        effective_path = self.run_dir() / "task_plan.json"
        canonical_bytes = effective_path.read_bytes()
        expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
        self.assertEqual(self.checkpoint()["plan_sha256"], expected_hash)
        self.assertIn("CREATE_EXISTING_TO_WRITE", canonical_bytes.decode())
        self.assertIn("WRITE_MISSING_TO_CREATE", canonical_bytes.decode())
        self.assertIn("DROP_MISSING_DELETE", canonical_bytes.decode())

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(effective_path.read_bytes(), canonical_bytes)
        self.assertEqual(hashlib.sha256(effective_path.read_bytes()).hexdigest(), expected_hash)
        plan = read_effective_plan(self.run_dir(), expected_hash)
        step = plan.steps[0]
        self.assertEqual(step.read_set, ("feature.txt :: current content",))
        self.assertEqual(step.write_set, ("feature.txt",))
        self.assertEqual(step.create_set, ("missing.txt",))
        self.assertEqual(step.delete_set, ())

    def test_unreadable_checkpoint_blocks_control_writes(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        store = RunStateStore(self.run_dir() / "state.json")
        phase = RunPhase(self.checkpoint()["phase"])
        self.assertIs(store.machine_state().phase, phase)
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        (self.run_dir() / "resume_checkpoint.json").write_text("{not json", encoding="utf-8")
        for mutate in (
            store.machine_state,
            store.identity,
            lambda: store.update_metadata(marker=True),
            lambda: store.set_run_state(RunMachineState(RunPhase.PUBLISH)),
        ):
            with self.assertRaises(RunCheckpointError):
                mutate()


if __name__ == "__main__":
    unittest.main()
