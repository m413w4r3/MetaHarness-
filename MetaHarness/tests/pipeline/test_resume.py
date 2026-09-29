"""Git is the durable authority; the checkpoint names the next operation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from metaharness.config import load_config
from metaharness.approval import ApprovalDecision, compute_plan_identity_from_run, write_plan_approval
from metaharness.gitops import current_head, index_tree_sha, resolve_tree
from metaharness.execution_selection import ensure_execution_selection, read_execution_selection
from metaharness.models import ExecutionRole, RunMachineState, RunPhase, RunStatus
from metaharness.orchestration.gates import GateService
from metaharness.resume import resume_info
from metaharness.state import RunCheckpointError, RunStateStore
from metaharness.orchestration.run_resume import prepare_resume
from metaharness.planning.artifacts import (
    iteration_plan_dir, persist_iteration_plan, read_effective_plan,
    read_iteration_plan, validate_implementation_bundle,
)
from metaharness.resume import ResumeCheckpoint, ResumeIntegrityError, ResumePhase, read_checkpoint, write_checkpoint
from tests.pipeline.support import (
    SPEC, STEP, PipelineHarness, audit_report, continuation_answer, crash_at_checkpoint, git, initial_plan, write,
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
    def test_later_milestone_resume_validates_initial_human_approval(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "second\n"),
        )
        second = initial_plan(("S01", "other.txt", "Second")).replace(
            "MILESTONE_ID: M01", "MILESTONE_ID: M02",
        )
        config = self.config()
        original = self.orchestrator(config, planner=[initial_plan(STEP)], continuation=[
            continuation_answer("NEXT", milestone="M02", plan_text=second),
        ])
        with crash_at_checkpoint(original, "deterministic_gate", occurrence=3):
            original.run_text(SPEC, run_id="run")
        self.assertEqual(self.checkpoint()["iteration"], 2)
        write_plan_approval(
            self.run_dir(), decision=ApprovalDecision.APPROVE,
            identity=compute_plan_identity_from_run(self.run_dir(), iteration=1), source="test",
        )
        config = replace(config, approval=replace(config.approval, require_plan_approval=True))
        approval_bytes = (self.run_dir() / "plan_approval.json").read_bytes()
        tampered = self.state()
        tampered["plan_identity"]["execution_sha256"] = "0" * 64
        with self.assertRaisesRegex(ResumeIntegrityError, "durable plan identity"):
            prepare_resume(
                config=config, run_dir=self.run_dir(), run_id="run", state=tampered,
                checkpoint=read_checkpoint(self.run_dir()), restore_worktree=False,
            )
        resumed = self.orchestrator(config, planner=["unused"], continuation=[
            continuation_answer("COMPLETE"),
        ]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual((self.run_dir() / "plan_approval.json").read_bytes(), approval_bytes)

    def test_resume_preserves_gate_snapshot_when_live_check_is_added(self) -> None:
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "implement_step"):
            original.run_text(SPEC, run_id="run")
        authority = (self.run_dir() / "check_authority.json").read_bytes()
        config = self.config(
            per_step_gate="test-collection",
            extra_checks='''[[check_catalog]]
id = "test-collection"
argv = ["must-not-execute-new-check"]
''',
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        resumed = self.orchestrator(config, planner=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual((self.run_dir() / "check_authority.json").read_bytes(), authority)

    def test_legacy_resume_excludes_check_added_after_approval(self) -> None:
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "implement_step"):
            original.run_text(SPEC, run_id="run")
        # Reproduce a run created before the per-step policy was captured.
        store = RunStateStore(self.run_dir() / "state.json")
        state = store.load()
        state.pop("per_step_check_ids")
        store._write(state)
        config = self.config(
            per_step_gate="test-collection",
            extra_checks='''[[check_catalog]]
id = "test-collection"
argv = ["must-not-execute-new-check"]
''',
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        resumed = self.orchestrator(config, planner=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertTrue(any("test-collection" in item for item in self.state()["check_warnings"]))

    def test_own_remote_tracking_ref_may_lag_behind_local_steps(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "second\n"),
        )
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP, ("S02", "other.txt", "Second"))],
        )
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        first_step = git(self.worktree(), "rev-parse", "HEAD^")
        branch = self.state()["branch"]
        git(self.repo, "update-ref", f"refs/remotes/origin/{branch}", first_step)
        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))

    def test_unowned_ancestor_remote_tracking_ref_is_refused(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        branch = self.state()["branch"]
        git(self.repo, "update-ref", f"refs/remotes/origin/{branch}", self.base_sha)
        roles = self.workers.roles()
        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")
        self.assertEqual(resumed.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles)

    def test_pause_after_step_is_durable_and_resumes_at_the_next_step(self) -> None:
        def write_and_request_pause(request):
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            RunStateStore(self.run_dir() / "state.json").update_metadata(
                pause_requested=True,
            )
            return "done\n"

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write_and_request_pause,
            write("other.txt", "second\n"),
        )
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP, ("S02", "other.txt", "Write other"))],
        )

        paused = original.run_text(SPEC, run_id="run")

        self.assertEqual(paused.status, RunStatus.WAITING_EXTERNAL, paused.state)
        self.assertEqual(paused.state["failure"]["reason"], "PAUSED")
        self.assertFalse(paused.state["pause_requested"])
        self.assertEqual(self.checkpoint()["phase"], "implement_step")
        self.assertEqual(self.checkpoint()["step_index"], 1)
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(self.workers.roles().count("implementer"), 1)

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 2)

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

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        self.assertFalse((worktree / "attempt-new.txt").exists())
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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertGreaterEqual(calls, 2)
        self.assertEqual(self.workers.roles().count("implementer"), 1)

    def test_resume_preserves_ignored_workspace_setup_and_cleans_attempt_residue(self) -> None:
        (self.repo / ".gitignore").write_text(".workspace-cache/\n", encoding="utf-8")
        git(self.repo, "add", ".gitignore")
        git(self.repo, "commit", "-qm", "ignore prepared workspace cache")
        git(self.repo, "push", "-q", "origin", "main")
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        real = GateService.run_gate
        calls = 0
        worktree = self.worktree()
        cache = worktree / ".workspace-cache"
        ready = cache / "dependency-ready"
        residue = worktree / "attempt-residue.tmp"

        def crash_once(service, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                cache.mkdir()
                ready.write_text("installed before crash\n", encoding="utf-8")
                residue.write_text("uncommitted attempt output\n", encoding="utf-8")
                raise RuntimeError("crash during deterministic gate")

            green = self.checkpoint()["last_green_commit"]
            self.assertEqual(current_head(worktree), green)
            self.assertEqual(index_tree_sha(worktree), resolve_tree(worktree, green))
            self.assertTrue(ready.is_file())
            self.assertFalse(residue.exists())
            return real(service, *args, **kwargs)

        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with mock.patch.object(GateService, "run_gate", crash_once):
            failed = original.run_text(SPEC, run_id="run")
            self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
            self.assertEqual(self.checkpoint()["phase"], "deterministic_gate")
            self.assertTrue(ready.is_file())
            self.assertTrue(residue.is_file())

            resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertTrue(ready.is_file())
        self.assertFalse(residue.exists())
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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), roles)

    def test_missing_accepted_chain_report_does_not_block_resume(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "candidate_push"):
            original.run_text(SPEC, run_id="run")
        (self.run_dir() / "accepted-chain.json").unlink()

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))

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

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
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
        effective_path = self.run_dir() / "iterations/01/plan/task_plan.json"
        canonical_bytes = effective_path.read_bytes()
        expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
        self.assertEqual(self.checkpoint()["plan_sha256"], expected_hash)
        self.assertIn("CREATE_EXISTING_TO_WRITE", canonical_bytes.decode())
        self.assertIn("WRITE_MISSING_TO_CREATE", canonical_bytes.decode())
        self.assertIn("DROP_MISSING_DELETE", canonical_bytes.decode())

        resumed = self.orchestrator(self.config(), planner=["unused"]).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(effective_path.read_bytes(), canonical_bytes)
        self.assertEqual(hashlib.sha256(effective_path.read_bytes()).hexdigest(), expected_hash)
        plan = read_effective_plan(effective_path.parent, expected_hash)
        step = plan.steps[0]
        self.assertEqual(step.read_set, ("feature.txt :: current content",))
        self.assertEqual(step.write_set, ("feature.txt",))
        self.assertEqual(step.create_set, ("missing.txt",))
        self.assertEqual(step.delete_set, ())

    def test_resume_uses_durable_iteration_two_plan_and_selection(self) -> None:
        original = self.orchestrator(self.config(), planner=[raw_normalization_plan()])
        with crash_at_checkpoint(original, "implement_step"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        run_dir = self.run_dir()
        checkpoint1 = self.checkpoint()
        first_plan = read_iteration_plan(run_dir, 1, checkpoint1["plan_sha256"])
        second_step = replace(first_plan.steps[0], title="Second milestone step")
        second_plan = replace(
            first_plan, milestone_id="M02", milestone_title="Second milestone",
            milestone_goal="Finish the next milestone.", steps=(second_step,),
        )
        second_hash = persist_iteration_plan(run_dir, 2, second_plan)
        second_dir = iteration_plan_dir(run_dir, 2)
        _bundle, _bundle_hash = validate_implementation_bundle(
            second_dir, expected_step_ids=["S01"],
        )
        effective_bytes = (second_dir / "task_plan.json").read_bytes()
        self.assertTrue(first_plan.normalizations)
        self.assertEqual(
            second_hash,
            hashlib.sha256(effective_bytes).hexdigest(),
        )
        normalized = read_iteration_plan(run_dir, 2, second_hash)
        self.assertEqual(persist_iteration_plan(run_dir, 2, normalized), second_hash)
        self.assertEqual((second_dir / "task_plan.json").read_bytes(), effective_bytes)
        self.assertNotEqual(
            (iteration_plan_dir(run_dir, 1) / "steps/S01/contract.md").read_bytes(),
            (second_dir / "steps/S01/contract.md").read_bytes(),
        )

        selection1 = read_execution_selection(run_dir, iteration=1)
        ensure_execution_selection(run_dir, selection1, iteration=2)
        checkpoint2 = ResumeCheckpoint(
            phase=ResumePhase.IMPLEMENT_STEP, iteration=2,
            step_index=checkpoint1["step_index"],
            last_green_commit=checkpoint1["last_green_commit"],
            plan_sha256=second_hash,
        )
        write_checkpoint(run_dir, checkpoint2)
        resumed = prepare_resume(
            config=self.config(), run_dir=run_dir, run_id="run",
            state=self.state(), checkpoint=checkpoint2, restore_worktree=False,
        )
        self.assertEqual(resumed.checkpoint, checkpoint2)
        self.assertEqual(resumed.plan.milestone_id, "M02")
        self.assertEqual([step.id for step in resumed.plan.steps], ["S01"])
        self.assertEqual([item.step_id for item in resumed.selection.steps], ["S01"])
        self.assertEqual(resumed.selection.planner.profile_id, self.state()["run_options"]["profiles"]["planner_profile"])
        self.assertEqual(resumed.selection.audit.profile_id, self.state()["run_options"]["profiles"]["audit_profile"])

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
