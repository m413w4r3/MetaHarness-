"""Resume: durable boundaries are honored, tampered authority fails closed.

The only post-implementation authority is the deterministic gate and its
audit: a resume re-derives that decision from the durable artifacts and never
buys a second plan, a second worker or a second audit report for work the
crash already recorded.
"""

from __future__ import annotations

import json
import unittest

from metaharness.config import load_config
from metaharness.models import RunDisposition, RunMachineState, RunPhase, RunStatus
from metaharness.orchestration.resume_integrity import validate_resume
from metaharness.resume import read_checkpoint, resume_info
from metaharness.state import RunCheckpointError, RunStateStore

from metaharness.models import ExecutionRole
from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    audit,
    crash_at_checkpoint,
    initial_plan,
    write,
)


def raw_plan(
    *, read_set: str = "NONE", write_set: str = "NONE",
    create_set: str = "NONE", delete_set: str = "NONE",
) -> str:
    """The standard plan with exactly these declared change sets."""

    return (
        initial_plan(STEP)
        .replace("READ_SET\n- feature.txt :: current content\n", f"READ_SET\n{read_set}\n")
        .replace("WRITE_SET\n- feature.txt\n", f"WRITE_SET\n{write_set}\n")
        .replace("CREATE_SET\nNONE\n", f"CREATE_SET\n{create_set}\n")
        .replace("DELETE_SET\nNONE\n", f"DELETE_SET\n{delete_set}\n")
    )


class ResumeTests(PipelineHarness):
    def test_crash_after_worker_resumes_checks_without_replaying_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "INTERNAL_HARNESS_ERROR")
        self.assertEqual(failure["detail"]["exception_type"], "RuntimeError")
        self.assertNotIn("Traceback", json.dumps(failure))
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(self.workers.roles(), ["implementer"])

        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=[audit()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        # The worker was never replayed: the resumed run only re-derived the
        # gate and bought the audit the crashed run never recorded.
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])

    def test_profile_selection_snapshot_survives_live_default_changes(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        config = self.config()

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

        original = self.orchestrator(config, planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run", on_created=change_live_defaults)
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)

        resumed = self.orchestrator(
            load_config(self.config_path), planner=["unused"], auditor=[audit()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        selection = json.loads((self.run_dir() / "execution_selection.json").read_text())
        self.assertEqual(selection["planner"]["profile_id"], "planner")
        self.assertEqual(selection["audit"]["profile_id"], "auditor")
        self.assertEqual(selection["steps"][0]["implementer"]["profile_id"], "worker")
        self.assertEqual(
            [call.profile_id for call in self.workers.calls], ["worker", "auditor"],
        )

    def test_candidate_boundary_resume_reuses_the_recorded_candidate(self) -> None:
        """A crash after the candidate boundary reuses the recorded candidate."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "publish"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        record = (self.run_dir() / "cycles/001/candidate/commit.json").read_bytes()
        roles = self.workers.roles()

        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        # No agent is bought again, and the durable candidate is the published one.
        self.assertEqual(self.workers.roles(), roles)
        self.assertEqual((self.run_dir() / "cycles/001/candidate/commit.json").read_bytes(), record)
        published = [
            event["data"]["commit_sha"] for event in self.trace_events()
            if event["event"] == "publish.completed"
        ]
        self.assertEqual(published, [json.loads(record)["commit_sha"]])

    def test_tampered_accepted_scope_is_rejected_before_resume_agents(self) -> None:
        """A tampered gate acceptance is refused by the step authority replay."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "publish"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        accepted_path = self.run_dir() / "cycles/001/checks/post-implementation/accepted.json"
        accepted = json.loads(accepted_path.read_text())
        accepted["mutable_scope"] = ["feature.txt", "tests/test_feature.py"]
        accepted_path.write_text(json.dumps(accepted), encoding="utf-8")

        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        self.assertEqual(self.planner.requests, [])

    def test_corrupt_latest_gate_evidence_fails_resume_integrity(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        roles_before_resume = self.workers.roles()
        evidence_path = self.run_dir() / "cycles/001/checks/post-implementation/evidence.json"
        evidence_path.write_text("{not json", encoding="utf-8")

        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles_before_resume)


class ResumeAuthorityTests(PipelineHarness):
    """Legitimate durable states resume; tampered authority fails closed."""

    def _crash_before_candidate_ready(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "publish"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")

    def test_crash_after_an_existing_head_acceptance_resumes(self) -> None:
        self._crash_before_candidate_ready()
        self.assertTrue(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").is_file()
        )
        roles = self.workers.roles()
        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), roles)

    def test_a_head_not_recorded_by_the_gate_acceptance_fails_closed(self) -> None:
        self._crash_before_candidate_ready()
        path = self.run_dir() / "cycles/001/checks/post-implementation/accepted.json"
        accepted = json.loads(path.read_text())
        accepted["commit_sha"] = "0" * 40
        path.write_text(json.dumps(accepted), encoding="utf-8")
        roles = self.workers.roles()
        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles)
        self.assertEqual(self.planner.requests, [])

    def test_tampered_run_execution_selection_fails_before_any_call(self) -> None:
        self._crash_before_candidate_ready()
        path = self.run_dir() / "execution_selection.json"
        payload = json.loads(path.read_text())
        payload["steps"][0]["implementer"]["model"] = "tampered-model"
        path.write_text(json.dumps(payload), encoding="utf-8")
        roles = self.workers.roles()
        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.workers.roles(), roles)

    def test_run_options_without_their_state_hash_are_never_resumed(self) -> None:
        from metaharness.resume import ResumeNotAllowedError

        self._crash_before_candidate_ready()
        options = self.run_dir() / "run_options.json"
        payload = json.loads(options.read_text())
        payload["max_step_contract_repairs"] = 9
        options.write_text(json.dumps(payload), encoding="utf-8")
        store = RunStateStore(self.run_dir() / "state.json")
        store.update_metadata(run_options_sha256=None)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(
                self.config(), planner=["unused"], auditor=["unused"],
            ).resume("run")
        self.assertEqual(self.workers.roles().count("implementer"), 1)

    def test_a_corrupt_checkpoint_refuses_control_writes_and_never_falls_back(self) -> None:
        """A checkpoint that exists is the phase authority, corruption included."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)])
        with crash_at_checkpoint(original, "deterministic_gate"):
            original.run_text(SPEC, run_id="run")
        store = RunStateStore(self.run_dir() / "state.json")
        phase = RunPhase(self.checkpoint()["phase"])
        self.assertIs(store.machine_state().phase, phase)
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)

        (self.run_dir() / "resume_checkpoint.json").write_text("{not json", encoding="utf-8")
        for name, mutate in (
            ("machine_state", lambda: store.machine_state()),
            ("identity", lambda: store.identity()),
            ("update_metadata", lambda: store.update_metadata(marker=True)),
            ("set_run_state", lambda: store.set_run_state(RunMachineState(RunPhase.PUBLISH))),
        ):
            with self.subTest(operation=name), self.assertRaises(RunCheckpointError):
                mutate()

        # The terminal reporting read still describes the last recorded phase,
        # and the recorded phase can never make the run resumable again.
        reported = store.reported_machine_state()
        self.assertIs(reported.phase, phase)
        failed = store.record_failure("RESUME_INTEGRITY_FAILURE", "the checkpoint is unreadable")
        self.assertEqual(failed["disposition"], RunDisposition.FAILED.value)
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)


class ResumePlanNormalizationTests(PipelineHarness):
    """A resume rebuilds the effective plan the run really approved."""

    def _crash_at_the_gate(self, raw: str) -> None:
        original = self.orchestrator(self.config(), planner=[raw])
        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)

    def _resumed_plan(self):
        return validate_resume(
            config=load_config(self.config_path),
            run_dir=self.run_dir(), state=self.state(),
            checkpoint=read_checkpoint(self.run_dir()),
            staging_remote="origin",
        ).plan

    def test_a_raw_create_of_an_existing_path_resumes_as_the_write(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self._crash_at_the_gate(raw_plan(read_set="NONE", create_set="- feature.txt"))

        step = self._resumed_plan().steps[0]
        self.assertEqual(step.write_set, ("feature.txt",))
        self.assertEqual(step.create_set, ())
        self.assertEqual(step.read_set, ("feature.txt :: current content",))

    def test_a_raw_delete_of_a_missing_path_is_not_resumed_scope(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self._crash_at_the_gate(raw_plan(
            read_set="- feature.txt :: current content", write_set="- feature.txt",
            delete_set="- missing.txt",
        ))

        step = self._resumed_plan().steps[0]
        self.assertEqual(step.delete_set, ())
        self.assertEqual(step.write_set, ("feature.txt",))
        self.assertNotIn("missing.txt", (*step.read_set, *step.write_set, *step.create_set))

    def test_a_raw_write_of_a_missing_path_resumes_as_the_create(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("missing.txt", "good\n"))
        self._crash_at_the_gate(raw_plan(
            read_set="- missing.txt :: current content", write_set="- missing.txt",
        ))

        step = self._resumed_plan().steps[0]
        self.assertEqual(step.create_set, ("missing.txt",))
        self.assertEqual(step.write_set, ())
        self.assertEqual(step.read_set, ())

    def test_an_incoherent_normalization_record_refuses_the_resume(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self._crash_at_the_gate(raw_plan(read_set="NONE", create_set="- feature.txt"))
        record = self.run_dir() / "plan.normalizations.json"
        self.assertIn("CREATE_EXISTING_TO_WRITE", record.read_text(encoding="utf-8"))
        record.write_text('{"schema": 1, "steps": {}}\n', encoding="utf-8")
        roles = self.workers.roles()

        resumed = self.orchestrator(
            self.config(), planner=["unused"], auditor=["unused"],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles)
        self.assertEqual(self.planner.requests, [])


if __name__ == "__main__":
    unittest.main()
