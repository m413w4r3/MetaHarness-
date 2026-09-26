"""Resume: durable boundaries are honored, tampered authority fails closed."""

from __future__ import annotations

import json
import unittest

from metaharness.config import load_config
from metaharness.llm.chat import LLMError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    check_repair_result,
    correction_plan,
    crash_at_checkpoint,
    crash_on_review,
    crash_on_revision,
    git,
    initial_plan,
    ladder_ledger,
    repaired_step_contract,
    review,
    write,
)


class ResumeTests(PipelineHarness):
    def _interrupt_before_replan_planner(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial_plan(STEP), LLMError("planner transport down")],
            reviewer=[review("REVISE", "REPLAN")],
        )
        failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "review_replan")
        self.assertEqual(self.checkpoint()["review_cycle"], 2)

    def test_crash_after_worker_resumes_checks_without_replaying_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "INTERNAL_HARNESS_ERROR")
        self.assertEqual(failure["detail"]["exception_type"], "RuntimeError")
        # The aborted boundary was the gate: the worker stays durable.
        self.assertEqual(failure["detail"]["phase"], "step_acceptance")
        self.assertEqual(failure["detail"]["operation"], "pipeline_coordinator")

        self.assertNotIn("Traceback", json.dumps(failure))
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(self.checkpoint()["phase"], "step_acceptance")
        self.assertEqual(self.workers.roles(), ["implementer"])

        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(
            [event["event"] for event in self.trace_events() if event["event"] == "checks.started"],
            ["checks.started"],
        )

    def test_tampered_correction_kind_fails_before_resume_planner(self) -> None:
        self._interrupt_before_replan_planner()
        path = self.run_dir() / "cycles/002/cycle.json"
        record = json.loads(path.read_text())
        record["kind"] = "review-implementation"
        path.write_text(json.dumps(record), encoding="utf-8")

        resumed = self.orchestrator(
            self.config(review_repair=1), planner=["unused"], reviewer=["unused"],
        )
        result = resumed.resume("run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_tampered_previous_reviewer_artifact_fails_before_resume_planner(self) -> None:
        self._interrupt_before_replan_planner()
        path = self.run_dir() / "cycles/001/review/review.json"
        review_payload = json.loads(path.read_text())
        review_payload["route"] = "IMPLEMENTATION"
        path.write_text(json.dumps(review_payload), encoding="utf-8")

        resumed = self.orchestrator(
            self.config(review_repair=1), planner=["unused"], reviewer=["unused"],
        )
        result = resumed.resume("run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_cycle_record_route_kind_mismatch_fails_closed_without_repair_or_worker(self) -> None:
        self._interrupt_before_replan_planner()
        path = self.run_dir() / "cycles/002/cycle.json"
        record = json.loads(path.read_text())
        record["source_route"] = "IMPLEMENTATION"
        path.write_text(json.dumps(record), encoding="utf-8")

        resumed = self.orchestrator(
            self.config(review_repair=1), planner=["unused"], reviewer=["unused"],
        )
        result = resumed.resume("run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_cycle_four_resume_is_not_limited_to_two_cycles(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
            write("feature.txt", "good\n"),
        )
        original = self.orchestrator(
            self.config(review_repair=3), planner=[initial_plan(STEP)],
            reviewer=[
                review("REVISE", "IMPLEMENTATION"), review("REVISE", "IMPLEMENTATION"),
                review("REVISE", "IMPLEMENTATION"), review(),
            ],
        )
        initial_planner = self.planner
        with crash_on_review(original, 4):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["review_cycle"], 4)
        self.assertEqual(self.checkpoint()["phase"], "final_review")

        resumed = self.orchestrator(
            self.config(review_repair=3), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.state()["cycle"], 4)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser", "reviser"])
        self.assertEqual(len(initial_planner.requests), 1)
        self.assertEqual(len(self.planner.requests), 0)

    def test_profile_selection_snapshot_survives_live_default_changes(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'good', 'good semantic'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good semantic\n"), write("feature.txt", "good semantic\n"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        config = self.config(check_repair=1, review_repair=1, semantic_revision=True)

        def change_live_defaults(_run_dir):
            text = self.config_path.read_text(encoding="utf-8")
            for old, new in {
                'default_planner_profile = "planner"': 'default_planner_profile = "live_planner"',
                'default_implementer_profile = "worker"': 'default_implementer_profile = "live_worker"',
                'default_reviewer_profile = "reviewer"': 'default_reviewer_profile = "live_reviewer"',
                'default_reviser_profile = "reviser"': 'default_reviser_profile = "live_reviser"',
                'default_repair_profile = "repairer"': 'default_repair_profile = "live_repairer"',
            }.items():
                text = text.replace(old, new)
            self.config_path.write_text(text, encoding="utf-8")

        original = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "REPLAN"), review()],
        )

        with crash_at_checkpoint(original, "deterministic_gate"):
            failed = original.run_text(SPEC, run_id="run", on_created=change_live_defaults)
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)

        resumed = self.orchestrator(
            load_config(self.config_path),
            planner=[correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "REPLAN"), review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        selection = json.loads((self.run_dir() / "execution_selection.json").read_text())
        self.assertEqual(selection["planner"]["profile_id"], "planner")
        self.assertEqual(selection["semantic_reviser"]["profile_id"], "reviser")
        self.assertEqual(selection["check_repair"]["profile_id"], "repairer")
        self.assertEqual(selection["final_reviewer"]["profile_id"], "reviewer")
        correction_selection = json.loads(
            (self.run_dir() / "cycles/002/correction/execution_selection.json").read_text()
        )
        self.assertEqual(correction_selection["steps"][0]["implementer"]["profile_id"], "worker")
        self.assertEqual(
            [call.profile_id for call in self.workers.calls],
            ["worker", "repairer", "reviser", "worker", "reviser"],
        )

    def test_reviewer_transport_failure_resumes_at_the_final_review_of_its_cycle(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        config = self.config(review_repair=2)
        orchestrator = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION"), LLMError("transport down")],
        )
        failed = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["review_cycle"]), ("final_review", 2))

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

    def test_green_evidence_without_acceptance_is_reused_on_resume(self) -> None:
        """A green evidence file is not resume authority without its acceptance."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        accepted_path = self.run_dir() / "cycles/001/checks/post-implementation/accepted.json"
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_at_checkpoint(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertTrue(accepted_path.is_file())
        # The accepted tree stays green; the durable acceptance record is the
        # only resume authority, so a resume re-derives it before publishing.
        accepted_path.unlink()

        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertTrue(accepted_path.is_file())

    def test_tampered_accepted_scope_is_rejected_before_resume_agents(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_at_checkpoint(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        accepted_path = self.run_dir() / "cycles/001/checks/post-implementation/accepted.json"
        accepted = json.loads(accepted_path.read_text())
        accepted["mutable_scope"] = ["feature.txt", "tests/test_feature.py"]
        accepted_path.write_text(json.dumps(accepted), encoding="utf-8")

        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(self.reviewer.requests, [])

    def test_resume_after_the_re_decomposition_never_buys_a_second_plan(self) -> None:
        """A crash after the new plan artifact reuses it instead of re-planning."""

        rewritten = ("S01", "feature.txt", "Rewrite the feature so the configured test passes")
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
            write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "bad\n"))
        config = self.config(check_repair=1)
        original = self.orchestrator(
            config,
            planner=[
                initial_plan(STEP), repaired_step_contract(), correction_plan(rewritten),
            ],
            reviewer=[review()],
        )
        with crash_at_checkpoint(original, "check_replan"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL, self.state().get("failure"))
        self.assertEqual(len(original._runtime.planner_client.requests), 3)
        root = self.run_dir()
        record = root / "cycles/002/check-replan/check_replan.plan.json"
        durable = record.read_bytes()
        self.assertTrue(record.is_file())
        # The rung is durably done and nothing of its new cycle was executed:
        # the crash landed between the new plan and its implementation.
        self.assertEqual(
            [entry["state"] for entry in ladder_ledger(self)["entries"]],
            ["done", "done", "done"],
        )
        self.assertFalse((root / "cycles/002/implementation/steps/S01/step.json").exists())
        self.assertTrue(resume_info(root, self.state()).resumable)

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The one durable answer is the plan the resumed run executed: the
        # planner was never asked again for it.
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(record.read_bytes(), durable)
        self.assertEqual(
            json.loads((root / "cycles/002/cycle.json").read_text())["kind"], "check-replan",
        )
        step = json.loads((root / "cycles/002/implementation/steps/S01/step.json").read_text())
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual((self.worktree() / "feature.txt").read_text(), "good\n")

    def test_corrupt_latest_gate_evidence_fails_resume_integrity(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
        )
        self.workers.on(ExecutionRole.REPAIR, lambda _request: check_repair_result())
        config = self.config(check_repair=1)
        waiting = self.orchestrator(
            config,
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                # The last rung re-decomposes the cycle; the plan already in
                # force re-decomposes nothing, so it is refused and spent.
                initial_plan(STEP),
            ],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        evidence_path = self.run_dir() / "cycles/001/checks/post-implementation/evidence.json"
        evidence_path.write_text(evidence_path.read_text() + " ", encoding="utf-8")
        roles_before_resume = self.workers.roles()

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles_before_resume)


class ResumeAuthorityTests(PipelineHarness):
    """Legitimate durable states resume; tampered authority fails closed."""

    def _crash_before_no_change_review(self) -> str:
        (self.repo / "feature.txt").write_text("good\n", encoding="utf-8")
        git(self.repo, "add", "feature.txt")
        git(self.repo, "commit", "-qm", "already satisfied")
        base = git(self.repo, "rev-parse", "HEAD")
        self.workers.on(ExecutionRole.IMPLEMENTER, lambda _request: "done\n", lambda _request: "done\n")
        original = self.orchestrator(
            self.config(max_step_contract_repairs=1),
            planner=[initial_plan(STEP), repaired_step_contract()], reviewer=["unused"],
        )
        with crash_on_review(original, 1):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "final_review", self.state().get("failure"))
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), base)
        return base

    def test_no_change_resume_reuses_existing_candidate(self) -> None:
        base = self._crash_before_no_change_review()
        confirmed = review().replace("SUMMARY: scripted review", "SUMMARY: SPEC_ALREADY_SATISFIED: feature.txt is good")
        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=[confirmed],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), base)
        self.assertEqual(self.state()["no_change_candidate_sha"], base)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    def _reject_tampered_no_change_candidate(self, field: str) -> None:
        self._crash_before_no_change_review()
        path = self.run_dir() / "cycles/001/candidate/commit.json"
        candidate = json.loads(path.read_text())
        candidate[field] = "a" * 40
        path.write_text(json.dumps(candidate), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.reviewer.requests, [])

    def test_no_change_resume_rejects_tampered_tree(self) -> None:
        self._reject_tampered_no_change_candidate("tree_sha")

    def test_no_change_resume_rejects_tampered_commit(self) -> None:
        self._reject_tampered_no_change_candidate("commit_sha")

    def test_changed_candidate_resume_still_rejects_wrong_parent(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=["unused"],
        )
        with crash_on_review(original, 1):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        path = self.run_dir() / "cycles/001/candidate/commit.json"
        candidate = json.loads(path.read_text())
        candidate["parent_sha"] = "a" * 40
        path.write_text(json.dumps(candidate), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")

    def test_changed_direct_correction_resumes_at_its_final_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        # A real correction: the accepted tree differs from the reviewed one.
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        original = self.orchestrator(
            self.config(review_repair=1), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        )
        with crash_on_review(original, 2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual((self.checkpoint()["phase"], self.checkpoint()["review_cycle"]), ("final_review", 2))

        resumed = self.orchestrator(
            self.config(review_repair=1), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])
        self.assertEqual(self.planner.requests, [])
        candidate = json.loads((self.run_dir() / "cycles/002/candidate/commit.json").read_text())
        first = json.loads((self.run_dir() / "cycles/001/candidate/commit.json").read_text())
        self.assertEqual(candidate["parent_sha"], first["commit_sha"])
        self.assertEqual(self.state()["commit_sha"], candidate["commit_sha"])

    def test_crash_after_an_existing_head_acceptance_resumes(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(self.config(), planner=[initial_plan(STEP)], reviewer=[review()])
        with crash_at_checkpoint(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertTrue(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").is_file()
        )
        resumed = self.orchestrator(self.config(), planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])

    def _crash_after_repair_commit(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_at_checkpoint(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["check_repair_attempt"]), ("deterministic_gate", 1))
        accepted = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").read_text()
        )
        self.assertEqual(accepted["acceptance_kind"], "repair")
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), accepted["commit_sha"])

    def test_crash_after_an_accepted_repair_commit_resumes_without_a_new_repair(self) -> None:
        self._crash_after_repair_commit()
        resumed = self.orchestrator(
            self.config(check_repair=1), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_a_head_not_recorded_by_the_gate_acceptance_fails_closed(self) -> None:
        self._crash_after_repair_commit()
        path = self.run_dir() / "cycles/001/checks/post-implementation/accepted.json"
        accepted = json.loads(path.read_text())
        accepted["commit_created"] = False
        path.write_text(json.dumps(accepted), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(check_repair=1), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])
        self.assertEqual(self.reviewer.requests, [])

    def _crash_in_semantic_revision(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(semantic_revision=True), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_on_revision(original):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(
            self.checkpoint()["phase"], "semantic_revision", self.state().get("failure"),
        )
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_semantic_revision_resumes_after_its_green_gate(self) -> None:
        self._crash_in_semantic_revision()
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        resumed = self.orchestrator(
            self.config(semantic_revision=True), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

    def test_semantic_revision_never_resumes_without_its_green_gate_acceptance(self) -> None:
        self._crash_in_semantic_revision()
        (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").unlink()
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        resumed = self.orchestrator(
            self.config(semantic_revision=True), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        # The checkpoint lacked its green gate acceptance, so no reviser runs.
        self.assertEqual(self.workers.roles(), ["implementer"])

    def _three_changed_cycles_crashing_at_the_third_review(self):
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n\n"), write("feature.txt", "good\n\n\n"),
        )
        original = self.orchestrator(
            self.config(review_repair=2), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review("REVISE", "IMPLEMENTATION")],
        )
        with crash_on_review(original, 3):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual((self.checkpoint()["phase"], self.checkpoint()["review_cycle"]), ("final_review", 3))

    def test_cycle_three_resume_publishes_the_third_candidate(self) -> None:
        self._three_changed_cycles_crashing_at_the_third_review()
        resumed = self.orchestrator(
            self.config(review_repair=2), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        candidates = [
            json.loads((self.run_dir() / f"cycles/{number:03d}/candidate/commit.json").read_text())
            for number in (1, 2, 3)
        ]
        self.assertEqual(len({item["commit_sha"] for item in candidates}), 3)
        self.assertEqual(self.state()["commit_sha"], candidates[2]["commit_sha"])

    def _assert_tampering_fails_before_any_call(self, relative: str, tamper) -> None:
        self._three_changed_cycles_crashing_at_the_third_review()
        path = self.run_dir() / relative
        path.write_text(json.dumps(tamper(json.loads(path.read_text()))), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(review_repair=2), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])

    def test_tampered_earlier_candidate_parent_fails_before_any_call(self) -> None:
        self._assert_tampering_fails_before_any_call(
            "cycles/001/candidate/commit.json",
            lambda payload: {**payload, "parent_sha": "0" * 40},
        )

    def test_tampered_initial_cycle_record_fails_before_any_call(self) -> None:
        self._assert_tampering_fails_before_any_call(
            "cycles/001/cycle.json", lambda payload: {**payload, "kind": "review-replan"},
        )

    def test_tampered_run_execution_selection_fails_before_any_call(self) -> None:
        self._three_changed_cycles_crashing_at_the_third_review()
        path = self.run_dir() / "execution_selection.json"
        payload = json.loads(path.read_text())
        payload["steps"][0]["implementer"]["model"] = "tampered-model"
        path.write_text(json.dumps(payload), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(review_repair=2), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.reviewer.requests, [])

    def test_tampered_replan_cycle_execution_selection_fails_before_any_call(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        original = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "REPLAN")],
        )
        with crash_on_review(original, 2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        path = self.run_dir() / "cycles/002/correction/execution_selection.json"
        payload = json.loads(path.read_text())
        payload["steps"][0]["implementer"]["model"] = "tampered-model"
        path.write_text(json.dumps(payload), encoding="utf-8")
        resumed = self.orchestrator(
            self.config(review_repair=1), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    def _interrupt_before_the_second_repair_pass(self):
        """Attempt 001 leaves the gate red; the pending pass 002 never starts."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "still bad\n"))
        config = self.config(check_repair=2)
        original = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with crash_at_checkpoint(original, "check_repair", occurrence=2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        path = self.run_dir() / "resume_checkpoint.json"
        checkpoint = json.loads(path.read_text())
        self.assertEqual(checkpoint["check_repair_attempt"], 1)
        return config, path, checkpoint

    def test_an_interrupted_pending_attempt_never_replays_a_recorded_attempt(self) -> None:
        config, _path, _checkpoint = self._interrupt_before_the_second_repair_pass()
        attempt_one = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        record = (attempt_one / "attempt.json").read_text()

        self.workers.scripts[ExecutionRole.REPAIR].clear()
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # Only the unfinished attempt 002 is retried; 001 is never replayed.
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        self.assertEqual((attempt_one / "attempt.json").read_text(), record)
        self.assertEqual(
            sorted(item.name for item in attempt_one.parent.iterdir()), ["001", "002"],
        )

    def test_a_check_repair_counter_beyond_the_durable_attempts_fails_closed(self) -> None:
        config, path, checkpoint = self._interrupt_before_the_second_repair_pass()
        checkpoint["check_repair_attempt"] = 3
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        resumed = self.orchestrator(config, planner=["unused"], reviewer=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_a_lowered_check_repair_counter_never_replays_a_recorded_attempt(self) -> None:
        config, path, checkpoint = self._interrupt_before_the_second_repair_pass()
        attempt_one = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        record = (attempt_one / "attempt.json").read_text()
        checkpoint["check_repair_attempt"] = 0
        path.write_text(json.dumps(checkpoint), encoding="utf-8")

        resumed = self.orchestrator(config, planner=["unused"], reviewer=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        # The recorded attempt is never replayed to fill the fabricated gap.
        self.assertEqual((attempt_one / "attempt.json").read_text(), record)
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_run_options_without_their_state_hash_are_never_resumed(self) -> None:
        from metaharness.resume import ResumeNotAllowedError
        from metaharness.state import RunStateStore

        self._crash_in_semantic_revision()
        options = self.run_dir() / "run_options.json"
        payload = json.loads(options.read_text())
        payload["max_check_repair_attempts"] = 9
        options.write_text(json.dumps(payload), encoding="utf-8")
        store = RunStateStore(self.run_dir() / "state.json")
        store.update_metadata(run_options_sha256=None)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(
                self.config(semantic_revision=True), planner=["unused"], reviewer=["unused"],
            ).resume("run")
        self.assertEqual(self.workers.roles(), ["implementer"])

if __name__ == "__main__":
    unittest.main()
