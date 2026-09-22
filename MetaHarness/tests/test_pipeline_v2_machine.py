"""End-to-end behavior of the generic pipeline-v2 state machine."""

from __future__ import annotations

import json
import unittest

from metaharness.llm.chat import LLMError
from metaharness.models import ExecutionRole, RunStatus
from tests.pipeline_support import (
    PipelineHarness,
    correction_plan,
    git,
    initial_plan,
    review,
    write,
)

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")


class SingleCycleTests(PipelineHarness):
    def test_green_pass_commits_the_accepted_step_as_the_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED)
        state = self.state()
        run_dir = self.run_dir()
        self.assertEqual(state["pipeline_version"], 2)
        self.assertEqual(state["cycle"], 1)
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD^"), self.base_sha)
        self.assertEqual(state["commit_sha"], git(self.worktree(), "rev-parse", "HEAD"))
        candidate = json.loads((run_dir / "cycles/001/candidate/commit.json").read_text())
        self.assertEqual(candidate["commit_sha"], state["commit_sha"])
        self.assertEqual(candidate["gate_stage"], "POST_IMPLEMENTATION")
        self.assertEqual(sorted(state["candidate"]), ["001"])
        for relative in (
            "cycles/001/cycle.json",
            "cycles/001/implementation/steps/S01/step.json",
            "cycles/001/checks/post-implementation/evidence.json",
            "cycles/001/review/review.json",
        ):
            self.assertTrue((run_dir / relative).is_file(), relative)
        self.assertEqual(
            json.loads((run_dir / "cycles/001/cycle.json").read_text()),
            {"kind": "initial", "number": 1, "schema_version": 1},
        )
        self.assertEqual(self.checkpoint()["status"], "completed")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)

    def test_new_run_writes_only_current_option_and_selection_names(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.orchestrator(
            self.config(check_repair=1, review_repair=1),
            planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        options = json.loads((self.run_dir() / "run_options.json").read_text())
        selection = json.loads((self.run_dir() / "execution_selection.json").read_text())
        self.assertEqual(options["pipeline_version"], 2)
        option_names = set(options) | {
            name for section in ("planning", "pipeline", "profiles") for name in options[section]
        }
        for legacy in (
            "claude_revision_enabled", "repair_cycles", "reviser_profile",
            "repair_profile", "reviewer_profile",
        ):
            self.assertNotIn(legacy, option_names)
        self.assertEqual(self.state()["run_options"], options)
        self.assertEqual(
            set(selection),
            {"schema_version", "planner", "steps", "check_repair", "semantic_reviser", "final_reviewer"},
        )
        for legacy in ("reviser", "repair_implementer", "reviewer", "implementer"):
            self.assertNotIn(legacy, selection)
        self.assertEqual(selection["check_repair"]["profile_id"], "repairer")
        self.assertEqual(selection["final_reviewer"]["profile_id"], "reviewer")

    def test_red_gate_without_repair_budget_never_reaches_the_reviewer(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(self.reviewer.requests, [])
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

    def test_review_revise_without_budget_hands_the_task_to_an_operator(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_REVISE")
        task = json.loads((self.run_dir() / "repair_task.json").read_text())
        self.assertEqual(task["route"], "IMPLEMENTATION")


class CheckRepairTests(PipelineHarness):
    def test_red_gate_is_repaired_then_committed_as_a_repair_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        attempt = run_dir / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertTrue((run_dir / "cycles/001/checks/post-implementation/attempts/01/evidence.json").is_file())
        self.assertEqual(
            git(self.worktree(), "log", "-1", "--format=%s"),
            "metaharness(check-repair): cycle 1",
        )
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_every_attempt_of_the_budget_is_used_then_the_gate_is_exhausted(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n"), write("feature.txt", "worse\n"),
            write("feature.txt", "nope\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=3), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair", "repair"])
        attempts = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts"
        self.assertEqual(sorted(path.name for path in attempts.iterdir()), ["001", "002", "003"])
        self.assertEqual(self.reviewer.requests, [])


class ReviewCorrectionCycleTests(PipelineHarness):
    def test_review_corrections_run_generic_cycles_beyond_two(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("other.txt", "second\n"), write("other.txt", "third\n"),
        )
        result = self.orchestrator(
            self.config(review_repair=3),
            planner=[
                initial_plan(STEP),
                correction_plan(("S01", "other.txt", "Correct other")),
                correction_plan(("S01", "other.txt", "Correct other again")),
            ],
            reviewer=[
                review("REVISE", "IMPLEMENTATION"), review("REVISE", "REPLAN"), review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        state = self.state()
        self.assertEqual(state["cycle"], 3)
        self.assertEqual(sorted(state["candidate"]), ["001", "002", "003"])
        kinds = [json.loads((run_dir / f"cycles/{n:03d}/cycle.json").read_text())["kind"] for n in (1, 2, 3)]
        self.assertEqual(kinds, ["initial", "review-implementation", "review-replan"])
        self.assertTrue((run_dir / "cycles/002/checks/post-review-implementation/evidence.json").is_file())
        self.assertTrue((run_dir / "cycles/003/checks/post-review-replan/evidence.json").is_file())
        self.assertTrue((run_dir / "cycles/003/correction/implementation_bundle.json").is_file())
        # One linear chain: BASE <- S01 <- correction 002 <- correction 003.
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "4")
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        self.assertEqual(len(self.planner.requests), 3)
        self.assertEqual([cycle["number"] for cycle in state["cycles"]], [1, 2, 3])

    def test_review_budget_bounds_the_number_of_correction_cycles(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REPAIR_EXHAUSTED")
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertFalse((self.run_dir() / "cycles/003").exists())


class ScopeApprovalTests(PipelineHarness):
    def test_a_correction_scope_expansion_waits_for_approval_then_resumes(self) -> None:
        from metaharness.run_options import RunOptions
        from metaharness.web.api import approve_repair_scope

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, write("other.txt", "second\n"))
        config = self.config(review_repair=1)
        options = RunOptions.from_config(config, repair_scope_policy="require-approval")
        waiting = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run", run_options=options)
        self.assertEqual(waiting.status, RunStatus.WAITING_SCOPE_APPROVAL)
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["review_cycle"]), ("review_replan", 2))
        delta = json.loads((self.run_dir() / "cycles/002/correction/scope_delta.json").read_text())
        self.assertEqual(delta["added_paths"], ["other.txt"])
        self.assertEqual(self.workers.roles(), ["implementer"])

        approve_repair_scope(self.root / "runs", "run", "APPROVE")
        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The durable correction answer is reused: no second planner call.
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])


class SemanticRevisionTests(PipelineHarness):
    def test_revision_runs_before_its_own_gate_stage(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "almost\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        self.assertTrue((run_dir / "cycles/001/semantic-revision/report.json").is_file())
        self.assertTrue((run_dir / "cycles/001/checks/post-semantic-revision/evidence.json").is_file())
        self.assertEqual(
            git(self.worktree(), "log", "-1", "--format=%s"),
            "metaharness(semantic-revision): accepted",
        )


class ResumeTests(PipelineHarness):
    def test_reviewer_transport_failure_resumes_at_the_final_review_of_its_cycle(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, write("other.txt", "second\n"))
        config = self.config(review_repair=2)
        orchestrator = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION"), LLMError("transport down")],
        )
        failed = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["review_cycle"]), ("final_review", 2))

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_a_failed_check_repair_attempt_resumes_at_that_attempt(self) -> None:
        def timeout(request):  # the repair worker edits in scope, then times out
            (request.worktree / "feature.txt").write_text("half\n", encoding="utf-8")
            from metaharness.agent import AgentRunResult
            from metaharness.gitops import candidate_tree_sha
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=candidate_tree_sha(request.worktree), usage=None,
                external_session_id=None, report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, timeout, write("feature.txt", "good\n"))
        config = self.config(check_repair=2)
        failed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_TIMEOUT")
        checkpoint = self.checkpoint()
        self.assertEqual(
            (checkpoint["phase"], checkpoint["stage"], checkpoint["check_repair_attempt"]),
            ("check_repair", "POST_IMPLEMENTATION", 1),
        )

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertTrue((attempt / "attempt.json").is_file())
        # The failed try keeps its own artifacts; the durable attempt record
        # belongs to the successful one only.
        self.assertTrue((attempt / "attempts/01/failure.json").is_file())
        self.assertFalse((attempt / "failure.json").exists())
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])

    def test_a_failed_step_is_rerun_alone_after_its_partial_tree_is_restored(self) -> None:
        def crash(request):  # the worker edits in scope, then fails
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            from metaharness.agent import AgentRunResult
            from metaharness.gitops import candidate_tree_sha
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=candidate_tree_sha(request.worktree), usage=None,
                external_session_id=None, report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, crash, write("feature.txt", "good\n"))
        config = self.config()
        failed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_TIMEOUT")
        self.assertEqual(self.checkpoint()["phase"], "implement_step")

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.state()["resume"]["restored_paths"], ["feature.txt"])
        self.assertTrue(
            (self.run_dir() / "cycles/001/implementation/steps/S01/attempts/01/step.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
