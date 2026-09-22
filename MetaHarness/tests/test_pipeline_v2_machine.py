"""End-to-end behavior of the generic pipeline-v2 state machine."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest import mock

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

    def test_review_revise_without_budget_is_exhausted_without_a_repair_task(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_REPAIR_EXHAUSTED")
        self.assertFalse((self.run_dir() / "repair_task.json").exists())


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
    def test_three_direct_implementation_corrections_use_reviser_only(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
            write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(review_repair=3),
            planner=[initial_plan(STEP)],
            reviewer=[
                review("REVISE", "IMPLEMENTATION"), review("REVISE", "IMPLEMENTATION"),
                review("REVISE", "IMPLEMENTATION"), review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        state = self.state()
        self.assertEqual(state["cycle"], 4)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser", "reviser"])
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(len(self.reviewer.requests), 4)
        self.assertEqual((run_dir / "cycles/002/correction/execution_selection.json").exists(), False)

    def test_replan_uses_one_repair_planner_and_implementer_step(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "REPLAN"), review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])
        self.assertEqual(len(self.planner.requests), 2)
        selection = self.run_dir() / "cycles/002/correction/execution_selection.json"
        self.assertTrue(selection.is_file())
        self.assertEqual(json.loads(selection.read_text())["steps"][0]["implementer"]["profile_id"], "worker")
        self.assertEqual(len(self.reviewer.requests), 2)

    def test_implementation_then_replan_has_distinct_traces(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(review_repair=2),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review("REVISE", "REPLAN"), review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "implementer"])

    def test_review_budget_exhaustion_is_exact(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"), write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=2), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "REVIEW_REPAIR_EXHAUSTED")
        self.assertEqual(failure["detail"]["last_review_cycle"], 3)
        self.assertEqual(failure["detail"]["corrections_used"], 2)
        self.assertEqual(len(self.reviewer.requests), 3)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])

    def test_review_budget_without_check_repair_profile_is_valid(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=2, check_repair=0), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        selection = json.loads((self.run_dir() / "execution_selection.json").read_text())
        self.assertIsNone(selection["check_repair"])
        self.assertEqual(selection["semantic_reviser"]["profile_id"], "reviser")
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])


class ScopeApprovalTests(PipelineHarness):
    def test_a_correction_scope_expansion_waits_for_approval_then_resumes(self) -> None:
        from metaharness.run_options import RunOptions
        from metaharness.web.api import approve_repair_scope

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        config = self.config(review_repair=1)
        options = RunOptions.from_config(config, repair_scope_policy="require-approval")
        waiting = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "REPLAN")],
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
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])


class SemanticRevisionTests(PipelineHarness):
    def test_initial_gate_runs_before_semantic_revision(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'good', 'good semantic'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good semantic\n"))
        result = self.orchestrator(
            self.config(check_repair=2, semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "reviser"])
        run_dir = self.run_dir()
        self.assertTrue((run_dir / "cycles/001/semantic-revision/report.json").is_file())
        self.assertTrue((run_dir / "cycles/001/checks/post-implementation/accepted.json").is_file())
        self.assertTrue((run_dir / "cycles/001/checks/post-semantic-revision/evidence.json").is_file())
        self.assertEqual(
            git(self.worktree(), "log", "-1", "--format=%s"),
            "metaharness(semantic-revision): accepted",
        )

    def test_semantic_red_tree_is_not_committed_before_repair(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(check_repair=2, semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "repair"])
        accepted = json.loads((self.run_dir() / "accepted-chain.json").read_text())['commits']
        red_tree = json.loads(
            (self.run_dir() / "cycles/001/checks/post-semantic-revision/attempts/01/evidence.json").read_text()
        )["staged_tree_sha"]
        self.assertNotIn(red_tree, {item.get("tree_sha", item.get("tree_after")) for item in accepted})
        self.assertEqual(
            git(self.worktree(), "show", "HEAD:feature.txt").strip(), "good",
        )

    def test_gate_episode_budgets_are_independent(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(check_repair=2, semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair", "reviser", "repair"])
        self.assertEqual(
            len(list((self.run_dir() / "cycles/001/check-repair/post-implementation/attempts").iterdir())), 2,
        )
        self.assertEqual(
            len(list((self.run_dir() / "cycles/001/check-repair/post-semantic-revision/attempts").iterdir())), 1,
        )


class ResumeTests(PipelineHarness):
    def test_green_evidence_without_acceptance_is_reused_on_resume(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with mock.patch.object(
            type(original), "_accept_gate_state",
            side_effect=RuntimeError("crash after green evidence"),
        ):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertFalse(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )
        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertTrue(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )

    def test_integrity_failure_does_not_start_repair_or_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        original = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        real_gate = original._run_gate

        def unexpected_head(_self, store, ctx, cycle_plan, stage):
            evidence = real_gate(store, ctx, cycle_plan, stage)
            return replace(
                evidence, deterministic_passed=False,
                failures=("UNEXPECTED_HEAD: moved",),
            )

        with mock.patch.object(type(original), "_run_gate", side_effect=unexpected_head):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(self.reviewer.requests, [])
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
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["review_cycle"]), ("final_review", 2))

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

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
