"""The fundamental pipeline-v2 end-to-end paths."""

from __future__ import annotations

import json
import unittest

from metaharness.models import ExecutionRole, RunStatus
from metaharness.gitops import candidate_tree_sha
from metaharness.orchestrator import Orchestrator
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    ScriptedChat,
    correction_plan,
    git,
    initial_plan,
    ladder_ledger,
    ladder_strategies,
    review,
    write,
)


class LifecycleTests(PipelineHarness):
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
        for forbidden in (
            "claude_revision_enabled", "repair_cycles", "reviser_profile",
            "repair_profile", "reviewer_profile",
        ):
            self.assertNotIn(forbidden, option_names)
        self.assertEqual(self.state()["run_options"], options)
        self.assertEqual(
            set(selection),
            {
                "schema_version", "planner", "steps", "check_repair",
                "check_repair_fallbacks", "semantic_reviser",
                "semantic_reviser_fallbacks", "final_reviewer",
            },
        )
        for forbidden in ("reviser", "repair_implementer", "reviewer", "implementer"):
            self.assertNotIn(forbidden, selection)
        self.assertEqual(selection["check_repair"]["profile_id"], "repairer")
        self.assertEqual(selection["final_reviewer"]["profile_id"], "reviewer")

    def test_red_gate_without_repair_budget_never_reaches_the_reviewer(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
        )
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        # A zero budget refuses the worker pass; the autonomous replan rung of
        # the ladder still ran before the operator was asked.
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])
        self.assertEqual(ladder_strategies(self), ["replan_step"])
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/attempts").exists()
        )
        self.assertEqual(self.reviewer.requests, [])
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

    def test_red_gate_is_repaired_then_committed_as_a_repair_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
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
        # The first distinct strategy of a red gate is the targeted repair and
        # it passed, so no later rung of the ladder was ever consumed.
        self.assertEqual(ladder_strategies(self), ["repair_targeted"])
        entries = ladder_ledger(self)["entries"]
        self.assertEqual(
            [(entry["state"], entry["repair_attempt"]) for entry in entries],
            [("done", 1)],
        )
        self.assertEqual(entries[0]["tree_after"], candidate_tree_sha(self.worktree()))

    def test_review_revise_without_budget_is_exhausted_without_a_repair_task(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "WAITING_REPAIR_EXHAUSTED")
        self.assertEqual(self.checkpoint()["phase"], "final_review")
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertFalse((self.run_dir() / "repair_task.json").exists())

    def test_human_route_stops_without_automatic_correction_or_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=3), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "HUMAN")],
        ).run_text(
            "Return either a file containing 'green' or one containing 'blue'; both outcomes are allowed.",
            run_id="run",
        )

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "HUMAN_REQUIRED")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_malformed_review_is_repaired_to_pass_without_repeating_review(self) -> None:
        malformed = "VERDICT: PASS\nROUTE: NONE\nREQUIRED FIXES: NONE\n"
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[malformed, review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertIn("Do not redo the review", self.reviewer.requests[1])
        self.assertEqual(self.workers.roles(), ["implementer"])

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
        self.assertNotIn("repair", self.workers.roles())
        self.assertEqual(len(self.planner.requests), 2)
        selection = self.run_dir() / "cycles/002/correction/execution_selection.json"
        self.assertTrue(selection.is_file())
        self.assertEqual(json.loads(selection.read_text())["steps"][0]["implementer"]["profile_id"], "worker")
        self.assertEqual(len(self.reviewer.requests), 2)

    def test_transient_step_failure_is_rolled_back_and_retried_automatically(self) -> None:
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
        completed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(completed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.state()["recovery_counters"]["agent-step:001:S01"], 1)
        self.assertTrue(
            (self.run_dir() / "cycles/001/implementation/steps/S01/attempts/01").is_dir()
        )
        recovery = [
            json.loads(line)
            for line in (self.run_dir() / "trace" / "events.v1.jsonl").read_text().splitlines()
            if '"event":"recovery.started"' in line
        ]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(
            recovery[0]["data"]["tree_before"], recovery[0]["data"]["tree_after"],
        )

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

    def test_accepted_git_chain_contains_only_reviewable_green_trees(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'good', 'semantic', 'semantic 2', 'semantic 3', 'semantic 4'} else 1)\n",
            encoding="utf-8",
        )
        steps = (
            ("S01", "feature.txt", "Write the feature"),
            ("S02", "other.txt", "Write the companion"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "semantic\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "bad 2\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "semantic 4\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "semantic 3\n"))
        result = self.orchestrator(
            self.config(check_repair=1, semantic_revision=True, review_repair=2, publish=True),
            planner=[
                initial_plan(*steps),
                correction_plan(("S01", "feature.txt", "Replan the feature")),
            ],
            reviewer=[
                review("REVISE", "IMPLEMENTATION"), review("REVISE", "REPLAN"), review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        commits = git(self.worktree(), "rev-list", "--reverse", "HEAD").splitlines()
        self.assertEqual(commits[0], self.base_sha)
        for commit in commits[1:]:
            self.assertEqual(git(self.worktree(), "rev-list", "--parents", "-n", "1", commit).split().__len__(), 2)
        red_trees = {
            json.loads(path.read_text(encoding="utf-8"))["staged_tree_sha"]
            for path in self.run_dir().glob("cycles/**/checks/**/attempts/**/evidence.json")
            if not json.loads(path.read_text(encoding="utf-8")).get("deterministic_passed", True)
            and "post-implementation" not in path.parts
        }
        commit_trees = {git(self.worktree(), "rev-parse", f"{commit}^{{tree}}") for commit in commits}
        self.assertTrue(red_trees.isdisjoint(commit_trees))
        candidate = json.loads(
            (self.run_dir() / "cycles/003/candidate/commit.json").read_text(encoding="utf-8")
        )
        review_record = json.loads(
            (self.run_dir() / "cycles/003/review/review.json").read_text(encoding="utf-8")
        )
        published = json.loads((self.run_dir() / "publish.json").read_text(encoding="utf-8"))
        reviewed_sha = [
            event["data"]["candidate_sha"]
            for event in self.trace_events()
            if event["event"] == "review.completed" and event.get("cycle") == 3
        ][-1]
        self.assertEqual(review_record["verdict"], "PASS")
        self.assertEqual(candidate["commit_sha"], reviewed_sha)
        self.assertEqual(published["commit_sha"], reviewed_sha)
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), candidate["commit_sha"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])

    def test_trace_proves_pipeline_order_and_model_metadata(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        events = self.trace_events()
        names = [event["event"] for event in events]
        positions = {
            name: names.index(name)
            for name in (
                "run.created", "plan.started", "step.started", "checks.started",
                "checks.completed", "step.committed", "candidate.pushed",
                "review.started", "publish.completed",
            )
        }
        self.assertLess(positions["run.created"], positions["plan.started"])
        self.assertLess(positions["plan.started"], positions["step.started"])
        self.assertLess(positions["step.committed"], positions["checks.started"])
        self.assertLess(positions["checks.completed"], positions["candidate.pushed"])
        self.assertLess(positions["candidate.pushed"], positions["review.started"])
        self.assertLess(positions["review.started"], positions["publish.completed"])
        for event in events:
            data = event.get("data", {})
            session = data.get("session")
            if session is not None:
                self.assertTrue(session.get("driver"))
                self.assertTrue(session.get("provider"))
                self.assertTrue(session.get("model"))
                self.assertIn("effort", session)
                self.assertTrue(session.get("profile_fingerprint"))
                if event["event"] in {"plan.completed", "step.agent.completed", "review.completed"}:
                    self.assertIsInstance(session.get("prompt_bytes"), int)
        stages = [
            event["data"].get("stage")
            for event in events
            if event["event"] == "checks.completed"
        ]
        self.assertEqual(stages, ["POST_IMPLEMENTATION"])

    def test_a_failing_external_observer_cannot_change_the_run(self) -> None:
        class BrokenObserver:
            def __init__(self) -> None:
                self.calls = 0

            def emit(self, event):
                self.calls += 1
                raise RuntimeError("cockpit unavailable")

        observer = BrokenObserver()
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        self.planner = ScriptedChat([initial_plan(STEP)], name="planner", events=self.events)
        self.reviewer = ScriptedChat([review()], name="reviewer", events=self.events)
        result = Orchestrator(
            self.config(check_repair=1, publish=True),
            planner_client=self.planner, reviewer_client=self.reviewer,
            trace_sink=observer,
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertGreater(observer.calls, 0)
        candidate = json.loads((self.run_dir() / "cycles/001/candidate/commit.json").read_text())
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])
        self.assertIn("publish.completed", self.trace_names())

if __name__ == "__main__":
    unittest.main()
