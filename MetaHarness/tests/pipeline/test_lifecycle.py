"""The fundamental pipeline-v2 end-to-end paths.

One authority owns the post-implementation decision: the deterministic gate
hands its evidence to the writable audit, and the accepted tree is what the
candidate and the publication carry.
"""

from __future__ import annotations

import json
import hashlib
import unittest

from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    ScriptedChat,
    audit,
    git,
    initial_plan,
    write,
)
from tests.pipeline_support import ScriptedPlannerMux, continuation_answer


class LifecycleTests(PipelineHarness):
    def test_green_pass_commits_the_accepted_step_as_the_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(self.config(), planner=[initial_plan(STEP)]).run_text(
            SPEC, run_id="run",
        )

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
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
            "cycles/001/checks/post-implementation/accepted.json",
            "cycles/001/audit/001/report.json",
        ):
            self.assertTrue((run_dir / relative).is_file(), relative)
        self.assertEqual(
            json.loads((run_dir / "cycles/001/cycle.json").read_text()),
            {"kind": "initial", "number": 1, "schema_version": 1},
        )
        checkpoint = self.checkpoint()
        self.assertEqual(set(checkpoint), {
            "schema_version", "iteration", "phase", "step_index",
            "last_green_commit", "plan_sha256",
        })
        self.assertEqual(checkpoint["iteration"], 1)
        self.assertEqual(checkpoint["phase"], "publish")
        self.assertIsNone(checkpoint["step_index"])
        self.assertEqual(checkpoint["last_green_commit"], state["commit_sha"])
        plan_path = self.run_dir() / "iterations/01/plan/task_plan.json"
        plan_dir = plan_path.parent
        self.assertEqual(checkpoint["plan_sha256"], hashlib.sha256(plan_path.read_bytes()).hexdigest())
        self.assertTrue((self.run_dir() / "iterations/01/execution_selection.json").is_file())
        check_authority = json.loads((self.run_dir() / "check_authority.json").read_text())
        self.assertEqual(check_authority["schema_version"], 3)
        self.assertEqual(check_authority["default_check_ids"], ["test"])
        for name in (
            "task_plan.json", "implementation_bundle.json", "plan.normalizations.json",
            "implementation_contract.md", "steps/S01/contract.md",
        ):
            self.assertTrue((plan_dir / name).is_file(), name)
        for name in (
            "task_plan.json", "implementation_bundle.json", "plan.normalizations.json",
            "implementation_contract.md", "execution_selection.json",
        ):
            self.assertFalse((self.run_dir() / name).exists(), name)
        self.assertFalse((self.run_dir() / "steps").exists())
        self.assertEqual(state["iteration"], 1)
        self.assertEqual(state["current_milestone"]["id"], "M01")
        self.assertTrue(state["current_milestone"]["title"])
        self.assertEqual([item["id"] for item in state["steps"]], ["S01"])
        self.assertNotIn("steps", state["planner"])
        # The gate accepted the tree and the audit confirmed it: exactly one
        # writable authority ran after the implementation worker.
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])

    def test_new_run_writes_only_current_option_and_selection_names(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.orchestrator(self.config(), planner=[initial_plan(STEP)]).run_text(SPEC, run_id="run")
        options = json.loads((self.run_dir() / "run_options.json").read_text())
        selection = json.loads((self.run_dir() / "iterations/01/execution_selection.json").read_text())
        self.assertFalse((self.run_dir() / "execution_selection.json").exists())
        self.assertEqual(options["schema_version"], 8)
        self.assertEqual(options["pipeline_version"], 2)
        option_names = set(options) | {
            name for section in ("planning", "profiles", "execution_fallbacks", "budget")
            for name in options[section]
        }
        for forbidden in (
            "semantic_revision_enabled", "max_check_repair_attempts",
            "max_correction_cycles", "check_repair_profile",
            "semantic_reviser_profile", "final_reviewer_profile",
            "max_review_transport_retries", "recovery",
        ):
            self.assertNotIn(forbidden, option_names)
        self.assertEqual(
            set(options["profiles"]),
            {
                "planner_profile", "mechanical_profile", "reasoning_profile",
                "agentic_profile", "audit_profile",
            },
        )
        self.assertNotIn("pipeline", options)
        self.assertEqual(
            set(options["budget"]),
            {"step_attempts", "audit_repairs", "max_iterations",
             "max_wall_clock_hours", "max_cost"},
        )
        self.assertEqual(
            set(options["execution_fallbacks"]),
            {"mechanical", "reasoning", "agentic"},
        )
        self.assertEqual(self.state()["run_options"], options)
        self.assertEqual(
            set(selection), {"schema_version", "planner", "steps", "audit"},
        )
        self.assertEqual(selection["audit"]["profile_id"], "auditor")
        self.assertEqual(selection["planner"]["profile_id"], "planner")
        self.assertEqual(
            sorted(selection["steps"][0]),
            ["execution_class", "fallbacks", "implementer", "step_id"],
        )
        self.assertEqual(selection["steps"][0]["implementer"]["profile_id"], "worker")

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
            config, planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(completed.status, RunStatus.PUBLISHED, self.state().get("failure"))
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

    def test_accepted_git_chain_contains_only_green_trees(self) -> None:
        steps = (
            ("S01", "feature.txt", "Write the feature"),
            ("S02", "other.txt", "Write the companion"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(*steps)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        commits = git(self.worktree(), "rev-list", "--reverse", "HEAD").splitlines()
        self.assertEqual(commits[0], self.base_sha)
        for commit in commits[1:]:
            self.assertEqual(
                len(git(self.worktree(), "rev-list", "--parents", "-n", "1", commit).split()), 2,
            )
        accepted = json.loads((self.run_dir() / "accepted-chain.json").read_text())["commits"]
        accepted_shas = [record["commit_sha"] for record in accepted]
        self.assertEqual(accepted_shas, commits[1:])
        self.assertTrue(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").is_file()
        )
        evidence = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/evidence.json").read_text()
        )
        self.assertTrue(evidence["deterministic_passed"])
        candidate = json.loads((self.run_dir() / "cycles/001/candidate/commit.json").read_text())
        published = json.loads((self.run_dir() / "publish.json").read_text())
        self.assertEqual(candidate["commit_sha"], accepted_shas[-1])
        self.assertEqual(published["commit_sha"], accepted_shas[-1])
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), candidate["commit_sha"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])

    def test_trace_proves_pipeline_order_and_model_metadata(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        events = self.trace_events()
        names = [event["event"] for event in events]
        positions = {
            name: names.index(name)
            for name in (
                "run.created", "plan.started", "step.started", "checks.started",
                "checks.completed", "step.committed", "gate.accepted",
                "candidate.pushed", "publish.completed",
            )
        }
        self.assertLess(positions["run.created"], positions["plan.started"])
        self.assertLess(positions["plan.started"], positions["step.started"])
        self.assertLess(positions["step.committed"], positions["checks.started"])
        self.assertLess(positions["checks.completed"], positions["candidate.pushed"])
        self.assertLess(positions["gate.accepted"], positions["candidate.pushed"])
        self.assertLess(positions["candidate.pushed"], positions["publish.completed"])
        for event in events:
            data = event.get("data", {})
            session = data.get("session")
            if session is not None:
                self.assertTrue(session.get("driver"))
                self.assertTrue(session.get("provider"))
                self.assertTrue(session.get("model"))
                self.assertIn("effort", session)
                self.assertTrue(session.get("profile_fingerprint"))
                if event["event"] in {"plan.completed", "step.agent.completed"}:
                    self.assertIsInstance(session.get("prompt_bytes"), int)
        stages = [
            event["data"].get("stage")
            for event in events
            if event["event"] == "checks.completed"
        ]
        self.assertEqual(set(stages), {"POST_IMPLEMENTATION"})

    def test_a_failing_external_observer_cannot_change_the_run(self) -> None:
        class BrokenObserver:
            def __init__(self) -> None:
                self.calls = 0

            def emit(self, event):
                self.calls += 1
                raise RuntimeError("cockpit unavailable")

        observer = BrokenObserver()
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.planner = ScriptedChat([initial_plan(STEP)], name="planner", events=self.events)
        continuation = ScriptedChat(
            [continuation_answer("COMPLETE")], name="planner_continue", events=self.events,
        )
        self.workers.on(ExecutionRole.AUDITOR, audit())
        result = Orchestrator(
            self.config(publish=True),
            planner_client=ScriptedPlannerMux(self.planner, continuation), trace_sink=observer,
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertGreater(observer.calls, 0)
        candidate = json.loads((self.run_dir() / "cycles/001/candidate/commit.json").read_text())
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])
        self.assertEqual(self.workers.roles(), ["implementer", "auditor"])
        self.assertIn("publish.completed", self.trace_names())


if __name__ == "__main__":
    unittest.main()
