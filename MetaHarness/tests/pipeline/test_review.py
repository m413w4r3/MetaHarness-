"""Review verdicts, correction routes and semantic revision."""

from __future__ import annotations

import hashlib
import json
import unittest

from metaharness.llm.chat import LLMHTTPError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestrator import Orchestrator
from metaharness.resume import resume_info
from metaharness.run_options import RunOptions

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    ScriptedChat,
    correction_plan,
    git,
    initial_plan,
    ladder_strategies,
    review,
    write,
)


class ReviewCorrectionTests(PipelineHarness):
    def test_reviewer_fail_stops_without_correction_or_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(correction_cycles=3), planner=[initial_plan(STEP)],
            reviewer=[review("FAIL", "HUMAN")],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_EVIDENCE_UNRESOLVED")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertEqual(self.checkpoint()["phase"], "final_review")
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_malformed_review_twice_keeps_final_review_checkpoint_and_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        failed = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=["VERDICT: PASS\nROUTE: NONE", "still malformed"],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(failed.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_FORMAT_INVALID")
        self.assertEqual(self.checkpoint()["phase"], "final_review")
        self.assertTrue((self.run_dir() / "cycles/001/candidate/commit.json").is_file())
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_review_http_503_recovers_on_the_same_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=[LLMHTTPError("LLM endpoint returned HTTP 503 after 1 attempt(s)"), review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_transport_retry_does_not_spend_replan_correction_budget(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "second\n"),
        )
        result = self.orchestrator(
            self.config(correction_cycles=1),
            planner=[
                initial_plan(STEP),
                correction_plan(("S01", "other.txt", "Correct other")),
            ],
            reviewer=[
                LLMHTTPError("LLM endpoint returned HTTP 503 after 1 attempt(s)"),
                review("REVISE", "REPLAN"), review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.state()["cycle"], 2)
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(len(self.reviewer.requests), 3)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    def test_fail_for_remote_unavailable_retries_from_local_inline_evidence(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
            reviewer=[
                "VERDICT: FAIL\nROUTE: NONE\nSUMMARY: remote unavailable\n"
                "FINDINGS: EVIDENCE_UNAVAILABLE | remote candidate view is unavailable\n"
                "REQUIRED FIXES: NONE\nMISSING TESTS: NONE\nRESIDUAL RISKS: NONE\n",
                review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertIn('"review_evidence_mode": "LOCAL_INLINE_ONLY"', self.reviewer.requests[1])
        self.assertIn("<BOUNDED DIFF EXCERPT>", self.reviewer.requests[1])
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_fail_with_corrupted_local_evidence_is_a_hard_integrity_failure(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        harness = self

        class CorruptingReviewer:
            def __init__(self):
                self.requests = []

            def complete(self, prompt):
                self.requests.append(prompt)
                evidence = harness.run_dir() / "cycles/001/checks/post-implementation/evidence.json"
                evidence.write_text("{}", encoding="utf-8")
                return (
                    "VERDICT: FAIL\nROUTE: NONE\nSUMMARY: evidence invalid\n"
                    "FINDINGS: EVIDENCE_INVALID | durable evidence is corrupt\n"
                    "REQUIRED FIXES: NONE\nMISSING TESTS: NONE\nRESIDUAL RISKS: NONE\n"
                )

        reviewer = CorruptingReviewer()
        result = Orchestrator(
            self.config(),
            planner_client=ScriptedChat([initial_plan(STEP)], name="planner", events=self.events),
            reviewer_client=reviewer,
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(len(reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_three_direct_implementation_corrections_use_reviser_only(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
            write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(correction_cycles=3),
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

    def test_implementation_then_replan_has_distinct_traces(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(correction_cycles=2),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review("REVISE", "REPLAN"), review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "implementer"])

    def test_mixed_review_routes_spend_only_the_route_specific_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(correction_cycles=3),
            planner=[initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other"))],
            reviewer=[
                review("REVISE", "IMPLEMENTATION"), review("REVISE", "REPLAN"),
                review("REVISE", "IMPLEMENTATION"), review(),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "implementer", "reviser"])
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(len(self.reviewer.requests), 4)
        self.assertEqual(self.state()["cycle"], 4)
        expected_routes = {
            2: ("IMPLEMENTATION", "review-implementation"),
            3: ("REPLAN", "review-replan"),
            4: ("IMPLEMENTATION", "review-implementation"),
        }
        for number, (route, kind) in expected_routes.items():
            record = json.loads(
                (self.run_dir() / f"cycles/{number:03d}/cycle.json").read_text()
            )
            self.assertEqual(record["schema_version"], 2)
            self.assertEqual(record["source_review_cycle"], number - 1)
            self.assertEqual(record["source_route"], route)
            self.assertEqual(record["kind"], kind)
            review_path = self.run_dir() / f"cycles/{number - 1:03d}/review/review.json"
            self.assertEqual(
                record["source_review_sha256"],
                hashlib.sha256(review_path.read_bytes()).hexdigest(),
            )
            previous_candidate = json.loads(
                (self.run_dir() / f"cycles/{number - 1:03d}/candidate/commit.json").read_text()
            )
            self.assertEqual(record["source_candidate_sha"], previous_candidate["commit_sha"])

    def test_review_budget_exhaustion_is_exact(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"), write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(correction_cycles=2), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "WAITING_REPAIR_EXHAUSTED")
        self.assertEqual(failure["detail"]["last_review_cycle"], 3)
        self.assertEqual(failure["detail"]["corrections_used"], 2)
        self.assertTrue(failure["detail"]["same_findings_as_previous_cycle"])
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(len(self.reviewer.requests), 3)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])

    def test_review_budget_without_check_repair_profile_is_valid(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(correction_cycles=2, check_repair=0), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        selection = json.loads((self.run_dir() / "execution_selection.json").read_text())
        self.assertIsNone(selection["check_repair"])
        self.assertEqual(selection["semantic_reviser"]["profile_id"], "reviser")
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

    def test_review_and_check_replans_share_one_budget(self) -> None:
        """The unit a review replan spends is the one the gate rung is refused.

        Cycle 001 commits, the review routes it to a whole-plan correction, and
        that single unit opens cycle 002.  The red gate of that very cycle is
        then refused the re-decomposition rung it would otherwise be admitted:
        one budget bounds both, so no third cycle and no planner call exists.
        """

        counter = self.root / "gate-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "sys.exit(0 if count == 0 else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        self.workers.on(ExecutionRole.REPAIR, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(check_repair=1, correction_cycles=1),
            planner=[
                initial_plan(STEP), correction_plan(("S01", "other.txt", "Correct other")),
            ],
            reviewer=[review("REVISE", "REPLAN")],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(
            json.loads((self.run_dir() / "cycles/002/cycle.json").read_text())["kind"],
            "review-replan",
        )
        # The review's own correction cycle is refused the rung that would
        # spend one more unit: nothing was re-decomposed and nothing planned.
        self.assertNotIn(
            "replan_cycle", ladder_strategies(self, cycle=2, stage="post-review-replan"),
        )
        self.assertFalse((self.run_dir() / "cycles/003").exists())
        self.assertEqual(
            [request for request in self.planner.requests if "re-decomposition" in request], [],
        )


class SemanticRevisionTests(PipelineHarness):
    def test_exhausted_semantic_infrastructure_still_reaches_gate_and_reviewer(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, timeout, timeout, timeout)
        result = self.orchestrator(
            self.config(semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser", "reviser"])
        self.assertTrue((self.run_dir() / "cycles/001/checks/post-semantic-revision/evidence.json").is_file())
        self.assertIn("SEMANTIC REVISION: UNAVAILABLE", self.reviewer.requests[0])
        self.assertIn("reason=AGENT_TIMEOUT", self.reviewer.requests[0])
        self.assertEqual(
            json.loads((self.run_dir() / "cycles/001/semantic-revision/status.json").read_text())[
                "status"],
            "UNAVAILABLE",
        )

    def test_semantic_worker_scope_violation_is_a_hard_stop(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, write("other.txt", "unsafe\n"))
        result = self.orchestrator(
            self.config(semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])
        self.assertEqual(self.reviewer.requests, [])

    def test_semantic_scope_request_within_auto_bound_is_authorized_and_retried(self) -> None:
        request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe correction depends on the related file.\n\n"
            "PATHS\n- other.txt\n\n"
            "EVIDENCE\n- The related file supplies required context.\n\n"
            "END META SCOPE REQUEST"
        )

        def use_added_scope(agent_request):
            self.assertIn("other.txt", agent_request.mutable_paths)
            (agent_request.worktree / "other.txt").write_text("authorized\n", encoding="utf-8")
            return "updated within requested scope\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, lambda _request: request, use_added_scope)
        config = self.config(semantic_revision=True)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])
        authority = json.loads((
            self.run_dir() / "cycles/001/semantic-revision/scope_requests/001/authority.json"
        ).read_text())
        self.assertEqual(authority["added_paths"], ["other.txt"])
        self.assertEqual(git(self.worktree(), "show", "HEAD:other.txt").strip(), "authorized")

    def test_semantic_scope_request_waits_for_approval_then_resumes(self) -> None:
        from metaharness.web.api import approve_repair_scope

        request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe correction depends on the related file.\n\n"
            "PATHS\n- other.txt\n\n"
            "EVIDENCE\n- The related file supplies required context.\n\n"
            "END META SCOPE REQUEST"
        )

        def use_added_scope(agent_request):
            self.assertIn("other.txt", agent_request.mutable_paths)
            (agent_request.worktree / "other.txt").write_text("approved\n", encoding="utf-8")
            return "updated within approved scope\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, lambda _request: request, use_added_scope)
        config = self.config(semantic_revision=True)
        options = RunOptions.from_config(
            config, repair_scope_policy="require-approval", repair_scope_max_added_paths=1,
        )
        waiting = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(waiting.status, RunStatus.WAITING_SCOPE_APPROVAL)
        approval_artifact = self.state()["scope_delta"]["approval_artifact"]
        self.assertTrue((self.run_dir() / approval_artifact).is_file())
        approve_repair_scope(self.root / "runs", "run", "APPROVE")

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])
        self.assertEqual(git(self.worktree(), "show", "HEAD:other.txt").strip(), "approved")

    def test_denied_semantic_scope_expansion_routes_to_replan(self) -> None:
        request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe correction depends on the related file.\n\n"
            "PATHS\n- other.txt\n\n"
            "EVIDENCE\n- The related file supplies required context.\n\n"
            "END META SCOPE REQUEST"
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            lambda _request: request,
            lambda _request: "no additional correction needed\n",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n\n"))
        config = self.config(semantic_revision=True, correction_cycles=1)
        options = RunOptions.from_config(config, repair_scope_policy="deny-expansion")
        result = self.orchestrator(
            config,
            planner=[initial_plan(STEP), correction_plan(STEP)],
            reviewer=[review("REVISE", "REPLAN"), review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.state()["cycle"], 2)
        self.assertIn("SEMANTIC REVISION: SCOPE EXPANSION DENIED\nroute=REPLAN", self.reviewer.requests[0])
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "implementer", "reviser"])

    def test_gate_before_revision_has_the_exact_target_order(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'good', 'good semantic'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good semantic\n"))
        result = self.orchestrator(
            self.config(check_repair=2, semantic_revision=True), planner=[initial_plan(STEP)],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        filtered = [
            (event["event"], event["data"].get("stage"))
            for event in self.trace_events()
            if event["event"] in {
                "step.agent.completed", "checks.started", "checks.completed",
                "check_repair.started", "revision.started", "review.started",
                "review.completed", "publish.completed",
            }
        ]
        self.assertEqual(filtered, [
            ("step.agent.completed", None),
            ("checks.started", "POST_IMPLEMENTATION"),
            ("checks.completed", "POST_IMPLEMENTATION"),
            ("check_repair.started", None),
            ("checks.started", "POST_IMPLEMENTATION"),
            ("checks.completed", "POST_IMPLEMENTATION"),
            ("check_repair.started", None),
            ("checks.started", "POST_IMPLEMENTATION"),
            ("checks.completed", "POST_IMPLEMENTATION"),
            ("revision.started", None),
            ("checks.started", "POST_SEMANTIC_REVISION"),
            ("checks.completed", "POST_SEMANTIC_REVISION"),
            ("review.started", None),
            ("review.completed", None),
            ("publish.completed", None),
        ])
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair", "reviser"])

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


class ScopeApprovalTests(PipelineHarness):
    def test_a_correction_scope_expansion_waits_for_approval_then_resumes(self) -> None:
        from metaharness.run_options import RunOptions
        from metaharness.web.api import approve_repair_scope

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        config = self.config(correction_cycles=1)
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

if __name__ == "__main__":
    unittest.main()
