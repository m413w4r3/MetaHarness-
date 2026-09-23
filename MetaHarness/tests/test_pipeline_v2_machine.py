"""End-to-end behavior of the generic pipeline-v2 state machine."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metaharness.llm.chat import LLMError
from metaharness.config import load_config
from metaharness.gitops import GitError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.run_options import RunOptions
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
    def test_publish_disabled_still_pushes_exact_candidate_before_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(candidate["remote"], "origin")
        self.assertEqual(candidate["remote_branch"], self.state()["branch"])
        self.assertEqual(candidate["remote_sha"], candidate["commit_sha"])
        self.assertTrue(candidate["pushed_at"])
        self.assertEqual(self.remote_tip(self.state()["branch"]), candidate["commit_sha"])
        self.assertIn(candidate["commit_sha"], self.reviewer.requests[0])
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_candidate_push_failure_never_calls_reviewer(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.push_run_branch",
            side_effect=GitError("simulated push failure"),
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "PUSH_FAILED")
        self.assertEqual(self.reviewer.requests, [])

    def test_different_remote_tip_fails_closed_before_reviewer(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            return_value="d" * 40,
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REMOTE_AUTHORITY_MISMATCH")
        self.assertEqual(self.reviewer.requests, [])

    def test_remote_push_without_web_url_uses_bounded_diff_fallback(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        request = self.reviewer.requests[0]
        self.assertIn('"remote_exploration": "UNAVAILABLE"', request)
        self.assertIn("<BOUNDED DIFF EXCERPT>", request)

    def test_pipeline_v2_stays_backend_neutral(self) -> None:
        source = Path(__file__).parents[1].joinpath(
            "src", "metaharness", "orchestration", "pipeline_v2.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("CodexAgent", "ClaudeCodeAgent", "C01", "C02"):
            self.assertNotIn(forbidden, source)

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
            {"schema_version", "planner", "steps", "check_repair", "semantic_reviser", "final_reviewer"},
        )
        for forbidden in ("reviser", "repair_implementer", "reviewer", "implementer"):
            self.assertNotIn(forbidden, selection)
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

    def test_human_route_stops_without_automatic_correction_or_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=3), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "HUMAN")],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "HUMAN_REQUIRED")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertFalse((self.run_dir() / "publish.json").exists())

    def test_reviewer_fail_stops_without_correction_or_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=3), planner=[initial_plan(STEP)],
            reviewer=[review("FAIL", "HUMAN")],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_FAILED")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertFalse((self.run_dir() / "publish.json").exists())


class CheckRepairTests(PipelineHarness):
    def _add_tracked_paths(self, *paths: str) -> None:
        for path in paths:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base test\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add test fixtures")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")

    def test_repair_scope_is_used_by_commit_gate_and_candidate(self) -> None:
        self._add_tracked_paths("tests/test_feature.py")
        self.check.write_text(
            "import pathlib, sys\n"
            "if pathlib.Path('feature.txt').read_text().strip() != 'good':\n"
            "    print('tests/test_feature.py: expected fixture update', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda request: (
                (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8"),
                (request.worktree / "tests/test_feature.py").write_text(
                    "repaired test\n", encoding="utf-8",
                ),
                "repaired\n",
            )[-1],
        )
        config = self.config(check_repair=2)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        accepted = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").read_text()
        )
        self.assertEqual(accepted["mutable_scope"], ["feature.txt", "tests/test_feature.py"])
        self.assertRegex(accepted["mutable_scope_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("tests/test_feature.py", git(self.worktree(), "show", "--format=", "--name-only", "HEAD"))

    def test_repair_scope_rejects_paths_over_the_auto_bound(self) -> None:
        self._add_tracked_paths("tests/test_feature.py", "tests/test_other.py")
        self.check.write_text(
            "import pathlib, sys\n"
            "if not (pathlib.Path('feature.txt').read_text().strip() == 'good'\n"
            "        and pathlib.Path('tests/test_feature.py').read_text().strip() == 'repaired'\n"
            "        and pathlib.Path('tests/test_other.py').read_text().strip() == 'repaired'):\n"
            "    print('tests/test_feature.py tests/test_other.py', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda request: (
                (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8"),
                (request.worktree / "tests/test_feature.py").write_text(
                    "repaired\n", encoding="utf-8",
                ),
                (request.worktree / "tests/test_other.py").write_text(
                    "repaired\n", encoding="utf-8",
                ),
                "repaired\n",
            )[-1],
        )
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVISION_SCOPE_VIOLATION")
        self.assertFalse(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )

    def test_noop_repair_attempt_is_existing_head_not_an_empty_repair_commit(self) -> None:
        counter = self.root / "flaky-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    print('tests/test_feature.py: flaky failure', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, lambda _request: "no change\n")
        result = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        accepted = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").read_text()
        )
        self.assertEqual(accepted["acceptance_kind"], "existing-head")
        self.assertFalse(accepted["commit_created"])
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")

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
        self.assertNotIn("repair", self.workers.roles())
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

    def test_mixed_review_routes_spend_only_the_route_specific_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.REVISER,
            write("feature.txt", "good\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("other.txt", "second\n"))
        result = self.orchestrator(
            self.config(review_repair=3),
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
    def test_gate_before_revision_has_the_exact_target_order(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'good', 'good semantic'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
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
    def _interrupt_before_replan_planner(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "REPLAN")],
        )
        with mock.patch.object(
            type(original), "_plan_correction",
            side_effect=AssertionError("planner must not run in this setup"),
        ):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.checkpoint()["phase"], "review_replan")
        self.assertEqual(self.checkpoint()["review_cycle"], 2)

    def test_crash_after_worker_resumes_checks_without_replaying_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        initial_planner = self.planner

        def crash(_self, *_args):
            raise RuntimeError("crash after worker before checks")

        with mock.patch.object(type(original), "_run_gate", side_effect=crash):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.checkpoint()["phase"], "deterministic_gate")
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

    def test_push_then_crash_resumes_review_with_the_same_remote_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        def crash_after_push(owner, *_args, **_kwargs):
            raise RuntimeError("crash after candidate push")

        with mock.patch.object(type(original), "_review_candidate", side_effect=crash_after_push):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text(encoding="utf-8")
        )
        branch = self.state()["branch"]
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])

        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"], reviewer=[review()],
        )
        with mock.patch(
            "metaharness.orchestrator.push_run_branch",
            side_effect=AssertionError("resume must not repush an exact remote SHA"),
        ):
            resumed = resumed.resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.remote_tip(branch), candidate["commit_sha"])
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)

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
        calls = 0
        real_review = type(original)._review_candidate

        def crash_on_cycle_four(owner, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("crash at cycle four review")
            return real_review(original, owner, *args, **kwargs)

        with mock.patch.object(type(original), "_review_candidate", side_effect=crash_on_cycle_four):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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

        def crash_once(_self, *_args):
            raise RuntimeError("crash before first checks")

        with mock.patch.object(type(original), "_run_gate", side_effect=crash_once):
            failed = original.run_text(SPEC, run_id="run", on_created=change_live_defaults)
        self.assertEqual(failed.status, RunStatus.FAILED)

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


class GitChainAndTraceTests(PipelineHarness):
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
    def test_green_evidence_without_acceptance_is_reused_on_resume(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with mock.patch(
            "metaharness.orchestration.check_repair.GateAcceptanceService.accept",
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

    def test_tampered_accepted_scope_is_rejected_before_resume_agents(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with mock.patch(
            "metaharness.orchestration.candidate.CandidateLifecycle.create",
            side_effect=RuntimeError("crash after acceptance"),
        ):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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


class ResumeAuthorityTests(PipelineHarness):
    """Legitimate durable states resume; tampered authority fails closed."""

    def _crash_at(self, orchestrator, phase):
        from metaharness.resume import ResumePhase

        real = type(orchestrator)._write_checkpoint
        fired: list[bool] = []

        def write(run_dir, next_phase, **fields):
            if next_phase is ResumePhase(phase) and not fired:
                fired.append(True)
                raise RuntimeError(f"crash before {phase}")
            return real(run_dir, next_phase, **fields)

        return mock.patch.object(type(orchestrator), "_write_checkpoint", staticmethod(write))

    def _crash_on_review(self, orchestrator, number):
        calls = 0
        real = type(orchestrator)._review_candidate

        def review_or_crash(owner, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == number:
                raise RuntimeError("crash at final review")
            return real(orchestrator, owner, *args, **kwargs)

        return mock.patch.object(type(orchestrator), "_review_candidate", side_effect=review_or_crash)

    def test_changed_direct_correction_resumes_at_its_final_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        # A real correction: the accepted tree differs from the reviewed one.
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        original = self.orchestrator(
            self.config(review_repair=1), planner=[initial_plan(STEP)],
            reviewer=[review("REVISE", "IMPLEMENTATION")],
        )
        with self._crash_on_review(original, 2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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
        with self._crash_at(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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
        with self._crash_at(original, "candidate_ready"):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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
        from metaharness.agent import AgentError

        def crash(_request):
            raise AgentError("reviser crashed")

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REVISER, crash)
        failed = self.orchestrator(
            self.config(semantic_revision=True), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(self.checkpoint()["phase"], "semantic_revision")

    def test_semantic_revision_resumes_after_its_green_gate(self) -> None:
        self._crash_in_semantic_revision()
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        resumed = self.orchestrator(
            self.config(semantic_revision=True), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "reviser", "reviser"])

    def test_semantic_revision_never_resumes_without_its_green_gate_acceptance(self) -> None:
        self._crash_in_semantic_revision()
        (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").unlink()
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n\n"))
        resumed = self.orchestrator(
            self.config(semantic_revision=True), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        # The crashed reviser call is the only one: no model call on resume.
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

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
        with self._crash_on_review(original, 3):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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
        with self._crash_on_review(original, 2):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
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

    def _crash_in_the_second_check_repair_attempt(self):
        from metaharness.agent import AgentError

        def crash(_request):  # fails before touching the worktree
            raise AgentError("repair worker crashed")

        # Attempt 001 leaves the gate red; attempt 002 crashes cleanly.
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "still bad\n"), crash)
        config = self.config(check_repair=2)
        failed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED)
        path = self.run_dir() / "resume_checkpoint.json"
        checkpoint = json.loads(path.read_text())
        self.assertEqual((checkpoint["phase"], checkpoint["check_repair_attempt"]), ("check_repair", 2))
        return config, path, checkpoint

    def test_a_lowered_check_repair_counter_never_replays_a_recorded_attempt(self) -> None:
        config, path, checkpoint = self._crash_in_the_second_check_repair_attempt()
        attempt_one = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        record = (attempt_one / "attempt.json").read_text()
        checkpoint["check_repair_attempt"] = 1
        path.write_text(json.dumps(checkpoint), encoding="utf-8")

        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # Only the unfinished attempt 002 is retried; 001 is never replayed.
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair", "repair"])
        self.assertEqual((attempt_one / "attempt.json").read_text(), record)
        self.assertEqual(
            sorted(item.name for item in attempt_one.parent.iterdir()), ["001", "002"],
        )

    def test_a_check_repair_counter_beyond_the_durable_attempts_fails_closed(self) -> None:
        config, path, checkpoint = self._crash_in_the_second_check_repair_attempt()
        checkpoint["check_repair_attempt"] = 3
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        resumed = self.orchestrator(config, planner=["unused"], reviewer=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])

    def test_run_options_without_their_state_hash_are_never_resumed(self) -> None:
        from metaharness.resume import ResumeNotAllowedError
        from metaharness.state import RunStateStore

        self._crash_in_semantic_revision()
        options = self.run_dir() / "run_options.json"
        payload = json.loads(options.read_text())
        payload["max_check_repair_attempts"] = 9
        options.write_text(json.dumps(payload), encoding="utf-8")
        store = RunStateStore(self.run_dir() / "state.json")
        store.update(status=store.load()["status"], run_options_sha256=None)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(
                self.config(semantic_revision=True), planner=["unused"], reviewer=["unused"],
            ).resume("run")
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

class ObservationAndPublicationAuthorityTests(PipelineHarness):
    def test_a_failing_external_observer_cannot_change_the_run(self) -> None:
        from metaharness.orchestrator import Orchestrator
        from tests.pipeline_support import ScriptedChat

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

    def test_publication_requires_a_durable_pass_for_the_exact_candidate(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        orchestrator = self.orchestrator(
            self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        with mock.patch("metaharness.orchestrator._accepted_review", return_value=None):
            result = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_AUTHORITY_MISSING")
        self.assertFalse((self.run_dir() / "publish.json").exists())
        self.assertEqual(len(self.reviewer.requests), 1)


if __name__ == "__main__":
    unittest.main()
