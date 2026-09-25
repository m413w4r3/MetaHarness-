"""End-to-end behavior of the generic pipeline-v2 state machine."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metaharness.llm.chat import LLMError, LLMHTTPError
from metaharness.config import load_config
from metaharness.agent.protocol import (
    CONTRACT_MISMATCH_HEADER,
    CheckRepairResult,
    parse_check_repair_result,
)
from metaharness.gitops import (
    GitError,
    candidate_tree_sha,
    remote_run_branch_tip as real_remote_run_branch_tip,
)
from metaharness.models import ExecutionRole, RunStatus
from metaharness.recovery_policy import ExecutionFallbacks, RecoveryBudgets
from metaharness.orchestration.pipeline_v2 import PipelineFailure
from metaharness.run_options import RunOptions
from metaharness.resume import resume_info
from tests.pipeline_support import (
    PipelineHarness,
    check_repair_result,
    correction_plan,
    git,
    initial_plan,
    review,
    write,
)

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")


class CheckRepairProtocolTests(unittest.TestCase):
    def test_check_repair_result_parser_requires_one_final_strict_block(self) -> None:
        valid = check_repair_result("DONE", "FAIL", "NONE", "targeted test ran and failed")
        self.assertEqual(
            parse_check_repair_result("worker summary\n\n" + valid),
            CheckRepairResult("DONE", "FAIL", "NONE", "targeted test ran and failed"),
        )
        blocked = check_repair_result(
            "BLOCKED", "NOT_RUN", "INFRASTRUCTURE", "Docker daemon unavailable",
        )
        self.assertIsNotNone(parse_check_repair_result(blocked))
        for invalid in (
            valid + valid,
            valid + "extra text\n",
            valid.replace("FAIL\n", "MAYBE\n"),
            valid.replace("BLOCKED_KIND\nNONE", "BLOCKED_KIND\nINFRASTRUCTURE"),
            blocked.replace("INFRASTRUCTURE", "NONE"),
            valid.replace("NOTE\ntargeted test ran and failed", "NOTE\n"),
        ):
            with self.subTest(invalid=invalid[:80]):
                self.assertIsNone(parse_check_repair_result(invalid))


def repaired_step_contract() -> str:
    return """META STEP CONTRACT REPAIR v1
STEP_ID: S01
TITLE: Write the feature
EXECUTION_CLASS: MECHANICAL
DEPENDS_ON: NONE

OBJECTIVE
Write feature.txt with the SPEC-required content.

READ_SET
- feature.txt :: current content

WRITE_SET
- feature.txt

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. Set the file to the required content.

VERIFY
- Run the configured test.

FORBIDDEN
- Do not edit paths outside the approved set.

END META STEP CONTRACT REPAIR
"""


class SingleCycleTests(PipelineHarness):
    def test_unknown_pipeline_failure_stops_without_recovery_model_calls(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.PipelineV2Coordinator.run",
            side_effect=PipelineFailure("TOTALLY_NEW_FAILURE", "retained diagnostic"),
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"], {
            "reason": "TOTALLY_NEW_FAILURE", "detail": "retained diagnostic",
        })
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
        self.assertEqual(len(self.planner.requests), 1)  # initial planning only
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.workers.calls, [])

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

    def test_optional_candidate_push_failure_uses_local_review_and_commits(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            side_effect=GitError("simulated network outage"),
        ):
            result = self.orchestrator(
                self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 1)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")
        request = self.reviewer.requests[0]
        self.assertIn('"remote_exploration": "UNAVAILABLE"', request)
        self.assertIn("<BOUNDED DIFF EXCERPT>", request)
        self.assertIn("candidate.push_unavailable", self.trace_names())

    def test_different_remote_tip_does_not_block_local_candidate_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            return_value="d" * 40,
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.reviewer.requests), 1)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertEqual(candidate["remote_status"], "unavailable")

    def test_remote_push_without_web_url_uses_bounded_diff_fallback(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(publish=False), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        request = self.reviewer.requests[0]
        self.assertIn('"remote_exploration": "UNAVAILABLE"', request)
        self.assertIn("<BOUNDED DIFF EXCERPT>", request)

    def test_required_candidate_push_waits_and_resumes_at_candidate_push(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            side_effect=GitError("simulated network outage"),
        ):
            waiting = self.orchestrator(
                self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.reviewer.requests, [])
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
        candidate = json.loads(
            (self.run_dir() / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertIsNone(candidate["remote_sha"])
        self.assertIsNone(candidate["pushed_at"])
        self.assertEqual(candidate["remote_status"], "unavailable")

        resumed = self.orchestrator(
            self.config(publish=True), planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 1)

    def test_contract_mismatch_uses_planner_repair_not_blind_retry(self) -> None:
        observed_before_retry: list[str] = []

        def mismatch_after_edit(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            return CONTRACT_MISMATCH_HEADER + "\nThe Verify anchor cannot run in this worker."

        def write_after_repair(request):
            observed_before_retry.append(
                (request.worktree / "feature.txt").read_text(encoding="utf-8")
            )
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "done\n"

        self.workers.on(
            ExecutionRole.IMPLEMENTER, mismatch_after_edit, write_after_repair,
        )
        result = self.orchestrator(
            self.config(max_step_contract_repairs=1),
            planner=[initial_plan(STEP), repaired_step_contract()],
            reviewer=[review()],
        ).run_text(SPEC, run_id="contract-repair")

        self.assertEqual(
            result.status, RunStatus.COMMITTED,
            self.state("contract-repair").get("failure"),
        )
        self.assertEqual(observed_before_retry, ["base\n"])
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(len(self.planner.requests), 2)
        repair_request = self.planner.requests[1]
        self.assertIn(SPEC.strip(), repair_request)
        self.assertIn("Verify anchor cannot run", repair_request)
        retry_prompt = self.workers.calls[1].prompt
        self.assertNotIn("CONTRACT REPAIR REQUIRED", retry_prompt)
        self.assertNotIn("PREVIOUS ATTEMPT MISMATCH REPORT", retry_prompt)
        repair_dir = (
            self.run_dir("contract-repair")
            / "cycles/001/implementation/steps/S01/contract_repairs/01"
        )
        validation = json.loads((repair_dir / "validation.json").read_text())
        self.assertEqual(
            validation["current_tree_sha"], git(self.repo, "rev-parse", "HEAD^{tree}"),
        )

    def test_residual_mismatch_edits_roll_back_to_the_exact_pre_step_tree(self) -> None:
        def mismatch_with(content: str):
            def action(request):
                (request.worktree / "feature.txt").write_text(content, encoding="utf-8")
                return CONTRACT_MISMATCH_HEADER + "\nThe Verify anchor cannot run in this worker."
            return action

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch_with("first partial\n"),
            mismatch_with("residual partial\n"),
        )
        result = self.orchestrator(
            self.config(max_step_contract_repairs=1),
            planner=[initial_plan(STEP), repaired_step_contract()],
            reviewer=[review()],
        ).run_text(SPEC, run_id="residual-mismatch")

        worktree = self.worktree("residual-mismatch")
        # Exhausted contract repair is a correctness decision for an operator.
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(
            self.state("residual-mismatch")["failure"]["reason"],
            "AGENT_CONTRACT_MISMATCH",
        )
        self.assertEqual(git(worktree, "rev-parse", "HEAD"), self.base_sha)
        self.assertEqual(git(worktree, "status", "--porcelain"), "")
        self.assertEqual((worktree / "feature.txt").read_text(), "base\n")
        self.assertEqual(len(self.workers.calls), 2)

    def test_environment_verify_failure_is_not_contract_mismatch(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write(
                "feature.txt", "good\n",
                "VERIFY: FAIL (environment: missing optional tool)\n",
            ),
        )
        result = self.orchestrator(
            self.config(max_step_contract_repairs=1),
            planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="environment-verify")

        self.assertEqual(
            result.status, RunStatus.COMMITTED,
            self.state("environment-verify").get("failure"),
        )
        self.assertEqual(len(self.workers.calls), 1)
        self.assertEqual(len(self.planner.requests), 1)
        steps_dir = (
            self.run_dir("environment-verify")
            / "cycles/001/implementation/steps/S01"
        )
        self.assertFalse((steps_dir / "contract_repairs").exists())
        self.assertNotIn("AGENT_CONTRACT_MISMATCH", self.trace_names("environment-verify"))

    def test_no_change_after_contract_repair_is_reviewed_before_implementation_correction(self) -> None:
        # The required check passes on the unchanged tree and on the corrected
        # tree; the independent reviewer decides that the SPEC still needs work.
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'base', 'good'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, lambda _request: "done\n", lambda _request: "done\n")
        self.workers.on(ExecutionRole.REVISER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(max_step_contract_repairs=1, review_repair=1),
            planner=[initial_plan(STEP), repaired_step_contract()],
            reviewer=[review("REVISE", "IMPLEMENTATION"), review()],
        ).run_text(SPEC, run_id="reviewed-no-change")

        self.assertEqual(
            result.status, RunStatus.COMMITTED,
            self.state("reviewed-no-change").get("failure"),
        )
        first_candidate = json.loads(
            (self.run_dir("reviewed-no-change") / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertTrue(first_candidate["no_change"])
        self.assertEqual(first_candidate["commit_sha"], self.base_sha)
        self.assertEqual(self.state("reviewed-no-change").get("no_change"), None)
        self.assertEqual(git(self.worktree("reviewed-no-change"), "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(self.workers.roles(), ["implementer", "implementer", "reviser"])
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertIn("candidate delta is empty", self.reviewer.requests[0])

    def test_pr_creation_requires_remote_candidate_before_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            side_effect=GitError("simulated network outage"),
        ):
            waiting = self.orchestrator(
                self.config(publish=True, github_pr=True),
                planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status.value, "waiting_remote")
        self.assertEqual(self.checkpoint()["phase"], "candidate_push")
        self.assertEqual(self.reviewer.requests, [])

    def test_publication_rejects_remote_tip_change_after_local_review_pass(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        calls = 0

        def moved_after_review(repo, *, remote, branch):
            nonlocal calls
            calls += 1
            actual = real_remote_run_branch_tip(repo, remote=remote, branch=branch)
            return "d" * 40 if calls >= 4 else actual

        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            side_effect=moved_after_review,
        ):
            result = self.orchestrator(
                self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "COMMIT_TREE_MISMATCH")
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertGreaterEqual(calls, 4)

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
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(self.reviewer.requests, [])
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

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

    def test_reviewer_fail_stops_without_correction_or_publication(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(review_repair=3), planner=[initial_plan(STEP)],
            reviewer=[review("FAIL", "HUMAN")],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEW_EVIDENCE_UNRESOLVED")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(len(self.reviewer.requests), 2)
        self.assertEqual(self.checkpoint()["phase"], "final_review")
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)
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
            self.config(review_repair=1),
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

        orchestrator = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        reviewer = CorruptingReviewer()
        orchestrator._reviewer_client = reviewer
        result = orchestrator.run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(len(reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer"])


class CheckRepairTests(PipelineHarness):
    def _add_tracked_paths(self, *paths: str) -> None:
        for path in paths:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base test\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add test fixtures")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")

    @staticmethod
    def _plan_with_approved_paths(*paths: str) -> str:
        plan_text = initial_plan(STEP)
        reads = "".join(f"- {path} :: current content\n" for path in paths)
        writes = "".join(f"- {path}\n" for path in paths)
        return plan_text.replace(
            "- feature.txt :: current content\n",
            "- feature.txt :: current content\n" + reads,
            1,
        ).replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n" + writes, 1)

    def test_failed_test_file_is_readable_but_not_auto_writable(self) -> None:
        self._add_tracked_paths("tests/test_feature.py")
        self.check.write_text(
            "import pathlib, sys\n"
            "if pathlib.Path('feature.txt').read_text().strip() != 'good':\n"
            "    print('feature.txt: required value; tests/test_feature.py: expected fixture update', file=sys.stderr)\n"
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
                check_repair_result(),
            )[-1],
        )
        config = self.config(check_repair=2)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        repair_request = self.workers.calls[-1]
        self.assertEqual(repair_request.mutable_paths, ("feature.txt",))
        self.assertIn("tests/test_feature.py", repair_request.prompt)
        self.assertIn("remain read-only", repair_request.prompt)
        self.assertFalse(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )

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
                check_repair_result(),
            )[-1],
        )
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
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
        self.workers.on(ExecutionRole.REPAIR, lambda _request: check_repair_result())
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

    def test_every_attempt_of_the_budget_is_used_then_the_gate_is_exhausted(self) -> None:
        def blocked_after_mutation(request):
            (request.worktree / "feature.txt").write_text("half repair\n", encoding="utf-8")
            return check_repair_result(
                "BLOCKED", "NOT_RUN", "INFRASTRUCTURE", "Docker socket access denied",
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            blocked_after_mutation,
            write("feature.txt", "good\n"),
        )
        config = self.config(check_repair=2)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_UNAVAILABLE")
        self.assertEqual(self.state()["check_repair"]["attempt_count"], 0)
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertFalse((attempt / "attempt.json").exists())
        self.assertEqual((self.worktree() / "feature.txt").read_text(encoding="utf-8"), "bad\n")
        evidence = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/evidence.json").read_text()
        )
        self.assertEqual(candidate_tree_sha(self.worktree()), evidence["staged_tree_sha"])
        report = json.loads((attempt / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["check_repair_result"]["blocked_kind"], "INFRASTRUCTURE")
        self.assertNotEqual(report["tree_before"], report["tree_after"])
        self.assertEqual(
            (self.checkpoint()["phase"], self.checkpoint()["check_repair_attempt"]),
            ("check_repair", 1),
        )

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertTrue((attempt / "attempts/01/report.json").is_file())

    def test_two_verified_check_repairs_can_exhaust_the_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "targeted check ran and failed: regression-redaction-secret",
            )),
            write("feature.txt", "worse\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "targeted check ran and failed: regression-redaction-secret",
            )),
        )
        with mock.patch(
            "metaharness.orchestrator.config_secret_values",
            return_value=("regression-redaction-secret",),
        ):
            result = self.orchestrator(
                self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertIsInstance(self.state()["failure"]["detail"], dict)
        self.assertEqual(self.state()["failure"]["detail"]["failed_check_ids"], ["test"])
        self.assertEqual(self.state()["failure"]["detail"]["attempt_count"], 2)
        self.assertEqual(self.state()["check_repair"]["failure_classification"], "product_check")
        self.assertEqual(self.state()["check_repair"]["next_action"], "Retry deterministic gate")
        self.assertRegex(self.state()["check_repair"]["latest_evidence_sha256"], r"^[0-9a-f]{64}$")
        reports = self.state()["check_repair"]["repair_reports"]
        self.assertEqual([item["attempt"] for item in reports], [1, 2])
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in reports))
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["phase"], "deterministic_gate")
        self.assertEqual(checkpoint["stage"], "POST_IMPLEMENTATION")
        self.assertEqual(checkpoint["check_repair_attempt"], 2)
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        diagnostics_path = self.run_dir() / "diagnostics.md"
        self.assertTrue(
            diagnostics_path.is_file(),
            (self.run_dir() / "diagnostics.error.txt").read_text(encoding="utf-8")
            if (self.run_dir() / "diagnostics.error.txt").is_file() else "diagnostics missing",
        )
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        recovery_summary = diagnostics.split("## DETERMINISTIC GATE RECOVERY", 1)[1].split("\n## ", 1)[0]
        for expected in (
            "deterministic gate attempt: 3",
            "check-repair attempts used / budget: 2 / 2",
            "latest failed check IDs: test",
            "failure classification: product_check",
            "next recovery action: Retry deterministic gate",
        ):
            self.assertIn(expected, recovery_summary, recovery_summary)
        attempts = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts"
        self.assertEqual(sorted(path.name for path in attempts.iterdir()), ["001", "002"])
        self.assertEqual(self.reviewer.requests, [])

    def test_blocked_scope_request_uses_authority_before_retrying_same_attempt(self) -> None:
        from metaharness.run_options import RunOptions

        self._add_tracked_paths("other.txt")
        self.check.write_text(
            "import pathlib, sys\n"
            "if (pathlib.Path('feature.txt').read_text().strip() != 'good'\n"
            "        or pathlib.Path('other.txt').read_text().strip() != 'good'):\n"
            "    print('feature.txt: required files are not repaired', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        scope_request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe failing check requires the related fixture.\n\n"
            "PATHS\n- other.txt\n\n"
            "EVIDENCE\n- The configured test reads other.txt.\n\n"
            "END META SCOPE REQUEST"
        )

        def blocked(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            return scope_request + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "other.txt is outside the current authority",
            )

        def repair(request):
            self.assertIn("other.txt", request.mutable_paths)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            (request.worktree / "other.txt").write_text("good\n", encoding="utf-8")
            return check_repair_result()

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, blocked, repair)
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[self._plan_with_approved_paths("other.txt")], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertEqual(json.loads((attempt / "attempt.json").read_text())["number"], 1)
        authority = json.loads((attempt / "scope_requests/001/authority.json").read_text())
        self.assertEqual(authority["added_paths"], ["other.txt"])
        self.assertEqual(json.loads((attempt / "scope.json").read_text())["added_paths"], ["other.txt"])

    def test_failure_trace_narrows_repair_and_scope_requests_obey_the_addition_limit(self) -> None:
        from metaharness.run_options import RunOptions

        modules = [f"src/repair_{index:02}.py" for index in range(38)]
        approved = ["feature.txt", *modules, "tests/test_failure.py"]
        self.assertEqual(len(approved), 40)
        self._add_tracked_paths(*approved)
        groups = [approved[index:index + 5] for index in range(0, len(approved), 5)]
        plan_text = initial_plan(*(
            (f"S{index:02}", group[0], f"Update {group[0]}")
            for index, group in enumerate(groups, start=1)
        ))
        for index, group in enumerate(groups, start=1):
            primary = group[0]
            read_entries = "".join(f"- {path} :: current content\n" for path in group)
            write_entries = "".join(f"- {path}\n" for path in group)
            plan_text = plan_text.replace(
                f"READ_SET\n- {primary} :: current content\n",
                f"READ_SET\n{read_entries}", 1,
            ).replace(
                f"WRITE_SET\n- {primary}\n",
                f"WRITE_SET\n{write_entries}", 1,
            )
        counter = self.root / "synthetic-check-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    print('=== FAILURES ===')\n"
            "    print('________________ test_gate_failure ________________')\n"
            "    print('Traceback (most recent call last):')\n"
            "    print(f'  File {str(pathlib.Path(\"src/repair_00.py\").resolve())!r}, line 12, in run')\n"
            "    print(f'  File {str(pathlib.Path(\"src/repair_01.py\").resolve())!r}, line 27, in check')\n"
            "    print('AssertionError: synthetic regression')\n"
            "    print('E   AssertionError: synthetic regression')\n"
            "    sys.stdout.write('\\n' * 5000)\n"
            "    print('FAILED tests/test_failure.py::test_gate_failure - AssertionError')\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        def scope_request_for(path: str) -> str:
            return (
                "META SCOPE REQUEST v1\n\n"
                "REASON\nThe failing assertion depends on this approved source file.\n\n"
                f"PATHS\n- {path}\n\n"
                "EVIDENCE\n- The traceback and check output identify the dependency.\n\n"
                "END META SCOPE REQUEST"
            )

        def request_third_path(request):
            self.assertEqual(request.mutable_paths, tuple(modules[:2]))
            self.assertIn("Traceback (most recent call last)", request.prompt)
            self.assertIn("src/repair_00.py", request.prompt)
            self.assertIn("src/repair_01.py", request.prompt)
            self.assertIn("tests/test_failure.py", request.prompt)
            self.assertIn("checks/test.stdout.log", request.prompt)
            return scope_request_for(modules[2]) + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "The related source path needs authority.",
            )

        def request_over_limit(request):
            self.assertEqual(request.mutable_paths, tuple(modules[:3]))
            return scope_request_for(modules[3]) + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "A second path needs authority.",
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, *(
            write(group[0], "good\n") for group in groups
        ))
        self.workers.on(ExecutionRole.REPAIR, request_third_path, request_over_limit)
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[plan_text], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(
            result.status, RunStatus.WAITING_SCOPE_APPROVAL,
            self.state().get("failure"),
        )
        self.assertEqual(self.workers.roles(), ["implementer"] * 8 + ["repair", "repair"])
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        checks = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/checks.json").read_text()
        )
        self.assertEqual(
            checks[0]["stdout_tail"].strip(),
            "FAILED tests/test_failure.py::test_gate_failure - AssertionError",
        )
        scope = json.loads((attempt / "scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["approved_mutable_scope"], sorted(approved))
        self.assertEqual(scope["initial_repair_scope"], sorted(modules[:2]))
        self.assertEqual(scope["added_paths"], [modules[2]])
        self.assertEqual(scope["effective_repair_scope"], sorted(modules[:3]))
        attempts = sorted((attempt / "scope_requests").glob("*/authority.json"))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(json.loads(attempts[1].read_text())["bound"], 1)

    def test_gate_docker_outage_waits_without_consuming_check_repair_budget(self) -> None:
        self.check.write_text(
            "import sys\nprint('Cannot connect to the Docker daemon')\nraise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_CHECK_INFRASTRUCTURE)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_INFRASTRUCTURE_UNAVAILABLE")
        self.assertNotIn("attempt_count", self.state().get("check_repair", {}))
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_operator_retry_reruns_gate_without_replaying_steps_or_repair_workers(self) -> None:
        counter = self.root / "gate-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count < 3:\n"
            "    print('FAILED tests/test_feature.py::test_behavior')\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda _request: check_repair_result(), lambda _request: check_repair_result(),
        )
        config = self.config(check_repair=2)
        original = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        )
        first_planner = original._planner_client
        waiting = original.run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        retry_info = resume_info(self.run_dir(), self.state())
        self.assertTrue(retry_info.resumable, retry_info.reason)
        self.assertEqual(retry_info.label, "Retry deterministic gate (POST_IMPLEMENTATION)")
        roles_before_resume = self.workers.roles()

        resumed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), roles_before_resume)
        self.assertEqual(len(first_planner.requests), 1)
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(counter.read_text(), "4")

    def test_same_red_gate_after_operator_retry_becomes_fixed_point(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "targeted check still fails",
            )),
        )
        config = self.config(check_repair=1)
        waiting = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        self.assertEqual(self.state()["check_repair"]["next_action"], "Retry deterministic gate")

        resumed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).resume("run")

        state = self.state()
        self.assertEqual(resumed.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(state["failure"]["reason"], "CHECK_REPAIR_FIXED_POINT")
        self.assertEqual(state["check_repair"]["status"], "fixed_point")
        self.assertEqual(
            state["check_repair"]["next_action"],
            "Code change or additional repair authority required",
        )
        self.assertEqual(
            state["failure"]["detail"]["operator_message"],
            "Code change or additional repair authority required",
        )
        self.assertFalse(state["recovery_resumable"])
        self.assertFalse(resume_info(self.run_dir(), state).resumable)
        self.assertEqual(self.workers.roles(), ["implementer", "repair"])

    def test_fixed_point_fingerprint_changes_with_tree_or_failed_check_set(self) -> None:
        from metaharness.orchestration.pipeline_v2 import check_repair_fingerprint

        original = check_repair_fingerprint(
            "a" * 40, ["test-integration"], "POST_IMPLEMENTATION",
        )
        self.assertEqual(original, check_repair_fingerprint(
            "a" * 40, ["test-integration"], "POST_IMPLEMENTATION",
        ))
        self.assertNotEqual(original, check_repair_fingerprint(
            "b" * 40, ["test-integration"], "POST_IMPLEMENTATION",
        ))
        self.assertNotEqual(original, check_repair_fingerprint(
            "a" * 40, ["another-check"], "POST_IMPLEMENTATION",
        ))

    def test_historical_attributeerror_shape_resumes_the_same_gate_checkpoint(self) -> None:
        counter = self.root / "legacy-gate-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count < 3:\n"
            "    print('FAILED tests/test_feature.py::test_behavior')\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda _request: check_repair_result(), lambda _request: check_repair_result(),
        )
        config = self.config(check_repair=2)
        waiting = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        checkpoint_before = self.checkpoint()

        # Present the exact durable state written by the historical bug.
        state = self.state()
        state["status"] = "failed"
        state["failure"] = {
            "reason": "ATTRIBUTEERROR",
            "detail": "'dict' object has no attribute 'replace'",
        }
        state.pop("recovery_resumable", None)
        for key in (
            "candidate_tree", "failed_check_ids", "failure_classification",
            "latest_evidence_sha256", "next_action", "repair_reports",
        ):
            state["check_repair"].pop(key, None)
        state["check_repair"]["status"] = "completed"
        (self.run_dir() / "state.json").write_text(json.dumps(state), encoding="utf-8")

        eligible = resume_info(self.run_dir(), state)
        self.assertTrue(eligible.resumable, eligible.reason)
        self.assertEqual(eligible.phase, "deterministic_gate")
        self.assertEqual(eligible.label, "Retry deterministic gate (POST_IMPLEMENTATION)")
        self.assertEqual(self.checkpoint(), checkpoint_before)
        roles_before_resume = self.workers.roles()
        resumed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), roles_before_resume)
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(counter.read_text(), "4")
        self.assertEqual(self.state()["resume"]["migration"], "historical_check_repair_redaction_crash")

    def test_corrupt_latest_gate_evidence_fails_resume_integrity(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda _request: check_repair_result(), lambda _request: check_repair_result(),
        )
        config = self.config(check_repair=2)
        waiting = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        evidence_path = self.run_dir() / "cycles/001/checks/post-implementation/evidence.json"
        evidence_path.write_text(evidence_path.read_text() + " ", encoding="utf-8")
        roles_before_resume = self.workers.roles()

        resumed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).resume("run")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.workers.roles(), roles_before_resume)


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
        config = self.config(semantic_revision=True, review_repair=1)
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
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
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
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "INTERNAL_HARNESS_ERROR")
        self.assertEqual(failure["detail"]["exception_type"], "RuntimeError")
        self.assertEqual(failure["detail"]["phase"], "deterministic_gate")
        self.assertEqual(failure["detail"]["operation"], "pipeline_coordinator")
        self.assertNotIn("Traceback", json.dumps(failure))
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)
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
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
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

        def crash_once(_self, *_args):
            raise RuntimeError("crash before first checks")

        with mock.patch.object(type(original), "_run_gate", side_effect=crash_once):
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
            side_effect=PipelineFailure("CHECK_INFRASTRUCTURE_UNAVAILABLE", "acceptance service unavailable"),
        ):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_CHECK_INFRASTRUCTURE)
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

    def test_integrity_failure_does_not_start_repair_or_review(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        original = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        real_gate = original._run_gate

        def unexpected_head(store, ctx, cycle_plan, stage):
            evidence = real_gate(store, ctx, cycle_plan, stage)
            return replace(
                evidence, deterministic_passed=False,
                failures=("UNEXPECTED_HEAD: moved",),
            )

        with mock.patch.object(type(original), "_run_gate", side_effect=unexpected_head):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.FAILED, self.state().get("failure"))
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
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.state()["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["review_cycle"]), ("final_review", 2))

        resumed = self.orchestrator(config, planner=["unused"], reviewer=[review()]).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.reviewer.requests), 1)
        self.assertEqual(self.workers.roles(), ["implementer", "reviser"])

    def test_a_dirty_check_repair_timeout_rolls_back_and_retries_exact_contract(self) -> None:
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
        completed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(completed.status, RunStatus.COMMITTED, self.state().get("failure"))
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertTrue((attempt / "attempts/01/failure.json").is_file())
        self.assertFalse((attempt / "failure.json").exists())
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])

    def test_check_repair_infrastructure_exhaustion_resumes_at_the_red_gate(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, timeout, write("feature.txt", "good\n"))
        config = self.config(check_repair=1)
        options = RunOptions.from_config(
            config,
            recovery=RecoveryBudgets(max_transient_attempts=0, max_executor_fallbacks=0),
        )
        failed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_UNAVAILABLE")
        self.assertEqual(
            (self.checkpoint()["phase"], self.checkpoint()["check_repair_attempt"]),
            ("check_repair", 1),
        )
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
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

    def test_clean_implementer_timeout_retries_the_exact_effective_contract(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, timeout, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        first, second = self.workers.calls
        self.assertEqual(first.contract, second.contract)
        self.assertEqual(first.prompt, second.prompt)
        self.assertEqual(first.mutable_paths, second.mutable_paths)
        self.assertEqual(first.profile_id, second.profile_id)

    def test_out_of_scope_timeout_is_a_hard_stop_without_retry(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            (request.worktree / "other.txt").write_text("unsafe\n", encoding="utf-8")
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, timeout, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_in_scope_timeout_without_exact_rollback_requires_operator(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def dirty_timeout(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(
            ExecutionRole.IMPLEMENTER, dirty_timeout, write("feature.txt", "good\n"),
        )
        original = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        )
        # The shared attempt transaction cannot restore the partial edit.
        with mock.patch(
            "metaharness.attempt_transaction.restore_paths_from_tree", lambda *_args: None,
        ):
            result = original.run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_REQUIRES_OPERATOR")
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_frozen_executor_fallback_keeps_the_same_contract_and_scope(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, timeout, write("feature.txt", "good\n"))
        config = self.config()
        config = replace(config, recovery=RecoveryBudgets(
            max_transient_attempts=0,
            max_executor_fallbacks=1,
            execution_fallbacks=ExecutionFallbacks(mechanical=("live_worker",)),
        ))
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        primary, fallback = self.workers.calls
        self.assertEqual((primary.profile_id, fallback.profile_id), ("worker", "live_worker"))
        self.assertEqual(primary.contract, fallback.contract)
        self.assertEqual(primary.prompt, fallback.prompt)
        self.assertEqual(primary.mutable_paths, fallback.mutable_paths)
        self.assertEqual(
            json.loads((self.run_dir() / "execution_selection.json").read_text())[
                "steps"][0]["fallbacks"][0]["profile_id"],
            "live_worker",
        )

    def test_auth_failure_is_resumable_and_never_tries_the_fallback(self) -> None:
        from metaharness.agent import AGENT_AUTH_FAILURE, AgentError

        class AuthFailure(AgentError):
            code = AGENT_AUTH_FAILURE

        def auth_failure(_request):
            raise AuthFailure("missing credentials")

        self.workers.on(ExecutionRole.IMPLEMENTER, auth_failure, write("feature.txt", "good\n"))
        config = self.config()
        config = replace(config, recovery=RecoveryBudgets(
            execution_fallbacks=ExecutionFallbacks(mechanical=("live_worker",)),
        ))
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.state()["failure"]["reason"], "EXTERNAL_AUTH_REQUIRED")
        self.assertEqual(self.workers.roles(), ["implementer"])


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
        with self._crash_on_review(original, 1):
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
        with self._crash_on_review(original, 1):
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
        with self._crash_on_review(original, 2):
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
        with self._crash_at(original, "candidate_ready"):
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
        with self._crash_at(original, "candidate_ready"):
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
        with mock.patch.object(
            original, "_run_revision_with_recovery",
            side_effect=RuntimeError("crash before semantic worker"),
        ):
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
        with self._crash_on_review(original, 3):
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
        with self._crash_on_review(original, 2):
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

    def _crash_in_the_second_check_repair_attempt(self):
        # Attempt 001 leaves the gate red; interrupt before attempt 002 starts.
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "still bad\n"))
        config = self.config(check_repair=2)
        original = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        )
        real = type(original)._run_check_repair_attempt

        def crash_before_second(owner, *args, **kwargs):
            if len(args) >= 5 and args[4] == 2:
                raise RuntimeError("crash before repair attempt 2")
            return real(owner, *args, **kwargs)

        with mock.patch.object(
            type(original), "_run_check_repair_attempt", crash_before_second,
        ):
            failed = original.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
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
        config, path, checkpoint = self._crash_in_the_second_check_repair_attempt()
        checkpoint["check_repair_attempt"] = 3
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        resumed = self.orchestrator(config, planner=["unused"], reviewer=["unused"]).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
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
        store.update(status=store.load()["status"], run_options_sha256=None)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(
                self.config(semantic_revision=True), planner=["unused"], reviewer=["unused"],
            ).resume("run")
        self.assertEqual(self.workers.roles(), ["implementer"])

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
