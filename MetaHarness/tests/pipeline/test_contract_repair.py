"""Step-contract repair: planner repair, rollback and review of a no-change tree."""

from __future__ import annotations

import json
import unittest

from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.models import ExecutionRole, RunStatus

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    git,
    initial_plan,
    repaired_step_contract,
    review,
    write,
)


class ContractRepairTests(PipelineHarness):
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
        # The mismatch is resolved inside its own step: the durable record is
        # COMPLETED and no step status ever reports a deferred contract.
        step = self.run_dir("contract-repair") / "cycles/001/implementation/steps/S01/step.json"
        self.assertEqual(json.loads(step.read_text())["status"], "COMPLETED")
        self.assertEqual(
            [row["status"] for row in self.state("contract-repair")["steps"]],
            ["completed"],
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
            self.config(max_step_contract_repairs=1, correction_cycles=1),
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

if __name__ == "__main__":
    unittest.main()
