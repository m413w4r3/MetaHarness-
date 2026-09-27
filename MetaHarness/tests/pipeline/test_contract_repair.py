"""Step-contract repair: planner repair, rollback and the no-change tree."""

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
        ).run_text(SPEC, run_id="contract-repair")

        self.assertEqual(
            result.status, RunStatus.PUBLISHED,
            self.state("contract-repair").get("failure"),
        )
        self.assertEqual(observed_before_retry, ["base\n"])
        self.assertEqual(self.workers.roles(), ["implementer", "implementer", "auditor"])
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
        ).run_text(SPEC, run_id="residual-mismatch")

        worktree = self.worktree("residual-mismatch")
        # An exhausted contract replan settles the step; no operator decision.
        self.assertNotIn(result.status, {RunStatus.WAITING_HUMAN, RunStatus.FAILED})
        step = self.run_dir("residual-mismatch") / "cycles/001/implementation/steps/S01/step.json"
        record = json.loads(step.read_text())
        self.assertEqual((record["status"], record["reason"]), ("FAILED_CONTINUED", "AGENT_CONTRACT_MISMATCH"))
        self.assertEqual(git(worktree, "rev-parse", "HEAD"), self.base_sha)
        self.assertEqual(git(worktree, "status", "--porcelain"), "")
        self.assertEqual((worktree / "feature.txt").read_text(), "base\n")
        self.assertEqual(
            [request.role for request in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 2,
        )

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
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="environment-verify")

        self.assertEqual(
            result.status, RunStatus.PUBLISHED,
            self.state("environment-verify").get("failure"),
        )
        self.assertEqual(
            [request.role for request in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 1,
        )
        self.assertEqual(len(self.planner.requests), 1)
        steps_dir = (
            self.run_dir("environment-verify")
            / "cycles/001/implementation/steps/S01"
        )
        self.assertFalse((steps_dir / "contract_repairs").exists())
        self.assertNotIn("AGENT_CONTRACT_MISMATCH", self.trace_names("environment-verify"))

    def test_a_no_change_tree_is_settled_by_the_deterministic_gate_alone(self) -> None:
        # The worker changes nothing and the required check passes on the
        # unchanged tree. One authority settles that tree: the deterministic
        # gate. No audit, no replan and no human wait is spent on a candidate
        # that carries no delta.
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            lambda _request: "done\n", lambda _request: "done\n",
        )
        result = self.orchestrator(
            self.config(max_step_contract_repairs=1),
            planner=[initial_plan(STEP), repaired_step_contract()],
        ).run_text(SPEC, run_id="no-change")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state("no-change").get("failure"))
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])
        candidate = json.loads(
            (self.run_dir("no-change") / "cycles/001/candidate/commit.json").read_text()
        )
        self.assertTrue(candidate["no_change"])
        self.assertEqual(candidate["commit_sha"], self.base_sha)
        self.assertEqual(self.state("no-change").get("no_change"), True)
        self.assertIn("run.completed_no_change", self.trace_names("no-change"))


if __name__ == "__main__":
    unittest.main()
