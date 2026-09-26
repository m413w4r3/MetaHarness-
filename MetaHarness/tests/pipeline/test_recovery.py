"""Recovery policy as the pipeline applies it: budgets, ladders, fallbacks."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest import mock

from metaharness.models import ExecutionRole, RunStatus
from metaharness.recovery_policy import ExecutionFallbacks, RecoveryBudgets
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    check_repair_result,
    initial_plan,
    ladder_strategies,
    review,
    write,
)


class RecoveryPathTests(PipelineHarness):
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
        # An unreachable infrastructure leaves for the external terminal before
        # any correctness strategy is consumed or counted.
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/ladder.json").exists()
        )

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
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
        )
        self.workers.on(ExecutionRole.REPAIR, lambda _request: check_repair_result())
        config = self.config(check_repair=1)
        original = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        )
        first_planner = original._runtime.planner_client
        waiting = original.run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        # The ladder walked every distinct strategy of this red gate: the
        # budgeted pass left the tree unchanged, the replan replayed the
        # approved step without changing it either.
        self.assertEqual(ladder_strategies(self), ["repair_targeted", "replan_step"])
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.checkpoint()["check_repair_attempt"], 1)
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
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "still bad\n"),
        )
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
        # Every distinct correctness strategy is durably spent before the
        # operator is asked: the repair left the failed check red and the one
        # replan of the responsible approved step reproduced that exact tree.
        self.assertEqual(ladder_strategies(self), ["repair_targeted", "replan_step"])
        self.assertEqual(self.checkpoint()["check_repair_attempt"], 1)
        fingerprint = self.state()["check_repair"]["operator_retry_fingerprint"]

        resumed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).resume("run")

        state = self.state()
        self.assertEqual(resumed.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(state["failure"]["reason"], "CHECK_REPAIR_FIXED_POINT")
        self.assertEqual(state["check_repair"]["status"], "fixed_point")
        self.assertEqual(state["failure"]["detail"]["fixed_point_fingerprint"], fingerprint)
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
        # The retry only re-ran the deterministic gate: no strategy was
        # replayed for facts the ladder had already spent.
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "implementer"])
        self.assertEqual(ladder_strategies(self), ["repair_targeted", "replan_step"])

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
        # The ladder position is part of the identity: a retry that stopped
        # after a different set of distinct strategies is a different fact.
        self.assertNotEqual(original, check_repair_fingerprint(
            "a" * 40, ["test-integration"], "POST_IMPLEMENTATION",
            strategy="repair_targeted+replan_step",
        ))

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

if __name__ == "__main__":
    unittest.main()
