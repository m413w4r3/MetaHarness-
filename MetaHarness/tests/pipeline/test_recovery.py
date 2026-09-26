"""Recovery policy as the pipeline applies it: budgets, ladders, fallbacks."""

from __future__ import annotations

import json
import re
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from metaharness.models import (
    CycleKind,
    ExecutionRole,
    RunStatus,
    correction_cycles_used,
    is_correction_cycle,
    is_replan_cycle,
)
from metaharness.orchestration.revision import EffectivePlanView
from metaharness.recovery_policy import ExecutionFallbacks, RecoveryBudgets
from metaharness.resume import resume_info

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    check_repair_result,
    correction_plan,
    git,
    initial_plan,
    ladder_strategies,
    repaired_step_contract,
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
        config = self.config(check_repair=1, correction_cycles=1)
        original = self.orchestrator(
            config,
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                # The re-decomposition rung answers with the plan already in
                # force: the same decomposition for the same facts is refused
                # and spends the rung instead of replaying approved steps.
                initial_plan(STEP),
            ],
            reviewer=[review()],
        )
        first_planner = original._runtime.planner_client
        waiting = original.run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        # The ladder walked every distinct strategy of this red gate: the
        # budgeted pass left the tree unchanged, the one replan of the
        # responsible approved step rewrote its contract and re-ran it, and the
        # cycle replan re-decomposed nothing.
        self.assertEqual(
            ladder_strategies(self), ["repair_targeted", "replan_step", "replan_cycle"],
        )
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
        # The first run bought the plan, its one contract replan and its one
        # re-decomposition; the operator retry re-ran the deterministic gate
        # and nothing else.
        self.assertEqual(len(first_planner.requests), 3)
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
        config = self.config(check_repair=1, correction_cycles=1)
        waiting = self.orchestrator(
            config,
            planner=[
                initial_plan(STEP), repaired_step_contract(), initial_plan(STEP),
            ],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR)
        self.assertEqual(self.state()["check_repair"]["next_action"], "Retry deterministic gate")
        # Every distinct correctness strategy is durably spent before the
        # operator is asked: the repair left the failed check red, the one
        # replan of the responsible approved step reproduced that exact tree,
        # and the re-decomposition answered nothing new.
        self.assertEqual(
            ladder_strategies(self), ["repair_targeted", "replan_step", "replan_cycle"],
        )
        unchanged = json.loads(
            (self.run_dir() / "cycles/002/check-replan/check_replan.plan.json").read_text()
        )
        self.assertEqual(unchanged["plan_identity_after"], unchanged["plan_identity_before"])
        self.assertNotIn("implementation_bundle_sha256", unchanged)
        self.assertEqual(self.checkpoint()["check_repair_attempt"], 1)
        fingerprint = self.state()["check_repair"]["operator_retry_fingerprint"]

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")

        # The identical answer is durable: the retry re-ran the gate and asked
        # the planner nothing.
        self.assertEqual(self.planner.requests, [])
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
        self.assertEqual(
            ladder_strategies(self), ["repair_targeted", "replan_step", "replan_cycle"],
        )

    def test_two_verified_check_repairs_can_exhaust_the_budget(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "worse still\n"),
        )
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "first targeted repair ran and failed",
            )),
            write("feature.txt", "worse\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "second targeted repair ran and failed",
            )),
        )
        result = self.orchestrator(
            self.config(check_repair=2, correction_cycles=1),
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                # The last rung re-decomposes the cycle and answers with the
                # plan already in force: it is refused instead of replaying it.
                initial_plan(STEP),
            ],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertIsInstance(self.state()["failure"]["detail"], dict)
        self.assertEqual(self.state()["failure"]["detail"]["failed_check_ids"], ["test"])
        self.assertEqual(self.state()["failure"]["detail"]["attempt_count"], 2)
        self.assertEqual(
            self.state()["failure"]["detail"]["strategy"],
            "repair_targeted+repair_targeted+replan_step+replan_cycle",
        )
        self.assertEqual(
            self.state()["failure"]["detail"]["strategies"],
            ["repair_targeted", "repair_targeted", "replan_step", "replan_cycle"],
        )
        self.assertEqual(self.state()["check_repair"]["failure_classification"], "product_check")
        self.assertEqual(self.state()["check_repair"]["next_action"], "Retry deterministic gate")
        self.assertRegex(self.state()["check_repair"]["latest_evidence_sha256"], r"^[0-9a-f]{64}$")
        reports = self.state()["check_repair"]["repair_reports"]
        self.assertEqual([item["attempt"] for item in reports], [1, 2])
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in reports))
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["phase"], "deterministic_gate")
        self.assertEqual(checkpoint["stage"], "POST_IMPLEMENTATION")
        # The ladder rewrote the tree after the last recorded pass, so the
        # checkpoint attributes it to no pass at all.
        self.assertIsNone(checkpoint["check_repair_attempt"])
        # Two budgeted passes, then the one distinct replan of the responsible
        # approved step and the one re-decomposition of the cycle, before the
        # exhausted ladder waits for the operator.
        self.assertEqual(
            self.workers.roles(), ["implementer", "repair", "repair", "implementer"],
        )
        self.assertEqual(
            ladder_strategies(self),
            ["repair_targeted", "repair_targeted", "replan_step", "replan_cycle"],
        )
        diagnostics_path = self.run_dir() / "diagnostics.md"
        self.assertTrue(
            diagnostics_path.is_file(),
            (self.run_dir() / "diagnostics.error.txt").read_text(encoding="utf-8")
            if (self.run_dir() / "diagnostics.error.txt").is_file() else "diagnostics missing",
        )
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        recovery_summary = diagnostics.split("## DETERMINISTIC GATE RECOVERY", 1)[1].split("\n## ", 1)[0]
        for expected in (
            # Two repaired red gates, the replan rung and their reruns.
            "deterministic gate attempt: 4",
            "check-repair attempts used / budget: 2 / 2",
            "latest failed check IDs: test",
            "failure classification: product_check",
            "next recovery action: Retry deterministic gate",
        ):
            self.assertIn(expected, recovery_summary, recovery_summary)
        attempts = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts"
        self.assertEqual(sorted(path.name for path in attempts.iterdir()), ["001", "002"])
        self.assertEqual(self.reviewer.requests, [])

    def test_a_spent_correction_budget_moves_the_ladder_to_its_fallback_rung(self) -> None:
        """A refused cycle rung is never an immediate human stop.

        This run has no correction unit to spend, so its cycle rung is refused
        without a planner call; the ladder simply consumes the rungs that
        follow it, and its frozen fallback executor is the one that turns the
        deterministic gate green.
        """

        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
        )
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n"), write("feature.txt", "still bad\n"),
            timeout, write("feature.txt", "good\n"),
        )
        config = self.config(check_repair=3, correction_cycles=0)
        config = replace(config, recovery=RecoveryBudgets(
            max_transient_attempts=0,
            execution_fallbacks=ExecutionFallbacks(check_repair=("live_repairer",)),
        ))
        result = self.orchestrator(
            config,
            planner=[initial_plan(STEP), repaired_step_contract()],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The rung the budget refused is absent from the episode; the rung that
        # follows it in the ladder is the one that finished the job.
        self.assertEqual(
            ladder_strategies(self),
            ["repair_targeted", "repair_targeted", "replan_step", "fallback_executor"],
        )
        self.assertEqual(self.workers.calls[-2].profile_id, "repairer")
        self.assertEqual(self.workers.calls[-1].profile_id, "live_repairer")
        self.assertEqual(
            [request for request in self.planner.requests if "re-decomposition" in request], [],
        )


    def test_a_cycle_replan_that_only_repeats_the_plan_never_loops(self) -> None:
        """The same answer for the same failure facts is spent, never replayed.

        Cycle 002 fails its own gate under the decomposition it just executed,
        so its re-decomposition rung is admitted -- and an answer that only
        repeats the plan already in force for these exact facts is refused and
        durably bound to them instead of opening a third cycle.
        """

        rewritten = ("S01", "feature.txt", "Rewrite the feature so the configured test passes")
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
            write("feature.txt", "worse\n"),
        )
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "bad\n"), write("feature.txt", "worse repaired\n"),
        )
        config = self.config(check_repair=1, correction_cycles=2)
        original = self.orchestrator(
            config,
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                correction_plan(rewritten), correction_plan(rewritten),
            ],
            reviewer=[review()],
        )
        waiting = original.run_text(SPEC, run_id="run")

        self.assertEqual(waiting.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(
            ladder_strategies(self, cycle=2, stage="post-check-replan"),
            ["repair_targeted", "replan_step", "replan_cycle"],
        )
        first = json.loads(
            (self.run_dir() / "cycles/002/check-replan/check_replan.plan.json").read_text()
        )
        second = json.loads(
            (self.run_dir() / "cycles/003/check-replan/check_replan.plan.json").read_text()
        )
        # Cycle 002 executed a genuinely new decomposition; cycle 003 was
        # refused the one already in force, so nothing was replayed.
        self.assertNotEqual(first["plan_identity_after"], first["plan_identity_before"])
        self.assertEqual(second["plan_identity_before"], first["plan_identity_after"])
        self.assertEqual(second["plan_identity_after"], second["plan_identity_before"])
        self.assertEqual(second["ladder_fingerprint"], [
            second["candidate_tree_sha"], ["test"], "replan_cycle",
            second["plan_identity_before"],
        ])
        self.assertFalse((self.run_dir() / "cycles/003/cycle.json").exists())
        self.assertEqual(
            self.workers.roles(), ["implementer", "repair", "implementer", "implementer", "repair"],
        )
        # One plan, one contract replan and one re-decomposition per cycle: the
        # repeated answer was never a second planner completion.
        self.assertEqual(len(original._runtime.planner_client.requests), 4)

    def test_zero_correction_budget_skips_cycle_replan_without_planner_call(self) -> None:
        """A spent correction budget refuses the cycle rung before any planning.

        The only approved path is a test file the failure names, so no step is
        proven responsible either: the ladder has no rung left, asks the
        planner nothing and never opens the cycle the rung would have.
        """

        target = self.repo / "tests/test_feature.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("base test\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add test fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.check.write_text(
            "import sys\n"
            "print('tests/test_feature.py: the fixture does not match the spec')\n"
            "raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("tests/test_feature.py", "bad fixture\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=0, correction_cycles=0),
            planner=[initial_plan(("S01", "tests/test_feature.py", "Write the fixture"))],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.state()["failure"]["detail"]["strategies"], [])
        # No rung was consumed, no planner was asked and no cycle exists: the
        # budget refused the rung before its transaction could run.
        self.assertEqual(len(self.planner.requests), 1)
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/ladder.json").exists()
        )
        self.assertFalse((self.run_dir() / "cycles/002").exists())
        # No durable trace of a replan exists: no plan artifact, no request and
        # no planner completion was ever paid for by this red gate.
        self.assertEqual(list(self.run_dir().glob("cycles/*/check-replan/*")), [])
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_one_correction_budget_allows_exactly_one_check_replan(self) -> None:
        """One unit admits one cycle replan; the cycle it opened cannot buy a second.

        Cycle 001's red gate spends the run's single correction unit on the
        re-decomposition that opens cycle 002.  Cycle 002 executes it and fails
        its own gate, and the rung that would re-decompose once more is refused
        by the budget the running cycle already paid with: no third cycle, no
        second re-decomposition and no planner call for one.
        """

        rewritten = ("S01", "feature.txt", "Rewrite the feature so the configured test passes")
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "worse\n"),
            write("feature.txt", "worse2\n"), write("feature.txt", "worse3\n"),
        )
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "worse4\n"), write("feature.txt", "worse5\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=1, correction_cycles=1),
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                correction_plan(rewritten),
                repaired_step_contract("Rewrite the feature so the configured test passes"),
            ],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        # The one unit bought exactly one re-decomposition: the plan cycle 002
        # executed, answered by the planner exactly once.
        plans = sorted(self.run_dir().glob("cycles/*/check-replan/check_replan.plan.json"))
        self.assertEqual([path.parent.parent.name for path in plans], ["002"])
        self.assertEqual(
            len([request for request in self.planner.requests if "re-decomposition" in request]),
            1,
        )
        self.assertEqual(
            json.loads((self.run_dir() / "cycles/002/cycle.json").read_text())["kind"],
            "check-replan",
        )
        # The red gate of that cycle is refused the rung before a third cycle
        # could exist, and the ladder still reaches its own terminal.
        self.assertNotIn(
            "replan_cycle", ladder_strategies(self, cycle=2, stage="post-check-replan"),
        )
        self.assertFalse((self.run_dir() / "cycles/003").exists())


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


class PipelineOperationsContractTests(unittest.TestCase):
    """The machine cannot be bound without its single recovery ladder."""

    def test_pipeline_operations_requires_recovery_ladder(self) -> None:
        import dataclasses

        from metaharness.orchestration.pipeline_v2 import PipelineV2Operations

        fields = dataclasses.fields(PipelineV2Operations)
        ladder = next(item for item in fields if item.name == "recovery_operations")
        self.assertIs(ladder.default, dataclasses.MISSING)
        self.assertIs(ladder.default_factory, dataclasses.MISSING)
        without = {
            item.name: (lambda *args, **kwargs: None)
            for item in fields
            if item.name != "recovery_operations"
        }
        with self.assertRaises(TypeError):
            PipelineV2Operations(**without)
        with self.assertRaises(TypeError):
            PipelineV2Operations(recovery_operations=None, **without)


class CorrectionCycleVocabularyTests(unittest.TestCase):
    """One canonical pair of predicates decides what a correction cycle is."""

    def test_correction_and_replan_kinds_are_canonical(self) -> None:
        self.assertEqual(
            [kind.value for kind in CycleKind if is_correction_cycle(kind)],
            ["review-implementation", "review-replan", "check-replan"],
        )
        self.assertEqual(
            [kind.value for kind in CycleKind if is_replan_cycle(kind)],
            ["review-replan", "check-replan"],
        )
        self.assertFalse(is_correction_cycle(CycleKind.INITIAL))
        # The counter every admission reads is derived from the cycle number:
        # nothing durable can hand a resumed run back a spent unit.
        self.assertEqual([correction_cycles_used(number) for number in (1, 2, 3)], [0, 1, 2])


class EffectivePlanViewTests(unittest.TestCase):
    """The plan authority treats a red-gate re-decomposition as a correction."""

    @staticmethod
    def _cycle_plan(number: int, kind: str, path: str) -> SimpleNamespace:
        plan = SimpleNamespace(
            title=f"Plan {number}",
            objective=f"Plan {number} objective",
            constraints="ORIGINAL CONSTRAINTS",
            required_checks=("unit", "integration"),
            steps=(SimpleNamespace(
                id=f"S{number:02d}",
                title=f"Step {number}",
                depends_on=None,
                objective=f"step objective {number}",
                write_set=(path,),
                create_set=(),
                delete_set=(),
                verify="run checks",
                forbidden="do not widen scope",
            ),),
        )
        return SimpleNamespace(
            cycle=SimpleNamespace(number=number, kind=CycleKind(kind)),
            plan=plan,
            correction_bundle_sha256=f"{number:064x}",
        )

    def test_check_replan_is_present_in_effective_plan_view(self) -> None:
        initial = self._cycle_plan(1, "initial", "src/initial.py")
        replan = self._cycle_plan(2, "check-replan", "src/correction.py")
        view = EffectivePlanView.from_cycle_plans(initial.plan, [initial, replan])

        accepted = view.accepted_correction_plans
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["cycle"], 2)
        self.assertEqual(accepted[0]["cycle_kind"], "check-replan")
        self.assertEqual(accepted[0]["plan_sha256"], replan.correction_bundle_sha256)
        # The re-decomposed plan is the one in force: the view carries its step
        # index, its required checks and its objective as the current authority.
        self.assertEqual([step["id"] for step in view.current_step_index], ["S02"])
        self.assertEqual(
            view.required_deterministic_check_ids, ("unit", "integration"),
        )
        self.assertEqual(view.current_cycle_correction_objective, "Plan 2 objective")

    def test_check_replan_scope_contributes_to_cumulative_scope(self) -> None:
        initial = self._cycle_plan(1, "initial", "src/initial.py")
        review = self._cycle_plan(2, "review-replan", "src/review.py")
        check = self._cycle_plan(3, "check-replan", "src/check.py")
        view = EffectivePlanView.from_cycle_plans(initial.plan, [initial, review, check])

        self.assertEqual(
            view.current_cumulative_approved_mutable_scope,
            ("src/check.py", "src/initial.py", "src/review.py"),
        )
        self.assertEqual(
            [
                (item["cycle_kind"], item["approved_scope_delta"]["added_paths"])
                for item in view.accepted_correction_plans
            ],
            [("review-replan", ["src/review.py"]), ("check-replan", ["src/check.py"])],
        )


if __name__ == "__main__":
    unittest.main()
