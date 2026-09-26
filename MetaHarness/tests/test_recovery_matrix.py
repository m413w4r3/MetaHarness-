"""The architectural recovery matrix, from classification to durable outcome.

Each row states the failure code, its deterministic failure class and the
first ladder strategy the policy chooses, then the durable run state a failure
that escaped every loop projects onto: one phase, one run disposition and the
derived status.  Path-level rows drive the real facade with a failure escaping
the coordinator and check the checkpoint, the disposition and the model calls.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metaharness.attempt_transaction import (
    AttemptViolation,
    CandidateAttemptTransaction,
    contain_trusted_process,
)
from metaharness.gitops import snapshot_candidate_state
from metaharness.models import ExecutionRole, RunDisposition, RunStatus
from metaharness.orchestration.pipeline_v2 import PipelineFailure
from metaharness.orchestration.recovery import RecoveryCoordinator, project_exit
from metaharness.recovery_policy import (
    FailureClass,
    RecoveryStrategy,
    classify_failure,
    recovery_ladder,
)
from metaharness.resume import (
    CHECKPOINT_INTEGRITY_OPERATION,
    RUN_SCHEMA_UNSUPPORTED,
    ResumeNotAllowedError,
    resume_info,
)
from metaharness.resume import (
    ResumePhase as P,
)
from metaharness.state import RunCheckpointError, RunStateStore
from tests.pipeline_support import (
    PipelineHarness,
    initial_plan,
    ladder_strategies,
    repaired_step_contract,
    review,
    write,
)

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")

RD = RunDisposition
T, F, S, X = (
    FailureClass.TRANSIENT, FailureClass.FIXABLE, FailureClass.SPEC_DECISION, FailureClass.FATAL,
)

# (name, code, failure class, exit phase, exit disposition, exit status)
RECOVERY_MATRIX = (
    ("planner format", "PLANNER_FORMAT_INVALID", F, P.PLANNER, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("planner repository precondition", "PLAN_REPOSITORY_PRECONDITION_INVALID", F, P.PLANNER, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("contract mismatch", "AGENT_CONTRACT_MISMATCH", F, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("scope violation", "AGENT_SCOPE_VIOLATION", F, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("Git ownership violation", "AGENT_GIT_VIOLATION", F, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("ordinary check failure", "CHECK_FAILED:unit", F, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("check repair exhausted", "CHECK_REPAIR_EXHAUSTED", F, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_REPAIR),
    ("contract repair slot", "STEP_CONTRACT_REPAIR_OUTPUT_INVALID", F, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_CONTRACT_REPAIR),
    ("review format", "REVIEW_FORMAT_INVALID", F, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("review repair exhausted", "WAITING_REPAIR_EXHAUSTED", F, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("vague operator code", "HUMAN_REQUIRED", F, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("unknown failure code", "TOTALLY_NEW_FAILURE", F, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("agent timeout", "AGENT_TIMEOUT", T, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("transport horizon", "LLM_TRANSPORT_EXHAUSTED", T, P.PLANNER, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("auth", "EXTERNAL_AUTH_REQUIRED", T, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("check infrastructure", "CHECK_INFRASTRUCTURE_UNAVAILABLE", T, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
    ("push failed", "PUSH_FAILED", T, P.CANDIDATE_PUSH, RD.WAIT_EXTERNAL, RunStatus.WAITING_REMOTE),
    ("semantic reviser unavailable", "SEMANTIC_REVISER_UNAVAILABLE", T, P.SEMANTIC_REVISION, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("product decision", "SPEC_DECISION_REQUIRED", S, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("reviewer product decision", "REVIEW_HUMAN_REQUIRED", S, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("secret", "SECRET_IN_DIFF", X, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("unscannable staged source", "UNSCANNABLE_STAGED_BLOB", X, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("rollback failure", "ROLLBACK_FAILED", X, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("corrupt artifact", "DURABLE_ARTIFACT_CORRUPTED", X, P.FINAL_REVIEW, RD.FAILED, RunStatus.FAILED),
    ("foreign branch", "BRANCH_MODIFIED_OUTSIDE_AUTHORITY", X, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
)


class RecoveryMatrixTests(unittest.TestCase):
    def test_each_row_names_its_class_its_first_step_and_its_exit_state(self) -> None:
        for name, code, failure_class, phase, disposition, status in RECOVERY_MATRIX:
            with self.subTest(failure=name):
                decision = classify_failure(code)
                self.assertIs(decision.failure_class, failure_class)
                self.assertIs(decision.strategy, recovery_ladder(failure_class)[0])
                exit_decision, terminal = project_exit(code, phase=phase)
                self.assertIs(exit_decision.strategy, recovery_ladder(failure_class)[-1])
                self.assertIs(terminal.phase, phase)
                self.assertIs(terminal.disposition, disposition)
                self.assertEqual(terminal.status, status)
                self.assertIs(terminal.resumable, disposition is RD.WAIT_EXTERNAL)

    def test_the_autonomous_classes_start_with_an_autonomous_step(self) -> None:
        for failure_class in (T, F):
            with self.subTest(failure_class=failure_class):
                self.assertFalse(recovery_ladder(failure_class)[0].terminal)
                self.assertNotIn(RecoveryStrategy.WAIT_HUMAN, recovery_ladder(failure_class))
                self.assertNotIn(RecoveryStrategy.HARD_STOP, recovery_ladder(failure_class))


class RecoveryCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RunStateStore(Path(self.temp.name) / "state.json")
        self.store.initialize("run")
        self.events: list[tuple[str, dict]] = []

    def coordinator(self) -> RecoveryCoordinator:
        return RecoveryCoordinator(
            self.store, emit=lambda event, **kwargs: self.events.append((event, kwargs)),
        )

    def admit(self, coordinator: RecoveryCoordinator, **overrides):
        options = dict(
            reason="AGENT_TIMEOUT", budget=2, phase="implementation", cycle=1,
            step_id="S01", profile_id="worker", tree_before="a" * 40, tree_after="a" * 40,
        )
        options.update(overrides)
        return coordinator.admit("agent-step:001:S01", **options)

    def test_budget_is_durable_and_never_reset_by_a_resume(self) -> None:
        self.assertTrue(self.admit(self.coordinator()).admitted)
        state = self.store.load()
        self.store.update_metadata(resume={"attempts": 3})
        # A fresh coordinator after a resume sees the same consumed budget.
        second = self.admit(self.coordinator())
        self.assertTrue(second.admitted)
        self.assertEqual(second.used, 2)
        exhausted = self.admit(self.coordinator())
        self.assertFalse(exhausted.admitted)
        self.assertTrue(exhausted.exhausted)
        self.assertIs(exhausted.decision.strategy, RecoveryStrategy.WAIT_EXTERNAL)
        self.assertEqual(self.store.load()["recovery_counters"], {"agent-step:001:S01": 2})
        data = self.events[-1][1]["data"]
        self.assertEqual(self.events[-1][0], "recovery.exhausted")
        self.assertEqual(data["strategy"], RecoveryStrategy.WAIT_EXTERNAL.value)
        self.assertEqual(data["terminal_status"], RunStatus.WAITING_EXTERNAL.value)
        self.assertEqual(data["checkpoint_phase"], P.IMPLEMENT_STEP.value)

    def test_each_consumed_attempt_has_a_durable_identity(self) -> None:
        self.admit(self.coordinator(), tree_after="b" * 40)
        (record,) = self.store.load()["recovery_attempts"]
        self.assertEqual(record, {
            "phase": "implementation", "reason": "AGENT_TIMEOUT", "attempt": 1,
            "budget_key": "agent-step:001:S01", "budget": 2, "budget_consumed": 1,
            "strategy": "retry_targeted", "cycle": 1, "step_id": "S01",
            "operation_id": "recovery:agent-step:001:S01:01",
            "profile_id": "worker", "tree_before": "a" * 40, "tree_after": "b" * 40,
        })

    def test_every_recovery_event_speaks_the_strategy_vocabulary(self) -> None:
        self.admit(self.coordinator())
        classified = self.events[0]
        self.assertEqual(classified[0], "recovery.classified")
        self.assertEqual(classified[1]["data"]["strategy"], RecoveryStrategy.RETRY_TARGETED.value)
        self.assertEqual(classified[1]["data"]["failure_class"], FailureClass.TRANSIENT.value)
        # The removed second vocabulary never reaches a durable trace.
        self.assertNotIn("disposition", classified[1]["data"])

    def test_disallowed_strategy_consumes_nothing(self) -> None:
        refused = self.admit(
            self.coordinator(), reason="AGENT_TIMEOUT",
            allowed={RecoveryStrategy.FALLBACK_EXECUTOR},
        )
        self.assertFalse(refused.admitted)
        self.assertFalse(refused.exhausted)
        self.assertEqual(self.store.load()["recovery_counters"], {})
        self.assertNotIn("recovery_attempts", self.store.load())

    def test_malformed_counters_fail_closed(self) -> None:
        state = self.store.load()
        self.store.update_metadata(recovery_counters={"agent-step:001:S01": -1})
        with self.assertRaises(PipelineFailure) as caught:
            self.admit(self.coordinator())
        self.assertEqual(caught.exception.reason, "DURABLE_ARTIFACT_CORRUPTED")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


class AttemptTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.name", "t")
        _git(self.repo, "config", "user.email", "t@example.invalid")
        (self.repo / "a.txt").write_text("a\n", encoding="utf-8")
        (self.repo / "b.txt").write_text("b\n", encoding="utf-8")
        _git(self.repo, "add", "--all")
        _git(self.repo, "commit", "-qm", "base")
        self.tree = _git(self.repo, "rev-parse", "HEAD^{tree}")

    def begin(self, secrets: tuple[str, ...] = ()) -> CandidateAttemptTransaction:
        return CandidateAttemptTransaction.begin(
            self.repo, self.repo, branch_ref="refs/heads/main", secrets=secrets,
        )

    def test_in_scope_failed_attempt_is_rolled_back_exactly(self) -> None:
        transaction = self.begin()
        (self.repo / "a.txt").write_text("partial\n", encoding="utf-8")
        (self.repo / "new.txt").write_text("new\n", encoding="utf-8")
        rollback = transaction.abort({"a.txt", "new.txt"})
        self.assertEqual(set(rollback.changed_paths), {"a.txt", "new.txt"})
        self.assertEqual(_git(self.repo, "write-tree"), self.tree)
        self.assertEqual(_git(self.repo, "status", "--porcelain"), "")

    def test_out_of_scope_mutation_is_a_fixable_scope_violation(self) -> None:
        transaction = self.begin()
        (self.repo / "b.txt").write_text("outside\n", encoding="utf-8")
        with self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertEqual(caught.exception.code, "AGENT_SCOPE_VIOLATION")
        # A neighbour edit is never fatal by itself; only a failed restore is.
        self.assertIs(classify_failure(caught.exception.code).failure_class, FailureClass.FIXABLE)

    def test_secret_in_a_failed_attempt_is_never_silently_rolled_back(self) -> None:
        transaction = self.begin(secrets=("sk-live-secret-value-123456",))
        (self.repo / "a.txt").write_text("key=sk-live-secret-value-123456\n", encoding="utf-8")
        with self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertTrue(caught.exception.code.startswith("SECRET_"), caught.exception.code)
        self.assertIs(classify_failure(caught.exception.code).strategy, RecoveryStrategy.HARD_STOP)

    def test_git_ownership_change_is_an_authority_violation(self) -> None:
        transaction = self.begin()
        _git(self.repo, "branch", "rogue")
        with self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertEqual(caught.exception.code, "AGENT_GIT_VIOLATION")

    def test_non_exact_rollback_requires_an_operator(self) -> None:
        transaction = self.begin()
        (self.repo / "a.txt").write_text("partial\n", encoding="utf-8")
        with mock.patch(
            "metaharness.attempt_transaction.restore_paths_from_tree", lambda *_args: None,
        ), self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertEqual(caught.exception.code, "RESUME_REQUIRES_OPERATOR")
        self.assertIs(classify_failure(caught.exception.code).strategy, RecoveryStrategy.HARD_STOP)

    def test_trusted_process_side_effects_are_contained(self) -> None:
        before = snapshot_candidate_state(self.repo)
        (self.repo / "a.txt").write_text("formatted\n", encoding="utf-8")
        effect = contain_trusted_process(self.repo, before, label="check")
        self.assertEqual(effect.changed_paths, ("a.txt",))
        self.assertEqual(snapshot_candidate_state(self.repo), before)
        _git(self.repo, "branch", "rogue")
        with self.assertRaises(AttemptViolation) as caught:
            contain_trusted_process(self.repo, before, label="check")
        self.assertEqual(caught.exception.code, "AGENT_GIT_VIOLATION")


class RecoveryPathTests(PipelineHarness):
    """A failure escaping the coordinator: status, checkpoint, model calls."""

    # (code, durable phase, durable disposition, derived status, resumable)
    PATHS = (
        # An unknown or ordinary failure never stops the run.
        ("TOTALLY_NEW_FAILURE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("AGENT_SCOPE_VIOLATION", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("SECRET_IN_DIFF", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("ROLLBACK_FAILED", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("AGENT_AUTH_FAILURE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("CHECK_INFRASTRUCTURE_UNAVAILABLE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE, True),
        ("REVIEWER_TRANSPORT_FAILURE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("SPEC_DECISION_REQUIRED", P.IMPLEMENT_STEP, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN, False),
        ("REVIEW_REPLAN", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("AGENT_TIMEOUT", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
    )

    def test_escaped_failures_project_without_calling_models(self) -> None:
        for index, (code, phase, disposition, status, resumable) in enumerate(self.PATHS):
            run_id = f"path-{index}"
            with self.subTest(code=code), mock.patch(
                "metaharness.orchestration.runtime.PipelineV2Coordinator.run",
                side_effect=PipelineFailure(code, "diagnostic"),
            ):
                result = self.orchestrator(
                    self.config(), planner=[initial_plan(STEP)], reviewer=["unused"],
                ).run_text(SPEC, run_id=run_id)
                state = self.state(run_id)
                self.assertEqual(result.status, status)
                self.assertEqual(state["status"], status.value)
                self.assertEqual(state["disposition"], disposition.value)
                self.assertEqual(state["failure"]["reason"], (
                    "EXTERNAL_AUTH_REQUIRED" if code == "AGENT_AUTH_FAILURE" else code
                ))
                # The exact pre-execution checkpoint is preserved for a resume,
                # and it is the phase authority of the run.
                self.assertEqual(self.checkpoint(run_id)["phase"], phase.value)
                self.assertEqual(resume_info(self.run_dir(run_id), state).resumable, resumable)
                self.assertEqual(len(self.planner.requests), 1)
                self.assertEqual(self.reviewer.requests, [])
                self.assertEqual(self.workers.calls, [])

    def _wait_at_a_resumable_checkpoint(self) -> None:
        with mock.patch(
            "metaharness.orchestration.runtime.PipelineV2Coordinator.run",
            side_effect=PipelineFailure("AGENT_TIMEOUT", "diagnostic"),
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=["unused"],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL)
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)

    def test_an_older_checkpoint_schema_is_a_plain_refusal(self) -> None:
        self._wait_at_a_resumable_checkpoint()
        path = self.run_dir() / "resume_checkpoint.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] -= 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        state_before = (self.run_dir() / "state.json").read_bytes()

        info = resume_info(self.run_dir(), self.state())

        self.assertFalse(info.resumable)
        self.assertEqual(info.operation, RUN_SCHEMA_UNSUPPORTED)
        resumed = self.orchestrator(self.config(), planner=["unused"], reviewer=["unused"])
        with self.assertRaises(ResumeNotAllowedError):
            resumed.resume("run")
        # An incompatible runtime is refused, never recorded as an incident.
        self.assertEqual((self.run_dir() / "state.json").read_bytes(), state_before)
        self.assertEqual(self.planner.requests, [])

    def test_a_corrupt_current_checkpoint_fails_resume_integrity(self) -> None:
        self._wait_at_a_resumable_checkpoint()
        path = self.run_dir() / "resume_checkpoint.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["step_id"] = "not-a-step"
        path.write_text(json.dumps(payload), encoding="utf-8")

        info = resume_info(self.run_dir(), self.state())

        self.assertFalse(info.resumable)
        self.assertEqual(info.operation, CHECKPOINT_INTEGRITY_OPERATION)
        resumed = self.orchestrator(
            self.config(), planner=["unused"], reviewer=["unused"],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(self.workers.calls, [])
        # Same refusal once the file cannot even be parsed: the failure stays
        # recordable, the fallback phase stays descriptive, and no control read
        # nor any later resume may read it back.
        path.write_text("{not json", encoding="utf-8")
        store = RunStateStore(self.run_dir() / "state.json")
        self.assertIs(store.reported_machine_state().phase, P.IMPLEMENT_STEP)
        with self.assertRaises(RunCheckpointError):
            store.machine_state()
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)

    def test_persistent_worker_timeout_waits_after_its_bounded_retries(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, timeout, timeout, timeout)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], P.IMPLEMENT_STEP.value)
        self.assertEqual(self.workers.roles(), ["implementer"] * 3)
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(self.reviewer.requests, [])
        attempts = self.state()["recovery_attempts"]
        self.assertEqual([item["attempt"] for item in attempts], [1, 2])
        self.assertTrue(all(item["step_id"] == "S01" and item["cycle"] == 1 for item in attempts))
        self.assertTrue(resume_info(self.run_dir(), self.state()).resumable)

    def test_red_gate_without_repair_budget_waits_for_an_operator(self) -> None:
        """A zero budget refuses the worker rungs, never the autonomous ones."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "bad\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=0, correction_cycles=1),
            planner=[
                initial_plan(STEP), repaired_step_contract(),
                # The last autonomous rung re-decomposes the cycle; the plan it
                # answers with is the one already in force, so it is spent.
                initial_plan(STEP),
            ],
            reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.checkpoint()["phase"], P.DETERMINISTIC_GATE.value)
        # The budgeted repair pass is refused, so the ladder consumes both
        # autonomous replan rungs before the operator is ever asked.
        self.assertEqual(ladder_strategies(self), ["replan_step", "replan_cycle"])
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/attempts").exists()
        )
        self.assertEqual(self.reviewer.requests, [])
        events = [
            json.loads(line) for line in
            (self.run_dir() / "trace/events.v1.jsonl").read_text().splitlines()
        ]
        self.assertFalse(any(item["event"] == "run.failed" for item in events))

    def test_a_passing_run_still_commits(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertNotIn("recovery_attempts", self.state())


if __name__ == "__main__":
    unittest.main()
