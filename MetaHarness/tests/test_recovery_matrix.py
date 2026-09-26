"""The architectural recovery matrix, from classification to durable outcome.

Each row states the recovery disposition the policy must choose, then the
durable run state a failure that left its recovery loop must project onto:
one phase, one run disposition, and the derived legacy status.  No row states
a status without the phase and disposition it is derived from.  Path-level
rows drive the real façade with a failure escaping the coordinator and check
the checkpoint, the disposition and the model calls.
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
from metaharness.orchestration.recovery import (
    RecoveryCoordinator,
    project_exit,
    strategy_terminal_state,
    terminal_state_for,
)
from metaharness.recovery_policy import (
    FailureClass,
    RecoveryFacts,
    RecoveryProgression,
    RecoveryStrategy,
    classify_failure,
    recovery_ladder,
)
from metaharness.recovery_policy import RecoveryDisposition as D
from metaharness.resume import (
    CHECKPOINT_INTEGRITY_OPERATION,
    RUN_SCHEMA_UNSUPPORTED,
    ResumeNotAllowedError,
    resume_info,
)
from metaharness.resume import (
    ResumePhase as P,
)
from metaharness.state import RunStateStore
from tests.pipeline_support import PipelineHarness, initial_plan, review, write

SPEC = "Make feature.txt good.\n"
STEP = ("S01", "feature.txt", "Write the feature")

# The durable vocabulary every row of this matrix is written in.
RD = RunDisposition

# (category, name, code, facts, disposition, exit phase, exit disposition, status or None)
# ``None`` exit: the recovery disposition is consumed in its loop and never ends a run.
RECOVERY_MATRIX = (
    ("auto", "planner format", "PLANNER_FORMAT_INVALID", {}, D.REPLAN, P.PLANNER, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "planner repository precondition", "PLAN_REPOSITORY_PRECONDITION_INVALID", {}, D.REPLAN, P.PLANNER, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "repository evidence blocker", "PLANNER_REPOSITORY_EVIDENCE", {}, D.REPLAN, P.PLANNER, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "clean contract mismatch", "AGENT_CONTRACT_MISMATCH", {"clean_contract_mismatch": True}, D.CONTRACT_REPAIR, P.IMPLEMENT_STEP, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "agent timeout", "AGENT_TIMEOUT", {}, D.RETRY_SAME, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("auto", "agent runtime", "AGENT_RUNTIME_FAILED", {}, D.RETRY_SAME, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("auto", "agent runtime after scoped edit", "AGENT_RUNTIME_FAILED", {"tree_changed_in_scope": True}, D.RETRY_AFTER_ROLLBACK, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("auto", "workspace setup transient", "WORKSPACE_SETUP_FAILED", {}, D.RETRY_SAME, P.WORKTREE_SETUP, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
    ("auto", "check preflight transient", "CHECK_PREFLIGHT_FAILED:unit", {}, D.RETRY_SAME, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
    ("auto", "check timeout", "CHECK_TIMEOUT:unit", {}, D.RETRY_SAME, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
    ("auto", "review format", "REVIEW_FORMAT_INVALID", {}, D.CONTRACT_REPAIR, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "review transport", "REVIEWER_TRANSPORT_FAILURE", {}, D.RETRY_SAME, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("auto", "review evidence", "REVIEW_EVIDENCE_RETRY", {}, D.RETRY_SAME, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("auto", "semantic reviser unavailable", "SEMANTIC_REVISER_UNAVAILABLE", {}, D.CONTINUE_WITH_WARNING, P.SEMANTIC_REVISION, None, None),
    ("auto", "optional remote unavailable", "CANDIDATE_REMOTE_UNAVAILABLE", {}, D.CONTINUE_WITH_WARNING, P.CANDIDATE_PUSH, None, None),
    ("auto", "optional push failed", "PUSH_FAILED", {"remote_required": False}, D.CONTINUE_WITH_WARNING, P.CANDIDATE_PUSH, None, None),
    ("model", "ordinary check failure", "CHECK_FAILED:unit", {}, D.CHECK_REPAIR, P.DETERMINISTIC_GATE, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("model", "review IMPLEMENTATION", "REVIEW_IMPLEMENTATION", {}, D.CONTRACT_REPAIR, P.REVIEW_IMPLEMENTATION, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("model", "review REPLAN", "REVIEW_REPLAN", {}, D.REPLAN, P.REVIEW_REPLAN, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("model", "bounded approved scope request", "BOUNDED_SCOPE_REQUEST", {}, D.CONTRACT_REPAIR, P.SEMANTIC_REVISION, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("wait", "auth", "AGENT_AUTH_FAILURE", {}, D.WAIT_EXTERNAL, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("wait", "external auth required", "EXTERNAL_AUTH_REQUIRED", {}, D.WAIT_EXTERNAL, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("wait", "persistent provider outage", "LLM_503", {"budget_exhausted": True}, D.WAIT_EXTERNAL, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("wait", "persistent check infrastructure", "CHECK_INFRASTRUCTURE_UNAVAILABLE", {}, D.WAIT_EXTERNAL, P.DETERMINISTIC_GATE, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
    ("wait", "required remote unavailable", "PUSH_FAILED", {"remote_required": True, "remote_unavailable": True}, D.WAIT_EXTERNAL, P.CANDIDATE_PUSH, RD.WAIT_EXTERNAL, RunStatus.WAITING_REMOTE),
    ("wait", "required publication remote unavailable", "PUSH_FAILED", {"remote_required": True, "remote_unavailable": True}, D.WAIT_EXTERNAL, P.PUBLISH, RD.WAIT_EXTERNAL, RunStatus.WAITING_REMOTE),
    ("wait", "true product decision", "SPEC_DECISION_REQUIRED", {}, D.WAIT_HUMAN, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("wait", "security/policy decision", "SECURITY_POLICY_DECISION_REQUIRED", {}, D.WAIT_HUMAN, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("wait", "check repair exhausted", "CHECK_REPAIR_EXHAUSTED", {}, D.WAIT_HUMAN, P.DETERMINISTIC_GATE, RD.WAIT_HUMAN, RunStatus.WAITING_CHECK_REPAIR),
    ("wait", "review repair exhausted", "WAITING_REPAIR_EXHAUSTED", {}, D.WAIT_HUMAN, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("hard", "unknown failure code", "TOTALLY_NEW_FAILURE", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "secret", "SECRET_IN_DIFF", {}, D.HARD_STOP, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("hard", "unscannable staged source", "UNSCANNABLE_STAGED_BLOB", {}, D.HARD_STOP, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("hard", "scope violation", "AGENT_SCOPE_VIOLATION", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "Git ownership violation", "AGENT_GIT_VIOLATION", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "check authority tampering", "CHECK_AUTHORITY_TAMPERING", {}, D.HARD_STOP, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("hard", "corrupt artifact", "DURABLE_ARTIFACT_CORRUPTED", {}, D.HARD_STOP, P.FINAL_REVIEW, RD.FAILED, RunStatus.FAILED),
    ("hard", "approval mismatch", "PLAN_APPROVAL_IDENTITY_MISMATCH", {}, D.HARD_STOP, P.PLAN_APPROVAL, RD.FAILED, RunStatus.FAILED),
    ("hard", "resume mismatch", "RESUME_IDENTITY_MISMATCH", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "unexplained repository drift", "REPOSITORY_TREE_DRIFT_UNEXPLAINED", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "rollback failure", "ROLLBACK_FAILED", {}, D.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("hard", "rollback not exact", "AGENT_RUNTIME_FAILED", {"rollback_succeeded": False}, D.HARD_STOP, None, None, None),
    ("hard", "out-of-scope tree change", "AGENT_TIMEOUT", {"tree_changed_out_of_scope": True}, D.HARD_STOP, None, None, None),
)
_MODEL_DISPOSITIONS = {D.CONTRACT_REPAIR, D.CHECK_REPAIR, D.REPLAN}
_AUTOMATIC = {
    D.RETRY_SAME, D.RETRY_AFTER_ROLLBACK, D.FALLBACK_EXECUTOR,
    D.CONTINUE_WITH_WARNING, *_MODEL_DISPOSITIONS,
}


class RecoveryMatrixTests(unittest.TestCase):
    def test_each_category_has_its_disposition_and_exit_status(self) -> None:
        for category, name, code, facts, disposition, phase, exit_disposition, status in RECOVERY_MATRIX:
            with self.subTest(category=category, failure=name):
                decision = classify_failure(code, **facts)
                self.assertEqual(decision.disposition, disposition)
                if category in {"wait", "hard"}:
                    self.assertNotIn(decision.disposition, _AUTOMATIC)
                    self.assertFalse(decision.consumes_budget)
                if category == "model":
                    self.assertIn(decision.disposition, _MODEL_DISPOSITIONS)
                if category == "hard":
                    self.assertFalse(decision.rollback_required)
                if phase is None or status is None:
                    continue
                exit_decision, terminal = project_exit(
                    code, phase=phase, remote_required=facts.get("remote_required", False),
                )
                # The durable state is the (phase, disposition) pair; the
                # status is only its derived projection.
                self.assertIs(terminal.phase, phase)
                self.assertIs(terminal.disposition, exit_disposition)
                self.assertEqual(terminal.status, status)
                # A waiting run owns a durable retry boundary only where its
                # phase still owns the operation a retry would run.
                self.assertIs(terminal.resumable, exit_disposition is RD.WAIT_EXTERNAL or (
                    exit_disposition is RD.WAIT_HUMAN and status in {
                        RunStatus.WAITING_CHECK_REPAIR, RunStatus.WAITING_CONTRACT_REPAIR,
                    }
                ))
                self.assertNotIn(exit_decision.disposition, _AUTOMATIC)

    def test_every_automatic_recovery_is_explicitly_allowlisted(self) -> None:
        for code in ("", "UNKNOWN", "AGENT_WEIRD", "CHECK_", "REVIEW_", "PLANNER_", "LLM_"):
            if not code:
                with self.assertRaises(ValueError):
                    classify_failure(code)
                continue
            with self.subTest(code=code):
                self.assertIs(classify_failure(code).disposition, D.HARD_STOP)
                self.assertIs(classify_failure(code, budget_exhausted=True).disposition, D.HARD_STOP)

    def test_a_non_terminal_escape_fails_closed(self) -> None:
        # An optional push is consumed inside its loop; if it escapes anyway
        # the projection never invents a retry or a waiting condition.
        decision, terminal = project_exit("SEMANTIC_REVISER_UNAVAILABLE", phase=P.SEMANTIC_REVISION)
        self.assertIs(decision.disposition, D.HARD_STOP)
        self.assertIs(terminal.status, RunStatus.FAILED)


# (name, code, facts, failure class, next strategy, exit phase, exit disposition, status)
# ``None`` exit: the ladder step is executed inside its loop, never at exit.
LADDER_MATRIX = (
    ("check failed", "CHECK_FAILED:unit", {}, FailureClass.CORRECTNESS, RecoveryStrategy.REPAIR_TARGETED, P.DETERMINISTIC_GATE, None, None),
    ("check repair exhausted", "CHECK_FAILED:unit", {"budget_exhausted": True}, FailureClass.CORRECTNESS, RecoveryStrategy.REPLAN_STEP, P.DETERMINISTIC_GATE, None, None),
    ("check repair with proof", "CHECK_FAILED:unit", {"budget_exhausted": True, "proof_required": True}, FailureClass.CORRECTNESS, RecoveryStrategy.EXPAND_SCOPE, P.DETERMINISTIC_GATE, None, None),
    ("review evidence retry", "REVIEW_EVIDENCE_RETRY", {}, FailureClass.CORRECTNESS, RecoveryStrategy.EXPAND_SCOPE, P.FINAL_REVIEW, None, None),
    ("review evidence unresolved", "REVIEW_EVIDENCE_UNRESOLVED", {}, FailureClass.CORRECTNESS, RecoveryStrategy.EXPAND_SCOPE, P.FINAL_REVIEW, None, None),
    ("clean contract mismatch", "AGENT_CONTRACT_MISMATCH", {"clean_contract_mismatch": True}, FailureClass.CONTRACT, RecoveryStrategy.REPAIR_TARGETED, P.IMPLEMENT_STEP, None, None),
    ("contract mismatch replan", "AGENT_CONTRACT_MISMATCH", {}, FailureClass.CONTRACT, RecoveryStrategy.REPLAN_STEP, P.IMPLEMENT_STEP, None, None),
    ("contract repair exhausted", "AGENT_CONTRACT_MISMATCH", {"clean_contract_mismatch": True, "budget_exhausted": True}, FailureClass.CONTRACT, RecoveryStrategy.REPLAN_STEP, P.IMPLEMENT_STEP, None, None),
    ("planner format", "PLANNER_FORMAT_INVALID", {}, FailureClass.MODEL_PROTOCOL, RecoveryStrategy.RETRY_TARGETED, P.PLANNER, None, None),
    ("planner correction exhausted", "PLANNER_FORMAT_INVALID", {"budget_exhausted": True}, FailureClass.MODEL_PROTOCOL, RecoveryStrategy.WAIT_HUMAN, P.PLANNER, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("transient provider", "LLM_503", {}, FailureClass.EXTERNAL, RecoveryStrategy.RETRY_TARGETED, P.FINAL_REVIEW, None, None),
    ("executor fallback", "AGENT_RUNTIME_FAILED", {"fallback_executor_available": True}, FailureClass.EXTERNAL, RecoveryStrategy.FALLBACK_EXECUTOR, P.IMPLEMENT_STEP, None, None),
    ("persistent provider outage", "LLM_503", {"budget_exhausted": True}, FailureClass.EXTERNAL, RecoveryStrategy.WAIT_EXTERNAL, P.FINAL_REVIEW, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("missing credentials", "AGENT_AUTH_FAILURE", {}, FailureClass.EXTERNAL, RecoveryStrategy.WAIT_EXTERNAL, P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL),
    ("product decision", "SPEC_DECISION_REQUIRED", {}, FailureClass.SPEC_DECISION, RecoveryStrategy.WAIT_HUMAN, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("policy decision", "SECURITY_POLICY_DECISION_REQUIRED", {}, FailureClass.SECURITY, RecoveryStrategy.WAIT_HUMAN, P.FINAL_REVIEW, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("secret in diff", "SECRET_IN_DIFF", {}, FailureClass.SECURITY, RecoveryStrategy.HARD_STOP, P.DETERMINISTIC_GATE, RD.FAILED, RunStatus.FAILED),
    ("tree mismatch", "TREE_MISMATCH", {}, FailureClass.INTEGRITY, RecoveryStrategy.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("scope violation", "AGENT_SCOPE_VIOLATION", {}, FailureClass.AUTHORITY, RecoveryStrategy.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
    ("scope approval required", "REPAIR_SCOPE_APPROVAL_REQUIRED", {}, FailureClass.AUTHORITY, RecoveryStrategy.WAIT_HUMAN, P.SEMANTIC_REVISION, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN),
    ("unknown failure code", "TOTALLY_NEW_FAILURE", {}, FailureClass.UNKNOWN, RecoveryStrategy.HARD_STOP, P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED),
)


class LadderMatrixTests(unittest.TestCase):
    """Each ladder row states its class, its next step and its exit status."""

    def test_each_row_exposes_its_class_and_next_strategy(self) -> None:
        for name, code, facts, failure_class, strategy, phase, exit_disposition, status in LADDER_MATRIX:
            with self.subTest(failure=name):
                decision = classify_failure(code, **facts)
                self.assertIs(decision.failure_class, failure_class)
                self.assertIs(decision.strategy, strategy)
                self.assertIn(strategy, recovery_ladder(failure_class))
                if status is None:
                    with self.assertRaises(ValueError):
                        strategy_terminal_state(strategy, failure_code=code, phase=phase)
                    continue
                terminal = strategy_terminal_state(strategy, failure_code=code, phase=phase)
                self.assertIs(terminal.disposition, exit_disposition)
                self.assertEqual(terminal.status, status)
                # The ladder terminal and the disposition projection agree on
                # the durable state and the status derived from it.
                projected = terminal_state_for(decision, failure_code=code, phase=phase)
                self.assertEqual(
                    (projected.phase, projected.disposition, projected.status),
                    (terminal.phase, terminal.disposition, terminal.status),
                )

    def test_check_failed_walks_its_ladder_before_a_human_wait(self) -> None:
        tree = "a" * 40
        facts = RecoveryFacts(
            candidate_tree=tree, proof_required=True, fallback_executor_available=True,
        )
        progression = RecoveryProgression()
        walked = []
        while True:
            strategy = progression.next_strategy(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, code="CHECK_FAILED:unit",
            )
            walked.append(strategy)
            if strategy.terminal:
                break
            progression.consume(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=strategy,
            )
        self.assertEqual(walked, [
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
            RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
            RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.WAIT_HUMAN,
        ])
        terminal = strategy_terminal_state(
            walked[-1], failure_code="CHECK_FAILED:unit", phase=P.DETERMINISTIC_GATE,
        )
        self.assertEqual(
            (terminal.disposition, terminal.status, terminal.resumable),
            (RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN, False),
        )
        # The same fingerprint never proposes a consumed step again.
        self.assertIs(
            progression.next_strategy(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, code="CHECK_FAILED:unit",
            ),
            RecoveryStrategy.WAIT_HUMAN,
        )

    def test_boundary_classes_never_expose_an_autonomous_step(self) -> None:
        for code in ("SECRET_IN_DIFF", "TREE_MISMATCH", "AGENT_SCOPE_VIOLATION"):
            with self.subTest(code=code):
                decision = classify_failure(code)
                self.assertTrue(all(step.terminal for step in recovery_ladder(decision.failure_class)))
                self.assertTrue(decision.strategy.terminal)


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
        self.store.update(status=state["status"], resume={"attempts": 3})
        # A fresh coordinator after a resume sees the same consumed budget.
        second = self.admit(self.coordinator())
        self.assertTrue(second.admitted)
        self.assertEqual(second.used, 2)
        exhausted = self.admit(self.coordinator())
        self.assertFalse(exhausted.admitted)
        self.assertTrue(exhausted.exhausted)
        self.assertIs(exhausted.decision.disposition, D.WAIT_EXTERNAL)
        self.assertEqual(self.store.load()["recovery_counters"], {"agent-step:001:S01": 2})
        data = self.events[-1][1]["data"]
        self.assertEqual(self.events[-1][0], "recovery.exhausted")
        self.assertEqual(data["terminal_status"], RunStatus.WAITING_EXTERNAL.value)
        self.assertEqual(data["checkpoint_phase"], P.IMPLEMENT_STEP.value)

    def test_each_consumed_attempt_has_a_durable_identity(self) -> None:
        self.admit(self.coordinator(), tree_after="b" * 40)
        (record,) = self.store.load()["recovery_attempts"]
        self.assertEqual(record, {
            "phase": "implementation", "reason": "AGENT_TIMEOUT", "attempt": 1,
            "budget_key": "agent-step:001:S01", "budget": 2, "budget_consumed": 1,
            "disposition": "retry_same", "cycle": 1, "step_id": "S01",
            "operation_id": "recovery:agent-step:001:S01:01",
            "profile_id": "worker", "tree_before": "a" * 40, "tree_after": "b" * 40,
        })

    def test_disallowed_disposition_consumes_nothing(self) -> None:
        refused = self.admit(
            self.coordinator(), reason="AGENT_AUTH_FAILURE", allowed={D.RETRY_SAME},
        )
        self.assertFalse(refused.admitted)
        self.assertFalse(refused.exhausted)
        self.assertEqual(self.store.load()["recovery_counters"], {})
        self.assertNotIn("recovery_attempts", self.store.load())

    def test_malformed_counters_fail_closed(self) -> None:
        state = self.store.load()
        self.store.update(status=state["status"], recovery_counters={"agent-step:001:S01": -1})
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

    def test_out_of_scope_mutation_is_an_authority_violation(self) -> None:
        transaction = self.begin()
        (self.repo / "b.txt").write_text("outside\n", encoding="utf-8")
        with self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertEqual(caught.exception.code, "AGENT_SCOPE_VIOLATION")
        self.assertIs(classify_failure(caught.exception.code).disposition, D.HARD_STOP)

    def test_secret_in_a_failed_attempt_is_never_silently_rolled_back(self) -> None:
        transaction = self.begin(secrets=("sk-live-secret-value-123456",))
        (self.repo / "a.txt").write_text("key=sk-live-secret-value-123456\n", encoding="utf-8")
        with self.assertRaises(AttemptViolation) as caught:
            transaction.abort({"a.txt"})
        self.assertTrue(caught.exception.code.startswith("SECRET_"), caught.exception.code)
        self.assertIs(classify_failure(caught.exception.code).disposition, D.HARD_STOP)

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
        self.assertIs(
            classify_failure(caught.exception.code, rollback_succeeded=False).disposition,
            D.HARD_STOP,
        )

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
        ("TOTALLY_NEW_FAILURE", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("SECRET_IN_DIFF", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("AGENT_SCOPE_VIOLATION", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("ROLLBACK_FAILED", P.IMPLEMENT_STEP, RD.FAILED, RunStatus.FAILED, False),
        ("AGENT_AUTH_FAILURE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("CHECK_INFRASTRUCTURE_UNAVAILABLE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE, True),
        ("REVIEWER_TRANSPORT_FAILURE", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
        ("SPEC_DECISION_REQUIRED", P.IMPLEMENT_STEP, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN, False),
        # An escaped automatic recovery is never re-authorized at the exit.
        ("REVIEW_REPLAN", P.IMPLEMENT_STEP, RD.WAIT_HUMAN, RunStatus.WAITING_HUMAN, False),
        ("AGENT_TIMEOUT", P.IMPLEMENT_STEP, RD.WAIT_EXTERNAL, RunStatus.WAITING_EXTERNAL, True),
    )

    def test_escaped_failures_project_without_calling_models(self) -> None:
        for index, (code, phase, disposition, status, resumable) in enumerate(self.PATHS):
            run_id = f"path-{index}"
            with self.subTest(code=code), mock.patch(
                "metaharness.orchestrator.PipelineV2Coordinator.run",
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
            "metaharness.orchestrator.PipelineV2Coordinator.run",
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
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        result = self.orchestrator(
            self.config(check_repair=0), planner=[initial_plan(STEP)], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.checkpoint()["phase"], P.DETERMINISTIC_GATE.value)
        self.assertEqual(self.workers.roles(), ["implementer"])
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
