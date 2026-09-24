from __future__ import annotations

import unittest

from metaharness.recovery_policy import (
    RecoveryBudgets,
    RecoveryDisposition,
    classify_failure,
)
from metaharness.orchestration.recovery import terminal_state_for
from metaharness.models import RunStatus
from metaharness.resume import ResumePhase


# One central architecture table covers all pipeline outcomes. Changes to the
# classifier must update this policy invariant and its path-level tests.
FAILURE_POLICY_MATRIX = (
    ("planner format", "PLANNER_FORMAT_INVALID", RecoveryDisposition.REPLAN, {}),
    ("planner topology", "PLAN_REPOSITORY_PRECONDITION_INVALID", RecoveryDisposition.REPLAN, {}),
    ("planner repository evidence blocker", "PLANNER_REPOSITORY_EVIDENCE", RecoveryDisposition.REPLAN, {}),
    ("contract mismatch", "AGENT_CONTRACT_MISMATCH", RecoveryDisposition.CONTRACT_REPAIR, {"clean_contract_mismatch": True}),
    ("agent timeout", "AGENT_TIMEOUT", RecoveryDisposition.RETRY_SAME, {}),
    ("agent runtime", "AGENT_RUNTIME_FAILED", RecoveryDisposition.RETRY_SAME, {}),
    ("workspace setup timeout", "WORKSPACE_SETUP_TIMEOUT", RecoveryDisposition.RETRY_SAME, {}),
    ("check preflight", "CHECK_PREFLIGHT_FAILED", RecoveryDisposition.RETRY_SAME, {}),
    ("check timeout", "CHECK_TIMEOUT", RecoveryDisposition.RETRY_SAME, {}),
    ("review format", "REVIEW_FORMAT_INVALID", RecoveryDisposition.CONTRACT_REPAIR, {}),
    ("review transport", "REVIEWER_TRANSPORT_FAILURE", RecoveryDisposition.RETRY_SAME, {}),
    ("semantic reviser unavailable", "SEMANTIC_REVISER_UNAVAILABLE", RecoveryDisposition.CONTINUE_WITH_WARNING, {}),
    ("candidate staging remote unavailable", "CANDIDATE_REMOTE_UNAVAILABLE", RecoveryDisposition.CONTINUE_WITH_WARNING, {}),
    ("ordinary check failed", "CHECK_FAILED:unit", RecoveryDisposition.CHECK_REPAIR, {}),
    ("review IMPLEMENTATION", "REVIEW_IMPLEMENTATION", RecoveryDisposition.CONTRACT_REPAIR, {}),
    ("review REPLAN", "REVIEW_REPLAN", RecoveryDisposition.REPLAN, {}),
    ("bounded scope request", "BOUNDED_SCOPE_REQUEST", RecoveryDisposition.CONTRACT_REPAIR, {}),
    ("secret", "SECRET_IN_DIFF", RecoveryDisposition.HARD_STOP, {}),
    ("out-of-scope mutation", "AGENT_SCOPE_VIOLATION", RecoveryDisposition.HARD_STOP, {}),
    ("Git ownership mutation", "AGENT_GIT_VIOLATION", RecoveryDisposition.HARD_STOP, {}),
    ("corrupt durable artifact", "DURABLE_ARTIFACT_CORRUPTED", RecoveryDisposition.HARD_STOP, {}),
    ("approval identity mismatch", "PLAN_APPROVAL_IDENTITY_MISMATCH", RecoveryDisposition.HARD_STOP, {}),
    ("resume identity mismatch", "RESUME_IDENTITY_MISMATCH", RecoveryDisposition.HARD_STOP, {}),
    ("true SPEC decision", "SPEC_DECISION_REQUIRED", RecoveryDisposition.WAIT_HUMAN, {}),
    ("security policy decision", "SECURITY_POLICY_DECISION_REQUIRED", RecoveryDisposition.WAIT_HUMAN, {}),
    ("scope configured require-approval", "REPAIR_SCOPE_APPROVAL_REQUIRED", RecoveryDisposition.WAIT_HUMAN, {}),
)


class RecoveryPolicyTests(unittest.TestCase):
    def test_unknown_failure_fails_closed(self) -> None:
        decision = classify_failure("TOTALLY_NEW_FAILURE")
        self.assertEqual(decision.disposition, RecoveryDisposition.HARD_STOP)
        self.assertFalse(decision.consumes_budget)
        self.assertEqual(
            terminal_state_for(decision, failure_code="TOTALLY_NEW_FAILURE", phase=ResumePhase.IMPLEMENT_STEP).status,
            RunStatus.FAILED,
        )

    def test_exhausted_outcomes_have_distinct_waiting_states(self) -> None:
        cases = (
            ("AGENT_AUTH_FAILURE", {}, ResumePhase.IMPLEMENT_STEP, RunStatus.WAITING_EXTERNAL),
            ("LLM_503", {"budget_exhausted": True}, ResumePhase.FINAL_REVIEW, RunStatus.WAITING_EXTERNAL),
            ("CHECK_TIMEOUT", {"budget_exhausted": True}, ResumePhase.DETERMINISTIC_GATE, RunStatus.WAITING_CHECK_INFRASTRUCTURE),
            ("PUSH_FAILED", {"remote_required": True, "budget_exhausted": True}, ResumePhase.CANDIDATE_PUSH, RunStatus.WAITING_REMOTE),
            ("CHECK_REPAIR_EXHAUSTED", {}, ResumePhase.DETERMINISTIC_GATE, RunStatus.WAITING_HUMAN),
            ("PLAN_REPOSITORY_PRECONDITION_INVALID", {"budget_exhausted": True}, ResumePhase.PLANNER, RunStatus.WAITING_HUMAN),
        )
        for code, options, phase, status in cases:
            with self.subTest(code=code):
                decision = classify_failure(code, **options)
                self.assertEqual(terminal_state_for(decision, failure_code=code, phase=phase).status, status)

    def test_failure_policy_matrix_is_an_architectural_invariant(self) -> None:
        for name, reason, expected, kwargs in FAILURE_POLICY_MATRIX:
            with self.subTest(policy=name):
                self.assertEqual(classify_failure(reason, **kwargs).disposition, expected)

    def test_agent_timeout_retries_same_executor(self) -> None:
        decision = classify_failure("AGENT_TIMEOUT")
        self.assertEqual(decision.disposition, RecoveryDisposition.RETRY_SAME)
        self.assertTrue(decision.consumes_budget)

    def test_runtime_failure_with_scoped_tree_change_requires_rollback(self) -> None:
        decision = classify_failure(
            "AGENT_RUNTIME_FAILED", tree_changed_in_scope=True,
        )
        self.assertEqual(decision.disposition, RecoveryDisposition.RETRY_AFTER_ROLLBACK)
        self.assertTrue(decision.rollback_required)

    def test_authority_and_security_failures_stop(self) -> None:
        for reason in (
            "AGENT_SCOPE_VIOLATION", "SECRET_IN_DIFF", "RESUME_INTEGRITY_FAILURE",
            "CHECK_AUTHORITY_TAMPERING", "STAGED_BLOB_SCAN_FAILED",
            "UNSCANNABLE_STAGED_BLOB", "UNREVIEWABLE_TEXT_DIFF",
            "SECRET_IN_STAGED_BLOB", "AGENT_GIT_VIOLATION",
            "ROLLBACK_FAILED", "DURABLE_ARTIFACT_CORRUPTED",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(
                    classify_failure(reason).disposition,
                    RecoveryDisposition.HARD_STOP,
                )

    def test_checks_repair_only_test_failures_and_retry_infrastructure(self) -> None:
        self.assertEqual(
            classify_failure("CHECK_FAILED:unit").disposition,
            RecoveryDisposition.CHECK_REPAIR,
        )
        for reason in ("CHECK_TIMEOUT:unit", "CHECK_PREFLIGHT_FAILED:unit"):
            with self.subTest(reason=reason):
                self.assertEqual(
                    classify_failure(reason).disposition,
                    RecoveryDisposition.RETRY_SAME,
                )
        for reason in (
            "CHECK_INFRASTRUCTURE_UNAVAILABLE:unit",
            "CHECK_SIDE_EFFECT_REPEATED:formatter",
        ):
            with self.subTest(reason=reason):
                self.assertEqual(
                    classify_failure(reason).disposition,
                    RecoveryDisposition.WAIT_EXTERNAL,
                )
        self.assertEqual(
            classify_failure("CHECK_TIMEOUT", budget_exhausted=True).disposition,
            RecoveryDisposition.WAIT_EXTERNAL,
        )

    def test_clean_contract_and_review_format_failures_are_recoverable(self) -> None:
        self.assertEqual(
            classify_failure(
                "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True,
            ).disposition,
            RecoveryDisposition.CONTRACT_REPAIR,
        )
        self.assertEqual(
            classify_failure("REVIEW_PARSE_INVALID").disposition,
            RecoveryDisposition.CONTRACT_REPAIR,
        )

    def test_push_depends_on_remote_authority(self) -> None:
        self.assertEqual(
            classify_failure("PUSH_FAILED", remote_required=False).disposition,
            RecoveryDisposition.CONTINUE_WITH_WARNING,
        )
        self.assertEqual(
            classify_failure("PUSH_FAILED", remote_required=True).disposition,
            RecoveryDisposition.RETRY_SAME,
        )
        self.assertEqual(
            classify_failure(
                "PUSH_FAILED", remote_required=True, remote_unavailable=True,
            ).disposition,
            RecoveryDisposition.WAIT_EXTERNAL,
        )

    def test_authentication_waits_for_external_change(self) -> None:
        for reason in ("AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "MISSING_PROVIDER_CREDENTIALS"):
            with self.subTest(reason=reason):
                decision = classify_failure(reason)
                self.assertEqual(decision.disposition, RecoveryDisposition.WAIT_EXTERNAL)
                self.assertFalse(decision.consumes_budget)

    def test_protocol_planning_workspace_and_transport_failures_are_bounded(self) -> None:
        for reason in (
            "AGENT_START_FAILED", "AGENT_PROTOCOL_FAILED", "LLM_429",
            "LLM_5XX", "LLM_TIMEOUT", "WORKSPACE_SETUP_FAILED",
            "WORKSPACE_SETUP_TIMEOUT", "CHECK_PREFLIGHT_FAILED",
        ):
            with self.subTest(reason=reason):
                self.assertTrue(classify_failure(reason).consumes_budget)
        self.assertEqual(
            classify_failure("PLAN_REPOSITORY_PRECONDITION_INVALID").disposition,
            RecoveryDisposition.REPLAN,
        )
        self.assertEqual(
            classify_failure("CANDIDATE_REMOTE_UNAVAILABLE").disposition,
            RecoveryDisposition.CONTINUE_WITH_WARNING,
        )

    def test_budgets_have_bounded_durable_defaults(self) -> None:
        self.assertEqual(RecoveryBudgets(), RecoveryBudgets(
            max_transient_attempts=2,
            max_executor_fallbacks=1,
            max_check_infra_retries=2,
            max_review_transport_retries=2,
            max_workspace_setup_retries=2,
        ))
        with self.assertRaises(ValueError):
            RecoveryBudgets(max_transient_attempts=11)


if __name__ == "__main__":
    unittest.main()
