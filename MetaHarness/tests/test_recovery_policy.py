from __future__ import annotations

import unittest

from metaharness.recovery_policy import (
    RecoveryBudgets,
    RecoveryDisposition,
    classify_failure,
)


class RecoveryPolicyTests(unittest.TestCase):
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
