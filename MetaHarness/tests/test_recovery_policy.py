from __future__ import annotations

import ast
import inspect
import re
import unittest

from metaharness import recovery_policy
from metaharness.models import RunStatus
from metaharness.orchestration.recovery import terminal_state_for
from metaharness.recovery_policy import (
    FailureClass,
    RecoveryBudgets,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryFacts,
    RecoveryFingerprint,
    RecoveryProgression,
    RecoveryStrategy,
    classify_failure,
    failure_class_for,
    recovery_ladder,
    terminal_strategy,
)
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
    ("check repair fixed point", "CHECK_REPAIR_FIXED_POINT", RecoveryDisposition.WAIT_HUMAN, {}),
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
    def test_invalid_contract_repair_output_is_never_a_worker_mismatch(self) -> None:
        code = "STEP_CONTRACT_REPAIR_OUTPUT_INVALID"
        self.assertEqual(classify_failure(code).disposition, RecoveryDisposition.CONTRACT_REPAIR)
        exhausted = classify_failure(code, budget_exhausted=True)
        self.assertEqual(exhausted.disposition, RecoveryDisposition.WAIT_HUMAN)
        terminal = terminal_state_for(exhausted, failure_code=code, phase=ResumePhase.IMPLEMENT_STEP)
        self.assertEqual((terminal.status, terminal.resumable), (RunStatus.WAITING_CONTRACT_REPAIR, True))
        # A genuine decision keeps the non-resumable WAITING_HUMAN projection.
        for genuine in ("SPEC_DECISION_REQUIRED", "SECURITY_POLICY_DECISION_REQUIRED", "CONTRACT_REPAIR_SCOPE_DENIED"):
            decision = classify_failure(genuine, budget_exhausted=True)
            terminal = terminal_state_for(decision, failure_code=genuine, phase=ResumePhase.IMPLEMENT_STEP)
            self.assertEqual((terminal.status, terminal.resumable), (RunStatus.WAITING_HUMAN, False))

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
            ("CHECK_REPAIR_EXHAUSTED", {}, ResumePhase.DETERMINISTIC_GATE, RunStatus.WAITING_CHECK_REPAIR),
            ("PLAN_REPOSITORY_PRECONDITION_INVALID", {"budget_exhausted": True}, ResumePhase.PLANNER, RunStatus.WAITING_HUMAN),
        )
        for code, options, phase, status in cases:
            with self.subTest(code=code):
                decision = classify_failure(code, **options)
                self.assertEqual(terminal_state_for(decision, failure_code=code, phase=phase).status, status)
        exhausted = terminal_state_for(
            classify_failure("CHECK_REPAIR_EXHAUSTED"),
            failure_code="CHECK_REPAIR_EXHAUSTED",
            phase=ResumePhase.DETERMINISTIC_GATE,
        )
        self.assertTrue(exhausted.resumable)

    def test_failure_policy_matrix_is_an_architectural_invariant(self) -> None:
        for name, reason, expected, kwargs in FAILURE_POLICY_MATRIX:
            with self.subTest(policy=name):
                self.assertEqual(classify_failure(reason, **kwargs).disposition, expected)

    def test_agent_timeout_retries_same_executor(self) -> None:
        decision = classify_failure("AGENT_TIMEOUT")
        self.assertEqual(decision.disposition, RecoveryDisposition.RETRY_SAME)
        self.assertTrue(decision.consumes_budget)

    def test_check_repair_fixed_point_requires_human_and_does_not_resume(self) -> None:
        decision = classify_failure("CHECK_REPAIR_FIXED_POINT")
        terminal = terminal_state_for(
            decision, failure_code="CHECK_REPAIR_FIXED_POINT",
            phase=ResumePhase.DETERMINISTIC_GATE,
        )
        self.assertEqual((terminal.status, terminal.resumable), (RunStatus.WAITING_HUMAN, False))
        self.assertIn("code change or additional repair authority", decision.reason)

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


def _policy_codes() -> tuple[str, ...]:
    """Every stable failure code literal the policy module decides on.

    Literals that are only ``startswith`` patterns, that end with ``_`` or that
    are a strict prefix of another literal are patterns, not codes.
    """

    tree = ast.parse(inspect.getsource(recovery_policy))
    patterns: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"startswith", "endswith"}
        ):
            for arg in node.args:
                for item in ast.walk(arg):
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        patterns.add(item.value)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in patterns
    }
    codes = tuple(sorted(
        value for value in literals if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", value)
    ))
    return tuple(
        code for code in codes
        if not code.endswith("_")
        and not any(other != code and other.startswith(code) for other in codes)
    )


class RecoveryLadderTests(unittest.TestCase):
    """The autonomy contract: one class per code, one ladder per class."""

    def test_every_class_owns_one_documented_ladder(self) -> None:
        self.assertEqual(recovery_ladder(FailureClass.CORRECTNESS), (
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
            RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
            RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.WAIT_HUMAN,
        ))
        self.assertEqual(recovery_ladder(FailureClass.CONTRACT), (
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.REPLAN_STEP,
            RecoveryStrategy.REPLAN_CYCLE, RecoveryStrategy.FALLBACK_EXECUTOR,
            RecoveryStrategy.WAIT_HUMAN,
        ))
        self.assertEqual(recovery_ladder(FailureClass.MODEL_PROTOCOL), (
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.RETRY_TARGETED,
            RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.WAIT_HUMAN,
        ))
        self.assertEqual(recovery_ladder(FailureClass.EXTERNAL), (
            RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
            RecoveryStrategy.WAIT_EXTERNAL,
        ))
        self.assertEqual(recovery_ladder(FailureClass.SPEC_DECISION), (
            RecoveryStrategy.WAIT_HUMAN,
        ))
        for failure_class in FailureClass:
            with self.subTest(failure_class=failure_class):
                ladder = recovery_ladder(failure_class)
                self.assertTrue(ladder)
                self.assertTrue(ladder[-1].terminal)

    def test_boundary_classes_only_offer_a_hard_stop_or_a_human_wait(self) -> None:
        for failure_class in (
            FailureClass.SECURITY, FailureClass.INTEGRITY, FailureClass.AUTHORITY,
        ):
            with self.subTest(failure_class=failure_class):
                self.assertEqual(
                    set(recovery_ladder(failure_class)),
                    {RecoveryStrategy.HARD_STOP, RecoveryStrategy.WAIT_HUMAN},
                )
                self.assertEqual(
                    [step for step in recovery_ladder(failure_class) if not step.terminal], [],
                )
        self.assertEqual(recovery_ladder(FailureClass.UNKNOWN), (RecoveryStrategy.HARD_STOP,))
        # Only the authority tables already in force pick the boundary terminal.
        self.assertIs(terminal_strategy(FailureClass.SECURITY, "SECRET_IN_DIFF"), RecoveryStrategy.HARD_STOP)
        self.assertIs(terminal_strategy(FailureClass.INTEGRITY, "TREE_MISMATCH"), RecoveryStrategy.HARD_STOP)
        self.assertIs(terminal_strategy(FailureClass.AUTHORITY, "AGENT_SCOPE_VIOLATION"), RecoveryStrategy.HARD_STOP)
        self.assertIs(
            terminal_strategy(FailureClass.SECURITY, "SECURITY_POLICY_DECISION_REQUIRED"),
            RecoveryStrategy.WAIT_HUMAN,
        )
        self.assertIs(
            terminal_strategy(FailureClass.AUTHORITY, "REPAIR_SCOPE_APPROVAL_REQUIRED"),
            RecoveryStrategy.WAIT_HUMAN,
        )
        self.assertIs(
            terminal_strategy(FailureClass.UNKNOWN, "TOTALLY_NEW_FAILURE"),
            RecoveryStrategy.HARD_STOP,
        )

    def test_failure_classes_come_from_stable_codes_only(self) -> None:
        cases = (
            ("CHECK_FAILED:unit", FailureClass.CORRECTNESS),
            ("CHECK_REPAIR_FIXED_POINT", FailureClass.CORRECTNESS),
            ("REVIEW_IMPLEMENTATION", FailureClass.CORRECTNESS),
            ("AGENT_CONTRACT_MISMATCH", FailureClass.CONTRACT),
            ("CONTRACT_INSUFFICIENCY", FailureClass.CONTRACT),
            ("PLAN_REPOSITORY_PRECONDITION_INVALID", FailureClass.CONTRACT),
            ("PLANNER_FORMAT_INVALID", FailureClass.MODEL_PROTOCOL),
            ("STEP_CONTRACT_REPAIR_OUTPUT_INVALID", FailureClass.MODEL_PROTOCOL),
            ("REVIEW_PARSE_TRUNCATED", FailureClass.MODEL_PROTOCOL),
            ("LLM_429", FailureClass.EXTERNAL),
            ("AGENT_TIMEOUT", FailureClass.EXTERNAL),
            ("AGENT_AUTH_FAILURE", FailureClass.EXTERNAL),
            ("CHECK_INFRASTRUCTURE_UNAVAILABLE", FailureClass.EXTERNAL),
            ("WORKSPACE_SETUP_TIMEOUT", FailureClass.EXTERNAL),
            ("SPEC_DECISION_REQUIRED", FailureClass.SPEC_DECISION),
            ("REVIEW_HUMAN_REQUIRED", FailureClass.SPEC_DECISION),
            ("SECRET_IN_STAGED_BLOB", FailureClass.SECURITY),
            ("DURABLE_ARTIFACT_CORRUPTED", FailureClass.INTEGRITY),
            ("REPOSITORY_TREE_DRIFT_UNEXPLAINED", FailureClass.INTEGRITY),
            ("AGENT_SCOPE_VIOLATION", FailureClass.AUTHORITY),
            ("CHECK_AUTHORITY_TAMPERING", FailureClass.AUTHORITY),
            ("RESUME_IDENTITY_MISMATCH", FailureClass.AUTHORITY),
            # A pipeline code the policy never classified stays unknown, which
            # fails closed instead of inventing autonomy.
            ("REPLAN_CYCLE_REQUIRED", FailureClass.UNKNOWN),
            ("TOCTOU_FAILURE", FailureClass.UNKNOWN),
        )
        for code, expected in cases:
            with self.subTest(code=code):
                self.assertIs(failure_class_for(code), expected)
        with self.assertRaises(ValueError):
            failure_class_for("  ")

    def test_check_failed_exposes_the_next_ladder_step(self) -> None:
        first = classify_failure("CHECK_FAILED:unit")
        self.assertIs(first.failure_class, FailureClass.CORRECTNESS)
        self.assertIs(first.strategy, RecoveryStrategy.REPAIR_TARGETED)
        # Exhaustion no longer collapses onto a flat operator wait: it exposes
        # the following step of the same ladder.
        exhausted = classify_failure("CHECK_FAILED:unit", budget_exhausted=True)
        self.assertIs(exhausted.disposition, RecoveryDisposition.WAIT_HUMAN)
        self.assertIs(exhausted.strategy, RecoveryStrategy.REPLAN_STEP)
        proven = classify_failure(
            "CHECK_FAILED:unit", budget_exhausted=True, proof_required=True,
        )
        self.assertIs(proven.strategy, RecoveryStrategy.EXPAND_SCOPE)
        fixed_point = classify_failure("CHECK_REPAIR_FIXED_POINT")
        self.assertIs(fixed_point.strategy, RecoveryStrategy.REPLAN_STEP)
        for decision in (first, exhausted, proven, fixed_point):
            self.assertIn(decision.strategy, recovery_ladder(decision.failure_class))

    def test_contract_and_protocol_exhaustion_keep_their_own_ladder(self) -> None:
        contract = classify_failure(
            "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True,
        )
        self.assertIs(contract.failure_class, FailureClass.CONTRACT)
        self.assertIs(contract.strategy, RecoveryStrategy.REPAIR_TARGETED)
        exhausted = classify_failure(
            "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True, budget_exhausted=True,
        )
        self.assertIs(exhausted.strategy, RecoveryStrategy.REPLAN_STEP)
        protocol = classify_failure("STEP_CONTRACT_REPAIR_OUTPUT_INVALID")
        self.assertIs(protocol.failure_class, FailureClass.MODEL_PROTOCOL)
        self.assertIs(protocol.strategy, RecoveryStrategy.REPAIR_TARGETED)
        # Format correction and a fresh completion are spent; nothing remains
        # without a configured fallback profile.
        spent = classify_failure("STEP_CONTRACT_REPAIR_OUTPUT_INVALID", budget_exhausted=True)
        self.assertIs(spent.strategy, RecoveryStrategy.WAIT_HUMAN)

    def test_external_failures_retry_then_wait_and_never_retry_credentials(self) -> None:
        transient = classify_failure("LLM_429")
        self.assertIs(transient.failure_class, FailureClass.EXTERNAL)
        self.assertIs(transient.strategy, RecoveryStrategy.RETRY_TARGETED)
        self.assertIs(
            classify_failure("AGENT_TIMEOUT", budget_exhausted=True).strategy,
            RecoveryStrategy.WAIT_EXTERNAL,
        )
        for code in (
            "AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "MISSING_PROVIDER_CREDENTIALS",
            "EXTERNAL_AUTH_REQUIRED",
        ):
            with self.subTest(code=code):
                self.assertIs(classify_failure(code).strategy, RecoveryStrategy.WAIT_EXTERNAL)
                self.assertFalse(RecoveryFacts.for_failure(code).retry_allowed)

    def test_a_spec_decision_never_authorizes_an_autonomous_step(self) -> None:
        decision = classify_failure("SPEC_DECISION_REQUIRED")
        self.assertIs(decision.failure_class, FailureClass.SPEC_DECISION)
        self.assertIs(decision.strategy, RecoveryStrategy.WAIT_HUMAN)
        self.assertEqual(
            [step for step in recovery_ladder(FailureClass.SPEC_DECISION) if not step.terminal],
            [],
        )

    def test_a_consumed_strategy_is_never_proposed_twice(self) -> None:
        tree = "a" * 40
        facts = RecoveryFacts(proof_required=True)
        progression = RecoveryProgression()
        seen = []
        for expected in (
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
            RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
            RecoveryStrategy.WAIT_HUMAN,
        ):
            strategy = progression.next_strategy(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS, facts=facts,
            )
            self.assertIs(strategy, expected)
            seen.append(strategy)
            if not strategy.terminal:
                progression.consume(
                    candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                    facts=facts, strategy=strategy,
                )
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(progression.consumed), 4)
        # A consumed step is refused, and a terminal is never consumed at all.
        with self.assertRaises(ValueError):
            progression.consume(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=RecoveryStrategy.REPAIR_TARGETED,
            )
        with self.assertRaises(ValueError):
            progression.consume(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=RecoveryStrategy.WAIT_HUMAN,
            )

    def test_a_new_tree_or_new_facts_open_a_new_progression(self) -> None:
        tree = "a" * 40
        facts = RecoveryFacts(proof_required=True)
        progression = RecoveryProgression()
        consumed = progression.consume(
            candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
            facts=facts, strategy=RecoveryStrategy.REPAIR_TARGETED,
        )
        self.assertTrue(progression.is_consumed(consumed))
        self.assertIs(
            progression.next_strategy(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS, facts=facts,
            ),
            RecoveryStrategy.EXPAND_SCOPE,
        )
        for candidate_tree, failure_class, other_facts in (
            ("b" * 40, FailureClass.CORRECTNESS, facts),
            (tree, FailureClass.CONTRACT, facts),
            (tree, FailureClass.CORRECTNESS, RecoveryFacts(proof_required=True, retry_allowed=True)),
            # A different observed failure fact is a different fingerprint.
            (
                tree, FailureClass.CORRECTNESS,
                RecoveryFacts(proof_required=True, observed_facts={"failed_checks": "unit"}),
            ),
        ):
            with self.subTest(tree=candidate_tree, failure_class=failure_class, facts=other_facts):
                self.assertIs(
                    progression.next_strategy(
                        candidate_tree=candidate_tree, failure_class=failure_class,
                        facts=other_facts,
                    ),
                    RecoveryStrategy.REPAIR_TARGETED,
                )
        self.assertEqual(len(progression.consumed), 1)

    def test_a_fingerprint_canonicalizes_its_stable_facts(self) -> None:
        ordered = RecoveryFingerprint(
            "a" * 40, FailureClass.CORRECTNESS,
            (("proof_required", "true"), ("retry_allowed", "false")),
            RecoveryStrategy.EXPAND_SCOPE,
        )
        shuffled = RecoveryFingerprint(
            "a" * 40, "correctness",
            {"retry_allowed": "false", "proof_required": "true"}, "expand_scope",
        )
        self.assertEqual(ordered, shuffled)
        self.assertEqual(len({ordered, shuffled}), 1)
        self.assertEqual(
            ordered.stable_failure_facts,
            (("proof_required", "true"), ("retry_allowed", "false")),
        )
        self.assertEqual(
            RecoveryFacts(proof_required=True).stable_items(),
            (
                ("fallback_executor_available", "false"),
                ("proof_required", "true"), ("retry_allowed", "false"),
            ),
        )
        self.assertEqual(
            RecoveryFacts(
                proof_required=True, observed_facts={"failed_checks": "unit"},
            ).stable_items(),
            (
                ("failed_checks", "unit"), ("fallback_executor_available", "false"),
                ("proof_required", "true"), ("retry_allowed", "false"),
            ),
        )
        with self.assertRaises(ValueError):
            RecoveryFingerprint(
                "a" * 40, FailureClass.CORRECTNESS,
                (("proof_required", "true"), ("proof_required", "true")),
                RecoveryStrategy.EXPAND_SCOPE,
            )
        with self.assertRaises(TypeError):
            RecoveryFacts(proof_required="yes")
        with self.assertRaises(ValueError):
            RecoveryFacts(candidate_tree="a" * 200)
        # An observed fact never shadows the ladder facts of one occurrence.
        with self.assertRaises(ValueError):
            RecoveryFacts(observed_facts={"proof_required": "true"})
        with self.assertRaises(TypeError):
            RecoveryFacts(observed_facts=42)
        with self.assertRaises(ValueError):
            RecoveryFacts(observed_facts=["unit"])

    def test_a_decision_refuses_a_foreign_ladder_vocabulary(self) -> None:
        with self.assertRaises(TypeError):
            RecoveryDecision(
                RecoveryDisposition.CHECK_REPAIR, "deterministic check failed",
                True, False, "correctness",
            )
        with self.assertRaises(TypeError):
            RecoveryDecision(
                RecoveryDisposition.CHECK_REPAIR, "deterministic check failed",
                True, False, FailureClass.CORRECTNESS, "repair",
            )
        with self.assertRaises(ValueError):
            RecoveryDecision(RecoveryDisposition.CHECK_REPAIR, "   ", True, False)

    def test_no_classification_exposes_an_autonomous_step_off_its_ladder(self) -> None:
        fact_cases = (
            {}, {"budget_exhausted": True}, {"proof_required": True},
            {"fallback_executor_available": True}, {"tree_changed_in_scope": True},
            {"clean_contract_mismatch": True}, {"remote_required": True},
            {"remote_required": True, "remote_unavailable": True},
            {"tree_changed_out_of_scope": True}, {"rollback_succeeded": False},
        )
        codes = _policy_codes()
        self.assertGreater(len(codes), 60)
        for code in codes:
            self.assertIsNot(failure_class_for(code), FailureClass.UNKNOWN, code)
            for facts in fact_cases:
                with self.subTest(code=code, facts=facts):
                    decision = classify_failure(code, **facts)
                    if decision.strategy is RecoveryStrategy.HARD_STOP:
                        # A stopped boundary exposes no ladder step at all.
                        self.assertIs(decision.disposition, RecoveryDisposition.HARD_STOP)
                        continue
                    self.assertIn(decision.strategy, recovery_ladder(decision.failure_class))

    def test_a_boundary_fact_outranks_the_ladder_of_its_code(self) -> None:
        # An out-of-scope mutation or a rollback that did not restore the
        # expected tree stops the run whatever the failure code was, and an
        # automatic recovery that escaped its loop fails closed identically.
        cases = (
            {"tree_changed_out_of_scope": True}, {"rollback_succeeded": False},
        )
        for facts in cases:
            for code in ("CHECK_FAILED:unit", "AGENT_CONTRACT_MISMATCH", "LLM_429"):
                with self.subTest(code=code, facts=facts):
                    decision = classify_failure(code, **facts)
                    self.assertIs(decision.disposition, RecoveryDisposition.HARD_STOP)
                    self.assertIs(decision.strategy, RecoveryStrategy.HARD_STOP)
        escaped = classify_failure("SEMANTIC_REVISER_UNAVAILABLE", budget_exhausted=True)
        self.assertEqual(escaped.failure_class, FailureClass.EXTERNAL)
        self.assertIs(escaped.disposition, RecoveryDisposition.HARD_STOP)
        self.assertIs(escaped.strategy, RecoveryStrategy.HARD_STOP)

    def test_recovery_facts_are_derived_from_codes_only(self) -> None:
        self.assertEqual(
            RecoveryFacts.for_failure("CHECK_FAILED:unit"),
            RecoveryFacts(candidate_tree="", retry_allowed=True),
        )
        self.assertTrue(RecoveryFacts.for_failure("REVIEW_EVIDENCE_RETRY").proof_required)
        self.assertFalse(RecoveryFacts.for_failure("CHECK_FAILED:unit", budget_exhausted=True).retry_allowed)
        self.assertTrue(
            RecoveryFacts.for_failure("CHECK_FAILED:unit", candidate_tree="c" * 40)
            .candidate_tree == "c" * 40
        )
        with self.assertRaises(ValueError):
            RecoveryFacts.for_failure("")


if __name__ == "__main__":
    unittest.main()
