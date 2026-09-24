"""Deterministic classification of pipeline failures and recovery budgets.

This module only consumes stable failure codes and observed boundary facts. It
never asks a model to decide whether a failure is safe to recover from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RecoveryDisposition(StrEnum):
    RETRY_SAME = "retry_same"
    RETRY_AFTER_ROLLBACK = "retry_after_rollback"
    FALLBACK_EXECUTOR = "fallback_executor"
    CONTRACT_REPAIR = "contract_repair"
    CHECK_REPAIR = "check_repair"
    REPLAN = "replan"
    WAIT_EXTERNAL = "wait_external"
    WAIT_HUMAN = "wait_human"
    CONTINUE_WITH_WARNING = "continue_with_warning"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True)
class RecoveryDecision:
    disposition: RecoveryDisposition
    reason: str
    consumes_budget: bool
    rollback_required: bool


@dataclass(frozen=True)
class ExecutionFallbacks:
    """Configured executor fallback profile IDs, grouped by worker authority."""

    mechanical: tuple[str, ...] = ()
    reasoning: tuple[str, ...] = ()
    agentic: tuple[str, ...] = ()
    semantic_reviser: tuple[str, ...] = ()
    check_repair: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("mechanical", "reasoning", "agentic", "semantic_reviser", "check_repair"):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(values, tuple):
                raise ValueError(f"execution_fallbacks.{name} must be a tuple of profile IDs")
            if len(values) > 10 or any(
                not isinstance(value, str)
                or not value
                or len(value) > 64
                or not value[0].isalnum()
                or any(not (char.isalnum() or char in "_.-") for char in value)
                for value in values
            ) or len(set(values)) != len(values):
                raise ValueError(f"execution_fallbacks.{name} contains an invalid profile ID")

    def for_execution_class(self, execution_class: str) -> tuple[str, ...]:
        key = str(execution_class).casefold()
        if key not in {"mechanical", "reasoning", "agentic"}:
            raise ValueError("execution class is invalid")
        return getattr(self, key)


@dataclass(frozen=True)
class RecoveryBudgets:
    """Retry limits frozen into each run's immutable options snapshot."""

    max_transient_attempts: int = 2
    max_executor_fallbacks: int = 1
    max_check_infra_retries: int = 2
    max_review_transport_retries: int = 2
    max_workspace_setup_retries: int = 2
    execution_fallbacks: ExecutionFallbacks = ExecutionFallbacks()

    def __post_init__(self) -> None:
        for name in (
            "max_transient_attempts", "max_executor_fallbacks",
            "max_check_infra_retries", "max_review_transport_retries",
            "max_workspace_setup_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
                raise ValueError(f"{name} must be an integer between 0 and 10")
        if not isinstance(self.execution_fallbacks, ExecutionFallbacks):
            raise ValueError("execution_fallbacks is invalid")


_HARD_STOP_CODES = frozenset({
    "SECRET_IN_DIFF", "SECRET_IN_BLOB", "SECRET_IN_STAGED_BLOB", "SECRET_DETECTED",
    "STAGED_BLOB_SCAN_FAILED", "UNSCANNABLE_STAGED_BLOB",
    "SOURCE_STAGED_BLOB_NOT_REVIEWABLE", "UNREVIEWABLE_TEXT_DIFF",
    "BLOB_SCAN_FAILED", "SOURCE_LIKE_STAGED_BLOB_NOT_REVIEWABLE",
    "AGENT_SCOPE_VIOLATION", "AGENT_GIT_VIOLATION",
    "HEAD_MODIFIED_OUTSIDE_AUTHORITY", "TREE_MODIFIED_OUTSIDE_AUTHORITY",
    "BRANCH_MODIFIED_OUTSIDE_AUTHORITY", "DURABLE_ARTIFACT_CORRUPTED",
    "CORRUPTED_DURABLE_ARTIFACT", "RESUME_INTEGRITY_FAILURE",
    "RESUME_IDENTITY_MISMATCH", "APPROVED_PLAN_HASH_MISMATCH",
    "APPROVED_PLAN_SHA_MISMATCH", "PLAN_HASH_MISMATCH",
    "RESUME_REQUIRES_OPERATOR", "PLAN_APPROVAL_IDENTITY_MISMATCH",
    "CHECK_AUTHORITY_TAMPERING", "CHECK_AUTHORITY_MISMATCH",
    "CHECK_AUTHORITY_INVALID", "CHECK_AUTHORITY_CORRUPTED", "SECURITY_VIOLATION",
    "SECRET_SECURITY_VIOLATION", "REPOSITORY_TREE_DRIFT_UNEXPLAINED",
    "UNEXPLAINED_REPOSITORY_TREE_DRIFT",
    "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH",
    "UNEXPECTED_HEAD", "UNEXPECTED_TREE", "TREE_MISMATCH",
    "INTEGRITY_MISMATCH", "HEAD_MISMATCH", "BRANCH_MISMATCH",
    "COMMIT_TREE_MISMATCH", "BASE_MOVED_SINCE_RUN",
    "CHECK_MUTATED_FORBIDDEN_FILES",
    "REMOTE_AUTHORITY_MISMATCH",
    "APPROVAL_IDENTITY_MISMATCH", "RESUME_IDENTITY_INVALID",
})

_AUTH_CODES = frozenset({
    "AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "LLM_AUTH_FAILURE",
    "MISSING_PROVIDER_CREDENTIALS", "PROVIDER_CREDENTIALS_MISSING",
    "EXTERNAL_AUTH_REQUIRED",
})

_TRANSIENT_AGENT_CODES = frozenset({
    "AGENT_START_FAILED", "AGENT_RUNTIME_FAILED", "AGENT_TIMEOUT",
    "AGENT_PROTOCOL_FAILED", "AGENT_FAILURE",
})

_TRANSIENT_EXTERNAL_CODES = frozenset({
    "LLM_429", "LLM_5XX", "LLM_TIMEOUT", "LLM_TRANSPORT_FAILURE",
    "LLM_FAILURE", "DNS_UNAVAILABLE", "NETWORK_UNAVAILABLE",
    "DOCKER_DAEMON_UNAVAILABLE", "REMOTE_TEMPORARILY_UNAVAILABLE",
})


# Model corrections whose exhaustion needs an operator, never a hard stop:
# the candidate is still the exact, rolled-back pre-attempt tree.
_CORRECTNESS_REPAIR_CODES = frozenset({
    "AGENT_CONTRACT_MISMATCH", "CONTRACT_INSUFFICIENT", "CONTRACT_INSUFFICIENCY",
    "REVIEW_IMPLEMENTATION", "REVIEW_REPLAN", "BOUNDED_SCOPE_REQUEST",
    "PLANNER_REPOSITORY_EVIDENCE",
})


def classify_failure(
    failure: str,
    *,
    tree_changed_in_scope: bool = False,
    tree_changed_out_of_scope: bool = False,
    rollback_succeeded: bool = True,
    clean_contract_mismatch: bool = False,
    remote_required: bool = False,
    remote_unavailable: bool = False,
    fallback_executor_available: bool = False,
    budget_exhausted: bool = False,
) -> RecoveryDecision:
    """Classify stable failure facts; unknown failures fail closed.

    ``failure`` may include a check ID suffix (for example
    ``CHECK_FAILED:unit``); only its stable code is consulted.
    """

    if not isinstance(failure, str) or not failure.strip():
        raise ValueError("failure must be a non-empty code")
    code = failure.strip().split(":", 1)[0].upper()

    def decision(
        disposition: RecoveryDisposition, reason: str, *, consumes: bool = False,
        rollback: bool = False,
    ) -> RecoveryDecision:
        return RecoveryDecision(disposition, reason, consumes, rollback)

    if (
        code in _HARD_STOP_CODES
        or code.startswith(("SECRET_", "SECURITY_VIOLATION", "DURABLE_ARTIFACT_CORRUPTED"))
    ):
        return decision(RecoveryDisposition.HARD_STOP, "authority, integrity, or security boundary failed")
    if tree_changed_out_of_scope:
        return decision(RecoveryDisposition.HARD_STOP, "tree changed outside approved scope")
    if not rollback_succeeded:
        return decision(RecoveryDisposition.HARD_STOP, "rollback did not restore the expected tree")
    if code in {
        "CHECK_REPAIR_EXHAUSTED", "DETERMINISTIC_GATE_FAILED", "WAITING_REPAIR_EXHAUSTED",
        "REVIEW_EVIDENCE_UNRESOLVED", "HUMAN_REQUIRED", "REPOSITORY_EVIDENCE_RECOVERY_EXHAUSTED",
    }:
        return decision(RecoveryDisposition.WAIT_HUMAN, "correctness repair or operator decision is required")
    if code == "CHECK_INFRA_RETRIES_EXHAUSTED":
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "check infrastructure retries were exhausted")
    if code == "TRANSIENT_ATTEMPTS_EXHAUSTED":
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "external executor retries were exhausted")
    if code == "CHECK_REPAIR_UNAVAILABLE":
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "check repair executor is unavailable")
    if code in {
        "CHECK_INFRASTRUCTURE_UNAVAILABLE", "CHECK_SIDE_EFFECT_REPEATED",
        "CHECK_SIDE_EFFECT_UNSTABLE",
    }:
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "check infrastructure needs operator or environment recovery")
    if budget_exhausted and code in {
        "CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE",
        "WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT",
    }:
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "bounded infrastructure retries were exhausted")
    if code == "REVIEWER_TRANSPORT_FAILURE" and budget_exhausted:
        return decision(
            RecoveryDisposition.WAIT_EXTERNAL,
            "review is retained at its final-review checkpoint for retry or operator action",
        )
    if code in {
        "SPEC_DECISION_REQUIRED", "SECURITY_POLICY_DECISION_REQUIRED",
        "ATOMIC_SCOPE_POLICY_LIMIT", "REPAIR_SCOPE_APPROVAL_REQUIRED",
        "REVIEW_HUMAN_REQUIRED",
    }:
        return decision(RecoveryDisposition.WAIT_HUMAN, "an operator decision is required")
    if code in {"PUSH_FAILED", "CANDIDATE_PUSH_FAILED"}:
        if not remote_required:
            return decision(RecoveryDisposition.CONTINUE_WITH_WARNING, "optional remote publication failed")
        if remote_unavailable:
            return decision(RecoveryDisposition.WAIT_EXTERNAL, "required remote is temporarily unavailable")
        if budget_exhausted:
            return decision(RecoveryDisposition.WAIT_EXTERNAL, "required remote publication retries were exhausted")
        return decision(RecoveryDisposition.RETRY_SAME, "required remote publication did not complete", consumes=True)
    if code in _AUTH_CODES or code.startswith(("LLM_401", "LLM_403")):
        return decision(RecoveryDisposition.WAIT_EXTERNAL, "credentials or external authorization required")
    if budget_exhausted:
        if code in _TRANSIENT_AGENT_CODES or code in _TRANSIENT_EXTERNAL_CODES or code.startswith(("LLM_5", "LLM_429", "LLM_HTTP_5", "LLM_HTTP_429")) or code in {"REVIEW_TRANSPORT_FAILURE", "REVIEWER_TRANSPORT_FAILURE"}:
            return decision(RecoveryDisposition.WAIT_EXTERNAL, "external recovery budget exhausted")
        if code.startswith(("REVIEW_FORMAT_INVALID", "REVIEWER_OUTPUT_INVALID", "REVIEW_PARSE")):
            return decision(RecoveryDisposition.WAIT_HUMAN, "review repair budget exhausted")
        if code == "REVIEW_EVIDENCE_RETRY":
            return decision(RecoveryDisposition.WAIT_HUMAN, "reviewer evidence recovery was exhausted")
        if code in _CORRECTNESS_REPAIR_CODES or code.startswith("CHECK_FAILED"):
            return decision(RecoveryDisposition.WAIT_HUMAN, "bounded correctness repair was exhausted")
        if code.startswith(("PLANNER_FORMAT_INVALID", "PLANNER_OUTPUT_INVALID")):
            return decision(RecoveryDisposition.WAIT_HUMAN, "planner correction budget exhausted")
        if code in {"PLAN_REPOSITORY_PRECONDITION_INVALID", "PLAN_REPOSITORY_PRECONDITION_ERROR", "PLANNER_REPOSITORY_PRECONDITION_ERROR"}:
            return decision(RecoveryDisposition.WAIT_HUMAN, "planning correction budget exhausted")
        return decision(RecoveryDisposition.HARD_STOP, "bounded recovery budget exhausted")

    if code in {"CANDIDATE_REMOTE_UNAVAILABLE", "REMOTE_UNAVAILABLE"}:
        return decision(
            RecoveryDisposition.WAIT_EXTERNAL if remote_required else RecoveryDisposition.CONTINUE_WITH_WARNING,
            "candidate remote availability depends on publication authority",
        )
    if code in {"AGENT_CONTRACT_MISMATCH", "CONTRACT_INSUFFICIENT", "CONTRACT_INSUFFICIENCY"}:
        if clean_contract_mismatch:
            return decision(RecoveryDisposition.CONTRACT_REPAIR, "clean contract mismatch is repairable", consumes=True)
        return decision(RecoveryDisposition.REPLAN, "contract facts require a bounded replan", consumes=True)
    if code in {"REVIEW_IMPLEMENTATION", "BOUNDED_SCOPE_REQUEST"}:
        return decision(RecoveryDisposition.CONTRACT_REPAIR, "bounded implementation correction is available", consumes=True)
    if code == "REVIEW_REPLAN":
        return decision(RecoveryDisposition.REPLAN, "review requested a bounded planning correction", consumes=True)
    if code.startswith("CHECK_FAILED"):
        return decision(RecoveryDisposition.CHECK_REPAIR, "deterministic check failed", consumes=True)
    if code in {"CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE"}:
        return decision(RecoveryDisposition.RETRY_SAME, "check infrastructure can be retried", consumes=True)
    if code == "REVIEW_EVIDENCE_RETRY":
        return decision(
            RecoveryDisposition.RETRY_SAME,
            "reviewer FAIL is retried once on rebuilt local evidence", consumes=True,
        )
    if code.startswith(("REVIEW_PARSE", "REVIEWER_OUTPUT_INVALID", "REVIEW_FORMAT_INVALID")):
        return decision(RecoveryDisposition.CONTRACT_REPAIR, "review output format can be repaired", consumes=True)
    if code.startswith(("PLANNER_FORMAT_INVALID", "PLANNER_PROTOCOL_FAILED", "PLANNER_OUTPUT_INVALID")):
        return decision(RecoveryDisposition.REPLAN, "planner protocol can be retried", consumes=True)
    if code == "PLANNER_REPOSITORY_EVIDENCE":
        return decision(RecoveryDisposition.REPLAN, "named immutable repository evidence can repair the plan", consumes=True)
    if code in {
        "PLAN_REPOSITORY_PRECONDITION_ERROR", "PLAN_REPOSITORY_PRECONDITION_INVALID",
        "PLANNER_REPOSITORY_PRECONDITION_ERROR",
    }:
        return decision(RecoveryDisposition.REPLAN, "repository precondition needs a bounded replan", consumes=True)
    if code in {"WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT"}:
        return decision(RecoveryDisposition.RETRY_SAME, "workspace setup can be retried", consumes=True)
    if code in _TRANSIENT_AGENT_CODES:
        if fallback_executor_available:
            return decision(RecoveryDisposition.FALLBACK_EXECUTOR, "another authorized executor is available", consumes=True)
        if tree_changed_in_scope:
            return decision(
                RecoveryDisposition.RETRY_AFTER_ROLLBACK,
                "agent failed after an in-scope tree change", consumes=True, rollback=True,
            )
        return decision(RecoveryDisposition.RETRY_SAME, "transient agent failure", consumes=True)
    if code == "SEMANTIC_REVISER_UNAVAILABLE":
        return decision(RecoveryDisposition.CONTINUE_WITH_WARNING, "semantic reviser is unavailable; deterministic checks remain authoritative")
    if (
        code in _TRANSIENT_EXTERNAL_CODES
        or code.startswith(("LLM_5", "LLM_429", "LLM_HTTP_5", "LLM_HTTP_429"))
    ):
        return decision(RecoveryDisposition.RETRY_SAME, "transient provider or network failure", consumes=True)
    if code in {"REVIEWER_TRANSPORT_FAILURE", "REVIEW_TRANSPORT_FAILURE"}:
        return decision(RecoveryDisposition.RETRY_SAME, "review transport can be retried", consumes=True)

    return decision(RecoveryDisposition.HARD_STOP, "unclassified failure has no authorized automatic recovery")


__all__ = [
    "ExecutionFallbacks", "RecoveryBudgets", "RecoveryDecision", "RecoveryDisposition",
    "classify_failure",
]
