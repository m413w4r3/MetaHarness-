"""Deterministic classification of pipeline failures and recovery budgets.

This module only consumes stable failure codes and observed boundary facts. It
never asks a model to decide whether a failure is safe to recover from.

The recovery ladder is the autonomy contract: one failure *code* maps to exactly
one :class:`FailureClass`, every class owns one ordered tuple of
:class:`RecoveryStrategy`, and a strategy already consumed for an exact
:class:`RecoveryFingerprint` is never proposed twice.  A new candidate tree or
new stable failure facts open a new progression.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class FailureClass(StrEnum):
    """Who may decide, and how, after one stable failure code."""

    CORRECTNESS = "correctness"
    CONTRACT = "contract"
    MODEL_PROTOCOL = "model_protocol"
    EXTERNAL = "external"
    AUTHORITY = "authority"
    INTEGRITY = "integrity"
    SECURITY = "security"
    SPEC_DECISION = "spec_decision"
    UNKNOWN = "unknown"


class RecoveryStrategy(StrEnum):
    """One deterministic step of one recovery ladder."""

    RETRY_TARGETED = "retry_targeted"
    REPAIR_TARGETED = "repair_targeted"
    EXPAND_SCOPE = "expand_scope"
    REPLAN_STEP = "replan_step"
    REPLAN_CYCLE = "replan_cycle"
    FALLBACK_EXECUTOR = "fallback_executor"
    WAIT_EXTERNAL = "wait_external"
    WAIT_HUMAN = "wait_human"
    HARD_STOP = "hard_stop"

    @property
    def terminal(self) -> bool:
        """Whether this step ends autonomous progression instead of acting."""

        return self in _TERMINAL_STRATEGIES


_TERMINAL_STRATEGIES = frozenset({
    RecoveryStrategy.WAIT_EXTERNAL, RecoveryStrategy.WAIT_HUMAN, RecoveryStrategy.HARD_STOP,
})


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
    """One classification: what the pipeline may do, and the ladder position.

    ``disposition`` is the durable vocabulary of the recovery loops that exist
    today.  ``failure_class`` and ``strategy`` carry the deterministic ladder
    position of the same failure code: an exhausted bounded step exposes the
    following step instead of only an operator wait.
    """

    disposition: RecoveryDisposition
    reason: str
    consumes_budget: bool
    rollback_required: bool
    failure_class: FailureClass = FailureClass.UNKNOWN
    strategy: RecoveryStrategy = RecoveryStrategy.HARD_STOP

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, RecoveryDisposition):
            raise TypeError("recovery disposition must be a RecoveryDisposition")
        if not isinstance(self.reason, str):
            raise TypeError("recovery decision reason must be a string")
        if not self.reason.strip():
            raise ValueError("recovery decision reason must not be empty")
        if not isinstance(self.failure_class, FailureClass):
            raise TypeError("recovery failure class must be a FailureClass")
        if not isinstance(self.strategy, RecoveryStrategy):
            raise TypeError("recovery strategy must be a RecoveryStrategy")


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
    # Protocol corrections of one StepContractRepairPlanner answer inside the
    # same semantic repair slot; never a new ``max_step_contract_repairs``.
    max_contract_repair_output_corrections: int = 2
    # Self-contained planner restarts of one exhausted output-correction
    # budget, still inside the same semantic repair slot.
    max_contract_repair_planner_restarts: int = 1
    execution_fallbacks: ExecutionFallbacks = ExecutionFallbacks()

    def __post_init__(self) -> None:
        for name in (
            "max_transient_attempts", "max_executor_fallbacks",
            "max_check_infra_retries", "max_review_transport_retries",
            "max_workspace_setup_retries", "max_contract_repair_output_corrections",
            "max_contract_repair_planner_restarts",
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
    # A refused step commit under its effective authority is never retried:
    # scope, parent, tree, worktree, security and verification refusals.
    "COMMIT_GATE_FAILED", "COMMIT_SCOPE_VIOLATION", "COMMIT_PARENT_MISMATCH",
    "COMMIT_WORKTREE_DRIFT", "COMMIT_SECURITY_FAILURE", "COMMIT_VERIFICATION_FAILURE",
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


# --------------------------------------------------------------------------
# The recovery ladder: deterministic code -> class -> ordered strategies
# --------------------------------------------------------------------------

_SECURITY_STRATEGY_CODES = frozenset({
    "SECRET_IN_DIFF", "SECRET_IN_BLOB", "SECRET_IN_STAGED_BLOB", "SECRET_DETECTED",
    "SECRET_SECURITY_VIOLATION", "SECURITY_VIOLATION",
    "STAGED_BLOB_SCAN_FAILED", "BLOB_SCAN_FAILED", "UNSCANNABLE_STAGED_BLOB",
    "UNREVIEWABLE_TEXT_DIFF", "SOURCE_STAGED_BLOB_NOT_REVIEWABLE",
    "SOURCE_LIKE_STAGED_BLOB_NOT_REVIEWABLE", "CHECK_MUTATED_FORBIDDEN_FILES",
    "COMMIT_SECURITY_FAILURE",
})

_INTEGRITY_STRATEGY_CODES = frozenset({
    "DURABLE_ARTIFACT_CORRUPTED", "CORRUPTED_DURABLE_ARTIFACT",
    "RESUME_INTEGRITY_FAILURE", "INTEGRITY_MISMATCH", "TREE_MISMATCH",
    "HEAD_MISMATCH", "UNEXPECTED_HEAD", "UNEXPECTED_TREE", "BRANCH_MISMATCH",
    "COMMIT_TREE_MISMATCH", "COMMIT_WORKTREE_DRIFT", "BASE_MOVED_SINCE_RUN",
    "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH", "REPOSITORY_TREE_DRIFT_UNEXPLAINED",
    "UNEXPLAINED_REPOSITORY_TREE_DRIFT", "CHECK_AUTHORITY_CORRUPTED",
})

_AUTHORITY_STRATEGY_CODES = frozenset({
    "AGENT_SCOPE_VIOLATION", "AGENT_GIT_VIOLATION", "HEAD_MODIFIED_OUTSIDE_AUTHORITY",
    "TREE_MODIFIED_OUTSIDE_AUTHORITY", "BRANCH_MODIFIED_OUTSIDE_AUTHORITY",
    "COMMIT_GATE_FAILED", "COMMIT_SCOPE_VIOLATION", "COMMIT_PARENT_MISMATCH",
    "COMMIT_VERIFICATION_FAILURE", "CHECK_AUTHORITY_TAMPERING",
    "CHECK_AUTHORITY_MISMATCH", "CHECK_AUTHORITY_INVALID", "PLAN_APPROVAL_INVALID",
    "PLAN_APPROVAL_IDENTITY_MISMATCH", "APPROVED_PLAN_HASH_MISMATCH",
    "APPROVED_PLAN_SHA_MISMATCH", "PLAN_HASH_MISMATCH", "APPROVAL_IDENTITY_MISMATCH",
    "RESUME_IDENTITY_MISMATCH", "RESUME_IDENTITY_INVALID", "RESUME_REQUIRES_OPERATOR",
    "REMOTE_AUTHORITY_MISMATCH", "CHECK_REPAIR_NOT_AUTHORIZED", "REVIEW_AUTHORITY_MISSING",
    "REPAIR_SCOPE_APPROVAL_REQUIRED", "ATOMIC_SCOPE_POLICY_LIMIT",
    "CONTRACT_REPAIR_SCOPE_DENIED",
})

_CORRECTNESS_STRATEGY_CODES = frozenset({
    "CHECK_FAILED", "DETERMINISTIC_GATE_FAILED", "CHECK_REPAIR_EXHAUSTED",
    "CHECK_REPAIR_FIXED_POINT", "WAITING_REPAIR_EXHAUSTED", "REVIEW_IMPLEMENTATION",
    "REVIEW_REPLAN", "REVIEW_EVIDENCE_RETRY", "REVIEW_EVIDENCE_UNRESOLVED",
    "BOUNDED_SCOPE_REQUEST", "PLANNER_REPOSITORY_EVIDENCE",
    "REPOSITORY_EVIDENCE_RECOVERY_EXHAUSTED",
})

_CONTRACT_STRATEGY_CODES = frozenset({
    "AGENT_CONTRACT_MISMATCH", "CONTRACT_INSUFFICIENT", "CONTRACT_INSUFFICIENCY",
    "PLAN_REPOSITORY_PRECONDITION_INVALID", "PLAN_REPOSITORY_PRECONDITION_ERROR",
    "PLANNER_REPOSITORY_PRECONDITION_ERROR",
})

_MODEL_PROTOCOL_STRATEGY_CODES = frozenset({
    "PLANNER_FORMAT_INVALID", "PLANNER_OUTPUT_INVALID", "PLANNER_PROTOCOL_FAILED",
    "STEP_CONTRACT_REPAIR_OUTPUT_INVALID", "REVIEW_PARSE_INVALID",
    "REVIEWER_OUTPUT_INVALID", "REVIEW_FORMAT_INVALID",
})

_EXTERNAL_STRATEGY_CODES = frozenset({
    "AGENT_START_FAILED", "AGENT_RUNTIME_FAILED", "AGENT_TIMEOUT", "AGENT_PROTOCOL_FAILED",
    "AGENT_FAILURE", "AGENT_AUTH_FAILURE", "LLM_401", "LLM_403", "LLM_429", "LLM_5XX",
    "LLM_TIMEOUT", "LLM_TRANSPORT_FAILURE", "LLM_FAILURE", "LLM_AUTH_FAILURE",
    "MISSING_PROVIDER_CREDENTIALS", "PROVIDER_CREDENTIALS_MISSING",
    "EXTERNAL_AUTH_REQUIRED", "DNS_UNAVAILABLE", "NETWORK_UNAVAILABLE",
    "DOCKER_DAEMON_UNAVAILABLE", "REMOTE_TEMPORARILY_UNAVAILABLE", "CHECK_TIMEOUT",
    "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE", "CHECK_INFRA_RETRIES_EXHAUSTED",
    "CHECK_INFRASTRUCTURE_UNAVAILABLE", "CHECK_SIDE_EFFECT_REPEATED",
    "CHECK_SIDE_EFFECT_UNSTABLE", "CHECK_REPAIR_UNAVAILABLE", "CHECK_REPAIR_PROFILE_MISSING",
    "WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT", "TRANSIENT_ATTEMPTS_EXHAUSTED",
    "REVIEW_TRANSPORT_FAILURE", "REVIEWER_TRANSPORT_FAILURE", "PUSH_FAILED",
    "CANDIDATE_PUSH_FAILED", "CANDIDATE_REMOTE_UNAVAILABLE", "REMOTE_UNAVAILABLE",
    "SEMANTIC_REVISER_UNAVAILABLE", "SEMANTIC_REVISER_PROFILE_MISSING",
})

_SPEC_DECISION_STRATEGY_CODES = frozenset({
    "SPEC_DECISION_REQUIRED", "REVIEW_HUMAN_REQUIRED", "HUMAN_REQUIRED",
})

# Stable failure codes whose occurrence is itself the proof that the current
# candidate scope is insufficient: they admit ``EXPAND_SCOPE``.
_EVIDENCE_REQUIRED_CODES = frozenset({
    "REVIEW_EVIDENCE_RETRY", "REVIEW_EVIDENCE_UNRESOLVED", "PLANNER_REPOSITORY_EVIDENCE",
    "REPOSITORY_EVIDENCE_RECOVERY_EXHAUSTED", "BOUNDED_SCOPE_REQUEST",
})

# Provider credentials are an external waiting condition; retrying with the
# same credentials can never succeed.
_NEVER_RETRY_CODES = frozenset(_AUTH_CODES)

# Operator decisions the existing authority already grants.  Only these codes
# may turn an authority, integrity or security boundary into a human wait.
_OPERATOR_DECISION_CODES = frozenset({
    "SPEC_DECISION_REQUIRED", "SECURITY_POLICY_DECISION_REQUIRED", "REVIEW_HUMAN_REQUIRED",
    "ATOMIC_SCOPE_POLICY_LIMIT", "REPAIR_SCOPE_APPROVAL_REQUIRED",
    "CONTRACT_REPAIR_SCOPE_DENIED",
})

_CODE_CLASSES: Mapping[str, FailureClass] = {
    **{code: FailureClass.SECURITY for code in _SECURITY_STRATEGY_CODES},
    **{code: FailureClass.INTEGRITY for code in _INTEGRITY_STRATEGY_CODES},
    **{code: FailureClass.AUTHORITY for code in _AUTHORITY_STRATEGY_CODES},
    **{code: FailureClass.CORRECTNESS for code in _CORRECTNESS_STRATEGY_CODES},
    **{code: FailureClass.CONTRACT for code in _CONTRACT_STRATEGY_CODES},
    **{code: FailureClass.MODEL_PROTOCOL for code in _MODEL_PROTOCOL_STRATEGY_CODES},
    **{code: FailureClass.EXTERNAL for code in _EXTERNAL_STRATEGY_CODES},
    **{code: FailureClass.SPEC_DECISION for code in _SPEC_DECISION_STRATEGY_CODES},
}

_CODE_PREFIX_CLASSES: tuple[tuple[str, FailureClass], ...] = (
    ("SECRET_", FailureClass.SECURITY),
    ("SECURITY_", FailureClass.SECURITY),
    ("CHECK_FAILED", FailureClass.CORRECTNESS),
    ("REVIEW_PARSE", FailureClass.MODEL_PROTOCOL),
    ("REVIEW_FORMAT_INVALID", FailureClass.MODEL_PROTOCOL),
    ("REVIEWER_OUTPUT_INVALID", FailureClass.MODEL_PROTOCOL),
    ("PLANNER_FORMAT_INVALID", FailureClass.MODEL_PROTOCOL),
    ("PLANNER_OUTPUT_INVALID", FailureClass.MODEL_PROTOCOL),
    ("PLANNER_PROTOCOL_FAILED", FailureClass.MODEL_PROTOCOL),
    ("LLM_", FailureClass.EXTERNAL),
)

_RECOVERY_LADDERS: Mapping[FailureClass, tuple[RecoveryStrategy, ...]] = {
    # REPAIR_TARGETED -> EXPAND_SCOPE if proof is required -> REPLAN_STEP ->
    # REPLAN_CYCLE -> FALLBACK_EXECUTOR if one is configured -> WAIT_HUMAN.
    FailureClass.CORRECTNESS: (
        RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
        RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
        RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.WAIT_HUMAN,
    ),
    # REPAIR_TARGETED -> REPLAN_STEP -> REPLAN_CYCLE -> fallback -> WAIT_HUMAN.
    FailureClass.CONTRACT: (
        RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.REPLAN_STEP,
        RecoveryStrategy.REPLAN_CYCLE, RecoveryStrategy.FALLBACK_EXECUTOR,
        RecoveryStrategy.WAIT_HUMAN,
    ),
    # Format correction -> fresh clean completion -> fallback profile -> WAIT_HUMAN.
    FailureClass.MODEL_PROTOCOL: (
        RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.RETRY_TARGETED,
        RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.WAIT_HUMAN,
    ),
    # Bounded retry -> configured fallback executor -> WAIT_EXTERNAL.
    FailureClass.EXTERNAL: (
        RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.FALLBACK_EXECUTOR,
        RecoveryStrategy.WAIT_EXTERNAL,
    ),
    FailureClass.SPEC_DECISION: (RecoveryStrategy.WAIT_HUMAN,),
    # Authority, integrity and security boundaries never have an autonomous
    # step; only the authority tables in force pick between these terminals.
    FailureClass.SECURITY: (RecoveryStrategy.HARD_STOP, RecoveryStrategy.WAIT_HUMAN),
    FailureClass.INTEGRITY: (RecoveryStrategy.HARD_STOP, RecoveryStrategy.WAIT_HUMAN),
    FailureClass.AUTHORITY: (RecoveryStrategy.HARD_STOP, RecoveryStrategy.WAIT_HUMAN),
    FailureClass.UNKNOWN: (RecoveryStrategy.HARD_STOP,),
}

# The ladder steps one disposition names, most specific first.  A step
# that is not a member of the class ladder or not admitted by the facts is
# skipped, so the same disposition projects onto the step its class owns.
_DISPOSITION_LADDER_STEPS: Mapping[RecoveryDisposition, tuple[RecoveryStrategy, ...]] = {
    RecoveryDisposition.RETRY_SAME: (
        RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
        RecoveryStrategy.REPAIR_TARGETED,
    ),
    RecoveryDisposition.RETRY_AFTER_ROLLBACK: (
        RecoveryStrategy.RETRY_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
        RecoveryStrategy.REPAIR_TARGETED,
    ),
    RecoveryDisposition.FALLBACK_EXECUTOR: (
        RecoveryStrategy.FALLBACK_EXECUTOR, RecoveryStrategy.RETRY_TARGETED,
    ),
    RecoveryDisposition.CONTRACT_REPAIR: (
        RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.REPLAN_STEP,
    ),
    RecoveryDisposition.CHECK_REPAIR: (RecoveryStrategy.REPAIR_TARGETED,),
    RecoveryDisposition.REPLAN: (
        RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.RETRY_TARGETED,
    ),
}


def _stable_code(code: object) -> str:
    """The stable code of a failure, without its check ID suffix."""

    if not isinstance(code, str):
        return ""
    return code.strip().split(":", 1)[0].upper()


def failure_class_for(code: str) -> FailureClass:
    """Map one stable failure code onto its failure class, deterministically.

    An unknown code is ``UNKNOWN`` and therefore fails closed; no model, no
    heuristic and no caller-supplied hint ever chooses a class.
    """

    stable = _stable_code(code)
    if not stable:
        raise ValueError("failure must be a non-empty code")
    known = _CODE_CLASSES.get(stable)
    if known is not None:
        return known
    for prefix, failure_class in _CODE_PREFIX_CLASSES:
        if stable.startswith(prefix):
            return failure_class
    return FailureClass.UNKNOWN


def recovery_ladder(failure_class: FailureClass) -> tuple[RecoveryStrategy, ...]:
    """The ordered ladder of one failure class, terminals included."""

    return _RECOVERY_LADDERS[FailureClass(failure_class)]


def terminal_strategy(failure_class: FailureClass, code: str = "") -> RecoveryStrategy:
    """The single terminal the existing authority allows for one occurrence."""

    fclass = FailureClass(failure_class)
    if fclass is FailureClass.EXTERNAL:
        return RecoveryStrategy.WAIT_EXTERNAL
    if fclass is FailureClass.UNKNOWN:
        return RecoveryStrategy.HARD_STOP
    if fclass in {
        FailureClass.SECURITY, FailureClass.INTEGRITY, FailureClass.AUTHORITY,
    }:
        return (
            RecoveryStrategy.WAIT_HUMAN
            if _stable_code(code) in _OPERATOR_DECISION_CODES
            else RecoveryStrategy.HARD_STOP
        )
    return RecoveryStrategy.WAIT_HUMAN


@dataclass(frozen=True)
class RecoveryFacts:
    """The stable, observed facts one ladder step may depend on.

    Every field is deterministic evidence: a candidate tree, an observed fact
    such as the failing check set, or a boolean the pipeline observed.  None of
    them is authored or chosen by a model.
    """

    candidate_tree: str = ""
    # Caller-observed facts (failed check set, failing step, ...).  They are
    # part of the anti-loop identity: new failure facts open a new progression.
    observed_facts: tuple[tuple[str, str], ...] = ()
    proof_required: bool = False
    fallback_executor_available: bool = False
    retry_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_tree, str):
            raise TypeError("candidate_tree must be a string")
        if len(self.candidate_tree) > 128:
            raise ValueError("candidate_tree must be a bounded string")
        observed = _canonical_facts(self.observed_facts)
        if {key for key, _ in observed} & set(_LADDER_FACT_KEYS):
            raise ValueError("observed failure facts must not shadow a ladder fact")
        object.__setattr__(self, "observed_facts", observed)
        for name in ("proof_required", "fallback_executor_available", "retry_allowed"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean fact")

    @classmethod
    def for_failure(
        cls,
        code: str,
        *,
        candidate_tree: str = "",
        observed_facts: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        proof_required: bool = False,
        budget_exhausted: bool = False,
        fallback_executor_available: bool = False,
    ) -> RecoveryFacts:
        """Derive the ladder facts of one stable failure code."""

        stable = _stable_code(code)
        if not stable:
            raise ValueError("failure must be a non-empty code")
        return cls(
            candidate_tree=candidate_tree,
            observed_facts=observed_facts,
            proof_required=proof_required or stable in _EVIDENCE_REQUIRED_CODES,
            fallback_executor_available=fallback_executor_available,
            retry_allowed=not budget_exhausted and stable not in _NEVER_RETRY_CODES,
        )

    def stable_items(self) -> tuple[tuple[str, str], ...]:
        """The canonical fingerprint facts: sorted, candidate-independent."""

        return _canonical_facts((
            *self.observed_facts,
            ("fallback_executor_available", _flag(self.fallback_executor_available)),
            ("proof_required", _flag(self.proof_required)),
            ("retry_allowed", _flag(self.retry_allowed)),
        ))


_LADDER_FACT_KEYS = (
    "fallback_executor_available", "proof_required", "retry_allowed",
)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _canonical_facts(
    facts: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Canonical, sorted ``(key, value)`` pairs, refusing duplicates."""

    if isinstance(facts, Mapping):
        items: Iterable[object] = facts.items()
    elif isinstance(facts, (tuple, list, set, frozenset)):
        items = facts
    else:
        raise TypeError("stable failure facts must be a mapping or a sequence of pairs")
    canonical: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("stable failure facts must be (key, value) pairs")
        key, value = item
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError("stable failure facts must be non-empty string pairs")
        canonical.append((key, value))
    if len({key for key, _ in canonical}) != len(canonical):
        raise ValueError("stable failure facts must not repeat a key")
    return tuple(sorted(canonical))


def _admitted(strategy: RecoveryStrategy, facts: RecoveryFacts) -> bool:
    if strategy is RecoveryStrategy.RETRY_TARGETED:
        return facts.retry_allowed
    if strategy is RecoveryStrategy.EXPAND_SCOPE:
        return facts.proof_required
    if strategy is RecoveryStrategy.FALLBACK_EXECUTOR:
        return facts.fallback_executor_available
    return True


def admitted_strategies(
    failure_class: FailureClass, facts: RecoveryFacts,
) -> tuple[RecoveryStrategy, ...]:
    """The ladder steps this occurrence may execute, terminals included."""

    if not isinstance(facts, RecoveryFacts):
        raise TypeError("recovery facts must be a RecoveryFacts instance")
    return tuple(
        step for step in recovery_ladder(failure_class) if _admitted(step, facts)
    )


def _advance(
    admitted: tuple[RecoveryStrategy, ...],
    *,
    after: RecoveryStrategy | None,
    terminal: RecoveryStrategy,
) -> RecoveryStrategy:
    """The next autonomous step after ``after``, or the class terminal."""

    steps = [step for step in admitted if not step.terminal]
    if after in steps:
        remaining = steps[steps.index(after) + 1:]
        return remaining[0] if remaining else terminal
    return steps[0] if steps else terminal


def project_strategy(
    failure_class: FailureClass,
    disposition: RecoveryDisposition,
    facts: RecoveryFacts,
    *,
    code: str = "",
) -> RecoveryStrategy:
    """Project one classification onto the ladder position it authorizes.

    A step the pipeline may execute now is projected as that step.  A bounded
    step that was exhausted advances to the following step of the class ladder
    instead of collapsing onto an operator wait.  A stopped run, a waiting
    condition and a boundary that existing authority refuses project their own
    terminal: authority always outranks the ladder of the failure code.
    """

    fclass = FailureClass(failure_class)
    if not isinstance(disposition, RecoveryDisposition):
        raise TypeError("recovery disposition must be a RecoveryDisposition")
    if not isinstance(facts, RecoveryFacts):
        raise TypeError("recovery facts must be a RecoveryFacts instance")
    if disposition is RecoveryDisposition.HARD_STOP:
        return RecoveryStrategy.HARD_STOP
    if disposition is RecoveryDisposition.WAIT_EXTERNAL:
        return RecoveryStrategy.WAIT_EXTERNAL
    if disposition is RecoveryDisposition.CONTINUE_WITH_WARNING:
        # The run continues without the optional external capability; the
        # ladder exposes no autonomous step for a missing capability.
        return terminal_strategy(fclass, code)
    ladder = recovery_ladder(fclass)
    terminal = terminal_strategy(fclass, code)
    if disposition is RecoveryDisposition.WAIT_HUMAN:
        return _advance(
            admitted_strategies(fclass, facts),
            after=RecoveryStrategy.REPAIR_TARGETED,
            terminal=terminal,
        )
    for step in _DISPOSITION_LADDER_STEPS[disposition]:
        if step in ladder and _admitted(step, facts):
            return step
    return _advance(admitted_strategies(fclass, facts), after=None, terminal=terminal)


@dataclass(frozen=True)
class RecoveryFingerprint:
    """The exact identity of one consumed ladder step.

    The same strategy may never be proposed again for the same candidate tree,
    the same failure class and the same stable failure facts.
    """

    candidate_tree: str
    failure_class: FailureClass
    stable_failure_facts: tuple[tuple[str, str], ...]
    strategy: RecoveryStrategy

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_tree, str) or len(self.candidate_tree) > 128:
            raise ValueError("candidate_tree must be a bounded string")
        object.__setattr__(self, "failure_class", FailureClass(self.failure_class))
        object.__setattr__(self, "strategy", RecoveryStrategy(self.strategy))
        object.__setattr__(
            self, "stable_failure_facts",
            _canonical_facts(self.stable_failure_facts),
        )


class RecoveryProgression:
    """The anti-loop ledger of one failure: one ladder step per exact fingerprint.

    A strategy already consumed for an exact :class:`RecoveryFingerprint` is
    never proposed again.  A new candidate tree or new stable failure facts form
    a new fingerprint and therefore open a new progression.
    """

    def __init__(self, consumed: Iterable[RecoveryFingerprint] = ()) -> None:
        self._consumed: set[RecoveryFingerprint] = set()
        for fingerprint in consumed:
            if not isinstance(fingerprint, RecoveryFingerprint):
                raise TypeError("consumed strategies must be recovery fingerprints")
            self._consumed.add(fingerprint)

    @property
    def consumed(self) -> frozenset[RecoveryFingerprint]:
        return frozenset(self._consumed)

    def fingerprint(
        self,
        *,
        candidate_tree: str,
        failure_class: FailureClass,
        facts: RecoveryFacts,
        strategy: RecoveryStrategy,
    ) -> RecoveryFingerprint:
        if not isinstance(facts, RecoveryFacts):
            raise TypeError("recovery facts must be a RecoveryFacts instance")
        return RecoveryFingerprint(
            candidate_tree, failure_class, facts.stable_items(), strategy,
        )

    def is_consumed(self, fingerprint: RecoveryFingerprint) -> bool:
        if not isinstance(fingerprint, RecoveryFingerprint):
            raise TypeError("expected a recovery fingerprint")
        return fingerprint in self._consumed

    def consume(
        self,
        *,
        candidate_tree: str,
        failure_class: FailureClass,
        facts: RecoveryFacts,
        strategy: RecoveryStrategy,
    ) -> RecoveryFingerprint:
        """Consume one autonomous step; a repeat of the same fingerprint refuses."""

        fingerprint = self.fingerprint(
            candidate_tree=candidate_tree, failure_class=failure_class,
            facts=facts, strategy=strategy,
        )
        if fingerprint.strategy.terminal:
            raise ValueError("a terminal strategy is never consumed as a recovery step")
        if fingerprint in self._consumed:
            raise ValueError("this strategy was already consumed for this exact fingerprint")
        self._consumed.add(fingerprint)
        return fingerprint

    def next_strategy(
        self,
        *,
        candidate_tree: str,
        failure_class: FailureClass,
        facts: RecoveryFacts,
        code: str = "",
    ) -> RecoveryStrategy:
        """The next unconsumed autonomous step, or the class terminal."""

        fclass = FailureClass(failure_class)
        for step in recovery_ladder(fclass):
            if step.terminal or not _admitted(step, facts):
                continue
            if self.fingerprint(
                candidate_tree=candidate_tree, failure_class=fclass,
                facts=facts, strategy=step,
            ) in self._consumed:
                continue
            return step
        return terminal_strategy(fclass, code)


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
    proof_required: bool = False,
) -> RecoveryDecision:
    """Classify stable failure facts; unknown failures fail closed.

    ``failure`` may include a check ID suffix (for example
    ``CHECK_FAILED:unit``); only its stable code is consulted.  The returned
    decision always carries the deterministic failure class of that code and
    the ladder position the classification authorizes.
    """

    if not isinstance(failure, str) or not failure.strip():
        raise ValueError("failure must be a non-empty code")
    code = failure.strip().split(":", 1)[0].upper()
    failure_class = failure_class_for(code)
    ladder_facts = RecoveryFacts.for_failure(
        code,
        proof_required=proof_required,
        budget_exhausted=budget_exhausted,
        fallback_executor_available=fallback_executor_available,
    )

    def decision(
        disposition: RecoveryDisposition, reason: str, *, consumes: bool = False,
        rollback: bool = False,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            disposition, reason, consumes, rollback, failure_class,
            project_strategy(failure_class, disposition, ladder_facts, code=code),
        )

    if (
        code in _HARD_STOP_CODES
        or code.startswith(("SECRET_", "SECURITY_VIOLATION", "DURABLE_ARTIFACT_CORRUPTED"))
    ):
        return decision(RecoveryDisposition.HARD_STOP, "authority, integrity, or security boundary failed")
    if tree_changed_out_of_scope:
        return decision(RecoveryDisposition.HARD_STOP, "tree changed outside approved scope")
    if not rollback_succeeded:
        return decision(RecoveryDisposition.HARD_STOP, "rollback did not restore the expected tree")
    if code == "CHECK_REPAIR_FIXED_POINT":
        return decision(
            RecoveryDisposition.WAIT_HUMAN,
            "the same candidate and failed checks remain after the repair budget was exhausted; "
            "code change or additional repair authority is required",
        )
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
    if code == "STEP_CONTRACT_REPAIR_OUTPUT_INVALID":
        # A planner protocol defect of one repair slot, never a worker
        # contract mismatch: corrected in place, then an operator retry.
        if budget_exhausted:
            return decision(
                RecoveryDisposition.WAIT_HUMAN,
                "contract repair output corrections were exhausted; retry the contract repair planner",
            )
        return decision(RecoveryDisposition.CONTRACT_REPAIR, "contract repair planner output can be corrected", consumes=True)
    if code in {
        "SPEC_DECISION_REQUIRED", "SECURITY_POLICY_DECISION_REQUIRED",
        "ATOMIC_SCOPE_POLICY_LIMIT", "REPAIR_SCOPE_APPROVAL_REQUIRED",
        "REVIEW_HUMAN_REQUIRED", "CONTRACT_REPAIR_SCOPE_DENIED",
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
    "ExecutionFallbacks", "FailureClass", "RecoveryBudgets", "RecoveryDecision",
    "RecoveryDisposition", "RecoveryFacts", "RecoveryFingerprint", "RecoveryProgression",
    "RecoveryStrategy", "admitted_strategies", "classify_failure", "failure_class_for",
    "project_strategy", "recovery_ladder", "terminal_strategy",
]
