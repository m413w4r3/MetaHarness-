"""Modèles de données de configuration et de contrôle."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .recovery_policy import RecoveryBudgets


class ProfileDriver(StrEnum):
    OPENAI_CHAT = "openai-chat"
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"
    # A protocol-neutral trusted process driver.  It is deliberately not a
    # DeepSeek-specific integration: concrete harnesses register their own
    # driver ID when their local contract is known.
    EXTERNAL = "external"


def profile_driver_name(value: ProfileDriver | str) -> str:
    """Return a stable driver ID for built-ins and registered extensions."""

    if isinstance(value, ProfileDriver):
        return value.value
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("profile driver must be a non-empty string")


@dataclass(frozen=True)
class AgentExecutorCapabilities:
    """Facts an executor can truthfully expose to orchestration observers."""

    edits_workspace: bool = False
    exposes_session_id: bool = False
    exposes_usage: bool = False
    exposes_reasoning_usage: bool = False
    exposes_tool_count: bool = False
    isolation_mode: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "edits_workspace",
            "exposes_session_id",
            "exposes_usage",
            "exposes_reasoning_usage",
            "exposes_tool_count",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.isolation_mode is not None and (
            not isinstance(self.isolation_mode, str) or not self.isolation_mode.strip()
        ):
            raise ValueError("isolation_mode must be a non-empty string or null")


class SelectionMode(StrEnum):
    REQUEST = "request"
    CLI = "cli"
    EXTERNAL_UI = "external-ui"


class ExecutionRole(StrEnum):
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    REPAIR = "repair"
    AUDITOR = "auditor"
    REVISER = "reviser"


class PlanDecision(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class BlockerKind(StrEnum):
    REPOSITORY_EVIDENCE = "REPOSITORY_EVIDENCE"
    SPEC_DECISION = "SPEC_DECISION"
    SECURITY_POLICY = "SECURITY_POLICY"
    ATOMIC_SCOPE = "ATOMIC_SCOPE"


class ExecutionMode(StrEnum):
    SINGLE = "SINGLE"
    STAGED = "STAGED"


class ExecutionClass(StrEnum):
    """Planner classification consumed by the execution router."""

    MECHANICAL = "MECHANICAL"
    REASONING = "REASONING"
    AGENTIC = "AGENTIC"


@dataclass(frozen=True)
class RoutingConfig:
    """Profiles selected for planner-produced execution classes."""

    mechanical_profile: str = ""
    reasoning_profile: str = ""
    agentic_profile: str = ""

    def __post_init__(self) -> None:
        for name in ("mechanical_profile", "reasoning_profile", "agentic_profile"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"routing {name} must be a profile id")

    def profile_for(self, execution_class: ExecutionClass | str) -> str:
        try:
            klass = ExecutionClass(execution_class)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution class is invalid") from exc
        return {
            ExecutionClass.MECHANICAL: self.mechanical_profile,
            ExecutionClass.REASONING: self.reasoning_profile,
            ExecutionClass.AGENTIC: self.agentic_profile,
        }[klass]


@dataclass(frozen=True)
class CodexProviderConfig:
    """Secret-free provider wiring for a managed Codex runtime."""

    name: str
    base_url: str
    wire_api: str
    api_key_env: str

    def __post_init__(self) -> None:
        for name in ("name", "base_url", "wire_api", "api_key_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Codex provider {name} must be non-empty")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    display_name: str
    roles: tuple[ExecutionRole, ...]
    driver: ProfileDriver | str
    model: str
    selection_mode: SelectionMode

    base_url: str | None = None
    endpoint_path: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int = 300
    retries: int = 2
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    # Trusted argv for the generic external-process driver.  It is never
    # derived from a prompt, model name, provider name, or role.
    argv: tuple[str, ...] = ()

    effort: str | None = None
    sandbox: str | None = None
    permission_mode: str | None = None

    description: str = ""
    strengths: tuple[str, ...] = ()
    cost_tier: str = "standard"
    latency_tier: str = "standard"
    # Provider metadata is independent from the execution driver/harness.
    provider: str = "openai"
    driver_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.driver, (ProfileDriver, str)) or not str(self.driver).strip():
            raise ValueError("profile driver must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("profile model must be a non-empty string")
        if not isinstance(self.argv, tuple) or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in self.argv
        ):
            raise ValueError("profile argv must be a tuple of non-empty strings")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("profile provider must be a non-empty string")
        if self.driver_version is not None and (
            not isinstance(self.driver_version, str) or not self.driver_version.strip()
        ):
            raise ValueError("profile driver_version must be a non-empty string or null")
        if not isinstance(self.description, str) or len(self.description) > 300:
            raise ValueError("profile description must be at most 300 characters")
        if not isinstance(self.strengths, tuple) or len(self.strengths) > 8:
            raise ValueError("profile strengths must contain at most 8 entries")
        if any(not isinstance(item, str) or len(item) > 80 for item in self.strengths):
            raise ValueError("profile strengths entries must be at most 80 characters")
        if not isinstance(self.cost_tier, str) or self.cost_tier not in {"low", "standard", "high"}:
            raise ValueError("profile cost_tier is invalid")
        if not isinstance(self.latency_tier, str) or self.latency_tier not in {"fast", "standard", "slow"}:
            raise ValueError("profile latency_tier is invalid")


@dataclass(frozen=True)
class ImplementationStep:
    id: str
    title: str
    execution_class: ExecutionClass
    depends_on: str | None
    objective: str
    read_set: tuple[str, ...]
    write_set: tuple[str, ...]
    instructions: str
    verify: str
    forbidden: str
    # New paths the step may create, existing paths it may delete.  Older
    # META PLAN v2 answers without these sections parse to empty tuples.
    create_set: tuple[str, ...] = ()
    delete_set: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlanV2:
    decision: PlanDecision
    title: str
    objective: str
    constraints: str
    execution_mode: ExecutionMode | None
    steps: tuple[ImplementationStep, ...]
    acceptance: str
    tests: str
    risks: str
    blockers: str
    raw: str
    required_checks: tuple[str, ...] = ()
    max_step_contract_chars: int = 5000
    blocker_kind: BlockerKind | None = None


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    BLOCKED = "blocked"
    WAITING_HUMAN = "waiting_human"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    WAITING_SCOPE_APPROVAL = "waiting_scope_approval"
    WAITING_EXTERNAL = "waiting_external"
    WAITING_CHECK_INFRASTRUCTURE = "waiting_check_infrastructure"
    WAITING_CHECK_REPAIR = "waiting_check_repair"
    WAITING_REMOTE = "waiting_remote"
    WAITING_CONTRACT_REPAIR = "waiting_contract_repair"
    PLAN_REJECTED = "plan_rejected"
    WORKTREE_READY = "worktree_ready"
    PREPARING = "preparing"
    IMPLEMENTING = "implementing"
    CONTRACT_REPAIRING = "contract_repairing"
    VALIDATING = "validating"
    PRE_REVISION_VALIDATING = "pre_revision_validating"
    REVISING = "revising"
    REVALIDATING = "revalidating"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    COMMITTED = "committed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunPhase(StrEnum):
    """The durable operation a run is at: the one running, or the next one.

    A phase names an operation, never a posture.  ``DETERMINISTIC_GATE`` stays
    the phase while a red gate waits for a repaired tree, and ``CHECK_REPAIR``
    names the bounded repair pass itself.  The posture of the run is
    :class:`RunDisposition`; the business detail is the failure reason.
    """

    CONTEXT = "context"
    PLANNER = "planner"
    PLAN_APPROVAL = "plan_approval"
    WORKTREE_SETUP = "worktree_setup"
    IMPLEMENT_STEP = "implement_step"
    # A successful worker candidate is durable (``step_candidate.json``);
    # only its deterministic acceptance and commit remain.  Never a worker.
    STEP_ACCEPTANCE = "step_acceptance"
    DETERMINISTIC_GATE = "deterministic_gate"
    CHECK_REPAIR = "check_repair"
    # The cycle a red gate's cycle replan opened: its approved decomposition is
    # loaded from the durable check-replan transaction and its steps run.
    CHECK_REPLAN = "check_replan"
    SEMANTIC_REVISION = "semantic_revision"
    CANDIDATE_READY = "candidate_ready"
    CANDIDATE_PUSH = "candidate_push"
    FINAL_REVIEW = "final_review"
    REVIEW_IMPLEMENTATION = "review_implementation"
    REVIEW_REPLAN = "review_replan"
    PUBLISH = "publish"


class RunDisposition(StrEnum):
    """The only postures a durable run can be in.

    Every finer business state is a ``(phase, disposition, reason)`` triple:
    no ``WAITING_CHECK_REPAIR``, ``REVALIDATING`` or ``CONTRACT_REPAIRING``
    posture exists, because those name an operation or a failure detail.
    """

    RUNNING = "RUNNING"
    WAIT_EXTERNAL = "WAIT_EXTERNAL"
    WAIT_HUMAN = "WAIT_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def waiting(self) -> bool:
        """Whether the run is stopped until something outside it changes."""

        return self in {RunDisposition.WAIT_EXTERNAL, RunDisposition.WAIT_HUMAN}

    @property
    def terminal(self) -> bool:
        """Whether no event may ever follow this disposition."""

        return self in {RunDisposition.COMPLETED, RunDisposition.FAILED}


class RunTransitionError(ValueError):
    """One invalid run transition.  Every refusal happens in ``transition``."""


def _run_reason(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunTransitionError("run reason must be a non-empty reason code or null")
    return value.strip()


@dataclass(frozen=True)
class RunMachineState:
    """The durable identity of one run: one phase, one disposition.

    ``reason`` is the business detail (a stable failure reason code) that
    :func:`project_run_outcome` needs to name a waiting flavour.  It is never
    a status, and it never competes with the disposition.
    """

    phase: RunPhase | None = None
    disposition: RunDisposition = RunDisposition.RUNNING
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.phase is not None and not isinstance(self.phase, RunPhase):
            try:
                object.__setattr__(self, "phase", RunPhase(self.phase))
            except (TypeError, ValueError) as exc:
                raise RunTransitionError("run phase is unknown") from exc
        if not isinstance(self.disposition, RunDisposition):
            try:
                object.__setattr__(self, "disposition", RunDisposition(self.disposition))
            except (TypeError, ValueError) as exc:
                raise RunTransitionError("run disposition is unknown") from exc
        object.__setattr__(self, "reason", _run_reason(self.reason))


class RunEventKind(StrEnum):
    """What a run can be told; the strategy detail stays in the failure."""

    ADVANCE = "advance"
    WAIT = "wait"
    FAIL = "fail"
    COMPLETE = "complete"
    RESUME = "resume"


@dataclass(frozen=True)
class RunEvent:
    """One event applied to a :class:`RunMachineState`."""

    kind: RunEventKind
    target: RunPhase | None = None
    disposition: RunDisposition | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RunEventKind):
            try:
                object.__setattr__(self, "kind", RunEventKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise RunTransitionError("run event kind is unknown") from exc
        if self.target is not None and not isinstance(self.target, RunPhase):
            try:
                object.__setattr__(self, "target", RunPhase(self.target))
            except (TypeError, ValueError) as exc:
                raise RunTransitionError("run event target is unknown") from exc
        if self.disposition is not None and not isinstance(self.disposition, RunDisposition):
            try:
                object.__setattr__(self, "disposition", RunDisposition(self.disposition))
            except (TypeError, ValueError) as exc:
                raise RunTransitionError("run event disposition is unknown") from exc
        object.__setattr__(self, "reason", _run_reason(self.reason))

    @classmethod
    def advance(cls, target: RunPhase | str) -> "RunEvent":
        """The current operation succeeded; *target* is the next one."""

        return cls(RunEventKind.ADVANCE, target=target)

    @classmethod
    def wait(
        cls, disposition: RunDisposition | str, *, reason: str | None = None,
    ) -> "RunEvent":
        """The run stops without failing, at its current operation."""

        return cls(RunEventKind.WAIT, disposition=disposition, reason=reason)

    @classmethod
    def fail(cls, *, reason: str | None = None) -> "RunEvent":
        """The run stops for good, at its current operation."""

        return cls(RunEventKind.FAIL, reason=reason)

    @classmethod
    def complete(cls) -> "RunEvent":
        """The final operation of the run succeeded."""

        return cls(RunEventKind.COMPLETE)

    @classmethod
    def resume(cls) -> "RunEvent":
        """A waiting run is claimed again; its phase is unchanged."""

        return cls(RunEventKind.RESUME)


# The one phase graph of the pipeline: the operation each phase may hand over
# to.  Self-transitions are explicit: a cycle implements several steps, a gate
# episode runs several attempts.
_RUN_PHASE_SUCCESSORS: Mapping[RunPhase, frozenset[RunPhase]] = {
    RunPhase.CONTEXT: frozenset({RunPhase.PLANNER}),
    RunPhase.PLANNER: frozenset({RunPhase.PLAN_APPROVAL}),
    RunPhase.PLAN_APPROVAL: frozenset({RunPhase.WORKTREE_SETUP}),
    RunPhase.WORKTREE_SETUP: frozenset({RunPhase.IMPLEMENT_STEP, RunPhase.REVIEW_IMPLEMENTATION}),
    RunPhase.IMPLEMENT_STEP: frozenset({
        RunPhase.IMPLEMENT_STEP, RunPhase.STEP_ACCEPTANCE,
        RunPhase.DETERMINISTIC_GATE, RunPhase.REVIEW_IMPLEMENTATION,
    }),
    RunPhase.STEP_ACCEPTANCE: frozenset({RunPhase.IMPLEMENT_STEP, RunPhase.DETERMINISTIC_GATE}),
    RunPhase.DETERMINISTIC_GATE: frozenset({
        RunPhase.DETERMINISTIC_GATE, RunPhase.CHECK_REPAIR,
        RunPhase.SEMANTIC_REVISION, RunPhase.CANDIDATE_READY, RunPhase.CHECK_REPLAN,
    }),
    RunPhase.CHECK_REPAIR: frozenset({RunPhase.CHECK_REPAIR, RunPhase.DETERMINISTIC_GATE}),
    # A new decomposition is implemented by its own cycle and returns to the
    # deterministic gate, exactly like every other cycle of the pipeline.
    RunPhase.CHECK_REPLAN: frozenset({RunPhase.IMPLEMENT_STEP, RunPhase.DETERMINISTIC_GATE}),
    RunPhase.SEMANTIC_REVISION: frozenset({RunPhase.DETERMINISTIC_GATE, RunPhase.CANDIDATE_READY}),
    RunPhase.CANDIDATE_READY: frozenset({RunPhase.CANDIDATE_PUSH}),
    RunPhase.CANDIDATE_PUSH: frozenset({RunPhase.FINAL_REVIEW}),
    RunPhase.FINAL_REVIEW: frozenset({
        RunPhase.PUBLISH, RunPhase.SEMANTIC_REVISION,
        RunPhase.REVIEW_IMPLEMENTATION, RunPhase.REVIEW_REPLAN,
    }),
    RunPhase.REVIEW_IMPLEMENTATION: frozenset({
        RunPhase.REVIEW_IMPLEMENTATION, RunPhase.DETERMINISTIC_GATE, RunPhase.SEMANTIC_REVISION,
    }),
    RunPhase.REVIEW_REPLAN: frozenset({
        RunPhase.IMPLEMENT_STEP, RunPhase.REVIEW_IMPLEMENTATION, RunPhase.DETERMINISTIC_GATE,
    }),
    RunPhase.PUBLISH: frozenset(),
}
# The operations that can end a run successfully: the reviewed candidate
# commit itself, or its publication.
_RUN_COMPLETABLE_PHASES = frozenset({RunPhase.CANDIDATE_PUSH, RunPhase.PUBLISH})


def phase_successors(phase: RunPhase | str) -> frozenset[RunPhase]:
    """The operations *phase* may legally hand over to."""

    try:
        return _RUN_PHASE_SUCCESSORS[RunPhase(phase)]
    except (TypeError, ValueError) as exc:
        raise RunTransitionError("run phase is unknown") from exc


def transition(current: RunMachineState, event: RunEvent) -> RunMachineState:
    """Apply *event* to *current*; the single place an invalid one fails.

    Pure: it reads nothing durable, writes nothing, and returns the next
    state.  Every illegal combination raises :class:`RunTransitionError`
    here, so no caller ever needs its own compatibility matrix.
    """

    if not isinstance(current, RunMachineState):
        raise RunTransitionError("transition expects a RunMachineState")
    if not isinstance(event, RunEvent):
        raise RunTransitionError("transition expects a RunEvent")
    if current.disposition.terminal:
        raise RunTransitionError(
            f"a {current.disposition.value} run accepts no further event"
        )
    if event.kind is RunEventKind.ADVANCE:
        if event.target is None:
            raise RunTransitionError("advance requires a target phase")
        if current.disposition is not RunDisposition.RUNNING:
            raise RunTransitionError(
                f"a {current.disposition.value} run must be resumed before it advances"
            )
        if current.phase is None:
            raise RunTransitionError("a run without a phase cannot advance")
        if event.target not in _RUN_PHASE_SUCCESSORS[current.phase]:
            raise RunTransitionError(
                f"{current.phase.value} never advances to {event.target.value}"
            )
        return RunMachineState(event.target, RunDisposition.RUNNING)
    if event.kind is RunEventKind.WAIT:
        if event.disposition is None or not event.disposition.waiting:
            raise RunTransitionError(
                "a wait event requires the WAIT_EXTERNAL or WAIT_HUMAN disposition"
            )
        return RunMachineState(current.phase, event.disposition, event.reason)
    if event.kind is RunEventKind.FAIL:
        return RunMachineState(current.phase, RunDisposition.FAILED, event.reason)
    if event.kind is RunEventKind.COMPLETE:
        if current.disposition is not RunDisposition.RUNNING:
            raise RunTransitionError(
                f"a {current.disposition.value} run must be resumed before it completes"
            )
        if current.phase not in _RUN_COMPLETABLE_PHASES:
            raise RunTransitionError(
                "only a reviewed candidate push or its publication completes a run"
            )
        return RunMachineState(current.phase, RunDisposition.COMPLETED)
    if event.kind is RunEventKind.RESUME:
        if not current.disposition.waiting:
            raise RunTransitionError(
                f"a {current.disposition.value} run is not waiting and cannot be resumed"
            )
        return RunMachineState(current.phase, RunDisposition.RUNNING)
    raise RunTransitionError(f"run event {event.kind!r} is unknown")


# -- the single derived projection -------------------------------------------

# The status of a running phase.  It is the only place these strings
# are chosen; every waiting or terminal status is derived below.
_RUNNING_STATUS: Mapping[RunPhase, RunStatus] = {
    RunPhase.CONTEXT: RunStatus.PLANNING,
    RunPhase.PLANNER: RunStatus.PLANNING,
    RunPhase.PLAN_APPROVAL: RunStatus.AWAITING_PLAN_APPROVAL,
    RunPhase.WORKTREE_SETUP: RunStatus.PREPARING,
    RunPhase.IMPLEMENT_STEP: RunStatus.IMPLEMENTING,
    RunPhase.STEP_ACCEPTANCE: RunStatus.IMPLEMENTING,
    RunPhase.DETERMINISTIC_GATE: RunStatus.VALIDATING,
    RunPhase.CHECK_REPAIR: RunStatus.REVISING,
    RunPhase.CHECK_REPLAN: RunStatus.IMPLEMENTING,
    RunPhase.SEMANTIC_REVISION: RunStatus.REVISING,
    RunPhase.CANDIDATE_READY: RunStatus.APPROVED,
    RunPhase.CANDIDATE_PUSH: RunStatus.APPROVED,
    RunPhase.FINAL_REVIEW: RunStatus.REVIEWING,
    RunPhase.REVIEW_IMPLEMENTATION: RunStatus.IMPLEMENTING,
    RunPhase.REVIEW_REPLAN: RunStatus.PLANNING,
    RunPhase.PUBLISH: RunStatus.PUBLISHING,
}
_COMPLETED_STATUS: Mapping[RunPhase, RunStatus] = {
    RunPhase.CANDIDATE_PUSH: RunStatus.COMMITTED,
    RunPhase.PUBLISH: RunStatus.PUBLISHED,
}
# Failure reasons that name *what* an external wait waits for; the phase stays
# the operation to retry, the reason carries the business meaning.
_CHECK_INFRASTRUCTURE_REASONS = frozenset({
    "CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE",
    "CHECK_INFRA_RETRIES_EXHAUSTED", "CHECK_INFRASTRUCTURE_UNAVAILABLE",
    "CHECK_SIDE_EFFECT_REPEATED", "CHECK_SIDE_EFFECT_UNSTABLE",
    "WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT",
})
_REMOTE_REASONS = frozenset({
    "PUSH_FAILED", "CANDIDATE_PUSH_FAILED", "CANDIDATE_REMOTE_UNAVAILABLE",
    "REMOTE_UNAVAILABLE", "REMOTE_TEMPORARILY_UNAVAILABLE",
})
_REMOTE_PHASES = frozenset({RunPhase.CANDIDATE_PUSH, RunPhase.PUBLISH})
# A human wait that still owns a durable retry slot: the bounded repair pass
# it refused is pending, so an operator retry resumes exactly that pass.
_REPAIR_SLOT_WAITS: Mapping[tuple[str, RunPhase], RunStatus] = {
    ("STEP_CONTRACT_REPAIR_OUTPUT_INVALID", RunPhase.IMPLEMENT_STEP): RunStatus.WAITING_CONTRACT_REPAIR,
    ("CHECK_REPAIR_EXHAUSTED", RunPhase.DETERMINISTIC_GATE): RunStatus.WAITING_CHECK_REPAIR,
}
# A human wait that owns a durable operator gate: the exact delta of the
# expansion is pending, so the operator's answer resumes the same operation.
_SCOPE_APPROVAL_REASON = "WAITING_SCOPE_APPROVAL"


@dataclass(frozen=True)
class RunOutcome:
    """The derived status view of one durable run state."""

    phase: RunPhase | None
    disposition: RunDisposition
    status: RunStatus
    # A durable retry boundary remains for exactly this outcome.
    resumable: bool
    # A resume may claim the run; the failure reason decides whether it may.
    resume_eligible: bool


def project_run_outcome(state: RunMachineState) -> RunOutcome:
    """Derive the durable status of one run state; never stored as a source.

    Pure and total: the phase says which operation, the disposition says the
    posture, and the failure reason names the waiting flavour.  No caller
    derives a status, and no caller compares a status to a phase.
    """

    if not isinstance(state, RunMachineState):
        raise RunTransitionError("project_run_outcome expects a RunMachineState")
    phase, disposition, reason = state.phase, state.disposition, state.reason
    if disposition is RunDisposition.RUNNING:
        status = _RUNNING_STATUS[phase] if phase is not None else RunStatus.CREATED
        return RunOutcome(phase, disposition, status, False, False)
    if disposition is RunDisposition.COMPLETED:
        return RunOutcome(
            phase, disposition, _COMPLETED_STATUS.get(phase, RunStatus.PUBLISHED), False, False,
        )
    if disposition is RunDisposition.FAILED:
        # The failure reason decides whether a resume may still claim the run.
        return RunOutcome(phase, disposition, RunStatus.FAILED, False, True)
    if disposition is RunDisposition.WAIT_EXTERNAL:
        if reason in _CHECK_INFRASTRUCTURE_REASONS:
            status = RunStatus.WAITING_CHECK_INFRASTRUCTURE
        elif reason in _REMOTE_REASONS and phase in _REMOTE_PHASES:
            status = RunStatus.WAITING_REMOTE
        else:
            status = RunStatus.WAITING_EXTERNAL
        return RunOutcome(phase, disposition, status, True, True)
    slot = _REPAIR_SLOT_WAITS.get((reason or "", phase))
    if slot is not None:
        return RunOutcome(phase, disposition, slot, True, True)
    if reason == _SCOPE_APPROVAL_REASON:
        return RunOutcome(phase, disposition, RunStatus.WAITING_SCOPE_APPROVAL, True, True)
    return RunOutcome(phase, disposition, RunStatus.WAITING_HUMAN, False, False)

# The single status bridge: a durable status names exactly one disposition.
# Nothing else maps a status onto a disposition.
_STATUS_DISPOSITIONS: Mapping[RunStatus, RunDisposition] = {
    RunStatus.CREATED: RunDisposition.RUNNING,
    RunStatus.PLANNING: RunDisposition.RUNNING,
    RunStatus.BLOCKED: RunDisposition.WAIT_HUMAN,
    RunStatus.WAITING_HUMAN: RunDisposition.WAIT_HUMAN,
    RunStatus.AWAITING_PLAN_APPROVAL: RunDisposition.RUNNING,
    RunStatus.WAITING_SCOPE_APPROVAL: RunDisposition.WAIT_HUMAN,
    RunStatus.WAITING_EXTERNAL: RunDisposition.WAIT_EXTERNAL,
    RunStatus.WAITING_CHECK_INFRASTRUCTURE: RunDisposition.WAIT_EXTERNAL,
    RunStatus.WAITING_CHECK_REPAIR: RunDisposition.WAIT_HUMAN,
    RunStatus.WAITING_REMOTE: RunDisposition.WAIT_EXTERNAL,
    RunStatus.WAITING_CONTRACT_REPAIR: RunDisposition.WAIT_HUMAN,
    RunStatus.PLAN_REJECTED: RunDisposition.WAIT_HUMAN,
    RunStatus.WORKTREE_READY: RunDisposition.RUNNING,
    RunStatus.PREPARING: RunDisposition.RUNNING,
    RunStatus.IMPLEMENTING: RunDisposition.RUNNING,
    RunStatus.CONTRACT_REPAIRING: RunDisposition.RUNNING,
    RunStatus.VALIDATING: RunDisposition.RUNNING,
    RunStatus.PRE_REVISION_VALIDATING: RunDisposition.RUNNING,
    RunStatus.REVISING: RunDisposition.RUNNING,
    RunStatus.REVALIDATING: RunDisposition.RUNNING,
    RunStatus.REVIEWING: RunDisposition.RUNNING,
    RunStatus.APPROVED: RunDisposition.RUNNING,
    RunStatus.PUBLISHING: RunDisposition.RUNNING,
    RunStatus.PUBLISHED: RunDisposition.COMPLETED,
    RunStatus.COMMITTED: RunDisposition.COMPLETED,
    RunStatus.FAILED: RunDisposition.FAILED,
    RunStatus.INTERRUPTED: RunDisposition.FAILED,
}


# A waiting status names the operator gate it stopped at when the failure
# reason alone does not carry it.
_STATUS_WAIT_REASONS: Mapping[RunStatus, str] = {
    RunStatus.WAITING_SCOPE_APPROVAL: "WAITING_SCOPE_APPROVAL",
}


def disposition_for_status(status: RunStatus | str) -> RunDisposition:
    """The disposition a durable run status stands for."""

    try:
        return _STATUS_DISPOSITIONS[RunStatus(status)]
    except (TypeError, ValueError) as exc:
        raise RunTransitionError(f"run status {status!r} is unknown") from exc


def wait_reason_for_status(status: RunStatus | str) -> str | None:
    """The operator gate a durable waiting status stands for."""

    try:
        return _STATUS_WAIT_REASONS.get(RunStatus(status))
    except (TypeError, ValueError):
        return None


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    REVISE = "REVISE"
    FAIL = "FAIL"


class ReviewRoute(StrEnum):
    NONE = "NONE"
    IMPLEMENTATION = "IMPLEMENTATION"
    REPLAN = "REPLAN"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class LLMEndpointConfig:
    base_url: str
    endpoint_path: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: int = 300
    retries: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextConfig:
    always_files: tuple[str, ...] = (
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
    )
    locator_argv: tuple[str, ...] = ()
    locator_timeout_seconds: int = 120
    max_hits: int = 8
    max_bytes: int = 160_000
    require_locator_head_at_base: bool = True


@dataclass(frozen=True)
class RepositoryConfig:
    remote: str = "origin"
    planner_remote_exploration: bool = True
    web_url: str | None = None


@dataclass(frozen=True)
class WorkstreamRef:
    """Stable references that identify one MetaHarness workstream.

    This is deliberately smaller than the durable Git state.  It is useful at
    integration boundaries (for example GitHub) without becoming a second
    state representation.
    """

    run_id: str
    base_ref: str
    base_sha: str
    local_branch: str
    worktree: str
    remote_branch: str | None = None
    issue_number: int | None = None
    pull_request_number: int | None = None


@dataclass(frozen=True)
class GitHubConfig:
    """Optional, credential-free configuration for workstream metadata."""

    enabled: bool = False
    issue_mode: str = "off"
    pull_request_mode: str = "off"
    issue_number: int | None = None
    # This is only an environment-variable name.  The value is never part of
    # config serialization, run options, state, trace, or diagnostics.
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("github.enabled must be a boolean")
        if not isinstance(self.issue_mode, str) or self.issue_mode not in {
            "off", "link-existing", "create"
        }:
            raise ValueError("github.issue_mode is invalid")
        if not isinstance(self.pull_request_mode, str) or self.pull_request_mode not in {
            "off", "create"
        }:
            raise ValueError("github.pull_request_mode is invalid")
        if self.issue_number is not None and (
            isinstance(self.issue_number, bool)
            or not isinstance(self.issue_number, int)
            or self.issue_number <= 0
        ):
            raise ValueError("github.issue_number must be a positive integer or null")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)
        ):
            raise ValueError("github.api_key_env must be an environment variable name or null")


class ExecutionModePolicy(StrEnum):
    """Planner freedom over EXECUTION_MODE, fixed by configuration."""

    AUTO = "auto"
    REQUIRE_STAGED = "require-staged"


class PublishMode(StrEnum):
    """Where a final reviewed commit is published."""

    # Push only the run branch ``harness/<plan>/<run-id>``.
    RUN_BRANCH = "run-branch"
    # Compare-and-swap fast-forward of the local base branch to the reviewed
    # commit, then push that exact commit to the remote base branch.
    FAST_FORWARD_BASE = "fast-forward-base"


@dataclass(frozen=True)
class PublishConfig:
    enabled: bool = False
    remote: str = "origin"
    mode: str = "run-branch"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("publish.enabled must be a boolean")
        if (
            not isinstance(self.remote, str)
            or not self.remote.strip()
            or any(char.isspace() for char in self.remote)
            or "\x00" in self.remote
            or self.remote.startswith("-")
        ):
            raise ValueError("publish.remote must be a valid remote name")
        if self.mode not in {item.value for item in PublishMode}:
            raise ValueError("publish.mode must be 'run-branch' or 'fast-forward-base'")


@dataclass(frozen=True)
class PlanningConfig:
    protocol: str = "v2"
    decomposition: str = "aggressive"
    single_step_max_mutable_paths: int = 2
    staged_step_max_mutable_paths: int = 5
    execution_mode_policy: str = "auto"
    max_steps_per_plan: int = 8
    max_read_paths_per_step: int = 8
    max_step_contract_chars: int = 5000
    max_preapproval_corrections: int = 2

    def __post_init__(self) -> None:
        if self.protocol != "v2":
            raise ValueError("planning protocol must be 'v2'")
        if self.decomposition not in {"balanced", "aggressive"}:
            raise ValueError("planning decomposition must be 'balanced' or 'aggressive'")
        for name in ("single_step_max_mutable_paths", "staged_step_max_mutable_paths"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer greater than zero")
        for name in ("max_steps_per_plan", "max_read_paths_per_step", "max_step_contract_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer greater than zero")
        if self.max_steps_per_plan > 99:
            raise ValueError("max_steps_per_plan must not exceed the protocol maximum of 99")
        validate_revision_budget(self.max_preapproval_corrections, "max_preapproval_corrections")
        if self.execution_mode_policy not in {item.value for item in ExecutionModePolicy}:
            raise ValueError("planning execution_mode_policy must be 'auto' or 'require-staged'")


MAX_REVISION_BUDGET = 10


def validate_revision_budget(value: int, name: str) -> int:
    """Validate one bounded correction budget in one central place."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_REVISION_BUDGET
    ):
        raise ValueError(
            f"{name} must be an integer between 0 and {MAX_REVISION_BUDGET}"
        )
    return value


@dataclass(frozen=True)
class RevisionConfig:
    """Independent, bounded semantic-revision and check-repair budgets.

    The default is no correction pipeline: every budget needs its profile.
    """

    enabled: bool = False
    max_review_repair_cycles: int = 0
    max_check_repair_attempts: int = 0
    max_step_contract_repairs: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("revision.enabled must be a boolean")
        validate_revision_budget(
            self.max_review_repair_cycles, "revision.max_review_repair_cycles"
        )
        validate_revision_budget(
            self.max_check_repair_attempts, "revision.max_check_repair_attempts"
        )
        validate_revision_budget(
            self.max_step_contract_repairs, "revision.max_step_contract_repairs"
        )

@dataclass(frozen=True)
class PromptBudgetConfig:
    """Byte budgets for role-specific model payloads.

    Authority sections are never truncated.  A payload may therefore record
    a soft overrun when the authority alone is larger than its configured
    budget.
    """

    planner_max_bytes: int = 160_000
    implementer_max_bytes: int = 120_000
    check_repair_max_bytes: int = 40_000
    semantic_revision_max_bytes: int = 120_000
    final_review_max_bytes: int = 120_000

    def __post_init__(self) -> None:
        for name in (
            "planner_max_bytes",
            "implementer_max_bytes",
            "check_repair_max_bytes",
            "semantic_revision_max_bytes",
            "final_review_max_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer greater than zero")


@dataclass(frozen=True)
class AgentConfig:
    """Codex harness settings derived from one codex ``ModelProfile``."""

    model: str = "gpt-5.6-luna"
    provider: str = "openai"
    provider_base_url: str | None = None
    provider_wire_api: str | None = None
    provider_api_key_env: str | None = None
    effort: str = "high"
    sandbox: str = "workspace-write"
    timeout_seconds: int = 5400
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
    )


@dataclass(frozen=True)
class ApprovalConfig:
    require_plan_approval: bool = False
    poll_interval_seconds: float = 0.5


@dataclass(frozen=True)
class UIConfig:
    max_active_runs: int = 1
    default_planner_profile: str | None = None
    default_reviewer_profile: str | None = None
    default_reviser_profile: str | None = None
    default_repair_profile: str | None = None
    enable_profile_recommendation: bool = True


@dataclass(frozen=True)
class CheckConfig:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 3600
    required: bool = True
    preflight_argv: tuple[str, ...] = ()
    description: str = ""

    @property
    def id(self) -> str:
        """Stable trusted catalogue identifier."""

        return self.name



@dataclass(frozen=True)
class EnvironmentConfig:
    files: tuple[Path, ...] = ()


@dataclass(frozen=True)
class CodexRuntimeConfig:
    home: Path
    env_allowlist: tuple[str, ...] = AgentConfig.env_allowlist


@dataclass(frozen=True)
class ClaudeRuntimeConfig:
    home: Path


@dataclass(frozen=True)
class WorkspaceSetupCommand:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: int = 1200
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "PNPM_HOME",
    )


@dataclass(frozen=True)
class HarnessConfig:
    repo: Path
    base_ref: str
    runs_root: Path
    worktrees_root: Path
    require_clean_base: bool
    context: ContextConfig
    check_catalog: tuple[CheckConfig, ...]
    max_diff_bytes: int = 400_000
    allow_no_required_checks: bool = False
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    model_profiles: Mapping[str, ModelProfile] = field(default_factory=dict)
    routing: RoutingConfig = field(
        default_factory=RoutingConfig
    )
    codex_providers: Mapping[str, CodexProviderConfig] = field(default_factory=dict)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runtime_environment: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    codex_runtime: CodexRuntimeConfig = field(
        default_factory=lambda: CodexRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "codex"
        )
    )
    claude_runtime: ClaudeRuntimeConfig = field(
        default_factory=lambda: ClaudeRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "claude"
        )
    )
    workspace_setup: tuple[WorkspaceSetupCommand, ...] = ()
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)
    recovery: RecoveryBudgets = field(default_factory=RecoveryBudgets)
    prompt_budget: PromptBudgetConfig = field(default_factory=PromptBudgetConfig)
    repository: RepositoryConfig = field(default_factory=RepositoryConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    repository_section_explicit: bool = field(default=False, repr=False, compare=False)
    default_check_ids: tuple[str, ...] = ()

    def trusted_checks(self) -> tuple[CheckConfig, ...]:
        return self.check_catalog

    def trusted_check_map(self) -> dict[str, CheckConfig]:
        return {check.id: check for check in self.trusted_checks()}

    def required_check_ids(self) -> tuple[str, ...]:
        configured = self.default_check_ids
        if configured:
            return configured
        return tuple(check.id for check in self.trusted_checks() if check.required)

    def select_checks(self, ids: tuple[str, ...] | list[str] | None = None) -> tuple[CheckConfig, ...]:
        catalogue = self.trusted_check_map()
        if ids is None:
            selected = self.required_check_ids()
        else:
            selected = tuple(ids)
        if len(set(selected)) != len(selected):
            raise ValueError("required check IDs must be unique")
        unknown = [check_id for check_id in selected if check_id not in catalogue]
        if unknown:
            raise ValueError("unknown trusted check ID(s): " + ", ".join(unknown))
        return tuple(catalogue[check_id] for check_id in selected)


@dataclass(frozen=True)
class SelectedProfile:
    profile_id: str
    driver: str
    model: str
    selection_mode: str
    effort: str | None = None
    sandbox: str | None = None
    permission_mode: str | None = None
    # SHA-256 of the execution-relevant profile configuration (schema 2).
    config_sha256: str | None = None
    provider: str = "openai"


@dataclass(frozen=True)
class StepExecutionSelection:
    """The immutable implementer snapshot selected for one v2 step."""

    step_id: str
    implementer: SelectedProfile
    execution_class: ExecutionClass = ExecutionClass.MECHANICAL
    fallbacks: tuple[SelectedProfile, ...] = ()


@dataclass(frozen=True)
class ExecutionSelection:
    """The complete immutable execution authority for one v2 run."""

    schema_version: int
    planner: SelectedProfile
    steps: tuple[StepExecutionSelection, ...]
    check_repair: SelectedProfile | None
    semantic_reviser: SelectedProfile | None
    final_reviewer: SelectedProfile
    check_repair_fallbacks: tuple[SelectedProfile, ...] = ()
    semantic_reviser_fallbacks: tuple[SelectedProfile, ...] = ()


@dataclass(frozen=True)
class CycleExecutionSelection:
    """Immutable implementer selections for one review-replan cycle."""

    schema_version: int
    cycle: int
    steps: tuple[StepExecutionSelection, ...]


class CycleKind(StrEnum):
    """Why one pipeline cycle exists; every cycle names its kind explicitly."""

    INITIAL = "initial"
    REVIEW_IMPLEMENTATION = "review-implementation"
    REVIEW_REPLAN = "review-replan"
    # A red deterministic gate whose bounded repair pass and single-step
    # replan were both durably spent, so the decomposition itself is replanned.
    CHECK_REPLAN = "check-replan"


class GateStage(StrEnum):
    """Why one deterministic gate episode runs.

    ``(review_cycle, stage)`` identifies exactly one gate episode.
    """

    POST_IMPLEMENTATION = "POST_IMPLEMENTATION"
    POST_SEMANTIC_REVISION = "POST_SEMANTIC_REVISION"
    POST_CHECK_REPLAN = "POST_CHECK_REPLAN"
    POST_REVIEW_IMPLEMENTATION = "POST_REVIEW_IMPLEMENTATION"
    POST_REVIEW_REPLAN = "POST_REVIEW_REPLAN"


@dataclass(frozen=True)
class RunCycle:
    """One generic orchestration cycle; its budget is configured elsewhere."""

    number: int
    kind: CycleKind

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 1:
            raise ValueError("run cycle number must be a positive integer")
        try:
            object.__setattr__(self, "kind", CycleKind(self.kind))
        except ValueError as exc:
            raise ValueError("run cycle kind is unknown") from exc
