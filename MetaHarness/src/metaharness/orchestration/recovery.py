"""Project classified recovery outcomes onto durable run states."""

from dataclasses import dataclass

from ..models import RunStatus
from ..recovery_policy import RecoveryDecision, RecoveryDisposition
from ..resume import ResumePhase


@dataclass(frozen=True)
class RecoveryTerminalState:
    status: RunStatus
    resumable: bool
    reason: str


_CHECK_INFRA = frozenset({
    "CHECK_TIMEOUT", "CHECK_PREFLIGHT_FAILED", "CHECK_INFRA_FAILURE",
    "CHECK_INFRA_RETRIES_EXHAUSTED", "CHECK_INFRASTRUCTURE_UNAVAILABLE",
    "CHECK_SIDE_EFFECT_REPEATED", "CHECK_SIDE_EFFECT_UNSTABLE",
    "WORKSPACE_SETUP_FAILED", "WORKSPACE_SETUP_TIMEOUT",
})
_REMOTE = frozenset({
    "PUSH_FAILED", "CANDIDATE_PUSH_FAILED", "CANDIDATE_REMOTE_UNAVAILABLE",
    "REMOTE_UNAVAILABLE", "REMOTE_TEMPORARILY_UNAVAILABLE",
})


def terminal_state_for(
    decision: RecoveryDecision, *, failure_code: str, phase: ResumePhase,
) -> RecoveryTerminalState:
    """Only terminal dispositions may cross the coordinator boundary."""

    code = failure_code.split(":", 1)[0].upper()
    if decision.disposition is RecoveryDisposition.HARD_STOP:
        return RecoveryTerminalState(RunStatus.FAILED, False, decision.reason)
    if decision.disposition is RecoveryDisposition.WAIT_HUMAN:
        return RecoveryTerminalState(RunStatus.WAITING_HUMAN, False, decision.reason)
    if decision.disposition is RecoveryDisposition.WAIT_EXTERNAL:
        if code in _CHECK_INFRA:
            status = RunStatus.WAITING_CHECK_INFRASTRUCTURE
        elif code in _REMOTE and phase is ResumePhase.CANDIDATE_PUSH:
            status = RunStatus.WAITING_REMOTE
        else:
            status = RunStatus.WAITING_EXTERNAL
        return RecoveryTerminalState(status, True, decision.reason)
    raise ValueError(f"recovery disposition {decision.disposition} is not terminal")


__all__ = ["RecoveryTerminalState", "terminal_state_for"]
