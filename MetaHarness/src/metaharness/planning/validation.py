"""Plan constraints: execution mode, decomposition and repository preconditions.

This module owns the deterministic policy the harness applies to a parsed plan
and the policy text the planner is asked to honour.  It never calls a model.
"""

from __future__ import annotations

from typing import Sequence

from ..models import (
    ExecutionMode,
    ExecutionModePolicy,
    ImplementationStep,
    PlanDecision,
    PlanningConfig,
    TaskPlanV2,
)
from ..plan_repository_validation import (
    PathPreconditionViolation,
    RepositoryPreconditions,
    plan_repository_violations,
    render_conflict_evidence,
    render_precondition_correction,
)
from ..step_ids import MAX_STEPS
from .protocol import V2PlanParseError

_PROTOCOL_ANCHOR = "The answer must use exactly this protocol."


def render_require_staged_policy_text(max_steps_per_plan: int = PlanningConfig.max_steps_per_plan) -> str:
    if isinstance(max_steps_per_plan, bool) or not isinstance(max_steps_per_plan, int):
        raise ValueError("max_steps_per_plan must be an integer")
    if not 2 <= max_steps_per_plan <= MAX_STEPS:
        raise ValueError("max_steps_per_plan must be between 2 and 99 for STAGED policy")
    return f"""This run REQUIRES STAGED execution.

You must return between 2 and {max_steps_per_plan} coherent implementation steps.

Do not create artificial "implementation then tests" steps when tests belong
to the same local behavior.

Instead decompose by independently understandable behavioral transformation,
layer, subsystem, or dependency boundary.

Each worker must receive a mechanically executable contract and must not need
architectural discovery.
"""



def render_decomposition_policy_text(
    single_step_max_mutable_paths: int, staged_step_max_mutable_paths: int
) -> str:
    """Render the AGGRESSIVE mutable-scope policy from the configured limits.

    The numbers come from :class:`PlanningConfig`, the same values that
    :func:`validate_decomposition_policy` enforces after parsing.
    """

    for name, value in (
        ("single_step_max_mutable_paths", single_step_max_mutable_paths),
        ("staged_step_max_mutable_paths", staged_step_max_mutable_paths),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an integer greater than zero")
    single = single_step_max_mutable_paths
    staged = staged_step_max_mutable_paths
    return f"""This run uses AGGRESSIVE decomposition.

A READY SINGLE plan may modify at most {single} distinct mutable paths across
the union of WRITE_SET, CREATE_SET and DELETE_SET.

Every STAGED step may modify at most {staged} distinct mutable paths across the
union of WRITE_SET, CREATE_SET and DELETE_SET.

This limit applies to the UNION of the three sets, not to each section
independently. A path counts once in the union. A READY plan must never exceed
the active limit; the harness rejects it deterministically.

The mutable-path limit bounds the scope of one worker; it is not an order to
fragment an atomic operation unsafely. When a transformation exceeds the
limit, decompose it only where a mechanically coherent decomposition exists:
by transformation, layer, or dependency boundary. Do not create an artificial
"implementation" step followed by a "tests" step to satisfy the limit; tests
directly associated with a local transformation stay in the same step. Every
step must leave the repository in a coherent state for the next step.

A large task is not in itself a reason to return BLOCKED. BLOCKED remains
reserved for when the architecture, paths or operations cannot be determined
precisely, or when one atomic operation needs more mutable paths than the
limit and no safe decomposition exists. In that case return BLOCKED instead
of an invalid plan or a plan that asks the worker to discover a solution.
"""


def render_repair_decomposition_policy_text(
    staged_step_max_mutable_paths: int,
) -> str:
    """Render the correction mutable-scope policy from its one real limit.

    A bounded correction step is bounded by what a single worker is already
    allowed to touch, so the STAGED per-step maximum is the authority here --
    for a SINGLE ``S01`` exactly as for a STAGED step.
    """

    if (
        isinstance(staged_step_max_mutable_paths, bool)
        or not isinstance(staged_step_max_mutable_paths, int)
        or staged_step_max_mutable_paths <= 0
    ):
        raise ValueError(
            "staged_step_max_mutable_paths "
            "must be an integer greater than zero"
        )

    return f"""REPAIR DECOMPOSITION POLICY

This is a bounded correction plan.

Every repair implementation step, including a SINGLE S01,
may modify at most {staged_step_max_mutable_paths} distinct mutable
paths across the UNION of WRITE_SET, CREATE_SET and DELETE_SET.

A path counts once in that union.

If one corrective step requires more than
{staged_step_max_mutable_paths} mutable paths, decompose it into
multiple ordered STAGED steps.

Do not return a READY repair plan containing any step above this limit.
"""


def insert_before_protocol(prompt: str, text: str) -> str:
    index = prompt.find(_PROTOCOL_ANCHOR)
    if index < 0:
        return prompt.rstrip("\n") + "\n\n" + text
    return prompt[:index] + text + "\n" + prompt[index:]


def validate_execution_mode_policy(plan: TaskPlanV2, planning: PlanningConfig) -> None:
    """Fail closed on a READY SINGLE plan when STAGED is required.

    A BLOCKED plan is always allowed: the policy constrains decomposition,
    never the planner's ability to refuse an under-specified SPEC.
    """

    if not isinstance(plan, TaskPlanV2) or not isinstance(planning, PlanningConfig):
        raise TypeError("plan and planning must be v2 model values")
    if planning.execution_mode_policy != ExecutionModePolicy.REQUIRE_STAGED.value:
        return
    if plan.decision is PlanDecision.READY and plan.execution_mode is not ExecutionMode.STAGED:
        raise V2PlanParseError("execution policy requires STAGED")


def validate_decomposition_policy(
    plan: TaskPlanV2,
    planning: PlanningConfig,
) -> None:
    """Apply the configured mutable-scope policy after strict v2 parsing."""

    if not isinstance(plan, TaskPlanV2) or not isinstance(planning, PlanningConfig):
        raise TypeError("plan and planning must be v2 model values")
    if plan.decision is not PlanDecision.READY or planning.decomposition != "aggressive":
        return

    def mutable_count(step: ImplementationStep) -> int:
        return len(set(step.write_set) | set(step.create_set) | set(step.delete_set))

    if plan.execution_mode is ExecutionMode.SINGLE:
        limit = planning.single_step_max_mutable_paths
    else:
        limit = planning.staged_step_max_mutable_paths
    mode = plan.execution_mode.value if plan.execution_mode else "UNKNOWN"
    for step in plan.steps:
        count = mutable_count(step)
        if count > limit:
            raise V2PlanParseError(
                f"aggressive {mode} step {step.id} may modify at most {limit} "
                f"distinct mutable paths; got {count}"
            )


def validate_repair_decomposition_policy(
    plan: TaskPlanV2,
    planning: PlanningConfig,
) -> None:
    """Bound every correction step by the normal staged worker limit.

    The SINGLE limit of :func:`validate_decomposition_policy` decides when an
    *initial* task must be decomposed.  A correction is already bounded to one
    corrective cycle, to the approved repair scope, to a reviewed immutable
    candidate and to the deterministic gates plus reviewer #2 that follow it,
    so the only remaining question is whether one worker may touch that many
    paths -- and ``staged_step_max_mutable_paths`` already answers it.  The
    execution mode therefore does not change this limit.
    """

    if not isinstance(plan, TaskPlanV2) or not isinstance(
        planning,
        PlanningConfig,
    ):
        raise TypeError(
            "plan and planning must be v2 model values"
        )

    if (
        plan.decision is not PlanDecision.READY
        or planning.decomposition != "aggressive"
    ):
        return

    limit = planning.staged_step_max_mutable_paths

    for step in plan.steps:
        count = len(
            set(step.write_set)
            | set(step.create_set)
            | set(step.delete_set)
        )

        if count > limit:
            raise V2PlanParseError(
                f"aggressive repair step {step.id} "
                f"may modify at most {limit} "
                f"distinct mutable paths; got {count}"
            )


def plan_precondition_violations(
    preconditions: RepositoryPreconditions | None, plan: TaskPlanV2,
) -> tuple[PathPreconditionViolation, ...]:
    if preconditions is None:
        return ()
    return plan_repository_violations(preconditions.repo, preconditions.start_tree_sha, plan)


def render_plan_precondition_correction(
    preconditions: RepositoryPreconditions,
    violations: Sequence[PathPreconditionViolation],
    previous_raw: str,
) -> str:
    return render_precondition_correction(
        violations,
        previous_raw=previous_raw,
        evidence=render_conflict_evidence(
            preconditions.repo, preconditions.start_tree_sha, violations,
        ),
    )


__all__ = [
    "insert_before_protocol",
    "plan_precondition_violations",
    "render_decomposition_policy_text",
    "render_plan_precondition_correction",
    "render_repair_decomposition_policy_text",
    "render_require_staged_policy_text",
    "validate_decomposition_policy",
    "validate_execution_mode_policy",
    "validate_repair_decomposition_policy",
]
