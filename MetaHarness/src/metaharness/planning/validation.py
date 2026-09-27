"""Plan constraints: execution mode, granularity policy, repository facts.

This module owns the deterministic policy the harness applies to a parsed plan
and the policy text the planner is asked to honour.  It never calls a model.

Granularity is no longer a hard limit: one step is one testable, coherent unit
with a default mutable scope of a few paths, and a genuinely atomic
transformation is allowed to exceed that default as long as its CONTEXT
explains the atomicity.  A transformation the limit cannot express becomes a
milestone, never a BLOCKED plan.
"""

from __future__ import annotations

from typing import Sequence

from ..models import (
    ExecutionMode,
    ExecutionModePolicy,
    PlanDecision,
    PlanningConfig,
    TaskPlanV2,
)
from ..plan_repository_validation import (
    PathPreconditionViolation,
    RepositoryPreconditions,
    normalize_plan_contracts,
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
    """Render the AGGRESSIVE granularity policy from the configured targets.

    The numbers come from :class:`PlanningConfig`.  They are the default
    mutable scope of one step, deliberately a target and not a limit: the
    harness never rejects a step for scope size, because a genuinely atomic
    transformation is allowed to exceed it when CONTEXT explains why.
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

One step is one testable, coherent unit: one main reasoning responsibility, one
observable result and one targeted verification.

Default mutable scope: 1 to {single} distinct paths for a SINGLE plan, and 1 to
{staged} distinct paths per step for STAGED. The default applies to the union
of WRITE_SET, CREATE_SET and DELETE_SET, not to each section independently.

A step may exceed that default only when a genuinely atomic transformation
requires it, and CONTEXT must then explain that atomicity. Never return BLOCKED
for scope size: split by transformation, layer or dependency boundary, or plan
only the next milestone.
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


def normalize_plan_repository(
    preconditions: RepositoryPreconditions | None, plan: TaskPlanV2,
) -> TaskPlanV2:
    """The effective plan after every deterministic contract normalization.

    Without a start tree there is nothing to normalize against, and the plan
    is returned unchanged: the execution boundary normalizes again against the
    real tree.
    """

    if preconditions is None:
        return plan
    return normalize_plan_contracts(preconditions.repo, preconditions.start_tree_sha, plan)


def render_plan_precondition_correction(
    violations: Sequence[PathPreconditionViolation],
    previous_raw: str,
) -> str:
    return render_precondition_correction(violations, previous_raw=previous_raw)


__all__ = [
    "insert_before_protocol",
    "normalize_plan_repository",
    "render_decomposition_policy_text",
    "render_plan_precondition_correction",
    "render_require_staged_policy_text",
    "validate_execution_mode_policy",
]
