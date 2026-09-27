"""META PLAN v2 wire protocol: grammar, parsing and deterministic rendering.

Pure protocol authority: this module parses planner answers into v2 model
values and renders the step contracts and planner-visible catalogues.  It
performs no I/O, calls no model and never touches Git.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from ..models import (
    ADD_DEFAULT_REQUIRED_CHECK,
    BlockerKind,
    CheckConfig,
    ContractNormalization,
    DROP_UNKNOWN_REQUIRED_CHECK,
    ExecutionClass,
    ExecutionMode,
    ImplementationStep,
    NORMALIZE_STEP_COUNT,
    PlanDecision,
    PlanningConfig,
    TaskPlanV2,
)
from ..plan_repository_validation import MAX_BLOCKERS_CHARS
from ..step_ids import LAST_STEP_ID, MAX_STEPS, STEP_ID_RE, step_ids
from .grammar import (
    PlanParseError,
    V2PlanParseError,
    change_sets,
    lines,
    nonempty,
    parse_labeled_body,
    read_set,
    read_set_paths,
    validate_step_text_limits,
)

MAX_STEP_CONTRACT_CHARS = 9_000
# Canonical layout of the approved step contracts, written at planning time
# and executed byte-for-byte: ``steps/<STEP>/contract.md``.
STEP_CONTRACT_NAME = "contract.md"

_HEADER = "META PLAN v2"
_END = "END META PLAN"
_STEP_BEGIN = re.compile(r"^BEGIN STEP (.+)$")
_STEP_END = re.compile(r"^END STEP (.+)$")
_STEP_ID = STEP_ID_RE
STEP_ID_RANGE = f"S01 through {LAST_STEP_ID}"


def _extract_step_blocks(
    lines: list[str], start: int, end: int
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split the envelope from the ``BEGIN STEP`` blocks, in order."""

    blocks: list[tuple[str, list[str]]] = []
    envelope: list[str] = []
    index = start
    while index < end:
        stripped = lines[index].strip()
        begin = _STEP_BEGIN.fullmatch(stripped)
        if _STEP_END.fullmatch(stripped) is not None:
            raise V2PlanParseError("stray END STEP block")
        if begin is None:
            envelope.append(lines[index])
            index += 1
            continue
        step_id = begin.group(1)
        if _STEP_ID.fullmatch(step_id) is None:
            raise V2PlanParseError(f"step ID must be exactly {STEP_ID_RANGE}")
        close: int | None = None
        for candidate in range(index + 1, end):
            candidate_line = lines[candidate].strip()
            if _STEP_BEGIN.fullmatch(candidate_line) is not None:
                raise V2PlanParseError("nested BEGIN STEP block")
            closing = _STEP_END.fullmatch(candidate_line)
            if closing is not None:
                if closing.group(1) != step_id:
                    raise V2PlanParseError("BEGIN STEP and END STEP IDs do not match")
                close = candidate
                break
        if close is None:
            raise V2PlanParseError(f"missing END STEP {step_id}")
        blocks.append((step_id, lines[index + 1 : close]))
        index = close + 1
    return blocks, envelope

_ENVELOPE_INLINE = frozenset({
    "STATUS", "TITLE", "MILESTONE_ID", "MILESTONE_TITLE",
    "EXECUTION_MODE", "STEP_COUNT", "BLOCKER_KIND",
})
_ENVELOPE_SECTIONS = frozenset({
    "OBJECTIVE", "CONSTRAINTS", "MILESTONE_GOAL", "PROJECT_REMAINDER",
    "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS", "REQUIRED_CHECKS",
})
_STEP_INLINE = frozenset({"TITLE", "EXECUTION_CLASS", "DEPENDS_ON"})
# The rich contract of one step.  EXAMPLES is the single optional section.
_STEP_SECTIONS = frozenset({
    "CONTEXT", "READ_SET", "WRITE_SET", "CREATE_SET", "DELETE_SET",
    "INSTRUCTIONS", "INTERFACES", "EXAMPLES", "TESTS", "PITFALLS",
    "DONE_WHEN", "VERIFY",
})
MILESTONE_ID_RE = re.compile(r"^M\d{2,3}$")


def _parse_step(
    step_id: str,
    body: Sequence[str],
    prior_ids: frozenset[str],
    *,
    max_read_paths_per_step: int,
) -> ImplementationStep:
    values, _ = parse_labeled_body(
        body, inline_names=_STEP_INLINE, section_names=_STEP_SECTIONS, where=f"step {step_id}"
    )
    for name in (
        "TITLE", "EXECUTION_CLASS", "DEPENDS_ON", "CONTEXT", "READ_SET",
        "WRITE_SET", "INSTRUCTIONS", "TESTS", "PITFALLS", "DONE_WHEN", "VERIFY",
    ):
        if not values.get(name, "").strip():
            raise V2PlanParseError(f"step {step_id} is missing {name}")
    # ``NONE`` is a real answer for the sections that may not apply.
    for name in ("CREATE_SET", "DELETE_SET", "INTERFACES"):
        if not values.get(name, "").strip():
            raise V2PlanParseError(f"step {step_id} is missing {name}")
    if "EXAMPLES" in values and not values["EXAMPLES"].strip():
        raise V2PlanParseError(f"step {step_id} has an empty EXAMPLES; use NONE")
    validate_step_text_limits(step_id, values)
    title = nonempty(values["TITLE"], f"step {step_id} TITLE")
    execution_class = values["EXECUTION_CLASS"]
    if execution_class not in {item.value for item in ExecutionClass}:
        raise V2PlanParseError(f"unknown execution class in {step_id}")
    dependency = values["DEPENDS_ON"]
    if dependency != "NONE":
        if _STEP_ID.fullmatch(dependency) is None or dependency not in prior_ids:
            raise V2PlanParseError(f"invalid or future dependency in {step_id}")
    context = nonempty(values["CONTEXT"], f"step {step_id} CONTEXT")
    read_entries = read_set(values["READ_SET"], max_paths=max_read_paths_per_step)
    write_set, create_set, delete_set = change_sets(values)
    return ImplementationStep(
        id=step_id,
        title=title,
        execution_class=ExecutionClass(execution_class),
        depends_on=None if dependency == "NONE" else dependency,
        context=context,
        read_set=read_entries,
        write_set=write_set,
        create_set=create_set,
        delete_set=delete_set,
        instructions=nonempty(values["INSTRUCTIONS"], f"step {step_id} INSTRUCTIONS"),
        interfaces=values["INTERFACES"].strip(),
        examples=values.get("EXAMPLES", "").strip() or "NONE",
        tests=nonempty(values["TESTS"], f"step {step_id} TESTS"),
        pitfalls=nonempty(values["PITFALLS"], f"step {step_id} PITFALLS"),
        done_when=nonempty(values["DONE_WHEN"], f"step {step_id} DONE_WHEN"),
        verify=nonempty(values["VERIFY"], f"step {step_id} VERIFY"),
    )


def _render_step_contract_unchecked(plan: TaskPlanV2, step: ImplementationStep) -> str:
    def lines(items: tuple[str, ...]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "NONE"

    return "\n\n".join(
        (
            "META IMPLEMENTATION STEP v2",
            "RUN TITLE\n" + plan.title,
            f"STEP\n{step.id} / {len(plan.steps):02d}",
            "TITLE\n" + step.title,
            "EXECUTION CLASS\n" + step.execution_class.value,
            "DEPENDS_ON\n" + (step.depends_on or "NONE"),
            "CONTEXT\n" + step.context,
            "READ SET\n" + lines(step.read_set),
            "WRITE SET\n" + lines(step.write_set),
            "CREATE SET\n" + lines(step.create_set),
            "DELETE SET\n" + lines(step.delete_set),
            "INSTRUCTIONS\n" + step.instructions,
            "INTERFACES\n" + step.interfaces,
            "EXAMPLES\n" + step.examples,
            "TESTS\n" + step.tests,
            "PITFALLS\n" + step.pitfalls,
            "DONE_WHEN\n" + step.done_when,
            "VERIFY\n" + step.verify,
            "END META IMPLEMENTATION STEP",
        )
    ) + "\n"


def render_step_contract(plan: TaskPlanV2, step: ImplementationStep) -> str:
    if not isinstance(plan, TaskPlanV2) or not isinstance(step, ImplementationStep):
        raise TypeError("plan and step must be v2 model values")
    if plan.decision is not PlanDecision.READY or step not in plan.steps:
        raise V2PlanParseError("step contract requires a step from a READY plan")
    rendered = _render_step_contract_unchecked(plan, step)
    if len(rendered) > plan.max_step_contract_chars:
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    return rendered


def validate_step_contract_bounds(plan: TaskPlanV2) -> None:
    contracts = [_render_step_contract_unchecked(plan, step) for step in plan.steps]
    if any(len(contract) > plan.max_step_contract_chars for contract in contracts):
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")


def _parse_required_checks(
    value: str,
    *,
    check_catalog: Sequence[CheckConfig],
    default_check_ids: Sequence[str],
    inherited_check_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[ContractNormalization, ...]]:
    """The effective required checks, and every mechanical normalization made.

    The trusted catalogue is the harness authority: a planner can never create
    a check by naming it, and every configured default is present whether the
    answer listed it or not.  Both facts are recorded instead of being refused.
    """

    catalog_ids = tuple(check.id for check in check_catalog)
    if len(set(catalog_ids)) != len(catalog_ids):
        raise V2PlanParseError("trusted check catalogue contains duplicate IDs")
    if not catalog_ids:
        return (), ()
    unknown_required = [
        check_id for check_id in (*default_check_ids, *inherited_check_ids)
        if check_id not in catalog_ids
    ]
    if unknown_required:
        raise V2PlanParseError("unknown configured required check ID: " + unknown_required[0])
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise V2PlanParseError("READY plan requires REQUIRED_CHECKS")
    records: list[ContractNormalization] = []
    selected: list[str] = []
    for line in lines:
        match = re.fullmatch(r"-\s+([A-Za-z0-9][A-Za-z0-9_.-]{0,63})", line)
        if match is None:
            raise V2PlanParseError("REQUIRED_CHECKS must contain only '- <catalogue id>' lines")
        check_id = match.group(1)
        if check_id not in catalog_ids:
            records.append(ContractNormalization(
                code=DROP_UNKNOWN_REQUIRED_CHECK, detail=f"check={check_id}",
            ))
            continue
        if check_id not in selected:
            selected.append(check_id)
    required = tuple(dict.fromkeys((*default_check_ids, *inherited_check_ids)))
    for check_id in required:
        if check_id not in selected:
            selected.append(check_id)
            records.append(ContractNormalization(
                code=ADD_DEFAULT_REQUIRED_CHECK, detail=f"check={check_id}",
            ))
    # The model chooses membership; the harness owns deterministic ordering.
    return tuple(check_id for check_id in catalog_ids if check_id in selected), tuple(records)


def parse_task_plan_v2(
    raw: str,
    *,
    planning: PlanningConfig | None = None,
    check_catalog: Sequence[CheckConfig] = (),
    default_check_ids: Sequence[str] = (),
    inherited_check_ids: Sequence[str] = (),
) -> TaskPlanV2:
    """Parse the exact v2 wire protocol and preserve ``raw`` unchanged."""

    if not isinstance(raw, str):
        raise TypeError("raw planner response must be a string")
    if not raw.strip():
        raise V2PlanParseError("planner response is empty")
    planning = planning or PlanningConfig()
    if not isinstance(planning, PlanningConfig):
        raise TypeError("planning must be a PlanningConfig")
    raw_lines = lines(raw)
    first = next((index for index, line in enumerate(raw_lines) if line.strip()), None)
    if first is None or raw_lines[first].strip() != _HEADER:
        raise V2PlanParseError("missing META PLAN v2 header")
    ends = [index for index, line in enumerate(raw_lines) if line.strip() == _END]
    if len(ends) != 1 or ends[0] <= first:
        raise V2PlanParseError("missing or duplicate END META PLAN")
    end = ends[0]
    if any(line.strip() for line in raw_lines[:first]) or any(
        line.strip() for line in raw_lines[end + 1:]
    ):
        raise V2PlanParseError("content outside META PLAN v2 envelope")

    blocks, envelope_lines = _extract_step_blocks(raw_lines, first + 1, end)
    inline, sections = parse_labeled_body(
        envelope_lines,
        inline_names=_ENVELOPE_INLINE,
        section_names=_ENVELOPE_SECTIONS,
        where="plan",
    )
    status = inline.get("STATUS")
    if status not in {PlanDecision.READY.value, PlanDecision.BLOCKED.value}:
        raise V2PlanParseError("STATUS must be exactly READY or BLOCKED")
    decision = PlanDecision(status)
    title = nonempty(inline.get("TITLE", ""), "TITLE")
    objective = nonempty(sections.get("OBJECTIVE", ""), "OBJECTIVE")
    blockers = sections.get("BLOCKERS", "").strip()
    if decision is PlanDecision.BLOCKED:
        if blocks or any(
            name in inline
            for name in ("EXECUTION_MODE", "STEP_COUNT", "MILESTONE_ID", "MILESTONE_TITLE")
        ):
            raise V2PlanParseError("BLOCKED plan must not contain execution metadata or steps")
        if not blockers or blockers.casefold() in {"none", "n/a", "na", "-", "—", "nil"}:
            raise V2PlanParseError("BLOCKED plan requires real BLOCKERS")
        if len(blockers) > MAX_BLOCKERS_CHARS:
            raise V2PlanParseError("BLOCKERS exceeds its protocol limit")
        try:
            blocker_kind = BlockerKind(inline.get("BLOCKER_KIND", ""))
        except ValueError as exc:
            raise V2PlanParseError(
                "BLOCKED plan requires a valid BLOCKER_KIND"
            ) from exc
        allowed = {"OBJECTIVE", "BLOCKERS"}
        if set(sections) - allowed or "CONSTRAINTS" in sections or "ACCEPTANCE" in sections or "TESTS" in sections or "RISKS" in sections:
            raise V2PlanParseError("BLOCKED plan contains READY-only sections")
        return TaskPlanV2(
            decision=decision, title=title, objective=objective, constraints="",
            execution_mode=None, steps=(), acceptance="", tests="", risks="",
            blockers=blockers, raw=raw,
            max_step_contract_chars=planning.max_step_contract_chars,
            blocker_kind=blocker_kind,
        )
    if "BLOCKER_KIND" in inline:
        raise V2PlanParseError("BLOCKER_KIND is only valid for BLOCKED plans")
    milestone_id = nonempty(inline.get("MILESTONE_ID", ""), "MILESTONE_ID")
    if MILESTONE_ID_RE.fullmatch(milestone_id) is None:
        raise V2PlanParseError("MILESTONE_ID must look like M01")
    milestone_title = nonempty(inline.get("MILESTONE_TITLE", ""), "MILESTONE_TITLE")
    milestone_goal = nonempty(sections.get("MILESTONE_GOAL", ""), "MILESTONE_GOAL")
    # ``NONE`` is the explicit, valid statement that nothing is left over.
    project_remainder = sections.get("PROJECT_REMAINDER", "").strip()
    if not project_remainder:
        raise V2PlanParseError("MILESTONE_GOAL and PROJECT_REMAINDER are required")

    mode = inline.get("EXECUTION_MODE")
    if mode not in {item.value for item in ExecutionMode}:
        raise V2PlanParseError("EXECUTION_MODE must be exactly SINGLE or STAGED")
    # The real step blocks are the count: the wire field is redundant metadata
    # whose mismatch is recorded, never a reason to reject a coherent plan.
    count_text = inline.get("STEP_COUNT", "").strip()
    step_count = len(blocks)
    if step_count < 1 or step_count > MAX_STEPS:
        raise V2PlanParseError("STEP_COUNT exceeds protocol bounds")
    if step_count > planning.max_steps_per_plan:
        raise V2PlanParseError(
            f"plan has {step_count} steps; planning.max_steps_per_plan is "
            f"{planning.max_steps_per_plan}"
        )
    if mode == ExecutionMode.SINGLE.value and step_count != 1:
        raise V2PlanParseError("SINGLE requires exactly one step")
    if mode == ExecutionMode.STAGED.value and not 2 <= step_count <= MAX_STEPS:
        raise V2PlanParseError(f"STAGED requires between 2 and {MAX_STEPS} steps")
    if [step_id for step_id, _ in blocks] != list(step_ids(step_count)):
        raise V2PlanParseError(f"step IDs must be contiguous {STEP_ID_RANGE}")
    normalizations: list[ContractNormalization] = []
    if count_text != str(step_count):
        normalizations.append(ContractNormalization(
            code=NORMALIZE_STEP_COUNT,
            detail=f"declared={count_text[:32] or 'missing'} real={step_count}",
        ))
    steps: list[ImplementationStep] = []
    for step_id, body in blocks:
        steps.append(
            _parse_step(
                step_id,
                body,
                frozenset(step.id for step in steps),
                max_read_paths_per_step=planning.max_read_paths_per_step,
            )
        )
    for name in ("CONSTRAINTS", "MILESTONE_GOAL", "PROJECT_REMAINDER",
                 "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"):
        if name not in sections or not sections[name].strip():
            raise V2PlanParseError(f"READY plan is missing {name}")
    if blockers and blockers.casefold() not in {"none", "n/a", "na", "-", "—", "nil"}:
        raise V2PlanParseError("READY plan cannot contain real BLOCKERS")
    required_checks, check_normalizations = _parse_required_checks(
        sections.get("REQUIRED_CHECKS", ""),
        check_catalog=check_catalog,
        default_check_ids=default_check_ids,
        inherited_check_ids=inherited_check_ids,
    )
    normalizations.extend(check_normalizations)
    plan = TaskPlanV2(
        decision=decision,
        title=title,
        objective=objective,
        constraints=sections["CONSTRAINTS"].strip(),
        execution_mode=ExecutionMode(mode),
        steps=tuple(steps),
        acceptance=sections["ACCEPTANCE"].strip(),
        tests=sections["TESTS"].strip(),
        risks=sections["RISKS"].strip(),
        blockers=blockers or "NONE",
        raw=raw,
        required_checks=required_checks,
        normalizations=tuple(normalizations),
        max_step_contract_chars=planning.max_step_contract_chars,
        milestone_id=milestone_id,
        milestone_title=milestone_title,
        milestone_goal=milestone_goal,
        project_remainder=project_remainder,
    )
    validate_step_contract_bounds(plan)
    return plan


def render_plan_summary_v2(plan: TaskPlanV2) -> str:
    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    mode = plan.execution_mode.value if plan.execution_mode is not None else "NONE"
    summaries = []
    for step in plan.steps:
        summaries.append(f"{step.id} — {step.title} [{step.execution_class.value}]\n{step.context}")
    milestone = plan.milestone_id or "NONE"
    return "\n\n".join(
        (
            "# Implementation contract",
            f"## Title\n{plan.title}",
            f"## Milestone\n{milestone} — {plan.milestone_title or 'NONE'}",
            f"## Milestone goal\n{plan.milestone_goal or 'NONE'}",
            f"## Project remainder\n{plan.project_remainder or 'NONE'}",
            f"## Objective\n{plan.objective}",
            f"## Constraints\n{plan.constraints or 'NONE'}",
            f"## Execution mode\n{mode}",
            "## Required checks\n" + ("\n".join(f"- {check_id}" for check_id in plan.required_checks) or "NONE"),
            "## Ordered steps\n" + ("\n\n".join(summaries) if summaries else "NONE"),
            f"## Acceptance\n{plan.acceptance or 'NONE'}",
            f"## Tests\n{plan.tests or 'NONE'}",
            f"## Risks\n{plan.risks or 'NONE'}",
        )
    ) + "\n"


def render_safe_check_catalogue(checks: Sequence[CheckConfig]) -> str:
    """Render only trusted check IDs and human descriptions for the planner."""

    output: list[str] = []
    for check in checks:
        if not isinstance(check, CheckConfig):
            raise TypeError("checks must contain CheckConfig values")
        output.append(
            "\n".join((
                "CHECK",
                f"ID: {check.id}",
                f"DESCRIPTION: {check.description or 'No description configured.'}",
                "END CHECK",
            ))
        )
    return "\n\n".join(output) or "NONE"


__all__ = [
    "MAX_STEP_CONTRACT_CHARS",
    "MILESTONE_ID_RE",
    "STEP_CONTRACT_NAME",
    "STEP_ID_RANGE",
    "PlanParseError",
    "V2PlanParseError",
    "parse_task_plan_v2",
    "read_set_paths",
    "render_plan_summary_v2",
    "render_safe_check_catalogue",
    "render_step_contract",
    "validate_step_contract_bounds",
]
