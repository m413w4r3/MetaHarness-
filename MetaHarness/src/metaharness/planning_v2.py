"""Strict META PLAN v2 parsing and bounded implementation contracts.

This module is deliberately parallel to :mod:`metaharness.planning`.  The v1
parser and its historical artifacts remain the compatibility path; v2 is a
new protocol that can be selected by a later execution milestone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence

from .gitops import RepositoryReference, render_repository_reference
from .llm.chat import LLMConversationHandle, TextLLMResult, conversation_handle
from .models import (
    ExecutionMode,
    ExecutionModePolicy,
    ImplementationStep,
    ModelProfile,
    CheckConfig,
    PlanningConfig,
    TaskPlanV2,
)
from .planning import PlanDecision, PlanParseError
from .result import atomic_write_text
from .usage import PLANNER_USAGE_ARTIFACT, completion_usage, write_usage_artifact


MAX_STEPS = 8
MAX_STEP_CONTRACT_CHARS = 8_000
MAX_TOTAL_STEP_CONTRACT_CHARS = 32_000
MAX_READ_SET = 8
MAX_WRITE_SET = 6
MAX_CREATE_SET = 6
MAX_DELETE_SET = 6
# Canonical layout of the approved step contracts, written at planning time
# and executed byte-for-byte: ``steps/<STEP>/contract.md``.
STEP_CONTRACT_NAME = "contract.md"

_HEADER = "META PLAN v2"
_END = "END META PLAN"
_STEP_BEGIN = re.compile(r"^BEGIN STEP (.+)$")
_STEP_END = re.compile(r"^END STEP (.+)$")
_STEP_ID = re.compile(r"S0[1-8]")
_INLINE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$")
_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_TEXT_SECTIONS = frozenset(
    {"OBJECTIVE", "CONSTRAINTS", "READ_SET", "WRITE_SET", "CREATE_SET", "DELETE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"}
)
_ENVELOPE_INLINE = frozenset({"STATUS", "TITLE", "EXECUTION_MODE", "STEP_COUNT", "REVIEWER_PROFILE"})
_ENVELOPE_SECTIONS = frozenset({"OBJECTIVE", "CONSTRAINTS", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS", "REQUIRED_CHECKS"})
_STEP_INLINE = frozenset({"TITLE", "IMPLEMENTER_PROFILE", "DEPENDS_ON"})
_STEP_SECTIONS = frozenset({"OBJECTIVE", "READ_SET", "WRITE_SET", "CREATE_SET", "DELETE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"})


class V2PlanParseError(PlanParseError):
    """A META PLAN v2 response is not safe to execute."""


def _lines(raw: str) -> list[str]:
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _nonempty(value: str, name: str) -> str:
    value = value.strip()
    if not value or value.casefold() in {"none", "n/a", "na", "-", "—", "nil", "tbd"}:
        raise V2PlanParseError(f"{name} is missing or a placeholder")
    return value


def _section_name(line: str, allowed: frozenset[str]) -> str | None:
    candidate = line.strip()
    candidate = re.sub(r"^#{1,6}\s+", "", candidate)
    if candidate.endswith(":"):
        candidate = candidate[:-1].rstrip()
    for name in allowed:
        if candidate.casefold() == name.casefold():
            return name
    return None


def _parse_labeled_body(
    body: Sequence[str],
    *,
    inline_names: frozenset[str],
    section_names: frozenset[str],
    where: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse one strict labeled body, tolerating Markdown headings for prose."""

    inline: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in body:
        stripped = raw_line.strip()
        if not stripped:
            if current is not None:
                sections[current].append("")
            continue

        match = _INLINE.fullmatch(stripped)
        if match and match.group(1) in inline_names:
            name, value = match.groups()
            if name in inline:
                raise V2PlanParseError(f"duplicate {where} field {name}")
            inline[name] = value.strip()
            current = None
            continue

        if match and match.group(1) in section_names:
            name, value = match.groups()
            if name in sections:
                raise V2PlanParseError(f"duplicate {where} section {name}")
            sections[name] = [value] if value.strip() else []
            current = name if not value.strip() else None
            continue

        section = _section_name(stripped, section_names)
        if section is not None:
            if section in sections:
                raise V2PlanParseError(f"duplicate {where} section {section}")
            sections[section] = []
            current = section
            continue

        if current is None:
            raise V2PlanParseError(f"unexpected content in {where}: {stripped[:80]}")
        sections[current].append(raw_line)

    values = {name: value.strip() for name, value in inline.items()}
    values.update({name: "\n".join(lines).strip() for name, lines in sections.items()})
    return values, {name: value for name, value in values.items() if name in section_names}


def _extract_step_blocks(lines: list[str], start: int, end: int) -> tuple[list[tuple[str, list[str]]], list[str]]:
    blocks: list[tuple[str, list[str]]] = []
    envelope: list[str] = []
    index = start
    while index < end:
        stripped = lines[index].strip()
        begin = _STEP_BEGIN.fullmatch(stripped)
        step_end = _STEP_END.fullmatch(stripped)
        if step_end is not None:
            raise V2PlanParseError("stray END STEP block")
        if begin is None:
            envelope.append(lines[index])
            index += 1
            continue
        step_id = begin.group(1)
        if _STEP_ID.fullmatch(step_id) is None:
            raise V2PlanParseError("step ID must be exactly S01 through S08")
        close: int | None = None
        for candidate in range(index + 1, end):
            candidate_line = lines[candidate].strip()
            nested = _STEP_BEGIN.fullmatch(candidate_line)
            if nested is not None:
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


def _repo_path(value: str, *, kind: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise V2PlanParseError(f"unsafe {kind} path")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise V2PlanParseError(f"unsafe {kind} path")
    path = PurePosixPath(value)
    if not value or path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise V2PlanParseError(f"unsafe {kind} path")
    if any(char in value for char in "*?["):
        raise V2PlanParseError(f"wildcard {kind} path")
    return value


def _read_set(value: str) -> tuple[str, ...]:
    if not value.strip():
        raise V2PlanParseError("READ_SET is missing")
    result: list[str] = []
    paths: set[str] = set()
    for line in value.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- ") or " :: " not in line:
            raise V2PlanParseError("each READ_SET line must be '- path :: anchor'")
        path, anchor = line[2:].split(" :: ", 1)
        path = _repo_path(path.strip(), kind="READ_SET")
        anchor = anchor.strip()
        if not anchor:
            raise V2PlanParseError("READ_SET anchor must not be empty")
        if path in paths:
            raise V2PlanParseError("duplicate READ_SET path")
        paths.add(path)
        result.append(path + " :: " + anchor)
    if not result:
        raise V2PlanParseError("READ_SET is missing")
    if len(result) > MAX_READ_SET:
        raise V2PlanParseError(f"READ_SET may contain at most {MAX_READ_SET} paths")
    return tuple(result)


def read_set_paths(read_set: Sequence[str]) -> tuple[str, ...]:
    """The repo-relative paths of ``'path :: anchor'`` READ_SET entries."""

    return tuple(item.split(" :: ", 1)[0] for item in read_set)


def _path_set(value: str, *, name: str, limit: int) -> tuple[str, ...]:
    """Parse a ``- path`` list; exactly ``NONE`` is the explicit empty set."""

    if not value.strip():
        raise V2PlanParseError(f"{name} is missing")
    if value.strip() == "NONE":
        return ()
    result: list[str] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- ") or " :: " in line:
            raise V2PlanParseError(f"each {name} line must be '- path'")
        path = _repo_path(line[2:].strip(), kind=name)
        if path in result:
            raise V2PlanParseError(f"duplicate {name} path")
        result.append(path)
    if not result:
        raise V2PlanParseError(f"{name} is missing")
    if len(result) > limit:
        raise V2PlanParseError(f"{name} may contain at most {limit} paths")
    return tuple(result)


def _change_sets(
    values: dict[str, str], read_set: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate WRITE/CREATE/DELETE sets against READ_SET and each other."""

    reads = set(read_set_paths(read_set))
    write_set = _path_set(values["WRITE_SET"], name="WRITE_SET", limit=MAX_WRITE_SET)
    # Plans emitted before CREATE_SET/DELETE_SET existed omit both sections.
    create_set = (
        _path_set(values["CREATE_SET"], name="CREATE_SET", limit=MAX_CREATE_SET)
        if "CREATE_SET" in values
        else ()
    )
    delete_set = (
        _path_set(values["DELETE_SET"], name="DELETE_SET", limit=MAX_DELETE_SET)
        if "DELETE_SET" in values
        else ()
    )
    if any(path not in reads for path in write_set):
        raise V2PlanParseError("every WRITE_SET path must also appear in READ_SET")
    if any(path not in reads for path in delete_set):
        raise V2PlanParseError("every DELETE_SET path must also appear in READ_SET")
    if any(path in reads for path in create_set):
        # A READ_SET path must exist and a CREATE_SET path must not.
        raise V2PlanParseError("a CREATE_SET path cannot appear in READ_SET")
    if (
        set(write_set) & set(create_set)
        or set(write_set) & set(delete_set)
        or set(create_set) & set(delete_set)
    ):
        raise V2PlanParseError(
            "a path may appear in only one of WRITE_SET, CREATE_SET and DELETE_SET"
        )
    if not (write_set or create_set or delete_set):
        raise V2PlanParseError("a step must write, create or delete at least one path")
    return write_set, create_set, delete_set


def _parse_step(step_id: str, body: Sequence[str], implementer_ids: frozenset[str], prior_ids: frozenset[str]) -> ImplementationStep:
    values, _ = _parse_labeled_body(
        body, inline_names=_STEP_INLINE, section_names=_STEP_SECTIONS, where=f"step {step_id}"
    )
    for name in ("TITLE", "IMPLEMENTER_PROFILE", "DEPENDS_ON", "OBJECTIVE", "READ_SET", "WRITE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"):
        if not values.get(name, "").strip():
            raise V2PlanParseError(f"step {step_id} is missing {name}")
    for name in ("CREATE_SET", "DELETE_SET"):
        if name in values and not values[name].strip():
            raise V2PlanParseError(f"step {step_id} has an empty {name}; use NONE")
    title = _nonempty(values["TITLE"], f"step {step_id} TITLE")
    profile = values["IMPLEMENTER_PROFILE"]
    if not _PROFILE.fullmatch(profile) or profile not in implementer_ids:
        raise V2PlanParseError(f"unknown implementer profile in {step_id}")
    dependency = values["DEPENDS_ON"]
    if dependency != "NONE":
        if _STEP_ID.fullmatch(dependency) is None or dependency not in prior_ids:
            raise V2PlanParseError(f"invalid or future dependency in {step_id}")
    objective = _nonempty(values["OBJECTIVE"], f"step {step_id} OBJECTIVE")
    read_set = _read_set(values["READ_SET"])
    write_set, create_set, delete_set = _change_sets(values, read_set)
    instructions = _nonempty(values["INSTRUCTIONS"], f"step {step_id} INSTRUCTIONS")
    verify = _nonempty(values["VERIFY"], f"step {step_id} VERIFY")
    forbidden = _nonempty(values["FORBIDDEN"], f"step {step_id} FORBIDDEN")
    return ImplementationStep(
        step_id, title, profile, None if dependency == "NONE" else dependency,
        objective, read_set, write_set, instructions, verify, forbidden,
        create_set=create_set, delete_set=delete_set,
    )


def _render_step_contract_unchecked(plan: TaskPlanV2, step: ImplementationStep) -> str:
    def lines(items: tuple[str, ...]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "NONE"

    return "\n\n".join(
        (
            "META IMPLEMENTATION STEP v1",
            "RUN TITLE\n" + plan.title,
            f"STEP\n{step.id} / {len(plan.steps):02d}",
            "TITLE\n" + step.title,
            "OBJECTIVE\n" + step.objective,
            "READ SET\n" + lines(step.read_set),
            "WRITE SET\n" + lines(step.write_set),
            "CREATE SET\n" + lines(step.create_set),
            "DELETE SET\n" + lines(step.delete_set),
            "INSTRUCTIONS\n" + step.instructions,
            "VERIFY\n" + step.verify,
            "FORBIDDEN\n" + step.forbidden,
            "END META IMPLEMENTATION STEP",
        )
    ) + "\n"


def render_step_contract(plan: TaskPlanV2, step: ImplementationStep) -> str:
    if not isinstance(plan, TaskPlanV2) or not isinstance(step, ImplementationStep):
        raise TypeError("plan and step must be v2 model values")
    if plan.decision is not PlanDecision.READY or step not in plan.steps:
        raise V2PlanParseError("step contract requires a step from a READY plan")
    rendered = _render_step_contract_unchecked(plan, step)
    if len(rendered) > MAX_STEP_CONTRACT_CHARS:
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    return rendered


def _validate_bounds(plan: TaskPlanV2) -> None:
    contracts = [_render_step_contract_unchecked(plan, step) for step in plan.steps]
    if any(len(contract) > MAX_STEP_CONTRACT_CHARS for contract in contracts):
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    if sum(map(len, contracts)) > MAX_TOTAL_STEP_CONTRACT_CHARS:
        raise V2PlanParseError("step contracts exceed MAX_TOTAL_STEP_CONTRACT_CHARS")


def _parse_required_checks(
    value: str,
    *,
    check_catalog: Sequence[CheckConfig],
    default_check_ids: Sequence[str],
    inherited_check_ids: Sequence[str],
) -> tuple[str, ...]:
    catalog_ids = tuple(check.id for check in check_catalog)
    if len(set(catalog_ids)) != len(catalog_ids):
        raise V2PlanParseError("trusted check catalogue contains duplicate IDs")
    if not catalog_ids:
        return ()
    unknown_required = [
        check_id for check_id in (*default_check_ids, *inherited_check_ids)
        if check_id not in catalog_ids
    ]
    if unknown_required:
        raise V2PlanParseError("unknown configured required check ID: " + unknown_required[0])
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise V2PlanParseError("READY plan requires REQUIRED_CHECKS")
    selected: list[str] = []
    for line in lines:
        match = re.fullmatch(r"-\s+([A-Za-z0-9][A-Za-z0-9_.-]{0,63})", line)
        if match is None:
            raise V2PlanParseError("REQUIRED_CHECKS must contain only '- <catalogue id>' lines")
        check_id = match.group(1)
        if check_id not in catalog_ids:
            raise V2PlanParseError(f"unknown required check ID: {check_id}")
        if check_id not in selected:
            selected.append(check_id)
    required = tuple(dict.fromkeys((*default_check_ids, *inherited_check_ids)))
    missing = [check_id for check_id in required if check_id not in selected]
    if missing:
        raise V2PlanParseError("REQUIRED_CHECKS is missing defaults: " + ", ".join(missing))
    # The model chooses membership; the harness owns deterministic ordering.
    return tuple(check_id for check_id in catalog_ids if check_id in selected)


def parse_task_plan_v2(
    raw: str,
    *,
    implementer_ids: frozenset[str],
    reviewer_ids: frozenset[str],
    check_catalog: Sequence[CheckConfig] = (),
    default_check_ids: Sequence[str] = (),
    inherited_check_ids: Sequence[str] = (),
) -> TaskPlanV2:
    """Parse the exact v2 wire protocol and preserve ``raw`` unchanged."""

    if not isinstance(raw, str):
        raise TypeError("raw planner response must be a string")
    if not raw.strip():
        raise V2PlanParseError("planner response is empty")
    if not isinstance(implementer_ids, frozenset) or not isinstance(reviewer_ids, frozenset):
        raise TypeError("profile IDs must be frozensets")
    lines = _lines(raw)
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None or lines[first].strip() != _HEADER:
        raise V2PlanParseError("missing META PLAN v2 header")
    ends = [index for index, line in enumerate(lines) if line.strip() == _END]
    if len(ends) != 1 or ends[0] <= first:
        raise V2PlanParseError("missing or duplicate END META PLAN")
    end = ends[0]
    if any(line.strip() for line in lines[:first]) or any(line.strip() for line in lines[end + 1:]):
        raise V2PlanParseError("content outside META PLAN v2 envelope")

    blocks, envelope_lines = _extract_step_blocks(lines, first + 1, end)
    inline, sections = _parse_labeled_body(
        envelope_lines,
        inline_names=_ENVELOPE_INLINE,
        section_names=_ENVELOPE_SECTIONS,
        where="plan",
    )
    status = inline.get("STATUS")
    if status not in {PlanDecision.READY.value, PlanDecision.BLOCKED.value}:
        raise V2PlanParseError("STATUS must be exactly READY or BLOCKED")
    decision = PlanDecision(status)
    title = _nonempty(inline.get("TITLE", ""), "TITLE")
    objective = _nonempty(sections.get("OBJECTIVE", ""), "OBJECTIVE")
    blockers = sections.get("BLOCKERS", "").strip()
    if decision is PlanDecision.BLOCKED:
        if blocks or any(name in inline for name in ("EXECUTION_MODE", "STEP_COUNT", "REVIEWER_PROFILE")):
            raise V2PlanParseError("BLOCKED plan must not contain execution metadata or steps")
        if not blockers or blockers.casefold() in {"none", "n/a", "na", "-", "—", "nil"}:
            raise V2PlanParseError("BLOCKED plan requires real BLOCKERS")
        allowed = {"OBJECTIVE", "BLOCKERS"}
        if set(sections) - allowed or "CONSTRAINTS" in sections or "ACCEPTANCE" in sections or "TESTS" in sections or "RISKS" in sections:
            raise V2PlanParseError("BLOCKED plan contains READY-only sections")
        return TaskPlanV2(decision, title, objective, "", None, None, (), "", "", "", blockers, raw)

    mode = inline.get("EXECUTION_MODE")
    if mode not in {item.value for item in ExecutionMode}:
        raise V2PlanParseError("EXECUTION_MODE must be exactly SINGLE or STAGED")
    count_text = inline.get("STEP_COUNT", "")
    if not re.fullmatch(r"[0-9]+", count_text):
        raise V2PlanParseError("STEP_COUNT must be an integer")
    step_count = int(count_text)
    if step_count < 1 or step_count > MAX_STEPS:
        raise V2PlanParseError("STEP_COUNT exceeds allowed bounds")
    if mode == ExecutionMode.SINGLE.value and step_count != 1:
        raise V2PlanParseError("SINGLE requires exactly one step")
    if mode == ExecutionMode.STAGED.value and not 2 <= step_count <= MAX_STEPS:
        raise V2PlanParseError("STAGED requires between two and eight steps")
    reviewer = inline.get("REVIEWER_PROFILE", "")
    if not _PROFILE.fullmatch(reviewer) or reviewer not in reviewer_ids:
        raise V2PlanParseError("unknown reviewer profile")
    if len(blocks) != step_count:
        raise V2PlanParseError("STEP_COUNT does not match step blocks")
    expected = [f"S{index:02d}" for index in range(1, step_count + 1)]
    if [step_id for step_id, _ in blocks] != expected:
        raise V2PlanParseError("step IDs must be contiguous S01 through S08")
    steps: list[ImplementationStep] = []
    for step_id, body in blocks:
        steps.append(_parse_step(step_id, body, implementer_ids, frozenset(step.id for step in steps)))
    for name in ("CONSTRAINTS", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"):
        if name not in sections or not sections[name].strip():
            raise V2PlanParseError(f"READY plan is missing {name}")
    if blockers and blockers.casefold() not in {"none", "n/a", "na", "-", "—", "nil"}:
        raise V2PlanParseError("READY plan cannot contain real BLOCKERS")
    required_checks = _parse_required_checks(
        sections.get("REQUIRED_CHECKS", ""),
        check_catalog=check_catalog,
        default_check_ids=default_check_ids,
        inherited_check_ids=inherited_check_ids,
    )
    plan = TaskPlanV2(
        decision, title, objective, sections["CONSTRAINTS"].strip(), ExecutionMode(mode), reviewer,
        tuple(steps), sections["ACCEPTANCE"].strip(), sections["TESTS"].strip(), sections["RISKS"].strip(), blockers or "NONE", raw,
        required_checks,
    )
    _validate_bounds(plan)
    return plan


def render_plan_summary_v2(plan: TaskPlanV2) -> str:
    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    mode = plan.execution_mode.value if plan.execution_mode is not None else "NONE"
    summaries = []
    for step in plan.steps:
        summaries.append(f"{step.id} — {step.title} [{step.implementer_profile}]\n{step.objective}")
    return "\n\n".join(
        (
            "# Implementation contract",
            f"## Title\n{plan.title}",
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


def render_repair_plan_summary(plan: TaskPlanV2) -> str:
    """Render only the original-plan facts not repeated by step contracts."""

    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    mode = plan.execution_mode.value if plan.execution_mode is not None else "NONE"
    steps = "\n".join(
        f"- {step.id} | {step.title} | depends_on: {step.depends_on or 'NONE'}"
        for step in plan.steps
    ) or "NONE"
    return "\n".join((
        "TITLE: " + (plan.title or "NONE"),
        "OBJECTIVE:\n" + (plan.objective or "NONE"),
        "CONSTRAINTS:\n" + (plan.constraints or "NONE"),
        "EXECUTION MODE: " + mode,
        "ACCEPTANCE:\n" + (plan.acceptance or "NONE"),
        "TESTS:\n" + (plan.tests or "NONE"),
        "RISKS:\n" + (plan.risks or "NONE"),
        "STEPS (id | title | depends_on):\n" + steps,
    )) + "\n"


def render_safe_profile_catalogue(profiles: Sequence[ModelProfile]) -> str:
    """Render planner-visible profile metadata, excluding endpoint credentials."""

    output: list[str] = []
    for profile in profiles:
        if not isinstance(profile, ModelProfile):
            raise TypeError("profiles must contain ModelProfile values")
        output.append(
            "\n".join(
                (
                    "PROFILE",
                    f"ID: {profile.id}",
                    f"DISPLAY_NAME: {profile.display_name}",
                    f"DRIVER: {profile.driver.value}",
                    f"MODEL_LABEL: {profile.model}",
                    f"EFFORT: {profile.effort or 'NONE'}",
                    f"DESCRIPTION: {profile.description}",
                    "STRENGTHS:",
                    *(f"- {item}" for item in profile.strengths),
                    f"COST_TIER: {profile.cost_tier}",
                    f"LATENCY_TIER: {profile.latency_tier}",
                    f"SELECTION_MODE: {profile.selection_mode.value}",
                    "END PROFILE",
                )
            )
        )
    return "\n\n".join(output)


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


REQUIRE_STAGED_POLICY_TEXT = """This run REQUIRES STAGED execution.

You must return between 2 and 6 coherent implementation steps.

Do not create artificial "implementation then tests" steps when tests belong
to the same local behavior.

Instead decompose by independently understandable behavioral transformation,
layer, subsystem, or dependency boundary.

Each worker must receive a mechanically executable contract and must not need
architectural discovery.
"""
_PROTOCOL_ANCHOR = "The answer must use exactly this protocol."


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
independently. A path counts once in the union. WRITE_SET, CREATE_SET and
DELETE_SET also keep their own structural limits. A READY plan must never
exceed the active limit; the harness rejects it deterministically.

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


def _insert_before_protocol(prompt: str, text: str) -> str:
    index = prompt.find(_PROTOCOL_ANCHOR)
    if index < 0:
        return prompt.rstrip("\n") + "\n\n" + text
    return prompt[:index] + text + "\n" + prompt[index:]


def _apply_execution_mode_policy(prompt: str, policy: str) -> str:
    """Insert the configured EXECUTION_MODE policy before the wire protocol."""

    if policy == ExecutionModePolicy.AUTO.value:
        return prompt
    if policy != ExecutionModePolicy.REQUIRE_STAGED.value:
        raise ValueError("unknown execution mode policy")
    return _insert_before_protocol(prompt, REQUIRE_STAGED_POLICY_TEXT)


def _apply_decomposition_policy(
    prompt: str,
    decomposition: str,
    single_step_max_mutable_paths: int,
    staged_step_max_mutable_paths: int,
) -> str:
    """Insert the AGGRESSIVE mutable-scope policy before the wire protocol."""

    if decomposition == "balanced":
        return prompt
    if decomposition != "aggressive":
        raise ValueError("unknown planning decomposition")
    return _insert_before_protocol(
        prompt,
        render_decomposition_policy_text(
            single_step_max_mutable_paths, staged_step_max_mutable_paths
        ),
    )


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


def build_planner_prompt_v2(
    spec: str,
    context: str,
    *,
    repository_reference: RepositoryReference | None = None,
    implementer_profiles: Sequence[ModelProfile] = (),
    reviewer_profiles: Sequence[ModelProfile] = (),
    template: str | None = None,
    execution_mode_policy: str = ExecutionModePolicy.AUTO.value,
    decomposition: str = PlanningConfig.decomposition,
    single_step_max_mutable_paths: int = PlanningConfig.single_step_max_mutable_paths,
    staged_step_max_mutable_paths: int = PlanningConfig.staged_step_max_mutable_paths,
    check_catalog: Sequence[CheckConfig] = (),
    default_check_ids: Sequence[str] = (),
) -> str:
    if not isinstance(spec, str) or not isinstance(context, str):
        raise TypeError("spec and context must be strings")
    if repository_reference is not None and not isinstance(repository_reference, RepositoryReference):
        raise TypeError("repository_reference must be a RepositoryReference")
    if template is None:
        template = (Path(__file__).with_name("prompts") / "planner_v2.txt").read_text(encoding="utf-8")
    values = {
        "{{SPEC}}": spec,
        "{{CONTEXT}}": context,
        "{{REPOSITORY}}": render_repository_reference(repository_reference) if repository_reference else (
            "WEB URL:\nUNAVAILABLE\n\nBASE SHA:\nUNAVAILABLE\n\nIMMUTABLE BASE URL:\nUNAVAILABLE\n\nREMOTE EXPLORATION:\nUNAVAILABLE"
        ),
        "{{IMPLEMENTER_PROFILES}}": render_safe_profile_catalogue(implementer_profiles),
        "{{REVIEWER_PROFILES}}": render_safe_profile_catalogue(reviewer_profiles),
        "{{CHECK_CATALOG}}": render_safe_check_catalogue(check_catalog),
        "{{DEFAULT_CHECK_IDS}}": "\n".join(f"- {check_id}" for check_id in default_check_ids) or "NONE",
    }
    # Policies are inserted into the template before substitution so that
    # SPEC or context text can never impersonate or displace them.
    template = _apply_execution_mode_policy(template, execution_mode_policy)
    template = _apply_decomposition_policy(
        template, decomposition,
        single_step_max_mutable_paths, staged_step_max_mutable_paths,
    )
    return re.sub(r"\{\{(?:SPEC|CONTEXT|REPOSITORY|IMPLEMENTER_PROFILES|REVIEWER_PROFILES|CHECK_CATALOG|DEFAULT_CHECK_IDS)\}\}", lambda match: values[match.group(0)], template)


def build_repair_planner_prompt(
    *,
    repository_reference: str,
    original_spec: str,
    original_step_contracts: str,
    original_plan_summary: str | None = None,
    current_repository_state: str,
    current_cumulative_diff: str,
    final_checks_cycle_1: str,
    claude_revision_report_cycle_1: str,
    reviewer_required_fixes: str,
    original_approved_mutable_scope: str,
    candidate_commit_sha: str = "",
    candidate_immutable_url: str = "",
    reviewer_result: str = "",
    reviewer_missing_tests: str = "",
    implementer_profiles: Sequence[ModelProfile] = (),
    reviewer_profiles: Sequence[ModelProfile] = (),
    template: str | None = None,
    original_meta_plan: str | None = None,
    check_catalog: Sequence[CheckConfig] = (),
    original_required_check_ids: Sequence[str] = (),
) -> str:
    """Build the bounded corrective planner request."""

    if original_plan_summary is None:
        # Compatibility for callers from the pre-P33 API.  The prompt itself
        # always uses the renamed compact-summary placeholder.
        original_plan_summary = original_meta_plan
    elif original_meta_plan is not None and original_plan_summary != original_meta_plan:
        raise ValueError("original_plan_summary and original_meta_plan disagree")
    if original_plan_summary is None:
        raise TypeError("original_plan_summary must be a string")
    values = {
        "{{REPOSITORY}}": repository_reference,
        "{{SPEC}}": original_spec,
        "{{ORIGINAL_PLAN_SUMMARY}}": original_plan_summary,
        "{{ORIGINAL_STEP_CONTRACTS}}": original_step_contracts,
        "{{CURRENT_REPOSITORY_STATE}}": current_repository_state,
        "{{CURRENT_CUMULATIVE_DIFF}}": current_cumulative_diff,
        "{{FINAL_CHECKS_CYCLE_1}}": final_checks_cycle_1,
        "{{CLAUDE_REVISION_REPORT_CYCLE_1}}": claude_revision_report_cycle_1,
        "{{REVIEWER_REQUIRED_FIXES}}": reviewer_required_fixes,
        "{{ORIGINAL_APPROVED_MUTABLE_SCOPE}}": original_approved_mutable_scope,
        "{{CANDIDATE_COMMIT_SHA}}": candidate_commit_sha,
        "{{CANDIDATE_IMMUTABLE_URL}}": candidate_immutable_url,
        "{{REVIEWER_RESULT}}": reviewer_result,
        "{{REVIEWER_MISSING_TESTS}}": reviewer_missing_tests,
        "{{IMPLEMENTER_PROFILES}}": render_safe_profile_catalogue(implementer_profiles),
        "{{REVIEWER_PROFILES}}": render_safe_profile_catalogue(reviewer_profiles),
        "{{CHECK_CATALOG}}": render_safe_check_catalogue(check_catalog),
        "{{ORIGINAL_REQUIRED_CHECKS}}": "\n".join(f"- {check_id}" for check_id in original_required_check_ids) or "NONE",
    }
    for name, value in values.items():
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
    if template is None:
        template = (Path(__file__).with_name("prompts") / "repair_planner_v2.txt").read_text(encoding="utf-8")
    return re.sub(
        r"\{\{(?:REPOSITORY|SPEC|ORIGINAL_PLAN_SUMMARY|ORIGINAL_STEP_CONTRACTS|CURRENT_REPOSITORY_STATE|CURRENT_CUMULATIVE_DIFF|FINAL_CHECKS_CYCLE_1|CLAUDE_REVISION_REPORT_CYCLE_1|REVIEWER_REQUIRED_FIXES|REVIEWER_MISSING_TESTS|REVIEWER_RESULT|ORIGINAL_APPROVED_MUTABLE_SCOPE|CANDIDATE_COMMIT_SHA|CANDIDATE_IMMUTABLE_URL|IMPLEMENTER_PROFILES|REVIEWER_PROFILES|CHECK_CATALOG|ORIGINAL_REQUIRED_CHECKS)\}\}",
        lambda match: values[match.group(0)],
        template,
    )


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


def write_implementation_bundle(directory: str | Path, plan: TaskPlanV2) -> dict[str, Any]:
    """Write the human summary, bounded step contracts and secret-free index."""

    if not isinstance(plan, TaskPlanV2) or plan.decision is not PlanDecision.READY:
        raise V2PlanParseError("implementation bundle requires a READY v2 plan")
    _validate_bounds(plan)
    target = Path(directory)
    contracts = {step.id: render_step_contract(plan, step) for step in plan.steps}
    entries = []
    for step in plan.steps:
        entries.append(
            {
                "id": step.id,
                "title": step.title,
                "implementer_profile": step.implementer_profile,
                "depends_on": step.depends_on,
                "contract_sha256": hashlib.sha256(contracts[step.id].encode("utf-8")).hexdigest(),
            }
        )
    bundle = {
        "schema_version": 1,
        "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
        "reviewer_profile": plan.reviewer_profile,
        "required_checks": list(plan.required_checks),
        "steps": entries,
    }
    atomic_write_text(target / "implementation_contract.md", render_plan_summary_v2(plan))
    for step in plan.steps:
        # The only copy of each contract: approval hashes and runtime reads
        # these exact bytes; nothing re-renders them after this point.
        atomic_write_text(step_contract_path(target, step.id), contracts[step.id])
    atomic_write_text(target / "implementation_bundle.json", json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    # ``task_plan.json`` is the stable v2 artifact name approved by the human.
    # Keep the older suffixed name as a compatibility alias for existing tools.
    atomic_write_text(
        target / "task_plan.json",
        json.dumps(
            {**asdict(plan), "decision": plan.decision.value,
             "execution_mode": plan.execution_mode.value if plan.execution_mode else None},
            ensure_ascii=False, indent=2,
        ) + "\n",
    )
    return bundle


def step_contract_path(directory: str | Path, step_id: str) -> Path:
    """Canonical path of one step contract: ``steps/<STEP>/contract.md``."""

    if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None:
        raise V2PlanParseError("step ID must be exactly S01 through S08")
    return Path(directory) / "steps" / step_id / STEP_CONTRACT_NAME


def read_approved_step_contract(
    directory: str | Path, bundle: dict[str, Any], step_id: str
) -> str:
    """Read the exact contract bytes declared by a validated bundle.

    The bytes are hashed again at read time, so the text handed to the worker
    is exactly the file whose hash the approval bound.
    """

    entries = bundle.get("steps") if isinstance(bundle, dict) else None
    entry = next(
        (item for item in entries or () if isinstance(item, dict) and item.get("id") == step_id),
        None,
    )
    if entry is None:
        raise V2PlanParseError(f"implementation bundle has no step {step_id}")
    path = step_contract_path(Path(directory).expanduser().resolve(), step_id)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise V2PlanParseError(f"missing contract for {step_id}") from exc
    if hashlib.sha256(data).hexdigest() != entry.get("contract_sha256"):
        raise V2PlanParseError(f"contract hash mismatch for {step_id}")
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise V2PlanParseError(f"contract for {step_id} is not UTF-8") from exc
    if len(text) > MAX_STEP_CONTRACT_CHARS:
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    return text


def validate_implementation_bundle(
    directory: str | Path,
    *,
    expected_step_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the immutable v2 bundle and every contract hash it declares.

    With *expected_step_ids*, the bundle must declare exactly those steps in
    that order.  Contracts are read from the canonical per-step layout only;
    the P20 ``steps/Sxx.contract.md`` layout is never executed.
    """

    target = Path(directory).expanduser().resolve()
    bundle_path = target / "implementation_bundle.json"
    try:
        bundle_bytes = bundle_path.read_bytes()
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2PlanParseError("implementation bundle is missing or invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise V2PlanParseError("implementation bundle schema_version is invalid")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise V2PlanParseError("implementation bundle steps are invalid")
    expected_ids = [f"S{index:02d}" for index in range(1, len(steps) + 1)]
    actual_ids: list[str] = []
    for entry in steps:
        if not isinstance(entry, dict) or set(entry) != {"id", "title", "implementer_profile", "depends_on", "contract_sha256"}:
            raise V2PlanParseError("implementation bundle step entry is invalid")
        step_id = entry.get("id")
        if not isinstance(step_id, str) or step_id in actual_ids:
            raise V2PlanParseError("implementation bundle step ID is invalid")
        actual_ids.append(step_id)
        declared = entry.get("contract_sha256")
        if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
            raise V2PlanParseError("implementation bundle contract hash is invalid")
        if _STEP_ID.fullmatch(step_id) is None:
            raise V2PlanParseError("implementation bundle step ID is invalid")
        contract_path = step_contract_path(target, step_id)
        try:
            actual = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise V2PlanParseError(f"missing contract for {step_id}") from exc
        if actual != declared:
            raise V2PlanParseError(f"contract hash mismatch for {step_id}")
    if actual_ids != expected_ids:
        raise V2PlanParseError("implementation bundle step IDs are not contiguous")
    if expected_step_ids is not None and list(expected_step_ids) != actual_ids:
        raise V2PlanParseError("implementation bundle steps do not match the plan")
    return payload, hashlib.sha256(bundle_bytes).hexdigest()


persist_implementation_bundle = write_implementation_bundle
render_profile_catalogue = render_safe_profile_catalogue


def persist_planning_v2_artifacts(
    directory: str | Path,
    *,
    spec: str,
    context: str,
    request: str,
    plan: TaskPlanV2,
) -> None:
    """Persist the v2 exchange and publish its implementation bundle."""

    target = Path(directory)
    atomic_write_text(target / "spec.md", spec)
    atomic_write_text(target / "context.txt", context)
    atomic_write_text(target / "planner.request.txt", request)
    atomic_write_text(target / "planner.raw.md", plan.raw)
    atomic_write_text(
        target / "task_plan_v2.json",
        json.dumps(
            {
                **asdict(plan),
                "decision": plan.decision.value,
                "execution_mode": plan.execution_mode.value
                if plan.execution_mode is not None
                else None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    # The unsuffixed artifact is the v2 approval surface.
    if plan.decision is PlanDecision.BLOCKED:
        atomic_write_text(
            target / "task_plan.json",
            json.dumps({**asdict(plan), "decision": plan.decision.value, "execution_mode": None}, ensure_ascii=False, indent=2) + "\n",
        )
    if plan.decision is PlanDecision.READY:
        write_implementation_bundle(target, plan)


persist_planning_artifacts_v2 = persist_planning_v2_artifacts


class TextCompletionClient(Protocol):
    def complete(self, prompt: str) -> TextLLMResult | str: ...


class PlannerV2:
    """Standalone v2 planner entry point; it never invokes P17 recommender."""

    def __init__(self, client: TextCompletionClient, *, implementer_ids: frozenset[str], reviewer_ids: frozenset[str], implementer_profiles: Sequence[ModelProfile] = (), reviewer_profiles: Sequence[ModelProfile] = (), repository_reference: RepositoryReference | None = None, planning: PlanningConfig | None = None, template: str | None = None, check_catalog: Sequence[CheckConfig] = (), default_check_ids: Sequence[str] = ()):
        self.client = client
        self.implementer_ids = implementer_ids
        self.reviewer_ids = reviewer_ids
        self.implementer_profiles = implementer_profiles
        self.reviewer_profiles = reviewer_profiles
        self.repository_reference = repository_reference
        self.planning = planning or PlanningConfig(protocol="v2")
        self.template = template
        self.check_catalog = tuple(check_catalog)
        self.default_check_ids = tuple(default_check_ids)
        self.last_conversation: LLMConversationHandle | None = None

    def plan(self, spec: str, context: str, *, repository_reference: RepositoryReference | None = None, artifacts_dir: str | Path | None = None) -> TaskPlanV2:
        reference = repository_reference if repository_reference is not None else self.repository_reference
        request = build_planner_prompt_v2(
            spec, context, repository_reference=reference,
            implementer_profiles=self.implementer_profiles,
            reviewer_profiles=self.reviewer_profiles, template=self.template,
            execution_mode_policy=self.planning.execution_mode_policy,
            decomposition=self.planning.decomposition,
            single_step_max_mutable_paths=self.planning.single_step_max_mutable_paths,
            staged_step_max_mutable_paths=self.planning.staged_step_max_mutable_paths,
            check_catalog=self.check_catalog,
            default_check_ids=self.default_check_ids,
        )
        target = Path(artifacts_dir) if artifacts_dir is not None else None
        if target is not None:
            atomic_write_text(target / "planner.request.txt", request)
        result = self.client.complete(request)
        self.last_conversation = conversation_handle(result)
        raw = result if isinstance(result, str) else getattr(result, "text", None)
        if target is not None:
            # Tokens were consumed whether or not the answer parses.
            write_usage_artifact(target / PLANNER_USAGE_ARTIFACT, completion_usage(result))
        if not isinstance(raw, str):
            raise V2PlanParseError("planner client did not return text")
        if target is not None:
            atomic_write_text(target / "planner.raw.md", raw)
        plan = parse_task_plan_v2(raw, implementer_ids=self.implementer_ids, reviewer_ids=self.reviewer_ids,
                                  check_catalog=self.check_catalog, default_check_ids=self.default_check_ids)
        # Only the initial planner is bound by the execution mode policy; the
        # bounded C02 repair plan keeps its own (possibly single-step) shape.
        validate_execution_mode_policy(plan, self.planning)
        validate_decomposition_policy(plan, self.planning)
        if artifacts_dir is not None:
            persist_planning_v2_artifacts(
                artifacts_dir,
                spec=spec,
                context=context,
                request=request,
                plan=plan,
            )
        return plan


class RepairPlannerV2:
    """Planner facade for one P26 corrective cycle.

    It deliberately shares the strict META PLAN v2 parser and bundle writer;
    only its request envelope and implementer catalogue are different.
    """

    def __init__(
        self,
        client: TextCompletionClient,
        *,
        implementer_ids: frozenset[str],
        reviewer_ids: frozenset[str],
        implementer_profiles: Sequence[ModelProfile] = (),
        reviewer_profiles: Sequence[ModelProfile] = (),
        planning: PlanningConfig | None = None,
        template: str | None = None,
        check_catalog: Sequence[CheckConfig] = (),
        original_required_check_ids: Sequence[str] = (),
    ):
        self.client = client
        self.implementer_ids = implementer_ids
        self.reviewer_ids = reviewer_ids
        self.implementer_profiles = implementer_profiles
        self.reviewer_profiles = reviewer_profiles
        self.planning = planning or PlanningConfig(protocol="v2")
        self.template = template
        self.check_catalog = tuple(check_catalog)
        self.original_required_check_ids = tuple(original_required_check_ids)

    def plan(
        self,
        *,
        repository_reference: str,
        original_spec: str,
        original_step_contracts: str,
        original_plan_summary: str | None = None,
        current_repository_state: str,
        current_cumulative_diff: str,
        final_checks_cycle_1: str,
        claude_revision_report_cycle_1: str,
        reviewer_required_fixes: str,
        original_approved_mutable_scope: str,
        artifacts_dir: str | Path,
        conversation: LLMConversationHandle | None = None,
        original_meta_plan: str | None = None,
        candidate_commit_sha: str = "",
        candidate_immutable_url: str = "",
        reviewer_result: str = "",
        reviewer_missing_tests: str = "",
    ) -> TaskPlanV2:
        request = build_repair_planner_prompt(
            repository_reference=repository_reference,
            original_spec=original_spec,
            original_plan_summary=original_plan_summary,
            original_step_contracts=original_step_contracts,
            current_repository_state=current_repository_state,
            current_cumulative_diff=current_cumulative_diff,
            final_checks_cycle_1=final_checks_cycle_1,
            claude_revision_report_cycle_1=claude_revision_report_cycle_1,
            reviewer_required_fixes=reviewer_required_fixes,
            original_approved_mutable_scope=original_approved_mutable_scope,
            candidate_commit_sha=candidate_commit_sha,
            candidate_immutable_url=candidate_immutable_url,
            reviewer_result=reviewer_result,
            reviewer_missing_tests=reviewer_missing_tests,
            implementer_profiles=self.implementer_profiles,
            reviewer_profiles=self.reviewer_profiles,
            template=self.template,
            original_meta_plan=original_meta_plan,
            check_catalog=self.check_catalog,
            original_required_check_ids=self.original_required_check_ids,
        )
        target = Path(artifacts_dir)
        atomic_write_text(target / "planner.request.txt", request)
        resume_conversation = getattr(self.client, "complete_in_conversation", None)
        if isinstance(conversation, LLMConversationHandle) and callable(resume_conversation):
            # The only allowed conversational reuse: the driver officially
            # exposed the initial planner conversation handle.
            result = resume_conversation(conversation, request)
        else:
            result = self.client.complete(request)
        raw = result if isinstance(result, str) else getattr(result, "text", None)
        write_usage_artifact(target / PLANNER_USAGE_ARTIFACT, completion_usage(result))
        if not isinstance(raw, str):
            raise V2PlanParseError("repair planner client did not return text")
        atomic_write_text(target / "planner.raw.md", raw)
        plan = parse_task_plan_v2(
            raw, implementer_ids=self.implementer_ids, reviewer_ids=self.reviewer_ids,
            check_catalog=self.check_catalog, inherited_check_ids=self.original_required_check_ids,
        )
        validate_decomposition_policy(plan, self.planning)
        if plan.decision is PlanDecision.READY:
            persist_planning_v2_artifacts(
                target, spec=original_spec, context=current_repository_state,
                request=request, plan=plan,
            )
        else:
            atomic_write_text(
                target / "task_plan.json",
                json.dumps({**asdict(plan), "decision": plan.decision.value, "execution_mode": None}, ensure_ascii=False, indent=2) + "\n",
            )
        return plan


def run_planner_v2(
    client: TextCompletionClient,
    spec: str,
    context: str,
    *,
    implementer_ids: frozenset[str],
    reviewer_ids: frozenset[str],
    implementer_profiles: Sequence[ModelProfile] = (),
    reviewer_profiles: Sequence[ModelProfile] = (),
    repository_reference: RepositoryReference | None = None,
    planning: PlanningConfig | None = None,
    artifacts_dir: str | Path | None = None,
    template: str | None = None,
    check_catalog: Sequence[CheckConfig] = (),
    default_check_ids: Sequence[str] = (),
) -> TaskPlanV2:
    return PlannerV2(
        client,
        implementer_ids=implementer_ids,
        reviewer_ids=reviewer_ids,
        implementer_profiles=implementer_profiles,
        reviewer_profiles=reviewer_profiles,
        repository_reference=repository_reference,
        planning=planning,
        template=template,
        check_catalog=check_catalog,
        default_check_ids=default_check_ids,
    ).plan(spec, context, artifacts_dir=artifacts_dir)


__all__ = [
    "ExecutionMode", "ImplementationStep", "MAX_CREATE_SET", "MAX_DELETE_SET", "MAX_READ_SET",
    "MAX_STEPS", "MAX_STEP_CONTRACT_CHARS", "MAX_TOTAL_STEP_CONTRACT_CHARS", "MAX_WRITE_SET",
    "PlannerV2", "STEP_CONTRACT_NAME", "TaskPlanV2", "V2PlanParseError",
    "PlanDecision", "PlanParseError",
    "build_planner_prompt_v2", "parse_task_plan_v2", "persist_implementation_bundle",
    "build_repair_planner_prompt", "RepairPlannerV2",
    "persist_planning_artifacts_v2", "persist_planning_v2_artifacts",
    "read_approved_step_contract", "read_set_paths",
    "render_plan_summary_v2", "render_repair_plan_summary", "render_profile_catalogue", "render_safe_profile_catalogue", "render_step_contract",
    "run_planner_v2", "step_contract_path", "validate_decomposition_policy", "validate_implementation_bundle", "write_implementation_bundle",
    "REQUIRE_STAGED_POLICY_TEXT", "validate_execution_mode_policy",
    "render_decomposition_policy_text",
]
