"""META PLAN v2 wire protocol: grammar, parsing and deterministic rendering.

Pure protocol authority: this module parses planner answers into v2 model
values and renders the step contracts and planner-visible catalogues.  It
performs no I/O, calls no model and never touches Git.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Sequence

from ..models import (
    BlockerKind,
    CheckConfig,
    ExecutionClass,
    ExecutionMode,
    ImplementationStep,
    PlanDecision,
    PlanningConfig,
    TaskPlanV2,
)
from ..plan_repository_validation import MAX_BLOCKERS_CHARS
from ..step_ids import LAST_STEP_ID, MAX_STEPS, STEP_ID_RE, step_ids

MAX_STEP_CONTRACT_CHARS = 5_000
# Canonical layout of the approved step contracts, written at planning time
# and executed byte-for-byte: ``steps/<STEP>/contract.md``.
STEP_CONTRACT_NAME = "contract.md"

_HEADER = "META PLAN v2"
_END = "END META PLAN"
_STEP_BEGIN = re.compile(r"^BEGIN STEP (.+)$")
_STEP_END = re.compile(r"^END STEP (.+)$")
_STEP_ID = STEP_ID_RE
STEP_ID_RANGE = f"S01 through {LAST_STEP_ID}"
_INLINE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$")

_ENVELOPE_INLINE = frozenset({"STATUS", "TITLE", "EXECUTION_MODE", "STEP_COUNT", "BLOCKER_KIND"})
_ENVELOPE_SECTIONS = frozenset({"OBJECTIVE", "CONSTRAINTS", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS", "REQUIRED_CHECKS"})
_STEP_INLINE = frozenset({"TITLE", "EXECUTION_CLASS", "DEPENDS_ON"})
_STEP_SECTIONS = frozenset({"OBJECTIVE", "READ_SET", "WRITE_SET", "CREATE_SET", "DELETE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"})


class PlanParseError(ValueError):
    """A planner answer was received but cannot be interpreted unambiguously."""


class V2PlanParseError(PlanParseError):
    """A META PLAN v2 response is not safe to execute."""


def _lines(raw: str) -> list[str]:
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _nonempty(value: str, name: str) -> str:
    value = value.strip()
    if not value or value.casefold() in {"none", "n/a", "na", "-", "—", "nil", "tbd"}:
        raise V2PlanParseError(f"{name} is missing or a placeholder")
    return value


def _validate_step_text_limits(step_id: str, values: dict[str, str]) -> None:
    title = values["TITLE"].strip()
    if len(title) > 100:
        raise V2PlanParseError(f"step {step_id} TITLE exceeds 100 characters")
    instruction_lines = [line for line in values["INSTRUCTIONS"].splitlines() if line.strip()]
    numbered = [line for line in instruction_lines if re.match(r"^\s*\d+[.)]\s+", line)]
    if len(numbered) > 6:
        raise V2PlanParseError(f"step {step_id} INSTRUCTIONS exceeds 6 operations")
    verify_lines = [line for line in values["VERIFY"].splitlines() if line.strip()]
    if len(verify_lines) > 3:
        raise V2PlanParseError(f"step {step_id} VERIFY exceeds 3 lines")
    forbidden_lines = [line for line in values["FORBIDDEN"].splitlines() if line.strip()]
    if len(forbidden_lines) > 4:
        raise V2PlanParseError(f"step {step_id} FORBIDDEN exceeds 4 rules")


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
            raise V2PlanParseError(f"step ID must be exactly {STEP_ID_RANGE}")
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


def _read_set(value: str, *, max_paths: int) -> tuple[str, ...]:
    if not value.strip():
        raise V2PlanParseError("READ_SET is missing")
    anchors_by_path: dict[str, list[str]] = {}
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
        # Repeated paths merge their anchors instead of failing: the scope is
        # unchanged, only the serialisation was split across lines.
        anchors = anchors_by_path.setdefault(path, [])
        if anchor not in anchors:
            anchors.append(anchor)
    if not anchors_by_path:
        raise V2PlanParseError("READ_SET is missing")
    if len(anchors_by_path) > max_paths:
        raise V2PlanParseError(
            f"READ_SET contains {len(anchors_by_path)} unique paths; "
            f"maximum is {max_paths}"
        )
    return tuple(path + " :: " + "; ".join(anchors) for path, anchors in anchors_by_path.items())


def read_set_paths(read_set: Sequence[str]) -> tuple[str, ...]:
    """The repo-relative paths of ``'path :: anchor'`` READ_SET entries."""

    return tuple(item.split(" :: ", 1)[0] for item in read_set)


def _path_set(value: str, *, name: str) -> tuple[str, ...]:
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
    return tuple(result)


def _change_sets(
    values: dict[str, str], read_set: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate WRITE/CREATE/DELETE sets against READ_SET and each other."""

    reads = set(read_set_paths(read_set))
    write_set = _path_set(values["WRITE_SET"], name="WRITE_SET")
    create_set = _path_set(values["CREATE_SET"], name="CREATE_SET")
    delete_set = _path_set(values["DELETE_SET"], name="DELETE_SET")
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


def _parse_step(
    step_id: str,
    body: Sequence[str],
    prior_ids: frozenset[str],
    *,
    max_read_paths_per_step: int,
) -> ImplementationStep:
    values, _ = _parse_labeled_body(
        body, inline_names=_STEP_INLINE, section_names=_STEP_SECTIONS, where=f"step {step_id}"
    )
    for name in ("TITLE", "EXECUTION_CLASS", "DEPENDS_ON", "OBJECTIVE", "READ_SET", "WRITE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"):
        if not values.get(name, "").strip():
            raise V2PlanParseError(f"step {step_id} is missing {name}")
    for name in ("CREATE_SET", "DELETE_SET"):
        if name not in values:
            raise V2PlanParseError(f"step {step_id} is missing {name}")
        if not values[name].strip():
            raise V2PlanParseError(f"step {step_id} has an empty {name}; use NONE")
    _validate_step_text_limits(step_id, values)
    title = _nonempty(values["TITLE"], f"step {step_id} TITLE")
    execution_class = values["EXECUTION_CLASS"]
    if execution_class not in {item.value for item in ExecutionClass}:
        raise V2PlanParseError(f"unknown execution class in {step_id}")
    dependency = values["DEPENDS_ON"]
    if dependency != "NONE":
        if _STEP_ID.fullmatch(dependency) is None or dependency not in prior_ids:
            raise V2PlanParseError(f"invalid or future dependency in {step_id}")
    objective = _nonempty(values["OBJECTIVE"], f"step {step_id} OBJECTIVE")
    read_set = _read_set(values["READ_SET"], max_paths=max_read_paths_per_step)
    write_set, create_set, delete_set = _change_sets(values, read_set)
    instructions = _nonempty(values["INSTRUCTIONS"], f"step {step_id} INSTRUCTIONS")
    verify = _nonempty(values["VERIFY"], f"step {step_id} VERIFY")
    forbidden = _nonempty(values["FORBIDDEN"], f"step {step_id} FORBIDDEN")
    return ImplementationStep(
        step_id, title, ExecutionClass(execution_class), None if dependency == "NONE" else dependency,
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
            "EXECUTION CLASS\n" + step.execution_class.value,
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
    if len(rendered) > plan.max_step_contract_chars:
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    return rendered


STEP_REPAIR_HEADER = "META STEP CONTRACT REPAIR v1"
STEP_REPAIR_END = "END META STEP CONTRACT REPAIR"
_STEP_REPAIR_INLINE = frozenset({"STEP_ID", "TITLE", "EXECUTION_CLASS", "DEPENDS_ON"})
_STEP_REPAIR_SECTIONS = frozenset({
    "OBJECTIVE", "READ_SET", "WRITE_SET", "CREATE_SET", "DELETE_SET",
    "INSTRUCTIONS", "VERIFY", "FORBIDDEN",
})
_STEP_REPAIR_ID_WITH_COUNT = re.compile(r"^(S\d{2})\s*/\s*(\d{1,2})$")


def normalize_repair_step_id(
    value: str,
    *,
    expected_step_id: str | None,
    expected_plan_step_count: int | None,
) -> str:
    """Normalize only the approved ``STEP_ID / plan-count`` display form.

    A suffix is accepted only when its primary ID exactly matches the expected
    step identity.  The general step-ID grammar remains strict everywhere
    else, including ``step_ids.py``.
    """

    if expected_plan_step_count is not None and (
        isinstance(expected_plan_step_count, bool)
        or not isinstance(expected_plan_step_count, int)
        or not 1 <= expected_plan_step_count <= MAX_STEPS
    ):
        raise V2PlanParseError("expected plan step count is invalid")
    if not isinstance(value, str):
        raise V2PlanParseError("step contract repair STEP_ID is invalid")
    candidate = value.strip()
    if STEP_ID_RE.fullmatch(candidate) is not None:
        if expected_step_id is not None and candidate != expected_step_id:
            raise V2PlanParseError(
                f"step contract repair STEP_ID changed: expected {expected_step_id!r}, got {candidate!r}"
            )
        return candidate
    match = _STEP_REPAIR_ID_WITH_COUNT.fullmatch(candidate)
    if match is None:
        raise V2PlanParseError("step contract repair STEP_ID is invalid")
    primary_id, count_text = match.groups()
    if (
        expected_step_id is None
        or STEP_ID_RE.fullmatch(expected_step_id) is None
        or primary_id != expected_step_id
    ):
        raise V2PlanParseError("step contract repair STEP_ID is invalid")
    suffix_count = int(count_text)
    if (
        not 1 <= suffix_count <= MAX_STEPS
        or (expected_plan_step_count is None and len(count_text) != 2)
        or (
            expected_plan_step_count is not None
            and suffix_count != expected_plan_step_count
        )
    ):
        raise V2PlanParseError("step contract repair STEP_ID is invalid")
    return primary_id


def parse_step_contract_repair(
    raw: str,
    *,
    max_read_paths_per_step: int,
    expected_step_id: str | None = None,
    expected_title: str | None = None,
    expected_execution_class: str | None = None,
    expected_depends_on: str | None = None,
    expected_plan_step_count: int | None = None,
    _normalizations: list[dict[str, str]] | None = None,
) -> ImplementationStep:
    """Parse one complete, standalone repaired step contract."""

    if not isinstance(raw, str) or not raw.strip():
        raise V2PlanParseError("step contract repair is empty")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
        if _normalizations is not None:
            _normalizations.append({
                "field": "ENVELOPE", "rule": "single_utf8_bom_prefix",
                "raw": "\ufeff", "canonical": "",
            })
    lines = [line.rstrip() for line in _lines(raw)]
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None or lines[first].strip() != STEP_REPAIR_HEADER:
        raise V2PlanParseError("missing META STEP CONTRACT REPAIR v1 header")
    ends = [index for index, line in enumerate(lines) if line.strip() == STEP_REPAIR_END]
    if len(ends) != 1 or ends[0] <= first:
        raise V2PlanParseError("missing or duplicate END META STEP CONTRACT REPAIR")
    end = ends[0]
    if any(line.strip() for line in lines[:first]) or any(line.strip() for line in lines[end + 1:]):
        raise V2PlanParseError("content outside step contract repair envelope")
    inline, sections = _parse_labeled_body(
        lines[first + 1:end],
        inline_names=_STEP_REPAIR_INLINE,
        section_names=_STEP_REPAIR_SECTIONS,
        where="step contract repair",
    )
    raw_step_id = inline.get("STEP_ID", "")
    step_id = normalize_repair_step_id(
        raw_step_id,
        expected_step_id=expected_step_id,
        expected_plan_step_count=expected_plan_step_count,
    )
    if step_id != raw_step_id and _normalizations is not None:
        _normalizations.append({
            "field": "STEP_ID",
            "rule": (
                "step_id_with_plan_count_suffix"
                if expected_plan_step_count is not None
                else "step_id_with_identity_suffix"
            ),
            "raw": raw_step_id,
            "canonical": step_id,
        })
    title = _nonempty(inline.get("TITLE", ""), "step contract repair TITLE")
    execution_class = inline.get("EXECUTION_CLASS")
    if execution_class not in {item.value for item in ExecutionClass}:
        raise V2PlanParseError("step contract repair EXECUTION_CLASS is invalid")
    dependency = inline.get("DEPENDS_ON", "")
    if dependency != "NONE" and _STEP_ID.fullmatch(dependency) is None:
        raise V2PlanParseError("step contract repair DEPENDS_ON is invalid")
    for name, expected, actual in (
        ("STEP_ID", expected_step_id, step_id),
        ("TITLE", expected_title, title),
        ("EXECUTION_CLASS", expected_execution_class, execution_class),
        ("DEPENDS_ON", expected_depends_on, dependency),
    ):
        if expected is not None and expected != actual:
            raise V2PlanParseError(
                f"step contract repair {name} changed: expected {expected!r}, got {actual!r}"
            )
    for name in _STEP_REPAIR_SECTIONS:
        if not sections.get(name, "").strip():
            raise V2PlanParseError(f"step contract repair is missing {name}")
    _validate_step_text_limits(step_id, sections | {"TITLE": title})
    read_set = _read_set(sections["READ_SET"], max_paths=max_read_paths_per_step)
    write_set, create_set, delete_set = _change_sets(sections, read_set)
    return ImplementationStep(
        step_id, title, ExecutionClass(execution_class),
        None if dependency == "NONE" else dependency,
        _nonempty(sections["OBJECTIVE"], "step contract repair OBJECTIVE"),
        read_set,
        write_set,
        _nonempty(sections["INSTRUCTIONS"], "step contract repair INSTRUCTIONS"),
        _nonempty(sections["VERIFY"], "step contract repair VERIFY"),
        _nonempty(sections["FORBIDDEN"], "step contract repair FORBIDDEN"),
        create_set=create_set,
        delete_set=delete_set,
    )


def render_repaired_step_contract(step: ImplementationStep) -> str:
    """Render the durable canonical form of an effective repaired contract."""

    def lines(items: tuple[str, ...]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "NONE"

    return "\n\n".join((
        STEP_REPAIR_HEADER,
        f"STEP_ID: {step.id}",
        f"TITLE: {step.title}",
        f"EXECUTION_CLASS: {step.execution_class.value}",
        f"DEPENDS_ON: {step.depends_on or 'NONE'}",
        "OBJECTIVE\n" + step.objective,
        "READ_SET\n" + lines(step.read_set),
        "WRITE_SET\n" + lines(step.write_set),
        "CREATE_SET\n" + lines(step.create_set),
        "DELETE_SET\n" + lines(step.delete_set),
        "INSTRUCTIONS\n" + step.instructions,
        "VERIFY\n" + step.verify,
        "FORBIDDEN\n" + step.forbidden,
        STEP_REPAIR_END,
    )) + "\n"


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
        if blocks or any(name in inline for name in ("EXECUTION_MODE", "STEP_COUNT")):
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

    mode = inline.get("EXECUTION_MODE")
    if mode not in {item.value for item in ExecutionMode}:
        raise V2PlanParseError("EXECUTION_MODE must be exactly SINGLE or STAGED")
    count_text = inline.get("STEP_COUNT", "")
    if not re.fullmatch(r"[0-9]+", count_text):
        raise V2PlanParseError("STEP_COUNT must be an integer")
    step_count = int(count_text)
    if step_count < 1 or step_count > MAX_STEPS:
        raise V2PlanParseError("STEP_COUNT exceeds protocol bounds")
    if step_count > planning.max_steps_per_plan:
        raise V2PlanParseError(
            f"STEP_COUNT exceeds planning.max_steps_per_plan ({planning.max_steps_per_plan})"
        )
    if mode == ExecutionMode.SINGLE.value and step_count != 1:
        raise V2PlanParseError("SINGLE requires exactly one step")
    if mode == ExecutionMode.STAGED.value and not 2 <= step_count <= MAX_STEPS:
        raise V2PlanParseError(f"STAGED requires between 2 and {MAX_STEPS} steps")
    if len(blocks) != step_count:
        raise V2PlanParseError("STEP_COUNT does not match step blocks")
    if [step_id for step_id, _ in blocks] != list(step_ids(step_count)):
        raise V2PlanParseError(f"step IDs must be contiguous {STEP_ID_RANGE}")
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
        max_step_contract_chars=planning.max_step_contract_chars,
    )
    validate_step_contract_bounds(plan)
    return plan


def render_plan_summary_v2(plan: TaskPlanV2) -> str:
    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    mode = plan.execution_mode.value if plan.execution_mode is not None else "NONE"
    summaries = []
    for step in plan.steps:
        summaries.append(f"{step.id} — {step.title} [{step.execution_class.value}]\n{step.objective}")
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


def render_repair_step_index(plan: TaskPlanV2) -> str:
    """Render the compact original-plan index the repair planner needs.

    The immutable candidate commit is the authority on the implementation, so
    the index carries only what the plan itself decided: objective, approved
    mutation scope, verification and prohibitions.  Instructions, READ_SET,
    profiles and worker reports are deliberately absent.
    """

    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")

    payload = []

    for step in plan.steps:
        payload.append(
            {
                "id": step.id,
                "title": step.title,
                "depends_on": step.depends_on,
                "execution_class": step.execution_class.value,
                "objective": step.objective,
                "mutation_scope": {
                    "write": list(step.write_set),
                    "create": list(step.create_set),
                    "delete": list(step.delete_set),
                },
                "verify": step.verify,
                "forbidden": step.forbidden,
            }
        )

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


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
    "STEP_CONTRACT_NAME",
    "STEP_ID_RANGE",
    "STEP_REPAIR_END",
    "STEP_REPAIR_HEADER",
    "PlanParseError",
    "V2PlanParseError",
    "normalize_repair_step_id",
    "parse_step_contract_repair",
    "parse_task_plan_v2",
    "read_set_paths",
    "render_plan_summary_v2",
    "render_repair_plan_summary",
    "render_repair_step_index",
    "render_repaired_step_contract",
    "render_safe_check_catalogue",
    "render_step_contract",
    "validate_step_contract_bounds",
]
