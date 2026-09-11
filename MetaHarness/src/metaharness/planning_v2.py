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

from .llm.chat import TextLLMResult
from .models import ExecutionMode, ImplementationStep, ModelProfile, TaskPlanV2
from .planning import PlanDecision, PlanParseError
from .result import atomic_write_text


MAX_STEPS = 6
MAX_STEP_CONTRACT_CHARS = 8_000
MAX_TOTAL_STEP_CONTRACT_CHARS = 32_000

_HEADER = "META PLAN v2"
_END = "END META PLAN"
_STEP_BEGIN = re.compile(r"^BEGIN STEP (.+)$")
_STEP_END = re.compile(r"^END STEP (.+)$")
_STEP_ID = re.compile(r"S0[1-6]")
_INLINE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$")
_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_TEXT_SECTIONS = frozenset(
    {"OBJECTIVE", "CONSTRAINTS", "READ_SET", "WRITE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"}
)
_ENVELOPE_INLINE = frozenset({"STATUS", "TITLE", "EXECUTION_MODE", "STEP_COUNT", "REVIEWER_PROFILE"})
_ENVELOPE_SECTIONS = frozenset({"OBJECTIVE", "CONSTRAINTS", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"})
_STEP_INLINE = frozenset({"TITLE", "IMPLEMENTER_PROFILE", "DEPENDS_ON"})
_STEP_SECTIONS = frozenset({"OBJECTIVE", "READ_SET", "WRITE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"})


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
            raise V2PlanParseError("step ID must be exactly S01 through S06")
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
    if len(result) > 8:
        raise V2PlanParseError("READ_SET may contain at most 8 paths")
    return tuple(result)


def _write_set(value: str, read_set: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        raise V2PlanParseError("WRITE_SET is missing")
    reads = {item.split(" :: ", 1)[0] for item in read_set}
    result: list[str] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- ") or " :: " in line:
            raise V2PlanParseError("each WRITE_SET line must be '- path'")
        path = _repo_path(line[2:].strip(), kind="WRITE_SET")
        if path in result:
            raise V2PlanParseError("duplicate WRITE_SET path")
        if path not in reads:
            raise V2PlanParseError("every WRITE_SET path must also appear in READ_SET")
        result.append(path)
    if not result:
        raise V2PlanParseError("WRITE_SET is missing")
    if len(result) > 6:
        raise V2PlanParseError("WRITE_SET may contain at most 6 paths")
    return tuple(result)


def _parse_step(step_id: str, body: Sequence[str], implementer_ids: frozenset[str], prior_ids: frozenset[str]) -> ImplementationStep:
    values, _ = _parse_labeled_body(
        body, inline_names=_STEP_INLINE, section_names=_STEP_SECTIONS, where=f"step {step_id}"
    )
    for name in ("TITLE", "IMPLEMENTER_PROFILE", "DEPENDS_ON", "OBJECTIVE", "READ_SET", "WRITE_SET", "INSTRUCTIONS", "VERIFY", "FORBIDDEN"):
        if not values.get(name, "").strip():
            raise V2PlanParseError(f"step {step_id} is missing {name}")
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
    write_set = _write_set(values["WRITE_SET"], read_set)
    instructions = _nonempty(values["INSTRUCTIONS"], f"step {step_id} INSTRUCTIONS")
    verify = _nonempty(values["VERIFY"], f"step {step_id} VERIFY")
    forbidden = _nonempty(values["FORBIDDEN"], f"step {step_id} FORBIDDEN")
    return ImplementationStep(step_id, title, profile, None if dependency == "NONE" else dependency, objective, read_set, write_set, instructions, verify, forbidden)


def _render_step_contract_unchecked(plan: TaskPlanV2, step: ImplementationStep) -> str:
    def lines(items: tuple[str, ...]) -> str:
        return "\n".join(f"- {item}" for item in items)

    return "\n\n".join(
        (
            "META IMPLEMENTATION STEP v1",
            "RUN TITLE\n" + plan.title,
            f"STEP\n{step.id} / {len(plan.steps):02d}",
            "TITLE\n" + step.title,
            "OBJECTIVE\n" + step.objective,
            "READ SET\n" + lines(step.read_set),
            "WRITE SET\n" + lines(step.write_set),
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


def parse_task_plan_v2(raw: str, *, implementer_ids: frozenset[str], reviewer_ids: frozenset[str]) -> TaskPlanV2:
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
        raise V2PlanParseError("STAGED requires between two and six steps")
    reviewer = inline.get("REVIEWER_PROFILE", "")
    if not _PROFILE.fullmatch(reviewer) or reviewer not in reviewer_ids:
        raise V2PlanParseError("unknown reviewer profile")
    if len(blocks) != step_count:
        raise V2PlanParseError("STEP_COUNT does not match step blocks")
    expected = [f"S{index:02d}" for index in range(1, step_count + 1)]
    if [step_id for step_id, _ in blocks] != expected:
        raise V2PlanParseError("step IDs must be contiguous S01 through S06")
    steps: list[ImplementationStep] = []
    for step_id, body in blocks:
        steps.append(_parse_step(step_id, body, implementer_ids, frozenset(step.id for step in steps)))
    for name in ("CONSTRAINTS", "ACCEPTANCE", "TESTS", "RISKS", "BLOCKERS"):
        if name not in sections or not sections[name].strip():
            raise V2PlanParseError(f"READY plan is missing {name}")
    if blockers and blockers.casefold() not in {"none", "n/a", "na", "-", "—", "nil"}:
        raise V2PlanParseError("READY plan cannot contain real BLOCKERS")
    plan = TaskPlanV2(
        decision, title, objective, sections["CONSTRAINTS"].strip(), ExecutionMode(mode), reviewer,
        tuple(steps), sections["ACCEPTANCE"].strip(), sections["TESTS"].strip(), sections["RISKS"].strip(), blockers or "NONE", raw
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
            "## Ordered steps\n" + ("\n\n".join(summaries) if summaries else "NONE"),
            f"## Acceptance\n{plan.acceptance or 'NONE'}",
            f"## Tests\n{plan.tests or 'NONE'}",
            f"## Risks\n{plan.risks or 'NONE'}",
        )
    ) + "\n"


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


def build_planner_prompt_v2(
    spec: str,
    context: str,
    *,
    implementer_profiles: Sequence[ModelProfile] = (),
    reviewer_profiles: Sequence[ModelProfile] = (),
    template: str | None = None,
) -> str:
    if not isinstance(spec, str) or not isinstance(context, str):
        raise TypeError("spec and context must be strings")
    if template is None:
        template = (Path(__file__).with_name("prompts") / "planner_v2.txt").read_text(encoding="utf-8")
    values = {
        "{{SPEC}}": spec,
        "{{CONTEXT}}": context,
        "{{IMPLEMENTER_PROFILES}}": render_safe_profile_catalogue(implementer_profiles),
        "{{REVIEWER_PROFILES}}": render_safe_profile_catalogue(reviewer_profiles),
    }
    return re.sub(r"\{\{SPEC\}\}|\{\{CONTEXT\}\}|\{\{IMPLEMENTER_PROFILES\}\}|\{\{REVIEWER_PROFILES\}\}", lambda match: values[match.group(0)], template)


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
        "steps": entries,
    }
    atomic_write_text(target / "implementation_contract.md", render_plan_summary_v2(plan))
    for step in plan.steps:
        atomic_write_text(target / "steps" / f"{step.id}.contract.md", contracts[step.id])
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


def validate_implementation_bundle(directory: str | Path) -> tuple[dict[str, Any], str]:
    """Validate the immutable v2 bundle and every contract hash it declares."""

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
        contract_path = target / "steps" / f"{step_id}.contract.md"
        try:
            actual = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise V2PlanParseError(f"missing contract for {step_id}") from exc
        if actual != declared:
            raise V2PlanParseError(f"contract hash mismatch for {step_id}")
    if actual_ids != expected_ids:
        raise V2PlanParseError("implementation bundle step IDs are not contiguous")
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

    def __init__(self, client: TextCompletionClient, *, implementer_ids: frozenset[str], reviewer_ids: frozenset[str], implementer_profiles: Sequence[ModelProfile] = (), reviewer_profiles: Sequence[ModelProfile] = (), template: str | None = None):
        self.client = client
        self.implementer_ids = implementer_ids
        self.reviewer_ids = reviewer_ids
        self.implementer_profiles = implementer_profiles
        self.reviewer_profiles = reviewer_profiles
        self.template = template

    def plan(self, spec: str, context: str, *, artifacts_dir: str | Path | None = None) -> TaskPlanV2:
        request = build_planner_prompt_v2(spec, context, implementer_profiles=self.implementer_profiles, reviewer_profiles=self.reviewer_profiles, template=self.template)
        result = self.client.complete(request)
        raw = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(raw, str):
            raise V2PlanParseError("planner client did not return text")
        plan = parse_task_plan_v2(raw, implementer_ids=self.implementer_ids, reviewer_ids=self.reviewer_ids)
        if artifacts_dir is not None:
            persist_planning_v2_artifacts(
                artifacts_dir,
                spec=spec,
                context=context,
                request=request,
                plan=plan,
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
    artifacts_dir: str | Path | None = None,
    template: str | None = None,
) -> TaskPlanV2:
    return PlannerV2(
        client,
        implementer_ids=implementer_ids,
        reviewer_ids=reviewer_ids,
        implementer_profiles=implementer_profiles,
        reviewer_profiles=reviewer_profiles,
        template=template,
    ).plan(spec, context, artifacts_dir=artifacts_dir)


__all__ = [
    "ExecutionMode", "ImplementationStep", "MAX_STEPS", "MAX_STEP_CONTRACT_CHARS",
    "MAX_TOTAL_STEP_CONTRACT_CHARS", "PlannerV2", "TaskPlanV2", "V2PlanParseError",
    "PlanDecision", "PlanParseError",
    "build_planner_prompt_v2", "parse_task_plan_v2", "persist_implementation_bundle",
    "persist_planning_artifacts_v2", "persist_planning_v2_artifacts",
    "render_plan_summary_v2", "render_profile_catalogue", "render_safe_profile_catalogue", "render_step_contract",
    "run_planner_v2", "validate_implementation_bundle", "write_implementation_bundle",
]
