"""PLANNER_CONTINUE: the authority that closes one audited milestone.

After the gate and the audit of one iteration it answers COMPLETE, NEXT (one
complete META PLAN v2 for the next milestone) or SPEC_DECISION (the single
product question that blocks progress).  This module owns that wire protocol,
its parser, the stagnation fingerprint and the service; request facts and their
durable files live in :mod:`metaharness.planning.continue_request`.  It never
touches Git and propagates transport failures unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Sequence

from ..llm.chat import LLMProtocolError
from ..models import PlanDecision, PlanningConfig, TaskPlanV2
from ..result import atomic_write_text
from ..usage import completion_usage
from . import TextCompletionClient
from .continue_request import (
    PlannerContinueFacts,
    build_planner_continue_payload,
    planner_continue_dir,
    planner_continue_request_record,
    planner_continue_result_record,
    write_planner_continue_raw,
    write_planner_continue_request,
    write_planner_continue_result,
)
from .grammar import V2PlanParseError, lines, nonempty, parse_labeled_body
from .protocol import MILESTONE_ID_RE, parse_task_plan_v2
from .validation import validate_execution_mode_policy

CONTINUE_HEADER, CONTINUE_END = "META CONTINUE v1", "END META CONTINUE"
CONTINUE_PLAN_BEGIN, CONTINUE_PLAN_END = "BEGIN NEXT PLAN", "END NEXT PLAN"
_FIELDS = ("DECISION", "SUMMARY", "REMAINING", "NEXT_MILESTONE", "SPEC_QUESTION")
_NONE_WORDS = frozenset({"none", "n/a", "na", "-", "—", "nil", "tbd"})
# One concrete question, bounded: a real decision never needs a paragraph.
MAX_SPEC_QUESTION_CHARS = 500


class ContinueDecision(StrEnum):
    COMPLETE = "COMPLETE"
    NEXT = "NEXT"
    SPEC_DECISION = "SPEC_DECISION"


@dataclass(frozen=True)
class PlannerContinueResult:
    """One parsed decision; ``next_plan`` is the only plan it may carry."""

    decision: ContinueDecision
    summary: str
    remaining: tuple[str, ...]
    next_milestone: str | None
    spec_question: str | None
    next_plan: TaskPlanV2 | None

    def __post_init__(self) -> None:
        paired = (self.next_plan is None) == (self.next_milestone is None)
        if not paired or (self.next_plan is not None) != (self.decision is ContinueDecision.NEXT):
            raise ValueError("only a NEXT continuation carries a milestone or plan")
        if (self.decision is ContinueDecision.SPEC_DECISION) != (self.spec_question is not None):
            raise ValueError("SPEC_QUESTION belongs to SPEC_DECISION only")


def _remaining(value: str) -> tuple[str, ...]:
    entries = [line.strip() for line in value.splitlines() if line.strip()]
    if any(not entry.startswith("- ") or len(entry) == 2 for entry in entries):
        raise V2PlanParseError("each REMAINING line must be a non-empty '- item'")
    items = [entry[2:].strip() for entry in entries]
    return tuple(item for item in items if item.casefold() != "none")


def parse_planner_continue(
    raw: str, *, planning: PlanningConfig | None = None,
    check_catalog: Sequence[Any] = (), default_check_ids: Sequence[str] = (),
) -> PlannerContinueResult:
    """Parse the exact META CONTINUE v1 protocol; no model, no correction."""

    if not isinstance(raw, str):
        raise TypeError("planner continue response must be a string")
    if not raw.strip():
        raise V2PlanParseError("planner continue response is empty")
    planning = planning or PlanningConfig()
    if not isinstance(planning, PlanningConfig):
        raise TypeError("planning must be a PlanningConfig")
    raw_lines = lines(raw)
    heads = [index for index, line in enumerate(raw_lines) if line.strip() == CONTINUE_HEADER]
    ends = [index for index, line in enumerate(raw_lines) if line.strip() == CONTINUE_END]
    first = next((index for index, line in enumerate(raw_lines) if line.strip()), None)
    if len(heads) != 1 or first != heads[0]:
        raise V2PlanParseError("answer needs exactly one META CONTINUE v1 envelope")
    if len(ends) != 1 or ends[0] <= heads[0]:
        raise V2PlanParseError("answer needs exactly one END META CONTINUE footer")
    if any(line.strip() for line in raw_lines[ends[0] + 1:]):
        raise V2PlanParseError("content after END META CONTINUE")
    # The embedded META PLAN v2 is one closed block, extracted byte-for-byte.
    body = raw_lines[heads[0] + 1:ends[0]]
    begins = [index for index, line in enumerate(body) if line.strip() == CONTINUE_PLAN_BEGIN]
    closes = [index for index, line in enumerate(body) if line.strip() == CONTINUE_PLAN_END]
    if len(begins) > 1 or len(begins) != len(closes) or (begins and closes[0] <= begins[0]):
        raise V2PlanParseError("NEXT PLAN must be exactly one closed block")
    inner_raw = None
    envelope = list(body)
    if begins:
        inner_raw = "\n".join(body[begins[0] + 1:closes[0]])
        envelope = list(body[:begins[0]]) + list(body[closes[0] + 1:])
    _inline, values = parse_labeled_body(
        envelope, inline_names=frozenset(_FIELDS), section_names=frozenset(_FIELDS),
        where="planner continue")
    if set(values) != set(_FIELDS) or any(not values[name].strip() for name in _FIELDS):
        raise V2PlanParseError("answer needs every META CONTINUE field, non-empty")
    if values["DECISION"] not in {item.value for item in ContinueDecision}:
        raise V2PlanParseError("DECISION must be exactly COMPLETE, NEXT or SPEC_DECISION")
    decision = ContinueDecision(values["DECISION"])
    milestone = values["NEXT_MILESTONE"]
    if milestone not in {"", "NONE"} and MILESTONE_ID_RE.fullmatch(milestone) is None:
        raise V2PlanParseError("NEXT_MILESTONE must be NONE or a milestone ID like M02")
    answer = values["SPEC_QUESTION"]
    question = None if not answer or answer.casefold() in _NONE_WORDS else answer
    if decision is not ContinueDecision.NEXT and inner_raw is not None:
        raise V2PlanParseError(f"{decision.value} must not contain a NEXT PLAN")
    plan = None if inner_raw is None else parse_task_plan_v2(
        inner_raw, planning=planning, check_catalog=check_catalog,
        default_check_ids=default_check_ids)
    if decision is ContinueDecision.NEXT:
        if milestone in {"", "NONE"}:
            raise V2PlanParseError("NEXT requires NEXT_MILESTONE")
        if plan is None:
            raise V2PlanParseError("NEXT requires one complete NEXT PLAN block")
        if plan.decision is not PlanDecision.READY:
            raise V2PlanParseError("NEXT PLAN must be READY, never BLOCKED")
        if plan.milestone_id != milestone:
            raise V2PlanParseError("NEXT_MILESTONE and the NEXT PLAN milestone differ")
        if question is not None:
            raise V2PlanParseError("NEXT must not carry SPEC_QUESTION")
        validate_execution_mode_policy(plan, planning)
        next_milestone = milestone
    else:
        if milestone not in {"", "NONE"}:
            raise V2PlanParseError(f"{decision.value} must set NEXT_MILESTONE to NONE")
        next_milestone = None
        if decision is ContinueDecision.SPEC_DECISION:
            # A question mark is recommended, never contractually required.
            if question is None or len(question) > MAX_SPEC_QUESTION_CHARS:
                raise V2PlanParseError("SPEC_DECISION requires one concrete SPEC_QUESTION")
        elif question is not None:
            raise V2PlanParseError("COMPLETE must set SPEC_QUESTION to NONE")
    return PlannerContinueResult(
        decision=decision, summary=nonempty(values["SUMMARY"], "SUMMARY"),
        remaining=_remaining(values["REMAINING"]), next_milestone=next_milestone,
        spec_question=question, next_plan=plan)


def stagnation_fingerprint(
    remaining: Sequence[str], candidate_tree: str | None, gate_failures: Sequence[str],
) -> str:
    """The canonical identity of one iteration; C8b owns budget and comparison.

    REMAINING and failure IDs are whitespace-collapsed, case-folded, sorted and
    deduplicated; the tree SHA is taken as given.  Nothing here touches a run.
    """

    if isinstance(remaining, (str, bytes)) or isinstance(gate_failures, (str, bytes)):
        raise TypeError("remaining and gate_failures must be sequences of strings")
    if candidate_tree is not None and not isinstance(candidate_tree, str):
        raise TypeError("candidate_tree must be a string")

    def canonical(items: Sequence[str]) -> list[str]:
        if any(not isinstance(item, str) for item in items):
            raise TypeError("fingerprint items must be strings")
        return sorted({" ".join(item.split()).casefold() for item in items} - {""})

    payload = {"remaining": canonical(remaining), "tree": (candidate_tree or "").strip(),
               "failures": canonical(gate_failures)}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class PlannerContinue:
    """One continuation decision, on the planner profile of the run."""

    client: TextCompletionClient
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    check_catalog: tuple[Any, ...] = ()
    default_check_ids: tuple[str, ...] = ()
    prompt_budget_bytes: int = 0
    last_usage: dict[str, Any] | None = None

    def decide(
        self, facts: PlannerContinueFacts, *, iterations_dir: str | Path | None = None,
        validate: Callable[[PlannerContinueResult], PlannerContinueResult] | None = None,
    ) -> PlannerContinueResult:
        """Ask and persist a corrected response when validation rejects one."""

        payload = build_planner_continue_payload(
            facts, planning=self.planning, check_catalog=self.check_catalog,
            default_check_ids=self.default_check_ids, budget_bytes=self.prompt_budget_bytes)
        target = None if iterations_dir is None else planner_continue_dir(
            iterations_dir, facts.iteration)
        if target is not None:
            write_planner_continue_request(target, planner_continue_request_record(facts, payload))
        raw_path = target / "raw.txt" if target is not None else None
        answer = None
        if raw_path is not None and raw_path.is_file():
            # A paid continuation is a durable parse boundary. A crash after
            # raw.txt must never buy the same decision a second time.
            answer = raw_path.read_text(encoding="utf-8")
        last_error = ""
        previous_answer = ""
        for attempt in range(self.planning.max_preapproval_corrections + 1):
            retry_dir = None if target is None or attempt == 0 else (
                target / "corrections" / f"{attempt:02d}"
            )
            if retry_dir is not None:
                retry_dir.mkdir(parents=True, exist_ok=True)
            attempt_raw_path = (
                retry_dir / "raw.txt" if retry_dir is not None else raw_path
            )
            if answer is None:
                if attempt_raw_path is not None and attempt_raw_path.is_file():
                    answer = attempt_raw_path.read_text(encoding="utf-8")
                else:
                    request = payload.rendered if attempt == 0 else _correction_prompt(
                        payload.rendered, last_error, previous_answer,
                    )
                    if retry_dir is not None:
                        atomic_write_text(retry_dir / "request.txt", request)
                    raw = self.client.complete(request)
                    self.last_usage = completion_usage(raw)
                    answer = raw if isinstance(raw, str) else getattr(raw, "text", None)
                    if not isinstance(answer, str):
                        raise LLMProtocolError("planner continue client did not return text")
                    if attempt == 0 and target is not None:
                        write_planner_continue_raw(target, answer)
                    elif attempt_raw_path is not None:
                        atomic_write_text(attempt_raw_path, answer)
            previous_answer = answer
            try:
                result = parse_planner_continue(
                    answer, planning=self.planning, check_catalog=self.check_catalog,
                    default_check_ids=self.default_check_ids)
                if validate is not None:
                    result = validate(result)
                if target is not None:
                    record = planner_continue_result_record(result)
                    record["accepted_attempt"] = attempt
                    write_planner_continue_result(target, record)
                return result
            except V2PlanParseError as exc:
                last_error = str(exc)
                rejection_dir = retry_dir if retry_dir is not None else target
                if rejection_dir is not None:
                    atomic_write_text(rejection_dir / "rejection.json", json.dumps(
                        {"reason": last_error}, ensure_ascii=False, indent=2,
                    ) + "\n")
            answer = None
        raise V2PlanParseError(f"planner continue remained invalid after correction: {last_error}")


def _correction_prompt(initial: str, reason: str, rejected: str) -> str:
    """Keep the full request authority while bounding rejected model output."""

    bounded = rejected[:24_000]
    if len(rejected) > len(bounded):
        bounded += "\n[planner response truncated for correction]"
    return (
        initial + "\n\nCONTINUATION VALIDATION FAILURE\n" + reason
        + "\n\nREJECTED CONTINUATION\n" + bounded
        + "\n\nReturn a corrected complete META CONTINUE v1 response."
    )


__all__ = [
    "CONTINUE_END", "CONTINUE_HEADER", "CONTINUE_PLAN_BEGIN", "CONTINUE_PLAN_END",
    "MAX_SPEC_QUESTION_CHARS",
    "ContinueDecision", "PlannerContinue", "PlannerContinueFacts", "PlannerContinueResult",
    "build_planner_continue_payload", "parse_planner_continue", "stagnation_fingerprint",
]
