"""The PLANNER_CONTINUE request: minimal facts, payload and durable files.

It owns the fact dataclass, the compact payload rendered from those facts and
``iterations/NN/planner-continue/{request.json,raw.txt,result.json}``; the protocol
that interprets an answer lives in :mod:`metaharness.planning.planner_continue`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ..models import PlanningConfig, TaskPlanV2
from ..prompt_contracts import PromptPayload, PromptSection, build_prompt_payload
from ..result import atomic_write_text
from .artifacts import read_bounded_json, render_json, sha256_bytes
from .protocol import (
    MILESTONE_ID_RE,
    render_plan_summary_v2,
    render_safe_check_catalogue,
)

if TYPE_CHECKING:  # typing only: the protocol module imports this one at runtime
    from .planner_continue import PlannerContinueResult

PLANNER_CONTINUE_DIR, PLANNER_CONTINUE_REQUEST = "planner-continue", "request.json"
PLANNER_CONTINUE_RAW, PLANNER_CONTINUE_RESULT = "raw.txt", "result.json"
MAX_CONTINUE_ARTIFACT_BYTES = 4_000_000
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SECTIONS = ("spec", "state", "plan", "audit", "evidence", "repository", "rules")
_TEXT_TUPLES = ("normalizations", "failed_steps", "audit_remaining", "audit_risks", "audit_fixed",
                "audit_refactored", "gate_failures", "gate_warnings", "gate_baseline_warnings",
                "modified_paths", "continuation_remaining", "prior_iteration_remaining")


@dataclass(frozen=True)
class PlannerContinueFacts:
    """The minimal facts of one continuation: no logs, no worker prompts."""

    spec: str
    plan: TaskPlanV2
    iteration: int
    milestone_id: str
    milestone_title: str
    milestone_goal: str
    audit_status: str
    milestones: tuple[tuple[str, str], ...] = ()
    normalizations: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    audit_remaining: tuple[str, ...] = ()
    audit_risks: tuple[str, ...] = ()
    audit_fixed: tuple[str, ...] = ()
    audit_refactored: tuple[str, ...] = ()
    gate_failures: tuple[str, ...] = ()
    gate_warnings: tuple[str, ...] = ()
    gate_baseline_warnings: tuple[str, ...] = ()
    diffstat: str = ""
    modified_paths: tuple[str, ...] = ()
    current_repository_context: str = ""
    continuation_remaining: tuple[str, ...] = ()
    prior_iteration_remaining: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(getattr(self, name), str) for name in (
                "spec", "milestone_title", "milestone_goal", "audit_status", "diffstat",
                "current_repository_context")):
            raise ValueError("every text fact must be a string")
        if not self.spec.strip() or not isinstance(self.plan, TaskPlanV2):
            raise ValueError("spec and plan must be the run's own values")
        if MILESTONE_ID_RE.fullmatch(self.milestone_id or "") is None:
            raise ValueError("milestone_id must look like M01")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration < 1:
            raise ValueError("iteration must be a positive integer")
        if not self.audit_status.strip():
            raise ValueError("audit_status must be the status of the last audit")
        if any(not isinstance(getattr(self, name), tuple) or any(
                not isinstance(item, str) and not (
                    isinstance(item, tuple) and len(item) == 2
                    and all(isinstance(part, str) for part in item))
                for item in getattr(self, name))
               for name in (*_TEXT_TUPLES, "milestones")):
            raise ValueError("evidence fields must be strings or (id, status) pairs")


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def _labeled(*pairs: tuple[str, str]) -> str:
    return "\n\n".join(f"{name}\n{text}" for name, text in pairs)


def build_planner_continue_payload(
    facts: PlannerContinueFacts, *, planning: PlanningConfig | None = None,
    check_catalog: Sequence[Any] = (), default_check_ids: Sequence[str] = (),
    template: str | None = None, budget_bytes: int = 0,
) -> PromptPayload:
    """Render the compact continuation request from *facts* only."""

    if not isinstance(facts, PlannerContinueFacts):
        raise TypeError("facts must be a PlannerContinueFacts")
    planning = planning or PlanningConfig()
    if template is None:
        template = (_PROMPTS_DIR / "planner_continue.txt").read_text(encoding="utf-8")
    # Harness-owned protocol numbers, never user text.
    constants = {
        "DEFAULT_CHECK_IDS": "\n".join(f"- {id}" for id in default_check_ids) or "NONE",
        "MAX_STEPS": str(planning.max_steps_per_plan),
        "LAST_STEP_ID": f"S{planning.max_steps_per_plan:02d}",
        "MAX_STEP_CONTRACT_CHARS": str(planning.max_step_contract_chars),
        "MAX_READ_PATHS_PER_STEP": str(planning.max_read_paths_per_step),
        "SINGLE_STEP_MAX_PATHS": str(planning.single_step_max_mutable_paths),
        "STAGED_STEP_MAX_PATHS": str(planning.staged_step_max_mutable_paths),
    }
    template = re.sub(
        r"\{\{([A-Z0-9_]+)\}\}",
        lambda match: constants.get(match.group(1), match.group(0)), template)
    texts = {
        "spec": facts.spec,
        "state": _labeled(
            ("MILESTONE", f"{facts.milestone_id} :: {facts.milestone_title or 'NONE'}"),
            ("GOAL", facts.milestone_goal or "NONE"),
            ("ITERATION", str(facts.iteration)),
            ("EXECUTION_MODE_POLICY", planning.execution_mode_policy),
            ("CLOSED MILESTONES", _bullets(
                f"{item} :: {status}" for item, status in facts.milestones))),
        "plan": render_plan_summary_v2(facts.plan) + "\n\n" + _labeled(
            ("NORMALIZATIONS", _bullets(facts.normalizations)),
            ("STEPS FAILED BUT CONTINUED", _bullets(facts.failed_steps))),
        "audit": _labeled(
            ("STATUS", facts.audit_status),
            ("REMAINING", _bullets(facts.audit_remaining)),
            ("CONTINUATION REMAINING", _bullets(facts.continuation_remaining)),
            ("PRIOR ITERATION REMAINING", _bullets(facts.prior_iteration_remaining)),
            ("RISKS", _bullets(facts.audit_risks)),
            ("FIXED", _bullets(facts.audit_fixed)),
            ("REFACTORED", _bullets(facts.audit_refactored))),
        "evidence": _labeled(
            ("FAILURES", _bullets(facts.gate_failures)),
            ("WARNINGS", _bullets(facts.gate_warnings)),
            ("BASELINE WARNINGS", _bullets(facts.gate_baseline_warnings)),
            ("DIFFSTAT SINCE BASE", facts.diffstat.strip() or "NONE"),
            ("MODIFIED PATHS", _bullets(facts.modified_paths))),
        "repository": facts.current_repository_context.strip() or "NONE",
        "rules": render_safe_check_catalogue(check_catalog),
    }
    return build_prompt_payload(
        role="planner-continue", template=template,
        sections=tuple(PromptSection.create(name, texts[name], True) for name in _SECTIONS),
        placeholders={f"{{{{{name.upper()}}}}}": name for name in _SECTIONS},
        budget_bytes=budget_bytes)


def planner_continue_dir(iterations_dir: str | Path, iteration: int) -> Path:
    """``iterations/NN/planner-continue``: the files of one continuation."""

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    return Path(iterations_dir) / f"{iteration:02d}" / PLANNER_CONTINUE_DIR


def write_planner_continue_request(directory: str | Path, request: dict[str, Any]) -> Path:
    return _write(directory, PLANNER_CONTINUE_REQUEST, render_json(request))


def write_planner_continue_raw(directory: str | Path, raw: str) -> Path:
    return _write(directory, PLANNER_CONTINUE_RAW, raw)


def write_planner_continue_result(directory: str | Path, result: dict[str, Any]) -> Path:
    return _write(directory, PLANNER_CONTINUE_RESULT, render_json(result))


def _write(directory: str | Path, name: str, text: str) -> Path:
    target = Path(directory)
    atomic_write_text(target / name, text)
    return target


def read_planner_continue_artifacts(
    directory: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Read one decision's ``(request, raw, result)`` triple back."""

    target = Path(directory)
    request = read_bounded_json(target / PLANNER_CONTINUE_REQUEST, MAX_CONTINUE_ARTIFACT_BYTES)
    result = read_bounded_json(target / PLANNER_CONTINUE_RESULT, MAX_CONTINUE_ARTIFACT_BYTES)
    if request is None or result is None:
        raise FileNotFoundError(f"planner continue artifacts are missing in {target}")
    if not isinstance(request, dict) or not isinstance(result, dict):
        raise TypeError("planner continue artifacts must be JSON objects")
    accepted_attempt = result.get("accepted_attempt", 0)
    if isinstance(accepted_attempt, bool) or not isinstance(accepted_attempt, int) or accepted_attempt < 0:
        raise TypeError("planner continue result has an invalid accepted_attempt")
    raw_path = (
        target / "raw.txt" if accepted_attempt == 0
        else target / "corrections" / f"{accepted_attempt:02d}" / "raw.txt"
    )
    return request, raw_path.read_text(encoding="utf-8"), result


def planner_continue_request_record(
    facts: PlannerContinueFacts, payload: PromptPayload,
    *, check_authority_sha256: str | None = None,
) -> dict[str, Any]:
    """The secret-free record of one request: section texts and pinned hashes."""

    return {
        "iteration": facts.iteration, "milestone_id": facts.milestone_id,
        "check_authority_sha256": check_authority_sha256,
        "spec_sha256": sha256_bytes(facts.spec.encode("utf-8")),
        "current_plan": {"milestone_id": facts.plan.milestone_id,
                         "steps": [step.id for step in facts.plan.steps],
                         "raw_sha256": sha256_bytes(facts.plan.raw.encode("utf-8"))},
        "facts": {section.name: section.text for section in payload.sections},
        "prompt": {"sha256": sha256_bytes(payload.rendered.encode("utf-8")),
                   "bytes": payload.total_bytes, "budget_bytes": payload.budget_bytes},
    }


def planner_continue_result_record(result: PlannerContinueResult) -> dict[str, Any]:
    """The durable form of one decision, including the NEXT PLAN bytes."""

    return {
        "decision": result.decision.value, "summary": result.summary,
        "remaining": list(result.remaining), "next_milestone": result.next_milestone,
        "spec_question": result.spec_question,
        "next_plan": result.next_plan.raw if result.next_plan is not None else None,
    }


__all__ = [
    "MAX_CONTINUE_ARTIFACT_BYTES", "PLANNER_CONTINUE_DIR", "PLANNER_CONTINUE_RAW",
    "PLANNER_CONTINUE_REQUEST", "PLANNER_CONTINUE_RESULT", "PlannerContinueFacts",
    "build_planner_continue_payload", "planner_continue_dir", "planner_continue_request_record",
    "planner_continue_result_record", "read_planner_continue_artifacts",
    "write_planner_continue_raw", "write_planner_continue_request", "write_planner_continue_result",
]
