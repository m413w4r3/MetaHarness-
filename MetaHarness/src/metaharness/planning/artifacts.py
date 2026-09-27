"""Durable planning artifacts: bundle persistence, hashes and identities.

Reads and writes the planning artifacts and derives their deterministic
hashes and identities.  It makes no planning decision and calls no model;
recovery revalidates a durable answer through the protocol and the policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from ..llm.chat import LLMConversationHandle, LLMProtocolError
from ..models import (
    BlockerKind,
    ContractNormalization,
    ExecutionClass,
    ExecutionMode,
    ImplementationStep,
    PlanDecision,
    TaskPlanV2,
)
from ..plan_repository_validation import (
    PathPreconditionViolation,
    PlanRepositoryPreconditionError,
)
from ..result import atomic_write_text
from ..step_ids import MAX_STEPS, STEP_ID_RE, step_ids
from .protocol import (
    MAX_STEP_CONTRACT_CHARS,
    STEP_CONTRACT_NAME,
    STEP_ID_RANGE,
    V2PlanParseError,
    render_plan_summary_v2,
    render_step_contract,
    validate_step_contract_bounds,
)
from .normalization import normalizations_payload

# The compact record of every deterministic normalization applied to the plan.
PLAN_NORMALIZATIONS_NAME = "plan.normalizations.json"


def read_bounded_json(path: Path, limit: int) -> Any:
    try:
        if path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iteration_dir(run_dir: str | Path, iteration: int) -> Path:
    """Return the durable directory for one positive iteration number."""

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    return Path(run_dir) / "iterations" / f"{iteration:02d}"


def iteration_plan_dir(run_dir: str | Path, iteration: int) -> Path:
    """Return ``iterations/NN/plan`` for one effective plan authority."""

    return iteration_dir(run_dir, iteration) / "plan"


def render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_implementation_bundle(directory: str | Path, plan: TaskPlanV2) -> dict[str, Any]:
    """Write the human summary, bounded step contracts and secret-free index."""

    bundle, contracts = _implementation_bundle(plan)
    target = Path(directory)
    atomic_write_text(target / "implementation_contract.md", render_plan_summary_v2(plan))
    for step in plan.steps:
        atomic_write_text(step_contract_path(target, step.id), contracts[step.id])
    atomic_write_text(target / "implementation_bundle.json", json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(target / PLAN_NORMALIZATIONS_NAME, render_json(normalizations_payload(plan)))
    atomic_write_text(target / "task_plan.json", render_json(effective_plan_payload(plan)))
    return bundle


def implementation_bundle_payload(plan: TaskPlanV2) -> dict[str, Any]:
    """Derive the disposable worker bundle from the canonical effective plan."""

    return _implementation_bundle(plan)[0]


def _implementation_bundle(plan: TaskPlanV2) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(plan, TaskPlanV2) or plan.decision is not PlanDecision.READY:
        raise V2PlanParseError("implementation bundle requires a READY v2 plan")
    validate_step_contract_bounds(plan)
    contracts = {step.id: render_step_contract(plan, step) for step in plan.steps}
    entries = []
    for step in plan.steps:
        entries.append(
            {
                "id": step.id,
                "title": step.title,
                "execution_class": step.execution_class.value,
                "depends_on": step.depends_on,
                "contract_sha256": hashlib.sha256(contracts[step.id].encode("utf-8")).hexdigest(),
            }
        )
    bundle = {
        "schema_version": 1,
        "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
        "max_step_contract_chars": plan.max_step_contract_chars,
        "required_checks": list(plan.required_checks),
        "steps": entries,
    }
    return bundle, contracts


def effective_plan_payload(plan: TaskPlanV2) -> dict[str, Any]:
    """One normalized plan representation; the raw response stays observational."""

    payload = asdict(plan)
    payload.pop("raw", None)
    payload["decision"] = plan.decision.value
    payload["execution_mode"] = plan.execution_mode.value if plan.execution_mode else None
    payload["blocker_kind"] = plan.blocker_kind.value if plan.blocker_kind else None
    return payload


def persist_effective_plan(directory: str | Path, plan: TaskPlanV2) -> str:
    """Persist the effective normalized plan and return its exact byte hash."""

    target = Path(directory)
    if plan.decision is PlanDecision.READY:
        write_implementation_bundle(target, plan)
    else:
        atomic_write_text(target / "task_plan.json", render_json(effective_plan_payload(plan)))
    return hashlib.sha256((target / "task_plan.json").read_bytes()).hexdigest()


def persist_iteration_plan(run_dir: str | Path, iteration: int, plan: TaskPlanV2) -> str:
    """Persist an iteration's effective plan and return its exact byte hash."""

    return persist_effective_plan(iteration_plan_dir(run_dir, iteration), plan)


def read_effective_plan(directory: str | Path, expected_sha256: str) -> TaskPlanV2:
    """Load the canonical plan only when its durable bytes match the checkpoint."""

    path = Path(directory) / "task_plan.json"
    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2PlanParseError("effective plan is missing or invalid") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha256 or not isinstance(payload, dict):
        raise V2PlanParseError("effective plan hash does not match the checkpoint")
    try:
        steps = tuple(ImplementationStep(
            **{
                **item,
                "execution_class": ExecutionClass(item["execution_class"]),
                **{name: tuple(item[name]) for name in (
                    "read_set", "write_set", "create_set", "delete_set",
                )},
            }
        ) for item in payload["steps"])
        normalizations = tuple(ContractNormalization(**item) for item in payload["normalizations"])
        return TaskPlanV2(
            decision=PlanDecision(payload["decision"]), title=payload["title"],
            objective=payload["objective"], constraints=payload["constraints"],
            execution_mode=ExecutionMode(payload["execution_mode"]) if payload["execution_mode"] else None,
            steps=steps, acceptance=payload["acceptance"], tests=payload["tests"],
            risks=payload["risks"], blockers=payload["blockers"], raw="",
            required_checks=tuple(payload["required_checks"]),
            max_step_contract_chars=payload["max_step_contract_chars"],
            blocker_kind=BlockerKind(payload["blocker_kind"]) if payload["blocker_kind"] else None,
            milestone_id=payload["milestone_id"], milestone_title=payload["milestone_title"],
            milestone_goal=payload["milestone_goal"], project_remainder=payload["project_remainder"],
            normalizations=normalizations,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V2PlanParseError("effective plan fields are invalid") from exc


def read_iteration_plan(run_dir: str | Path, iteration: int, expected_sha256: str) -> TaskPlanV2:
    """Read the effective plan named by an iteration checkpoint."""

    return read_effective_plan(iteration_plan_dir(run_dir, iteration), expected_sha256)


def step_contract_path(directory: str | Path, step_id: str) -> Path:
    """Canonical path of one step contract: ``steps/<STEP>/contract.md``."""

    if not isinstance(step_id, str) or STEP_ID_RE.fullmatch(step_id) is None:
        raise V2PlanParseError(f"step ID must be exactly {STEP_ID_RANGE}")
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
    limit = bundle.get("max_step_contract_chars", MAX_STEP_CONTRACT_CHARS)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise V2PlanParseError("implementation bundle contract limit is invalid")
    if len(text) > limit:
        raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
    return text


def validate_implementation_bundle(
    directory: str | Path,
    *,
    expected_step_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the immutable v2 bundle and every contract hash it declares.

    With *expected_step_ids*, the bundle must declare exactly those steps in
    that order.  Contracts are read from the canonical per-step layout
    ``steps/Sxx/contract.md`` only.
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
    expected_ids = list(step_ids(len(steps)))
    actual_ids: list[str] = []
    for entry in steps:
        if not isinstance(entry, dict) or set(entry) != {"id", "title", "execution_class", "depends_on", "contract_sha256"}:
            raise V2PlanParseError("implementation bundle step entry is invalid")
        step_id = entry.get("id")
        if not isinstance(step_id, str) or step_id in actual_ids:
            raise V2PlanParseError("implementation bundle step ID is invalid")
        actual_ids.append(step_id)
        if entry.get("execution_class") not in {item.value for item in ExecutionClass}:
            raise V2PlanParseError("implementation bundle execution class is invalid")
        declared = entry.get("contract_sha256")
        if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
            raise V2PlanParseError("implementation bundle contract hash is invalid")
        if STEP_ID_RE.fullmatch(step_id) is None:
            raise V2PlanParseError("implementation bundle step ID is invalid")
        contract_path = step_contract_path(target, step_id)
        try:
            contract_bytes = contract_path.read_bytes()
            limit = payload.get("max_step_contract_chars", MAX_STEP_CONTRACT_CHARS)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise V2PlanParseError("implementation bundle contract limit is invalid")
            if len(contract_bytes.decode("utf-8")) > limit:
                raise V2PlanParseError("step contract exceeds MAX_STEP_CONTRACT_CHARS")
            actual = hashlib.sha256(contract_bytes).hexdigest()
        except OSError as exc:
            raise V2PlanParseError(f"missing contract for {step_id}") from exc
        if actual != declared:
            raise V2PlanParseError(f"contract hash mismatch for {step_id}")
    if actual_ids != expected_ids:
        raise V2PlanParseError("implementation bundle step IDs are not contiguous")
    if expected_step_ids is not None and list(expected_step_ids) != actual_ids:
        raise V2PlanParseError("implementation bundle steps do not match the plan")
    return payload, hashlib.sha256(bundle_bytes).hexdigest()




def persist_planning_v2_artifacts(
    directory: str | Path,
    *,
    spec: str,
    context: str,
    request: str,
    plan: TaskPlanV2,
    iteration: int = 1,
) -> None:
    """Persist the v2 exchange and publish its implementation bundle."""

    target = Path(directory)
    atomic_write_text(target / "spec.md", spec)
    atomic_write_text(target / "context.txt", context)
    atomic_write_text(target / "planner.request.txt", request)
    atomic_write_text(target / "planner.raw.md", plan.raw)
    # Planner exchange artifacts remain at the run root; the executable
    # authority belongs to the first iteration from its first publication.
    plan_dir = iteration_plan_dir(target, iteration)
    if plan.decision is PlanDecision.BLOCKED:
        atomic_write_text(plan_dir / "task_plan.json", render_json(effective_plan_payload(plan)))
    if plan.decision is PlanDecision.READY:
        write_implementation_bundle(plan_dir, plan)


def read_planning_session(target: Path | None) -> dict[str, Any]:
    if target is None or not (target / "planner.session.json").exists():
        return {}
    try:
        value = json.loads((target / "planner.session.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("invalid schema")
        if (isinstance(value.get("latest_attempt"), bool)
                or not isinstance(value.get("latest_attempt"), int)
                or value["latest_attempt"] < 1
                or not isinstance(value.get("continuation_available"), bool)):
            raise ValueError("invalid session fields")
        if value["continuation_available"] != (planning_session_handle(value) is not None):
            raise ValueError("invalid continuation flag")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise LLMProtocolError("planner session artifact is invalid") from exc


def planning_session_handle(session: dict[str, Any]) -> LLMConversationHandle | None:
    provider = session.get("provider_id")
    identifier = session.get("conversation_id")
    if provider is None and identifier is None:
        return None
    try:
        return LLMConversationHandle(provider, identifier)
    except (TypeError, ValueError) as exc:
        raise LLMProtocolError("planner session handle is invalid") from exc


def write_planning_session(
    target: Path, attempt: int, handle: LLMConversationHandle | None,
    *, continuation_used: bool, fallback_fresh_request: bool,
    response_error: str | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "provider_id": handle.provider_id if handle else None,
        "conversation_id": handle.conversation_id if handle else None,
        "latest_attempt": attempt,
        "continuation_available": handle is not None,
        "continuation_used": continuation_used,
        "fallback_fresh_request": fallback_fresh_request,
    }
    if response_error is not None:
        value["response_error"] = response_error[:1000]
    path = target / "planner.session.json"
    atomic_write_text(path, json.dumps(value, ensure_ascii=False) + "\n")
    os.chmod(path, 0o600)
    return value


def validation_failure(
    exc: V2PlanParseError | PlanRepositoryPreconditionError,
    violations: Sequence[PathPreconditionViolation],
) -> dict[str, Any]:
    if violations:
        errors = [{"code": item.kind, "step_id": item.step_id, "path": item.path}
                  for item in violations]
    else:
        errors = [{"code": "plan_format_invalid", "detail": str(exc)[:1000]}]
    return {"valid": False, "errors": errors}


def read_attempt_validation(attempt: Path) -> dict[str, Any]:
    try:
        value = json.loads((attempt / "planner.validation.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise LLMProtocolError("planner validation artifact is invalid") from exc
    if not isinstance(value, dict) or value.get("valid") is not False or not isinstance(value.get("errors"), list):
        raise LLMProtocolError("planner validation artifact is invalid")
    return value


__all__ = [
    "PLAN_NORMALIZATIONS_NAME",
    "effective_plan_payload",
    "implementation_bundle_payload",
    "iteration_dir",
    "iteration_plan_dir",
    "persist_effective_plan",
    "persist_iteration_plan",
    "read_iteration_plan",
    "read_effective_plan",
    "persist_planning_v2_artifacts",
    "planning_session_handle",
    "read_approved_step_contract",
    "read_attempt_validation",
    "read_bounded_json",
    "read_planning_session",
    "render_json",
    "sha256_bytes",
    "step_contract_path",
    "validate_implementation_bundle",
    "validation_failure",
    "write_implementation_bundle",
    "write_planning_session",
]
