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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..llm.chat import LLMConversationHandle, LLMProtocolError
from ..models import (
    CheckConfig,
    ExecutionClass,
    PlanDecision,
    PlanningConfig,
    TaskPlanV2,
)
from ..plan_repository_validation import (
    PathPreconditionViolation,
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
    plan_repository_violations,
)
from ..result import atomic_write_text
from ..step_ids import MAX_STEPS, STEP_ID_RE, step_ids
from .protocol import (
    MAX_STEP_CONTRACT_CHARS,
    STEP_CONTRACT_NAME,
    STEP_ID_RANGE,
    V2PlanParseError,
    parse_task_plan_v2,
    render_plan_summary_v2,
    render_step_contract,
    validate_step_contract_bounds,
)
from .normalization import normalizations_payload
from .validation import (
    normalize_plan_repository,
    validate_repair_decomposition_policy,
)

STEP_CONTRACT_REPAIR_OUTPUT_INVALID = "STEP_CONTRACT_REPAIR_OUTPUT_INVALID"
# The compact record of every deterministic normalization applied to the plan.
PLAN_NORMALIZATIONS_NAME = "plan.normalizations.json"
# Paid answers of one semantic repair slot beyond the first:
# ``contract_repairs/NN/output_attempts/NNN``.
STEP_REPAIR_OUTPUT_ATTEMPTS_DIR = "output_attempts"


class StepContractRepairArtifactError(Exception):
    """Durable repair answers, hashes or metadata are inconsistent."""

    code = "RESUME_INTEGRITY_FAILURE"


def read_bounded_json(path: Path, limit: int) -> Any:
    try:
        if path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


@dataclass(frozen=True)
class StepRepairAttemptFiles:
    """Durable files of one paid StepContractRepairPlanner answer.

    Output attempt 001 keeps its historical slot-level request/raw/usage
    files; every later attempt owns ``output_attempts/NNN``.  Each attempt's
    ``parse_error.json`` and ``response.meta.json`` live in its own directory.
    """

    number: int
    directory: Path
    request: Path
    meta: Path
    raw: Path
    usage: Path

    @property
    def parse_error(self) -> Path:
        return self.directory / "parse_error.json"

    @property
    def response_meta(self) -> Path:
        return self.directory / "response.meta.json"


def step_repair_attempt_files(slot: str | Path, number: int) -> StepRepairAttemptFiles:
    slot = Path(slot)
    directory = slot / STEP_REPAIR_OUTPUT_ATTEMPTS_DIR / f"{number:03d}"
    if number == 1:
        return StepRepairAttemptFiles(
            1, directory, slot / "planner.request.txt", slot / "request.meta.json",
            slot / "planner.raw.md", slot / "usage.json",
        )
    return StepRepairAttemptFiles(
        number, directory, directory / "planner.request.txt", directory / "request.meta.json",
        directory / "planner.raw.md", directory / "usage.json",
    )


def step_repair_attempt_state(
    files: StepRepairAttemptFiles, *, current_tree_sha: str | None = None,
) -> tuple[str, str | None]:
    """``(state, raw)`` of one output attempt; inconsistent evidence raises.

    ``none``: no durable request; ``pending``: request durable, no answer;
    ``raw``: a paid answer is durable and not yet classified; ``invalid``: the
    answer was deterministically rejected (``parse_error.json``).
    """

    meta = read_bounded_json(files.meta, 64 * 1024)
    if meta is None:
        if files.meta.exists():
            raise StepContractRepairArtifactError(
                f"contract repair output attempt {files.number:03d} metadata is unreadable"
            )
        return "none", None
    status = meta.get("status") if isinstance(meta, dict) else None
    if status not in {"pending", "raw", "validated"}:
        raise StepContractRepairArtifactError(
            f"contract repair output attempt {files.number:03d} metadata is malformed"
        )
    try:
        request_sha = sha256_bytes(files.request.read_bytes())
    except OSError as exc:
        raise StepContractRepairArtifactError(
            f"contract repair output attempt {files.number:03d} request is missing"
        ) from exc
    if meta.get("request_sha256") != request_sha or (
        current_tree_sha is not None and meta.get("current_tree_sha") != current_tree_sha
    ):
        raise StepContractRepairArtifactError(
            f"contract repair output attempt {files.number:03d} request identity changed"
        )
    if status == "pending":
        return "pending", None
    try:
        raw_bytes = files.raw.read_bytes()
    except OSError as exc:
        raise StepContractRepairArtifactError(
            f"contract repair output attempt {files.number:03d} raw answer is missing"
        ) from exc
    raw_sha = sha256_bytes(raw_bytes)
    if raw_sha != meta.get("raw_sha256"):
        raise StepContractRepairArtifactError(
            f"contract repair output attempt {files.number:03d} raw answer hash changed"
        )
    if files.parse_error.exists():
        error = read_bounded_json(files.parse_error, 64 * 1024)
        if (
            not isinstance(error, dict)
            or error.get("raw_sha256") != raw_sha
            or error.get("request_sha256") != request_sha
            or error.get("code") != STEP_CONTRACT_REPAIR_OUTPUT_INVALID
        ):
            raise StepContractRepairArtifactError(
                f"contract repair output attempt {files.number:03d} parse error does not match its answer"
            )
        return "invalid", raw_bytes.decode("utf-8")
    return "raw", raw_bytes.decode("utf-8")


def write_implementation_bundle(directory: str | Path, plan: TaskPlanV2) -> dict[str, Any]:
    """Write the human summary, bounded step contracts and secret-free index."""

    if not isinstance(plan, TaskPlanV2) or plan.decision is not PlanDecision.READY:
        raise V2PlanParseError("implementation bundle requires a READY v2 plan")
    validate_step_contract_bounds(plan)
    target = Path(directory)
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
    atomic_write_text(target / "implementation_contract.md", render_plan_summary_v2(plan))
    for step in plan.steps:
        # The only copy of each contract: approval hashes and runtime reads
        # these exact bytes; nothing re-renders them after this point.
        atomic_write_text(step_contract_path(target, step.id), contracts[step.id])
    atomic_write_text(target / "implementation_bundle.json", json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    # Exactly one normalization record per plan, written with the plan whose
    # steps it made effective; the audit reads it without re-deriving anything.
    atomic_write_text(
        target / PLAN_NORMALIZATIONS_NAME,
        render_json(normalizations_payload(plan)),
    )
    # ``task_plan.json`` is the stable v2 artifact name approved by the human.
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
    if "reviewer_profile" in payload or "implementer_profiles" in payload:
        raise V2PlanParseError("implementation bundle contains planner-selected profiles")
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
) -> None:
    """Persist the v2 exchange and publish its implementation bundle."""

    target = Path(directory)
    atomic_write_text(target / "spec.md", spec)
    atomic_write_text(target / "context.txt", context)
    atomic_write_text(target / "planner.request.txt", request)
    atomic_write_text(target / "planner.raw.md", plan.raw)
    write_task_plan_v2(target, plan)
    # The unsuffixed artifact is the v2 approval surface.
    if plan.decision is PlanDecision.BLOCKED:
        atomic_write_text(
            target / "task_plan.json",
            json.dumps({**asdict(plan), "decision": plan.decision.value, "execution_mode": None}, ensure_ascii=False, indent=2) + "\n",
        )
    if plan.decision is PlanDecision.READY:
        write_implementation_bundle(target, plan)




def write_task_plan_v2(target: Path, plan: TaskPlanV2) -> None:
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


def persist_recovered_plan_artifacts(directory: str | Path, plan: TaskPlanV2) -> dict[str, Any]:
    """Publish an operator-supplied READY plan as the run's plan authority.

    Unlike :func:`persist_planning_v2_artifacts` this never touches
    ``spec.md``, ``context.txt`` or ``planner.request.txt``: no planner
    request exists for an operator recovery, and the run inputs are immutable.
    """

    if not isinstance(plan, TaskPlanV2) or plan.decision is not PlanDecision.READY:
        raise V2PlanParseError("plan recovery requires a READY v2 plan")
    target = Path(directory)
    atomic_write_text(target / "planner.raw.md", plan.raw)
    write_task_plan_v2(target, plan)
    return write_implementation_bundle(target, plan)


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


def _repair_plan_recovery_sources(target: Path) -> list[Path]:
    """The correction directories that may hold an already paid planner answer.

    ``target`` first, then its archived retry attempts newest-first, so a raw
    response that was already paid for and rejected only by local validation
    is found wherever it was kept.
    """

    sources = [target]
    attempts = target / "attempts"
    if attempts.is_dir():
        sources.extend(
            sorted((path for path in attempts.iterdir() if path.is_dir()), reverse=True)
        )
    return sources


def recover_existing_repair_plan(
    *,
    target: Path,
    current_evidence_text: str,
    original_spec: str,
    current_repository_state: str,
    check_catalog: Sequence[CheckConfig],
    inherited_check_ids: Sequence[str],
    planning: PlanningConfig,
    repository_preconditions: RepositoryPreconditions | None = None,
) -> TaskPlanV2 | None:
    """Revalidate an already produced correction answer locally, or return ``None``.

    A durable ``planner.raw.md`` is reusable only next to a
    ``planner.evidence.md`` byte-identical to *current_evidence_text*: the
    answer then belongs to exactly this candidate commit, reviewer result
    and approved mutable scope.  The strict parser and the repair policy are
    applied unchanged, no existing artifact is deleted, and the model is never
    called.  A structurally invalid or still out-of-policy answer is refused.
    """

    # Accepted for symmetry with the persistence step; a recovery decision
    # depends only on the evidence packet and on the raw answer itself.
    del original_spec, current_repository_state

    for source in _repair_plan_recovery_sources(target):
        try:
            raw = (source / "planner.raw.md").read_text(encoding="utf-8")
            evidence = (source / "planner.evidence.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if evidence != current_evidence_text:
            continue
        try:
            plan = parse_task_plan_v2(
                raw,
                planning=planning,
                check_catalog=check_catalog,
                inherited_check_ids=inherited_check_ids,
            )
            validate_repair_decomposition_policy(plan, planning)
        except V2PlanParseError:
            continue
        plan = normalize_plan_repository(repository_preconditions, plan)
        if plan_repository_violations(plan):
            continue
        if source is not target:
            # The retry archived the provenance of the answer being reused, so
            # restore it where the run expects it -- never over a present copy.
            for name, text in (
                ("planner.raw.md", raw),
                ("planner.evidence.md", evidence),
            ):
                if not (target / name).exists():
                    atomic_write_text(target / name, text)
        return plan
    return None


def persist_recovered_repair_artifacts(
    target: Path,
    *,
    original_spec: str,
    current_repository_state: str,
    plan: TaskPlanV2,
) -> None:
    """Publish a locally revalidated repair plan without rewriting its call.

    ``planner.request.txt``, ``planner.request.fallback.txt``,
    ``planner.evidence.md``, ``planner.request.meta.json``, ``planner.raw.md``
    and ``planner.usage.json`` describe the one exchange that really produced
    this answer, so they stay exactly as they are; the already durable raw
    response remains the authority of provenance.
    """

    atomic_write_text(target / "spec.md", original_spec)
    atomic_write_text(target / "context.txt", current_repository_state)
    write_task_plan_v2(target, plan)
    if plan.decision is PlanDecision.READY:
        write_implementation_bundle(target, plan)
    else:
        atomic_write_text(
            target / "task_plan.json",
            json.dumps(
                {**asdict(plan), "decision": plan.decision.value, "execution_mode": None},
                ensure_ascii=False, indent=2,
            ) + "\n",
        )


__all__ = [
    "PLAN_NORMALIZATIONS_NAME",
    "STEP_CONTRACT_REPAIR_OUTPUT_INVALID",
    "STEP_REPAIR_OUTPUT_ATTEMPTS_DIR",
    "StepContractRepairArtifactError",
    "StepRepairAttemptFiles",
    "persist_planning_v2_artifacts",
    "persist_recovered_plan_artifacts",
    "persist_recovered_repair_artifacts",
    "planning_session_handle",
    "read_approved_step_contract",
    "read_attempt_validation",
    "read_bounded_json",
    "read_planning_session",
    "recover_existing_repair_plan",
    "render_json",
    "sha256_bytes",
    "step_contract_path",
    "step_repair_attempt_files",
    "step_repair_attempt_state",
    "validate_implementation_bundle",
    "validation_failure",
    "write_implementation_bundle",
    "write_planning_session",
    "write_task_plan_v2",
]
