"""Durable transaction markers of one semantic step contract repair.

A clean ``AGENT_CONTRACT_MISMATCH`` opens exactly one numbered repair slot
(``contract_repairs/NN``).  The slot's ``transaction.json`` and archived
``mismatch.json`` are written before any planner call, so a resume continues
that repair instead of replaying the worker that already produced the
mismatch.  The markers are transaction state, never semantic authority: the
repaired contract is still produced and validated by the planner transaction.

Statuses only move forward.  A transport failure (HTTP 429/5xx, timeout,
connection loss) parks the slot in ``waiting_external``; a resume re-enters
``awaiting_planner`` for the same semantic repair number and only counts a
new planner transport attempt.

A durable planner answer that is deterministically invalid moves the slot to
``planner_output_invalid``; the next admitted output correction opens
``awaiting_output_correction`` for output attempt ``n + 1`` of the *same* slot.
Ordering is ``(output_attempt, rank)``, so the correction loop is monotone.
Three counters stay independent: the semantic repair number (the slot), the
planner transport attempts of the current output attempt, and the output
corrections.  An exhausted correction budget parks the slot in
``output_correction_exhausted`` until an operator retries the planner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..planning.artifacts import (
    StepContractRepairArtifactError,
    step_repair_attempt_files,
    step_repair_attempt_state,
)
from ..result import atomic_write_text

TRANSACTION_NAME = "transaction.json"
MISMATCH_NAME = "mismatch.json"
SCHEMA_VERSION = 1

AWAITING_PLANNER = "awaiting_planner"
WAITING_EXTERNAL = "waiting_external"
PLANNER_RESPONSE_DURABLE = "planner_response_durable"
PLANNER_OUTPUT_INVALID = "planner_output_invalid"
AWAITING_OUTPUT_CORRECTION = "awaiting_output_correction"
OUTPUT_CORRECTION_EXHAUSTED = "output_correction_exhausted"
PLANNER_VALIDATED = "planner_validated"
SCOPE_WAITING = "scope_waiting"
VALIDATED = "validated"
COMPLETED = "completed"
SUPERSEDED = "superseded"
LEGACY_PROMPT_BUG = "legacy_forbidden_as_invariants_prompt_bug"

_RANK = {
    AWAITING_PLANNER: 0, WAITING_EXTERNAL: 0, AWAITING_OUTPUT_CORRECTION: 0,
    PLANNER_RESPONSE_DURABLE: 1, PLANNER_OUTPUT_INVALID: 2,
    OUTPUT_CORRECTION_EXHAUSTED: 3, PLANNER_VALIDATED: 4, SCOPE_WAITING: 5,
    VALIDATED: 6, COMPLETED: 7, SUPERSEDED: 7,
}
FINISHED = frozenset({VALIDATED, COMPLETED, SUPERSEDED})
_AWAITING = frozenset({AWAITING_PLANNER, WAITING_EXTERNAL, AWAITING_OUTPUT_CORRECTION})
# Statuses of one output attempt after which a correction may open.
_CORRECTABLE = frozenset({PLANNER_RESPONSE_DURABLE, PLANNER_OUTPUT_INVALID, OUTPUT_CORRECTION_EXHAUSTED})
_MAX_JSON_BYTES = 256 * 1024


class ContractRepairIntegrityError(Exception):
    """A repair slot cannot be proven to belong to the resumed operation."""

    code = "RESUME_INTEGRITY_FAILURE"


@dataclass(frozen=True)
class PendingContractRepair:
    directory: Path
    number: int
    step_id: str
    tree_sha: str
    mismatch: str
    transaction: dict[str, Any]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def repair_identity(cycle: int, step_id: str, number: int) -> str:
    """The stable identity of one semantic repair, shared by every resume."""

    return f"contract-repair:cycle-{cycle:03d}:{step_id}:{number:02d}"


def repair_dirs(artifact_dir: Path) -> list[Path]:
    root = artifact_dir / "contract_repairs"
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def read_transaction(directory: Path) -> dict[str, Any] | None:
    path = directory / TRANSACTION_NAME
    if not path.exists():
        return None
    data = _read_json(path)
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != SCHEMA_VERSION
        or data.get("status") not in _RANK
        or data.get("repair_number") != int(directory.name)
        or not _positive_int(data.get("output_attempt", 1))
    ):
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} transaction marker is malformed"
        )
    return data


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def output_attempt(transaction: dict[str, Any]) -> int:
    """The current output attempt; slots predating corrections are at 1."""

    value = transaction.get("output_attempt", 1)
    return value if _positive_int(value) else 1


def _order(transaction: dict[str, Any]) -> tuple[int, int]:
    return output_attempt(transaction), _RANK[transaction["status"]]


def _write(directory: Path, data: dict[str, Any]) -> None:
    atomic_write_text(
        directory / TRANSACTION_NAME,
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def begin(
    directory: Path, *, number: int, cycle: int, step_id: str,
    current_contract: str, mismatch: str, tree_sha: str,
    output_correction_limit: int = 0,
) -> dict[str, Any]:
    """Durably open one semantic repair slot before any planner call."""

    if (directory / TRANSACTION_NAME).exists():
        raise ContractRepairIntegrityError(
            f"contract repair {number:02d} slot is already open"
        )
    directory.mkdir(parents=True, exist_ok=True)
    mismatch_sha = sha256_text(mismatch)
    # The archived mismatch lands first: a transaction marker always names
    # evidence that already exists.
    atomic_write_text(directory / MISMATCH_NAME, json.dumps({
        "schema_version": SCHEMA_VERSION, "step_id": step_id,
        "tree_before": tree_sha, "mismatch": mismatch,
        "mismatch_sha256": mismatch_sha,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    data = {
        "schema_version": SCHEMA_VERSION,
        "repair_id": repair_identity(cycle, step_id, number),
        "repair_number": number,
        "cycle": cycle,
        "step_id": step_id,
        "original_contract_sha256": sha256_text(current_contract),
        "mismatch_sha256": mismatch_sha,
        "tree_sha": tree_sha,
        "status": AWAITING_PLANNER,
        "planner_transport_attempt": 0,
        "output_attempt": 1,
        "output_correction_attempt": 0,
        "output_correction_limit": output_correction_limit,
    }
    _write(directory, data)
    return data


def advance(directory: Path, status: str, **fields: Any) -> dict[str, Any]:
    """Move a transaction forward; a backward transition fails closed."""

    current = read_transaction(directory)
    if current is None:
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} has no transaction marker"
        )
    updated = {**current, **fields, "status": status}
    before, after = output_attempt(current), updated.get("output_attempt", 1)
    opens_correction = (
        after == before + 1
        and status == AWAITING_OUTPUT_CORRECTION
        and current["status"] in _CORRECTABLE
    )
    if (
        status not in _RANK
        or not _positive_int(after)
        or (after != before and not opens_correction)
        or _order(updated) < _order(current)
    ):
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} cannot move from "
            f"{current['status']} (output attempt {before}) to {status} "
            f"(output attempt {after})"
        )
    if updated != current:
        _write(directory, updated)
    return updated


def ensure(directory: Path, status: str, **fields: Any) -> dict[str, Any]:
    """Advance to ``status`` unless the transaction is already beyond it."""

    current = read_transaction(directory)
    if current is not None:
        target = (fields.get("output_attempt", output_attempt(current)), _RANK[status])
        if _order(current) > target:
            return current
    return advance(directory, status, **fields)


def is_awaiting_planner(transaction: dict[str, Any]) -> bool:
    return transaction.get("status") in _AWAITING


def awaiting_status(transaction: dict[str, Any]) -> str:
    """The awaiting status of the transaction's current output attempt."""

    return AWAITING_PLANNER if output_attempt(transaction) == 1 else AWAITING_OUTPUT_CORRECTION


def planner_response_durable(directory: Path, attempt: int = 1) -> bool:
    """Whether a paid planner answer of one output attempt is durable."""

    files = step_repair_attempt_files(directory, attempt)
    meta = _read_json(files.meta)
    if not isinstance(meta, dict) or meta.get("status") not in {"raw", "validated"}:
        return False
    try:
        return hashlib.sha256(files.raw.read_bytes()).hexdigest() == meta.get("raw_sha256")
    except OSError:
        return False


def episode_summary(directory: Path) -> dict[str, Any]:
    """Bounded, secret-free facts of one repair slot for diagnostics and UI.

    Raw answers stay in their artifacts; only hashes and the bounded
    deterministic parse error are exposed.  Never raises.
    """

    summary: dict[str, Any] = {"slot": directory.name}
    try:
        transaction = read_transaction(directory)
    except ContractRepairIntegrityError as exc:
        return {**summary, "integrity_error": str(exc)}
    if transaction is None:
        return {**summary, "transaction": None}
    for key in (
        "repair_id", "repair_number", "step_id", "status", "tree_sha",
        "planner_transport_attempt", "output_attempt", "output_correction_attempt",
        "output_correction_limit", "operator_output_retries", "planner_restarts",
        "last_transport_failure",
    ):
        if key in transaction:
            summary[key] = transaction[key]
    summary.setdefault("output_attempt", 1)
    summary.setdefault("output_correction_attempt", 0)
    attempts: list[dict[str, Any]] = []
    for number in range(1, output_attempt(transaction) + 1):
        files = step_repair_attempt_files(directory, number)
        meta = _read_json(files.meta)
        item: dict[str, Any] = {"output_attempt": number}
        try:
            state, _raw = step_repair_attempt_state(files)
            item["parse_status"] = state
        except StepContractRepairArtifactError as exc:
            item["parse_status"] = "integrity_error"
            item["integrity_error"] = str(exc)
        if isinstance(meta, dict):
            item["request_sha256"] = meta.get("request_sha256")
            item["raw_sha256"] = meta.get("raw_sha256")
            if meta.get("status") == "validated":
                item["parse_status"] = "validated"
        error = _read_json(files.parse_error)
        if isinstance(error, dict):
            item["parse_error_code"] = error.get("code")
            item["parse_error_detail"] = str(error.get("detail") or "")[:500]
        attempts.append(item)
    summary["output_attempts"] = attempts
    validation = _read_json(directory / "validation.json")
    if isinstance(validation, dict):
        summary["validation"] = {
            key: validation.get(key) for key in (
                "status", "output_attempt", "original_step_contract_sha256",
                "repaired_contract_sha256", "added_mutable_paths",
            ) if key in validation
        }
    return summary


def durable_request_matches(directory: Path, tree_sha: str) -> bool | None:
    """``None`` without a durable request; else whether it is intact."""

    request_path = directory / "planner.request.txt"
    if not request_path.exists():
        return None
    meta = _read_json(directory / "request.meta.json")
    try:
        digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    except OSError:
        return False
    return (
        isinstance(meta, dict)
        and meta.get("request_sha256") == digest
        and meta.get("current_tree_sha") == tree_sha
    )


def semantic_repair_count(artifact_dir: Path) -> int:
    """Count semantic slots; superseded generator bugs use no budget."""

    return sum(
        (transaction := read_transaction(directory)) is None
        or transaction["status"] != SUPERSEDED
        for directory in repair_dirs(artifact_dir)
    )


def next_repair_number(artifact_dir: Path) -> int:
    dirs = repair_dirs(artifact_dir)
    return int(dirs[-1].name) + 1 if dirs else 1


def _prompt_section(prompt: bytes, label: bytes) -> bytes | None:
    opening = b"<" + label + b">\n"
    closing = b"\n</" + label + b">"
    if prompt.count(opening) != 1 or prompt.count(closing) != 1:
        return None
    after = prompt.split(opening, 1)[1]
    return after.split(closing, 1)[0] if closing in after else None


def _latest_attempt(artifact_dir: Path) -> Path | None:
    root = artifact_dir / "attempts"
    if not root.is_dir():
        return None
    attempts = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    return max(attempts, key=lambda path: int(path.name)) if attempts else None


def legacy_prompt_bug_candidate(artifact_dir: Path) -> bool:
    attempt = _latest_attempt(artifact_dir)
    if attempt is None:
        return False
    diagnostics = _read_json(attempt / "prompt.diagnostics.json")
    if not isinstance(diagnostics, dict) or diagnostics.get("role") != "implementer":
        return False
    sections = diagnostics.get("sections")
    return isinstance(sections, list) and any(
        isinstance(item, dict) and item.get("name") == "step_invariants"
        for item in sections
    )


def legacy_prompt_bug_proven(
    pending: PendingContractRepair, *, step_forbidden: str,
) -> bool:
    """Prove the archived worker received forbidden bytes under two labels."""

    attempt = _latest_attempt(pending.directory.parent.parent)
    if attempt is None:
        return False
    for attempt in (attempt,):
        record = _read_json(attempt / "step.json")
        diagnostics = _read_json(attempt / "prompt.diagnostics.json")
        if not isinstance(record, dict) or not isinstance(diagnostics, dict):
            continue
        if (
            record.get("id") != pending.step_id
            or record.get("reason") != "AGENT_CONTRACT_MISMATCH"
            or record.get("tree_before") != pending.tree_sha
            or record.get("tree_after") != pending.tree_sha
            or record.get("mismatch") != pending.mismatch
            or diagnostics.get("role") != "implementer"
        ):
            continue
        sections = diagnostics.get("sections")
        if not isinstance(sections, list):
            continue
        by_name = {
            item.get("name"): item for item in sections
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        invariants = by_name.get("step_invariants")
        forbidden = by_name.get("forbidden_contract")
        if not isinstance(invariants, dict) or not isinstance(forbidden, dict):
            continue
        try:
            prompt = (attempt / "agent.prompt.txt").read_bytes()
        except OSError:
            continue
        invariant_bytes = _prompt_section(prompt, b"STEP INVARIANTS")
        forbidden_bytes = _prompt_section(prompt, b"FORBIDDEN CONTRACT")
        if invariant_bytes is None or forbidden_bytes is None:
            continue
        digest = hashlib.sha256(invariant_bytes).hexdigest()
        if (
            invariant_bytes == forbidden_bytes == step_forbidden.encode("utf-8")
            and digest == invariants.get("sha256") == forbidden.get("sha256")
            and len(invariant_bytes) == invariants.get("bytes") == forbidden.get("bytes")
            and invariants.get("authority") is True
            and forbidden.get("authority") is True
            and invariants.get("truncated") is False
            and forbidden.get("truncated") is False
            and len(prompt) == diagnostics.get("prompt_bytes")
        ):
            return True
    return False


def supersede_legacy_prompt_bug(pending: PendingContractRepair) -> dict[str, Any]:
    if pending.transaction.get("status") not in _AWAITING or planner_response_durable(pending.directory):
        raise ContractRepairIntegrityError("only a pending planner repair can be superseded")
    return advance(
        pending.directory, SUPERSEDED,
        superseded_reason=LEGACY_PROMPT_BUG,
        superseded_at=datetime.now(timezone.utc).isoformat(),
        tree_sha=pending.tree_sha,
        repair_id=pending.transaction["repair_id"],
    )


def find_pending(
    artifact_dir: Path, *, cycle: int, step_id: str, current_contract: str,
    legacy_mismatch_sources: list[str],
) -> PendingContractRepair | None:
    """The single unfinished repair slot of a step, if any.

    ``legacy_mismatch_sources`` are archived worker mismatch reports (newest
    first), used only to adopt a slot created before transaction markers.
    """

    dirs = repair_dirs(artifact_dir)
    pending: list[Path] = []
    for directory in dirs:
        transaction = read_transaction(directory)
        if transaction is not None:
            if transaction["status"] not in FINISHED:
                pending.append(directory)
            continue
        validation = _read_json(directory / "validation.json")
        if isinstance(validation, dict) and validation.get("status") == "validated":
            continue
        pending.append(directory)
    if not pending:
        return None
    if len(pending) > 1 or pending[-1] != dirs[-1]:
        raise ContractRepairIntegrityError(
            "more than one unfinished contract repair slot exists"
        )
    directory = pending[0]
    transaction = read_transaction(directory)
    if transaction is None:
        transaction = _adopt_legacy(
            directory, cycle=cycle, step_id=step_id, current_contract=current_contract,
            legacy_mismatch_sources=legacy_mismatch_sources,
        )
    return _verified(directory, transaction, step_id=step_id, current_contract=current_contract)


def _verified(
    directory: Path, transaction: dict[str, Any], *, step_id: str, current_contract: str,
) -> PendingContractRepair:
    archived = _read_json(directory / MISMATCH_NAME)
    tree_sha = transaction.get("tree_sha")
    if (
        transaction.get("step_id") != step_id
        or not isinstance(tree_sha, str)
        or not isinstance(archived, dict)
        or not isinstance(archived.get("mismatch"), str)
        or archived.get("tree_before") != tree_sha
        or archived.get("step_id") != step_id
        or sha256_text(archived["mismatch"]) != transaction.get("mismatch_sha256")
    ):
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} mismatch evidence does not match its transaction"
        )
    if transaction.get("original_contract_sha256") != sha256_text(current_contract):
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} was opened for a different step contract"
        )
    if durable_request_matches(directory, tree_sha) is False:
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} durable planner request changed"
        )
    return PendingContractRepair(
        directory=directory, number=int(directory.name), step_id=step_id,
        tree_sha=tree_sha, mismatch=archived["mismatch"], transaction=transaction,
    )


def _adopt_legacy(
    directory: Path, *, cycle: int, step_id: str, current_contract: str,
    legacy_mismatch_sources: list[str],
) -> dict[str, Any]:
    """Adopt a pre-transaction slot only when its identity is provable.

    The durable request must be intact and must embed both the current
    contract and one archived worker mismatch verbatim; otherwise the slot
    is left untouched and the operator is asked to decide.
    """

    meta = _read_json(directory / "request.meta.json")
    tree_sha = meta.get("current_tree_sha") if isinstance(meta, dict) else None
    if not isinstance(tree_sha, str) or durable_request_matches(directory, tree_sha) is not True:
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} predates transaction markers and "
            "has no intact durable request; operator decision required"
        )
    request = (directory / "planner.request.txt").read_text(encoding="utf-8")
    mismatch = next(
        (text for text in legacy_mismatch_sources if text and text in request), None,
    )
    if current_contract not in request or mismatch is None:
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} predates transaction markers and "
            "its mismatch or contract cannot be identified; operator decision required"
        )
    number = int(directory.name)
    status = AWAITING_PLANNER
    if planner_response_durable(directory):
        status = PLANNER_RESPONSE_DURABLE
    validation = _read_json(directory / "validation.json")
    if isinstance(validation, dict) and validation.get("status") == "planner_validated":
        status = PLANNER_VALIDATED
    atomic_write_text(directory / MISMATCH_NAME, json.dumps({
        "schema_version": SCHEMA_VERSION, "step_id": step_id,
        "tree_before": tree_sha, "mismatch": mismatch,
        "mismatch_sha256": sha256_text(mismatch),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    data = {
        "schema_version": SCHEMA_VERSION,
        "repair_id": repair_identity(cycle, step_id, number),
        "repair_number": number,
        "cycle": cycle,
        "step_id": step_id,
        "original_contract_sha256": sha256_text(current_contract),
        "mismatch_sha256": sha256_text(mismatch),
        "tree_sha": tree_sha,
        "status": status,
        "planner_transport_attempt": 1,
        "adopted_legacy_slot": True,
    }
    _write(directory, data)
    return data


__all__ = [
    "AWAITING_OUTPUT_CORRECTION", "AWAITING_PLANNER", "COMPLETED", "OUTPUT_CORRECTION_EXHAUSTED",
    "PLANNER_OUTPUT_INVALID", "awaiting_status", "episode_summary", "output_attempt", "ContractRepairIntegrityError", "FINISHED",
    "PLANNER_RESPONSE_DURABLE", "PLANNER_VALIDATED", "PendingContractRepair",
    "SCOPE_WAITING", "SUPERSEDED", "VALIDATED", "WAITING_EXTERNAL", "advance", "begin",
    "durable_request_matches", "ensure", "find_pending", "is_awaiting_planner", "planner_response_durable", "repair_dirs",
    "legacy_prompt_bug_candidate", "legacy_prompt_bug_proven", "next_repair_number", "repair_identity",
    "semantic_repair_count", "sha256_text", "supersede_legacy_prompt_bug",
]
