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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..result import atomic_write_text

TRANSACTION_NAME = "transaction.json"
MISMATCH_NAME = "mismatch.json"
SCHEMA_VERSION = 1

AWAITING_PLANNER = "awaiting_planner"
WAITING_EXTERNAL = "waiting_external"
PLANNER_RESPONSE_DURABLE = "planner_response_durable"
PLANNER_VALIDATED = "planner_validated"
SCOPE_WAITING = "scope_waiting"
VALIDATED = "validated"
COMPLETED = "completed"

_RANK = {
    AWAITING_PLANNER: 0, WAITING_EXTERNAL: 0,
    PLANNER_RESPONSE_DURABLE: 1, PLANNER_VALIDATED: 2, SCOPE_WAITING: 3,
    VALIDATED: 4, COMPLETED: 5,
}
FINISHED = frozenset({VALIDATED, COMPLETED})
_AWAITING = frozenset({AWAITING_PLANNER, WAITING_EXTERNAL})
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
    ):
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} transaction marker is malformed"
        )
    return data


def _write(directory: Path, data: dict[str, Any]) -> None:
    atomic_write_text(
        directory / TRANSACTION_NAME,
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def begin(
    directory: Path, *, number: int, cycle: int, step_id: str,
    current_contract: str, mismatch: str, tree_sha: str,
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
    if status not in _RANK or _RANK[status] < _RANK[current["status"]]:
        raise ContractRepairIntegrityError(
            f"contract repair {directory.name} cannot move from "
            f"{current['status']} to {status}"
        )
    updated = {**current, **fields, "status": status}
    if updated != current:
        _write(directory, updated)
    return updated


def ensure(directory: Path, status: str, **fields: Any) -> dict[str, Any]:
    """Advance to ``status`` unless the transaction is already beyond it."""

    current = read_transaction(directory)
    if current is not None and _RANK[current["status"]] > _RANK[status]:
        return current
    return advance(directory, status, **fields)


def is_awaiting_planner(transaction: dict[str, Any]) -> bool:
    return transaction.get("status") in _AWAITING


def planner_response_durable(directory: Path) -> bool:
    """Whether a paid planner answer for this slot is already durable."""

    meta = _read_json(directory / "request.meta.json")
    raw = directory / "planner.raw.md"
    if not isinstance(meta, dict) or meta.get("status") not in {"raw", "validated"}:
        return False
    try:
        return hashlib.sha256(raw.read_bytes()).hexdigest() == meta.get("raw_sha256")
    except OSError:
        return False


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
    """The highest opened semantic slot; transport retries never add one."""

    dirs = repair_dirs(artifact_dir)
    return int(dirs[-1].name) if dirs else 0


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
    "AWAITING_PLANNER", "COMPLETED", "ContractRepairIntegrityError", "FINISHED",
    "PLANNER_RESPONSE_DURABLE", "PLANNER_VALIDATED", "PendingContractRepair",
    "SCOPE_WAITING", "VALIDATED", "WAITING_EXTERNAL", "advance", "begin",
    "durable_request_matches", "ensure", "find_pending", "is_awaiting_planner", "planner_response_durable", "repair_dirs",
    "repair_identity", "semantic_repair_count", "sha256_text",
]
