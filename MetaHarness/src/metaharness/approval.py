"""Durable human approval for a parsed implementation plan."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


class ApprovalError(ValueError):
    """The plan approval artifact is absent, invalid, or already decided."""


class ApprovalDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class PlanIdentity:
    """SHA-256 identity of the exact plan artifacts shown to a human."""

    raw_sha256: str
    contract_sha256: str
    execution_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.raw_sha256, "raw_sha256")
        _validate_sha256(self.contract_sha256, "contract_sha256")
        if self.execution_sha256 is not None:
            _validate_sha256(self.execution_sha256, "execution_sha256")


@dataclass(frozen=True)
class PlanApproval:
    decision: ApprovalDecision
    raw_sha256: str
    contract_sha256: str
    created_at: str
    source: str
    execution_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            decision = ApprovalDecision(self.decision)
        except (TypeError, ValueError) as exc:
            raise ApprovalError("approval decision is unknown") from exc
        if decision is not self.decision:
            object.__setattr__(self, "decision", decision)
        _validate_sha256(self.raw_sha256, "raw_sha256")
        _validate_sha256(self.contract_sha256, "contract_sha256")
        if self.execution_sha256 is not None:
            _validate_sha256(self.execution_sha256, "execution_sha256")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ApprovalError("approval created_at must be a non-empty string")
        if not isinstance(self.source, str) or self.source not in _SOURCES:
            raise ApprovalError("approval source is invalid")


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCES = frozenset({"cli", "web-ui", "test"})
_SCHEMA_VERSION = 2
_APPROVAL_FILENAME = "plan_approval.json"


def _validate_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ApprovalError(f"approval {field} must be lowercase SHA-256 hex")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_path(run_dir: str | Path) -> Path:
    return Path(run_dir).expanduser().resolve()


def compute_plan_identity(
    raw: str, contract: str, execution_sha256: str | None = None
) -> PlanIdentity:
    """Hash the exact UTF-8 encoding of both plan strings."""

    if not isinstance(raw, str) or not isinstance(contract, str):
        raise TypeError("raw and contract must be strings")
    return PlanIdentity(
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        contract_sha256=hashlib.sha256(contract.encode("utf-8")).hexdigest(),
        execution_sha256=execution_sha256,
    )


def compute_plan_identity_from_run(run_dir: str | Path) -> PlanIdentity:
    """Hash the bytes currently persisted in a run's two plan artifacts."""

    directory = _run_path(run_dir)
    try:
        raw = (directory / "planner.raw.md").read_bytes()
        contract = (directory / "implementation_contract.md").read_bytes()
    except (OSError, UnicodeError) as exc:
        raise ApprovalError(f"could not read plan artifacts: {exc}") from exc
    execution_path = directory / "execution_selection.json"
    execution_sha256: str | None = None
    if execution_path.exists():
        try:
            execution_sha256 = hashlib.sha256(execution_path.read_bytes()).hexdigest()
        except (OSError, UnicodeError) as exc:
            raise ApprovalError(f"could not read execution selection: {exc}") from exc
    return PlanIdentity(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_sha256=hashlib.sha256(contract).hexdigest(),
        execution_sha256=execution_sha256,
    )


def _approval_payload(approval: PlanApproval) -> dict[str, object]:
    return {
        "schema_version": 2 if approval.execution_sha256 is not None else 1,
        "decision": approval.decision.value,
        "raw_sha256": approval.raw_sha256,
        "contract_sha256": approval.contract_sha256,
        **(
            {"execution_sha256": approval.execution_sha256}
            if approval.execution_sha256 is not None
            else {}
        ),
        "created_at": approval.created_at,
        "source": approval.source,
    }


def _publish_exclusive(path: Path, content: str) -> None:
    """Publish atomically and fail if the destination already exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ApprovalError("plan approval already exists") from exc
        finally:
            # The hard link is the atomic publication.  Removing the private
            # temporary name leaves the published inode in place.
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except ApprovalError:
        raise
    except OSError as exc:
        raise ApprovalError(f"could not write {path.name}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_plan_approval(
    run_dir: Path,
    *,
    decision: ApprovalDecision,
    identity: PlanIdentity,
    source: str,
) -> None:
    """Write one approval decision without ever replacing an existing one."""

    try:
        normalized_decision = ApprovalDecision(decision)
    except (TypeError, ValueError) as exc:
        raise ApprovalError("approval decision is unknown") from exc
    approval = PlanApproval(
        decision=normalized_decision,
        raw_sha256=identity.raw_sha256,
        contract_sha256=identity.contract_sha256,
        execution_sha256=identity.execution_sha256,
        created_at=_now(),
        source=source,
    )
    content = json.dumps(_approval_payload(approval), ensure_ascii=False, indent=2) + "\n"
    _publish_exclusive(_run_path(run_dir) / _APPROVAL_FILENAME, content)


def read_plan_approval(
    run_dir: Path,
    *,
    expected_identity: PlanIdentity,
) -> PlanApproval | None:
    """Read and validate the decision for exactly ``expected_identity``."""

    path = _run_path(run_dir) / _APPROVAL_FILENAME
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise ApprovalError(f"could not read {path.name}: {exc}") from exc
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ApprovalError("approval JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ApprovalError("approval JSON must contain an object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in (1, 2):
        raise ApprovalError("approval schema_version is unsupported")
    required = ("decision", "raw_sha256", "contract_sha256", "created_at", "source")
    if any(field not in payload for field in required):
        raise ApprovalError("approval artifact is missing an essential field")
    if schema_version == 2:
        if "execution_sha256" not in payload:
            raise ApprovalError("approval artifact is missing an essential field")
    elif "execution_sha256" in payload:
        raise ApprovalError("historic approval contains an execution hash")
    try:
        approval = PlanApproval(
            decision=ApprovalDecision(payload["decision"]),
            raw_sha256=payload["raw_sha256"],
            contract_sha256=payload["contract_sha256"],
            execution_sha256=payload.get("execution_sha256"),
            created_at=payload["created_at"],
            source=payload["source"],
        )
    except (ApprovalError, TypeError, ValueError) as exc:
        if isinstance(exc, ApprovalError):
            raise
        raise ApprovalError("approval artifact is invalid") from exc
    if approval.execution_sha256 is not None:
        try:
            execution_bytes = (_run_path(run_dir) / "execution_selection.json").read_bytes()
        except OSError as exc:
            raise ApprovalError("execution selection is missing") from exc
        actual_execution = hashlib.sha256(execution_bytes).hexdigest()
        if actual_execution != approval.execution_sha256:
            raise ApprovalError("execution selection does not match approval")
    if (
        approval.raw_sha256 != expected_identity.raw_sha256
        or approval.contract_sha256 != expected_identity.contract_sha256
        or (
            expected_identity.execution_sha256 is not None
            and approval.execution_sha256 != expected_identity.execution_sha256
        )
        or (expected_identity.execution_sha256 is None and approval.execution_sha256 is not None
            and schema_version != 2)
    ):
        raise ApprovalError("approval does not match the expected plan identity")
    return approval


def wait_for_plan_approval(
    run_dir: Path,
    *,
    identity: PlanIdentity,
    poll_interval_seconds: float,
) -> PlanApproval:
    """Poll until one valid human decision is durably present."""

    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(float(poll_interval_seconds))
        or poll_interval_seconds <= 0
        or poll_interval_seconds > 10
    ):
        raise ValueError("poll_interval_seconds must be > 0 and <= 10")
    while True:
        approval = read_plan_approval(run_dir, expected_identity=identity)
        if approval is not None:
            return approval
        time.sleep(poll_interval_seconds)


__all__ = [
    "ApprovalDecision",
    "ApprovalError",
    "PlanApproval",
    "PlanIdentity",
    "compute_plan_identity",
    "compute_plan_identity_from_run",
    "read_plan_approval",
    "wait_for_plan_approval",
    "write_plan_approval",
]
