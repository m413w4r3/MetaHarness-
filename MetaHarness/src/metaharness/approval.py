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

from .models import CheckConfig


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
    bundle_sha256: str | None = None
    execution_sha256: str | None = None
    checks_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.raw_sha256, "raw_sha256")
        _validate_sha256(self.contract_sha256, "contract_sha256")
        if self.execution_sha256 is not None:
            _validate_sha256(self.execution_sha256, "execution_sha256")
        if self.bundle_sha256 is not None:
            _validate_sha256(self.bundle_sha256, "bundle_sha256")
        if self.checks_sha256 is not None:
            _validate_sha256(self.checks_sha256, "checks_sha256")


@dataclass(frozen=True)
class PlanApproval:
    decision: ApprovalDecision
    raw_sha256: str
    contract_sha256: str
    created_at: str
    source: str
    bundle_sha256: str | None = None
    execution_sha256: str | None = None
    checks_sha256: str | None = None

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
        if self.bundle_sha256 is not None:
            _validate_sha256(self.bundle_sha256, "bundle_sha256")
        if self.checks_sha256 is not None:
            _validate_sha256(self.checks_sha256, "checks_sha256")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ApprovalError("approval created_at must be a non-empty string")
        if not isinstance(self.source, str) or self.source not in _SOURCES:
            raise ApprovalError("approval source is invalid")


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCES = frozenset({"cli", "web-ui", "test"})
_SCHEMA_VERSION = 5
_APPROVAL_FILENAME = "plan_approval.json"
_SCOPE_APPROVAL_FILENAME = "scope_approval.json"
_CHECK_AUTHORITY_FILENAME = "check_authority.json"
_CHECK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _validate_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ApprovalError(f"approval {field} must be lowercase SHA-256 hex")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_path(run_dir: str | Path) -> Path:
    return Path(run_dir).expanduser().resolve()


def compute_plan_identity(
    raw: str,
    contract: str,
    execution_sha256: str | None = None,
    *,
    bundle_sha256: str | None = None,
    checks_sha256: str | None = None,
) -> PlanIdentity:
    """Hash the exact UTF-8 encoding of both plan strings."""

    if not isinstance(raw, str) or not isinstance(contract, str):
        raise TypeError("raw and contract must be strings")
    return PlanIdentity(
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        contract_sha256=hashlib.sha256(contract.encode("utf-8")).hexdigest(),
        execution_sha256=execution_sha256,
        bundle_sha256=bundle_sha256,
        checks_sha256=checks_sha256,
    )


def _check_authority_payload(checks: tuple[CheckConfig, ...]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for check in checks:
        entry: dict[str, object] = {
            "id": check.id,
            "argv": list(check.argv),
            "cwd": check.cwd,
            "timeout_seconds": check.timeout_seconds,
            "preflight_argv": list(check.preflight_argv),
            "required": check.required,
        }
        if check.description:
            entry["description"] = check.description
        entries.append(entry)
    return {
        "schema_version": 1,
        "required_check_ids": [check.id for check in checks],
        "checks": entries,
    }


def _canonical_check_authority(checks: tuple[CheckConfig, ...]) -> bytes:
    return (
        json.dumps(
            _check_authority_payload(checks),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_check_authority_check(value: object, index: int) -> CheckConfig:
    if not isinstance(value, dict):
        raise ApprovalError(f"check authority checks[{index}] must be an object")
    allowed = {
        "id", "argv", "cwd", "timeout_seconds", "preflight_argv", "required", "description",
    }
    if set(value) - allowed:
        raise ApprovalError(f"check authority checks[{index}] contains an unknown field")
    check_id = value.get("id")
    if not isinstance(check_id, str) or _CHECK_ID.fullmatch(check_id) is None:
        raise ApprovalError(f"check authority checks[{index}].id is invalid")
    argv = value.get("argv")
    if isinstance(argv, (str, bytes)) or not isinstance(argv, list) or not argv:
        raise ApprovalError(f"check authority checks[{index}].argv must be a non-empty array")
    if any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise ApprovalError(f"check authority checks[{index}].argv is invalid")
    cwd = value.get("cwd")
    if not isinstance(cwd, str) or not cwd.strip() or "\x00" in cwd or Path(cwd).is_absolute():
        raise ApprovalError(f"check authority checks[{index}].cwd is invalid")
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ApprovalError(f"check authority checks[{index}].timeout_seconds is invalid")
    preflight = value.get("preflight_argv", [])
    if isinstance(preflight, (str, bytes)) or not isinstance(preflight, list):
        raise ApprovalError(f"check authority checks[{index}].preflight_argv is invalid")
    if any(not isinstance(item, str) or "\x00" in item for item in preflight):
        raise ApprovalError(f"check authority checks[{index}].preflight_argv is invalid")
    required = value.get("required")
    if not isinstance(required, bool):
        raise ApprovalError(f"check authority checks[{index}].required is invalid")
    description = value.get("description", "")
    if not isinstance(description, str) or len(description) > 300:
        raise ApprovalError(f"check authority checks[{index}].description is invalid")
    try:
        return CheckConfig(
            name=check_id, argv=tuple(argv), cwd=cwd, timeout_seconds=timeout,
            preflight_argv=tuple(preflight), required=required, description=description,
        )
    except (TypeError, ValueError) as exc:
        raise ApprovalError(f"check authority checks[{index}] is invalid") from exc


def read_check_authority(
    run_dir: str | Path,
    *,
    expected_sha256: str | None = None,
    trusted_check_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[tuple[str, ...], tuple[CheckConfig, ...]] | None:
    """Read the immutable, command-bearing check authority for a run."""

    path = _run_path(run_dir) / _CHECK_AUTHORITY_FILENAME
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ApprovalError(f"could not read {path.name}: {exc}") from exc
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ApprovalError("check authority does not match its approved hash")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalError("check authority JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ApprovalError("check authority schema_version is unsupported")
    required_ids = payload.get("required_check_ids")
    raw_checks = payload.get("checks")
    if (
        isinstance(required_ids, (str, bytes)) or not isinstance(required_ids, list)
        or isinstance(raw_checks, (str, bytes)) or not isinstance(raw_checks, list)
    ):
        raise ApprovalError("check authority has an invalid shape")
    if any(not isinstance(item, str) or _CHECK_ID.fullmatch(item) is None for item in required_ids):
        raise ApprovalError("check authority required_check_ids are invalid")
    if len(set(required_ids)) != len(required_ids):
        raise ApprovalError("check authority contains duplicate check IDs")
    checks = tuple(_validate_check_authority_check(item, index) for index, item in enumerate(raw_checks))
    if tuple(check.id for check in checks) != tuple(required_ids):
        raise ApprovalError("check authority order does not match required_check_ids")
    if trusted_check_ids is not None:
        trusted = set(trusted_check_ids)
        missing = [check_id for check_id in required_ids if check_id not in trusted]
        if missing:
            raise ApprovalError("check authority references an unknown trusted check ID: " + missing[0])
    if content != _canonical_check_authority(checks):
        raise ApprovalError("check authority JSON is not canonical")
    return tuple(required_ids), checks


def write_check_authority(
    run_dir: str | Path, checks: tuple[CheckConfig, ...] | list[CheckConfig]
) -> str:
    """Publish check definitions once; identical publication is idempotent."""

    normalized = tuple(checks)
    if any(not isinstance(check, CheckConfig) for check in normalized):
        raise ApprovalError("check authority must contain only trusted CheckConfig values")
    content = _canonical_check_authority(normalized)
    path = _run_path(run_dir) / _CHECK_AUTHORITY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        _publish_exclusive(path, content.decode("utf-8"))
    except OSError as exc:
        raise ApprovalError(f"could not read {path.name}: {exc}") from exc
    else:
        if existing != content:
            raise ApprovalError("check authority already exists with different bytes")
    return hashlib.sha256(content).hexdigest()


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
    bundle_path = directory / "implementation_bundle.json"
    bundle_sha256: str | None = None
    if bundle_path.exists():
        try:
            bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        except (OSError, UnicodeError) as exc:
            raise ApprovalError(f"could not read implementation bundle: {exc}") from exc
    checks_path = directory / _CHECK_AUTHORITY_FILENAME
    checks_sha256: str | None = None
    if checks_path.exists():
        try:
            checks_sha256 = hashlib.sha256(checks_path.read_bytes()).hexdigest()
        except (OSError, UnicodeError) as exc:
            raise ApprovalError(f"could not read {checks_path.name}: {exc}") from exc
    return PlanIdentity(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_sha256=hashlib.sha256(contract).hexdigest(),
        execution_sha256=execution_sha256,
        bundle_sha256=bundle_sha256,
        checks_sha256=checks_sha256,
    )


def _approval_payload(approval: PlanApproval, *, schema_version: int | None = None) -> dict[str, object]:
    schema_version = schema_version or (
        5 if approval.checks_sha256 is not None else
        3 if approval.bundle_sha256 is not None else
        2 if approval.execution_sha256 is not None else 1
    )
    return {
        "schema_version": schema_version,
        "decision": approval.decision.value,
        "raw_sha256": approval.raw_sha256,
        "contract_sha256": approval.contract_sha256,
        **(
            {"execution_sha256": approval.execution_sha256}
            if approval.execution_sha256 is not None or approval.bundle_sha256 is not None
            else {}
        ),
        **(
            {"bundle_sha256": approval.bundle_sha256}
            if approval.bundle_sha256 is not None
            else {}
        ),
        **(
            {"checks_sha256": approval.checks_sha256}
            if approval.checks_sha256 is not None
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
    # The web API historically rebuilt the identity before the check hash was
    # introduced.  If this is a new run, recover the already durable hash
    # here so the approval still binds the authority without changing that
    # compatibility boundary.
    if identity.checks_sha256 is None:
        authority_path = _run_path(run_dir) / _CHECK_AUTHORITY_FILENAME
        try:
            authority_bytes = authority_path.read_bytes()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ApprovalError("could not read check authority") from exc
        else:
            authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
            # The v2 web approval compatibility adapter may pass an identity
            # reconstructed without the newly added field.  The run state is
            # the already-persisted pre-approval identity; never approve an
            # authority whose bytes differ from that identity.
            state_path = _run_path(run_dir) / "state.json"
            try:
                state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
                state_payload = None
            stored_identity = state_payload.get("plan_identity") if isinstance(state_payload, dict) else None
            stored_checks = stored_identity.get("checks_sha256") if isinstance(stored_identity, dict) else None
            if stored_checks is not None and stored_checks != authority_sha256:
                raise ApprovalError("check authority does not match the run identity")
            identity = PlanIdentity(
                raw_sha256=identity.raw_sha256,
                contract_sha256=identity.contract_sha256,
                bundle_sha256=identity.bundle_sha256,
                execution_sha256=identity.execution_sha256,
                checks_sha256=authority_sha256,
            )
    approval = PlanApproval(
        decision=normalized_decision,
        raw_sha256=identity.raw_sha256,
        contract_sha256=identity.contract_sha256,
        execution_sha256=identity.execution_sha256,
        bundle_sha256=identity.bundle_sha256,
        checks_sha256=identity.checks_sha256,
        created_at=_now(),
        source=source,
    )
    directory = _run_path(run_dir)
    schema_version = None
    if normalized_decision is ApprovalDecision.APPROVE and identity.checks_sha256 is not None:
        schema_version = 5
    elif normalized_decision is ApprovalDecision.APPROVE and identity.execution_sha256 is not None and identity.bundle_sha256 is not None:
        try:
            selection_payload = json.loads((directory / "execution_selection.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ApprovalError("execution selection is missing or invalid") from exc
        if isinstance(selection_payload, dict) and selection_payload.get("schema_version") == 4:
            schema_version = 4
    content = json.dumps(_approval_payload(approval, schema_version=schema_version), ensure_ascii=False, indent=2) + "\n"
    _publish_exclusive(directory / _APPROVAL_FILENAME, content)


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
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version not in (1, 2, 3, 4, 5):
        raise ApprovalError("approval schema_version is unsupported")
    required = ("decision", "raw_sha256", "contract_sha256", "created_at", "source")
    if any(field not in payload for field in required):
        raise ApprovalError("approval artifact is missing an essential field")
    if schema_version == 2:
        if "execution_sha256" not in payload:
            raise ApprovalError("approval artifact is missing an essential field")
        if "bundle_sha256" in payload:
            raise ApprovalError("schema 2 approval contains a bundle hash")
    elif schema_version in (3, 4):
        if "execution_sha256" not in payload or "bundle_sha256" not in payload:
            raise ApprovalError("approval artifact is missing an essential field")
        if payload.get("execution_sha256") is None and payload.get("decision") != ApprovalDecision.REJECT.value:
            raise ApprovalError("approved schema 3 decision must bind execution")
    elif schema_version == 5:
        if "checks_sha256" not in payload:
            raise ApprovalError("schema 5 approval is missing the check authority hash")
    elif "execution_sha256" in payload:
        raise ApprovalError("historic approval contains an execution hash")
    if schema_version not in (3, 4, 5) and "bundle_sha256" in payload:
        raise ApprovalError("historic approval contains a bundle hash")
    if schema_version != 5 and "checks_sha256" in payload:
        raise ApprovalError("historic approval contains a check authority hash")
    if schema_version == 4:
        try:
            selection_payload = json.loads((_run_path(run_dir) / "execution_selection.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ApprovalError("execution selection is missing or invalid") from exc
        if not isinstance(selection_payload, dict) or selection_payload.get("schema_version") != 4:
            raise ApprovalError("schema 4 approval requires execution selection schema 4")
    try:
        approval = PlanApproval(
            decision=ApprovalDecision(payload["decision"]),
            raw_sha256=payload["raw_sha256"],
            contract_sha256=payload["contract_sha256"],
            execution_sha256=payload.get("execution_sha256"),
            bundle_sha256=payload.get("bundle_sha256"),
            checks_sha256=payload.get("checks_sha256"),
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
    if approval.bundle_sha256 is not None:
        try:
            bundle_bytes = (_run_path(run_dir) / "implementation_bundle.json").read_bytes()
        except OSError as exc:
            raise ApprovalError("implementation bundle is missing") from exc
        if hashlib.sha256(bundle_bytes).hexdigest() != approval.bundle_sha256:
            raise ApprovalError("implementation bundle does not match approval")
    if approval.checks_sha256 is not None:
        try:
            checks_bytes = (_run_path(run_dir) / _CHECK_AUTHORITY_FILENAME).read_bytes()
        except OSError as exc:
            raise ApprovalError("check authority is missing") from exc
        if hashlib.sha256(checks_bytes).hexdigest() != approval.checks_sha256:
            raise ApprovalError("check authority does not match approval")
    if (
        approval.raw_sha256 != expected_identity.raw_sha256
        or approval.contract_sha256 != expected_identity.contract_sha256
        or (
            expected_identity.execution_sha256 is not None
            and approval.execution_sha256 != expected_identity.execution_sha256
        )
        or (
            expected_identity.bundle_sha256 is not None
            and approval.bundle_sha256 != expected_identity.bundle_sha256
        )
        or (
            expected_identity.checks_sha256 is not None
            and approval.checks_sha256 != expected_identity.checks_sha256
        )
        or (expected_identity.execution_sha256 is None and approval.execution_sha256 is not None
            and schema_version not in (2, 3, 4, 5))
        or (expected_identity.bundle_sha256 is None and approval.bundle_sha256 is not None)
        or (expected_identity.checks_sha256 is None and approval.checks_sha256 is not None)
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


@dataclass(frozen=True)
class ScopeApproval:
    """Immutable decision for an exact, planner-derived scope delta."""

    decision: ApprovalDecision
    scope_delta_sha256: str
    created_at: str
    source: str

    def __post_init__(self) -> None:
        try:
            decision = ApprovalDecision(self.decision)
        except (TypeError, ValueError) as exc:
            raise ApprovalError("scope approval decision is unknown") from exc
        object.__setattr__(self, "decision", decision)
        _validate_sha256(self.scope_delta_sha256, "scope_delta_sha256")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ApprovalError("scope approval created_at must be non-empty")
        if not isinstance(self.source, str) or self.source not in _SOURCES:
            raise ApprovalError("scope approval source is invalid")


def write_scope_approval(
    run_dir: Path, *, decision: ApprovalDecision, scope_delta_sha256: str, source: str,
) -> None:
    approval = ScopeApproval(decision, scope_delta_sha256, _now(), source)
    content = json.dumps({
        "schema_version": 1, "decision": approval.decision.value,
        "scope_delta_sha256": approval.scope_delta_sha256,
        "created_at": approval.created_at, "source": approval.source,
    }, ensure_ascii=False, indent=2) + "\n"
    _publish_exclusive(_run_path(run_dir) / _SCOPE_APPROVAL_FILENAME, content)


def read_scope_approval(run_dir: Path, *, expected_sha256: str) -> ScopeApproval | None:
    path = _run_path(run_dir) / _SCOPE_APPROVAL_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ApprovalError("scope approval JSON is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ApprovalError("scope approval schema_version is unsupported")
    try:
        result = ScopeApproval(
            ApprovalDecision(payload["decision"]), payload["scope_delta_sha256"],
            payload["created_at"], payload["source"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApprovalError("scope approval artifact is invalid") from exc
    if result.scope_delta_sha256 != expected_sha256:
        raise ApprovalError("scope approval does not match scope delta")
    return result


__all__ = [
    "ApprovalDecision",
    "ApprovalError",
    "PlanApproval",
    "PlanIdentity",
    "ScopeApproval",
    "compute_plan_identity",
    "compute_plan_identity_from_run",
    "read_check_authority",
    "read_plan_approval",
    "wait_for_plan_approval",
    "write_plan_approval",
    "write_check_authority",
    "read_scope_approval",
    "write_scope_approval",
]
