"""Safe, bounded HTTP-facing reads for the local MetaHarness UI."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ..agent.events import parse_event, summarize_event
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from ..models import RunStatus
from ..state import RunStateStore

ARTIFACT_ALLOWLIST = frozenset(
    {
        "state.json",
        "planner.raw.md",
        "implementation_contract.md",
        "agent.events.jsonl",
        "checks.json",
        "review.json",
        "reviewer.raw.md",
        "repair_task.md",
    }
)
PROGRESS_MAX_BYTES = 256 * 1024
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class WebAPIError(Exception):
    """An expected API failure with an HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def validate_run_id(value: str) -> str:
    """Validate one unquoted run-id path component."""

    if not isinstance(value, str):
        raise WebAPIError(400, "invalid run id")
    try:
        decoded = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise WebAPIError(400, "invalid run id") from exc
    if (
        decoded != value
        or not decoded
        or decoded in {".", ".."}
        or "/" in decoded
        or "\\" in decoded
        or "\x00" in decoded
        or _RUN_ID.fullmatch(decoded) is None
        or ".." in decoded
        or decoded.endswith((".", ".lock"))
    ):
        raise WebAPIError(400, "invalid run id")
    return decoded


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def _load_state(run_dir: Path) -> dict[str, Any]:
    """Load state once more after a short transient read failure."""

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            state = RunStateStore(run_dir / "state.json").load()
            if not isinstance(state, dict):
                raise TypeError("run state must contain an object")
            return state
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.01)
    if isinstance(last_error, FileNotFoundError):
        raise WebAPIError(404, "run not found") from last_error
    raise WebAPIError(503, "run state is temporarily unavailable") from last_error


def _run_dir(runs_root: Path, run_id: str) -> Path:
    """Resolve a run below exactly one configured runs-root component."""

    safe_id = validate_run_id(run_id)
    root = runs_root.expanduser().resolve()
    try:
        directory = (root / safe_id).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise WebAPIError(404, "run not found") from exc
    if directory.parent != root or not directory.is_dir():
        raise WebAPIError(404, "run not found")
    return directory


def _artifact_path(run_dir: Path, name: str) -> Path:
    if name not in ARTIFACT_ALLOWLIST:
        raise ValueError("artifact is not allowlisted")
    return run_dir / name


def _state_summary(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    return {
        "run_id": run_id,
        "status": state.get("status"),
        "updated_at": state.get("updated_at"),
        "plan_title": planner.get("title"),
        "commit_sha": state.get("commit_sha"),
        "failure": state.get("failure"),
    }


def list_runs(runs_root: Path) -> list[dict[str, Any]]:
    """List direct child runs, tolerating incomplete or transient entries."""

    root = runs_root.expanduser().resolve()
    try:
        entries = list(root.iterdir())
    except (FileNotFoundError, OSError):
        return []
    result: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            safe_id = validate_run_id(entry.name)
            directory = _run_dir(root, safe_id)
            state = _load_state(directory)
        except WebAPIError:
            continue
        result.append(_state_summary(safe_id, state))
    result.sort(
        key=lambda item: (
            str(item.get("updated_at") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )
    return result


def get_run(runs_root: Path, run_id: str) -> dict[str, Any]:
    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    state = _load_state(directory)
    raw_plan = _load_text(_artifact_path(directory, "planner.raw.md"))
    contract = _load_text(_artifact_path(directory, "implementation_contract.md"))
    return {
        **_state_summary(safe_id, state),
        "state": state,
        "plan": {
            "raw": raw_plan,
            "contract": contract,
        },
        # Named aliases keep the response convenient for small API clients;
        # both values still come exclusively from the allowlisted artifacts.
        "planner_raw": raw_plan,
        "implementation_contract": contract,
        "checks": _load_json(_artifact_path(directory, "checks.json")),
        "review": _load_json(_artifact_path(directory, "review.json")),
        "reviewer_raw": _load_text(_artifact_path(directory, "reviewer.raw.md")),
        "repair_task": _load_text(_artifact_path(directory, "repair_task.md")),
        "failure": state.get("failure"),
    }


def progress(runs_root: Path, run_id: str, offset: int) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise WebAPIError(400, "offset must be a non-negative integer")
    directory = _run_dir(runs_root, run_id)
    path = _artifact_path(directory, "agent.events.jsonl")
    try:
        size = path.stat().st_size
        offset = min(offset, size)
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(PROGRESS_MAX_BYTES)
    except FileNotFoundError:
        return {"next_offset": 0, "events": []}
    except OSError as exc:
        raise WebAPIError(503, "progress is temporarily unavailable") from exc

    complete_length = data.rfind(b"\n") + 1
    complete = data[:complete_length]
    events: list[str] = []
    for line in complete.splitlines():
        event = parse_event(line.decode("utf-8", errors="replace"))
        if event is None:
            continue
        summary = summarize_event(event)
        if summary:
            events.append(summary)
    return {"next_offset": offset + complete_length, "events": events}


def approve_run(runs_root: Path, run_id: str, decision: str) -> dict[str, Any]:
    """Perform the only web mutation through the core approval API."""

    try:
        selected = ApprovalDecision(decision)
    except (TypeError, ValueError) as exc:
        raise WebAPIError(400, "decision must be APPROVE or REJECT") from exc
    directory = _run_dir(runs_root, run_id)
    state = _load_state(directory)
    if state.get("status") != RunStatus.AWAITING_PLAN_APPROVAL.value:
        raise WebAPIError(409, "run is not awaiting plan approval")
    stored = state.get("plan_identity")
    if not isinstance(stored, dict):
        raise WebAPIError(409, "run has no valid plan identity")
    try:
        expected = PlanIdentity(
            raw_sha256=stored["raw_sha256"],
            contract_sha256=stored["contract_sha256"],
        )
        actual = compute_plan_identity_from_run(directory)
    except (KeyError, TypeError, ValueError, ApprovalError, OSError, UnicodeError) as exc:
        raise WebAPIError(409, "plan artifacts do not match run state") from exc
    if actual != expected:
        raise WebAPIError(409, "plan artifacts do not match run state")
    try:
        write_plan_approval(
            directory,
            decision=selected,
            identity=expected,
            source="web-ui",
        )
    except ApprovalError as exc:
        raise WebAPIError(409, "plan approval already exists or is invalid") from exc
    return {"ok": True, "decision": selected.value}


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "PROGRESS_MAX_BYTES",
    "WebAPIError",
    "approve_run",
    "get_run",
    "list_runs",
    "progress",
    "validate_run_id",
]
