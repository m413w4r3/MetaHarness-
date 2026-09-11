"""Safe, bounded HTTP-facing reads for the local MetaHarness UI."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote

from ..agent.events import parse_event, summarize_event
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from ..execution_selection import resolve_execution_selection, write_execution_selection
from ..models import ExecutionRole, HarnessConfig, RunStatus
from ..profiles import ProfileError, profile_for_role, profiles_for_config, safe_profile_metadata
from ..state import RunStateStore
from .run_manager import RunCapacityError, RunManager, RunManagerError

ARTIFACT_ALLOWLIST = frozenset(
    {
        "state.json",
        "spec.md",
        "planner.raw.md",
        "implementation_contract.md",
        "execution_recommendation.request.txt",
        "execution_recommendation.raw.md",
        "execution_recommendation.json",
        "execution_recommendation.error.txt",
        "agent.events.jsonl",
        "checks.json",
        "review.json",
        "reviewer.raw.md",
        "execution_selection.json",
        "repair_task.md",
    }
)
# Window of complete JSONL lines returned by one progress request.
PROGRESS_MAX_BYTES = 256 * 1024
# A longer single event is omitted from the UI (the artifact keeps it; the
# Codex adapter parses lines up to 8 MiB for usage/final message).
PROGRESS_MAX_EVENT_BYTES = 1 * 1024 * 1024
# Bytes of an oversized line skipped per request without being stored.
PROGRESS_MAX_SKIP_BYTES = 8 * 1024 * 1024
_PROGRESS_SCAN_CHUNK_BYTES = 64 * 1024
OVERSIZED_EVENT = "[oversized Codex event omitted]"
MAX_SPEC_BYTES = 48 * 1024
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class WebAPIError(Exception):
    """An expected API failure with an HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def validate_spec(value: object) -> str:
    if not isinstance(value, str):
        raise WebAPIError(400, "spec must be a string")
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise WebAPIError(400, "spec must be valid UTF-8 text") from exc
    if encoded_length > MAX_SPEC_BYTES:
        raise WebAPIError(400, "spec is too large")
    if not value.strip():
        raise WebAPIError(400, "spec must not be empty")
    return value


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
    spec = _load_text(_artifact_path(directory, "spec.md"))
    contract = _load_text(_artifact_path(directory, "implementation_contract.md"))
    reviewer_raw = _load_text(_artifact_path(directory, "reviewer.raw.md"))
    # This shape is polled by the run page every STATE_POLL_MS: status,
    # updated_at, state.{base_sha,branch,worktree,commit_sha,failure}, plan,
    # checks, review and reviewer_raw(_available) must stay stable.
    return {
        **_state_summary(safe_id, state),
        "state": state,
        "spec": spec,
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
        "reviewer_raw": reviewer_raw,
        "reviewer_raw_available": reviewer_raw is not None,
        "execution_selection": _load_json(
            _artifact_path(directory, "execution_selection.json")
        ),
        "execution_recommendation": _load_json(
            _artifact_path(directory, "execution_recommendation.json")
        ),
        "repair_task": _load_text(_artifact_path(directory, "repair_task.md")),
        "failure": state.get("failure"),
    }


def _summaries(complete_lines: bytes) -> list[str]:
    events: list[str] = []
    for line in complete_lines.splitlines():
        event = parse_event(line.decode("utf-8", errors="replace"))
        if event is None:
            continue
        summary = summarize_event(event)
        if summary:
            events.append(summary)
    return events


def _skip_past_newline(stream: BinaryIO, start: int, limit: int) -> tuple[int, bool]:
    """Scan at most *limit* bytes from *start* without keeping them.

    Return the position just after the next newline and ``True``, or the
    position where scanning stopped (limit or end of file) and ``False``.
    """

    stream.seek(start)
    position = start
    remaining = limit
    while remaining > 0:
        chunk = stream.read(min(_PROGRESS_SCAN_CHUNK_BYTES, remaining))
        if not chunk:
            break
        newline = chunk.find(b"\n")
        if newline >= 0:
            return position + newline + 1, True
        position += len(chunk)
        remaining -= len(chunk)
    return position, False


def _read_progress(stream: BinaryIO, offset: int) -> dict[str, Any]:
    if offset > 0:
        stream.seek(offset - 1)
        if stream.read(1) != b"\n":
            # The offset is inside a line already reported as oversized (or
            # was supplied mid-line): drop the rest of that line first.
            position, found = _skip_past_newline(stream, offset, PROGRESS_MAX_SKIP_BYTES)
            if not found:
                return {"next_offset": position, "events": []}
            offset = position

    stream.seek(offset)
    data = stream.read(PROGRESS_MAX_BYTES)
    complete_length = data.rfind(b"\n") + 1
    if complete_length:
        return {"next_offset": offset + complete_length, "events": _summaries(data[:complete_length])}
    if len(data) < PROGRESS_MAX_BYTES:
        # A partial final line that is still being written.
        return {"next_offset": offset, "events": []}

    # One line is longer than the window: it is read whole only up to the
    # event limit, so memory stays bounded whatever the line length.
    line = data + stream.read(max(0, PROGRESS_MAX_EVENT_BYTES - len(data)))
    newline = line.find(b"\n")
    if newline >= 0:
        return {"next_offset": offset + newline + 1, "events": _summaries(line[: newline + 1])}
    if len(line) < PROGRESS_MAX_EVENT_BYTES:
        # Still being written and may end under the limit.
        return {"next_offset": offset, "events": []}
    # Oversized: never parsed.  The offset always moves forward (after the
    # newline when found in the bounded scan, otherwise mid-line where the
    # next request resumes skipping), so the same range is never re-read.
    position, _found = _skip_past_newline(
        stream, offset + len(line), PROGRESS_MAX_SKIP_BYTES
    )
    return {"next_offset": position, "events": [OVERSIZED_EVENT]}


def progress(runs_root: Path, run_id: str, offset: int) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise WebAPIError(400, "offset must be a non-negative integer")
    directory = _run_dir(runs_root, run_id)
    path = _artifact_path(directory, "agent.events.jsonl")
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            return _read_progress(stream, min(offset, size))
    except FileNotFoundError:
        return {"next_offset": 0, "events": []}
    except OSError as exc:
        raise WebAPIError(503, "progress is temporarily unavailable") from exc


def approve_run(
    runs_root: Path,
    run_id: str,
    decision: str,
    *,
    config: HarnessConfig | None = None,
    implementer_profile: object = None,
    reviewer_profile: object = None,
) -> dict[str, Any]:
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
    modern_selection = isinstance(state.get("execution"), dict) and isinstance(
        state.get("execution", {}).get("planner"), dict
    ) and bool(state.get("execution", {}).get("planner", {}).get("profile_id"))
    if selected is ApprovalDecision.APPROVE and modern_selection:
        if config is None or not isinstance(implementer_profile, str) or not isinstance(reviewer_profile, str):
            raise WebAPIError(400, "implementer_profile and reviewer_profile are required")
    try:
        expected = PlanIdentity(
            raw_sha256=stored["raw_sha256"],
            contract_sha256=stored["contract_sha256"],
            execution_sha256=stored.get("execution_sha256"),
        )
        actual = compute_plan_identity_from_run(directory)
    except (KeyError, TypeError, ValueError, ApprovalError, OSError, UnicodeError) as exc:
        raise WebAPIError(409, "plan artifacts do not match run state") from exc
    if actual != expected:
        raise WebAPIError(409, "plan artifacts do not match run state")
    if selected is ApprovalDecision.APPROVE and modern_selection:
        planner_profile = state["execution"]["planner"]["profile_id"]
        try:
            selection = resolve_execution_selection(
                config,
                planner_profile_id=planner_profile,
                implementer_profile_id=implementer_profile,
                reviewer_profile_id=reviewer_profile,
            )
            write_execution_selection(directory, selection)
            identity = compute_plan_identity_from_run(directory)
            RunStateStore(directory / "state.json").update(
                status=state["status"],
                plan_identity=identity.__dict__,
                execution={
                    "planner": {
                        "profile_id": selection.planner.profile_id,
                        "model": selection.planner.model,
                        "selection_mode": selection.planner.selection_mode,
                    },
                    "implementer": {
                        "profile_id": selection.implementer.profile_id,
                        "model": selection.implementer.model,
                        "effort": selection.implementer.effort,
                        "selection_mode": selection.implementer.selection_mode,
                    },
                    "reviewer": {
                        "profile_id": selection.reviewer.profile_id,
                        "model": selection.reviewer.model,
                        "selection_mode": selection.reviewer.selection_mode,
                    },
                },
            )
        except (ProfileError, ValueError, OSError, UnicodeError) as exc:
            raise WebAPIError(400, "selected profile is invalid") from exc
    else:
        identity = expected
    try:
        write_plan_approval(
            directory,
            decision=selected,
            identity=identity,
            source="web-ui",
        )
    except ApprovalError as exc:
        raise WebAPIError(409, "plan approval already exists or is invalid") from exc
    return {"ok": True, "decision": selected.value}


def model_profiles(config: HarnessConfig) -> dict[str, Any]:
    profiles = profiles_for_config(config)
    defaults = {
        "planner": config.ui.default_planner_profile or "legacy-planner",
        "implementer": config.ui.default_implementer_profile or "legacy-implementer",
        "reviewer": config.ui.default_reviewer_profile or "legacy-reviewer",
    }
    return {
        "profiles": [safe_profile_metadata(profile) for profile in profiles.values()],
        "defaults": defaults,
    }


def create_run(
    manager: RunManager,
    *,
    spec: object,
    run_id: object = None,
    planner_profile: object = None,
) -> dict[str, str]:
    content = validate_spec(spec)
    if planner_profile is not None and not isinstance(planner_profile, str):
        raise WebAPIError(400, "planner_profile is invalid")
    selected_planner = planner_profile or manager._config.ui.default_planner_profile or "legacy-planner"
    try:
        profile_for_role(manager._config, selected_planner, ExecutionRole.PLANNER)
    except ProfileError as exc:
        raise WebAPIError(400, "planner_profile is invalid or incompatible") from exc
    selected_id: str | None = None
    if run_id is not None:
        if not isinstance(run_id, str):
            raise WebAPIError(400, "invalid run id")
        selected_id = validate_run_id(run_id)
        root = manager._config.runs_root.expanduser().resolve()
        if (root / selected_id).exists():
            raise WebAPIError(409, "run already exists")
    try:
        if planner_profile is None:
            created_id = manager.start_run(content, run_id=selected_id)
        else:
            created_id = manager.start_run(
                content, run_id=selected_id, planner_profile=selected_planner
            )
    except RunCapacityError as exc:
        raise WebAPIError(409, "maximum active runs reached") from exc
    except RunManagerError as exc:
        # A concurrent creator can win between the existence check and the
        # reservation; expose the same durable collision contract.
        if selected_id is not None:
            root = manager._config.runs_root.expanduser().resolve()
            if (root / selected_id).exists():
                raise WebAPIError(409, "run already exists") from exc
        raise WebAPIError(503, "run could not be created") from exc
    return {"ok": True, "run_id": created_id, "location": f"/runs/{created_id}"}  # type: ignore[dict-item]


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "MAX_SPEC_BYTES",
    "OVERSIZED_EVENT",
    "PROGRESS_MAX_BYTES",
    "PROGRESS_MAX_EVENT_BYTES",
    "PROGRESS_MAX_SKIP_BYTES",
    "WebAPIError",
    "approve_run",
    "create_run",
    "model_profiles",
    "get_run",
    "list_runs",
    "progress",
    "validate_run_id",
    "validate_spec",
]
