"""Safe, bounded HTTP-facing reads for the local MetaHarness UI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping
from urllib.parse import unquote

from ..agent.events import parse_event, summarize_event, summarize_step_event
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from ..execution_selection import (
    ExecutionSelectionConflict,
    ExecutionSelectionError,
    ensure_execution_selection,
    is_profile_aware_run,
    read_execution_selection_with_sha256,
    resolve_execution_selection,
    ensure_execution_selection_v3,
    read_execution_selection_v3_with_sha256,
    resolve_execution_selection_v3,
    validate_execution_selection_v3,
)
from ..models import ExecutionRole, ExecutionSelection, HarnessConfig, RunStatus
from ..planning_v2 import V2PlanParseError, step_contract_path, validate_implementation_bundle
from ..profiles import ProfileError, profile_for_role, profiles_for_config, safe_profile_metadata
from ..state import RunStateStore
from ..usage import (
    PLANNER_USAGE_ARTIFACT,
    REVIEWER_USAGE_ARTIFACT,
    add_usage,
    empty_usage,
    normalize_usage,
    read_usage_artifact,
)
from .run_manager import RunCapacityError, RunCollisionError, RunManager, RunManagerError

ARTIFACT_ALLOWLIST = frozenset(
    {
        "state.json",
        "spec.md",
        "planner.raw.md",
        "implementation_contract.md",
        "task_plan.json",
        "task_plan_v2.json",
        "implementation_bundle.json",
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
        "agent.result.json",
        "agent.final.md",
        "agent.stderr.log",
        "revision/agent.prompt.txt",
        "revision/agent.events.jsonl",
        "revision/agent.result.json",
        "revision/agent.final.md",
        "revision/agent.stderr.log",
        "changed-files.txt",
        "diff.patch",
        "plan_approval.json",
        "setup/results.json",
        PLANNER_USAGE_ARTIFACT,
        REVIEWER_USAGE_ARTIFACT,
    }
)
# A step contract is at most 8000 characters; anything larger is not shown.
MAX_STEP_CONTRACT_BYTES = 64 * 1024
STEP_EVENTS_MAX = 30
# Worker input above this many tokens is flagged (advisory only).
HIGH_WORKER_INPUT_TOKENS = 100_000
_STEP_ID = re.compile(r"S0[1-6]\Z")
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
MAX_DIAGNOSTIC_TAIL_BYTES = 32 * 1024
MAX_DIFF_BYTES = 64 * 1024
MAX_RESULT_BYTES = 128 * 1024
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


def _load_json(path: Path, *, max_bytes: int | None = None) -> Any:
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _load_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def _tail_text(path: Path, max_bytes: int) -> str | None:
    """Read only a bounded UTF-8 tail from an allowlisted artifact."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read(max_bytes).decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError, UnicodeError):
        return None


def _bounded_changed_files(path: Path) -> list[str]:
    content = _load_text_bounded(path, MAX_DIFF_BYTES)
    if content is None:
        return []
    return [line for line in content.splitlines() if line][:4096]


def _load_text_bounded(path: Path, max_bytes: int) -> str | None:
    try:
        with path.open("rb") as stream:
            data = stream.read(max_bytes)
        return data.decode("utf-8", errors="replace")
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
    approval_payload = _load_json(_artifact_path(directory, "plan_approval.json"))
    approval_decision = (
        approval_payload.get("decision")
        if isinstance(approval_payload, dict)
        and approval_payload.get("decision") in {ApprovalDecision.APPROVE.value, ApprovalDecision.REJECT.value}
        else None
    )
    agent_result = _load_json(
        _artifact_path(directory, "agent.result.json"), max_bytes=MAX_RESULT_BYTES
    )
    if not isinstance(agent_result, dict):
        agent_result = {}
    raw_usage = agent_result.get("usage")
    safe_usage = (
        {
            key: value
            for key, value in raw_usage.items()
            if key in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(value, int)
            and not isinstance(value, bool)
        }
        if isinstance(raw_usage, dict)
        else {}
    )
    safe_result = {
        key: agent_result[key]
        for key in ("exit_code", "timed_out", "usage")
        if key in agent_result
    }
    safe_result["usage"] = safe_usage
    agent_diagnostics = {
        "result": safe_result,
        "final_tail": _tail_text(
            _artifact_path(directory, "agent.final.md"), MAX_DIAGNOSTIC_TAIL_BYTES
        ),
        "stderr_tail": _tail_text(
            _artifact_path(directory, "agent.stderr.log"), MAX_DIAGNOSTIC_TAIL_BYTES
        ),
        "usage": safe_usage,
    }
    workspace_setup = _load_json(_artifact_path(directory, "setup/results.json"))
    if not isinstance(workspace_setup, list):
        workspace_setup = state.get("workspace_setup", [])
    bundle = _load_json(_artifact_path(directory, "implementation_bundle.json"), max_bytes=MAX_RESULT_BYTES)
    declared_hashes = {
        entry.get("id"): entry.get("contract_sha256")
        for entry in (bundle.get("steps") if isinstance(bundle, dict) and isinstance(bundle.get("steps"), list) else [])
        if isinstance(entry, dict)
    }
    step_artifacts: list[dict[str, Any]] = []
    raw_steps = state.get("steps") if isinstance(state.get("steps"), list) else []
    for item in raw_steps:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or _STEP_ID.fullmatch(item["id"]) is None:
            continue
        step_id = item["id"]
        step_dir = directory / "steps" / step_id
        contract = _step_contract(directory, step_id, declared_hashes.get(step_id))
        step_json = _load_json(step_dir / "step.json", max_bytes=MAX_RESULT_BYTES)
        step_usage = (
            normalize_usage(step_json.get("usage"))
            if isinstance(step_json, dict) and isinstance(step_json.get("usage"), dict)
            else None
        )
        step_artifacts.append({
            **item,
            "contract": contract["text"],
            "contract_sha256": contract["sha256"],
            "contract_matches_bundle": contract["matches"],
            "contract_layout": contract["layout"],
            "final": _tail_text(step_dir / "agent.final.md", 8 * 1024),
            "stderr": _tail_text(step_dir / "agent.stderr.log", 8 * 1024),
            "result": _load_json(step_dir / "agent.result.json", max_bytes=MAX_RESULT_BYTES),
            "events": step_progress_tail(directory, step_id, max_events=STEP_EVENTS_MAX),
            "usage": step_usage,
            "high_context": bool(
                step_usage and step_usage["input_tokens"] > HIGH_WORKER_INPUT_TOKENS
            ),
        })
    changed_files = _bounded_changed_files(
        _artifact_path(directory, "changed-files.txt")
    )
    diff_tail = _tail_text(_artifact_path(directory, "diff.patch"), MAX_DIFF_BYTES)
    # Keep this shape stable for both the JSON API and the server-rendered run
    # page; all newly exposed artifact data below is bounded or allowlisted.
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
        "task_plan": _load_json(_artifact_path(directory, "task_plan.json")),
        "implementation_bundle": _load_json(_artifact_path(directory, "implementation_bundle.json")),
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
        "approval": {"recorded": approval_decision is not None, "decision": approval_decision},
        "agent_diagnostics": agent_diagnostics,
        "progress_tail": progress_tail(runs_root, safe_id, max_events=50),
        "candidate": {"changed_files": changed_files, "diff_tail": diff_tail},
        "workspace_setup": workspace_setup,
        "step_artifacts": step_artifacts,
        "usage": _usage_summary(directory, step_artifacts, raw_usage),
    }


def _step_contract(
    directory: Path, step_id: str, declared_sha256: Any
) -> dict[str, Any]:
    """Read one step contract exactly as stored, never re-rendered.

    ``matches`` tells whether these bytes are the ones hashed in
    ``implementation_bundle.json``.  Historic P20 runs stored contracts as
    ``steps/Sxx.contract.md``; that layout is shown read-only.
    """

    for layout, path in (
        ("canonical", step_contract_path(directory, step_id)),
        ("legacy", directory / "steps" / f"{step_id}.contract.md"),
    ):
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_STEP_CONTRACT_BYTES + 1)
        except FileNotFoundError:
            continue
        except OSError:
            break
        if len(data) > MAX_STEP_CONTRACT_BYTES:
            return {"text": None, "sha256": None, "matches": False, "layout": layout}
        digest = hashlib.sha256(data).hexdigest()
        return {
            "text": data.decode("utf-8", errors="replace"),
            "sha256": digest,
            "matches": isinstance(declared_sha256, str) and digest == declared_sha256,
            "layout": layout,
        }
    return {"text": None, "sha256": None, "matches": False, "layout": None}


def _usage_summary(
    directory: Path, step_artifacts: list[dict[str, Any]], v1_agent_usage: Any
) -> dict[str, Any]:
    """Aggregate persisted token usage per phase; no pricing is derived."""

    planner = read_usage_artifact(directory / PLANNER_USAGE_ARTIFACT) or empty_usage()
    reviewer = read_usage_artifact(directory / REVIEWER_USAGE_ARTIFACT) or empty_usage()
    steps = [
        {"id": item["id"], "usage": item["usage"]}
        for item in step_artifacts
        if isinstance(item.get("usage"), dict)
    ]
    if steps:
        implementer = add_usage(step["usage"] for step in steps)
    elif isinstance(v1_agent_usage, dict) and v1_agent_usage:
        implementer = normalize_usage(v1_agent_usage)
    else:
        implementer = empty_usage()
    return {
        "planner": planner,
        "implementer": {"total": implementer, "steps": steps},
        "reviewer": reviewer,
        "grand_total": add_usage((planner, implementer, reviewer)),
    }


def _summaries(
    complete_lines: bytes,
    summarize: Callable[[dict[str, Any]], str | None] = summarize_event,
) -> list[str]:
    events: list[str] = []
    for line in complete_lines.splitlines():
        event = parse_event(line.decode("utf-8", errors="replace"))
        if event is None:
            continue
        summary = summarize(event)
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


def _read_progress(
    stream: BinaryIO,
    offset: int,
    summarize: Callable[[dict[str, Any]], str | None] = summarize_event,
) -> dict[str, Any]:
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
        return {"next_offset": offset + complete_length, "events": _summaries(data[:complete_length], summarize)}
    if len(data) < PROGRESS_MAX_BYTES:
        # A partial final line that is still being written.
        return {"next_offset": offset, "events": []}

    # One line is longer than the window: it is read whole only up to the
    # event limit, so memory stays bounded whatever the line length.
    line = data + stream.read(max(0, PROGRESS_MAX_EVENT_BYTES - len(data)))
    newline = line.find(b"\n")
    if newline >= 0:
        return {"next_offset": offset + newline + 1, "events": _summaries(line[: newline + 1], summarize)}
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


def progress_tail(runs_root: Path, run_id: str, max_events: int = 50) -> list[str]:
    """Return at most the latest human-readable progress events.

    The existing bounded JSONL reader is deliberately reused so a malformed
    or very large event never turns the HTML/API read into an unbounded load.
    """

    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise WebAPIError(400, "max_events must be a non-negative integer")
    if max_events == 0:
        return []
    directory = _run_dir(runs_root, run_id)
    return _tail_events(_artifact_path(directory, "agent.events.jsonl"), max_events)


def _tail_events(
    path: Path,
    max_events: int,
    summarize: Callable[[dict[str, Any]], str | None] = summarize_event,
) -> list[str]:
    """Latest summaries of a JSONL file, read through the bounded window."""

    events: deque[str] = deque(maxlen=max_events)
    offset = 0
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            while offset < size:
                payload = _read_progress(stream, offset, summarize)
                next_offset = int(payload["next_offset"])
                if next_offset <= offset:
                    break
                events.extend(str(item) for item in payload.get("events", []))
                offset = next_offset
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise WebAPIError(503, "progress is temporarily unavailable") from exc
    return list(events)


def step_progress_tail(
    run_dir: Path,
    step_id: str,
    *,
    max_events: int = STEP_EVENTS_MAX,
) -> list[str]:
    """Latest compact events of one v2 step (``steps/Sxx/agent.events.jsonl``).

    Uses the same bounded JSONL window as :func:`progress_tail`; tool
    arguments are never rendered.
    """

    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise WebAPIError(400, "max_events must be a non-negative integer")
    if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None:
        raise WebAPIError(400, "invalid step id")
    if max_events == 0:
        return []
    path = Path(run_dir) / "steps" / step_id / "agent.events.jsonl"
    return _tail_events(path, max_events, summarize_step_event)


def approve_run(
    runs_root: Path,
    run_id: str,
    decision: str,
    *,
    config: HarnessConfig | None = None,
    implementer_profile: object = None,
    reviewer_profile: object = None,
    reviser_profile: object = None,
    step_profiles: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Perform the only web mutation through the core approval API."""

    try:
        selected = ApprovalDecision(decision)
    except (TypeError, ValueError) as exc:
        raise WebAPIError(400, "decision must be APPROVE or REJECT") from exc
    # 1-2. Load state and require the approval gate.
    directory = _run_dir(runs_root, run_id)
    state = _load_state(directory)
    if state.get("status") != RunStatus.AWAITING_PLAN_APPROVAL.value:
        raise WebAPIError(409, "run is not awaiting plan approval")
    profile_aware = is_profile_aware_run(state)
    v2 = state.get("planning_protocol") == "v2"
    if v2:
        if selected is ApprovalDecision.APPROVE:
            if config is None or not isinstance(step_profiles, Mapping) or not isinstance(reviewer_profile, str):
                raise WebAPIError(400, "step profiles and reviewer_profile are required")
            if any(not isinstance(value, str) for value in step_profiles.values()):
                raise WebAPIError(400, "invalid step profile field")
    if selected is ApprovalDecision.APPROVE and profile_aware and not v2:
        if config is None or not isinstance(implementer_profile, str) or not isinstance(reviewer_profile, str):
            raise WebAPIError(400, "implementer_profile and reviewer_profile are required")
    # 3. Verify the plan artifacts shown to the human.
    stored = state.get("plan_identity")
    if not isinstance(stored, dict):
        raise WebAPIError(409, "run has no valid plan identity")
    try:
        expected = PlanIdentity(
            raw_sha256=stored["raw_sha256"],
            contract_sha256=stored["contract_sha256"],
            execution_sha256=stored.get("execution_sha256"),
            bundle_sha256=stored.get("bundle_sha256"),
        )
        actual = compute_plan_identity_from_run(directory)
    except (KeyError, TypeError, ValueError, ApprovalError, OSError, UnicodeError) as exc:
        raise WebAPIError(409, "plan artifacts do not match run state") from exc
    if v2:
        matches = (actual.raw_sha256, actual.contract_sha256, actual.bundle_sha256) == (
            expected.raw_sha256, expected.contract_sha256, expected.bundle_sha256,
        )
    elif profile_aware:
        # A selection already claimed by a concurrent request is checked by
        # ensure_execution_selection(), not here: only raw/contract matter.
        matches = (actual.raw_sha256, actual.contract_sha256) == (
            expected.raw_sha256,
            expected.contract_sha256,
        )
    else:
        matches = actual == expected
    if not matches:
        raise WebAPIError(409, "plan artifacts do not match run state")

    if v2 and selected is ApprovalDecision.REJECT:
        _publish_decision(directory, selected, expected)
        return {"ok": True, "decision": selected.value}

    if v2:
        try:
            bundle, _ = validate_implementation_bundle(directory)
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise WebAPIError(409, "plan artifacts do not match run state") from exc
        expected_ids = [entry["id"] for entry in bundle["steps"]]
        if set(step_profiles) != set(expected_ids):
            raise WebAPIError(400, "missing or unknown step profile field")
        try:
            requested = resolve_execution_selection_v3(
                config,
                planner_profile_id=state["execution"]["planner"]["profile_id"],
                step_profile_ids={key: value for key, value in step_profiles.items()},
                reviewer_profile_id=reviewer_profile,
                reviser_profile_id=(reviser_profile or config.ui.default_reviser_profile),
            )
        except (ProfileError, ExecutionSelectionError) as exc:
            raise WebAPIError(400, "selected profile is invalid") from exc
        try:
            ensure_execution_selection_v3(directory, requested)
            durable, execution_sha256 = read_execution_selection_v3_with_sha256(directory)
            validate_execution_selection_v3(config, durable)
        except ExecutionSelectionConflict as exc:
            raise WebAPIError(409, "a different execution selection is already recorded") from exc
        except ExecutionSelectionError as exc:
            raise WebAPIError(409, "execution selection is invalid") from exc
        if durable != requested:
            raise WebAPIError(409, "a different execution selection is already recorded")
        identity = PlanIdentity(
            raw_sha256=expected.raw_sha256,
            contract_sha256=expected.contract_sha256,
            execution_sha256=execution_sha256,
            bundle_sha256=expected.bundle_sha256,
        )
        _publish_decision(directory, selected, identity)
        # Compare-and-set: once the orchestrator has left the gate it owns the
        # state, and this write must not resurrect the approval status.
        RunStateStore(directory / "state.json").update_if_status(
            RunStatus.AWAITING_PLAN_APPROVAL,
            plan_identity=asdict(identity),
            execution=_execution_state_v3(durable),
        )
        return {"ok": True, "decision": selected.value}

    if not (selected is ApprovalDecision.APPROVE and profile_aware):
        # REJECT never executes anything; historic runs keep schema v1.
        identity = (
            PlanIdentity(expected.raw_sha256, expected.contract_sha256)
            if profile_aware
            else expected
        )
        _publish_decision(directory, selected, identity)
        return {"ok": True, "decision": selected.value}

    # 4-5. Resolve the requested profiles from trusted configuration only.
    try:
        requested = resolve_execution_selection(
            config,
            planner_profile_id=state["execution"]["planner"]["profile_id"],
            implementer_profile_id=implementer_profile,
            reviewer_profile_id=reviewer_profile,
            reviser_profile_id=(reviser_profile or config.ui.default_reviser_profile),
        )
    except ProfileError as exc:
        raise WebAPIError(400, "selected profile is invalid") from exc
    # 6. Claim the selection immutably, then read back the exact durable bytes.
    try:
        ensure_execution_selection(directory, requested)
        durable, execution_sha256 = read_execution_selection_with_sha256(directory)
    except ExecutionSelectionConflict as exc:
        raise WebAPIError(409, "a different execution selection is already recorded") from exc
    except ExecutionSelectionError as exc:
        raise WebAPIError(409, "execution selection is invalid") from exc
    if durable != requested:
        raise WebAPIError(409, "a different execution selection is already recorded")
    # 7-8. Bind the decision to that selection and publish it exclusively.
    identity = PlanIdentity(
        raw_sha256=expected.raw_sha256,
        contract_sha256=expected.contract_sha256,
        execution_sha256=execution_sha256,
    )
    _publish_decision(directory, selected, identity)
    # 9. Only now reflect the approved choice in state.json.  The orchestrator
    # may already have left the gate, in which case it owns the state: the
    # compare-and-set writes nothing then.
    RunStateStore(directory / "state.json").update_if_status(
        RunStatus.AWAITING_PLAN_APPROVAL,
        plan_identity=asdict(identity),
        execution=_execution_state(durable),
    )
    return {"ok": True, "decision": selected.value}


def _publish_decision(
    directory: Path, decision: ApprovalDecision, identity: PlanIdentity
) -> None:
    try:
        write_plan_approval(
            directory,
            decision=decision,
            identity=identity,
            source="web-ui",
        )
    except ApprovalError as exc:
        raise WebAPIError(409, "plan approval already exists or is invalid") from exc


def _execution_state(selection: ExecutionSelection) -> dict[str, Any]:
    state = {
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
    }
    if selection.reviser is not None:
        state["reviser"] = {
            "profile_id": selection.reviser.profile_id,
            "model": selection.reviser.model,
            "effort": selection.reviser.effort,
            "permission_mode": selection.reviser.permission_mode,
            "selection_mode": selection.reviser.selection_mode,
        }
    return state


def _execution_state_v3(selection: Any) -> dict[str, Any]:
    state = {
        "planner": asdict(selection.planner),
        "steps": [{"step_id": item.step_id, "implementer": asdict(item.implementer)} for item in selection.steps],
        "reviewer": asdict(selection.reviewer),
    }
    if selection.reviser is not None:
        state["reviser"] = asdict(selection.reviser)
    return state


def model_profiles(config: HarnessConfig) -> dict[str, Any]:
    profiles = profiles_for_config(config)
    defaults = {
        "planner": config.ui.default_planner_profile or "legacy-planner",
        "implementer": config.ui.default_implementer_profile or "legacy-implementer",
        "reviewer": config.ui.default_reviewer_profile or "legacy-reviewer",
        "reviser": config.ui.default_reviser_profile,
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
    except RunCollisionError as exc:
        raise WebAPIError(409, "run already exists or is active") from exc
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
    "HIGH_WORKER_INPUT_TOKENS",
    "MAX_SPEC_BYTES",
    "MAX_STEP_CONTRACT_BYTES",
    "MAX_DIAGNOSTIC_TAIL_BYTES",
    "MAX_DIFF_BYTES",
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
    "progress_tail",
    "step_progress_tail",
    "validate_run_id",
    "validate_spec",
]
