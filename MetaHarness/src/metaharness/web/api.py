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

from ..agent.diagnostics import TOKEN_DIAGNOSTICS_NAME
from ..agent.events import parse_event, summarize_event, summarize_step_event
from ..diagnostics import DIAGNOSTICS_ERROR_NAME, DIAGNOSTICS_NAME, MAX_REPORT_BYTES, build_run_diagnostics
from ..resume import resume_info
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    read_scope_approval,
    write_scope_approval,
    write_plan_approval,
)
from ..execution_selection import (
    ExecutionSelectionConflict,
    ExecutionSelectionError,
    ensure_execution_selection,
    is_profile_aware_run,
    read_execution_selection_with_sha256,
    resolve_execution_selection,
    validate_execution_selection,
)
from ..models import (
    ExecutionClass,
    ImplementationStep,
    ExecutionSelection,
    HarnessConfig,
    PublishMode,
    RunStatus,
)
from ..progress import display_event, sync_progress
from ..redaction import config_secret_values
from ..planning.artifacts import step_contract_path, validate_implementation_bundle
from ..planning.protocol import V2PlanParseError
from ..profiles import ProfileError, profiles_for_config, safe_profile_metadata
from ..run_options import RunOptions, RunOptionsError, effective_run_config, read_run_options_for_state
from ..state import RunStateStore
from ..step_ids import STEP_ID_PATTERN, STEP_ID_RE
from ..usage import (
    PLANNER_USAGE_ARTIFACT,
    REVIEWER_USAGE_ARTIFACT,
    add_usage,
    normalize_usage,
    phase_usage_summary,
    read_usage_artifact,
)
from ..plan_recovery import (
    MAX_REPLACEMENT_PLAN_BYTES,
    PLAN_RECOVERY_ARTIFACT,
    PlanRecoveryError,
    plan_recovery_info,
    read_plan_recovery_record,
    validate_replacement_text,
)
from .run_manager import (
    RunCapacityError,
    RunCollisionError,
    RunManager,
    RunManagerError,
    RunPlanRecoveryError,
    RunResumeNotAllowedError,
)

ARTIFACT_ALLOWLIST = frozenset(
    {
        "state.json",
        "run_options.json",
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
        "trace/events.v1.jsonl",
        "checks.json",
        "review.json",
        "reviewer.raw.md",
        "execution_selection.json",
        "repair_task.md",
        "agent.result.json",
        "agent.final.md",
        "agent.stderr.log",
        "changed-files.txt",
        "diff.patch",
        "plan_approval.json",
        PLAN_RECOVERY_ARTIFACT,
        "publish.json",
        DIAGNOSTICS_NAME,
        DIAGNOSTICS_ERROR_NAME,
        "setup/results.json",
        PLANNER_USAGE_ARTIFACT,
        REVIEWER_USAGE_ARTIFACT,
    }
)
# The parser caps a step contract at 5000 characters; the UI read bound is
# deliberately higher so a malformed artifact is reported rather than read
# without a bound.
MAX_STEP_CONTRACT_BYTES = 64 * 1024
STEP_EVENTS_MAX = 30
# Worker input above this many tokens is flagged (advisory only).
HIGH_WORKER_INPUT_TOKENS = 100_000
_STEP_ID = STEP_ID_RE
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
MAX_DIFF_BYTES = 64 * 1024
MAX_RESULT_BYTES = 128 * 1024
MAX_ARTIFACT_READ_BYTES = 512 * 1024
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


# Every durable artifact of any cycle, by its generic layout only.
_CYCLE_ARTIFACT = re.compile(
    r"cycles/[0-9]{3}/(?:"
    rf"implementation/steps/{STEP_ID_PATTERN}"
    rf"|correction(?:/steps/{STEP_ID_PATTERN})?"
    r"|semantic-revision"
    r"|checks/[a-z-]+"
    r"|check-repair/[a-z-]+/attempts/[0-9]{3}"
    r"|candidate|review"
    r")/[A-Za-z0-9_][A-Za-z0-9_.-]*"
)


def _artifact_path(run_dir: Path, name: str) -> Path:
    if name not in ARTIFACT_ALLOWLIST and _CYCLE_ARTIFACT.fullmatch(name) is None:
        raise ValueError("artifact is not allowlisted")
    return run_dir / name


def get_artifact(runs_root: Path, run_id: str, name: str) -> dict[str, Any]:
    """Read one allowlisted artifact with a fixed memory bound.

    Missing artifacts return ``exists=False``, ``size=0`` and ``content=None``.
    Oversized files return their UTF-8 tail.
    """

    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    try:
        path = _artifact_path(directory, name)
    except ValueError as exc:
        raise WebAPIError(404, "artifact not found") from exc
    # The allowlist controls names; also prevent symlinks from escaping a run.
    try:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise WebAPIError(404, "artifact not found")
        size = path.stat().st_size
    except FileNotFoundError:
        return {
            "run_id": safe_id, "name": name, "exists": False,
            "encoding": "utf-8", "content": None, "truncated": False, "size": 0,
        }
    except OSError as exc:
        raise WebAPIError(503, "artifact is temporarily unavailable") from exc

    truncated = size > MAX_ARTIFACT_READ_BYTES
    content = (
        _tail_text(path, MAX_ARTIFACT_READ_BYTES)
        if truncated else _load_text_bounded(path, MAX_ARTIFACT_READ_BYTES)
    )
    if content is None:
        if not path.exists():
            return {
                "run_id": safe_id, "name": name, "exists": False,
                "encoding": "utf-8", "content": None, "truncated": False, "size": 0,
            }
        raise WebAPIError(503, "artifact is temporarily unavailable")
    return {
        "run_id": safe_id, "name": name, "exists": True,
        "encoding": "utf-8", "content": content, "truncated": truncated, "size": size,
    }


def _state_summary(run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    return {
        "run_id": run_id,
        "status": state.get("status"),
        "updated_at": state.get("updated_at"),
        "plan_title": planner.get("title"),
        "commit_sha": state.get("commit_sha"),
        "candidate": state.get("candidate"),
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


def get_run(
    runs_root: Path, run_id: str, *, config: HarnessConfig | None = None,
) -> dict[str, Any]:
    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    state = _load_state(directory)
    raw_plan = _load_text(_artifact_path(directory, "planner.raw.md"))
    spec = _load_text(_artifact_path(directory, "spec.md"))
    contract = _load_text(_artifact_path(directory, "implementation_contract.md"))
    approval_payload = _load_json(_artifact_path(directory, "plan_approval.json"))
    approval_decision = (
        approval_payload.get("decision")
        if isinstance(approval_payload, dict)
        and approval_payload.get("decision") in {ApprovalDecision.APPROVE.value, ApprovalDecision.REJECT.value}
        else None
    )
    workspace_setup = _load_json(_artifact_path(directory, "setup/results.json"))
    if not isinstance(workspace_setup, list):
        workspace_setup = state.get("workspace_setup", [])
    cycle_artifacts = _cycle_artifacts(directory, state)
    # The initial cycle's steps, rebuilt from their durable records.
    step_artifacts = cycle_artifacts[0]["steps"] if cycle_artifacts else []
    # Top-level aliases describe the final cycle that exists.
    last = _cycle_numbers(directory)[-1] if _cycle_numbers(directory) else 1
    root = f"cycles/{last:03d}"
    gate = _cycle_gate_dir(directory, last)
    gate_root = (
        gate.relative_to(directory).as_posix() if gate is not None
        else f"{root}/checks/post-implementation"
    )
    checks_path, review_path = f"{gate_root}/checks.json", f"{root}/review/review.json"
    scope_delta_path = _artifact_path(directory, f"{root}/correction/scope_delta.json")
    scope_approval_path = _artifact_path(directory, f"{root}/correction/scope_approval.json")
    scope_delta = _load_json(scope_delta_path, max_bytes=256 * 1024)
    scope_approval_payload = _load_json(scope_approval_path, max_bytes=16 * 1024)
    try:
        with scope_delta_path.open("rb") as stream:
            scope_delta_bytes = stream.read(256 * 1024 + 1)
        scope_delta_sha256 = hashlib.sha256(scope_delta_bytes).hexdigest() if len(scope_delta_bytes) <= 256 * 1024 else None
    except OSError:
        scope_delta_sha256 = None
    scope_approval_recorded = isinstance(scope_approval_payload, dict) and scope_approval_payload.get("decision") in {
        ApprovalDecision.APPROVE.value, ApprovalDecision.REJECT.value,
    } and scope_approval_payload.get("scope_delta_sha256") == scope_delta_sha256 and scope_delta_sha256 is not None
    reviewer_raw = _load_text(_artifact_path(directory, f"{root}/review/reviewer.raw.md"))
    revision_path = f"{root}/semantic-revision/report.json"
    changed_path, diff_path = f"{gate_root}/changed-files.txt", f"{gate_root}/diff.patch"
    changed_files = _bounded_changed_files(_artifact_path(directory, changed_path))
    diff_tail = _tail_text(_artifact_path(directory, diff_path), MAX_DIFF_BYTES)
    diagnostics_path = _artifact_path(directory, DIAGNOSTICS_NAME)
    diagnostics_content = _load_text_bounded(diagnostics_path, MAX_REPORT_BYTES)
    diagnostics_live = False
    if config is not None:
        # Rebuilt read-only from the current artifacts, so a waiting run never
        # shows a stale report; nothing is written and no authority changes.
        try:
            diagnostics_content = build_run_diagnostics(config, directory)
            diagnostics_live = True
        except Exception:
            pass
    diagnostics_meta: dict[str, Any] = {
        "path": DIAGNOSTICS_NAME,
        "available": diagnostics_content is not None,
        "size": None,
        "generated_at": None,
        "content": diagnostics_content,
        "live": diagnostics_live,
    }
    if diagnostics_content is not None:
        generated_match = re.search(r'"generated_at"\s*:\s*"([^"]+)"', diagnostics_content[:16 * 1024])
        if generated_match:
            diagnostics_meta["generated_at"] = generated_match.group(1)
    if diagnostics_live and diagnostics_content is not None:
        diagnostics_meta["size"] = len(diagnostics_content.encode("utf-8"))
    else:
        try:
            diagnostics_meta["size"] = diagnostics_path.stat().st_size
        except OSError:
            pass
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
        "checks": _load_json(_artifact_path(directory, checks_path)),
        "review": _load_json(_artifact_path(directory, review_path)),
        "reviewer_raw": reviewer_raw,
        "reviewer_raw_available": reviewer_raw is not None,
        "revision": _load_json(_artifact_path(directory, revision_path)),
        "execution_selection": _load_json(
            _artifact_path(directory, "execution_selection.json")
        ),
        "run_options": _load_json(
            _artifact_path(directory, "run_options.json")
        ),
        "execution_recommendation": _load_json(
            _artifact_path(directory, "execution_recommendation.json")
        ),
        "repair_task": _load_text(_artifact_path(directory, "repair_task.md")),
        "scope_delta": scope_delta,
        "scope_approval": {
            "recorded": scope_approval_recorded,
            "decision": scope_approval_payload.get("decision") if scope_approval_recorded else None,
        },
        "failure": state.get("failure"),
        "publish": _load_json(_artifact_path(directory, "publish.json")),
        "approval": {"recorded": approval_decision is not None, "decision": approval_decision},
        "plan_recovery": _plan_recovery_payload(directory, state),
        "progress_tail": progress_tail(
            runs_root, safe_id, max_events=50,
            secrets=config_secret_values(config) if config is not None else (),
        ),
        "candidate": {
            **(state.get("candidate") if isinstance(state.get("candidate"), dict) else {}),
            "changed_files": changed_files, "diff_tail": diff_tail,
        },
        "workspace_setup": workspace_setup,
        "step_artifacts": step_artifacts,
        "cycle": _state_cycle(state),
        "cycle_artifacts": cycle_artifacts,
        "usage": phase_usage_summary(directory),
        "overview": run_overview(directory, state, config),
        "diagnostics": diagnostics_meta,
    }


def _state_cycle(state: Mapping[str, Any]) -> int:
    cycle = state.get("cycle")
    return cycle if isinstance(cycle, int) and not isinstance(cycle, bool) and cycle >= 1 else 1


def _cycle_root(directory: Path, cycle: int) -> Path:
    if not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 1:
        raise WebAPIError(400, "invalid review cycle")
    return directory / "cycles" / f"{cycle:03d}"


def _cycle_numbers(directory: Path) -> list[int]:
    """Every durable cycle of a run, in order."""

    try:
        return sorted(
            int(entry.name) for entry in (directory / "cycles").iterdir()
            if entry.is_dir() and re.fullmatch(r"[0-9]{3}", entry.name) and int(entry.name) >= 1
        )
    except OSError:
        return []


def _cycle_contracts_dir(directory: Path, cycle: int) -> Path:
    """Where the approved plan of one cycle keeps its bundle and contracts."""

    return directory if cycle == 1 else _cycle_root(directory, cycle) / "correction"


def _cycle_gate_dir(directory: Path, cycle: int) -> Path | None:
    """The one deterministic gate directory of a cycle, if it ran."""

    try:
        gates = sorted(
            entry for entry in (_cycle_root(directory, cycle) / "checks").iterdir()
            if entry.is_dir()
        )
    except OSError:
        return None
    return gates[-1] if gates else None


def _cycle_kind(directory: Path, cycle: int) -> str:
    record = _load_json(_cycle_root(directory, cycle) / "cycle.json", max_bytes=4096)
    kind = record.get("kind") if isinstance(record, dict) else None
    return kind if isinstance(kind, str) else ("initial" if cycle == 1 else "unknown")


def _bundle_entries(path: Path) -> list[dict[str, Any]]:
    bundle = _load_json(path, max_bytes=MAX_RESULT_BYTES)
    steps = bundle.get("steps") if isinstance(bundle, dict) else None
    if not isinstance(steps, list):
        return []
    return [
        entry for entry in steps
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        and _STEP_ID.fullmatch(entry["id"]) is not None
    ]


_DURABLE_STEP_STATUS = {
    "COMPLETED": "completed",
    "DEFERRED_CONTRACT_MISMATCH": "deferred",
    "FAILED": "failed",
}


def _cycle_steps(directory: Path, cycle: int, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Steps of exactly one cycle, from that cycle's own artifact tree.

    Live status comes from ``state.steps`` only while the state describes
    this same cycle; otherwise the durable ``step.json`` is authoritative.
    """

    root = _cycle_root(directory, cycle)
    contracts = _cycle_contracts_dir(directory, cycle)
    entries = {entry["id"]: entry for entry in _bundle_entries(contracts / "implementation_bundle.json")}
    raw_live = state.get("steps") if _state_cycle(state) == cycle else None
    live = {
        item["id"]: item for item in (raw_live if isinstance(raw_live, list) else [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
        and _STEP_ID.fullmatch(item["id"]) is not None
    }
    ids = list(entries) or list(live)
    result: list[dict[str, Any]] = []
    for step_id in ids:
        entry = entries.get(step_id, {})
        step_dir = root / "implementation" / "steps" / step_id
        step_json = _load_json(step_dir / "step.json", max_bytes=MAX_RESULT_BYTES)
        step_json = step_json if isinstance(step_json, dict) else {}
        step_usage = (
            normalize_usage(step_json["usage"]) if isinstance(step_json.get("usage"), dict) else None
        )
        contract = _step_contract(contracts, step_id, entry.get("contract_sha256"))
        item = live.get(step_id, {})
        status = item.get("status") or _DURABLE_STEP_STATUS.get(
            str(step_json.get("status")), "waiting"
        )
        result.append({
            **item,
            "token_diagnostics": _token_diagnostics(step_dir),
            "context_level": context_level(step_usage),
            "id": step_id,
            "cycle": cycle,
            "title": item.get("title", entry.get("title")),
            "execution_class": item.get("execution_class") or step_json.get("execution_class") or entry.get("execution_class"),
            "profile_id": item.get("profile_id") or step_json.get("profile_id"),
            "status": status,
            "failure_reason": step_json.get("reason"),
            "mismatch": step_json.get("mismatch"),
            "initial_mismatch": step_json.get("initial_mismatch"),
            "mismatch_retry_count": step_json.get("mismatch_retry_count"),
            "deferred_verify": step_json.get("deferred_verify"),
            "tree_before": step_json.get("tree_before"),
            "tree_after": step_json.get("tree_after"),
            "contract": contract["text"],
            "contract_sha256": contract["sha256"],
            "contract_matches_bundle": contract["matches"],
            "contract_layout": contract["layout"],
            "final": _tail_text(step_dir / "agent.final.md", 8 * 1024),
            "stderr": _tail_text(step_dir / "agent.stderr.log", 8 * 1024),
            "result": _load_json(step_dir / "agent.result.json", max_bytes=MAX_RESULT_BYTES),
            "events": cycle_step_progress_tail(directory, cycle, step_id, max_events=STEP_EVENTS_MAX),
            "usage": step_usage,
            "high_context": bool(
                step_usage and step_usage["input_tokens"] > HIGH_WORKER_INPUT_TOKENS
            ),
        })
    return result


def _cycle_revision(directory: Path, cycle: int) -> dict[str, Any] | None:
    source = _cycle_root(directory, cycle) / "semantic-revision"
    if not (source / "tree_before.txt").exists() and not (source / "report.json").exists():
        return None
    return {
        "report": _load_json(source / "report.json", max_bytes=MAX_RESULT_BYTES),
        "pre_checks": _load_json(source / "pre_checks.json", max_bytes=MAX_RESULT_BYTES),
        "scope": _load_json(source / "scope.json", max_bytes=MAX_RESULT_BYTES),
        "final": _tail_text(source / "agent.final.md", 8 * 1024),
        "stderr": _tail_text(source / "agent.stderr.log", 8 * 1024),
        "events": _tail_events(source / "agent.events.jsonl", STEP_EVENTS_MAX, summarize_step_event),
        "usage": read_usage_artifact(source / "usage.json"),
    }


def _cycle_checks(directory: Path, cycle: int) -> dict[str, Any] | None:
    source = _cycle_gate_dir(directory, cycle)
    if source is None:
        return None
    checks = _load_json(source / "checks.json", max_bytes=MAX_RESULT_BYTES)
    if checks is None:
        return None
    evidence = _load_json(source / "evidence.json", max_bytes=MAX_RESULT_BYTES + MAX_DIFF_BYTES * 8)
    gate = (
        {"passed": evidence.get("deterministic_passed"), "failures": evidence.get("failures")}
        if isinstance(evidence, dict) else None
    )
    repairs = sorted((_cycle_root(directory, cycle) / "check-repair").glob("*/attempts/[0-9][0-9][0-9]"))
    return {
        "stage": source.name,
        "checks": checks,
        "gate": gate,
        "check_repair_attempts": [
            {
                "attempt": item.name,
                "record": _load_json(item / "attempt.json", max_bytes=MAX_RESULT_BYTES),
                "failure": _load_json(item / "failure.json", max_bytes=MAX_RESULT_BYTES),
            }
            for item in repairs
        ],
        "changed_files": _bounded_changed_files(source / "changed-files.txt"),
        "diff_tail": _tail_text(source / "diff.patch", MAX_DIFF_BYTES),
    }


def _cycle_review(directory: Path, cycle: int) -> dict[str, Any] | None:
    source = _cycle_root(directory, cycle) / "review"
    review = _load_json(source / "review.json", max_bytes=MAX_RESULT_BYTES)
    raw = _load_text_bounded(source / "reviewer.raw.md", MAX_RESULT_BYTES)
    if review is None and raw is None:
        return None
    return {
        "review": review,
        "raw": raw,
        "usage": read_usage_artifact(source / REVIEWER_USAGE_ARTIFACT),
    }


def _cycle_artifacts(directory: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every cycle of the run, each read only from its own sources."""

    records = {
        item.get("number"): item for item in (state.get("cycles") or [])
        if isinstance(item, dict)
    } if isinstance(state.get("cycles"), list) else {}
    numbers = _cycle_numbers(directory) or [1]
    cycles = []
    for number in numbers:
        record = records.get(number, {})
        cycles.append({
            "number": number,
            "kind": _cycle_kind(directory, number),
            "status": record.get("status"),
            "failure": record.get("failure"),
            "plan_raw": (
                _load_text_bounded(
                    _cycle_root(directory, number) / "correction" / "planner.raw.md",
                    MAX_RESULT_BYTES,
                )
                if number > 1 else None
            ),
            "steps": _cycle_steps(directory, number, state),
            "revision": _cycle_revision(directory, number),
            "checks": _cycle_checks(directory, number),
            "candidate": _load_json(
                _cycle_root(directory, number) / "candidate" / "commit.json",
                max_bytes=MAX_RESULT_BYTES,
            ),
            "review": _cycle_review(directory, number),
        })
    return cycles


def _step_contract(
    directory: Path, step_id: str, declared_sha256: Any
) -> dict[str, Any]:
    """Read one step contract exactly as stored, never re-rendered.

    ``matches`` tells whether these bytes are the ones hashed in
    ``implementation_bundle.json``.
    """

    try:
        with step_contract_path(directory, step_id).open("rb") as stream:
            data = stream.read(MAX_STEP_CONTRACT_BYTES + 1)
    except OSError:
        return {"text": None, "sha256": None, "matches": False, "layout": None}
    if len(data) > MAX_STEP_CONTRACT_BYTES:
        return {"text": None, "sha256": None, "matches": False, "layout": "canonical"}
    digest = hashlib.sha256(data).hexdigest()
    return {
        "text": data.decode("utf-8", errors="replace"),
        "sha256": digest,
        "matches": isinstance(declared_sha256, str) and digest == declared_sha256,
        "layout": "canonical",
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


def progress(
    runs_root: Path, run_id: str, offset: int, config: HarnessConfig | None = None,
) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise WebAPIError(400, "offset must be a non-negative integer")
    directory = _run_dir(runs_root, run_id)
    try:
        secrets = config_secret_values(config) if config is not None else ()
        path = sync_progress(directory, secrets=secrets)
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            return _read_progress(stream, min(offset, size), lambda item: _summarize_progress(item, secrets))
    except FileNotFoundError:
        return {"next_offset": 0, "events": []}
    except OSError as exc:
        raise WebAPIError(503, "progress is temporarily unavailable") from exc


def progress_tail(
    runs_root: Path, run_id: str, max_events: int = 50,
    *, secrets: tuple[str, ...] = (),
) -> list[str]:
    """Return at most the latest human-readable progress events.

    The existing bounded JSONL reader is deliberately reused so a malformed
    or very large event never turns the HTML/API read into an unbounded load.
    """

    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise WebAPIError(400, "max_events must be a non-negative integer")
    if max_events == 0:
        return []
    directory = _run_dir(runs_root, run_id)
    path = sync_progress(directory, secrets=secrets)
    return _tail_events(path, max_events, lambda item: _summarize_progress(item, secrets))


def _summarize_progress(payload: dict[str, Any], secrets: tuple[str, ...] = ()) -> str | None:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sequence"), int):
        return None
    return display_event(payload, secrets)


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


def cycle_step_progress_tail(
    run_dir: Path,
    cycle: int,
    step_id: str,
    *,
    max_events: int = STEP_EVENTS_MAX,
) -> list[str]:
    """Latest compact events of one step of one cycle.

    Reads ``cycles/NNN/implementation/steps/Sxx/agent.events.jsonl`` with the
    same bounded JSONL window as :func:`progress_tail`; tool arguments are
    never rendered.
    """

    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise WebAPIError(400, "max_events must be a non-negative integer")
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 1:
        raise WebAPIError(400, "invalid review cycle")
    if not isinstance(step_id, str) or _STEP_ID.fullmatch(step_id) is None:
        raise WebAPIError(400, "invalid step id")
    if max_events == 0:
        return []
    path = _cycle_root(Path(run_dir), cycle) / "implementation" / "steps" / step_id / "agent.events.jsonl"
    return _tail_events(path, max_events, summarize_step_event)


def approve_run(
    runs_root: Path,
    run_id: str,
    decision: str,
    *,
    config: HarnessConfig | None = None,
    final_reviewer_profile: object = None,
    semantic_reviser_profile: object = None,
    check_repair_profile: object = None,
    step_profiles: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Perform the only web mutation through the core approval API."""

    try:
        selected = ApprovalDecision(decision)
    except (TypeError, ValueError) as exc:
        raise WebAPIError(400, "decision must be APPROVE or REJECT") from exc
    # 1-2. Load state and require the approval gate of a pipeline-v2 run.
    directory = _run_dir(runs_root, run_id)
    state = _load_state(directory)
    if state.get("status") != RunStatus.AWAITING_PLAN_APPROVAL.value:
        raise WebAPIError(409, "run is not awaiting plan approval")
    if state.get("planning_protocol") != "v2":
        raise WebAPIError(409, "run is not a pipeline v2 run")
    reviser_profile = semantic_reviser_profile
    repair_profile = check_repair_profile
    semantic_revision_enabled = check_repair_required = False
    if selected is ApprovalDecision.APPROVE:
        if not is_profile_aware_run(state):
            raise WebAPIError(409, "run has no planner profile")
        if config is None or not isinstance(step_profiles, Mapping) or not isinstance(final_reviewer_profile, str):
            raise WebAPIError(400, "step profiles and final_reviewer_profile are required")
        try:
            snapshot, _ = read_run_options_for_state(directory, state)
            config = effective_run_config(config, snapshot)
        except RunOptionsError as exc:
            raise WebAPIError(409, "run options are invalid") from exc
        # A semantic reviser or check-repair field never enables a pipeline
        # implicitly: only the immutable creation snapshot decides this.
        semantic_revision_enabled = snapshot.semantic_revision_enabled
        check_repair_required = (
            snapshot.max_check_repair_attempts > 0 or snapshot.max_review_repair_cycles > 0
        )
        if any(not isinstance(value, str) for value in step_profiles.values()):
            raise WebAPIError(400, "invalid step profile field")
        if reviser_profile is not None and not semantic_revision_enabled:
            raise WebAPIError(400, "semantic_reviser_profile requires semantic revision")
        if repair_profile is not None and not check_repair_required:
            raise WebAPIError(400, "check_repair_profile requires a correction budget")
        # Each default comes from its own frozen key: the check-repair
        # profile is never derived from the reviser.
        if semantic_revision_enabled and reviser_profile is None:
            reviser_profile = snapshot.semantic_reviser_profile
        if check_repair_required and repair_profile is None:
            repair_profile = snapshot.check_repair_profile
        for name, value in (
            ("semantic_reviser_profile", reviser_profile),
            ("check_repair_profile", repair_profile),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise WebAPIError(400, f"{name} is invalid")
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
            checks_sha256=stored.get("checks_sha256"),
        )
        actual = compute_plan_identity_from_run(directory)
    except (KeyError, TypeError, ValueError, ApprovalError, OSError, UnicodeError) as exc:
        raise WebAPIError(409, "plan artifacts do not match run state") from exc
    if (actual.raw_sha256, actual.contract_sha256, actual.bundle_sha256) != (
        expected.raw_sha256, expected.contract_sha256, expected.bundle_sha256,
    ):
        raise WebAPIError(409, "plan artifacts do not match run state")
    if (directory / "check_authority.json").is_file() and actual.checks_sha256 is None:
        raise WebAPIError(409, "check authority hash is missing from the plan identity")
    if expected.checks_sha256 is not None and expected.checks_sha256 != actual.checks_sha256:
        raise WebAPIError(409, "check authority does not match the durable plan identity")

    if selected is ApprovalDecision.REJECT:
        _publish_decision(directory, selected, expected)
        return {"ok": True, "decision": selected.value}

    # 4. Freeze the requested profiles from trusted configuration only.
    try:
        bundle, _ = validate_implementation_bundle(directory)
    except (V2PlanParseError, OSError, UnicodeError) as exc:
        raise WebAPIError(409, "plan artifacts do not match run state") from exc
    expected_ids = [entry["id"] for entry in bundle["steps"]]
    if set(step_profiles) != set(expected_ids):
        raise WebAPIError(400, "missing or unknown step profile field")
    try:
        requested = resolve_execution_selection(
            config,
            planner_profile_id=state["execution"]["planner"]["profile_id"],
            plan_steps=[
                ImplementationStep(
                    entry["id"], entry["title"], ExecutionClass(entry["execution_class"]),
                    None, "", (), (), "", "", "",
                )
                for entry in bundle["steps"]
            ],
            step_profile_ids={key: value for key, value in step_profiles.items()},
            semantic_reviser_profile_id=reviser_profile if semantic_revision_enabled else None,
            check_repair_profile_id=repair_profile if check_repair_required else None,
            final_reviewer_profile_id=final_reviewer_profile,
        )
    except (ProfileError, ExecutionSelectionError) as exc:
        raise WebAPIError(400, "selected profile is invalid") from exc
    # 5. Claim the selection immutably, then read back the exact durable bytes.
    try:
        ensure_execution_selection(directory, requested)
        durable, execution_sha256 = read_execution_selection_with_sha256(directory)
        validate_execution_selection(config, durable)
    except ExecutionSelectionConflict as exc:
        raise WebAPIError(409, "a different execution selection is already recorded") from exc
    except ExecutionSelectionError as exc:
        raise WebAPIError(409, "execution selection is invalid") from exc
    if durable != requested:
        raise WebAPIError(409, "a different execution selection is already recorded")
    # 6. Bind the decision to that selection and publish it exclusively.
    identity = PlanIdentity(
        raw_sha256=expected.raw_sha256,
        contract_sha256=expected.contract_sha256,
        execution_sha256=execution_sha256,
        bundle_sha256=expected.bundle_sha256,
        checks_sha256=actual.checks_sha256,
    )
    _publish_decision(directory, selected, identity)
    # Compare-and-set: once the orchestrator has left the gate it owns the
    # state, and this write must not resurrect the approval status.
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
        "steps": [
            {"step_id": item.step_id, "implementer": asdict(item.implementer)}
            for item in selection.steps
        ],
        "final_reviewer": asdict(selection.final_reviewer),
    }
    if selection.check_repair is not None:
        state["check_repair"] = asdict(selection.check_repair)
    if selection.semantic_reviser is not None:
        state["semantic_reviser"] = asdict(selection.semantic_reviser)
    return state




def model_profiles(config: HarnessConfig) -> dict[str, Any]:
    profiles = profiles_for_config(config)
    defaults = {
        "planner": config.ui.default_planner_profile,
        "mechanical": config.routing.mechanical_profile,
        "reasoning": config.routing.reasoning_profile,
        "agentic": config.routing.agentic_profile,
        "planner_profile": config.ui.default_planner_profile,
        "final_reviewer_profile": config.ui.default_reviewer_profile,
        "semantic_reviser_profile": config.ui.default_reviser_profile,
        "check_repair_profile": config.ui.default_repair_profile,
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
    mechanical_profile: object = None,
    reasoning_profile: object = None,
    agentic_profile: object = None,
    final_reviewer_profile: object = None,
    semantic_reviser_profile: object = None,
    check_repair_profile: object = None,
    semantic_revision_enabled: object = None,
    max_check_repair_attempts: object = None,
    max_review_repair_cycles: object = None,
    decomposition: object = None,
    execution_mode_policy: object = None,
    single_step_max_mutable_paths: object = None,
    staged_step_max_mutable_paths: object = None,
    repair_scope_policy: object = None,
    repair_scope_max_added_paths: object = None,
) -> dict[str, str]:
    content = validate_spec(spec)
    try:
        def profile(value: object, name: str) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise WebAPIError(400, f"{name} is invalid")
            return value

        def optional_profile(value: object, name: str) -> str | None:
            if value == "":
                return None
            return profile(value, name)

        def boolean(value: object, name: str) -> bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value in {"enabled", "disabled"}:
                return value == "enabled"
            raise WebAPIError(400, f"{name} is invalid")

        def integer(value: object, name: str) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                raise WebAPIError(400, f"{name} is invalid")
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            raise WebAPIError(400, f"{name} is invalid")

        overrides = {
            "planner_profile": profile(planner_profile, "planner_profile"),
            "mechanical_profile": profile(mechanical_profile, "mechanical_profile"),
            "reasoning_profile": profile(reasoning_profile, "reasoning_profile"),
            "agentic_profile": profile(agentic_profile, "agentic_profile"),
            "final_reviewer_profile": profile(final_reviewer_profile, "final_reviewer_profile"),
            "semantic_reviser_profile": optional_profile(semantic_reviser_profile, "semantic_reviser_profile"),
            "check_repair_profile": optional_profile(check_repair_profile, "check_repair_profile"),
            "semantic_revision_enabled": boolean(semantic_revision_enabled, "semantic_revision_enabled"),
            "max_check_repair_attempts": integer(max_check_repair_attempts, "max_check_repair_attempts"),
            "max_review_repair_cycles": integer(max_review_repair_cycles, "max_review_repair_cycles"),
            "decomposition": decomposition,
            "execution_mode_policy": execution_mode_policy,
            "single_step_max_mutable_paths": integer(single_step_max_mutable_paths, "single_step_max_mutable_paths"),
            "staged_step_max_mutable_paths": integer(staged_step_max_mutable_paths, "staged_step_max_mutable_paths"),
            "repair_scope_policy": repair_scope_policy,
            "repair_scope_max_added_paths": integer(repair_scope_max_added_paths, "repair_scope_max_added_paths"),
        }
        overrides = {key: value for key, value in overrides.items() if value is not None}
        options = RunOptions.from_config(manager._config, **overrides)
    except RunOptionsError as exc:
        raise WebAPIError(400, str(exc)) from exc
    selected_id: str | None = None
    if run_id is not None:
        if not isinstance(run_id, str):
            raise WebAPIError(400, "invalid run id")
        selected_id = validate_run_id(run_id)
        root = manager._config.runs_root.expanduser().resolve()
        if (root / selected_id).exists():
            raise WebAPIError(409, "run already exists")
    try:
        created_id = manager.start_run(
            content, run_id=selected_id, run_options=options,
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


# -- token diagnostics, resumability and the live overview --------------------

# Worker input tokens above these levels are flagged (advisory only; a run is
# never failed because of its context usage).
CONTEXT_WARNING_INPUT_TOKENS = HIGH_WORKER_INPUT_TOKENS
CONTEXT_SEVERE_INPUT_TOKENS = 250_000
LIVE_EVENTS_MAX = 20
_LIVE_EVENT_WINDOW_BYTES = 256 * 1024
# Statuses for which the run page stops polling: terminal, or waiting for a
# human decision that needs the complete server-rendered page.
LIVE_STOP_STATUSES = frozenset({
    "committed", "published", "failed", "blocked", "plan_rejected", "interrupted",
    "waiting_human",
    "awaiting_plan_approval", "waiting_check_infrastructure",
    "waiting_scope_approval", "waiting_remote", "waiting_external",
    "waiting_contract_repair", "waiting_check_repair",
})
_CONTRACT_REPAIR_PHASES = {
    "running": "Contract repair",
    "correcting_output": "Correcting planner output",
    "output_invalid": "Correcting planner output",
    "waiting_external": "Waiting external",
    "output_correction_exhausted": "Output correction exhausted",
}
_DIAGNOSTIC_COUNTERS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens",
    "event_count", "tool_call_count",
)
_DIAGNOSTIC_LISTS = ("files_read_observed", "commands_observed")
_CHECK_FAILURE_PREFIXES = (
    "CHECK_", "DETERMINISTIC_GATE", "EMPTY_DIFF", "DIFF_TOO_LARGE", "SECRET_IN",
    "UNSCANNABLE", "UNREVIEWABLE", "HEAD_MISMATCH",
)
_PUBLISH_FAILURES = frozenset({
    "PUSH_FAILED", "BASE_MOVED_SINCE_RUN", "COMMIT_TREE_MISMATCH", "TOCTOU_FAILURE",
})
_PIPELINE_STEP_STATE = {
    "completed": "complete", "running": "running", "failed": "failed",
    "deferred": "deferred",
    "interrupted": "failed", "waiting": "waiting",
}


def context_level(usage: Any) -> str:
    """``normal``, ``warning`` (> 100k input) or ``severe`` (> 250k input)."""

    tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        return "normal"
    if tokens > CONTEXT_SEVERE_INPUT_TOKENS:
        return "severe"
    if tokens > CONTEXT_WARNING_INPUT_TOKENS:
        return "warning"
    return "normal"


def _token_diagnostics(step_dir: Path) -> dict[str, Any] | None:
    """The bounded, argument-free ``token_diagnostics.json`` of one step."""

    payload = _load_json(step_dir / TOKEN_DIAGNOSTICS_NAME, max_bytes=64 * 1024)
    if not isinstance(payload, dict):
        return None
    result: dict[str, Any] = {}
    for key in _DIAGNOSTIC_COUNTERS:
        value = payload.get(key)
        result[key] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
    for key in _DIAGNOSTIC_LISTS:
        value = payload.get(key)
        items = value if isinstance(value, list) else []
        result[key] = [item[:300] for item in items[:100] if isinstance(item, str)]
    return result


def _plan_recovery_payload(directory: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    info = plan_recovery_info(directory, state)
    return {
        "eligible": info.eligible,
        "reason": info.reason,
        "recovered": read_plan_recovery_record(directory) is not None,
        "max_bytes": MAX_REPLACEMENT_PLAN_BYTES,
    }


def _resume_payload(directory: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    info = resume_info(directory, state)
    label = info.label
    if info.phase == "plan_approval" and read_plan_recovery_record(directory) is not None:
        # Never "Retry planner": the plan authority is the operator's.
        label = "Resume recovered plan approval"
    return {
        "resumable": info.resumable, "phase": info.phase, "label": label,
        "expected_tree": info.expected_tree, "review_cycle": info.review_cycle,
        "step_id": info.step_id, "reason": info.reason,
        "operation": info.operation,
    }


def contract_repair_view(
    state: Mapping[str, Any], resume: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Concise live phase of a pending step contract repair, if any.

    A genuine ``waiting_human`` decision never renders as a contract repair:
    only a proven recoverable slot (``resume.operation``) does.
    """

    repair = state.get("contract_repair")
    status = str(state.get("status") or "")
    if not isinstance(repair, Mapping) or repair.get("status") in {"completed", "superseded"}:
        return None
    if status == "waiting_human":
        if resume.get("operation") != "contract_repair":
            return None
        phase = "Correcting planner output"
    elif status == "waiting_contract_repair":
        phase = "Output correction exhausted"
    elif status == "waiting_scope_approval":
        phase = "Waiting scope approval"
    elif status in {"contract_repairing", "waiting_external", "interrupted", "failed"}:
        phase = _CONTRACT_REPAIR_PHASES.get(str(repair.get("status") or ""), "Contract repair")
    else:
        return None

    def count(key: str) -> int | None:
        value = repair.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    step = repair.get("step_id")
    step_id = step if isinstance(step, str) and _STEP_ID.fullmatch(step) else None
    corrections, limit = count("output_correction_attempt"), count("output_correction_limit")
    text = phase + (f" · {step_id}" if step_id else "")
    if corrections and limit is not None:
        text += f" · attempt {corrections} / {limit}"
    repair_id = repair.get("repair_id")
    return {
        "phase": phase, "text": text, "step_id": step_id,
        "repair_id": str(repair_id)[:80] if isinstance(repair_id, str) else None,
        "status": str(repair.get("status") or "")[:40],
        "output_correction_attempt": corrections, "output_correction_limit": limit,
        "planner_transport_attempt": count("planner_transport_attempt"),
    }


def _step_statuses(directory: Path, cycle: int, state: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Light (id, status) pairs of one cycle for the pipeline; no event reads."""

    root = _cycle_root(directory, cycle)
    ids = [
        entry["id"]
        for entry in _bundle_entries(_cycle_contracts_dir(directory, cycle) / "implementation_bundle.json")
    ]
    raw_live = state.get("steps") if _state_cycle(state) == cycle else None
    live = {
        item["id"]: item.get("status") for item in (raw_live if isinstance(raw_live, list) else [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
        and _STEP_ID.fullmatch(item["id"]) is not None
    }
    result: list[tuple[str, str]] = []
    for step_id in ids or list(live):
        status = live.get(step_id)
        if not isinstance(status, str):
            record = _load_json(
                root / "implementation" / "steps" / step_id / "step.json", max_bytes=MAX_RESULT_BYTES,
            )
            status = _DURABLE_STEP_STATUS.get(
                str(record.get("status")) if isinstance(record, dict) else "", "waiting"
            )
        result.append((step_id, status))
    return result


_STEP_PHASES = frozenset({"implement_step", "review_implementation"})
_GATE_PHASES = frozenset({"deterministic_gate", "check_repair"})


def run_pipeline(
    directory: Path,
    state: Mapping[str, Any],
    config: HarnessConfig | None = None,
    resume: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Server-computed pipeline states: complete/running/failed/waiting/resumable.

    Every cycle is rendered from its own durable directory; the number of
    cycles is never assumed.
    """

    status = str(state.get("status") or "")
    failure = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
    reason = str(failure.get("reason") or "") if failure else ""
    failed = status in {"failed", "interrupted"}
    current = _state_cycle(state)
    planner = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
    resumable = isinstance(resume, Mapping) and bool(resume.get("resumable"))
    resume_phase = resume.get("phase") if resumable else None
    resume_cycle = resume.get("review_cycle") if resumable else None
    snapshot = state.get("run_options") if isinstance(state.get("run_options"), Mapping) else {}
    pipeline_snapshot = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), Mapping) else {}
    revision_enabled = bool(pipeline_snapshot.get("semantic_revision_enabled"))
    items: list[dict[str, str]] = []

    def add(key: str, label: str, value: str) -> None:
        items.append({"key": key, "label": label, "state": value})

    def phase_state(
        done: bool, phases: frozenset[str], running_statuses: frozenset[str],
        failure_prefixes: tuple[str, ...], cycle: int,
    ) -> str:
        if done:
            return "complete"
        if resume_phase in phases and resume_cycle == cycle:
            return "resumable"
        if cycle == current and failed and reason.startswith(failure_prefixes):
            return "failed"
        if cycle == current and not failed and status in running_statuses:
            return "running"
        return "waiting"

    decision = planner.get("decision")
    recovered = read_plan_recovery_record(directory) is not None
    if resume_phase in {"context", "planner"}:
        add("planner", "Planner", "resumable")
    elif decision == "READY":
        add("planner", "Planner · operator recovery" if recovered else "Planner", "complete")
    elif status == "waiting_human":
        add("planner", "Planner · operator decision required", "waiting")
    elif decision == "BLOCKED" or status == "blocked" or failed:
        add("planner", "Planner", "failed")
    elif status in {"created", "planning"}:
        add("planner", "Planner", "running")
    else:
        add("planner", "Planner", "waiting")

    approval = _load_json(directory / "plan_approval.json", max_bytes=16 * 1024)
    approval_decision = approval.get("decision") if isinstance(approval, dict) else None
    if approval_decision == "APPROVE" or (
        approval_decision is None and decision == "READY"
        and (directory / "execution_selection.json").exists()
        and status != AWAITING_APPROVAL
    ):
        add("approval", "Approval", "complete")
    elif approval_decision == "REJECT" or status == "plan_rejected":
        add("approval", "Approval", "failed")
    elif resume_phase == "plan_approval":
        add("approval", "Recovered plan — awaiting approval" if recovered else "Approval", "resumable")
    elif status == AWAITING_APPROVAL:
        add("approval", "Recovered plan — awaiting approval" if recovered else "Approval", "running")
    elif failed and reason.startswith(("PLAN_APPROVAL", "EXECUTION_SELECTION")):
        add("approval", "Approval", "failed")
    else:
        add("approval", "Approval", "waiting")

    for cycle in _cycle_numbers(directory) or [1]:
        root = _cycle_root(directory, cycle)
        label = f"{cycle:03d}"
        if cycle > 1:
            planned = (root / "correction" / "implementation_bundle.json").is_file()
            value = phase_state(
                planned, frozenset({"review_replan"}), frozenset({"planning"}),
                ("REPAIR", "PLANNER", "LLM_FAILURE", "HUMAN_REQUIRED"), cycle,
            )
            if status == "waiting_scope_approval" and cycle == current:
                value = "resumable"
            add(f"c{cycle}-planner", f"{label} · correction planner ({_cycle_kind(directory, cycle)})", value)
        steps = _step_statuses(directory, cycle, state)
        for step_id, step_status in steps:
            value = _PIPELINE_STEP_STATE.get(step_status, "waiting")
            if value in {"failed", "running"} and resume_phase in _STEP_PHASES and resume_cycle == cycle:
                value = "resumable"
            add(f"c{cycle}-{step_id}", f"{label} · {step_id}", value)
        if not steps:
            add(f"c{cycle}-steps", f"{label} · steps", "waiting")
        if revision_enabled:
            report = _load_json(root / "semantic-revision" / "report.json", max_bytes=MAX_RESULT_BYTES)
            add(f"c{cycle}-revision", f"{label} · semantic revision", phase_state(
                isinstance(report, dict) and report.get("status") in {"COMPLETED", "NO_CHANGE"},
                frozenset({"semantic_revision"}),
                frozenset({"pre_revision_validating", "revising"}),
                ("AGENT_", "REVISION_SCOPE", "SEMANTIC_REVISER"), cycle,
            ))
        gate = _cycle_gate_dir(directory, cycle)
        evidence = _load_json(gate / "evidence.json", max_bytes=MAX_RESULT_BYTES + MAX_DIFF_BYTES * 8) if gate else None
        add(f"c{cycle}-checks", f"{label} · checks", phase_state(
            isinstance(evidence, dict) and evidence.get("deterministic_passed") is True,
            _GATE_PHASES, frozenset({"validating", "revalidating", "revising"}),
            (*_CHECK_FAILURE_PREFIXES, "REVISION_SCOPE"), cycle,
        ))
        review = _load_json(root / "review" / "review.json", max_bytes=MAX_RESULT_BYTES)
        add(f"c{cycle}-reviewer", f"{label} · reviewer", phase_state(
            isinstance(review, dict),
            frozenset({"candidate_ready", "candidate_push", "final_review"}),
            frozenset({"approved", "reviewing"}),
            ("REVIEW", "PUSH_FAILED"), cycle,
        ))

    publish_enabled = config.publish.enabled if config is not None else True
    fast_forward = config is not None and config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
    label = (
        f"Publish {config.base_ref}" if fast_forward and config is not None
        else "Publish run branch" if publish_enabled else "Commit"
    )
    if status in {"published", "committed"}:
        value = "complete"
    elif resume_phase == "publish":
        value = "resumable"
    elif failed and reason in _PUBLISH_FAILURES:
        value = "failed"
    elif status == "publishing":
        value = "running"
    else:
        value = "waiting"
    add("publish", label, value)
    return items


def _current_and_next(items: list[dict[str, str]], state: Mapping[str, Any]) -> tuple[str, str]:
    status = str(state.get("status") or "")
    if status == "published":
        return "Published", "—"
    if status == "committed":
        return "Committed", "—"
    index = next((i for i, item in enumerate(items) if item["state"] == "running"), None)
    if index is not None:
        current = items[index]["label"]
    else:
        index = next(
            (i for i, item in enumerate(items) if item["state"] in {"failed", "resumable"}), None
        )
        if index is None:
            completed = [i for i, item in enumerate(items) if item["state"] == "complete"]
            index = completed[-1] if completed else -1
            current = "—"
        else:
            current = items[index]["label"]
    following = next(
        (item["label"] for item in items[index + 1:] if item["state"] in {"waiting", "resumable"}),
        "—",
    )
    return current, following


def _usage_pair(value: Any) -> dict[str, int]:
    usage = normalize_usage(value)
    return {
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
    }


def _token_totals(directory: Path) -> dict[str, dict[str, int]]:
    usage = phase_usage_summary(directory)
    implementer = usage.get("implementer") if isinstance(usage.get("implementer"), dict) else {}
    planner = add_usage((
        normalize_usage(usage.get("planner")), normalize_usage(usage.get("correction_planner")),
    ))
    return {
        "planner": _usage_pair(planner),
        "implementer": _usage_pair(implementer.get("total")),
        "check_repair": _usage_pair(usage.get("check_repair")),
        "semantic_reviser": _usage_pair(usage.get("semantic_reviser")),
        "final_reviewer": _usage_pair(usage.get("final_reviewer")),
        "total": _usage_pair(usage.get("grand_total")),
    }


def publish_target(config: HarnessConfig | None, state: Mapping[str, Any]) -> tuple[str, str]:
    """(short target, one-line description) shown before approval and after."""

    if config is None:
        publish = state.get("publish") if isinstance(state.get("publish"), Mapping) else {}
        target = publish.get("target") or publish.get("branch") or "—"
        return str(target), str(target)
    if not config.publish.enabled:
        return "none", "local commit only (publication disabled)"
    if config.publish.mode == PublishMode.FAST_FORWARD_BASE.value:
        return config.base_ref, f"{config.base_ref} via safe fast-forward after final PASS"
    return "run branch", (
        f"run branch harness/<plan>/<run-id> on {config.publish.remote} after final PASS"
    )


def run_overview(
    directory: Path, state: Mapping[str, Any], config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Status/action-oriented summary shared by the page and the live endpoint."""

    resume = _resume_payload(directory, state)
    pipeline = run_pipeline(directory, state, config, resume)
    current, following = _current_and_next(pipeline, state)
    repair = contract_repair_view(state, resume)
    if repair is not None:
        current = repair["text"]
    planner = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
    steps = planner.get("steps") if isinstance(planner.get("steps"), list) else []
    mode = planner.get("execution_mode") or "—"
    target, target_detail = publish_target(config, state)
    return {
        "resume": resume,
        "pipeline": pipeline,
        "contract_repair": repair,
        "current_label": current,
        "next_label": following,
        "execution_label": f"{mode} · {len(steps)} step{'s' if len(steps) != 1 else ''}",
        "publish_target": target,
        "publish_target_detail": target_detail,
        "token_totals": _token_totals(directory),
    }


def _recent_events(
    path: Path,
    max_events: int,
    summarize: Callable[[dict[str, Any]], str | None] = summarize_step_event,
) -> list[str]:
    """Latest summaries from only the last bounded window of a JSONL file."""

    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            start = max(0, size - _LIVE_EVENT_WINDOW_BYTES)
            stream.seek(start)
            data = stream.read(_LIVE_EVENT_WINDOW_BYTES)
    except OSError:
        return []
    if start > 0:
        newline = data.find(b"\n")
        data = data[newline + 1:] if newline >= 0 else b""
    complete = data[: data.rfind(b"\n") + 1]
    return _summaries(complete, summarize)[-max_events:]


def _live_events(
    directory: Path, state: Mapping[str, Any], secrets: tuple[str, ...] = (),
) -> list[str]:
    # The durable projection covers waiting_external (where current_step is
    # deliberately cleared), terminal runs, and resumed worker activity.
    try:
        return progress_tail(directory.parent, directory.name, LIVE_EVENTS_MAX, secrets=secrets)
    except WebAPIError:
        pass
    cycle = _state_cycle(state)
    step = state.get("current_step")
    if isinstance(step, str) and _STEP_ID.fullmatch(step):
        path = _cycle_root(directory, cycle) / "implementation" / "steps" / step / "agent.events.jsonl"
        return _recent_events(path, LIVE_EVENTS_MAX)
    if state.get("status") == "revising":
        source = _cycle_root(directory, cycle) / "semantic-revision"
        return _recent_events(source / "agent.events.jsonl", LIVE_EVENTS_MAX)
    return []


def live_status(
    runs_root: Path, run_id: str, config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Small bounded live payload: no prompt, diff, secret or tool argument."""

    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    state = _load_state(directory)
    overview = run_overview(directory, state, config)
    status = str(state.get("status") or "")
    failure = state.get("failure")
    failure_payload = (
        {
            "reason": str(failure.get("reason") or "")[:120],
            "detail": " ".join(str(failure.get("detail") or "").split())[:300],
        }
        if isinstance(failure, dict) else None
    )
    step = state.get("current_step")
    current_step = step if isinstance(step, str) and _STEP_ID.fullmatch(step) else None
    cycle = _state_cycle(state)
    return {
        "run_id": safe_id,
        "status": status,
        "updated_at": state.get("updated_at"),
        "cycle": cycle,
        "phase": f"{status}:{current_step}" if current_step else status,
        "current_step": current_step,
        "failure": failure_payload,
        "resumable": overview["resume"]["resumable"],
        "resume_phase": overview["resume"]["phase"],
        "resume_label": overview["resume"]["label"],
        "contract_repair": overview["contract_repair"],
        "running": status not in LIVE_STOP_STATUSES,
        "token_totals": overview["token_totals"],
        "progress_events": _live_events(
            directory, state, config_secret_values(config) if config is not None else (),
        ),
        "pipeline": overview["pipeline"],
        "current_label": overview["current_label"],
        "next_label": overview["next_label"],
    }


def resume_run_request(manager: RunManager, runs_root: Path, run_id: str) -> dict[str, Any]:
    """Start the only resume mutation for the same run id."""

    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    if not resume_info(directory, _load_state(directory)).resumable:
        raise WebAPIError(409, "run is not resumable")
    try:
        manager.resume_run(safe_id)
    except RunResumeNotAllowedError as exc:
        raise WebAPIError(409, "run is not resumable") from exc
    except RunCollisionError as exc:
        raise WebAPIError(409, "run is already active") from exc
    except RunCapacityError as exc:
        raise WebAPIError(409, "maximum active runs reached") from exc
    except RunManagerError as exc:
        raise WebAPIError(503, "run could not be resumed") from exc
    return {"ok": True, "run_id": safe_id, "location": f"/runs/{safe_id}"}


def recover_plan_request(
    manager: RunManager, runs_root: Path, run_id: str, replacement: object,
) -> dict[str, Any]:
    """REPLACE PLAN: publish an operator META PLAN v2, then await approval.

    Only the raw replacement text is accepted: SPEC, context, BASE, run
    options and catalogues stay those of the run.  No planner is called.
    """

    directory = _run_dir(runs_root, run_id)
    safe_id = validate_run_id(run_id)
    try:
        text = validate_replacement_text(replacement)
    except PlanRecoveryError as exc:
        raise WebAPIError(400, str(exc)) from exc
    info = plan_recovery_info(directory, _load_state(directory))
    if not info.eligible:
        raise WebAPIError(409, f"plan recovery refused: {info.reason}")
    try:
        manager.recover_plan(safe_id, text)
    except RunPlanRecoveryError as exc:
        raise WebAPIError(400, f"plan recovery refused: {exc}") from exc
    except RunResumeNotAllowedError as exc:
        raise WebAPIError(409, "recovered run could not be resumed") from exc
    except RunCollisionError as exc:
        raise WebAPIError(409, "run is already active") from exc
    except RunCapacityError as exc:
        raise WebAPIError(409, "maximum active runs reached") from exc
    except RunManagerError as exc:
        raise WebAPIError(503, "plan could not be recovered") from exc
    return {"ok": True, "run_id": safe_id, "location": f"/runs/{safe_id}"}


def approve_repair_scope(
    runs_root: Path, run_id: str, decision: str,
) -> dict[str, Any]:
    """Record APPROVE/REJECT for the exact planner-derived correction delta."""

    try:
        selected = ApprovalDecision(decision)
    except (TypeError, ValueError) as exc:
        raise WebAPIError(400, "decision must be APPROVE or REJECT") from exc
    directory = _run_dir(runs_root, run_id)
    state = _load_state(directory)
    if state.get("status") != RunStatus.WAITING_SCOPE_APPROVAL.value:
        raise WebAPIError(409, "run is not waiting for scope approval")
    correction = _cycle_root(directory, _state_cycle(state)) / "correction"
    delta_path = correction / "scope_delta.json"
    approval_dir = correction
    state_delta = state.get("scope_delta") if isinstance(state.get("scope_delta"), dict) else {}
    requested_artifact = state_delta.get("approval_artifact")
    if isinstance(requested_artifact, str) and requested_artifact:
        candidate = (directory / requested_artifact).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exc:
            raise WebAPIError(409, "scope approval artifact is invalid") from exc
        if candidate.is_file():
            delta_path, approval_dir = candidate, candidate.parent
    if not delta_path.is_file():
        candidates = sorted(
            (
                path for path in (_cycle_root(directory, _state_cycle(state)) / "implementation" / "steps").glob(
                    "*/contract_repairs/*/scope_delta.json"
                ) if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        candidates.extend(
            path for path in (
                _cycle_root(directory, _state_cycle(state)) / "semantic-revision" / "scope_requests"
            ).glob("*/scope_delta.json") if path.is_file()
        )
        candidates.sort(key=lambda path: path.stat().st_mtime_ns)
        if candidates:
            delta_path = candidates[-1]
            approval_dir = delta_path.parent
    try:
        delta_hash = hashlib.sha256(delta_path.read_bytes()).hexdigest()
        if not isinstance(_load_json(delta_path, max_bytes=256 * 1024), dict):
            raise ValueError
        existing = read_scope_approval(approval_dir, expected_sha256=delta_hash)
        if existing is not None:
            raise WebAPIError(409, "scope approval already exists")
        write_scope_approval(approval_dir, decision=selected,
                             scope_delta_sha256=delta_hash, source="web-ui")
    except WebAPIError:
        raise
    except (OSError, ValueError, ApprovalError) as exc:
        raise WebAPIError(409, "scope delta is invalid") from exc
    if selected is ApprovalDecision.REJECT:
        RunStateStore(directory / "state.json").update(
            status=RunStatus.FAILED, failure={"reason": "HUMAN_REQUIRED", "detail": "repair scope rejected"}
        )
    return {"ok": True, "decision": selected.value, "scope_delta_sha256": delta_hash}


AWAITING_APPROVAL = RunStatus.AWAITING_PLAN_APPROVAL.value


__all__ = [
    "ARTIFACT_ALLOWLIST",
    "CONTEXT_SEVERE_INPUT_TOKENS",
    "CONTEXT_WARNING_INPUT_TOKENS",
    "LIVE_STOP_STATUSES",
    "HIGH_WORKER_INPUT_TOKENS",
    "context_level",
    "live_status",
    "publish_target",
    "recover_plan_request",
    "resume_run_request",
    "approve_repair_scope",
    "run_overview",
    "run_pipeline",
    "MAX_SPEC_BYTES",
    "MAX_STEP_CONTRACT_BYTES",
    "MAX_DIFF_BYTES",
    "OVERSIZED_EVENT",
    "PROGRESS_MAX_BYTES",
    "PROGRESS_MAX_EVENT_BYTES",
    "PROGRESS_MAX_SKIP_BYTES",
    "WebAPIError",
    "approve_run",
    "create_run",
    "cycle_step_progress_tail",
    "model_profiles",
    "get_run",
    "list_runs",
    "progress",
    "progress_tail",
    "validate_run_id",
    "validate_spec",
]
