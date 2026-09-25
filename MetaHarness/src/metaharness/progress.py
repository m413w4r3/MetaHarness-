"""Presentation-only, redacted projection of durable pipeline activity."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent.events import parse_event, summarize_step_event
from .redaction import REDACTED, redact

PROGRESS_RELATIVE_PATH = "progress/events.v1.jsonl"
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 20_000
MAX_MESSAGE = 300
_SOURCE_CACHE: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b((?:DEEPSEEK|BRIDGE|OPENAI|ANTHROPIC|API)_API_KEY\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]+"),
)


def _safe(text: Any, secrets: tuple[str, ...] = ()) -> str:
    value = " ".join(str(text or "").split())[:MAX_MESSAGE]
    value = redact(value, secrets)
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub(lambda match: match.group(1) + REDACTED, value)
    return value


def _category(event: str, phase: str | None) -> str:
    value = f"{event} {phase or ''}".casefold()
    if any(word in value for word in ("check", "validat", "test", "lint")):
        return "check"
    if "transport" in value:
        return "planner"
    if "repair" in value or "recovery" in value or "resume" in value or "waiting" in value:
        return "recovery"
    if "review" in value:
        return "review"
    if "plan" in value or "planner" in value:
        return "planner"
    if "step" in value or phase == "implementation":
        return "step"
    if any(word in value for word in ("git", "candidate", "remote")):
        return "git"
    return "system"


def _trace_message(payload: dict[str, Any], secrets: tuple[str, ...]) -> tuple[str, str, str] | None:
    event = str(payload.get("event") or "event")
    phase = str(payload.get("phase") or "")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    category = _category(event, phase)
    labels = {
        "run.created": "run started", "run.completed": "run completed", "run.failed": "hard failure",
        "planning.started": "planning started", "planning.completed": "planning completed",
        "plan.corrected": "plan correction", "approval.granted": "approval granted",
        "step.started": "step started", "step.completed": "step completed",
        "step.no_change": "step no-change", "step.failed": "step failed",
        "contract.mismatch": "contract mismatch", "contract_repair.started": "contract repair started",
        "contract_repair.waiting_external": "contract repair waiting external",
        "contract_repair.resumed": "contract repair resumed", "contract_repair.completed": "contract repair completed",
        "contract_repair.output_invalid": "contract repair planner output invalid",
        "contract_repair.output_correction.started": "contract repair output correction",
        "contract_repair.output_correction.exhausted": "contract repair output correction exhausted",
        "check.started": "check started", "check.completed": "check completed",
        "review.started": "review started", "review.completed": "review completed",
        "candidate.created": "candidate created", "resume.started": "resume",
    }
    message = labels.get(event, event.replace(".", " ").replace("_", " "))
    if event == "transport.http_response" and isinstance(data.get("http_status"), int):
        message = f"HTTP {data['http_status']}"
    elif event.startswith("transport."):
        message = event.removeprefix("transport.").replace("_", " ")
    details = []
    for key in ("check_id", "name", "status", "result", "reason", "detail", "action", "attempt", "http_status", "exit_code"):
        value = data.get(key)
        if isinstance(value, (str, int, float)) and value != "":
            details.append(f"{key}={value}")
    if details:
        message += ": " + ", ".join(details)
    lowered = message.casefold()
    level = "error" if any(word in lowered for word in ("failed", "failure", "mismatch", "unavailable", "timeout", "http 5")) else "info"
    if "waiting" in lowered or "unavailable" in lowered:
        level = "warning"
    cycle = payload.get("cycle")
    step = payload.get("step_id")
    return category, level, _safe(message, secrets)


def _timestamp(value: Any) -> str:
    if isinstance(value, str) and len(value) <= 64:
        return value
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sources(run_dir: Path) -> list[Path]:
    paths = [run_dir / "trace/events.v1.jsonl"]
    for pattern in (
        "cycles/*/implementation/steps/*/agent.events.jsonl",
        "cycles/*/semantic-revision/agent.events.jsonl",
        "cycles/*/check-repair/**/agent.events.jsonl",
        "semantic-revision/agent.events.jsonl",
        "check-repair/**/agent.events.jsonl",
    ):
        paths.extend(run_dir.glob(pattern))
    paths.extend(p for p in (run_dir / "agent.events.jsonl",) if p.exists())
    return sorted(set(paths), key=lambda p: str(p.relative_to(run_dir)))


def _state_failure(run_dir: Path, secrets: tuple[str, ...]) -> list[dict[str, Any]]:
    path = run_dir / "state.json"
    try:
        with path.open("rb") as stream:
            state = json.loads(stream.read(1024 * 1024 + 1))
        if not isinstance(state, dict):
            return []
        failure = state.get("failure") if isinstance(state, dict) else None
    except (OSError, UnicodeError, ValueError):
        return []
    if not isinstance(failure, dict):
        return []
    reason = _safe(failure.get("reason") or "run failed", secrets)
    detail = _safe(failure.get("detail"), secrets)
    message = f"{reason}: {detail}" if detail else reason
    context: list[str] = []
    step = state.get("current_step")
    if isinstance(step, str) and re.fullmatch(r"S\d{2,3}", step):
        context.append(step)
    recovery = state.get("recovery")
    if isinstance(recovery, dict):
        for key in ("operation", "status", "action"):
            val = recovery.get(key)
            if isinstance(val, str):
                context.append(_safe(val, secrets))
    stable = hashlib.sha256(json.dumps(failure, sort_keys=True, default=str).encode()).hexdigest()
    return [{
        "schema_version": 1, "timestamp": _timestamp(state.get("updated_at")),
        "level": "error", "category": "recovery", "phase": "failure",
        "cycle": state.get("cycle") if isinstance(state.get("cycle"), int) else None,
        "step_id": step if isinstance(step, str) and re.fullmatch(r"S\d{2,3}", step) else None,
        "message": _safe(" · ".join(context + [message]), secrets), "source_id": f"state.failure:{stable}",
    }]


def _project_source(run_dir: Path, path: Path, secrets: tuple[str, ...]) -> list[dict[str, Any]]:
    relative = str(path.relative_to(run_dir))
    cache_key = str(run_dir.resolve()) + ":" + relative + ":" + hashlib.sha256("\0".join(secrets).encode()).hexdigest()
    try:
        stat = path.stat()
        cached = _SOURCE_CACHE.get(cache_key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    except OSError:
        return []
    raw = raw[:MAX_SOURCE_BYTES]
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        if not line or len(line) > 64 * 1024:
            continue
        payload = parse_event(line.decode("utf-8", errors="replace"))
        if payload is None:
            continue
        identity = hashlib.sha256(relative.encode() + b"\0" + str(index).encode() + b"\0" + line).hexdigest()
        if relative == "trace/events.v1.jsonl":
            mapped = _trace_message(payload, secrets)
            if mapped is None:
                continue
            category, level, message = mapped
            phase = payload.get("phase") if isinstance(payload.get("phase"), str) else None
            cycle = payload.get("cycle") if isinstance(payload.get("cycle"), int) else None
            step = payload.get("step_id") if isinstance(payload.get("step_id"), str) else None
            timestamp = _timestamp(payload.get("timestamp"))
        else:
            event_type = str(payload.get("type") or "").casefold()
            if any(part in event_type for part in ("reasoning", "chain_of_thought", "hidden")):
                continue
            summary = summarize_step_event(payload)
            if not summary:
                continue
            category, level, message = "step", "info", _safe(summary, secrets)
            match = re.search(r"(?:/steps/|/)(S\d{2,3})(?:/|$)", relative)
            step = match.group(1) if match else None
            phase = "implementation" if "/steps/" in relative else "revision"
            cycle_match = re.search(r"cycles/(\d+)", relative)
            cycle = int(cycle_match.group(1)) if cycle_match else None
            timestamp = _timestamp(payload.get("timestamp") or payload.get("created_at"))
        rows.append({
            "schema_version": 1, "timestamp": timestamp, "level": level,
            "category": category, "phase": phase, "cycle": cycle, "step_id": step,
            "message": message, "source_id": identity,
        })
    _SOURCE_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, rows)
    return rows


def sync_progress(run_dir: Path, *, secrets: tuple[str, ...] = ()) -> Path:
    """Append newly observed safe summaries; repeated reads are idempotent."""
    directory = run_dir / "progress"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events.v1.jsonl"
    lock_path = directory / "events.v1.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        known: set[str] = set()
        last = 0
        if path.is_file():
            valid = bytearray()
            with path.open("rb") as stream:
                raw_existing = stream.read(MAX_EVENTS * 2048)
            # Drop a crash-truncated tail, then keep only the monotone valid
            # prefix. This presentation artifact can always be reconstructed.
            complete = raw_existing[: raw_existing.rfind(b"\n") + 1]
            for line in complete.splitlines(keepends=True):
                try:
                    record = json.loads(line)
                    sequence = record.get("sequence") if isinstance(record, dict) else None
                    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= last:
                        break
                except (UnicodeError, ValueError):
                    break
                last = sequence
                if isinstance(record.get("source_id"), str):
                    known.add(record["source_id"])
                valid.extend(line)
            if len(valid) != len(raw_existing):
                with path.open("wb") as stream:
                    stream.write(valid)
                    stream.flush()
                    os.fsync(stream.fileno())
        projected = [row for source in _sources(run_dir) for row in _project_source(run_dir, source, secrets)]
        projected.extend(_state_failure(run_dir, secrets))
        rows = [row for row in projected if row["source_id"] not in known]
        if len(known) + len(rows) > MAX_EVENTS:
            available = max(0, MAX_EVENTS - len(known))
            rows = rows[-available:] if available else []
        if rows:
            with path.open("ab") as stream:
                for row in rows:
                    last += 1
                    row["sequence"] = last
                    stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        return path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def display_event(payload: dict[str, Any], secrets: tuple[str, ...] = ()) -> str:
    timestamp = str(payload.get("timestamp") or "")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        clock = parsed.astimezone().strftime("%H:%M:%S")
    except ValueError:
        clock = "--:--:--"
    category = str(payload.get("category") or "system")
    step = payload.get("step_id")
    prefix = f" [{step}]" if isinstance(step, str) and re.fullmatch(r"S\d{2,3}", step) else ""
    return f"{clock} [{category}]{prefix} {_safe(payload.get('message'), secrets)}"
