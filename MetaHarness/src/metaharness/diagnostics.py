"""Bounded, redacted, consolidated diagnostics for one durable run.

This module is intentionally a read-only projection of run artifacts.  The
artifacts remain authoritative; this report is only a convenient hand-off for
debugging a run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .agent.events import parse_event, summarize_step_event
from .config import HarnessConfig
from .plan_recovery import PLAN_RECOVERY_ARTIFACT, PLAN_SOURCE_OPERATOR, plan_source
from .profiles import profiles_for_config, safe_profile_metadata
from .redaction import config_secret_values, redact
from .result import atomic_write_text
from .resume import ResumeCheckpointError, resume_info, read_checkpoint_record
from .run_options import RunOptionsError, read_run_options_with_sha256
from .step_ids import is_step_id
from .usage import normalize_usage, phase_usage_summary, read_usage_artifact

REPORT_SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 128 * 1024
MAX_PLANNER_REQUEST_BYTES = 256 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_EVENT_SCAN_BYTES = 256 * 1024
MAX_EVENT_BYTES = 1 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024
DIAGNOSTICS_NAME = "diagnostics.md"
DIAGNOSTICS_ERROR_NAME = "diagnostics.error.txt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


_AUTHORIZATION_LINE = re.compile(r"(?im)^(\s*authorization\s*:\s*).*$")
_AUTHORIZATION_JSON = re.compile(r'(?i)("authorization"\s*:\s*)"[^"]*"')
_AUTHORIZATION_INLINE = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\"']+")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|bridge[_-]?(?:credential|token)|access[_-]?token)\s*[:=]\s*)([^\s,;]+)"
)


def _clean(text: str, secrets: tuple[str, ...]) -> str:
    """Apply configured redaction and remove credential-shaped log fields."""

    text = redact(text, secrets)
    text = _AUTHORIZATION_LINE.sub(r"\1[REDACTED]", text)
    text = _AUTHORIZATION_JSON.sub(r'\1"[REDACTED]"', text)
    text = _AUTHORIZATION_INLINE.sub(r"\1[REDACTED]", text)
    return _CREDENTIAL_ASSIGNMENT.sub(r"\1[REDACTED]", text)


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class _Artifact:
    path: Path
    relative: str
    size: int | None
    sha256: str | None
    exists: bool


def _artifact(run_dir: Path, relative: str) -> _Artifact:
    path = run_dir / relative
    try:
        size = path.stat().st_size
    except OSError:
        return _Artifact(path, relative, None, None, False)
    return _Artifact(path, relative, size, _sha256(path), True)


def _read_bounded(path: Path, limit: int, secrets: tuple[str, ...], *, tail: bool = False) -> tuple[str, int | None, bool]:
    """Read at most ``limit`` bytes (plus a small redaction overlap)."""

    try:
        size = path.stat().st_size
        # Keep enough context for configured exact secrets and for the
        # existing credential-shaped redactors (Authorization/API-key).
        overlap = max(max((len(value.encode("utf-8")) for value in secrets), default=0), 4096)
        with path.open("rb") as stream:
            if tail:
                visible_start = max(0, size - limit)
                stream.seek(max(0, visible_start - overlap))
                data = stream.read(size - max(0, visible_start - overlap))
            else:
                data = stream.read(limit + overlap)
    except OSError:
        return "", None, False
    truncated = size > limit
    # Redact the extended buffer first.  Slicing the raw read before this
    # point could expose the visible half of a secret crossing the cutoff.
    cleaned = _clean(data.decode("utf-8", errors="replace"), secrets)
    cleaned_bytes = cleaned.encode("utf-8", errors="replace")
    bounded = cleaned_bytes[-limit:] if tail else cleaned_bytes[:limit]
    text = bounded.decode("utf-8", errors="ignore")
    if truncated:
        text += f"\n[TRUNCATED: original {size} bytes]"
    return text, size, truncated


def _artifact_header(item: _Artifact) -> str:
    lines = [f"Artifact: {item.relative}"]
    if not item.exists:
        lines.append("Status: missing")
    else:
        lines.append(f"Size: {item.size} bytes")
        if item.sha256:
            lines.append(f"SHA256: {item.sha256}")
    return "\n".join(lines) + "\n"


def _artifact_text(run_dir: Path, relative: str, secrets: tuple[str, ...], limit: int = MAX_ARTIFACT_BYTES, *, tail: bool = False) -> str:
    item = _artifact(run_dir, relative)
    result = _artifact_header(item)
    if not item.exists:
        return result
    text, _size, _truncated = _read_bounded(item.path, limit, secrets, tail=tail)
    return result + "Content:\n" + text + "\n"


def _artifact_json(run_dir: Path, relative: str, secrets: tuple[str, ...], limit: int = MAX_ARTIFACT_BYTES) -> str:
    item = _artifact(run_dir, relative)
    result = _artifact_header(item)
    if not item.exists:
        return result
    text, _size, _truncated = _read_bounded(item.path, limit, secrets)
    return result + "JSON:\n" + text + "\n"


def _safe_json_artifact(run_dir: Path, relative: str, secrets: tuple[str, ...], allowed: Iterable[str], limit: int = MAX_ARTIFACT_BYTES) -> str:
    item = _artifact(run_dir, relative)
    result = _artifact_header(item)
    if not item.exists:
        return result
    text, _size, truncated = _read_bounded(item.path, limit, secrets)
    try:
        payload = json.loads(text) if not truncated else None
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, Mapping):
        payload = {key: payload[key] for key in allowed if key in payload}
        return result + "JSON (safe subset):\n" + redact(_json(payload), secrets)
    return result + "JSON (bounded):\n" + text + "\n"


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.rstrip()}\n\n"


def _safe_state(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id", "status", "started_at", "updated_at", "repo", "base_ref", "base_sha",
        "branch", "worktree", "cycle", "current_step", "approved_tree_sha", "commit_sha",
        "review_iterations", "planning_protocol",
    )
    result = {key: state.get(key) for key in keys if key in state}
    failure = state.get("failure")
    result["failure"] = {
        key: failure.get(key)
        for key in ("reason", "detail")
        if isinstance(failure, Mapping) and key in failure
    }
    publish = state.get("publish")
    result["publish"] = {
        key: publish.get(key)
        for key in ("mode", "target", "remote", "branch", "run_branch", "base_sha", "commit_sha", "web_url", "status", "local_base_updated", "run_branch_cleanup")
        if isinstance(publish, Mapping) and key in publish
    }
    result["planner"] = {
        key: state.get("planner", {}).get(key)
        for key in ("decision", "title", "profile_id", "model", "source", "execution_mode", "steps", "reviewer_recommendation")
        if isinstance(state.get("planner"), Mapping) and key in state["planner"]
    }
    cycles = state.get("cycles")
    if isinstance(cycles, list):
        result["cycles"] = [
            {key: item.get(key) for key in ("number", "kind", "status", "failure") if key in item}
            for item in cycles if isinstance(item, Mapping)
        ]
    return result


def _profiles(config: HarnessConfig, state: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    execution = state.get("execution") if isinstance(state.get("execution"), Mapping) else {}
    for role, value in sorted(execution.items()):
        if not isinstance(value, Mapping):
            continue
        profile_id = value.get("profile_id")
        if isinstance(profile_id, str):
            selected[role] = {
                key: value.get(key)
                for key in ("profile_id", "model", "effort", "selection_mode")
                if key in value
            }
    catalog = {}
    for profile_id, profile in sorted(profiles_for_config(config).items()):
        metadata = safe_profile_metadata(profile)
        catalog[profile_id] = {
            key: metadata.get(key)
            for key in ("id", "display_name", "roles", "model_label", "effort", "selection_mode")
        }
    return {"selected": selected, "catalogue": catalog}


def _plan_summary(run_dir: Path, secrets: tuple[str, ...], relative: str) -> str:
    item = _artifact(run_dir, relative)
    if not item.exists:
        return _artifact_header(item)
    text, _size, truncated = _read_bounded(item.path, MAX_ARTIFACT_BYTES, secrets)
    try:
        payload = json.loads(text) if not truncated else None
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, Mapping):
        return _artifact_header(item) + "Summary unavailable; bounded JSON follows:\n" + text + "\n"
    safe: dict[str, Any] = {
        key: payload[key] for key in ("schema_version", "decision", "title", "objective", "execution_mode", "reviewer_profile", "constraints", "acceptance", "tests", "risks") if key in payload
    }
    steps = payload.get("steps")
    if isinstance(steps, list):
        safe["steps"] = []
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            safe["steps"].append({key: step.get(key) for key in ("id", "title", "implementer_profile", "depends_on", "read_set", "write_set", "create_set", "delete_set", "contract_sha256") if key in step})
    return _artifact_header(item) + "Structured summary:\n" + redact(_json(safe), secrets)


def _bundle_summary(run_dir: Path, secrets: tuple[str, ...], relative: str) -> str:
    item = _artifact(run_dir, relative)
    if not item.exists:
        return _artifact_header(item)
    text, _size, truncated = _read_bounded(item.path, MAX_ARTIFACT_BYTES, secrets)
    try:
        payload = json.loads(text) if not truncated else None
    except (TypeError, ValueError):
        payload = None
    safe: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        safe = {key: payload[key] for key in ("schema_version", "execution_mode", "reviewer_profile") if key in payload}
        safe["steps"] = []
        for step in payload.get("steps", []) if isinstance(payload.get("steps"), list) else []:
            if isinstance(step, Mapping):
                safe["steps"].append({key: step.get(key) for key in ("id", "title", "implementer_profile", "depends_on", "contract_sha256")})
    return _artifact_header(item) + "Bundle summary:\n" + redact(_json(safe), secrets)


def _selection_summary(run_dir: Path, secrets: tuple[str, ...]) -> str:
    """Render execution selection metadata without endpoint/credential fields."""

    relative = "execution_selection.json"
    item = _artifact(run_dir, relative)
    if not item.exists:
        return _artifact_header(item)
    text, _size, truncated = _read_bounded(item.path, MAX_ARTIFACT_BYTES, secrets)
    try:
        payload = json.loads(text) if not truncated else None
    except ValueError:
        payload = None
    safe: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        safe["schema_version"] = payload.get("schema_version")
        for role in ("planner", "implementer", "reviser", "repair_implementer", "reviewer"):
            value = payload.get(role)
            if isinstance(value, Mapping):
                safe[role] = {key: value.get(key) for key in ("profile_id", "model", "effort", "selection_mode") if key in value}
        steps = payload.get("steps")
        if isinstance(steps, list):
            safe["steps"] = []
            for step in steps:
                if isinstance(step, Mapping):
                    value = step.get("implementer")
                    row = {"step_id": step.get("step_id")}
                    if isinstance(value, Mapping):
                        row["implementer"] = {key: value.get(key) for key in ("profile_id", "model", "effort", "selection_mode") if key in value}
                    safe["steps"].append(row)
    return _artifact_header(item) + "Safe metadata:\n" + _clean(_json(safe), secrets)


def _summarized_events(path: Path, secrets: tuple[str, ...], max_events: int = 30) -> str:
    # The caller supplies a path that is already rooted; only the content is
    # returned here because provenance is rendered by the caller.
    if not path.is_file():
        return "No events.\n"
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            start = max(0, size - MAX_EVENT_SCAN_BYTES)
            stream.seek(start)
            data = stream.read(size - start)
    except OSError:
        return "No events.\n"
    if start > 0:
        first_newline = data.find(b"\n")
        data = data[first_newline + 1:] if first_newline >= 0 else b""
    # JSONL records are complete only when terminated by a newline.  This
    # also prevents a concurrent/truncated final record from being parsed.
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1] if b"\n" in data else b""
    summaries: list[str] = []
    for raw in data.splitlines():
        if len(raw) > MAX_EVENT_BYTES:
            continue
        event = parse_event(raw.decode("utf-8", errors="replace"))
        if event is None:
            continue
        summary = summarize_step_event(event)
        if summary:
            summaries.append(_clean(summary, secrets))
    return "\n".join(summaries[-max_events:]) + ("\n" if summaries else "No summarized events.\n")


def _input_tokens_for(path: Path, *, nested_usage: bool = False) -> int | None:
    if nested_usage:
        try:
            if path.stat().st_size > MAX_ARTIFACT_BYTES:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return None
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        return normalize_usage(usage).get("input_tokens") if isinstance(usage, Mapping) else None
    usage = read_usage_artifact(path)
    return usage.get("input_tokens") if usage is not None else None


def _prompt_footprint(run_dir: Path) -> str:
    """Render deterministic metadata for every persisted prompt request."""

    rows: list[tuple[str, Path, Path | None, bool]] = []

    def add(relative: str, usage: str | None = None, *, nested_usage: bool = False) -> None:
        artifact = run_dir / relative
        if artifact.is_file():
            rows.append((relative, artifact, run_dir / usage if usage else None, nested_usage))

    add("planner.request.txt", "planner.usage.json")
    for root, usage_name in (("steps", "step.json"), ("repair/C02/steps", "step.json")):
        base = run_dir / root
        try:
            step_dirs = sorted(
                path for path in base.iterdir()
                if path.is_dir() and is_step_id(path.name)
            )
        except OSError:
            step_dirs = []
        for step_dir in step_dirs:
            add(f"{root}/{step_dir.name}/agent.prompt.txt", f"{root}/{step_dir.name}/{usage_name}", nested_usage=True)
    add("revision/C01/agent.prompt.txt", "revision/C01/usage.json")
    add("review/C01/reviewer.request.txt", "review/C01/reviewer.usage.json")
    add("repair/C02/planner.request.txt", "repair/C02/planner.usage.json")
    add("revision/C02/agent.prompt.txt", "revision/C02/usage.json")
    add("review/C02/reviewer.request.txt", "review/C02/reviewer.usage.json")

    lines = ["Prompt artifacts:", "| Relative path | Bytes | SHA256 | Input tokens |", "|---|---:|---|---:|"]
    for relative, artifact, usage_path, nested_usage in rows:
        item = _artifact(run_dir, relative)
        input_tokens = (
            _input_tokens_for(usage_path, nested_usage=nested_usage)
            if usage_path is not None and usage_path.is_file() else None
        )
        lines.append(
            f"| {relative} | {item.size} | {item.sha256 or '—'} | "
            f"{input_tokens if input_tokens is not None else '—'} |"
        )
    for relative in ("context.txt", "spec.md"):
        item = _artifact(run_dir, relative)
        if item.exists:
            lines.append(f"| {relative} | {item.size} | {item.sha256 or '—'} | — |")
    if len(lines) == 3:
        lines.append("| (none) | — | — | — |")
    return "\n".join(lines) + "\n"


def _claude_status(config: HarnessConfig, run_dir: Path, revision_exists: bool) -> str:
    options_path = run_dir / "run_options.json"
    if options_path.is_file():
        try:
            options, _digest = read_run_options_with_sha256(run_dir)
        except RunOptionsError:
            return "Claude revision status unavailable: durable run options are malformed."
        if not options.claude_revision_enabled:
            return "Claude revision disabled by run options."
        if not revision_exists:
            return "Claude revision enabled by run options, but no revision artifact was produced/reached."
        return "Claude revision artifacts present."
    # Historical runs had no durable per-run switch; make the fallback
    # explicit instead of treating a missing artifact as proof of disablement.
    if not config.revision.enabled:
        return "Claude revision disabled for this legacy run (configuration fallback)."
    if not revision_exists:
        return "Claude revision enabled for this legacy run, but no revision artifact was produced/reached."
    return "Claude revision artifacts present (legacy configuration fallback)."


def _event_artifact(run_dir: Path, relative: str, secrets: tuple[str, ...]) -> str:
    item = _artifact(run_dir, relative)
    return _artifact_header(item) + "Summarized events (tool arguments omitted):\n" + (
        _summarized_events(item.path, secrets) if item.exists else "No events.\n"
    )


def _cycle(config: HarnessConfig, run_dir: Path, cycle: int, secrets: tuple[str, ...]) -> str:
    prefix = "" if cycle == 1 else "repair/C02/"
    label = f"CYCLE C0{cycle}"
    parts = [_section(label, "")]
    bundle = prefix + "implementation_bundle.json"
    plan = prefix + "task_plan_v2.json"
    parts.append(_section("Contract / plan summary", _plan_summary(run_dir, secrets, plan)))
    parts.append(_section("Bundle", _bundle_summary(run_dir, secrets, bundle)))
    steps_root = run_dir / prefix / "steps"
    try:
        step_ids = sorted(path.name for path in steps_root.iterdir() if path.is_dir() and path.name.startswith("S"))
    except OSError:
        step_ids = []
    if not step_ids:
        legacy_worker = "\n".join([
            _artifact_json(run_dir, "agent.result.json", secrets),
            _artifact_text(run_dir, "agent.final.md", secrets),
            _artifact_text(run_dir, "agent.stderr.log", secrets, MAX_STDERR_BYTES, tail=True),
            _event_artifact(run_dir, "agent.events.jsonl", secrets),
        ])
        has_legacy_worker = any(
            (run_dir / name).exists()
            for name in ("agent.result.json", "agent.final.md", "agent.stderr.log", "agent.events.jsonl")
        )
        parts.append(_section("Steps", legacy_worker if has_legacy_worker else "No worker steps recorded."))
    for step_id in step_ids:
        step_prefix = prefix + f"steps/{step_id}/"
        body = "\n".join([
            _artifact_text(run_dir, step_prefix + "contract.md", secrets),
            _artifact_json(run_dir, step_prefix + "step.json", secrets),
            _artifact_text(run_dir, step_prefix + "agent.final.md", secrets),
            _artifact_text(run_dir, step_prefix + "agent.stderr.log", secrets, MAX_STDERR_BYTES, tail=True),
            _artifact_json(run_dir, step_prefix + "usage.json", secrets),
            _event_artifact(run_dir, step_prefix + "agent.events.jsonl", secrets),
        ])
        parts.append(_section(f"{step_id}", body))
    if cycle == 1 and (run_dir / "checks/C01/checks.json").exists():
        checks, changed, diff = "checks/C01/checks.json", "checks/C01/changed-files.txt", "checks/C01/diff.patch"
    else:
        checks = prefix + "checks.json" if cycle == 1 else "checks/C02/checks.json"
        changed = prefix + "changed-files.txt" if cycle == 1 else "checks/C02/changed-files.txt"
        diff = prefix + "diff.patch" if cycle == 1 else "checks/C02/diff.patch"
    checks_body = "\n".join([
        _artifact_json(run_dir, checks, secrets),
        _safe_json_artifact(run_dir, checks.rsplit("/", 1)[0] + "/evidence.json" if "/" in checks else "evidence.json", secrets, ("base_sha", "staged_tree_sha", "deterministic_passed", "failures", "changed_files", "checks")),
        _artifact_text(run_dir, changed, secrets),
        "Diff artifact metadata (diff omitted from consolidated diagnostics):",
        _artifact_header(_artifact(run_dir, diff)),
        "Diff content omitted from consolidated diagnostics.",
    ])
    parts.append(_section(f"CHECKS C0{cycle}", checks_body))
    revision = f"revision/C0{cycle}/"
    if cycle == 1 and not (run_dir / revision).is_dir() and any(
        (run_dir / "revision" / name).exists()
        for name in ("agent.prompt.txt", "agent.result.json", "agent.final.md", "agent.stderr.log", "report.json")
    ):
        revision = "revision/"
    revision_exists = any(
        (run_dir / revision / name).exists()
        for name in ("agent.prompt.txt", "agent.result.json", "agent.final.md", "agent.stderr.log", "report.json", "tree_before.txt")
    )
    claude_body = _claude_status(config, run_dir, revision_exists)
    if revision_exists:
        claude_body += "\n" + "\n".join([
            _artifact_text(run_dir, revision + "agent.prompt.txt", secrets),
            _artifact_json(run_dir, revision + "agent.result.json", secrets),
            _artifact_text(run_dir, revision + "agent.final.md", secrets),
            _artifact_text(run_dir, revision + "agent.stderr.log", secrets, MAX_STDERR_BYTES, tail=True),
            _artifact_json(run_dir, revision + "report.json", secrets),
            _artifact_json(run_dir, revision + "usage.json", secrets),
            _event_artifact(run_dir, revision + "agent.events.jsonl", secrets),
            _artifact_text(run_dir, revision + "tree_before.txt", secrets),
            _artifact_text(run_dir, revision + "tree_after.txt", secrets),
        ])
    parts.append(_section(f"CLAUDE C0{cycle}", claude_body))
    review = f"review/C0{cycle}/"
    if cycle == 1 and not (run_dir / review).is_dir() and (run_dir / "review.json").exists():
        review = ""
    review_body = "\n".join([
        _artifact_text(run_dir, review + "reviewer.request.txt", secrets),
        _artifact_text(run_dir, review + "reviewer.raw.md", secrets),
        _artifact_json(run_dir, review + "review.json", secrets),
        _artifact_json(run_dir, review + "reviewer.usage.json", secrets),
    ])
    parts.append(_section(f"REVIEWER C0{cycle}", review_body))
    return "".join(parts)


def _attempts(run_dir: Path, secrets: tuple[str, ...]) -> str:
    rows: list[str] = []
    try:
        candidates = sorted(path for path in run_dir.rglob("attempts") if path.is_dir())
    except OSError:
        candidates = []
    for attempts_dir in candidates:
        try:
            attempts = sorted(path for path in attempts_dir.iterdir() if path.is_dir())
        except OSError:
            continue
        phase = str(attempts_dir.parent.relative_to(run_dir))
        if phase == ".":
            phase = "run"
        for attempt in attempts:
            status = "FAILED" if (attempt / "agent.stderr.log").exists() or (attempt / "planner.raw.md").exists() else "RECORDED"
            rows.append(f"### {phase} — ATTEMPT {attempt.name} — {status}\n")
            for name in ("planner.request.txt", "planner.raw.md", "agent.final.md", "agent.stderr.log", "step.json", "usage.json", "planner.usage.json", "reviewer.raw.md", "review.json"):
                if (attempt / name).exists():
                    relative = str((attempt / name).relative_to(run_dir))
                    rows.append(_artifact_text(run_dir, relative, secrets, MAX_STDERR_BYTES if name.endswith("stderr.log") else MAX_ARTIFACT_BYTES, tail=name.endswith("stderr.log")))
        current_index = max((int(path.name) for path in attempts if path.name.isdigit()), default=0) + 1
        current = attempts_dir.parent
        current_status = _current_attempt_status(current, secrets)
        if current_status is not None:
            rows.append(f"### {phase} — ATTEMPT {current_index:02d} — {current_status}\n")
    return _section("ATTEMPTS", "".join(rows) if rows else "No archived attempts recorded.")


def _current_attempt_status(directory: Path, secrets: tuple[str, ...]) -> str | None:
    """Classify the live artifact set after an archived retry."""

    step = directory / "step.json"
    try:
        if step.is_file():
            text, _size, truncated = _read_bounded(step, MAX_ARTIFACT_BYTES, secrets)
            payload = json.loads(text) if not truncated else None
            if isinstance(payload, Mapping) and payload.get("status") == "COMPLETED":
                return "SUCCESS"
            if isinstance(payload, Mapping) and payload.get("status") == "FAILED":
                return "FAILED"
        result = directory / "agent.result.json"
        if result.is_file():
            text, _size, truncated = _read_bounded(result, MAX_ARTIFACT_BYTES, secrets)
            payload = json.loads(text) if not truncated else None
            if isinstance(payload, Mapping) and payload.get("exit_code") == 0 and not payload.get("timed_out"):
                return "SUCCESS"
            return "FAILED"
        for name in ("review.json",):
            path = directory / name
            if path.is_file():
                text, _size, truncated = _read_bounded(path, MAX_ARTIFACT_BYTES, secrets)
                payload = json.loads(text) if not truncated else None
                return "SUCCESS" if isinstance(payload, Mapping) and payload.get("verdict") == "PASS" else "FAILED"
        planner = directory / "planner.raw.md"
        if planner.is_file():
            text, _size, _truncated = _read_bounded(planner, MAX_ARTIFACT_BYTES, secrets)
            return "SUCCESS" if '"decision": "READY"' in text or "STATUS: READY" in text else "FAILED"
    except (OSError, UnicodeError, ValueError):
        return "FAILED"
    return None


def _timeline(state: Mapping[str, Any]) -> str:
    rows: list[str] = []
    if state.get("started_at"):
        rows.append(f"- {state['started_at']} — CREATED — run initialized")
    timestamps = state.get("timestamps")
    if isinstance(timestamps, Mapping):
        for phase, timestamp in sorted(timestamps.items(), key=lambda pair: str(pair[1])):
            rows.append(f"- {timestamp} — {phase} — state timestamp")
    if state.get("updated_at"):
        rows.append(f"- {state['updated_at']} — {state.get('status', 'unknown').upper()} — final durable state")
    return "\n".join(rows) if rows else "No timestamps were persisted."


def build_run_diagnostics(config: HarnessConfig, run_dir: str | Path) -> str:
    """Build one bounded Markdown report without changing any source artifact."""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    directory = Path(run_dir).expanduser().resolve()
    secrets = config_secret_values(config)
    state_text, _state_size, state_truncated = _read_bounded(
        directory / "state.json", MAX_ARTIFACT_BYTES, secrets
    )
    try:
        state = json.loads(state_text) if not state_truncated else {}
    except (TypeError, ValueError):
        state = {}
    if not isinstance(state, Mapping):
        state = {}
    header = "# MetaHarness Run Diagnostics\n\n"
    header += _section("REPORT", _json({
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _now(),
        "run_id": state.get("run_id", directory.name),
        "status": state.get("status", "unknown"),
        "confidentiality": "This report may contain SPEC, repository context, prompts and model outputs. Configured secrets are redacted, but review this file before external sharing; it is not anonymized.",
    }))
    body = header
    summary_body = _artifact_header(_artifact(directory, "state.json")) + _clean(_json(_safe_state(state)), secrets)
    if state_truncated:
        summary_body += f"\n[TRUNCATED: original {_state_size} bytes]\n"
    body += _section("RUN SUMMARY", summary_body)
    body += _section("RUN OPTIONS", _safe_json_artifact(directory, "run_options.json", secrets, ("schema_version", "planning", "pipeline", "profiles")))
    try:
        checkpoint_record = read_checkpoint_record(directory)
        checkpoint_text = _artifact_json(directory, "resume_checkpoint.json", secrets)
        info = resume_info(directory, state)
        checkpoint_text += "Current resumable status:\n" + _json({"resumable": info.resumable, "phase": info.phase, "cycle": info.cycle, "step_id": info.step_id, "label": info.label, "reason": info.reason})
        if checkpoint_record:
            checkpoint_text += f"Checkpoint status: {checkpoint_record[1]}\n"
    except (OSError, ValueError, ResumeCheckpointError) as exc:
        checkpoint_text = _artifact_json(directory, "resume_checkpoint.json", secrets) + "Resume information unavailable: " + redact(str(exc), secrets)[:500]
    body += _section("RESUME", checkpoint_text)
    body += _section("EXECUTION PROFILES", _clean(_json(_profiles(config, state)), secrets))
    body += _section("SPEC", _artifact_text(directory, "spec.md", secrets))
    context_meta = state.get("context") if isinstance(state.get("context"), Mapping) else {}
    repository_meta = {"repository_reference": None, "base_sha": state.get("base_sha"), "context": context_meta}
    ref = _artifact(directory, "repository_reference.json")
    if ref.exists:
        text, _size, truncated = _read_bounded(ref.path, MAX_ARTIFACT_BYTES, secrets)
        try:
            repository_meta["repository_reference"] = json.loads(text) if not truncated else text
        except ValueError:
            repository_meta["repository_reference"] = text
    body += _section("REPOSITORY / CONTEXT SUMMARY", _artifact_header(ref) + _clean(_json(repository_meta), secrets) + "\n" + _artifact_header(_artifact(directory, "context.txt")))
    body += _section("PROMPT FOOTPRINT", _prompt_footprint(directory))
    body += _section("PLANNER REQUEST", _artifact_text(directory, "planner.request.txt", secrets, MAX_PLANNER_REQUEST_BYTES))
    recovered = plan_source(directory) == PLAN_SOURCE_OPERATOR
    body += _section("PLANNER RESPONSE", "\n".join([
        # An operator recovery replaces the planner answer without any model
        # call; the planner usage below belongs to the archived attempt.
        "plan source: operator recovery" if recovered else "plan source: planner model completion",
        *([_safe_json_artifact(directory, PLAN_RECOVERY_ARTIFACT, secrets, (
            "schema_version", "source", "previous_raw_sha256", "replacement_raw_sha256",
            "recovered_at", "archived_attempt", "planner_called",
        ))] if recovered else []),
        _artifact_text(directory, "planner.raw.md", secrets),
        _plan_summary(directory, secrets, "task_plan_v2.json"),
        _artifact_json(directory, "planner.usage.json", secrets),
    ]))
    body += _section("PLAN / BUNDLE", "\n".join([
        _artifact_text(directory, "implementation_contract.md", secrets),
        _bundle_summary(directory, secrets, "implementation_bundle.json"),
    ]))
    body += _section("APPROVAL", "\n".join([
        _safe_json_artifact(directory, "plan_approval.json", secrets, ("decision", "raw_sha256", "contract_sha256", "bundle_sha256", "execution_sha256", "source")),
        _selection_summary(directory, secrets),
    ]))
    body += _cycle(config, directory, 1, secrets)
    if (directory / "repair" / "C02").is_dir():
        body += _section("REPAIR C02", "\n".join([
            _artifact_text(directory, "repair/C02/planner.request.txt", secrets, MAX_PLANNER_REQUEST_BYTES),
            _artifact_text(directory, "repair/C02/planner.raw.md", secrets),
            _plan_summary(directory, secrets, "repair/C02/task_plan_v2.json"),
            _bundle_summary(directory, secrets, "repair/C02/implementation_bundle.json"),
        ]))
        body += _cycle(config, directory, 2, secrets)
    else:
        body += _section("REPAIR C02", "No repair cycle executed.")
    body += _section("COMMIT / PUBLISH", "\n".join([
        _safe_json_artifact(directory, "publish.json", secrets, ("mode", "target", "remote", "branch", "run_branch", "base_sha", "commit_sha", "web_url", "status", "local_base_updated", "run_branch_cleanup")),
        "Commit SHA: " + str(state.get("commit_sha", "—")),
    ]))
    body += _attempts(directory, secrets)
    body += _section("USAGE SUMMARY", _clean(_json(phase_usage_summary(directory)), secrets))
    body += _section("TIMELINE", _timeline(state))
    body = _clean(body, secrets)
    encoded = body.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_REPORT_BYTES:
        return body
    marker = f"\n[TRUNCATED: original {len(encoded)} bytes]\n"
    kept = MAX_REPORT_BYTES - len(marker.encode("utf-8"))
    return encoded[:kept].decode("utf-8", errors="ignore") + marker


def write_run_diagnostics(config: HarnessConfig, run_dir: str | Path) -> Path:
    """Write ``diagnostics.md`` atomically; persist a bounded error on failure."""

    directory = Path(run_dir).expanduser().resolve()
    target = directory / DIAGNOSTICS_NAME
    try:
        atomic_write_text(target, build_run_diagnostics(config, directory))
    except Exception as exc:
        try:
            secrets = config_secret_values(config)
            message = _clean(f"{type(exc).__name__}: {exc}", secrets)[:4096]
            atomic_write_text(directory / DIAGNOSTICS_ERROR_NAME, message + "\n")
        except Exception:
            pass
        raise
    return target


__all__ = [
    "DIAGNOSTICS_ERROR_NAME", "DIAGNOSTICS_NAME", "MAX_ARTIFACT_BYTES",
    "MAX_PLANNER_REQUEST_BYTES", "MAX_REPORT_BYTES", "build_run_diagnostics",
    "write_run_diagnostics",
]
