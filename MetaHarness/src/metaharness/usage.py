"""Provider-neutral token usage records persisted per phase.

Every record has exactly :data:`USAGE_FIELDS`.  A counter the provider did not
supply is ``0``; the only derived value is ``total_tokens``, which falls back
to ``input_tokens + output_tokens`` when the provider gives no total.  No
pricing or cost is ever computed here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .result import atomic_write_text
from .step_ids import STEP_ID_RE

USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
PLANNER_USAGE_ARTIFACT = "planner.usage.json"
REVIEWER_USAGE_ARTIFACT = "reviewer.usage.json"
_MAX_USAGE_ARTIFACT_BYTES = 16 * 1024

# Provider spellings accepted for each canonical counter, in priority order.
_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
    "cache_write_input_tokens": ("cache_write_input_tokens", "cache_creation_input_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "reasoning_output_tokens": ("reasoning_output_tokens",),
    "total_tokens": ("total_tokens",),
}
# OpenAI-style nested details: (container, key) pairs.
_NESTED: dict[str, tuple[tuple[str, str], ...]] = {
    "cached_input_tokens": (
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
    ),
    "cache_write_input_tokens": (
        ("prompt_tokens_details", "cache_write_tokens"),
        ("input_tokens_details", "cache_write_tokens"),
    ),
    "reasoning_output_tokens": (
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ),
}


def _count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def empty_usage() -> dict[str, int]:
    return {name: 0 for name in USAGE_FIELDS}


def normalize_usage(raw: Any) -> dict[str, int]:
    """Map one provider usage object to the exact canonical schema."""

    source: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    found: dict[str, int | None] = {}
    for name in USAGE_FIELDS:
        value: int | None = None
        for alias in _ALIASES[name]:
            value = _count(source.get(alias))
            if value is not None:
                break
        if value is None:
            for container, key in _NESTED.get(name, ()):
                details = source.get(container)
                if isinstance(details, Mapping):
                    value = _count(details.get(key))
                    if value is not None:
                        break
        found[name] = value
    usage = {name: found[name] or 0 for name in USAGE_FIELDS}
    if found["total_tokens"] is None:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def add_usage(records: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Sum canonical records field by field."""

    total = empty_usage()
    for record in records:
        for name in USAGE_FIELDS:
            value = _count(record.get(name)) if isinstance(record, Mapping) else None
            total[name] += value or 0
    return total


def completion_usage(result: Any) -> dict[str, Any]:
    """Raw usage of one client completion; a plain-text double has none."""

    usage = getattr(result, "usage", None)
    return dict(usage) if isinstance(usage, Mapping) else {}


def write_usage_artifact(path: str | Path, raw: Any) -> dict[str, int]:
    """Persist the canonical record of *raw* and return it."""

    usage = normalize_usage(raw)
    atomic_write_text(path, json.dumps(usage, indent=2) + "\n")
    return usage


def read_usage_artifact(path: str | Path) -> dict[str, int] | None:
    """Read a bounded canonical record; ``None`` when absent or invalid."""

    try:
        target = Path(path)
        if target.stat().st_size > _MAX_USAGE_ARTIFACT_BYTES:
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return normalize_usage(payload)


_STEP_DIRECTORY = STEP_ID_RE
_ATTEMPT_DIRECTORY = re.compile(r"[0-9]{2,}")
_MAX_STEP_RECORD_BYTES = 128 * 1024


def _step_record_usage(record: Path) -> dict[str, int] | None:
    """The canonical usage of one bounded ``step.json``, if it has any."""

    try:
        if record.stat().st_size > _MAX_STEP_RECORD_BYTES:
            return None
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
        return normalize_usage(payload["usage"])
    return None


def _archived_attempt_usage(step_dir: Path) -> list[dict[str, int]]:
    """Usage of every archived attempt of one step, oldest attempt first.

    ``_archive_attempt`` *moves* a finished attempt's artifacts into
    ``attempts/NN/``, so an archived record is never also the current one:
    summing them can not double-count a worker invocation.
    """

    try:
        entries = sorted((step_dir / "attempts").iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    rows: list[dict[str, int]] = []
    for entry in entries:
        if _ATTEMPT_DIRECTORY.fullmatch(entry.name) is None or not entry.is_dir():
            continue
        usage = _step_record_usage(entry / "step.json")
        if usage is not None:
            rows.append(usage)
    return rows


def persisted_step_usage(steps_dir: str | Path) -> list[dict[str, Any]]:
    """``[{"id", "usage", "attempts"}]`` for every ``Sxx`` under *steps_dir*.

    One logical step may have run several times (a bounded mismatch retry, a
    transient transport failure, a resume).  ``usage`` is the aggregate of
    every actual worker invocation of that step: its current ``step.json`` plus
    each archived ``attempts/NN/step.json``.  Worker totals are always derived
    from these durable records, never from ``state.steps``, which only
    describes the current cycle.
    """

    try:
        entries = sorted(Path(steps_dir).iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if _STEP_DIRECTORY.fullmatch(entry.name) is None or not entry.is_dir():
            continue
        records = _archived_attempt_usage(entry)
        current = _step_record_usage(entry / "step.json")
        if current is not None:
            records.append(current)
        if not records:
            continue
        rows.append({
            "id": entry.name, "usage": add_usage(records), "attempts": len(records),
        })
    return rows


def _first_usage(*paths: Path) -> dict[str, int]:
    for path in paths:
        usage = read_usage_artifact(path)
        if usage is not None:
            return usage
    return empty_usage()


def phase_usage_summary(run_dir: str | Path) -> dict[str, Any]:
    """Aggregate persisted usage from every generic pipeline cycle."""

    directory = Path(run_dir)
    planner = _first_usage(directory / PLANNER_USAGE_ARTIFACT)
    cycles: list[dict[str, Any]] = []
    implementer_rows: list[dict[str, Any]] = []
    correction_planner = empty_usage()
    semantic_reviser = empty_usage()
    check_repair = empty_usage()
    final_reviewer = empty_usage()
    for cycle_path in sorted((directory / "cycles").glob("[0-9][0-9][0-9]")):
        if not cycle_path.is_dir():
            continue
        number = int(cycle_path.name)
        step_rows = [
            {**row, "cycle": number}
            for row in persisted_step_usage(cycle_path / "implementation" / "steps")
        ]
        cycle_implementer = add_usage(row["usage"] for row in step_rows)
        cycle_planner = _first_usage(cycle_path / "correction" / PLANNER_USAGE_ARTIFACT)
        cycle_reviser = _first_usage(cycle_path / "semantic-revision" / "usage.json")
        cycle_repair = add_usage(
            _first_usage(path)
            for path in sorted(cycle_path.glob("check-repair/*/attempts/[0-9][0-9][0-9]/usage.json"))
        )
        cycle_reviewer = _first_usage(cycle_path / "review" / REVIEWER_USAGE_ARTIFACT)
        correction_planner = add_usage((correction_planner, cycle_planner))
        semantic_reviser = add_usage((semantic_reviser, cycle_reviser))
        check_repair = add_usage((check_repair, cycle_repair))
        final_reviewer = add_usage((final_reviewer, cycle_reviewer))
        implementer_rows.extend(step_rows)
        cycles.append({
            "number": number,
            "correction_planner": cycle_planner,
            "implementer": cycle_implementer,
            "semantic_reviser": cycle_reviser,
            "check_repair": cycle_repair,
            "final_reviewer": cycle_reviewer,
        })
    summary: dict[str, Any] = {
        "planner": planner,
        "correction_planner": correction_planner,
        "implementer": {"total": add_usage(row["usage"] for row in implementer_rows), "steps": implementer_rows},
        "check_repair": check_repair,
        "semantic_reviser": semantic_reviser,
        "final_reviewer": final_reviewer,
        "cycles": cycles,
    }
    summary["grand_total"] = add_usage((
        planner, correction_planner, summary["implementer"]["total"],
        check_repair, semantic_reviser, final_reviewer,
    ))
    return summary


__all__ = [
    "PLANNER_USAGE_ARTIFACT",
    "REVIEWER_USAGE_ARTIFACT",
    "USAGE_FIELDS",
    "add_usage",
    "completion_usage",
    "empty_usage",
    "normalize_usage",
    "persisted_step_usage",
    "phase_usage_summary",
    "read_usage_artifact",
    "write_usage_artifact",
]
