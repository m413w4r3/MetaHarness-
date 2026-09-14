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


_STEP_DIRECTORY = re.compile(r"S0[1-6]\Z")
_MAX_STEP_RECORD_BYTES = 128 * 1024


def persisted_step_usage(steps_dir: str | Path) -> list[dict[str, Any]]:
    """``[{"id", "usage"}]`` read from every ``Sxx/step.json`` under *steps_dir*.

    Luna totals are always derived from these durable records, never from
    ``state.steps``, which only describes the current cycle.
    """

    try:
        entries = sorted(Path(steps_dir).iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if _STEP_DIRECTORY.fullmatch(entry.name) is None or not entry.is_dir():
            continue
        record = entry / "step.json"
        try:
            if record.stat().st_size > _MAX_STEP_RECORD_BYTES:
                continue
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            rows.append({"id": entry.name, "usage": normalize_usage(payload["usage"])})
    return rows


def _first_usage(*paths: Path) -> dict[str, int]:
    for path in paths:
        usage = read_usage_artifact(path)
        if usage is not None:
            return usage
    return empty_usage()


def phase_usage_summary(
    run_dir: str | Path, *, v1_agent_usage: Any = None
) -> dict[str, Any]:
    """Aggregate persisted token usage per phase and cycle; no cost derived."""

    directory = Path(run_dir)
    planner = _first_usage(directory / PLANNER_USAGE_ARTIFACT)
    c01_steps = [{**row, "cycle": 1} for row in persisted_step_usage(directory / "steps")]
    c02_steps = [
        {**row, "cycle": 2}
        for row in persisted_step_usage(directory / "repair" / "C02" / "steps")
    ]
    if c01_steps:
        luna_c01 = add_usage(row["usage"] for row in c01_steps)
    elif isinstance(v1_agent_usage, Mapping) and v1_agent_usage:
        luna_c01 = normalize_usage(v1_agent_usage)
    else:
        luna_c01 = empty_usage()
    luna_c02 = add_usage(row["usage"] for row in c02_steps)
    claude_c01 = _first_usage(
        directory / "revision" / "C01" / "usage.json", directory / "revision" / "usage.json"
    )
    reviewer_c01 = _first_usage(
        directory / "review" / "C01" / REVIEWER_USAGE_ARTIFACT,
        directory / REVIEWER_USAGE_ARTIFACT,
    )
    repair_planner = _first_usage(directory / "repair" / "C02" / PLANNER_USAGE_ARTIFACT)
    claude_c02 = _first_usage(directory / "revision" / "C02" / "usage.json")
    reviewer_c02 = _first_usage(directory / "review" / "C02" / REVIEWER_USAGE_ARTIFACT)
    has_c02 = (directory / "repair" / "C02").is_dir() or any(
        value != empty_usage() for value in (repair_planner, luna_c02, claude_c02, reviewer_c02)
    )
    summary: dict[str, Any] = {
        "planner": planner,
        "implementer": {
            "total": add_usage((luna_c01, luna_c02)),
            "steps": c01_steps + c02_steps,
        },
        "reviser": add_usage((claude_c01, claude_c02)),
        "reviewer": add_usage((reviewer_c01, reviewer_c02)),
    }
    if has_c02:
        summary.update({
            "luna_c01": luna_c01,
            "claude_c01": claude_c01,
            "reviewer_c01": reviewer_c01,
            "repair_planner_c02": repair_planner,
            "luna_c02": luna_c02,
            "claude_c02": claude_c02,
            "reviewer_c02": reviewer_c02,
        })
    summary["grand_total"] = add_usage(
        (planner, luna_c01, claude_c01, reviewer_c01,
         repair_planner, luna_c02, claude_c02, reviewer_c02)
    )
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
