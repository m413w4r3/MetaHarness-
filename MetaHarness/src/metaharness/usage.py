"""Provider-neutral token usage records persisted per phase.

Every record has exactly :data:`USAGE_FIELDS`.  A counter the provider did not
supply is ``0``; the only derived value is ``total_tokens``, which falls back
to ``input_tokens + output_tokens`` when the provider gives no total.  No
pricing or cost is ever computed here.
"""

from __future__ import annotations

import json
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


__all__ = [
    "PLANNER_USAGE_ARTIFACT",
    "REVIEWER_USAGE_ARTIFACT",
    "USAGE_FIELDS",
    "add_usage",
    "completion_usage",
    "empty_usage",
    "normalize_usage",
    "read_usage_artifact",
    "write_usage_artifact",
]
