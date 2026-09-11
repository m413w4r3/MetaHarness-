"""Lecture tolérante du flux JSONL produit par ``codex exec --json``."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


def parse_event(line: str) -> dict[str, Any] | None:
    """Parse une ligne JSONL; les lignes non JSON ou non-object sont ignorées."""

    if not isinstance(line, str):
        raise TypeError("event line must be a string")
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def iter_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield only the valid JSON objects from an agent stream."""

    for line in lines:
        event = parse_event(line)
        if event is not None:
            yield event


def _nodes(event: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the common event envelopes used by Codex versions."""

    yield event
    for key in ("msg", "info", "item", "message", "result"):
        value = event.get(key)
        if isinstance(value, dict):
            yield value
            for nested_key in ("item", "message", "result"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    yield nested


def _integer_fields(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    accepted = {
        key: number
        for key, number in value.items()
        if isinstance(key, str) and isinstance(number, int) and not isinstance(number, bool)
    }
    wanted = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    }
    result = {key: number for key, number in accepted.items() if key in wanted}
    return result or None


def extract_usage(event: dict[str, Any]) -> dict[str, int] | None:
    """Return token counters found in an event, if any.

    The CLI has used both ``usage`` and ``token_usage`` and has placed them at
    the top level or in an envelope.  Unknown counters are deliberately not
    propagated into the public result.
    """

    if not isinstance(event, dict):
        raise TypeError("event must be an object")
    for node in _nodes(event):
        for key in ("total_token_usage", "token_usage", "usage", "last_token_usage"):
            usage = _integer_fields(node.get(key))
            if usage is not None:
                return usage
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        joined = "".join(parts)
        return joined if joined else None
    return None


def extract_final(event: dict[str, Any]) -> str | None:
    """Extract a final assistant message from known Codex event shapes."""

    if not isinstance(event, dict):
        raise TypeError("event must be an object")
    event_type = str(
        event.get("type") or (event.get("msg", {}).get("type") if isinstance(event.get("msg"), dict) else "")
    ).casefold()
    nodes = list(_nodes(event))
    # A normal agent_message is final only when Codex marks the turn complete,
    # or when it uses the explicit final-message event name.
    finalish = (
        "final" in event_type
        or "output_text" in event_type
        or event_type in {"turn.completed", "response.completed", "result"}
        or event.get("subtype") == "success"
    )
    for node in nodes:
        for key in ("final_message", "last_message", "result"):
            value = _text(node.get(key))
            if value is not None:
                return value
    if finalish:
        for node in nodes:
            for key in ("text", "content", "message"):
                value = _text(node.get(key))
                if value is not None:
                    return value
    for node in nodes:
        if node.get("type") in {"agent_message", "assistant_message"}:
            value = _text(node.get("text") or node.get("content"))
            if value is not None:
                return value
    return None


def summarize_event(event: dict[str, Any]) -> str | None:
    """Return a small human-readable description for optional progress logs."""

    msg = event.get("msg") if isinstance(event.get("msg"), dict) else {}
    event_type = event.get("type") or msg.get("type")
    item = event.get("item") or msg.get("item") or {}
    if isinstance(item, dict):
        tool_name = item.get("tool_name") or item.get("name") or item.get("tool")
        if isinstance(tool_name, str) and tool_name in {"apply_patch", "exec_command"}:
            message = item.get("message")
            if isinstance(message, str) and message:
                return f"tool: {tool_name}\nmessage: {message.replace(chr(10), ' ')[:110]}"
            return f"tool: {tool_name}"
        command = item.get("command")
        if command:
            return f"$ {str(command).replace(chr(10), ' ')[:110]}"
        if item.get("type") in {"file_change", "patch_apply", "apply_patch"}:
            paths = item.get("paths") or item.get("files") or []
            return f"édition : {', '.join(map(str, paths))[:110]}"
        if item.get("type") in {"agent_message", "assistant_message"}:
            text = item.get("text", "")
            return f"message : {str(text).replace(chr(10), ' ')[:110]}"
    if event_type:
        return str(event_type)
    return None


__all__ = ["extract_final", "extract_usage", "iter_events", "parse_event", "summarize_event"]
