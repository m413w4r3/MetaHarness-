"""Bounded, argument-free token diagnostics for one Codex step.

The artifact explains *why* a worker consumed a large context without ever
persisting prompts, tool arguments or file contents: only counters, command
program names and repository paths exposed by structured event fields.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from ..result import atomic_write_text
from ..usage import normalize_usage
from .events import parse_event

TOKEN_DIAGNOSTICS_NAME = "token_diagnostics.json"
MAX_OBSERVED_FILES = 100
MAX_OBSERVED_COMMANDS = 100
_MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
_COMMAND_NAME = re.compile(r"[A-Za-z0-9_.+-]{1,40}")
_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})
_SHELL_FLAGS = frozenset({"-c", "-lc", "-ic", "-lic"})
_TOOL_ITEM_TYPES = frozenset({
    "command_execution", "mcp_tool_call", "file_change", "patch_apply", "apply_patch",
    "web_search", "function_call", "tool_call", "local_shell_call", "custom_tool_call",
})
_TOOL_EVENT_TYPES = frozenset({"exec_command_begin", "mcp_tool_call_begin", "patch_apply_begin"})
_READ_TYPES = frozenset({"read", "file_read", "read_file"})


def _words(command: Any) -> list[str]:
    if isinstance(command, (list, tuple)):
        return [part for part in command if isinstance(part, str)]
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()
    return []


def command_name(command: Any) -> str | None:
    """Only the program name of a command, never an argument.

    ``bash -lc "<script>"`` is reduced to the first program of the script.
    """

    parts = _words(command)
    if len(parts) >= 3 and parts[0].rsplit("/", 1)[-1] in _SHELLS and parts[1] in _SHELL_FLAGS:
        parts = _words(parts[2])
    if not parts:
        return None
    name = parts[0].rsplit("/", 1)[-1]
    return name if _COMMAND_NAME.fullmatch(name) else None


def _repository_path(value: Any, worktree: Path | None) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 300 or "\x00" in value:
        return None
    if value.startswith("/"):
        if worktree is None:
            return None
        try:
            value = str(Path(os.path.normpath(value)).relative_to(worktree))
        except ValueError:
            return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _nodes(event: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield event
    for key in ("msg", "item"):
        value = event.get(key)
        if isinstance(value, Mapping):
            yield value
            nested = value.get("item")
            if isinstance(nested, Mapping):
                yield nested


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open("rb")
    except OSError:
        return
    with stream:
        skipping = False
        while True:
            chunk = stream.readline(_MAX_EVENT_LINE_BYTES)
            if not chunk:
                return
            complete = chunk.endswith(b"\n")
            if skipping or (not complete and len(chunk) >= _MAX_EVENT_LINE_BYTES):
                skipping = not complete
                continue
            event = parse_event(chunk.decode("utf-8", errors="replace"))
            if event is not None:
                yield event


def token_diagnostics(
    events_path: str | Path, usage: Any, *, worktree: str | Path | None = None
) -> dict[str, Any]:
    """Compute the bounded diagnostic record of one step's event stream."""

    root = Path(worktree).expanduser().resolve() if worktree is not None else None
    normalized = normalize_usage(usage)
    event_count = 0
    tool_ids: set[str] = set()
    anonymous_tools = 0
    files: list[str] = []
    commands: list[str] = []

    def add(bucket: list[str], value: str | None, limit: int) -> None:
        if value is not None and value not in bucket and len(bucket) < limit:
            bucket.append(value)

    for event in _iter_events(Path(events_path)):
        event_count += 1
        event_type = str(event.get("type") or "")
        for node in _nodes(event):
            node_type = str(node.get("type") or "")
            if node_type in _TOOL_ITEM_TYPES and event_type != "item.updated":
                identifier = node.get("id")
                if isinstance(identifier, str) and identifier:
                    tool_ids.add(identifier)
                elif event_type in {"item.completed", ""}:
                    anonymous_tools += 1
            if node_type in _TOOL_EVENT_TYPES:
                anonymous_tools += 1
            if node_type == "command_execution" or node_type == "exec_command_begin":
                add(commands, command_name(node.get("command")), MAX_OBSERVED_COMMANDS)
            if node_type in _READ_TYPES:
                add(files, _repository_path(node.get("path"), root), MAX_OBSERVED_FILES)
            parsed = node.get("parsed_cmd")
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, Mapping) and entry.get("type") in _READ_TYPES:
                        add(files, _repository_path(entry.get("path"), root), MAX_OBSERVED_FILES)
    return {
        "input_tokens": normalized["input_tokens"],
        "cached_input_tokens": normalized["cached_input_tokens"],
        "output_tokens": normalized["output_tokens"],
        "reasoning_output_tokens": normalized["reasoning_output_tokens"],
        "event_count": event_count,
        "tool_call_count": len(tool_ids) + anonymous_tools,
        "files_read_observed": files,
        "commands_observed": commands,
    }


def write_token_diagnostics(
    step_dir: str | Path, usage: Any, *, worktree: str | Path | None = None
) -> dict[str, Any]:
    directory = Path(step_dir)
    record = token_diagnostics(directory / "agent.events.jsonl", usage, worktree=worktree)
    atomic_write_text(directory / TOKEN_DIAGNOSTICS_NAME, json.dumps(record, indent=2) + "\n")
    return record


__all__ = [
    "MAX_OBSERVED_COMMANDS",
    "MAX_OBSERVED_FILES",
    "TOKEN_DIAGNOSTICS_NAME",
    "command_name",
    "token_diagnostics",
    "write_token_diagnostics",
]
