"""Bounded Claude Code execution for semantic revision."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..gitops import GitError, current_head
from ..models import ModelProfile, ProfileDriver
from ..procutil import run_bounded
from ..usage import normalize_usage
from ..agent.base import AgentError
from ..agent.events import extract_final, extract_terminal_result, extract_usage, parse_event


class ClaudeAgentError(AgentError):
    code = "CLAUDE_FAILED"


class ClaudeCommittedError(ClaudeAgentError):
    code = "CLAUDE_COMMITTED"

    def __init__(self, expected: str, actual: str):
        super().__init__(f"{self.code}: expected HEAD {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class ClaudeResult:
    exit_code: int
    timed_out: bool
    final_message: str
    usage: dict[str, int]
    stderr_tail: str
    terminal_type: str | None = None
    terminal_subtype: str | None = None
    terminal_is_error: bool | None = None
    terminal_num_turns: int | None = None
    terminal_stop_reason: str | None = None
    terminal_errors: tuple[str, ...] = ()


_DEFAULT_TAIL_BYTES = 16 * 1024
_MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
_REVISION_TOOLS = "Read,Edit,Write,Grep,Glob"
# Revision lifetime is bounded by profile.timeout_seconds through run_bounded,
# not by an arbitrary Claude turn count.
# Only these names are inherited; HOME, TMPDIR and XDG_CACHE_HOME are forced
# below the managed Claude home and never taken from the parent process.
_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TERM")
_AUTH_FAILURE_SIGNALS = (
    "401 unauthorized",
    "unauthorized",
    "authentication required",
    "not logged in",
    "not authenticated",
    "please log in",
    "authentication failed",
    "login required",
)


def build_claude_environment(
    source_environment: Mapping[str, str], *, claude_home: Path
) -> dict[str, str]:
    """Return the only environment inherited by Claude Code.

    The personal HOME, TMPDIR, XDG directories, CODEX_HOME and every API key
    are dropped.  The credential stays in ``CLAUDE_CONFIG_DIR``.
    """

    if not isinstance(source_environment, Mapping):
        raise TypeError("source_environment must be a mapping")
    home = Path(claude_home).expanduser().resolve()
    environment = {
        name: source_environment[name]
        for name in _ALLOWLIST
        if name in source_environment and isinstance(source_environment[name], str)
    }
    environment["HOME"] = str(home / "home")
    environment["CLAUDE_CONFIG_DIR"] = str(home)
    environment["XDG_CACHE_HOME"] = str(home / "cache")
    environment["TMPDIR"] = str(home / "tmp")
    return environment


def classify_claude_failure(stderr: str, events: str = "") -> str | None:
    if not isinstance(stderr, str) or not isinstance(events, str):
        raise TypeError("Claude failure diagnostics must be strings")
    haystack = f"{stderr}\n{events}".casefold()
    if any(signal in haystack for signal in _AUTH_FAILURE_SIGNALS):
        return "CLAUDE_AUTH_FAILURE"
    return None


def build_revision_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return (
        prompt.rstrip()
        + "\n\n"
        "MetaHarness revision constraints:\n"
        "- Read and analyze the worktree, then edit only files authorized by the implementation contract.\n"
        "- Do not create commits, move HEAD, switch branches, create or remove worktrees, or push.\n"
        "- Runtime capabilities are deliberately restricted; use only the tools exposed by MetaHarness.\n"
        "- The harness, not you, runs deterministic checks. Finish with a concise revision report.\n"
    )


def _tail(path: Path, limit: int) -> str:
    if limit <= 0 or not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def _scan_events(path: Path) -> tuple[dict[str, int], str | None, dict[str, Any] | None]:
    usage: dict[str, int] = {}
    final: str | None = None
    terminal: dict[str, Any] | None = None
    if not path.exists():
        return usage, final, terminal
    with path.open("rb") as stream:
        skipping = False
        while True:
            chunk = stream.readline(_MAX_EVENT_LINE_BYTES)
            if not chunk:
                break
            complete = chunk.endswith(b"\n")
            if skipping or (not complete and len(chunk) >= _MAX_EVENT_LINE_BYTES):
                skipping = not complete
                continue
            event = parse_event(chunk.decode("utf-8", errors="replace"))
            if event is None:
                continue
            found_usage = extract_usage(event)
            if found_usage is not None:
                usage = normalize_usage(found_usage)
            found_final = extract_final(event)
            if found_final is not None:
                final = found_final
            found_terminal = extract_terminal_result(event)
            if found_terminal is not None:
                terminal = found_terminal
    return usage, final, terminal


class ClaudeCodeAgent:
    """Run one isolated Claude Code revision process."""

    def __init__(
        self,
        *,
        executable: str = "claude",
        interrupt_grace_seconds: float = 5.0,
        stderr_tail_bytes: int = _DEFAULT_TAIL_BYTES,
    ) -> None:
        self.executable = executable
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self.stderr_tail_bytes = stderr_tail_bytes

    def build_argv(
        self, worktree: str | Path, *, profile: ModelProfile, claude_home: str | Path
    ) -> list[str]:
        if profile.driver is not ProfileDriver.CLAUDE_CODE:
            raise ClaudeAgentError("profile driver is not claude-code")
        if not profile.model or not profile.effort or not profile.permission_mode:
            raise ClaudeAgentError("Claude Code profile is incomplete")
        worktree_path = Path(worktree).expanduser().resolve()
        home = Path(claude_home).expanduser().resolve()
        # ``--print`` with ``--output-format stream-json`` is rejected by the
        # Claude Code CLI unless ``--verbose`` is present.  It is part of the
        # authoritative argv and deliberately not configurable.
        return [
            self.executable,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            # Safe mode, unlike ``--bare``, keeps Claude subscription/OAuth
            # authentication available from the managed CLAUDE_CONFIG_DIR
            # while disabling foreign customizations.  ``--restricted``
            # remains the authority for tool and filesystem confinement.
            "--safe-mode",
            "--restricted",
            "--tools",
            _REVISION_TOOLS,
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--model",
            profile.model,
            "--effort",
            profile.effort,
            "--permission-mode",
            profile.permission_mode,
            "--settings",
            str(home / "settings.json"),
            "--strict-mcp-config",
            "--mcp-config",
            str(home / "empty-mcp.json"),
        ]

    def run_revision(
        self,
        prompt: str,
        worktree: Path,
        *,
        artifacts_dir: Path,
        profile: ModelProfile,
        environment: Mapping[str, str],
        revision_dir: Path | None = None,
    ) -> ClaudeResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        worktree_path = Path(worktree).expanduser().resolve()
        directory = (
            Path(revision_dir).expanduser().resolve()
            if revision_dir is not None
            else Path(artifacts_dir).expanduser().resolve() / "revision"
        )
        directory.mkdir(parents=True, exist_ok=True)
        prompt_text = build_revision_prompt(prompt)
        prompt_path = directory / "agent.prompt.txt"
        events_path = directory / "agent.events.jsonl"
        stderr_path = directory / "agent.stderr.log"
        final_path = directory / "agent.final.md"
        result_path = directory / "agent.result.json"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        try:
            expected_head = current_head(worktree_path)
        except GitError as exc:
            raise ClaudeAgentError(f"cannot establish worktree HEAD: {exc}") from exc
        home_value = environment.get("CLAUDE_CONFIG_DIR")
        if not isinstance(home_value, str) or not home_value:
            raise ClaudeAgentError("CLAUDE_CONFIG_DIR is not forced")
        argv = self.build_argv(worktree_path, profile=profile, claude_home=Path(home_value))
        try:
            with prompt_path.open("rb") as stdin, events_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                exit_code, timed_out = run_bounded(
                    argv,
                    cwd=worktree_path,
                    timeout_seconds=profile.timeout_seconds,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    interrupt_signal=signal.SIGINT,
                    grace_seconds=self.interrupt_grace_seconds,
                    env=dict(environment),
                )
        except (OSError, ValueError) as exc:
            raise ClaudeAgentError(f"could not start Claude Code: {exc}") from exc
        try:
            actual_head = current_head(worktree_path)
        except GitError as exc:
            raise ClaudeAgentError(f"cannot establish worktree HEAD: {exc}") from exc
        if actual_head != expected_head:
            raise ClaudeCommittedError(expected_head, actual_head)
        usage, event_final, terminal = _scan_events(events_path)
        if final_path.exists():
            final_message = final_path.read_text(encoding="utf-8", errors="replace")
        else:
            final_message = event_final or ""
            final_path.write_text(final_message, encoding="utf-8")
        result = ClaudeResult(
            exit_code=124 if timed_out else exit_code,
            timed_out=timed_out,
            final_message=final_message,
            usage=usage,
            stderr_tail=_tail(stderr_path, self.stderr_tail_bytes),
            terminal_type=terminal["type"] if terminal is not None else None,
            terminal_subtype=terminal["subtype"] if terminal is not None else None,
            terminal_is_error=terminal["is_error"] if terminal is not None else None,
            terminal_num_turns=terminal["num_turns"] if terminal is not None else None,
            terminal_stop_reason=terminal["stop_reason"] if terminal is not None else None,
            terminal_errors=terminal["errors"] if terminal is not None else (),
        )
        result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


__all__ = [
    "ClaudeAgentError",
    "ClaudeCodeAgent",
    "ClaudeCommittedError",
    "ClaudeResult",
    "build_claude_environment",
    "build_revision_prompt",
    "classify_claude_failure",
    "_REVISION_TOOLS",
]
