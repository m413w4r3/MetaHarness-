"""Local authentication probe for the managed Claude Code runtime."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ClaudeAuthStatus:
    available: bool
    detail: str


_STATUS_COMMAND = re.compile(r"(?:^|\s)status(?:\s|$)", re.IGNORECASE | re.MULTILINE)
_AUTH_FAILURE_WORDS = (
    "not logged in",
    "not authenticated",
    "logged out",
    "authentication required",
)
_TIMEOUT = 20


def _run_local(
    argv: list[str], home: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str] | None:
    try:
        process_environment = dict(environment)
        process_environment["CLAUDE_CONFIG_DIR"] = str(home)
        return subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_TIMEOUT,
            env=process_environment,
            cwd=home,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def check_claude_authentication(
    claude_home: Path, *, environment: Mapping[str, str]
) -> ClaudeAuthStatus:
    """Verify auth only through ``claude auth --help`` and ``auth status``."""

    executable = shutil.which("claude", path=environment.get("PATH", ""))
    if executable is None:
        return ClaudeAuthStatus(False, "could not verify")
    help_result = _run_local([executable, "auth", "--help"], claude_home, environment)
    if help_result is None:
        return ClaudeAuthStatus(False, "could not verify")
    help_text = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
    if not _STATUS_COMMAND.search(help_text):
        return ClaudeAuthStatus(False, "could not verify")
    status_result = _run_local([executable, "auth", "status"], claude_home, environment)
    if status_result is None:
        return ClaudeAuthStatus(False, "could not verify")
    output = ((status_result.stdout or "") + "\n" + (status_result.stderr or "")).casefold()
    if any(marker in output for marker in _AUTH_FAILURE_WORDS):
        return ClaudeAuthStatus(False, "unavailable")
    if status_result.returncode == 0:
        return ClaudeAuthStatus(True, "available")
    return ClaudeAuthStatus(False, "unavailable")


__all__ = ["ClaudeAuthStatus", "check_claude_authentication"]
