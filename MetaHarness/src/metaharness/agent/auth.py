"""Local authentication checks for the managed Codex CLI runtime."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CodexAuthStatus:
    """Safe, non-secret result of a managed Codex authentication check."""

    available: bool
    detail: str


_AUTH_UNAVAILABLE = "codex authentication is unavailable"
_AUTH_UNVERIFIED = "codex authentication could not be verified"
_AUTH_STATE_FILENAMES = ("auth.json",)
_AUTH_FAILURE_WORDS = (
    "not logged in",
    "not authenticated",
    "logged out",
    "authentication required",
)
_STATUS_COMMAND = re.compile(r"^\s+status(?:\s|$)", re.IGNORECASE | re.MULTILINE)
_AUTH_CHECK_TIMEOUT_SECONDS = 20


def _codex_executable(environment: Mapping[str, str]) -> str | None:
    path = environment.get("PATH", "")
    return shutil.which("codex", path=path) if isinstance(path, str) else None


def _run_local(
    argv: list[str], codex_home: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str] | None:
    try:
        process_environment = dict(environment)
        process_environment["CODEX_HOME"] = str(codex_home)
        return subprocess.run(
            argv,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_AUTH_CHECK_TIMEOUT_SECONDS,
            env=process_environment,
            cwd=codex_home,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _status_result(result: subprocess.CompletedProcess[str]) -> CodexAuthStatus:
    # Only fixed status phrases are inspected.  Command output is never
    # returned, logged, or parsed as a credential.
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).casefold()
    if any(marker in output for marker in _AUTH_FAILURE_WORDS):
        return CodexAuthStatus(False, _AUTH_UNAVAILABLE)
    if result.returncode == 0:
        return CodexAuthStatus(True, "codex authentication available")
    return CodexAuthStatus(False, _AUTH_UNVERIFIED)


def _fallback_auth_state(codex_home: Path) -> CodexAuthStatus:
    """Check only the current CLI's non-empty local auth state marker."""

    for filename in _AUTH_STATE_FILENAMES:
        state_path = codex_home / filename
        try:
            stat = state_path.stat()
        except OSError:
            continue
        if stat.st_mode & 0o170000 == 0o100000 and stat.st_size > 0:
            return CodexAuthStatus(True, "codex authentication state available")
    return CodexAuthStatus(False, _AUTH_UNVERIFIED)


def check_codex_authentication(
    codex_home: Path,
    *,
    environment: Mapping[str, str],
) -> CodexAuthStatus:
    """Verify Codex authentication without making a model request.

    The installed CLI's local status command is authoritative when its
    presence is advertised by ``codex login --help``.  Older CLIs without a
    reliable status command fall back to the non-empty local state file known
    to that CLI family.  Neither path reads credential contents.
    """

    if not isinstance(codex_home, Path):
        raise TypeError("codex_home must be a Path")
    executable = _codex_executable(environment)
    if executable is None:
        return CodexAuthStatus(False, _AUTH_UNVERIFIED)

    help_result = _run_local(
        [executable, "login", "--help"], codex_home, environment
    )
    if help_result is None:
        return CodexAuthStatus(False, _AUTH_UNVERIFIED)
    help_text = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
    if _STATUS_COMMAND.search(help_text):
        status_result = _run_local(
            [executable, "login", "status"], codex_home, environment
        )
        if status_result is None:
            return CodexAuthStatus(False, _AUTH_UNVERIFIED)
        return _status_result(status_result)
    return _fallback_auth_state(codex_home)


__all__ = ["CodexAuthStatus", "check_codex_authentication"]
