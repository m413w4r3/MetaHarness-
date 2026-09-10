"""Execution of the deterministic checks configured for a run.

The important boundary in this module is that the command line comes only
from :attr:`HarnessConfig.checks`.  No text produced by another agent is ever
interpreted as a command.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gitops import current_head, status_porcelain
from .models import CheckConfig, HarnessConfig


DEFAULT_TAIL_BYTES = 4_096


class ValidationError(RuntimeError):
    """A configured check cannot be run safely."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    stdout_log: str
    stderr_log: str
    stdout_tail: str
    stderr_tail: str
    workspace_mutated: bool


@dataclass(frozen=True)
class _GitSnapshot:
    """The content that a subsequent ``git add -A`` would submit."""

    head: str
    tracked_diff: str
    untracked: tuple[tuple[str, str], ...]


def bounded_tail(value: str, max_bytes: int = DEFAULT_TAIL_BYTES) -> str:
    """Return a UTF-8 tail whose encoded size is at most *max_bytes*."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return encoded[-max_bytes:].decode("utf-8", errors="ignore") if max_bytes else ""


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _file_fingerprint(path: Path) -> str:
    """Fingerprint one untracked path without following a final symlink."""

    try:
        stat = path.lstat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_mode & 0o7777).encode("ascii"))
        if path.is_symlink():
            digest.update(b"symlink:")
            digest.update(path.readlink().as_posix().encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"special-or-directory")
        return digest.hexdigest()
    except (OSError, ValueError):
        return "<unreadable>"


def _untracked_snapshot(worktree: Path, status: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for line in status:
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        # Git quotes unusual paths in the non -z porcelain format.  Such a
        # path is still safely represented by the status line; normal paths,
        # including spaces, retain their exact spelling here.
        if relative.startswith('"') and relative.endswith('"'):
            relative = relative[1:-1]
        entries.append((relative, _file_fingerprint(worktree / relative)))
    return tuple(sorted(entries))


def _tracked_diff(worktree: Path) -> str:
    """Return the complete tracked working-tree delta from HEAD."""

    result = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValidationError(f"could not snapshot Git diff: {detail}")
    return result.stdout


def _git_snapshot(worktree: Path) -> _GitSnapshot:
    status = status_porcelain(worktree)
    return _GitSnapshot(
        head=current_head(worktree),
        tracked_diff=_tracked_diff(worktree),
        untracked=_untracked_snapshot(worktree, status),
    )


def resolve_check_cwd(worktree: str | Path, check: CheckConfig) -> Path:
    """Resolve and validate a check's cwd inside the worktree.

    ``Path.resolve`` is deliberately used before the containment test so a
    symlink cannot make a seemingly relative cwd escape the worktree.
    """

    root = Path(worktree).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"worktree is not a directory: {root}")
    if "\x00" in check.cwd:
        raise ValidationError(f"check cwd contains NUL: {check.name}")
    try:
        candidate = (Path(worktree).expanduser() / check.cwd).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(f"invalid cwd for check {check.name!r}: {exc}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationError(
            f"check cwd escapes worktree: {check.name!r} -> {check.cwd!r}"
        ) from exc
    if not candidate.is_dir():
        raise ValidationError(f"check cwd is not a directory: {check.cwd!r}")
    return candidate


def _safe_log_stem(name: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip(".") or "check"
    candidate = stem
    index = 2
    while candidate in used:
        candidate = f"{stem}-{index}"
        index += 1
    used.add(candidate)
    return candidate


def _persist_logs(
    logs_dir: Path,
    result: CheckResult,
    stem: str,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"{stem}.stdout.log").write_text(result.stdout_log, encoding="utf-8")
    (logs_dir / f"{stem}.stderr.log").write_text(result.stderr_log, encoding="utf-8")


def run_checks(
    worktree: str | Path,
    config: HarnessConfig,
    *,
    logs_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> tuple[CheckResult, ...]:
    """Run every configured check, continuing after failures and timeouts."""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    root = Path(worktree).expanduser().resolve()
    output_dir = Path(logs_dir).expanduser().resolve() if logs_dir is not None else None
    if output_dir is not None:
        try:
            output_dir.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValidationError("logs_dir must be outside the worktree")
    used_log_stems: set[str] = set()
    results: list[CheckResult] = []

    for check in config.checks:
        cwd = resolve_check_cwd(root, check)
        before = _git_snapshot(root)
        started = time.monotonic()
        stdout = ""
        stderr = ""
        exit_code = -1
        timed_out = False
        try:
            completed = subprocess.run(
                list(check.argv),
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=check.timeout_seconds,
                check=False,
                shell=False,
            )
            stdout = _decode_output(completed.stdout)
            stderr = _decode_output(completed.stderr)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
            stderr = f"{stderr}\ncheck timed out after {check.timeout_seconds}s".lstrip()
        except (OSError, ValueError) as exc:
            stderr = f"could not start check: {exc}"
        duration = time.monotonic() - started
        after = _git_snapshot(root)
        result = CheckResult(
            name=check.name,
            argv=tuple(check.argv),
            cwd=str(cwd),
            exit_code=exit_code,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_log=stdout,
            stderr_log=stderr,
            stdout_tail=bounded_tail(stdout, tail_bytes),
            stderr_tail=bounded_tail(stderr, tail_bytes),
            workspace_mutated=before != after,
        )
        results.append(result)
        if output_dir is not None:
            _persist_logs(output_dir, result, _safe_log_stem(check.name, used_log_stems))

    return tuple(results)


def check_result_json(result: CheckResult, *, log_stem: str | None = None) -> dict[str, Any]:
    """Serialize reviewer-safe check metadata; full logs remain in log files."""

    data = asdict(result)
    data.pop("stdout_log")
    data.pop("stderr_log")
    if log_stem is not None:
        data["stdout_log_path"] = f"checks/{log_stem}.stdout.log"
        data["stderr_log_path"] = f"checks/{log_stem}.stderr.log"
    return data


# Descriptive aliases make the boundary easy to find for callers.
execute_checks = run_checks
validate_checks = run_checks
validate = run_checks
