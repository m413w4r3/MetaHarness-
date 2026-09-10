"""Bounded execution of external processes.

Every long-running external command (Codex, deterministic checks, the context
locator) goes through :func:`run_bounded`:

- ``shell=False`` always; the argv comes from trusted configuration;
- the child runs in its own session, so the complete process group can be
  signalled;
- stdout and stderr go to files, never to pipes, so the deadline does not
  depend on anybody reading or writing a pipe;
- on timeout the group receives ``interrupt_signal``, then ``SIGKILL`` after a
  bounded grace period;
- after the leader exits, any process still alive in its group is terminated,
  so a lingering background child cannot modify the worktree after the
  harness has taken its evidence snapshot.

A descendant that deliberately creates a new session escapes the group; V0
does not use cgroups and documents this limitation.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import IO, Sequence

_POLL_SECONDS = 0.05
_LEFTOVER_GRACE_SECONDS = 1.0


def _signal_group(pgid: int, sig: signal.Signals) -> bool:
    """Signal a process group; return whether the group still existed."""

    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A member changed credentials; nothing more can be done safely.
        return False
    return True


def _group_alive(pgid: int) -> bool:
    return _signal_group(pgid, 0)  # type: ignore[arg-type]


def _wait_group_gone(pgid: int, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
    return True


def terminate_group(pgid: int, *, grace_seconds: float = _LEFTOVER_GRACE_SECONDS) -> None:
    """Terminate every process still alive in *pgid* in bounded time."""

    if not _signal_group(pgid, signal.SIGTERM):
        return
    if not _wait_group_gone(pgid, grace_seconds):
        _signal_group(pgid, signal.SIGKILL)
        _wait_group_gone(pgid, grace_seconds)


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    stdout: IO[bytes],
    stderr: IO[bytes],
    stdin: IO[bytes] | None = None,
    interrupt_signal: signal.Signals = signal.SIGTERM,
    grace_seconds: float = 5.0,
    env: dict[str, str] | None = None,
) -> tuple[int, bool]:
    """Run *argv* and return ``(exit_code, timed_out)``.

    ``OSError``/``ValueError`` are raised when the process cannot be started.
    A ``KeyboardInterrupt`` kills the whole group before propagating.
    """

    if isinstance(argv, str) or not argv:
        raise ValueError("argv must be a non-empty sequence of strings")
    process = subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        stdin=stdin if stdin is not None else subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=True,
        env=env,
    )
    pgid = process.pid
    timed_out = False
    try:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                _signal_group(pgid, interrupt_signal)
                grace_deadline = time.monotonic() + max(0.0, grace_seconds)
                while process.poll() is None and time.monotonic() < grace_deadline:
                    time.sleep(_POLL_SECONDS)
                if process.poll() is None:
                    _signal_group(pgid, signal.SIGKILL)
                break
            time.sleep(_POLL_SECONDS)
        exit_code = process.wait()
    finally:
        if process.poll() is None:
            _signal_group(pgid, signal.SIGKILL)
            process.wait()
        # Background descendants must not outlive the command: they could
        # otherwise mutate the candidate after its evidence snapshot.
        terminate_group(pgid)
    return exit_code, timed_out


def read_capped(path: Path, max_bytes: int) -> tuple[str, bool]:
    """Read at most the last *max_bytes* of a log file as text."""

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "", False
    with path.open("rb") as stream:
        if size > max_bytes:
            stream.seek(size - max_bytes)
            data = stream.read(max_bytes)
            text = data.decode("utf-8", errors="replace")
            omitted = size - max_bytes
            return f"[... {omitted} earlier bytes omitted; see the full log file ...]\n{text}", True
        return stream.read().decode("utf-8", errors="replace"), False


__all__ = ["read_capped", "run_bounded", "terminate_group"]
