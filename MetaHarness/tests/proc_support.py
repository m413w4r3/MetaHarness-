"""Deterministic process-liveness helpers shared by the test suite.

Tests about process groups used to wait for a descendant's scheduled late
write and then check that it never happened. Observing the descendant itself
is both faster and stronger: a process that no longer runs cannot mutate
anything afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path


def process_is_gone(pid: int) -> bool:
    """Return whether *pid* can no longer execute any code.

    A zombie (exited but not yet reaped by its new parent) counts as gone.
    """

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # The PID exists and belongs to someone else: be conservative.
        return False
    proc = Path("/proc")
    if not (proc / "self" / "stat").exists():
        return False
    try:
        stat = (proc / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    # The command name is parenthesised and may contain spaces.
    return stat.rsplit(")", 1)[1].split()[0] in {"Z", "X"}


def read_pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())
