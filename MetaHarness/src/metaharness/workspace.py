"""Deterministic setup commands for fresh run worktrees."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .gitops import GitError, status_porcelain
from .models import WorkspaceSetupCommand
from .procutil import read_capped, run_bounded
from .redaction import redact, redact_file
from .result import atomic_write_text


@dataclass(frozen=True)
class WorkspaceSetupResult:
    name: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str


class WorkspaceSetupError(RuntimeError):
    """A workspace setup command cannot safely be followed by the agent."""

    def __init__(self, reason: str, results: tuple[WorkspaceSetupResult, ...] = ()):
        self.code = reason
        self.results = results
        super().__init__(reason)


_TAIL_BYTES = 4096


def _safe_name(name: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip(".") or "setup"
    candidate = stem
    index = 2
    while candidate in used:
        candidate = f"{stem}-{index}"
        index += 1
    used.add(candidate)
    return candidate


def _cwd(worktree: Path, command: WorkspaceSetupCommand) -> Path:
    if Path(command.cwd).is_absolute() or "\x00" in command.cwd:
        raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED")
    root = worktree.resolve()
    candidate = (root / command.cwd).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED") from None
    if not candidate.is_dir():
        raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED")
    return candidate


def _persist_results(path: Path, results: tuple[WorkspaceSetupResult, ...]) -> None:
    atomic_write_text(path, json.dumps([asdict(item) for item in results], indent=2) + "\n")


def prepare_workspace(
    worktree: Path,
    commands: tuple[WorkspaceSetupCommand, ...],
    *,
    environment: Mapping[str, str],
    artifacts_dir: Path,
    secrets: tuple[str, ...],
) -> tuple[WorkspaceSetupResult, ...]:
    """Run trusted argv setup commands while requiring a Git-clean candidate."""

    root = Path(worktree).expanduser().resolve()
    artifact_root = Path(artifacts_dir).expanduser().resolve()
    try:
        artifact_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED")
    setup_dir = artifact_root / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    results: list[WorkspaceSetupResult] = []
    _persist_results(setup_dir / "results.json", tuple(results))
    used: set[str] = set()

    try:
        if status_porcelain(root):
            raise WorkspaceSetupError("WORKSPACE_SETUP_MUTATED")
    except GitError:
        raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED") from None

    for command in commands:
        safe_name = redact(command.name, secrets)
        stem = _safe_name(safe_name, used)
        stdout_path = setup_dir / f"{stem}.stdout.log"
        stderr_path = setup_dir / f"{stem}.stderr.log"
        try:
            cwd = _cwd(root, command)
            command_environment = {
                name: environment[name]
                for name in command.env_allowlist
                if name in environment
            }
            started = time.monotonic()
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    exit_code, timed_out = run_bounded(
                        command.argv,
                        cwd=cwd,
                        timeout_seconds=command.timeout_seconds,
                        stdout=stdout,
                        stderr=stderr,
                        env=command_environment,
                    )
                except (OSError, ValueError) as exc:
                    stderr.write(f"could not start workspace setup: {type(exc).__name__}\n".encode())
                    exit_code, timed_out = -1, False
            duration = time.monotonic() - started
            redact_file(stdout_path, secrets)
            redact_file(stderr_path, secrets)
            stdout_text, _ = read_capped(stdout_path, _TAIL_BYTES)
            stderr_text, _ = read_capped(stderr_path, _TAIL_BYTES)
            result = WorkspaceSetupResult(
                name=safe_name,
                exit_code=124 if timed_out else exit_code,
                timed_out=timed_out,
                duration_seconds=duration,
                stdout_tail=stdout_text,
                stderr_tail=stderr_text,
            )
            results.append(result)
            frozen = tuple(results)
            _persist_results(setup_dir / "results.json", frozen)
            try:
                mutated = bool(status_porcelain(root))
            except GitError:
                raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED", frozen) from None
            if mutated:
                raise WorkspaceSetupError("WORKSPACE_SETUP_MUTATED", frozen)
            if timed_out:
                raise WorkspaceSetupError("WORKSPACE_SETUP_TIMEOUT", frozen)
            if exit_code != 0:
                raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED", frozen)
        except WorkspaceSetupError:
            raise
        except (OSError, ValueError):
            frozen = tuple(results)
            _persist_results(setup_dir / "results.json", frozen)
            raise WorkspaceSetupError("WORKSPACE_SETUP_FAILED", frozen) from None
    return tuple(results)


__all__ = [
    "WorkspaceSetupError",
    "WorkspaceSetupResult",
    "prepare_workspace",
]
