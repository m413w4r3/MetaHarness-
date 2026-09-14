"""Public results and small durable artifact helpers for one run."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RunStatus


@dataclass(frozen=True)
class RunResult:
    """The outcome returned by :class:`~metaharness.orchestrator.Orchestrator`."""

    run_dir: Path
    status: RunStatus
    state: dict[str, Any]

    @property
    def committed(self) -> bool:
        return self.status in {RunStatus.COMMITTED, RunStatus.PUBLISHED}

    @property
    def published(self) -> bool:
        return self.status is RunStatus.PUBLISHED

    @property
    def commit_sha(self) -> str | None:
        value = self.state.get("commit_sha")
        return value if isinstance(value, str) else None

    @property
    def failure_reason(self) -> str | None:
        failure = self.state.get("failure")
        if isinstance(failure, dict) and isinstance(failure.get("reason"), str):
            return failure["reason"]
        return None


class ResultArtifactError(RuntimeError):
    """A run artifact could not be written safely."""


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write an artifact with replace-and-fsync semantics."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ResultArtifactError(f"could not write artifact {target}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_repair_task(run_dir: str | Path, *, fields: dict[str, Any]) -> None:
    """Persist the single explicit repair task requested by a REVISE review."""

    directory = Path(run_dir).expanduser().resolve()
    markdown = "\n".join(
        [
            "# MetaHarness repair task",
            "",
            f"Route: {fields['route']}",
            f"Run ID: {fields['run_id']}",
            f"Existing branch: {fields['existing_branch']}",
            f"Existing worktree: {fields['existing_worktree']}",
            "",
            "## Review summary",
            str(fields["review_summary"]),
            "",
            "## Findings",
            str(fields["findings"]),
            "",
            "## Required fixes",
            str(fields["required_fixes"]),
            "",
            "## Missing tests",
            str(fields["missing_tests"]),
            "",
        ]
    )
    atomic_write_text(directory / "repair_task.md", markdown)
    atomic_write_text(
        directory / "repair_task.json",
        json.dumps(fields, ensure_ascii=False, indent=2) + "\n",
    )


__all__ = ["ResultArtifactError", "RunResult", "atomic_write_text", "write_repair_task"]
