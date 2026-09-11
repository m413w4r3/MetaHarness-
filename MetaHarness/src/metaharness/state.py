"""Persistance atomique de l'état d'un run."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunStateStore:
    """Store one run state in a JSON file, using atomic replacements."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def initialize(
        self,
        run_id: str,
        *,
        started_at: str | None = None,
        base_sha: str | None = None,
        branch: str | None = None,
        worktree: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        started = started_at or _now()
        state: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": RunStatus.CREATED.value,
            "started_at": started,
            "updated_at": started,
            "base_sha": base_sha,
            "branch": branch,
            "worktree": worktree,
            "planner": {},
            "execution": {},
            "plan_identity": None,
            "agent": {},
            "checks": [],
            "review": {},
            "approved_tree_sha": None,
            "commit_sha": None,
            "failure": None,
        }
        self._write(state)
        return state

    def update(self, *, status: RunStatus | str, **fields: Any) -> dict[str, Any]:
        state = self.load()
        state["status"] = RunStatus(status).value
        state.update(fields)
        state["updated_at"] = _now()
        self._write(state)
        return state

    def record_failure(
        self, reason: str, detail: Any = None
    ) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        failure: dict[str, Any] = {"reason": reason}
        if detail is not None:
            failure["detail"] = detail
        state = self.load()
        state["status"] = RunStatus.FAILED.value
        state["failure"] = failure
        state["updated_at"] = _now()
        self._write(state)
        return state

    def load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        if not isinstance(state, dict):
            raise ValueError("run state must contain a JSON object")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            fd, temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                json.dump(state, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
