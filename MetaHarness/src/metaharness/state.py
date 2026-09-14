"""Persistance atomique de l'état d'un run."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import RunStatus

# Lock file serializing every read/modify/write of ``state.json``.  It is an
# internal coordination file: it never contains state and is never served.
STATE_LOCK_NAME = "state.lock"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@contextmanager
def _exclusive_state_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on *lock_path* (Linux).

    ``flock`` locks belong to the open file description, so two threads of one
    process opening the file independently exclude each other exactly like two
    processes (web server and orchestrator) do.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class RunStateStore:
    """Store one run state in a JSON file, using atomic replacements.

    Every mutation reads, merges and replaces the file under one exclusive
    lock, so concurrent writers (orchestrator thread, web approval) can never
    lose each other's fields or resurrect an older status.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.parent / STATE_LOCK_NAME

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
            "recommendation": {},
            "plan_identity": None,
            "agent": {},
            "agent_candidate_tree_before": None,
            "agent_candidate_tree_after": None,
            "workspace_setup": [],
            "checks": [],
            "review": {},
            "revision": {},
            "review_iterations": 0,
            "cycle": 1,
            "cycles": [],
            "approved_tree_sha": None,
            "commit_sha": None,
            "failure": None,
            "steps": [],
            "current_step": None,
            "agent_usage": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "steps": [],
            },
            "usage": {
                "planner": {},
                "luna": {"total": {}, "steps": []},
                "reviser": {},
                "reviewer": {},
                "grand_total": {},
            },
        }
        with _exclusive_state_lock(self.lock_path):
            self._write(state)
        return state

    def update(self, *, status: RunStatus | str, **fields: Any) -> dict[str, Any]:
        new_status = RunStatus(status).value
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            state["status"] = new_status
            state.update(fields)
            state["updated_at"] = _now()
            self._write(state)
        return state

    def update_if_status(
        self,
        expected_status: RunStatus | str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Compare-and-set: merge *fields* only while status is *expected*.

        Under one lock: read the current state, return ``None`` without
        writing if its status differs, otherwise merge and atomically write.
        The status itself is not changed.
        """

        expected = RunStatus(expected_status).value
        if "status" in fields:
            raise ValueError("update_if_status does not change the status")
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            if state.get("status") != expected:
                return None
            state.update(fields)
            state["updated_at"] = _now()
            self._write(state)
        return state

    def record_failure(
        self, reason: str, detail: Any = None, **fields: Any
    ) -> dict[str, Any]:
        """Mark the run FAILED; *fields* are merged in the same write."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if "status" in fields or "failure" in fields:
            raise ValueError("record_failure owns status and failure")
        failure: dict[str, Any] = {"reason": reason}
        if detail is not None:
            failure["detail"] = detail
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            state.update(fields)
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
