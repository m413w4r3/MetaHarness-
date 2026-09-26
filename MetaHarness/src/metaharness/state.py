"""Persistance atomique de l'état d'un run.

Un run n'a qu'une source durable d'état : sa phase (l'opération courante ou
prochaine, portée par le checkpoint) et sa disposition
(:class:`~metaharness.models.RunDisposition`, l'une des cinq postures).  Le
``status`` stocké ici n'est qu'une projection dérivée, calculée en un seul
endroit (:func:`~metaharness.models.project_run_outcome`) ; aucun écrivain ne
peut donc lui faire contredire la disposition.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .checkpoint_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointFormatError,
    CheckpointStamp,
    read_checkpoint_stamp,
)
from .models import (
    RunDisposition,
    RunEvent,
    RunIdentity,
    RunMachineState,
    RunOutcome,
    RunPhase,
    RunStatus,
    assemble_run_state,
    project_run_outcome,
    transition,
)

# Lock file serializing every read/modify/write of ``state.json``.  It is an
# internal coordination file: it never contains state and is never served.
STATE_LOCK_NAME = "state.lock"
# The fields of the canonical machine state.  One metadata mutation can never
# carry one: only ``set_run_state`` and ``transition_run`` own them.
CONTROL_FIELDS = frozenset({"status", "disposition", "phase", "reason"})


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


class RunCheckpointError(ValueError):
    """A run checkpoint exists but cannot be the phase authority.

    The store refuses here instead of reading the phase the state file happens
    to record: one command has one authority, and a corrupt checkpoint is an
    incident rather than a gap.  Only the terminal reporting path
    (:meth:`RunStateStore.record_failure`, :meth:`RunStateStore.reported_machine_state`)
    survives it, and only to describe where the run stopped.
    """


class RunStateStore:
    """Store one run state in a JSON file, using atomic replacements.

    Every mutation reads, merges and replaces the file under one exclusive
    lock, so concurrent writers (orchestrator thread, web approval) can never
    lose each other's fields or resurrect an older posture.

    The store owns the canonical :class:`RunMachineState` of the run: the
    checkpoint owns the phase, the file records the posture and its reason, and
    ``status`` is only ever the projection of the two.  No caller can name a
    status, and a metadata mutation can never move the machine.

    Two reads are named explicitly.  The control read
    (:meth:`machine_state`, :meth:`identity`, :meth:`update_metadata`,
    :meth:`set_run_state`, :meth:`transition_run`) treats the checkpoint as the
    sole phase authority and refuses a checkpoint it cannot read; the terminal
    reporting read (:meth:`record_failure`, :meth:`reported_machine_state`)
    still describes the last recorded phase so a corruption never prevents its
    own diagnostic.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.parent / STATE_LOCK_NAME

    def initialize(
        self,
        run_id: str,
        *,
        pipeline_version: int = 2,
        started_at: str | None = None,
        base_sha: str | None = None,
        branch: str | None = None,
        worktree: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(pipeline_version, bool) or pipeline_version != 2:
            raise ValueError("pipeline_version must be 2")
        started = started_at or _now()
        state: dict[str, Any] = {
            "schema_version": 1,
            "pipeline_version": pipeline_version,
            "run_id": run_id,
            "status": RunStatus.CREATED.value,
            "disposition": RunDisposition.RUNNING.value,
            "phase": None,
            "reason": None,
            "started_at": started,
            "updated_at": started,
            "base_sha": base_sha,
            "branch": branch,
            "worktree": worktree,
            "run_options_sha256": None,
            "run_options": {},
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
            "remote_branch": None,
            "remote_sha": None,
            "reviewed_candidate_sha": None,
            "issue_number": None,
            "pull_request_number": None,
            "approved_tree_sha": None,
            "commit_sha": None,
            "publish": {},
            "failure": None,
            "recovery_counters": {},
            "steps": [],
            "accepted_steps": [],
            "accepted_commits": [],
            "deferred_verifications": [],
            "current_step": None,
            "expected_head_sha": base_sha,
            "expected_parent_sha": base_sha,
            "expected_tree_sha": None,
            "next_step_id": None,
            "usage": {
                "planner": {},
                "correction_planner": {},
                "implementer": {"total": {}, "steps": []},
                "check_repair": {},
                "semantic_reviser": {},
                "final_reviewer": {},
                "cycles": [],
                "grand_total": {},
            },
        }
        with _exclusive_state_lock(self.lock_path):
            self._write(state)
        return state

    def update_metadata(
        self, *, expected: RunIdentity | None = None, **fields: Any,
    ) -> dict[str, Any] | None:
        """Merge *fields* without ever moving the machine state.

        The phase, the disposition, the reason and the projected status are
        re-derived from the durable facts, so a metadata writer can never
        contradict the checkpoint or pilot the run.  It reads the phase through
        the strict control checkpoint: a corrupt or foreign one refuses the
        write instead of merging under a guessed phase.  With *expected*, the
        merge happens only while the run still has exactly that canonical
        identity; ``None`` is returned without writing otherwise.
        """

        self._refuse_control_fields(fields)
        if expected is not None and not isinstance(expected, RunIdentity):
            raise TypeError("update_metadata expects a RunIdentity expectation")
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            stamp = self._control_stamp()
            machine = self._recorded_machine(state, stamp)
            if expected is not None and self._identity(state, machine, stamp) != expected:
                return None
            self._apply(state, machine, fields)
            self._write(state)
        return state

    def set_run_state(self, machine: RunMachineState, **fields: Any) -> dict[str, Any]:
        """Persist one canonical run state; the status is derived from it.

        The phase authority itself lives in the run checkpoint: this store
        only records the posture and its projection, so no second field can
        ever contradict the checkpoint's phase.  A machine state that names a
        phase the checkpoint does not own is refused, never silently applied,
        and so is a checkpoint this runtime cannot read.
        """

        if not isinstance(machine, RunMachineState):
            raise TypeError("set_run_state expects a RunMachineState")
        self._refuse_control_fields(fields)
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            stamp = self._control_stamp()
            self._apply(state, self._resolved_machine(state, stamp, machine), fields)
            self._write(state)
        return state

    def transition_run(
        self, event: RunEvent, *, expected: RunIdentity, **fields: Any,
    ) -> dict[str, Any] | None:
        """Compare-and-set the canonical run state under one event.

        Under one lock: return ``None`` without writing unless the run still
        has exactly the observed identity (phase, posture, state generation
        and checkpoint bytes); otherwise apply the event through the single
        state machine, merge *fields* and write atomically.  An illegal event
        is refused by :func:`~metaharness.models.transition` alone; a checkpoint
        this runtime cannot read refuses the transition before it is considered.
        """

        if not isinstance(event, RunEvent):
            raise TypeError("transition_run expects a RunEvent")
        if not isinstance(expected, RunIdentity):
            raise TypeError("transition_run expects the observed RunIdentity")
        self._refuse_control_fields(fields)
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            stamp = self._control_stamp()
            machine = self._recorded_machine(state, stamp)
            if self._identity(state, machine, stamp) != expected:
                return None
            self._apply(state, transition(machine, event), fields)
            self._write(state)
        return state

    def record_failure(
        self, reason: str, detail: str | Mapping[str, Any] | None = None, **fields: Any
    ) -> dict[str, Any]:
        """Record one escaped exception as the run's durable failure.

        It is the terminal projection of an exception, not a transition: the
        reason it carries is the machine reason, the posture becomes FAILED and
        *fields* are merged in the same write.

        The read is the terminal reporting one: it describes the last recorded
        phase and never refuses, so a corrupt checkpoint cannot stop the
        diagnostic that explains the run.  That phase stays descriptive — no
        RESUME or ADVANCE ever reads it back.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if detail is not None and not isinstance(detail, (str, Mapping)):
            raise TypeError("failure detail must be human-readable text or a structured mapping")
        if isinstance(detail, Mapping):
            if any(not isinstance(key, str) for key in detail):
                raise TypeError("structured failure detail keys must be strings")
            try:
                json.dumps(detail, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise TypeError("structured failure detail must contain JSON data") from exc
        self._refuse_control_fields(fields)
        if "failure" in fields:
            raise ValueError("record_failure owns the failure payload")
        failure: dict[str, Any] = {"reason": reason}
        if detail is not None:
            failure["detail"] = detail
        with _exclusive_state_lock(self.lock_path):
            state = self.load()
            machine = RunMachineState(
                self._recorded_phase(state, self._reporting_stamp()),
                RunDisposition.FAILED, reason,
            )
            self._apply(state, machine, {**fields, "failure": failure})
            self._write(state)
        return state

    # -- canonical reads -------------------------------------------------

    def machine_state(self) -> RunMachineState:
        """The canonical state of this run: the checkpoint phase and posture.

        Once a checkpoint exists it is the only phase authority.  A checkpoint
        this runtime cannot read refuses the read instead of silently falling
        back to the phase the state file records.
        """

        state = self.load()
        return self._recorded_machine(state, self._control_stamp())

    def reported_machine_state(self) -> RunMachineState:
        """The descriptive state a terminal report may still be written from.

        It is the reporting twin of :meth:`machine_state`: it never refuses, so
        a checkpoint that cannot be read degrades to the last recorded phase
        and the failure that explains the run stays recordable.  Nothing may
        decide from it — a transition, a resume and a metadata write all read
        :meth:`machine_state`.
        """

        state = self.load()
        return self._recorded_machine(state, self._reporting_stamp())

    def outcome(self) -> RunOutcome:
        """The projected view of the canonical state; never stored as a source."""

        return project_run_outcome(self.machine_state())

    def identity(self) -> RunIdentity:
        """The canonical identity a compare-and-set claim must observe."""

        state = self.load()
        stamp = self._control_stamp()
        return self._identity(state, self._recorded_machine(state, stamp), stamp)

    # -- internals -------------------------------------------------------

    @staticmethod
    def _refuse_control_fields(fields: Mapping[str, Any]) -> None:
        owned = fields.keys() & CONTROL_FIELDS
        if owned:
            raise ValueError(
                "the run machine owns " + ", ".join(sorted(owned))
                + "; use set_run_state or transition_run"
            )

    # -- the phase authority ---------------------------------------------

    def _control_stamp(self) -> CheckpointStamp | None:
        """The checkpoint that owns the phase, or ``None`` before the first one.

        Every control read and write goes through here, so the read is strict
        by construction: a checkpoint that is unreadable, structurally invalid
        or written by a foreign schema refuses the operation.  ``None`` means
        exactly one thing — no durable checkpoint was written yet, and the
        recorded phase is then the only boundary the run has.
        """

        try:
            stamp = read_checkpoint_stamp(self.path.parent)
        except CheckpointFormatError as exc:
            raise RunCheckpointError(
                f"the run checkpoint cannot own the phase: {exc}"
            ) from exc
        if stamp is None:
            return None
        if not stamp.current_schema:
            raise RunCheckpointError(
                f"the run checkpoint schema {stamp.schema_version} is not "
                f"{CHECKPOINT_SCHEMA_VERSION}: this runtime cannot read its phase"
            )
        return stamp

    def _reporting_stamp(self) -> CheckpointStamp | None:
        """The checkpoint a terminal report may still describe; it never refuses.

        Only :meth:`record_failure` and :meth:`reported_machine_state` read
        through here: a run whose checkpoint was corrupted must still be able
        to write why it stopped.  The result is descriptive — it names the
        last recorded phase — and it can never authorize RESUME or ADVANCE.
        """

        try:
            return read_checkpoint_stamp(self.path.parent)
        except CheckpointFormatError:
            return None

    @staticmethod
    def _phase_recorded_in(state: Mapping[str, Any]) -> RunPhase | None:
        recorded = state.get("phase")
        if not isinstance(recorded, str):
            return None
        try:
            return RunPhase(recorded)
        except ValueError:
            return None

    def _recorded_phase(
        self, state: Mapping[str, Any], stamp: CheckpointStamp | None,
    ) -> RunPhase | None:
        """The phase one operation must use: the checkpoint's, else the recorded one."""

        if stamp is not None:
            return stamp.phase
        return self._phase_recorded_in(state)

    def _recorded_machine(
        self, state: Mapping[str, Any], stamp: CheckpointStamp | None,
    ) -> RunMachineState:
        failure = state.get("failure")
        return assemble_run_state(
            self._recorded_phase(state, stamp),
            disposition=state.get("disposition"),
            status=state.get("status"),
            reason=state.get("reason"),
            failure_reason=(
                failure.get("reason") if isinstance(failure, Mapping) else None
            ),
        )

    def _resolved_machine(
        self, state: Mapping[str, Any], stamp: CheckpointStamp | None,
        machine: RunMachineState,
    ) -> RunMachineState:
        """Bind *machine* to the checkpoint phase, or refuse the contradiction."""

        if stamp is not None:
            if machine.phase is not None and machine.phase is not stamp.phase:
                raise ValueError(
                    f"the checkpoint owns the {stamp.phase.value} phase, not {machine.phase.value}"
                )
            return RunMachineState(stamp.phase, machine.disposition, machine.reason)
        phase = machine.phase if machine.phase is not None else self._phase_recorded_in(state)
        return RunMachineState(phase, machine.disposition, machine.reason)

    @staticmethod
    def _identity(
        state: Mapping[str, Any], machine: RunMachineState, stamp: CheckpointStamp | None,
    ) -> RunIdentity:
        updated_at = state.get("updated_at")
        return RunIdentity.of(
            machine,
            updated_at=updated_at if isinstance(updated_at, str) else None,
            checkpoint_sha256=stamp.sha256 if stamp is not None else None,
        )

    @staticmethod
    def _apply(
        state: dict[str, Any], machine: RunMachineState, fields: Mapping[str, Any],
    ) -> None:
        """Write one machine state, its projection and the merged metadata."""

        failure = fields.get("failure")
        if (
            isinstance(failure, Mapping)
            and failure.get("reason") is not None
            and failure.get("reason") != machine.reason
        ):
            raise ValueError("the failure reason contradicts the machine reason")
        state["status"] = project_run_outcome(machine).status.value
        state["disposition"] = machine.disposition.value
        state["phase"] = machine.phase.value if machine.phase is not None else None
        state["reason"] = machine.reason
        state.update(fields)
        state["updated_at"] = _now()

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
