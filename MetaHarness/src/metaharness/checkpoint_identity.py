"""The structural identity of a run checkpoint: schema, status, phase, bytes.

The checkpoint is the one durable owner of a run's phase, and two readers have
to agree on what it says without sharing the resume gate: the state store,
which must refuse to move a run whose checkpoint it cannot read, and the resume
gate, which validates the whole payload.  This module is their shared floor —
it decodes the file once, returns the three fields that frame it and the exact
SHA-256 of the bytes it read, and refuses anything it cannot establish.  It
knows no business rule, no Git identity and no orchestration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .models import RUN_CHECKPOINT_NAME, RunPhase

# The schema this runtime writes and reads.  Another one is not corruption,
# but it is not a phase authority either: the resume gate refuses it as
# ``RUN_SCHEMA_UNSUPPORTED`` and a control read refuses to read a phase from it.
CHECKPOINT_SCHEMA_VERSION = 4

CHECKPOINT_STATUSES = frozenset({"pending", "completed"})


class CheckpointFormatError(ValueError):
    """A checkpoint is unreadable or structurally invalid.

    It is never a gap: a caller that needs the durable phase must refuse its
    operation instead of reading a second, weaker authority.
    """


@dataclass(frozen=True)
class CheckpointStamp:
    """The exact bytes of one checkpoint and the fields that frame them."""

    schema_version: int
    status: str
    phase: RunPhase
    sha256: str

    @property
    def pending(self) -> bool:
        return self.status == "pending"

    @property
    def current_schema(self) -> bool:
        return self.schema_version == CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointFile:
    """One checkpoint read exactly once: its decoded payload and its digest."""

    payload: Any
    sha256: str


def stamp_from_payload(payload: Any, *, sha256: str) -> CheckpointStamp:
    """The structural stamp of a decoded checkpoint, or a refusal."""

    if not isinstance(payload, Mapping):
        raise CheckpointFormatError("checkpoint is not an object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CheckpointFormatError("checkpoint schema_version is not an integer")
    status = payload.get("status")
    if status not in CHECKPOINT_STATUSES:
        raise CheckpointFormatError("checkpoint status is invalid")
    if "phase" not in payload:
        raise CheckpointFormatError("checkpoint phase is missing")
    try:
        phase = RunPhase(payload["phase"])
    except (TypeError, ValueError) as exc:
        raise CheckpointFormatError("checkpoint phase is unknown") from exc
    return CheckpointStamp(schema_version, status, phase, sha256)


def read_checkpoint_file(run_dir: str | Path) -> CheckpointFile | None:
    """The decoded checkpoint of one run, or ``None`` when none was written.

    A file that exists but cannot be read or parsed raises
    :class:`CheckpointFormatError`: a corrupt checkpoint is an incident, never
    a missing one.
    """

    try:
        raw = (Path(run_dir) / RUN_CHECKPOINT_NAME).read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointFormatError("checkpoint is unreadable") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError("checkpoint is unreadable") from exc
    return CheckpointFile(payload, hashlib.sha256(raw).hexdigest())


def read_checkpoint_stamp(run_dir: str | Path) -> CheckpointStamp | None:
    """The structural stamp of one run's checkpoint, or ``None`` without one."""

    checkpoint = read_checkpoint_file(run_dir)
    if checkpoint is None:
        return None
    return stamp_from_payload(checkpoint.payload, sha256=checkpoint.sha256)


def checkpoint_sha256(run_dir: str | Path) -> str | None:
    """The exact bytes of the checkpoint one observation was built from."""

    try:
        return hashlib.sha256(
            (Path(run_dir) / RUN_CHECKPOINT_NAME).read_bytes()
        ).hexdigest()
    except OSError:
        return None


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "CHECKPOINT_STATUSES", "CheckpointFile",
    "CheckpointFormatError", "CheckpointStamp", "checkpoint_sha256",
    "read_checkpoint_file", "read_checkpoint_stamp", "stamp_from_payload",
]
