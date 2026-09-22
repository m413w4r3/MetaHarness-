"""Durable, provider-neutral META TRACE v1 event stream.

The trace is an observation channel.  It is deliberately not consulted by
the orchestration state machine for any gate, budget, Git, or publication
decision.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .redaction import REDACTED, redact


TRACE_SCHEMA_VERSION = 1
TRACE_RELATIVE_PATH = "trace/events.v1.jsonl"
TRACE_LOCK_NAME = "events.v1.lock"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any, *, secrets: tuple[str, ...]) -> Any:
    """Return a JSON-safe, secret-free observation value.

    Trace payloads are authored by MetaHarness, but the final boundary also
    protects against accidentally passing a credential-shaped field from a
    backend or a test double.  ``api_key_env`` is explicitly safe: it is a
    variable name, never the value stored in that variable.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in {
                "api_key", "api_key_value", "authorization", "password",
                "passwd", "secret", "secret_value", "credential",
                "credentials", "access_token", "refresh_token", "bearer",
                "token", "token_value", "api_token",
            } or normalized.endswith((
                "_api_key", "_authorization", "_secret", "_credential", "_token",
            )):
                result[key] = REDACTED
                continue
            result[key] = _json_value(raw_value, secrets=secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, secrets=secrets) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return redact(value, secrets) if isinstance(value, str) else value
    # Do not serialize arbitrary backend objects or their repr(), which can
    # contain request headers, URLs, or credentials.
    return REDACTED


@runtime_checkable
class TraceSink(Protocol):
    """Destination for one already-numbered trace event."""

    def emit(self, event: "TraceEvent") -> None:
        ...


@dataclass(frozen=True)
class TraceEvent:
    """One self-contained META TRACE v1 record."""

    sequence: int
    run_id: str
    pipeline_version: int
    event: str
    phase: str | None = None
    cycle: int | None = None
    step_id: str | None = None
    data: Mapping[str, Any] | None = None
    timestamp: str | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("trace sequence must be a positive integer")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("trace run_id must be a non-empty string")
        if isinstance(self.pipeline_version, bool) or not isinstance(self.pipeline_version, int):
            raise ValueError("trace pipeline_version must be an integer")
        if self.pipeline_version != 2:
            raise ValueError("trace pipeline_version must be 2")
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema version")
        if not isinstance(self.event, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]+", self.event):
            raise ValueError("trace event name is invalid")
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValueError("trace timestamp must be a string or null")
        if self.data is not None and not isinstance(self.data, Mapping):
            raise ValueError("trace data must be an object or null")

    def as_dict(self, *, secrets: tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp": self.timestamp or _now(),
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "event": self.event,
            "phase": self.phase,
            "cycle": self.cycle,
            "step_id": self.step_id,
            "data": _json_value(dict(self.data or {}), secrets=secrets),
        }

    def to_json(self, *, secrets: tuple[str, ...] = ()) -> str:
        return json.dumps(
            self.as_dict(secrets=secrets),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class NullTraceSink:
    """A no-op sink useful for isolated callers and tests."""

    def emit(self, event: TraceEvent) -> None:
        del event


@contextmanager
def _exclusive_trace_lock(lock_path: Path):
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


def _event_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.name == "events.v1.jsonl":
        return path
    # Passing a run directory is the ergonomic public form.
    return path / TRACE_RELATIVE_PATH


class JsonlTraceSink:
    """Append-only, fsynced JSONL trace sink.

    The sink validates the sequence under the same lock used for the append,
    so a resumed process starts after the last durable complete JSON record.
    """

    def __init__(
        self,
        path_or_run_dir: str | Path | None = None,
        *,
        run_dir: str | Path | None = None,
        secrets: Sequence[str] = (),
    ) -> None:
        if path_or_run_dir is None:
            if run_dir is None:
                raise TypeError("trace sink needs a run directory or events path")
            path_or_run_dir = run_dir
        elif run_dir is not None:
            raise TypeError("pass either path_or_run_dir or run_dir, not both")
        self.path = _event_path(path_or_run_dir)
        self.lock_path = self.path.parent / TRACE_LOCK_NAME
        self.secrets = tuple(secret for secret in secrets if isinstance(secret, str) and secret)

    @classmethod
    def for_run(cls, run_dir: str | Path, *, secrets: Sequence[str] = ()) -> "JsonlTraceSink":
        return cls(Path(run_dir) / TRACE_RELATIVE_PATH, secrets=secrets)

    def last_sequence(self) -> int:
        with _exclusive_trace_lock(self.lock_path):
            return self._last_sequence_unlocked()

    def _last_sequence_unlocked(self) -> int:
        if not self.path.is_file():
            return 0
        last = 0
        with self.path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                if not raw.endswith(b"\n"):
                    # A partial tail must not be treated as a valid prior
                    # stream: doing so could reuse a sequence after a partial
                    # write.
                    raise OSError("trace contains an incomplete JSONL record")
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    raise OSError("trace contains invalid JSONL") from exc
                sequence = payload.get("sequence") if isinstance(payload, dict) else None
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                    raise OSError("trace contains an invalid sequence")
                if sequence <= last:
                    raise OSError("trace sequence is not strictly increasing")
                last = sequence
        return last

    def emit(self, event: TraceEvent) -> None:
        if not isinstance(event, TraceEvent):
            raise TypeError("trace sink accepts TraceEvent")
        line = (event.to_json(secrets=self.secrets) + "\n").encode("utf-8")
        with _exclusive_trace_lock(self.lock_path):
            last = self._last_sequence_unlocked()
            if event.sequence <= last:
                raise ValueError(
                    f"trace sequence must increase: {event.sequence} <= {last}"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("ab") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                raise


class FanoutTraceSink:
    """Local required sink plus optional observer sinks.

    Observer failures are intentionally isolated.  The local JSONL sink is
    emitted first and its failure is propagated because it is the durable run
    artifact; optional sinks never participate in run success or failure.
    """

    def __init__(self, required: TraceSink, observers: Sequence[TraceSink] = ()) -> None:
        self.required = required
        self.observers = tuple(observers)

    def emit(self, event: TraceEvent) -> None:
        self.required.emit(event)
        for observer in self.observers:
            try:
                observer.emit(event)
            except Exception:
                # An external observation endpoint cannot become an execution
                # authority or alter the durable state machine.
                continue


class TraceStream:
    """Number and emit events for one run, continuing after resume."""

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        pipeline_version: int,
        *,
        sink: TraceSink | None = None,
        secrets: Sequence[str] = (),
    ) -> None:
        local = JsonlTraceSink.for_run(run_dir, secrets=secrets)
        self.sink: TraceSink = FanoutTraceSink(local, (sink,)) if sink is not None else local
        self.run_id = run_id
        self.pipeline_version = pipeline_version
        self.secrets = tuple(secret for secret in secrets if isinstance(secret, str) and secret)
        self._sequence = local.last_sequence()

    @property
    def path(self) -> Path:
        required = self.sink.required if isinstance(self.sink, FanoutTraceSink) else self.sink
        return required.path if isinstance(required, JsonlTraceSink) else Path(TRACE_RELATIVE_PATH)

    def emit(
        self,
        event: str,
        *,
        phase: str | None = None,
        cycle: int | None = None,
        step_id: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        self._sequence += 1
        record = TraceEvent(
            sequence=self._sequence,
            timestamp=_now(),
            run_id=self.run_id,
            pipeline_version=self.pipeline_version,
            event=event,
            phase=phase,
            cycle=cycle,
            step_id=step_id,
            data=_json_value(dict(data or {}), secrets=self.secrets),
        )
        try:
            self.sink.emit(record)
        except Exception:
            # The number belongs to the durable local stream only when emit
            # succeeds.  A failed required emit must be visible to the caller;
            # the next successful attempt may reuse this sequence.
            self._sequence -= 1
            raise
        return record

    def _read_events(self) -> list[dict[str, Any]]:
        path = self.path
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def has_event(
        self,
        event: str,
        *,
        cycle: int | None = None,
        step_id: str | None = None,
    ) -> bool:
        return any(
            row.get("event") == event
            and (cycle is None or row.get("cycle") == cycle)
            and (step_id is None or row.get("step_id") == step_id)
            for row in self._read_events()
        )

    def emit_once(self, event: str, **kwargs: Any) -> TraceEvent | None:
        if self.has_event(
            event,
            cycle=kwargs.get("cycle"),
            step_id=kwargs.get("step_id"),
        ):
            return None
        return self.emit(event, **kwargs)


__all__ = [
    "FanoutTraceSink",
    "JsonlTraceSink",
    "NullTraceSink",
    "TRACE_RELATIVE_PATH",
    "TRACE_SCHEMA_VERSION",
    "TraceEvent",
    "TraceSink",
    "TraceStream",
]
