"""The observation stream, session metadata and diagnostics of one run.

``RunObservability`` owns what a run records about itself: the trace stream and
its bounded events, the session metadata of every model call, the usage
projections read back from the durable artifacts, the normalization and
redaction of the small per-attempt artifact envelopes and the terminal
diagnostics of a run result.  It never decides an outcome: ``diagnose_result``
only projects a result the other authorities already produced.
"""

from __future__ import annotations

import hashlib, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from ..diagnostics import write_run_diagnostics
from ..models import ExecutionRole, ModelProfile, RunStatus, profile_driver_name
from ..profiles import profile_execution_fingerprint
from ..redaction import redact_file
from ..result import RunResult, atomic_write_text
from ..state import RunStateStore
from ..trace import TraceStream
from ..usage import normalize_usage, phase_usage_summary
from .shared import AGENT_ARTIFACTS, REVISION_ARTIFACTS, json_text

if TYPE_CHECKING:
    from .runtime import RunRuntime

_OUTPUT_DISCIPLINE_TARGETS = {
    ExecutionRole.PLANNER: "META PLAN v2 only",
    ExecutionRole.IMPLEMENTER: "<=8 lines; <=1200 characters",
    ExecutionRole.REPAIR: "<=6 lines; <=800 characters",
    ExecutionRole.REVISER: "<=10 lines; <=1500 characters",
    ExecutionRole.REVIEWER: "META REVIEW v1; terse material findings only",
}


def _safe_agent_result_payload(result: Any) -> dict[str, Any]:
    """Persist bounded protocol metadata, never the raw backend result."""

    return {
        "status": getattr(result, "status", None),
        "exit_reason": getattr(result, "exit_reason", None),
        "exit_code": getattr(result, "exit_code", None),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "usage": normalize_usage(getattr(result, "usage", None)),
        "driver": getattr(result, "driver", None),
        "backend_reason": getattr(result, "backend_reason", None),
    }


class RunObservability:
    """The observation stream, session metadata and diagnostics of one run."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def begin_trace(
        self, run_dir: Path, run_id: str, *, created: bool, pipeline_version: int = 2,
    ) -> None:
        """Attach the observation stream without changing run authority."""

        self.runtime.trace = TraceStream(
            run_dir,
            run_id,
            pipeline_version=pipeline_version,
            sink=self.runtime.trace_sink,
            secrets=self.runtime.secrets,
        )
        if created:
            self.runtime.trace.emit(
                "run.created",
                phase="run",
                cycle=1,
                data={"status": RunStatus.CREATED.value},
            )

    def trace_emit(
        self,
        event: str,
        *,
        phase: str | None = None,
        cycle: int | None = None,
        step_id: str | None = None,
        data: Mapping[str, Any] | None = None,
        once: bool = False,
    ) -> None:
        stream = self.runtime.trace
        if stream is None:
            return
        kwargs = {
            "phase": phase,
            "cycle": cycle,
            "step_id": step_id,
            "data": data or {},
        }
        if once:
            # Terminal events describe the run, not a cycle.  A resumed
            # terminal projection must remain idempotent even when the
            # durable state exposes a different cycle number.
            if event in {"run.created", "run.failed", "run.completed"}:
                if stream.has_event(event):
                    return
                stream.emit(event, **kwargs)
            else:
                stream.emit_once(event, **kwargs)
        else:
            stream.emit(event, **kwargs)

    def trace_transport(self, observation: dict[str, Any]) -> None:
        """Persist only bounded transport metadata, never request material."""
        event = observation.get("event")
        if not isinstance(event, str):
            return
        data = {
            key: observation[key]
            for key in (
                "operation", "attempt", "attempts", "http_status", "elapsed_ms",
                "max_wait_seconds",
            )
            if isinstance(observation.get(key), (str, int))
        }
        self.trace_emit(f"transport.{event}", phase="transport", data=data)

    @staticmethod
    def trace_time() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def trace_selected_profile(
        self, profile_id: str, role: ExecutionRole, *, step_id: str | None = None,
    ) -> Any | None:
        selection = self.runtime.last_selection
        if selection is None:
            return None
        if step_id is not None:
            for item in getattr(selection, "steps", ()):
                if getattr(item, "step_id", None) == step_id:
                    selected = getattr(item, "implementer", None)
                    if getattr(selected, "profile_id", None) == profile_id:
                        return selected
                    for fallback in getattr(item, "fallbacks", ()):
                        if getattr(fallback, "profile_id", None) == profile_id:
                            return fallback
        name = {
            ExecutionRole.PLANNER: "planner",
            ExecutionRole.REPAIR: "check_repair",
            ExecutionRole.REVIEWER: "final_reviewer",
            ExecutionRole.REVISER: "semantic_reviser",
        }.get(role)
        selected = getattr(selection, name, None) if name is not None else None
        if getattr(selected, "profile_id", None) == profile_id:
            return selected
        fallback_name = {
            ExecutionRole.REPAIR: "check_repair_fallbacks",
            ExecutionRole.REVISER: "semantic_reviser_fallbacks",
        }.get(role)
        for fallback in getattr(selection, fallback_name, ()) if fallback_name else ():
            if getattr(fallback, "profile_id", None) == profile_id:
                return fallback
        return None

    def trace_session(
        self,
        *,
        profile: ModelProfile | None,
        selected: Any | None,
        role: ExecutionRole,
        prompt_bytes: int | None,
        started_at: str,
        started_mono: float,
        tree_before: str | None = None,
        result: Any | None = None,
        exit_reason: str | None = None,
        final_message: str | None = None,
    ) -> dict[str, Any]:
        """Build session metadata without deriving unavailable metrics."""

        fingerprint = getattr(selected, "config_sha256", None)
        if fingerprint is None and profile is not None:
            try:
                fingerprint = profile_execution_fingerprint(
                    profile,
                    agent_env_allowlist=self.runtime.config.codex_runtime.env_allowlist,
                    codex_home=self.runtime.config.codex_runtime.home,
                    claude_config_home=self.runtime.config.claude_runtime.home,
                )
            except (TypeError, ValueError, AttributeError):
                fingerprint = None
        raw_result = getattr(result, "raw_result", None) if result is not None else None
        if final_message is None and result is not None:
            candidate_message = getattr(result, "final_message", None)
            if isinstance(candidate_message, str):
                final_message = candidate_message
        raw_usage = getattr(raw_result, "usage", None) if raw_result is not None else None
        usage = raw_usage if isinstance(raw_usage, Mapping) else (
            getattr(result, "usage", None)
            if result is not None and raw_result is None else None
        )

        aliases = {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
            "cache_write_input_tokens": (
                "cache_write_input_tokens", "cache_creation_input_tokens",
            ),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "reasoning_output_tokens": ("reasoning_output_tokens",),
        }
        nested = {
            "cached_input_tokens": (
                ("prompt_tokens_details", "cached_tokens"),
                ("input_tokens_details", "cached_tokens"),
            ),
            "cache_write_input_tokens": (
                ("prompt_tokens_details", "cache_write_tokens"),
                ("input_tokens_details", "cache_write_tokens"),
            ),
            "reasoning_output_tokens": (
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_tokens_details", "reasoning_tokens"),
            ),
        }

        def metric(name: str) -> int | None:
            if not isinstance(usage, Mapping):
                return None
            for alias in aliases.get(name, (name,)):
                value = usage.get(alias)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
            for container, key in nested.get(name, ()):
                details = usage.get(container)
                if isinstance(details, Mapping):
                    value = details.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        return value
            return None

        return {
            "driver": (
                profile_driver_name(profile.driver) if profile is not None
                else getattr(selected, "driver", None)
                or getattr(result, "driver", None)
            ),
            "driver_version": (
                getattr(result, "driver_version", None) if result is not None else None
            ) or getattr(selected, "driver_version", None)
            or (profile.driver_version if profile is not None else None),
            "provider": (
                profile.provider if profile is not None
                else getattr(selected, "provider", None)
            ),
            "model": profile.model if profile is not None else getattr(selected, "model", None),
            "effort": profile.effort if profile is not None else getattr(selected, "effort", None),
            "profile_id": getattr(selected, "profile_id", None) or (profile.id if profile is not None else None),
            "profile_fingerprint": fingerprint,
            "role": role.value,
            "started_at": started_at,
            "finished_at": self.trace_time() if result is not None or exit_reason is not None else None,
            "wall_time_ms": round((time.perf_counter() - started_mono) * 1000) if result is not None or exit_reason is not None else None,
            "prompt_bytes": prompt_bytes,
            "final_message_bytes": (
                len(final_message.encode("utf-8", errors="replace"))
                if isinstance(final_message, str) else None
            ),
            "output_discipline_target": _OUTPUT_DISCIPLINE_TARGETS.get(role),
            "input_tokens": metric("input_tokens"),
            "cached_input_tokens": metric("cached_input_tokens"),
            "cache_write_input_tokens": metric("cache_write_input_tokens"),
            "output_tokens": metric("output_tokens"),
            "reasoning_output_tokens": metric("reasoning_output_tokens"),
            "tool_call_count": None,
            "exit_reason": exit_reason if exit_reason is not None else getattr(result, "exit_reason", None),
            "tree_before": tree_before or getattr(result, "tree_before", None),
            "tree_after": getattr(result, "tree_after", None) if result is not None else None,
            "external_session_id": getattr(result, "external_session_id", None) if result is not None else None,
        }

    def trace_finished_model_session(
        self,
        *,
        profile: ModelProfile,
        selected: Any | None,
        role: ExecutionRole,
        prompt_bytes: int | None,
        started_at: str,
        started_mono: float,
        usage: Mapping[str, Any] | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        exit_reason: str | None = None,
        final_message: str | None = None,
    ) -> dict[str, Any]:
        session = self.trace_session(
            profile=profile,
            selected=selected,
            role=role,
            prompt_bytes=prompt_bytes,
            started_at=started_at,
            started_mono=started_mono,
            tree_before=tree_before,
            result=None,
            exit_reason=exit_reason,
            final_message=final_message,
        )
        session.update(
            finished_at=self.trace_time(),
            wall_time_ms=round((time.perf_counter() - started_mono) * 1000),
            tree_after=tree_after,
        )
        aliases = {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "cached_input_tokens": ("cached_input_tokens", "cache_read_input_tokens"),
            "cache_write_input_tokens": (
                "cache_write_input_tokens", "cache_creation_input_tokens",
            ),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "reasoning_output_tokens": ("reasoning_output_tokens",),
        }
        nested = {
            "cached_input_tokens": (
                ("prompt_tokens_details", "cached_tokens"),
                ("input_tokens_details", "cached_tokens"),
            ),
            "cache_write_input_tokens": (
                ("prompt_tokens_details", "cache_write_tokens"),
                ("input_tokens_details", "cache_write_tokens"),
            ),
            "reasoning_output_tokens": (
                ("completion_tokens_details", "reasoning_tokens"),
                ("output_tokens_details", "reasoning_tokens"),
            ),
        }
        for name in aliases:
            value = None
            if isinstance(usage, Mapping):
                for alias in aliases[name]:
                    value = usage.get(alias)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        break
                    value = None
                if value is None:
                    for container, key in nested.get(name, ()):
                        details = usage.get(container)
                        if isinstance(details, Mapping):
                            value = details.get(key)
                            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                                break
                            value = None
            session[name] = value
        return session

    @staticmethod
    def trace_diff_reference(path: Path) -> dict[str, Any]:
        try:
            payload = path.read_bytes()
        except OSError:
            return {"diff_artifact": str(path), "diff_sha256": None}
        return {
            "diff_artifact": str(path),
            "diff_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def update_v2_usage(self, store: RunStateStore, run_dir: Path) -> None:
        """Publish the per-cycle token totals.

        Always derived from persisted artifacts
        (``cycles/NNN/implementation/steps/Sxx/step.json`` and every worker
        report), never from ``state.steps``.
        """

        store.update_metadata(usage=phase_usage_summary(run_dir))

    @staticmethod
    def ensure_step_artifacts(step_dir: Path, result: Any) -> None:
        """Make the durable step envelope complete for test doubles too."""

        if not (step_dir / "agent.final.md").exists():
            atomic_write_text(step_dir / "agent.final.md", str(getattr(result, "final_message", "")))
        if not (step_dir / "agent.stderr.log").exists():
            atomic_write_text(step_dir / "agent.stderr.log", str(getattr(result, "stderr_tail", "")))
        if not (step_dir / "agent.events.jsonl").exists():
            atomic_write_text(step_dir / "agent.events.jsonl", "")
        if not (step_dir / "agent.result.json").exists():
            payload = _safe_agent_result_payload(result)
            atomic_write_text(step_dir / "agent.result.json", json_text(payload))

    @staticmethod
    def ensure_revision_artifacts(artifact_dir: Path, result: Any) -> None:
        """Complete the revision artifact set without replacing provider files."""

        if not (artifact_dir / "agent.final.md").exists():
            atomic_write_text(artifact_dir / "agent.final.md", str(getattr(result, "final_message", "")))
        if not (artifact_dir / "agent.stderr.log").exists():
            atomic_write_text(artifact_dir / "agent.stderr.log", str(getattr(result, "stderr_tail", "")))
        if not (artifact_dir / "agent.events.jsonl").exists():
            atomic_write_text(artifact_dir / "agent.events.jsonl", "")
        if not (artifact_dir / "agent.result.json").exists():
            payload = _safe_agent_result_payload(result)
            atomic_write_text(artifact_dir / "agent.result.json", json_text(payload))

    def diagnose_result(self, result: RunResult) -> RunResult:
        """Best-effort terminal projection; diagnostics never changes a run result."""

        if result.status in {RunStatus.COMMITTED, RunStatus.PUBLISHED}:
            state = result.state
            self.trace_emit(
                "run.completed",
                phase="run",
                cycle=state.get("cycle") if isinstance(state.get("cycle"), int) else None,
                data={
                    "status": result.status.value,
                    "commit_sha": state.get("commit_sha"),
                    "published": result.status is RunStatus.PUBLISHED,
                },
                once=True,
            )
        elif result.status in {
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.PLAN_REJECTED,
        }:
            failure = result.state.get("failure") if isinstance(result.state, Mapping) else None
            self.trace_emit(
                "run.failed",
                phase="run",
                cycle=result.state.get("cycle") if isinstance(result.state.get("cycle"), int) else None,
                data={
                    "status": result.status.value,
                    "reason": failure.get("reason") if isinstance(failure, Mapping) else result.status.value,
                },
                once=True,
            )
        elif result.status in {RunStatus.WAITING_HUMAN, RunStatus.WAITING_CHECK_REPAIR}:
            failure = result.state.get("failure") if isinstance(result.state, Mapping) else None
            self.trace_emit(
                "run.waiting_check_repair" if result.status is RunStatus.WAITING_CHECK_REPAIR else "run.waiting_human",
                phase="run",
                cycle=result.state.get("cycle") if isinstance(result.state, Mapping) else None,
                data={
                    "reason": failure.get("reason") if isinstance(failure, Mapping) else None,
                },
                once=True,
            )

        if result.status in {
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.PLAN_REJECTED,
            RunStatus.WAITING_HUMAN,
            RunStatus.WAITING_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE,
            RunStatus.WAITING_CHECK_REPAIR,
            RunStatus.WAITING_REMOTE, RunStatus.WAITING_SCOPE_APPROVAL,
            RunStatus.WAITING_CONTRACT_REPAIR, RunStatus.COMMITTED, RunStatus.PUBLISHED,
        }:
            try:
                write_run_diagnostics(self.runtime.config, result.run_dir)
            except Exception:
                # ``write_run_diagnostics`` records diagnostics.error.txt when
                # possible.  A reporting failure must not alter the pipeline.
                pass
        return result

    def redact_revision_artifacts(self, artifact_dir: Path) -> None:
        for name in REVISION_ARTIFACTS:
            redact_file(artifact_dir / name, self.runtime.secrets)

    def redact_step_artifacts(self, step_dir: Path) -> None:
        for name in AGENT_ARTIFACTS:
            redact_file(step_dir / name, self.runtime.secrets)
