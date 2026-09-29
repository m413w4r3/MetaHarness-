"""The run runtime: frozen facts, live dependencies and composition.

``RunRuntime`` owns everything one prepared run resolves once: the effective
config and run options, the secrets used for redaction, the trace stream, the
recovery services, the execution selection and the durable checkpoints.  It
composes the four run authorities -- bootstrap, composition, failure and
observability -- and keeps the run-level entries that span them: the recovery
coordinators, the cycle-record merge, the checkpoint boundary and the operator
plan recovery.

``metaharness.orchestrator`` is the façade that constructs this runtime and
hands it to the coordinator; no module of this package imports the façade.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..diagnostics import write_run_diagnostics
from ..integrations.github import GitHubWorkstreamClient, NullGitHubWorkstreamClient
from ..llm.chat import OpenAIChatTextClient
from ..models import (
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    RunCycle,
    RunStatus,
    profile_driver_name,
)
from ..profiles import profile_execution_fingerprint
from ..recovery_policy import elapsed_hours, wall_clock_exhausted
from ..redaction import redact_file
from ..result import RunResult, atomic_write_text
from ..resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumePhase,
    read_checkpoint_record,
    write_checkpoint,
)
from ..run_options import RunOptions
from ..state import RunStateStore
from ..trace import TraceSink, TraceStream
from ..usage import normalize_usage, phase_usage_summary
from ..validation import ValidationError
from .audit import AuditService
from .gates import GateService
from .pipeline_v2 import PipelineV2Context, PipelineV2Coordinator
from .publication import PublicationService
from .recovery import (
    CheckInfrastructureRecovery, RecoveryCoordinator, WorkerRecovery,
    normalize_exit_reason,
)
from .run_bootstrap import RunBootstrap
from .run_composition import RunComposition
from .run_failure import RunFailure
from .shared import AGENT_ARTIFACTS, OrchestrationError, chat_client, json_text
from .step_execution import StepAcceptanceService, StepExecutionService
from .worker_attempt import WorkerAttemptService


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def safe_run_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationError("run_id must be a non-empty path component")
    value = value.strip()
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise OrchestrationError("run_id must be one safe path component")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise OrchestrationError("run_id contains unsupported characters")
    # The run id is also the last component of the run branch name.
    if ".." in value or value.endswith(".") or value.endswith(".lock"):
        raise OrchestrationError("run_id must be a valid Git ref component")
    return value


class RunRuntime:
    """The frozen facts, live dependencies and services of one prepared run."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        planner_client: Any | None = None,
        github_client: GitHubWorkstreamClient | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self.planner_client = planner_client
        self.github_client = (
            github_client if github_client is not None else NullGitHubWorkstreamClient()
        )
        self.trace_sink = trace_sink
        self.environment = (
            config.runtime_environment if config.runtime_environment else os.environ
        )
        self.run_options: RunOptions | None = None
        self.secrets: tuple[str, ...] = ()
        self.last_selection: Any | None = None
        self.trace: TraceStream | None = None
        self.trace_cycle = 1
        # The step services of this run: the step execution ladder, the one
        # worker attempt it drives, and the durable acceptance boundary.
        self.step_execution = StepExecutionService(self)
        self.worker_attempt = WorkerAttemptService(self)
        self.step_acceptance = StepAcceptanceService(self)
        self.gates = GateService(self)
        self.audit = AuditService(self)
        # The candidate authority: the immutable candidate commit, its push
        # and the publication of the accepted HEAD.
        self.publication = PublicationService(self)

        # The four run authorities; each one owns its own module and reads the
        # runtime's frozen facts and live dependencies through this reference.
        self.observability = RunObservability(self)
        self.failure = RunFailure(self)
        self.composition = RunComposition(self)
        self.bootstrap = RunBootstrap(self)

    def chat(self, endpoint: LLMEndpointConfig) -> OpenAIChatTextClient:
        """The run's one text transport, on the run's one transport horizon.

        ``chat`` is the only place the ``[transport]`` budget is applied, so
        every consumer of a text endpoint (the planner and the audit) shares
        the same resilience.
        """

        return chat_client(
            dataclasses.replace(
                endpoint, max_wait_seconds=self.config.transport.max_wait_seconds
            ),
            self.environment, self.observability.trace_transport,
        )

    def run_pipeline(
        self, store: RunStateStore, pipeline: PipelineV2Context,
        start: ResumeCheckpoint, *, resumed: bool,
    ) -> RunResult:
        """Drive the generic coordinator and project the failure that left it."""

        engine = PipelineV2Coordinator(
            pipeline, self.composition.pipeline_operations(store)
        )
        return self.failure.run_pipeline(store, pipeline, start, engine, resumed=resumed)

    def recovery(self, store: RunStateStore) -> RecoveryCoordinator:
        """The recovery coordinator bound to this run's durable state."""

        return RecoveryCoordinator(store, emit=self.observability.trace_emit)

    def budget_exhausted(self, store: RunStateStore) -> str | None:
        """The PARTIAL reason the one global budget already proves, or ``None``.

        The wall-clock budget is measured from the run's durable creation
        timestamp, never from an in-memory timer: a crash and a resume both
        keep spending the same budget.
        """

        budget = self.run_options.budget if self.run_options is not None else None
        if budget is None:
            return None
        if wall_clock_exhausted(
            store.load().get("started_at"),
            max_wall_clock_hours=budget.max_wall_clock_hours,
        ):
            return "wall_clock"
        return None

    def check_recovery(self, store: RunStateStore) -> CheckInfrastructureRecovery:
        return CheckInfrastructureRecovery(
            self.recovery(store), store=store,
            attempts=self.run_options.budget.step_attempts,
            budget_exhausted=lambda: self.budget_exhausted(store),
        )

    def worker_recovery(self, store: RunStateStore) -> WorkerRecovery:
        return WorkerRecovery(
            self.recovery(store), store=store,
            secrets=self.secrets, scope=self.config.scope,
        )

    @staticmethod
    def cycle_update(store: RunStateStore, cycle: RunCycle | int, **fields: Any) -> None:
        """Merge one cycle record without replacing the other cycle records."""

        number = cycle.number if isinstance(cycle, RunCycle) else cycle
        state = store.load()
        cycles = list(state.get("cycles") or [])
        index = next(
            (i for i, item in enumerate(cycles)
             if isinstance(item, dict) and item.get("number") == number),
            None,
        )
        record: dict[str, Any] = {"number": number}
        if index is not None and isinstance(cycles[index], dict):
            record.update(cycles[index])
        if isinstance(cycle, RunCycle):
            record["kind"] = cycle.kind.value
        record.update(fields)
        if index is None:
            cycles.append(record)
        else:
            cycles[index] = record
        store.update_metadata(cycles=cycles)

    @staticmethod
    def approved_check_authority_sha256(run_dir: Path) -> str | None:
        """The check authority hash the run's durable boundary already binds.

        This is deliberately *not* a hash of the file being read: the expected
        value comes from the checkpoint written before the approval, so a
        rewritten ``check_authority.json`` is rejected on every live use, not
        only on resume.  A run created before the artifact existed has no hash
        and keeps its current behavior.
        """

        directory = Path(run_dir).expanduser().resolve()
        while not (directory / "check_authority.json").is_file():
            parent = directory.parent
            if parent == directory:
                return None
            directory = parent
        try:
            read_checkpoint_record(directory)
        except ResumeCheckpointError as exc:
            raise ValidationError(f"the run checkpoint is unreadable: {exc}") from exc
        state = RunStateStore(directory / "state.json").load()
        identity = state.get("plan_identity") if isinstance(state.get("plan_identity"), Mapping) else {}
        approved = identity.get("checks_sha256")
        if approved is None:
            raise ValidationError(
                "the run has a check authority but no durable approved hash"
            )
        return approved

    @staticmethod
    def write_checkpoint(
        run_dir: Path,
        phase: ResumePhase,
        *,
        head: str | None,
        step_index: int | None = None,
        iteration: int | None = None,
        plan_sha256: str | None = None,
    ) -> None:
        """Persist the next operation that has not succeeded yet.

        Git records the accepted commit; the checkpoint names only the next
        operation and the canonical effective plan bytes.
        """

        record = read_checkpoint_record(run_dir)
        if record is None:
            return
        previous = record
        write_checkpoint(run_dir, ResumeCheckpoint(
            phase=phase,
            iteration=iteration or previous.iteration,
            step_index=step_index,
            last_green_commit=head or previous.last_green_commit,
            plan_sha256=plan_sha256 or previous.plan_sha256,
        ))


if TYPE_CHECKING:
    from .runtime import RunRuntime

_OUTPUT_DISCIPLINE_TARGETS = {
    ExecutionRole.PLANNER: "META PLAN v2 only",
    ExecutionRole.IMPLEMENTER: "<=8 lines; <=1200 characters",
    ExecutionRole.AUDITOR: "<=8 lines; <=1200 characters",
}


def _safe_agent_result_payload(result: Any) -> dict[str, Any]:
    """Persist bounded protocol metadata, never the raw backend result."""

    exit_reason = getattr(result, "exit_reason", None)
    if isinstance(exit_reason, str) and exit_reason.strip():
        exit_reason = normalize_exit_reason(exit_reason)
    backend_reason = getattr(result, "backend_reason", None)
    if isinstance(backend_reason, str) and backend_reason.strip():
        backend_reason = normalize_exit_reason(backend_reason)
    return {
        "status": getattr(result, "status", None),
        "exit_reason": exit_reason,
        "exit_code": getattr(result, "exit_code", None),
        "timed_out": bool(getattr(result, "timed_out", False)),
        "usage": normalize_usage(getattr(result, "usage", None)),
        "driver": getattr(result, "driver", None),
        "backend_reason": backend_reason,
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
            ExecutionRole.AUDITOR: "audit",
        }.get(role)
        selected = getattr(selection, name, None) if name is not None else None
        if getattr(selected, "profile_id", None) == profile_id:
            return selected
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
        observed_exit_reason = (
            exit_reason if exit_reason is not None
            else getattr(result, "exit_reason", None)
        )
        if isinstance(observed_exit_reason, str) and observed_exit_reason.strip():
            observed_exit_reason = normalize_exit_reason(observed_exit_reason)

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
            "exit_reason": observed_exit_reason,
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
        """Publish the per-cycle token totals and the compact budget view.

        Always derived from persisted artifacts
        (``cycles/NNN/implementation/steps/Sxx/step.json`` and every worker
        report), never from ``state.steps``.  The budget block is descriptive:
        the durable counters and the creation timestamp stay the authority.
        """

        store.update_metadata(
            usage=phase_usage_summary(run_dir), budget=self.budget_view(store),
        )

    def budget_view(self, store: RunStateStore) -> dict[str, Any]:
        """What the one global budget is, what it consumed, and its known cost."""

        budget = self.runtime.run_options.budget if self.runtime.run_options else None
        state = store.load()
        elapsed = elapsed_hours(state.get("started_at"))
        iterations = state.get("current_iteration") or state.get("iteration")
        return {
            "configured": dataclasses.asdict(budget) if budget is not None else None,
            "iterations": iterations if isinstance(iterations, int) else 1,
            "elapsed_seconds": round(elapsed * 3600) if elapsed is not None else None,
            # No configured provider publishes an explicit cost, and a price
            # is never derived from token counts.
            "cost_usd": None,
        }

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


    def diagnose_result(self, result: RunResult) -> RunResult:
        """Best-effort terminal projection; diagnostics never changes a run result."""

        if result.status in {RunStatus.COMMITTED, RunStatus.PUBLISHED, RunStatus.PARTIAL}:
            state = result.state
            self.trace_emit(
                "run.completed",
                phase="run",
                cycle=state.get("cycle") if isinstance(state.get("cycle"), int) else None,
                data={
                    "status": result.status.value,
                    "commit_sha": state.get("commit_sha"),
                    "published": result.status is RunStatus.PUBLISHED,
                    "completion_kind": state.get("completion_kind"),
                },
                once=True,
            )
        elif result.status is RunStatus.FAILED:
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
        elif result.status is RunStatus.WAITING_HUMAN:
            failure = result.state.get("failure") if isinstance(result.state, Mapping) else None
            self.trace_emit(
                "run.waiting_human",
                phase="run",
                cycle=result.state.get("cycle") if isinstance(result.state, Mapping) else None,
                data={
                    "reason": failure.get("reason") if isinstance(failure, Mapping) else None,
                },
                once=True,
            )

        if result.status in {
            RunStatus.FAILED,
            RunStatus.WAITING_HUMAN,
            RunStatus.WAITING_EXTERNAL,
            RunStatus.COMMITTED, RunStatus.PUBLISHED, RunStatus.PARTIAL,
        }:
            try:
                write_run_diagnostics(self.runtime.config, result.run_dir)
            except Exception:
                # ``write_run_diagnostics`` records diagnostics.error.txt when
                # possible.  A reporting failure must not alter the pipeline.
                pass
        return result

    def redact_step_artifacts(self, step_dir: Path) -> None:
        for name in AGENT_ARTIFACTS:
            redact_file(step_dir / name, self.runtime.secrets)
