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

import dataclasses, os, re, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from ..integrations.github import GitHubWorkstreamClient, NullGitHubWorkstreamClient
from ..llm.chat import OpenAIChatTextClient
from ..models import (
    HarnessConfig, LLMEndpointConfig, RunCycle,
    RunDisposition, RunMachineState, RunPhase,
)
from ..result import RunResult
from ..resume import (
    ResumeCheckpoint, ResumeCheckpointError, ResumePhase,
    read_checkpoint_record, write_checkpoint,
)
from ..run_options import (
    RunOptions, RunOptionsError,
    effective_run_config,
    read_run_options_for_state,
)
from ..state import RunStateStore
from ..trace import TraceSink, TraceStream
from ..validation import ValidationError
from .audit import AuditService
from .check_recovery import CheckInfrastructureRecovery
from .gates import GateService
from .pipeline_v2 import PipelineV2Coordinator, PipelineV2Context
from .publication import PublicationService
from .recovery import RecoveryCoordinator
from .run_resume import ResumedRun
from .run_bootstrap import RunBootstrap
from .run_composition import RunComposition
from .run_failure import RunFailure
from .run_observability import RunObservability
from .shared import (
    OrchestrationError, chat_client,
)
from .step_acceptance import StepAcceptanceService
from .step_execution import StepExecutionService
from .worker_attempt import WorkerAttemptService
from .worker_recovery import WorkerRecovery

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

    def check_recovery(self, store: RunStateStore) -> CheckInfrastructureRecovery:
        return CheckInfrastructureRecovery(
            self.recovery(store), store=store, budgets=self.run_options.recovery,
        )

    def worker_recovery(self, store: RunStateStore) -> WorkerRecovery:
        return WorkerRecovery(
            self.recovery(store), store=store,
            budgets=self.run_options.recovery, secrets=self.secrets,
            scope=self.config.scope,
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
            record = read_checkpoint_record(directory)
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
