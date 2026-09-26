"""The run runtime: frozen facts, live dependencies and composition.

``RunRuntime`` owns everything one prepared run resolves once: the effective
config and run options, the secrets used for redaction, the trace stream, the
recovery services, the execution selection and the durable checkpoints.  It
builds the :class:`~metaharness.orchestration.pipeline_v2.PipelineV2Context`
of a run and assembles the :class:`PipelineV2Operations` the generic
coordinator sequences, wiring every operation to the service that owns it:

* :mod:`metaharness.orchestration.implementation` executes a step and accepts
  its candidate commit;
* :mod:`metaharness.orchestration.gates` runs the deterministic gate;
* :mod:`metaharness.orchestration.review_service` reviews the candidate and
  drives the correction cycles;
* :mod:`metaharness.orchestration.publication` publishes the accepted one.

``metaharness.orchestrator`` is the façade that constructs this runtime and
hands it to the coordinator; no module of this package imports the façade.
"""

from __future__ import annotations

import dataclasses, functools, hashlib, json, os, re, time, uuid
from dataclasses import asdict
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    NoReturn,
)
from ..agent.base import (
    AgentError,
    AgentRunRequest,
)
from ..agent.execution import (
    ExecutorRuntimeConfig,
    executor_for_profile,
)
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity_from_run,
    read_plan_approval,
    write_check_authority,
    wait_for_plan_approval,
)
from ..context import (
    build_context,
    render_context,
)
from ..validation import (
    ValidationError,
    config_with_check_authority,
)
from ..gitops import (
    GitError,
    WorktreeInfo,
    branch_exists,
    restore_paths_from_tree,
    candidate_tree_sha,
    create_run_worktree,
    current_head,
    git_root,
    index_tree_sha,
    local_branches,
    build_repository_reference,
    build_run_branch,
    registered_worktrees,
    resolve_commit,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
    status_porcelain,
    symbolic_head,
    stage_all,
)
from ..llm.chat import LLMError
from ..integrations.github import (
    GitHubIntegrationError,
    GitHubWorkstreamClient,
    NullGitHubWorkstreamClient,
)
from ..recommendation import (
    ExecutionRecommender,
    RecommendationError,
    write_recommendation_error,
)
from ..execution_selection import (
    SCHEMA_VERSION as EXECUTION_SELECTION_SCHEMA,
    ExecutionSelectionError,
    ensure_execution_selection,
    ensure_cycle_execution_selection,
    read_execution_selection_with_sha256,
    read_cycle_execution_selection,
    resolve_execution_selection,
    resolve_cycle_execution_selection,
    validate_cycle_execution_selection,
    validate_execution_selection,
)
from ..models import (
    BlockerKind,
    CycleKind,
    ExecutionRole,
    ExecutionSelection,
    GateStage,
    ModelProfile,
    HarnessConfig,
    PlanDecision,
    RunCycle,
    RunStatus,
    profile_driver_name,
)
from ..planning.artifacts import (
    persist_recovered_plan_artifacts,
    validate_implementation_bundle,
)
from ..planning.planner import PlannerV2
from ..planning.protocol import (
    PlanParseError,
    V2PlanParseError,
    parse_task_plan_v2,
)
from ..planning.validation import (
    validate_decomposition_policy,
    validate_execution_mode_policy,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
    validate_plan_repository_topology,
)
from ..plan_recovery import (
    PLAN_SOURCE_OPERATOR,
    PlanRecoveryError,
    plan_recovery_info,
    plan_source,
    recoverable_plan_source_status,
    validate_replacement_text,
    write_plan_recovery_record,
)
from ..resume import (
    PHASE_STATUS,
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeError,
    ResumeIntegrityError,
    ResumePhase,
    ResumeRequiresOperatorError,
    plan_identity_from_mapping,
    read_checkpoint,
    read_checkpoint_record,
    write_checkpoint,
)
from ..usage import (
    normalize_usage,
    empty_usage,
    phase_usage_summary,
    read_usage_artifact,
)
from ..redaction import (
    redact,
    redact_file,
    redact_mapping,
)
from ..diagnostics import write_run_diagnostics
from ..profiles import (
    ProfileError,
    build_llm_endpoint,
    profile_for_role,
    profile_execution_fingerprint,
    profiles_for_config,
)
from ..trace import (
    TraceSink,
    TraceStream,
)
from ..result import (
    RunResult,
    atomic_write_text,
)
from ..review import (
    Reviewer,
    ReviewParseError,
)
from ..recovery_policy import RecoveryDisposition
from ..state import RunStateStore
from ..workspace import (
    WorkspaceSetupError,
    prepare_workspace,
)
from ..run_options import (
    EffectiveRepairScopePolicy,
    RunOptions,
    RunOptionsError,
    effective_repair_scope_policy,
    effective_run_config,
    read_run_options_for_state,
)
from .shared import (
    CandidatePushError,
    CommitBoundaryError,
    CycleArtifactService,
    GitOwnership,
    OrchestrationError,
    ScopeApprovalRequired,
    StepExecutionFailure,
    _AGENT_ARTIFACTS,
    _RECOVERY_ATTEMPT_ARTIFACTS,
    _REVISION_ARTIFACTS,
    _archive_attempt,
    _archive_attempt_target,
    _archive_attempt_tree,
    bounded_v2_report,
    _git_ownership,
    _git_ownership_payload,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    _safe_candidate_tree,
    _status_has_unstaged_or_untracked,
    bounded_parse_detail,
    chat_client,
)
from .revision import has_deferred_contract_mismatches
from .check_repair import (
    CheckRepairLadder,
    GateAcceptanceService,
    _hard_integrity_failures,
    gate_mutable_authority,
    _soft_check_failures,
)
from .candidate import CandidateLifecycle
from .pipeline_v2 import (
    CyclePlan,
    FailureDetail,
    PipelineFailure,
    PipelineV2Context,
    PipelineV2Coordinator,
    PipelineV2Operations,
    check_repair_dir,
    check_repair_fingerprint,
    correction_dir,
    gate_dir,
    step_dir as cycle_step_dir,
)
from .recovery import (
    RecoveryCoordinator,
    normalize_exit_reason,
    project_exit,
)
from .check_recovery import CheckInfrastructureRecovery
from .review_recovery import ReviewRecovery
from .worker_recovery import WorkerRecovery
from .resume_validation import (
    ResumedRun,
    _load_evidence,
    _load_revision,
    _persist_planner_conversation,
    _read_repository_reference,
    _semantic_revision_scope,
    completed_step_records,
    load_correction_plan,
    read_candidate_record,
    read_cycle_record,
    validate_resume,
    verify_correction_scope,
)
from .implementation import ImplementationService
from .gates import GateService
from .publication import PublicationService
from .review_service import ReviewService

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

def _terminal_step_fields(
    state: Mapping[str, Any], failed_step: str | None, terminal: str = "failed"
) -> dict[str, Any]:
    """State fields that close every step when a run becomes terminal.

    The failed step and any step still marked ``running`` take *terminal*;
    later steps stay ``waiting``; ``current_step`` is cleared.
    """

    steps = state.get("steps") if isinstance(state, Mapping) else None
    if not isinstance(steps, list):
        return {"current_step": None}
    closed: list[Any] = []
    for item in steps:
        if isinstance(item, dict) and (
            item.get("id") == failed_step or item.get("status") == "running"
        ):
            item = {**item, "status": terminal}
        closed.append(item)
    return {"steps": closed, "current_step": None}

@dataclasses.dataclass(frozen=True)
class PreparedV2Run:
    """A planned, approved and prepared v2 run, ready for its first step."""

    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: Any
    info: WorktreeInfo
    ownership_before: GitOwnership
    base_tree_sha: str
    checkpoint: ResumeCheckpoint | None

def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (ResumeIntegrityError, ResumeRequiresOperatorError)):
        # Their stable ``.code`` is the authority: these reasons must stay
        # recognizable as permanently non-resumable.
        return exc.code
    if isinstance(exc, OrchestrationError) and str(exc).startswith("CHECK_PREFLIGHT_FAILED:"):
        return str(exc).split()[0]
    if isinstance(exc, OrchestrationError) and str(exc).startswith(
        "CHECK_INFRASTRUCTURE_UNAVAILABLE:"
    ):
        return "CHECK_INFRASTRUCTURE_UNAVAILABLE"
    if isinstance(exc, (OrchestrationError, ValidationError)):
        detail = str(exc)
        for code in (
            "AGENT_GIT_VIOLATION", "ROLLBACK_FAILED", "ROLLBACK_TREE_MISMATCH",
            "TREE_MISMATCH", "HEAD_MISMATCH", "DURABLE_ARTIFACT_CORRUPTED",
        ):
            if detail.startswith(code + ":"):
                return code
    if isinstance(exc, CandidatePushError):
        return exc.code
    if isinstance(exc, CommitBoundaryError):
        return "TOCTOU_FAILURE"
    if isinstance(exc, GitError):
        return "GIT_FAILURE"
    if isinstance(exc, AgentError):
        return getattr(exc, "code", "AGENT_FAILURE")
    if isinstance(exc, PlanRepositoryPreconditionError):
        return exc.code
    if isinstance(exc, ApprovalError):
        return "PLAN_APPROVAL_INVALID"
    if isinstance(exc, ExecutionSelectionError):
        return "EXECUTION_SELECTION_INVALID"
    if isinstance(exc, PlanParseError):
        return "PLANNER_OUTPUT_INVALID"
    if isinstance(exc, ReviewParseError):
        return "REVIEWER_OUTPUT_INVALID"
    if isinstance(exc, LLMError):
        return "LLM_FAILURE"
    if isinstance(exc, ValidationError):
        return "CHECK_SETUP_INVALID"
    if isinstance(exc, WorkspaceSetupError):
        return exc.code
    if isinstance(exc, GitHubIntegrationError):
        return getattr(exc, "code", "GITHUB_WORKSTREAM_FAILURE")
    return "INTERNAL_HARNESS_ERROR"

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
        reviewer_client: Any | None = None,
        recommender_client: Any | None = None,
        github_client: GitHubWorkstreamClient | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be a HarnessConfig")
        self.config = config
        self.planner_client = planner_client
        self.reviewer_client = reviewer_client
        self.recommender_client = recommender_client
        self.github_client = (
            github_client if github_client is not None else NullGitHubWorkstreamClient()
        )
        self.trace_sink = trace_sink
        self.environment = (
            config.runtime_environment if config.runtime_environment else os.environ
        )
        self.run_options: RunOptions | None = None
        self.secrets: tuple[str, ...] = ()
        self.repair_scope = EffectiveRepairScopePolicy("deny-expansion", 4, "run-options")
        self.last_selection: Any | None = None
        self.trace: TraceStream | None = None
        self.trace_cycle = 1
        self.implementation = ImplementationService(self)
        self.gates = GateService(self)
        self.reviews = ReviewService(self)
        self.publication = PublicationService(self)

    def begin_trace(
        self, run_dir: Path, run_id: str, *, created: bool, pipeline_version: int = 2,
    ) -> None:
        """Attach the observation stream without changing run authority."""

        self.trace = TraceStream(
            run_dir,
            run_id,
            pipeline_version=pipeline_version,
            sink=self.trace_sink,
            secrets=self.secrets,
        )
        if created:
            self.trace.emit(
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
        stream = self.trace
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
            for key in ("operation", "attempt", "attempts", "http_status", "elapsed_ms")
            if isinstance(observation.get(key), (str, int))
        }
        self.trace_emit(f"transport.{event}", phase="transport", data=data)
    def recovery(self, store: RunStateStore) -> RecoveryCoordinator:
        """The recovery coordinator bound to this run's durable state."""

        return RecoveryCoordinator(store, emit=self.trace_emit)
    def check_recovery(self, store: RunStateStore) -> CheckInfrastructureRecovery:
        return CheckInfrastructureRecovery(
            self.recovery(store), store=store, budgets=self.run_options.recovery,
        )
    def review_recovery(self, store: RunStateStore) -> ReviewRecovery:
        return ReviewRecovery(self.recovery(store), budgets=self.run_options.recovery)
    def worker_recovery(self, store: RunStateStore) -> WorkerRecovery:
        return WorkerRecovery(
            self.recovery(store), store=store,
            budgets=self.run_options.recovery, secrets=self.secrets,
        )
    @staticmethod
    def trace_time() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    def trace_selected_profile(
        self, profile_id: str, role: ExecutionRole, *, step_id: str | None = None,
    ) -> Any | None:
        selection = self.last_selection
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
                    agent_env_allowlist=self.config.codex_runtime.env_allowlist,
                    codex_home=self.config.codex_runtime.home,
                    claude_config_home=self.config.claude_runtime.home,
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
    def reviewer_for_profile(self, profile_id: str) -> Reviewer:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.REVIEWER)
        client = self.reviewer_client
        if client is None:
            client = chat_client(
                build_llm_endpoint(profile), self.environment, self.trace_transport
            )
        return Reviewer(client, allow_format_repair=True)
    def _recommender_for_profile(self, profile_id: str) -> ExecutionRecommender:
        profile = profile_for_role(self.config, profile_id, ExecutionRole.PLANNER)
        client = self.recommender_client
        if client is None:
            # This is deliberately a new client: the recommender has no
            # planner conversation/history, while using the same profile
            # endpoint and transport policy.
            client = chat_client(
                build_llm_endpoint(profile), self.environment, self.trace_transport
            )
        return ExecutionRecommender(client)
    def _maybe_recommend_profiles(
        self,
        store: RunStateStore,
        run_dir: Path,
        planner_profile_id: str,
    ) -> None:
        if not self.config.ui.enable_profile_recommendation:
            return
        profiles = profiles_for_config(self.config)
        implementers = tuple(
            profile for profile in profiles.values() if ExecutionRole.IMPLEMENTER in profile.roles
        )
        reviewers = tuple(
            profile for profile in profiles.values() if ExecutionRole.REVIEWER in profile.roles
        )
        if len(implementers) <= 1 and len(reviewers) <= 1:
            return
        try:
            contract = (run_dir / "implementation_contract.md").read_text(encoding="utf-8")
            recommendation = self._recommender_for_profile(planner_profile_id).recommend(
                contract,
                implementers,
                reviewers,
                artifacts_dir=run_dir,
            )
        except (LLMError, RecommendationError, OSError, UnicodeError) as exc:
            message = redact(" ".join(str(exc).split()), self.secrets)
            warning = f"{type(exc).__name__}: {message}"[:1000]
            try:
                write_recommendation_error(run_dir, warning)
            except (OSError, UnicodeError):
                pass
            store.update(
                status=RunStatus.PLANNING,
                recommendation={"status": "FAILED", "warning": warning},
            )
            return
        store.update(
            status=RunStatus.PLANNING,
            recommendation={
                "status": "READY",
                "implementer_profile": recommendation.implementer_profile,
                "reviewer_profile": recommendation.reviewer_profile,
                "rationale": recommendation.rationale,
            },
        )
    def executor_for_profile(
        self,
        profile_id: str,
        role: ExecutionRole,
        *,
        forbidden_env_names: tuple[str | None, ...] = (),
    ) -> Any:
        """Resolve one generic executor through the infrastructure boundary."""

        profile = profile_for_role(self.config, profile_id, role)
        runtime = ExecutorRuntimeConfig(
            config=self.config,
            environment=self.environment,
            codex_home=self.config.codex_runtime.home,
            claude_home=self.config.claude_runtime.home,
            forbidden_env_names=forbidden_env_names,
        )
        return executor_for_profile(profile, runtime)
    def run_revision(
        self,
        *,
        worktree: Path,
        prompt: str,
        artifact_dir: Path,
        profile_id: str,
        role: ExecutionRole,
        mutable_paths: tuple[str, ...] = (),
    ) -> Any:
        """Run one semantic-revision or check-repair worker pass."""

        profile = profile_for_role(self.config, profile_id, role)
        executor = self.executor_for_profile(profile.id, role)
        return executor.run(
            AgentRunRequest(
                role=role,
                profile_id=profile.id,
                prompt=redact(prompt, self.secrets),
                worktree=Path(worktree),
                artifact_dir=Path(artifact_dir),
                mutable_paths=mutable_paths,
                prompt_mode="revision",
            )
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
        store.update(status=state.get("status", RunStatus.PLANNING), cycles=cycles)
    def update_v2_usage(self, store: RunStateStore, run_dir: Path) -> None:
        """Publish the per-cycle token totals.

        Always derived from persisted artifacts
        (``cycles/NNN/implementation/steps/Sxx/step.json`` and every worker
        report), never from ``state.steps``.
        """

        store.update(
            status=store.load().get("status", RunStatus.PLANNING),
            usage=phase_usage_summary(run_dir),
        )
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
            atomic_write_text(step_dir / "agent.result.json", _json_text(payload))
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
            atomic_write_text(artifact_dir / "agent.result.json", _json_text(payload))
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
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.BLOCKED,
            RunStatus.PLAN_REJECTED,
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
            RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.BLOCKED,
            RunStatus.PLAN_REJECTED, RunStatus.WAITING_HUMAN,
            RunStatus.WAITING_EXTERNAL, RunStatus.WAITING_CHECK_INFRASTRUCTURE,
            RunStatus.WAITING_CHECK_REPAIR,
            RunStatus.WAITING_REMOTE, RunStatus.WAITING_SCOPE_APPROVAL,
            RunStatus.WAITING_CONTRACT_REPAIR, RunStatus.COMMITTED, RunStatus.PUBLISHED,
        }:
            try:
                write_run_diagnostics(self.config, result.run_dir)
            except Exception:
                # ``write_run_diagnostics`` records diagnostics.error.txt when
                # possible.  A reporting failure must not alter the pipeline.
                pass
        return result
    @staticmethod
    def closing_step_fields(store: RunStateStore, terminal: str) -> dict[str, Any]:
        try:
            return _terminal_step_fields(store.load(), None, terminal)
        except (OSError, ValueError):
            return {}
    def _prepare_v2_run(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        context: str,
        repository_reference: RepositoryReference,
        planner_profile: ModelProfile,
        existing_plan: TaskPlanV2 | None = None,
        existing_info: WorktreeInfo | None = None,
    ) -> "PreparedV2Run | RunResult":
        """Plan, obtain the approved selection, create the worktree and set up.

        Returns a terminal :class:`RunResult` for BLOCKED/REJECTED plans.  None
        of this is ever replayed by a resume.
        """

        revision_enabled = self.run_options.semantic_revision_enabled
        repair_enabled = self.run_options.max_review_repair_cycles > 0
        check_repair_enabled = self.run_options.max_check_repair_attempts > 0
        planner_profile_id = planner_profile.id
        if existing_plan is None:
            planner = PlannerV2(
                self.planner_client or chat_client(build_llm_endpoint(planner_profile), self.environment, self.trace_transport),
                repository_reference=repository_reference,
                planning=self.config.planning,
                check_catalog=self.config.check_catalog,
                default_check_ids=self.config.default_check_ids,
                prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
                repository_preconditions=RepositoryPreconditions(
                    repo, resolve_tree(repo, base_sha),
                ),
                on_event=lambda name, data: self.trace_emit(name, phase="planning", cycle=1, data=data),
            )
            plan_started_at = self.trace_time()
            plan_started_mono = time.perf_counter()
            self.trace_emit(
                "plan.started",
                phase="planning",
                cycle=1,
                data={
                    "session": self.trace_session(
                        profile=planner_profile,
                        selected=None,
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=None,
                        started_at=plan_started_at,
                        started_mono=plan_started_mono,
                        tree_before=resolve_tree(repo, base_sha),
                    )
                },
            )
            try:
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
            except Exception:
                # The failed planner answer is deliberately left in place and
                # will be archived by the next planner attempt.
                self.write_checkpoint(
                    run_dir, ResumePhase.PLANNER, head=base_sha,
                    tree=resolve_tree(repo, base_sha),
                )
                raise
            _persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            self.trace_emit(
                "plan.completed",
                phase="planning",
                cycle=1,
                data={
                    "decision": plan.decision.value,
                    "title": plan.title,
                    "session": self.trace_finished_model_session(
                        profile=planner_profile,
                        selected=None,
                        role=ExecutionRole.PLANNER,
                        prompt_bytes=(
                            (run_dir / "planner.request.txt").stat().st_size
                            if (run_dir / "planner.request.txt").is_file() else None
                        ),
                        started_at=plan_started_at,
                        started_mono=plan_started_mono,
                        usage=(
                            getattr(planner, "last_usage", None)
                            if getattr(planner, "last_usage", None) is not None
                            else read_usage_artifact(run_dir / "planner.usage.json")
                        ),
                        tree_before=resolve_tree(repo, base_sha),
                        tree_after=resolve_tree(repo, base_sha),
                        final_message=getattr(plan, "raw", None),
                    ),
                },
            )
        else:
            # Resume of PLANNER has already produced and durably parsed this
            # plan.  Re-entering setup must never call the planner again.
            plan = existing_plan
        store.update(
            status=RunStatus.PLANNING,
            planning_protocol="v2",
            planner={
                "decision": plan.decision.value,
                "blocker_kind": plan.blocker_kind.value if plan.blocker_kind else None,
                "blockers": plan.blockers if plan.decision is PlanDecision.BLOCKED else None,
                "title": plan.title,
                "model": planner_profile.model,
                "profile_id": planner_profile.id,
                "selection_mode": planner_profile.selection_mode.value,
                # operator_recovery: the plan was pasted by the operator and
                # no planner completion produced it.
                "source": plan_source(run_dir),
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
                "required_checks": list(plan.required_checks),
                "steps": [
                    {"id": step.id, "title": step.title,
                     "execution_class": step.execution_class.value,
                     "recommended_profile": self.config.routing.profile_for(step.execution_class),
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": self.run_options.final_reviewer_profile,
            },
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": self.config.routing.profile_for(step.execution_class)}
                for step in plan.steps
            ],
            current_step=None,
        )
        self.cycle_update(
            store, 1, status="running", plan_summary=plan.title,
            steps_summary=[{"id": step.id, "title": step.title} for step in plan.steps],
        )
        if plan.decision is PlanDecision.BLOCKED:
            kind = plan.blocker_kind
            if kind is BlockerKind.REPOSITORY_EVIDENCE:
                reason = "REPOSITORY_EVIDENCE_RECOVERY_EXHAUSTED"
                detail: dict[str, Any] = {
                    "blocker_kind": kind.value,
                    "blockers": plan.blockers,
                    "action": "operator must clarify the named repository path or symbol",
                }
            elif kind is BlockerKind.SPEC_DECISION:
                reason = "SPEC_DECISION_REQUIRED"
                detail = {
                    "blocker_kind": kind.value,
                    "blockers": plan.blockers,
                    "action": "operator must resolve the product choice left open by the SPEC",
                }
            elif kind is BlockerKind.SECURITY_POLICY:
                reason = "SECURITY_POLICY_DECISION_REQUIRED"
                detail = {
                    "blocker_kind": kind.value,
                    "blockers": plan.blockers,
                    "action": "operator must decide the security or policy question",
                }
            elif kind is BlockerKind.ATOMIC_SCOPE:
                reason = "ATOMIC_SCOPE_POLICY_LIMIT"
                detail = {
                    "blocker_kind": kind.value,
                    "blockers": plan.blockers,
                    "planning_limits": {
                        "max_steps_per_plan": self.config.planning.max_steps_per_plan,
                        "single_step_max_mutable_paths": self.config.planning.single_step_max_mutable_paths,
                        "staged_step_max_mutable_paths": self.config.planning.staged_step_max_mutable_paths,
                    },
                    "action": "operator must authorize a higher planning limit or split the requested work",
                }
            else:
                reason = "PLANNER_BLOCKED_REQUIRES_OPERATOR"
                detail = {"blocker_kind": None, "blockers": plan.blockers}
            state = store.update(
                status=RunStatus.WAITING_HUMAN,
                failure={"reason": reason, "detail": detail},
            )
            return RunResult(run_dir, RunStatus.WAITING_HUMAN, state)

        # Every source of this plan (planner, resumed planner answer, operator
        # recovery) must be possible against the base tree before it can be
        # offered for approval.  Resume re-enters here, so it is checked again.
        validate_plan_repository_topology(repo, resolve_tree(repo, base_sha), plan)
        try:
            # REQUIRED_CHECKS has already been parsed against the trusted
            # catalogue.  Materialize those exact trusted definitions before
            # the plan can become approval authority.
            selected_checks = self.config.select_checks(plan.required_checks)
            # Freeze the whole trusted catalogue, not just this selection: a
            # correction plan may legitimately require another approved check,
            # and it must still run the argv approved at this boundary.
            write_check_authority(
                run_dir, tuple(self.config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            _bundle, _bundle_sha = validate_implementation_bundle(run_dir)
            plan_identity = compute_plan_identity_from_run(run_dir)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update(status=RunStatus.PLANNING, plan_identity=asdict(plan_identity))
        self.write_checkpoint(
            run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha,
            tree=resolve_tree(repo, base_sha), plan_identity=plan_identity,
        )

        if self.config.approval.require_plan_approval:
            store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
            approval = wait_for_plan_approval(
                run_dir, identity=plan_identity,
                poll_interval_seconds=self.config.approval.poll_interval_seconds,
            )
            if approval.decision is ApprovalDecision.REJECT:
                state = store.update(status=RunStatus.PLAN_REJECTED)
                return RunResult(run_dir, RunStatus.PLAN_REJECTED, state)
            try:
                selection, execution_sha = read_execution_selection_with_sha256(run_dir)
                validate_execution_selection(self.config, selection)
                durable_identity = compute_plan_identity_from_run(run_dir)
                if durable_identity.execution_sha256 != execution_sha:
                    raise ApprovalError("execution selection hash mismatch")
                read = compute_plan_identity_from_run(run_dir)
                bound = read_plan_approval(run_dir, expected_identity=read)
                if bound is None or bound.bundle_sha256 != durable_identity.bundle_sha256:
                    raise ApprovalError("v2 approval is not bound to the exact bundle")
            except (ExecutionSelectionError, ApprovalError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        else:
            requested = resolve_execution_selection(
                self.config,
                planner_profile_id=planner_profile_id,
                plan_steps=plan.steps,
                semantic_reviser_profile_id=(
                    self.run_options.semantic_reviser_profile
                    if revision_enabled or repair_enabled else None
                ),
                check_repair_profile_id=(
                    self.run_options.check_repair_profile
                    if check_repair_enabled else None
                ),
                final_reviewer_profile_id=self.run_options.final_reviewer_profile,
                fallback_authority=self.run_options.recovery.execution_fallbacks,
            )
            selection = ensure_execution_selection(run_dir, requested)
            durable_identity = compute_plan_identity_from_run(run_dir)

        # Re-read the complete manifest after the approval transaction.  The
        # first validation protects the approval surface; this one closes the
        # race between approval and worktree creation.  The bundle bytes must
        # still be the ones hashed at planning time and bound by the approval.
        try:
            bundle, bundle_sha = validate_implementation_bundle(
                run_dir, expected_step_ids=[step.id for step in plan.steps]
            )
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        if bundle_sha != plan_identity.bundle_sha256:
            raise ApprovalError("PLAN_APPROVAL_INVALID: implementation bundle changed after planning")

        if selection.planner.profile_id != planner_profile_id:
            raise ExecutionSelectionError("execution selection planner is not the run planner")
        if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
            raise ExecutionSelectionError("execution selection steps do not match the plan")
        if [item.execution_class for item in selection.steps] != [step.execution_class for step in plan.steps]:
            raise ExecutionSelectionError("execution selection classes do not match the plan")
        if not isinstance(selection, ExecutionSelection) or selection.schema_version != EXECUTION_SELECTION_SCHEMA:
            raise ExecutionSelectionError("v2 execution requires the generic execution selection")
        validate_execution_selection(self.config, selection)
        self.last_selection = selection
        execution_state: dict[str, Any] = {
            "planner": asdict(selection.planner),
            "steps": [
                {"step_id": item.step_id, "implementer": asdict(item.implementer)}
                for item in selection.steps
            ],
        }
        if selection.semantic_reviser is not None:
            execution_state["semantic_reviser"] = asdict(selection.semantic_reviser)
        if selection.check_repair is not None:
            execution_state["check_repair"] = asdict(selection.check_repair)
        execution_state["final_reviewer"] = asdict(selection.final_reviewer)
        store.update(status=RunStatus.PLANNING, execution=execution_state,
                     plan_identity=asdict(durable_identity))
        self.trace_emit(
            "plan.approved",
            phase="planning",
            cycle=1,
            data={
                "plan_identity": asdict(durable_identity),
                "execution_selection_sha256": durable_identity.execution_sha256,
            },
            once=True,
        )
        base_tree_sha = resolve_tree(repo, base_sha)
        self.write_checkpoint(
            run_dir, ResumePhase.WORKTREE_SETUP, head=base_sha,
            tree=base_tree_sha, plan_identity=durable_identity,
            execution_selection_sha256=durable_identity.execution_sha256,
        )
        # Plan approval complete: the next operation is the first step.
        checkpoint = self._initial_checkpoint(plan, base_sha, base_tree_sha, durable_identity)

        branch = build_run_branch(plan.title, run_id)
        if existing_info is None:
            info = create_run_worktree(
                repo, base_ref=base_sha, branch=branch,
                worktree_path=self.config.worktrees_root / run_id,
                require_clean_base=self.config.require_clean_base,
            )
        else:
            info = existing_info
        store.update(status=RunStatus.WORKTREE_READY, branch=info.branch,
                     worktree=str(info.worktree), base_sha=info.base_sha)
        self.publication.ensure_github_issue_metadata(
            store=store, run_id=run_id, plan_title=plan.title, info=info,
        )
        self.trace_emit(
            "worktree.created",
            phase="setup",
            cycle=1,
            data={
                "branch": info.branch,
                "worktree": str(info.worktree),
                "base_sha": info.base_sha,
                "tree_sha": base_tree_sha,
                "resumed_setup": existing_info is not None,
            },
            once=True,
        )
        ownership_before = _git_ownership(repo, info.worktree)
        # Persisted with an explicit status so a later resume has the stronger
        # durable ownership proof of the pre-execution boundary.
        store.update(
            status=RunStatus.PREPARING,
            git_ownership=_git_ownership_payload(ownership_before),
        )
        setup_results = self.check_recovery(store).prepare_workspace(
            worktree=info.worktree, run_dir=run_dir,
            run_setup=lambda: prepare_workspace(
                info.worktree, self.config.workspace_setup,
                environment=self.environment, artifacts_dir=run_dir,
                secrets=self.secrets,
            ),
        )
        store.update(status=RunStatus.PREPARING,
                     workspace_setup=[asdict(result) for result in setup_results])
        # The candidate starts as the base tree and all later gates use the
        # actual index identity, never an inferred file list.
        stage_all(info.worktree)
        candidate_tree = index_tree_sha(info.worktree)
        if candidate_tree != base_tree_sha:
            raise OrchestrationError("initial candidate tree does not match base")
        # The workspace preflight is the first live use of the authority; the
        # expected hash is the one this run's identity already durably bound.
        check_config, check_ids = config_with_check_authority(
            self.config, run_dir, expected_sha256=durable_identity.checks_sha256,
        )
        preflight_failures = self.gates.run_check_preflights_recoverably(
            store=store, worktree=info.worktree, check_config=check_config,
            check_ids=check_ids or plan.required_checks,
            counter_key="check-preflight:workspace", phase="preparing",
        )
        if preflight_failures:
            raise OrchestrationError(preflight_failures[0])
        # Worktree and setup complete: still the first step.
        if checkpoint is not None:
            write_checkpoint(run_dir, checkpoint)
        return PreparedV2Run(plan, bundle, selection, info, ownership_before, base_tree_sha, checkpoint)
    @staticmethod
    def _initial_checkpoint(
        plan: TaskPlanV2, base_sha: str, base_tree_sha: str, identity: PlanIdentity,
    ) -> ResumeCheckpoint | None:
        if identity.execution_sha256 is None or not plan.steps:
            return None
        return ResumeCheckpoint(
            phase=ResumePhase.IMPLEMENT_STEP, review_cycle=1,
            step_id=plan.steps[0].id, expected_head_sha=base_sha,
            expected_tree_sha=base_tree_sha,
            execution_selection_sha256=identity.execution_sha256,
            plan_identity=identity,
        )
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
        identity = record[0].plan_identity if record is not None else None
        approved = identity.checks_sha256 if identity is not None else None
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
        tree: str | None,
        cycle: int | None = None,
        stage: GateStage | None = None,
        step_id: str | None = None,
        check_repair_attempt: int | None = None,
        correction_bundle_sha256: str | None = None,
        expected_parent_sha: str | None = None,
        next_step_id: str | None = None,
        plan_identity: PlanIdentity | None = None,
        execution_selection_sha256: str | None = None,
    ) -> None:
        """Persist the next operation that has not succeeded yet.

        Only the run identity (plan identity and execution selection) is
        carried over from the current checkpoint; every other field describes
        exactly the new boundary.  A run without a pending checkpoint has
        nothing to resume and is left untouched.
        """

        record = read_checkpoint_record(run_dir)
        if record is None or record[1] != "pending":
            return
        previous = record[0]
        write_checkpoint(run_dir, ResumeCheckpoint(
            phase=phase,
            review_cycle=cycle or previous.review_cycle,
            stage=stage,
            step_id=step_id,
            next_step_id=next_step_id,
            check_repair_attempt=check_repair_attempt,
            expected_head_sha=head,
            expected_parent_sha=expected_parent_sha,
            expected_tree_sha=tree,
            execution_selection_sha256=(
                execution_selection_sha256 or previous.execution_selection_sha256
            ),
            plan_identity=plan_identity or previous.plan_identity,
            correction_bundle_sha256=correction_bundle_sha256,
        ))
    def execute_v2(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
        repo: Path,
        base_sha: str,
        context: str,
        repository_reference: RepositoryReference,
        *,
        prepared: "PreparedV2Run | RunResult | None" = None,
    ) -> RunResult:
        """Prepare a new run, then delegate its execution to the coordinator."""

        if prepared is None:
            planner_profile = profile_for_role(
                self.config, store.load()["execution"]["planner"]["profile_id"],
                ExecutionRole.PLANNER,
            )
            prepared = self._prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                repository_reference, planner_profile,
            )
        if isinstance(prepared, RunResult):
            return prepared
        if prepared.checkpoint is None:
            # A v2 run always has an execution selection bound to its plan
            # identity once approved; never run without one.
            raise OrchestrationError("v2 run has no execution selection identity")
        pipeline = PipelineV2Context(
            run_dir=run_dir, run_id=run_id, spec=spec, context=context, repo=repo,
            base_sha=base_sha, base_tree_sha=prepared.base_tree_sha,
            repository_reference=repository_reference, info=prepared.info,
            plan=prepared.plan, bundle=prepared.bundle, selection=prepared.selection,
            options=self.run_options,
        )
        return pipeline, prepared.checkpoint
    def pipeline_operations(self, store: RunStateStore) -> PipelineV2Operations:
        """Bind every operation the coordinator sequences to this run."""

        bind = functools.partial
        candidate_lifecycle = CandidateLifecycle(
            staging_remote=self.config.repository.remote,
            authorize_tree=self.publication.authorize_candidate_tree,
            gate_mutable_authority=lambda ctx, cycle_plan, stage: gate_mutable_authority(
                ctx.run_dir, cycle_plan.cycle.number, stage,
                base_paths=self.effective_cycle_scope(ctx, cycle_plan),
                policy_config=self.repair_scope,
                require_attempt_records=True,
            ),
            push_tree=self.publication.push_candidate,
            cycle_update=self.cycle_update,
        )
        gate_acceptance = GateAcceptanceService(
            secrets=self.secrets,
            repair_scope_policy=self.repair_scope,
            authorize_candidate_tree=self.publication.authorize_candidate_tree,
            check_repair_attempts=self.gates.check_repair_attempt_records,
            load_revision=_load_revision,
            trace_emit=self.trace_emit,
            bounded_detail=bounded_parse_detail,
        )
        cycle_artifacts = CycleArtifactService(
            cycle_update=self.cycle_update,
            trace_emit=self.trace_emit,
            set_trace_cycle=lambda number: setattr(self, "trace_cycle", number),
        )
        return PipelineV2Operations(
            checkpoint=lambda ctx, phase, **fields: self.write_checkpoint(
                ctx.run_dir, phase, **fields
            ),
            current_head=lambda ctx: current_head(ctx.info.worktree),
            candidate_tree=lambda ctx: candidate_tree_sha(ctx.info.worktree),
            begin_cycle=bind(cycle_artifacts.begin, store),
            load_cycle=lambda ctx, number: read_cycle_record(ctx.run_dir, number),
            initial_plan=self._initial_cycle_plan,
            review_implementation_correction=self.reviews.review_implementation_correction,
            plan_correction=bind(self.reviews.plan_correction, store),
            load_correction=self._load_correction,
            completed_steps=self.completed_steps,
            execute_step=bind(self.implementation.execute_cycle_step, store),
            accept_step=bind(self.implementation.resume_step_acceptance, store),
            unresolved_mismatches=lambda ctx, plan: has_deferred_contract_mismatches(
                self.completed_steps(ctx, plan)
            ),
            semantic_revision=bind(self.reviews.semantic_revision, store),
            semantic_review_correction=bind(self.reviews.semantic_review_correction, store),
            run_gate=bind(self.gates.run_gate, store),
            load_gate_evidence=lambda ctx, number, stage: _load_evidence(
                gate_dir(ctx.run_dir, number, stage)
            ),
            load_accepted_gate_evidence=self.gates.load_accepted_gate_evidence,
            accept_gate_state=lambda ctx, cycle_plan, stage, evidence: gate_acceptance.accept(
                store, ctx, cycle_plan, stage, evidence,
                base_paths=self.effective_cycle_scope(ctx, cycle_plan),
            ),
            check_repair_attempts=lambda ctx, number, stage: self.gates.check_repair_attempt_records(
                ctx.run_dir, number, stage
            ),
            check_repair_attempt=bind(self.gates.run_check_repair_attempt, store),
            hard_failures=_hard_integrity_failures,
            soft_failures=_soft_check_failures,
            create_candidate=bind(candidate_lifecycle.create, store),
            load_candidate=lambda ctx, number: read_candidate_record(ctx.run_dir, number),
            push_candidate=bind(candidate_lifecycle.push, store),
            review_candidate=bind(self.reviews.review_candidate, store),
            record_review=bind(self.reviews.record_review, store),
            request_human=bind(self.reviews.request_human, store),
            review_repair_exhausted=bind(self.reviews.review_repair_exhausted, store),
            publish=bind(self.publication.publish_candidate, store),
            recovery_operations=CheckRepairLadder(
                replay_steps=functools.partial(self.gates.replay_approved_steps, store),
            ),
        )

    # -- cycle operations -----------------------------------------------------
    def _initial_cycle_plan(self, ctx: PipelineV2Context) -> CyclePlan:
        return CyclePlan(
            cycle=RunCycle(1, CycleKind.INITIAL),
            plan=ctx.plan,
            bundle=ctx.bundle,
            contracts_dir=ctx.run_dir,
            step_profile_ids={
                item.step_id: item.implementer.profile_id for item in ctx.selection.steps
            },
            step_fallback_profile_ids={
                item.step_id: tuple(profile.profile_id for profile in item.fallbacks)
                for item in ctx.selection.steps
            },
        )
    def cycle_plan(self, ctx: PipelineV2Context, number: int) -> CyclePlan:
        """The approved plan any durable cycle executed."""

        if number == 1:
            return self._initial_cycle_plan(ctx)
        cycle = read_cycle_record(ctx.run_dir, number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            previous = self.cycle_plan(ctx, number - 1)
            return CyclePlan(
                cycle=cycle,
                plan=previous.plan,
                bundle=previous.bundle,
                contracts_dir=previous.contracts_dir,
                step_profile_ids=previous.step_profile_ids,
                step_fallback_profile_ids=previous.step_fallback_profile_ids,
            )
        plan, bundle, bundle_sha = load_correction_plan(
            self.config, ctx.selection, ctx.run_dir, number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        return self.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )
    def correction_cycle_plan(
        self, ctx: PipelineV2Context, cycle: RunCycle, plan: TaskPlanV2,
        bundle: Mapping[str, Any], bundle_sha: str, *, creating: bool,
    ) -> CyclePlan:
        if cycle.kind is not CycleKind.REVIEW_REPLAN:
            raise PipelineFailure("REPLAN_CYCLE_REQUIRED")
        planned_profile_ids = {
            step.id: self.config.routing.profile_for(step.execution_class) for step in plan.steps
        }
        try:
            if creating:
                cycle_selection = ensure_cycle_execution_selection(
                    ctx.run_dir,
                    resolve_cycle_execution_selection(
                        self.config, cycle=cycle.number,
                        plan_steps=plan.steps,
                        step_profile_ids=planned_profile_ids,
                        fallback_authority=self.run_options.recovery.execution_fallbacks,
                    ),
                )
            else:
                cycle_selection = read_cycle_execution_selection(ctx.run_dir, cycle.number)
                validate_cycle_execution_selection(self.config, cycle_selection)
                if [item.step_id for item in cycle_selection.steps] != [step.id for step in plan.steps]:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
                if [item.execution_class for item in cycle_selection.steps] != [step.execution_class for step in plan.steps]:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
                if {
                    item.step_id: item.implementer.profile_id for item in cycle_selection.steps
                } != planned_profile_ids:
                    raise PipelineFailure("REPLAN_EXECUTION_SELECTION_MISMATCH")
        except (ExecutionSelectionError, OSError) as exc:
            raise PipelineFailure("REPLAN_EXECUTION_SELECTION_INVALID", str(exc)) from exc
        step_profile_ids = {
            item.step_id: item.implementer.profile_id for item in cycle_selection.steps
        }
        step_fallback_profile_ids = {
            item.step_id: tuple(profile.profile_id for profile in item.fallbacks)
            for item in cycle_selection.steps
        }
        return CyclePlan(
            cycle=cycle, plan=plan, bundle=bundle,
            contracts_dir=correction_dir(ctx.run_dir, cycle),
            step_profile_ids=step_profile_ids,
            correction_bundle_sha256=bundle_sha,
            step_fallback_profile_ids=step_fallback_profile_ids,
        )
    def approved_scope_before(self, ctx: PipelineV2Context, number: int) -> list[str]:
        """Every mutable path the plans of cycles ``1..number-1`` approved."""

        scope: set[str] = set()
        for earlier in range(1, number):
            earlier_plan = self.cycle_plan(ctx, earlier)
            scope |= set(self.effective_cycle_scope(ctx, earlier_plan))
        return sorted(scope)
    def _load_correction(
        self, ctx: PipelineV2Context, cycle: RunCycle, expected_sha: str | None,
    ) -> CyclePlan:
        """Read back the correction plan a checkpoint of this cycle is bound to."""

        plan, bundle, bundle_sha = load_correction_plan(
            self.config, ctx.selection, ctx.run_dir, cycle.number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        if expected_sha is None or bundle_sha != expected_sha:
            raise ResumeIntegrityError(f"cycle {cycle.number:03d} correction plan changed")
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha, self.repair_scope)
        return self.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )
    def completed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        return completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number, [step.id for step in cycle_plan.plan.steps],
        )
    def effective_cycle_scope(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> tuple[str, ...]:
        """The approved envelope plus the effective authority of every step.

        A contract repair contributes only through its step's effective
        authority (hash- and chain-verified); a semantic revision addition
        only through its own durable policy authority.  Nothing widens the
        scope merely because a repair exists.
        """

        scope = set(cycle_plan.mutable_scope)
        if cycle_plan.cycle.kind is CycleKind.REVIEW_IMPLEMENTATION and cycle_plan.cycle.number > 1:
            # A direct correction has no steps of its own: it inherits the
            # accepted authority of the cycle it corrects (as resume does).
            scope.update(self.effective_cycle_scope(
                ctx, self.cycle_plan(ctx, cycle_plan.cycle.number - 1),
            ))
        count = len(cycle_plan.plan.steps)
        for step in cycle_plan.plan.steps:
            artifact_dir = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id)
            if not (artifact_dir / "contract_repairs").is_dir():
                continue
            authority = self.implementation.resolve_step_authority(
                artifact_dir, step, self.implementation.approved_step_contract(cycle_plan, step),
                expected_tree=None, expected_plan_step_count=count,
            )
            scope.update(authority.mutable_scope)
        scope.update(_semantic_revision_scope(
            ctx.repo, ctx.run_dir, cycle_plan.cycle.number,
            self.repair_scope,
        ))
        return tuple(sorted(scope))
    def state_steps(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan, *, running: str | None = None,
    ) -> list[dict[str, Any]]:
        """The ``state.steps`` view of one cycle, derived from durable records."""

        ctx_steps = {record["id"]: record for record in self.completed_steps(ctx, cycle_plan)}
        rows: list[dict[str, Any]] = []
        for step in cycle_plan.plan.steps:
            record = ctx_steps.get(step.id)
            row: dict[str, Any] = {
                "id": step.id, "title": step.title,
                "profile_id": cycle_plan.step_profile_ids.get(step.id),
                "status": "waiting",
            }
            if record is not None:
                row.update(
                    status="deferred" if record["status"] == "DEFERRED_CONTRACT_MISMATCH" else "completed",
                    no_change=record.get("no_change", False),
                    usage=record["usage"],
                    input_tokens=record["usage"]["input_tokens"],
                    output_tokens=record["usage"]["output_tokens"],
                )
            elif step.id == running:
                row["status"] = "running"
            rows.append(row)
        return rows
    def step_failed(
        self, store: RunStateStore, run_dir: Path, failure: StepExecutionFailure,
    ) -> RunResult:
        _decision, terminal = project_exit(
            failure.reason, phase=self.checkpoint_phase(run_dir),
        )
        self.trace_emit(
            "step.failed",
            phase="implementation",
            cycle=self.trace_cycle,
            step_id=failure.step_id,
            data={
                "reason": failure.reason,
                "detail": failure.detail,
                "profile_id": failure.profile_id,
                "tree_before": failure.tree_before,
                "tree_after": failure.tree_after,
                "changed_paths": [],
                "commit_sha": None,
            },
        )
        return self.v2_failed(
            store, run_dir, failure.reason, failure.step_id, failure.detail,
            usage=failure.usage, profile_id=failure.profile_id,
            tree_before=failure.tree_before, tree_after=failure.tree_after,
            mismatch=failure.mismatch,
            mismatch_clean=failure.clean_contract_mismatch,
            mismatch_retry_count=failure.mismatch_retry_count,
            initial_mismatch=failure.initial_mismatch,
            index_tree_after=failure.index_tree_after,
            step_dir=failure.step_dir,
            terminal_status=terminal.status,
            auto_resumable=terminal.resumable,
        )
    def redact_revision_artifacts(self, artifact_dir: Path) -> None:
        for name in _REVISION_ARTIFACTS:
            redact_file(artifact_dir / name, self.secrets)
    def redact_step_artifacts(self, step_dir: Path) -> None:
        for name in _AGENT_ARTIFACTS:
            redact_file(step_dir / name, self.secrets)
    def v2_failed(
        self, store: RunStateStore, run_dir: Path, reason: str,
        step_id: str | None, detail: FailureDetail | None = None, *,
        usage: Mapping[str, int] | None = None,
        profile_id: str | None = None,
        tree_before: str | None = None,
        tree_after: str | None = None,
        mismatch: str | None = None,
        mismatch_clean: bool = False,
        mismatch_retry_count: int = 0,
        initial_mismatch: str | None = None,
        index_tree_after: str | None = None,
        step_dir: Path | None = None,
        terminal_status: RunStatus | None = None,
        auto_resumable: bool | None = None,
    ) -> RunResult:
        """Persist a failure that left its recovery loop, with its step record.

        Without an explicit ``terminal_status`` the failure is projected by
        the recovery policy: only a hard stop becomes ``FAILED``.
        """

        repeated_fixed_point = False
        retry_fingerprint: list[Any] | None = None
        if reason == "CHECK_REPAIR_EXHAUSTED" and isinstance(detail, Mapping):
            raw_failed_ids = detail.get("failed_check_ids")
            candidate_tree = detail.get("candidate_tree")
            try:
                checkpoint = read_checkpoint(run_dir)
            except ResumeCheckpointError:
                checkpoint = None
            if (
                isinstance(raw_failed_ids, list)
                and all(isinstance(item, str) for item in raw_failed_ids)
                and isinstance(candidate_tree, str)
                and checkpoint is not None
                and checkpoint.stage is not None
            ):
                fingerprint = check_repair_fingerprint(
                    candidate_tree, raw_failed_ids, checkpoint.stage,
                    detail.get("strategy") if isinstance(detail.get("strategy"), str) else "",
                )
                # The durable fingerprint is compared against a JSON round
                # trip: its failed-check set is normalized to a list so a
                # reloaded state is the same identity as the written one.
                retry_fingerprint = [fingerprint[0], list(fingerprint[1]), *fingerprint[2:]]
                prior_check_repair = store.load().get("check_repair")
                repeated_fixed_point = (
                    isinstance(prior_check_repair, Mapping)
                    and prior_check_repair.get("operator_retry_fingerprint") == retry_fingerprint
                )
                if repeated_fixed_point:
                    reason = "CHECK_REPAIR_FIXED_POINT"
                    detail = {
                        **dict(detail),
                        "fixed_point_fingerprint": retry_fingerprint,
                        "operator_message": "Code change or additional repair authority required",
                    }
                    terminal_status = None
                    auto_resumable = False
        if terminal_status is None:
            reason = normalize_exit_reason(reason)
            terminal_status = project_exit(
                reason, phase=self.checkpoint_phase(run_dir),
                remote_required=reason == "PUSH_FAILED",
            )[1].status
        if reason == "CHECK_REPAIR_EXHAUSTED":
            self.trace_emit(
                "check_repair.exhausted",
                phase="repair",
                cycle=self.trace_cycle,
                data={"reason": reason, "step_id": step_id},
                once=True,
            )
        elif reason == "CHECK_REPAIR_FIXED_POINT":
            self.trace_emit(
                "check_repair.fixed_point",
                phase="repair",
                cycle=self.trace_cycle,
                data={
                    "step_id": step_id,
                    "fingerprint": retry_fingerprint,
                    "operator_message": "Code change or additional repair authority required",
                },
                once=True,
            )
        try:
            self.update_v2_usage(store, run_dir)
        except (OSError, ValueError):
            pass
        state = store.load()
        fields = _terminal_step_fields(
            state, step_id, "waiting" if terminal_status is not RunStatus.FAILED else "failed",
        )
        if reason in {"CHECK_REPAIR_EXHAUSTED", "CHECK_REPAIR_FIXED_POINT"} and isinstance(detail, Mapping):
            failure_detail = dict(detail)
            evidence_sha = failure_detail.get("latest_evidence_sha256")
            failed_ids = failure_detail.get("failed_check_ids")
            candidate_tree = failure_detail.get("candidate_tree")
            attempt_count = failure_detail.get("attempt_count")
            budget = failure_detail.get("budget")
            try:
                checkpoint = read_checkpoint(run_dir)
            except ResumeCheckpointError:
                checkpoint = None
            reports: list[dict[str, Any]] = []
            if (
                checkpoint is not None and checkpoint.stage is not None
                and isinstance(attempt_count, int) and not isinstance(attempt_count, bool)
            ):
                root = check_repair_dir(run_dir, checkpoint.review_cycle, checkpoint.stage) / "attempts"
                for number in range(1, attempt_count + 1):
                    report_path = root / f"{number:03d}" / "report.json"
                    if report_path.is_file():
                        try:
                            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
                        except OSError:
                            digest = None
                        reports.append({
                            "attempt": number,
                            "artifact": report_path.relative_to(run_dir).as_posix(),
                            "sha256": digest,
                        })
            failure_detail["repair_reports"] = reports
            prior_check_repair = state.get("check_repair")
            if retry_fingerprint is None:
                fingerprint = check_repair_fingerprint(
                    candidate_tree if isinstance(candidate_tree, str) else "",
                    failed_ids if isinstance(failed_ids, list) else [],
                    checkpoint.stage if checkpoint is not None and checkpoint.stage is not None else "",
                    failure_detail.get("strategy")
                    if isinstance(failure_detail.get("strategy"), str) else "",
                )
                retry_fingerprint = [fingerprint[0], list(fingerprint[1]), *fingerprint[2:]]
            fixed_point = reason == "CHECK_REPAIR_FIXED_POINT"
            if fixed_point:
                failure_detail["fixed_point_fingerprint"] = retry_fingerprint
                failure_detail["operator_message"] = "Code change or additional repair authority required"
            fields["check_repair"] = {
                **(dict(prior_check_repair) if isinstance(prior_check_repair, Mapping) else {}),
                "status": "fixed_point" if fixed_point else "exhausted",
                "attempt_count": attempt_count,
                "budget": budget,
                "failed_check_ids": list(failed_ids) if isinstance(failed_ids, list) else [],
                "candidate_tree": candidate_tree,
                "repair_reports": reports,
                "latest_evidence_sha256": evidence_sha,
                "failure_classification": "product_check",
                "operator_retry_fingerprint": retry_fingerprint,
                "next_action": (
                    "Code change or additional repair authority required"
                    if fixed_point else "Retry deterministic gate"
                ),
            }
            detail = failure_detail
        if auto_resumable is not None:
            fields["recovery_resumable"] = auto_resumable
        cycles = list(state.get("cycles") or [])
        current_cycle = state.get("cycle", 1)
        for index, cycle in enumerate(cycles):
            if isinstance(cycle, dict) and cycle.get("number") == current_cycle:
                cycles[index] = {**cycle, "status": "waiting" if terminal_status is not RunStatus.FAILED else "failed", "failure": reason}
                break
        if cycles:
            fields["cycles"] = cycles
        if step_id is not None:
            detail = f"step={step_id}" + (f" {detail}" if detail is not None else "")
            step_usage = normalize_usage(usage) if usage is not None else empty_usage()
            if isinstance(fields.get("steps"), list):
                fields["steps"] = [
                    {**item, "usage": step_usage,
                     "input_tokens": step_usage["input_tokens"],
                     "output_tokens": step_usage["output_tokens"]}
                    if isinstance(item, dict) and item.get("id") == step_id else item
                    for item in fields["steps"]
                ]
            step_dir = step_dir or (
                cycle_step_dir(run_dir, current_cycle, step_id)
            )
            if not (step_dir / "step.json").exists():
                if profile_id is None:
                    state_steps = state.get("steps")
                    profile_id = next(
                        (item.get("profile_id") for item in state_steps
                         if isinstance(item, dict) and item.get("id") == step_id),
                        None,
                    ) if isinstance(state_steps, list) else None
                atomic_write_text(step_dir / "step.json", _json_text({
                    "id": step_id, "status": "FAILED", "reason": reason,
                    "profile_id": profile_id,
                    "tree_before": tree_before, "tree_after": tree_after,
                    **({"changed_paths": []} if reason == "AGENT_CONTRACT_MISMATCH" and tree_before == tree_after else {}),
                    **({"mismatch": bounded_v2_report(mismatch)} if mismatch else {}),
                    **({"mismatch_clean": mismatch_clean}
                       if reason == "AGENT_CONTRACT_MISMATCH" else {}),
                    **({"mismatch_retry_count": mismatch_retry_count}
                       if mismatch_retry_count else {}),
                    **({"initial_mismatch": bounded_v2_report(initial_mismatch)}
                       if initial_mismatch else {}),
                    **({"index_tree_after": index_tree_after}
                       if index_tree_after else {}),
                    "usage": step_usage,
                }))
        state = self.persist_exit(store, reason, detail, terminal_status, **fields)
        return RunResult(run_dir, terminal_status, state)
    @staticmethod
    def checkpoint_phase(
        run_dir: Path, *, default: ResumePhase = ResumePhase.IMPLEMENT_STEP,
    ) -> ResumePhase:
        try:
            checkpoint = read_checkpoint(run_dir)
        except ResumeCheckpointError:
            return default
        return checkpoint.phase if checkpoint is not None else default
    def persist_exit(
        self, store: RunStateStore, reason: str, detail: FailureDetail | None,
        status: RunStatus, **fields: Any,
    ) -> dict[str, Any]:
        """Write one terminal or waiting outcome; FAILED only for hard stops."""

        if isinstance(detail, str):
            detail = redact(detail, self.secrets)
        elif isinstance(detail, Mapping):
            detail = redact_mapping(detail, self.secrets)
        elif detail is not None:
            raise TypeError("failure detail must be text or a structured mapping")
        if status is RunStatus.FAILED:
            return store.record_failure(reason, detail, **fields)
        failure = {"reason": reason}
        if detail is not None:
            failure["detail"] = detail
        return store.update(status=status, failure=failure, **fields)
    def project_exception(
        self, store: RunStateStore, run_dir: Path, exc: Exception, *,
        operation: str = "orchestrator",
    ) -> RunResult:
        """Project an exception that escaped every recovery loop."""

        reason = normalize_exit_reason(_failure_reason(exc))
        phase = self.checkpoint_phase(run_dir)
        _decision, terminal = project_exit(reason, phase=phase)
        detail: FailureDetail
        auto_resumable: bool | None = None
        if reason == "INTERNAL_HARNESS_ERROR":
            message = " ".join(str(exc).split())[:500]
            detail = {
                "exception_type": type(exc).__name__,
                "message": message,
                "phase": phase.value,
                "operation": operation,
            }
            auto_resumable = self._checkpoint_is_retryable(store, run_dir)
            status = RunStatus.WAITING_EXTERNAL if auto_resumable else RunStatus.FAILED
        else:
            detail = " ".join(str(exc).split())[:500]
            status = terminal.status
        fields = self.closing_step_fields(
            store, "waiting" if status is not RunStatus.FAILED else "failed",
        )
        if auto_resumable is not None:
            fields["recovery_resumable"] = auto_resumable
        state = self.persist_exit(store, reason, detail, status, **fields)
        return RunResult(run_dir, status, state)
    def _checkpoint_is_retryable(self, store: RunStateStore, run_dir: Path) -> bool:
        """Only make an internal crash resumable after the full resume gate passes."""

        try:
            state = store.load()
            checkpoint = read_checkpoint(run_dir)
            if checkpoint is None:
                return False
            options, _digest = read_run_options_for_state(run_dir, state)
            config = effective_run_config(self.config, options)
            validate_resume(
                config=config,
                repair_scope=effective_repair_scope_policy(options),
                run_dir=run_dir,
                state=state,
                checkpoint=checkpoint,
                staging_remote=config.repository.remote,
            )
        except Exception:
            return False
        return True
    def restore_checkpoint_tree(self, resumed: ResumedRun) -> None:
        """Undo a failed attempt's in-scope edits, exactly and boundedly."""

        worktree = resumed.info.worktree
        expected = resumed.checkpoint.expected_tree_sha
        try:
            restore_paths_from_tree(worktree, expected, resumed.restore_paths)
            stage_all(worktree)
            restored = (
                index_tree_sha(worktree) == expected
                and candidate_tree_sha(worktree) == expected
                and not _status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except GitError:
            restored = False
        if not restored:
            raise ResumeRequiresOperatorError(
                "the checkpoint tree could not be restored exactly"
            )

    # -- operator plan recovery ----------------------------------------------
    def persist_recovered_plan(self, run_id: str, replacement_raw: str) -> None:
        def refuse(message: str) -> NoReturn:
            raise PlanRecoveryError(message)

        raw = validate_replacement_text(replacement_raw)
        try:
            selected = safe_run_id(run_id)
        except OrchestrationError as exc:
            refuse(str(exc))
        run_dir = (self.config.runs_root / selected).expanduser().resolve()
        if not (run_dir / "state.json").is_file():
            refuse("run state does not exist")
        store = RunStateStore(run_dir / "state.json")
        try:
            state = store.load()
        except (OSError, ValueError) as exc:
            refuse(f"run state is unreadable: {exc}")
        eligibility = plan_recovery_info(run_dir, state)
        if not eligibility.eligible:
            refuse(eligibility.reason or "run is not eligible for plan recovery")
        try:
            options, _ = read_run_options_for_state(run_dir, state)
            # The run's frozen options decide the policies and catalogues,
            # never the current defaults.
            config = effective_run_config(self.config, options)
        except RunOptionsError:
            refuse("run options are missing or invalid")
        checkpoint = read_checkpoint(run_dir)
        if checkpoint is None or checkpoint.phase is not ResumePhase.PLANNER:
            refuse("run is not at its PLANNER checkpoint")
        for name in ("spec.md", "context.txt"):
            if not (run_dir / name).is_file():
                refuse(f"{name} is missing")

        # The run is bound to its immutable stored BASE, not to where
        # base_ref points today.
        base_sha = state.get("base_sha")
        if not _is_object_id(base_sha):
            refuse("run base SHA is missing or invalid")
        if checkpoint.expected_head_sha is None or checkpoint.expected_tree_sha is None:
            refuse("PLANNER checkpoint has no BASE identity")
        try:
            repo = git_root(config.repo)
            if state.get("repo") not in {None, str(repo), str(config.repo)}:
                refuse("the configured repository is not the run repository")
            if resolve_commit(repo, base_sha) != base_sha:
                refuse("run base SHA does not resolve to itself")
            base_tree = resolve_tree(repo, base_sha)
            if checkpoint.expected_head_sha != base_sha:
                refuse("PLANNER checkpoint base SHA does not match the run")
            if checkpoint.expected_tree_sha != base_tree:
                refuse("PLANNER checkpoint base tree does not match the run")
            reference = _read_repository_reference(run_dir)
            if reference is None or reference.base_sha != base_sha:
                refuse("repository reference does not match the run base SHA")
            worktree_path = (config.worktrees_root / selected).expanduser().resolve()
            if worktree_path.exists() or str(worktree_path) in registered_worktrees(repo):
                refuse("a run worktree already exists")
            if any(
                ref.startswith("refs/heads/harness/") and ref.endswith(f"/{selected}")
                for ref in local_branches(repo)
            ):
                refuse("a run branch already exists")
        except GitError as exc:
            refuse(f"run Git identity cannot be verified: {exc}")

        try:
            profiles = tuple(profiles_for_config(config).values())
        except ProfileError as exc:
            refuse(f"profile catalogue is unavailable: {exc}")
        try:
            plan = parse_task_plan_v2(
                raw,
                planning=config.planning,
                check_catalog=config.check_catalog,
                default_check_ids=config.default_check_ids,
            )
            if plan.decision is not PlanDecision.READY:
                refuse("replacement plan must be STATUS: READY")
            validate_execution_mode_policy(plan, config.planning)
            validate_decomposition_policy(plan, config.planning)
        except V2PlanParseError as exc:
            refuse(f"replacement plan is invalid: {exc}")
        try:
            # An operator plan is bound by the same repository preconditions.
            validate_plan_repository_topology(repo, base_tree, plan)
        except PlanRepositoryPreconditionError as exc:
            refuse(f"{exc.code}: {exc}")
        except GitError as exc:
            refuse(f"run Git identity cannot be verified: {exc}")

        replacement_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        source_status = recoverable_plan_source_status(state)
        if source_status is None:
            refuse("run is not in an exact recoverable planner state")
        claimed = store.transition_if(
            source_status, state.get("updated_at"), status=source_status,
            plan_recovery={"status": "persisting", "replacement_raw_sha256": replacement_sha},
        )
        if claimed is None:
            refuse("run state changed while the plan recovery was validated")
        try:
            raw_path = run_dir / "planner.raw.md"
            previous_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None
            archived = _archive_attempt(run_dir, names=_RECOVERY_ATTEMPT_ARTIFACTS)
            if (run_dir / "steps").is_dir():
                # Only planning-time contracts can be there (eligibility):
                # retire them with the plan they belong to.
                if archived is None:
                    archived = _archive_attempt_target(run_dir)
                os.replace(run_dir / "steps", archived / "steps")
            persist_recovered_plan_artifacts(run_dir, plan)
            # The recovered plan becomes approval authority here, so the check
            # authority it selects must be frozen here too -- exactly as the
            # planner path does before its own approval gate.  Without it the
            # operator would be offered the gate while ``check_authority.json``
            # does not exist yet, and the write-once decision published in that
            # window could never bind the ``checks_sha256`` the run later
            # expects.  The whole trusted catalogue is frozen, not just this
            # selection, so a correction plan still runs approved argv.
            selected_checks = config.select_checks(plan.required_checks)
            write_check_authority(
                run_dir, tuple(config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            validate_implementation_bundle(run_dir, expected_step_ids=[step.id for step in plan.steps])
            identity = compute_plan_identity_from_run(run_dir)
            if identity.raw_sha256 != replacement_sha or identity.execution_sha256 is not None:
                raise ApprovalError("recovered plan identity does not match the replacement")
            record = write_plan_recovery_record(
                run_dir, previous_raw_sha256=previous_sha,
                replacement_raw_sha256=replacement_sha,
                archived_attempt=archived.relative_to(run_dir).as_posix() if archived else None,
            )
            self.write_checkpoint(
                run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha, tree=base_tree,
                plan_identity=identity,
            )
        except Exception as exc:
            store.update(
                status=source_status,
                plan_recovery={"status": "failed", "replacement_raw_sha256": replacement_sha,
                               "detail": redact(str(exc), self._secrets_or_empty())},
            )
            raise
        planner_state = state.get("planner") if isinstance(state.get("planner"), dict) else {}
        store.update(
            status=RunStatus.FAILED,
            plan_identity=asdict(identity),
            plan_recovery={**record, "status": "awaiting_approval"},
            planner={
                **planner_state,
                "decision": plan.decision.value,
                "title": plan.title,
                "source": PLAN_SOURCE_OPERATOR,
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
                "required_checks": list(plan.required_checks),
                "steps": [
                    {"id": step.id, "title": step.title,
                     "execution_class": step.execution_class.value,
                     "recommended_profile": self.config.routing.profile_for(step.execution_class),
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": self.run_options.final_reviewer_profile,
            },
        )
    def _secrets_or_empty(self) -> tuple[str, ...]:
        return tuple(self.secrets or ())
    def resume_pre_execution(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        state: Mapping[str, Any],
        checkpoint: ResumeCheckpoint,
        record: Mapping[str, Any],
        *,
        on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Resume context/planning/approval/setup without requiring later artifacts."""

        def refuse(message: str) -> NoReturn:
            raise ResumeIntegrityError(message)

        claimed = store.transition_if(
            state.get("status", RunStatus.FAILED), state.get("updated_at"),
            status=PHASE_STATUS[checkpoint.phase], failure=None, current_step=None,
            resume={**record, "status": "running"},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            repo = git_root(self.config.repo)
            if state.get("repo") not in {None, str(repo), str(self.config.repo)}:
                refuse("the configured repository is not the run repository")
            spec = (run_dir / "spec.md").read_text(encoding="utf-8")
            base_sha = state.get("base_sha")
            if base_sha is None:
                base_sha = resolve_commit(repo, self.config.base_ref)
                base_tree = resolve_tree(repo, base_sha)
                store.update(status=RunStatus.PLANNING, repo=str(repo), base_sha=base_sha)
            else:
                if not _is_object_id(base_sha):
                    refuse("run base SHA is invalid")
                base_tree = resolve_tree(repo, base_sha)
            if checkpoint.expected_head_sha is not None and checkpoint.expected_head_sha != base_sha:
                refuse("the checkpoint base SHA changed")
            if checkpoint.expected_tree_sha is not None and checkpoint.expected_tree_sha != base_tree:
                refuse("the checkpoint base tree changed")
            reference = _read_repository_reference(run_dir)
            if (run_dir / "repository_reference.json").exists() and reference is None:
                refuse("repository reference artifact is corrupted")
            if reference is None:
                if checkpoint.phase is not ResumePhase.CONTEXT:
                    refuse("repository reference artifact is missing")
                try:
                    reference = build_repository_reference(repo, base_sha=base_sha, config=self.config.repository)
                except GitError:
                    reference = RepositoryReference(self.config.repository.remote, None, base_sha, None)
                atomic_write_text(run_dir / "repository_reference.json", json.dumps(
                    repository_reference_dict(reference), indent=2
                ) + "\n")
            if reference.base_sha != base_sha:
                refuse("the base SHA changed for this run")

            context_path = run_dir / "context.txt"
            if checkpoint.phase is ResumePhase.CONTEXT:
                context_bundle = build_context(repo, base_sha, spec, self.config.context)
                context = render_context(context_bundle)
                atomic_write_text(context_path, context)
                store.update(status=RunStatus.PLANNING, context={
                    "base_sha": context_bundle.base_sha,
                    "locator_used": context_bundle.locator_used,
                    "locator_warning": context_bundle.locator_warning,
                    "omitted": list(context_bundle.omitted),
                    "total_bytes": context_bundle.total_bytes,
                })
                self.write_checkpoint(run_dir, ResumePhase.PLANNER,
                                             head=base_sha, tree=base_tree)
                checkpoint = ResumeCheckpoint(
                    phase=ResumePhase.PLANNER, review_cycle=1,
                    expected_head_sha=base_sha, expected_tree_sha=base_tree,
                )
            context = context_path.read_text(encoding="utf-8")
            planner_profile_id = (
                state.get("execution", {}).get("planner", {}).get("profile_id")
                if isinstance(state.get("execution"), Mapping)
                and isinstance(state.get("execution", {}).get("planner"), Mapping)
                else None
            )
            if not isinstance(planner_profile_id, str):
                refuse("run planner profile is missing")
            planner_profile = profile_for_role(self.config, planner_profile_id, ExecutionRole.PLANNER)
            if checkpoint.phase is ResumePhase.PLANNER:
                planner = PlannerV2(
                    self.planner_client or chat_client(
                        build_llm_endpoint(planner_profile), self.environment, self.trace_transport
                    ),
                    repository_reference=reference, planning=self.config.planning,
                    check_catalog=self.config.check_catalog,
                    default_check_ids=self.config.default_check_ids,
                    prompt_budget_bytes=self.config.prompt_budget.planner_max_bytes,
                    repository_preconditions=RepositoryPreconditions(repo, base_tree),
                    on_event=lambda name, data: self.trace_emit(name, phase="planning", cycle=1, data=data),
                )
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
                _persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            else:
                raw = (run_dir / "planner.raw.md").read_text(encoding="utf-8")
                plan = parse_task_plan_v2(
                    raw,
                    planning=self.config.planning,
                    check_catalog=self.config.check_catalog,
                    default_check_ids=self.config.default_check_ids,
                )
                if checkpoint.plan_identity is not None:
                    try:
                        if compute_plan_identity_from_run(run_dir) != checkpoint.plan_identity:
                            refuse("plan identity no longer matches")
                    except (ApprovalError, OSError, UnicodeError) as exc:
                        refuse(f"plan artifacts are invalid: {exc}")
            if checkpoint.phase in {ResumePhase.PLAN_APPROVAL, ResumePhase.WORKTREE_SETUP}:
                try:
                    _bundle, bundle_sha = validate_implementation_bundle(
                        run_dir, expected_step_ids=[step.id for step in plan.steps]
                    )
                    identity = compute_plan_identity_from_run(run_dir)
                except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
                    refuse(f"plan artifacts are invalid: {exc}")
                if checkpoint.plan_identity is not None and identity != checkpoint.plan_identity:
                    refuse("plan identity no longer matches")
                recorded_identity = state.get("plan_identity")
                if isinstance(recorded_identity, Mapping):
                    try:
                        if identity != plan_identity_from_mapping(recorded_identity):
                            refuse("plan identity no longer matches state")
                    except ResumeCheckpointError as exc:
                        refuse(str(exc))
                if checkpoint.phase is ResumePhase.WORKTREE_SETUP:
                    try:
                        _selection, selection_sha = read_execution_selection_with_sha256(run_dir)
                        validate_execution_selection(self.config, _selection)
                    except (ExecutionSelectionError, ProfileError, OSError, UnicodeError) as exc:
                        refuse(f"execution selection is invalid: {exc}")
                    if selection_sha != checkpoint.execution_selection_sha256:
                        refuse("execution selection hash changed")
                    if self.config.approval.require_plan_approval:
                        try:
                            approval = read_plan_approval(run_dir, expected_identity=identity)
                        except ApprovalError as exc:
                            refuse(f"approval artifact is invalid: {exc}")
                        if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                            refuse("approval was changed or is no longer APPROVE")
            existing_info = self._existing_setup_worktree(
                repo, run_dir, run_id, plan, base_sha, self.config.worktrees_root,
                self.config.base_ref,
            ) if checkpoint.phase is ResumePhase.WORKTREE_SETUP else None
            if existing_info is not None:
                _archive_attempt_tree(run_dir / "setup")
            prepared = self._prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                reference, planner_profile, existing_plan=plan,
                existing_info=existing_info,
            )
            if isinstance(prepared, RunResult):
                return prepared
            return self.execute_v2(
                store, run_dir, run_id, spec, repo, base_sha, context, reference,
                prepared=prepared,
            )
        except ResumeRequiresOperatorError as exc:
            failed = store.record_failure(exc.code, redact(str(exc), self.secrets),
                                          **self.closing_step_fields(store, "failed"))
            return RunResult(run_dir, RunStatus.FAILED, failed)
        except KeyboardInterrupt:
            interrupted = store.update(status=RunStatus.INTERRUPTED,
                                       failure={"reason": "INTERRUPTED"},
                                       **self.closing_step_fields(store, "interrupted"))
            return RunResult(run_dir, RunStatus.INTERRUPTED, interrupted)
        except Exception as exc:
            return self.project_exception(store, run_dir, exc)
    @staticmethod
    def _existing_setup_worktree(
        repo: Path, run_dir: Path, run_id: str, plan: TaskPlanV2, base_sha: str,
        worktrees_root: Path, base_ref: str,
    ) -> WorktreeInfo | None:
        """Return only an exactly recognizable partial setup; never delete/repair it."""

        expected_branch = build_run_branch(plan.title, run_id)
        raw_path = None
        try:
            raw_state = _read_json_artifact(run_dir / "state.json", 256 * 1024)
            raw_path = raw_state.get("worktree") if isinstance(raw_state, dict) else None
            recorded_branch = raw_state.get("branch") if isinstance(raw_state, dict) else None
        except Exception:
            recorded_branch = None
        path = Path(raw_path).expanduser().resolve() if isinstance(raw_path, str) else (
            Path(worktrees_root).expanduser().resolve() / run_id
        )
        if not path.exists():
            if branch_exists(repo, expected_branch):
                raise ResumeIntegrityError("expected run branch exists without its worktree")
            return None
        if recorded_branch not in {None, expected_branch}:
            raise ResumeIntegrityError("partial worktree has an unexpected branch")
        try:
            if not path.is_dir() or str(path) not in registered_worktrees(repo):
                raise ResumeIntegrityError("partial worktree is not registered")
            if not branch_exists(repo, expected_branch) or symbolic_head(path) != f"refs/heads/{expected_branch}":
                raise ResumeIntegrityError("partial worktree branch is not the expected run branch")
            if current_head(path) != base_sha or candidate_tree_sha(path) != resolve_tree(repo, base_sha):
                raise ResumeIntegrityError("partial worktree is not exactly at the run base")
            if status_porcelain(path):
                raise ResumeRequiresOperatorError("partial worktree is not clean")
        except GitError as exc:
            raise ResumeIntegrityError(f"partial worktree is unreadable: {exc}") from exc
        return WorktreeInfo(repo, path, expected_branch, base_ref, base_sha)

    def run_pipeline(
        self, store: RunStateStore, pipeline: PipelineV2Context,
        start: ResumeCheckpoint, engine: PipelineV2Coordinator, *, resumed: bool,
    ) -> RunResult:
        """Run the generic coordinator and project a failure that left it."""

        self.last_selection = pipeline.selection
        try:
            return engine.run(start, resumed=resumed)
        except PipelineFailure as failure:
            reason = normalize_exit_reason(failure.reason)
            if reason != failure.reason:
                failure.reason = reason
                failure.detail = "external executor authorization is required"
            checkpoint_phase = self.checkpoint_phase(pipeline.run_dir, default=start.phase)
            decision, terminal = project_exit(
                failure.reason, phase=checkpoint_phase,
                remote_required=failure.reason == "PUSH_FAILED",
            )
            state = store.load()
            tree = state.get("staged_tree_sha")
            if not isinstance(tree, str):
                tree = _safe_candidate_tree(pipeline.info.worktree)
            cycle = state.get("cycle") if isinstance(state.get("cycle"), int) else None
            phase = state.get("status") if isinstance(state.get("status"), str) else "run"
            recovery = self.recovery(store)
            initial = recovery.stop(
                failure.reason, phase=phase, cycle=cycle, step_id=failure.step_id,
                tree_before=tree, tree_after=_safe_candidate_tree(pipeline.info.worktree),
            )
            if (
                initial.disposition is not RecoveryDisposition.HARD_STOP
                or terminal.status is not RunStatus.FAILED
            ):
                recovery.trace(
                    "recovery.exhausted", reason=failure.reason, decision=decision,
                    attempt=1, tree_before=tree,
                    tree_after=_safe_candidate_tree(pipeline.info.worktree),
                    budget_remaining=0, phase=phase, cycle=cycle,
                    step_id=failure.step_id, terminal_status=terminal.status,
                    checkpoint_phase=checkpoint_phase,
                )
            return self.v2_failed(
                store, pipeline.run_dir, failure.reason, failure.step_id, failure.detail,
                terminal_status=terminal.status,
                auto_resumable=terminal.resumable,
            )
        except StepExecutionFailure as failure:
            return self.step_failed(store, pipeline.run_dir, failure)
        except ScopeApprovalRequired:
            return RunResult(pipeline.run_dir, RunStatus.WAITING_SCOPE_APPROVAL, store.load())
        except (ResumeIntegrityError, ResumeRequiresOperatorError):
            raise
        except Exception as exc:
            return self.project_exception(
                store, pipeline.run_dir, exc, operation="pipeline_coordinator",
            )

    # -- context ------------------------------------------------------------

    def pipeline_context(
        self, *, run_dir: Path, run_id: str, spec: str, context: str, repo: Path,
        info: WorktreeInfo, base_sha: str, base_tree_sha: str,
        repository_reference: RepositoryReference, plan: TaskPlanV2,
        bundle: Mapping[str, Any], selection: ExecutionSelection,
    ) -> PipelineV2Context:
        """The immutable facts of this run, as the coordinator reads them."""

        return PipelineV2Context(
            run_dir=run_dir, run_id=run_id, spec=spec, context=context, repo=repo,
            base_sha=base_sha, base_tree_sha=base_tree_sha,
            repository_reference=repository_reference, info=info, plan=plan,
            bundle=bundle, selection=selection, options=self.run_options,
        )
