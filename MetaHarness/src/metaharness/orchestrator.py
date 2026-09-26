"""The MetaHarness façade: prepare one run and drive its generic pipeline.

The façade owns construction, the run entry points (``run_text``, ``resume``,
``recover_plan``) and the delegation of the sequencing to
:class:`~metaharness.orchestration.pipeline_v2.PipelineV2Coordinator`.
Everything the coordinator sequences lives in
:mod:`metaharness.orchestration.runtime` and the services it composes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Callable,
)
from .config import load_config
from .context import (
    build_context,
    render_context,
)
from .gitops import (
    GitError,
    assert_clean,
    git_root,
    build_repository_reference,
    resolve_commit,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
)
from .integrations.github import GitHubWorkstreamClient
from .models import (
    ExecutionRole,
    HarnessConfig,
    INTERRUPTED_REASON,
    RunDisposition,
    RunEvent,
    RunMachineState,
)
from .resume import (
    CHECKPOINT_INTEGRITY_OPERATION,
    CHECK_REPAIR_INTEGRITY_OPERATION,
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeError,
    ResumeIntegrityError,
    ResumeNotAllowedError,
    ResumePhase,
    ResumeRequiresOperatorError,
    read_checkpoint,
    pipeline_version_from_state,
    resume_info,
    resume_label,
    run_identity,
    write_checkpoint,
)
from .redaction import (
    config_secret_values,
    redact,
)
from .profiles import profile_for_role
from .trace import TraceSink
from .result import (
    RunResult,
    atomic_write_text,
)
from .state import RunStateStore
from .run_options import (
    RunOptions,
    RunOptionsError,
    effective_repair_scope_policy,
    effective_run_config,
    read_run_options_for_state,
    write_run_options,
)
from .orchestration.shared import (
    CommitBoundaryError,
    OrchestrationError,
)
from .orchestration.pipeline_v2 import PipelineV2Context
from .orchestration.resume_integrity import validate_resume
from .orchestration.runtime import (
    RunRuntime,
    generate_run_id,
    safe_run_id,
)


class Orchestrator:
    """The façade of one pipeline-v2 run.

    It constructs the run's runtime, freezes its options and execution
    selection, then delegates every execution phase to the generic engine:
    :class:`~metaharness.orchestration.pipeline_v2.PipelineV2Coordinator`.
    """

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
        self._runtime = RunRuntime(
            config,
            planner_client=planner_client,
            reviewer_client=reviewer_client,
            recommender_client=recommender_client,
            github_client=github_client,
            trace_sink=trace_sink,
        )

    def run(
        self, spec: str | Path, *, run_id: str | None = None,
        planner_profile: str | None = None,
        run_options: RunOptions | None = None,
    ) -> RunResult:
        """Read one SPEC file and delegate execution to :meth:`run_text`."""

        spec_path = Path(spec).expanduser().resolve()
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise OrchestrationError(f"could not read spec {spec_path}: {exc}") from exc
        kwargs: dict[str, Any] = {"run_id": run_id}
        if planner_profile is not None:
            kwargs["planner_profile"] = planner_profile
        if run_options is not None:
            kwargs["run_options"] = run_options
        return self.run_text(spec_content, **kwargs)

    def run_text(
        self,
        spec_content: str,
        *,
        run_id: str | None = None,
        planner_profile: str | None = None,
        run_options: RunOptions | None = None,
        on_created: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Run one in-memory SPEC and return its durable final state.

        A worktree is deliberately never removed.  This keeps failed and
        interrupted runs inspectable and makes the run directory the handoff
        point for operators.
        """

        if not isinstance(spec_content, str):
            raise OrchestrationError("spec must be a string")
        if not spec_content.strip():
            raise OrchestrationError("spec must not be empty")

        original_config = self._runtime.config
        try:
            if run_options is None:
                overrides = {"planner_profile": planner_profile} if planner_profile is not None else {}
                run_options = RunOptions.from_config(original_config, **overrides)
            elif planner_profile is not None and planner_profile != run_options.planner_profile:
                raise OrchestrationError("planner profile conflicts with run options")
            run_options.validate_profiles(original_config)
            selected_planner = profile_for_role(
                original_config, run_options.planner_profile, ExecutionRole.PLANNER
            )
        except RunOptionsError as exc:
            raise OrchestrationError(str(exc)) from exc

        selected_run_id = safe_run_id(run_id) if run_id is not None else generate_run_id()
        run_dir = (self._runtime.config.runs_root / selected_run_id).expanduser().resolve()
        if run_dir.exists():
            raise OrchestrationError(f"run directory already exists: {run_dir}")
        self._runtime.secrets = config_secret_values(
            self._runtime.config, self._runtime.environment
        )

        store: RunStateStore | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            # The SPEC copy is created before state initialization, as the
            # CREATED phase contract requires both to exist together.
            (run_dir / "spec.md").write_text(spec_content, encoding="utf-8")
            options_sha256 = write_run_options(run_dir, run_options)
            store = RunStateStore(run_dir / "state.json")
            store.initialize(selected_run_id, pipeline_version=2)
            self._runtime.observability.begin_trace(run_dir, selected_run_id, created=True)
            # All downstream methods use this frozen per-run view.  The
            # caller's HarnessConfig object is never mutated.
            self._runtime.run_options = run_options
            self._runtime.repair_scope = effective_repair_scope_policy(run_options)
            self._runtime.config = effective_run_config(original_config, run_options)
            store.update_metadata(
                spec_path="spec.md",
                repo=str(self._runtime.config.repo),
                base_ref=self._runtime.config.base_ref,
                run_options_sha256=options_sha256,
                run_options=run_options.to_dict(),
                execution={
                    "planner": {
                        "profile_id": selected_planner.id,
                        "model": selected_planner.model,
                        "selection_mode": selected_planner.selection_mode.value,
                    }
                },
            )
            if on_created is not None:
                on_created(run_dir)
            # The first durable boundary, declared once creation is announced:
            # the run is CREATED until it exists, then CONTEXT owns the phase
            # it will resume.  It intentionally carries no Git/plan identity
            # yet, since context and repository discovery are themselves
            # resumable operations.
            write_checkpoint(run_dir, ResumeCheckpoint(phase=ResumePhase.CONTEXT))
            return self._runtime.observability.diagnose_result(
                self._execute(store, run_dir, selected_run_id, spec_content)
            )
        except KeyboardInterrupt:
            if store is None:
                raise
            state = store.set_run_state(
                RunMachineState(
                    disposition=RunDisposition.FAILED, reason=INTERRUPTED_REASON,
                ),
                failure={"reason": INTERRUPTED_REASON},
                **self._runtime.failure.closing_step_fields(store, "interrupted"),
            )
            return self._runtime.observability.diagnose_result(RunResult.of(run_dir, state))
        except Exception as exc:
            if store is None:
                raise
            return self._runtime.observability.diagnose_result(self._runtime.failure.project_exception(store, run_dir, exc))

    def _execute(
        self,
        store: RunStateStore,
        run_dir: Path,
        run_id: str,
        spec: str,
    ) -> RunResult:
        repo = git_root(self._runtime.config.repo)
        state = store.load()
        planner_profile_id = state.get("execution", {}).get("planner", {}).get("profile_id")
        if not isinstance(planner_profile_id, str):
            raise OrchestrationError("run planner profile is missing")
        if self._runtime.config.require_clean_base:
            assert_clean(repo)
        base_sha = resolve_commit(repo, self._runtime.config.base_ref)
        base_tree_sha = resolve_tree(repo, base_sha)
        store.update_metadata(
            repo=str(repo), base_sha=base_sha,
            planning_protocol="v2",
        )
        try:
            repository_reference = build_repository_reference(
                repo, base_sha=base_sha, config=self._runtime.config.repository
            )
        except GitError:
            repository_reference = RepositoryReference(
                self._runtime.config.repository.remote, None, base_sha, None
            )
        atomic_write_text(
            run_dir / "repository_reference.json",
            json.dumps(repository_reference_dict(repository_reference), indent=2) + "\n",
        )
        context_bundle = build_context(repo, base_sha, spec, self._runtime.config.context)
        context = render_context(context_bundle)
        atomic_write_text(run_dir / "context.txt", context)
        store.update_metadata(
            context={
                "base_sha": context_bundle.base_sha,
                "locator_used": context_bundle.locator_used,
                "locator_warning": context_bundle.locator_warning,
                "omitted": list(context_bundle.omitted),
                "total_bytes": context_bundle.total_bytes,
            },
        )
        self._runtime.write_checkpoint(
            run_dir, ResumePhase.PLANNER, head=base_sha, tree=base_tree_sha
        )
        outcome = self._runtime.composition.execute_v2(
            store, run_dir, run_id, spec, repo, base_sha, context,
            repository_reference,
        )
        if isinstance(outcome, RunResult):
            return outcome
        pipeline, start = outcome
        return self._runtime.run_pipeline(store, pipeline, start, resumed=False)

    def resume(
        self, run_id: str, *, on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Resume a failed or interrupted run at its durable checkpoint.

        Never replays a successful phase.  Every persisted invariant is
        validated first; a mismatch records ``RESUME_INTEGRITY_FAILURE`` (or
        ``RESUME_REQUIRES_OPERATOR``) without any model call.  A run that is
        not resumable raises :class:`ResumeNotAllowedError` and its state is
        left untouched.
        """

        try:
            selected = safe_run_id(run_id)
        except OrchestrationError as exc:
            raise ResumeError(str(exc)) from exc
        run_dir = (self._runtime.config.runs_root / selected).expanduser().resolve()
        if not run_dir.is_dir():
            raise ResumeError("run directory does not exist")
        if not (run_dir / "state.json").is_file():
            raise ResumeError("run state does not exist")
        store = RunStateStore(run_dir / "state.json")
        try:
            state = store.load()
            pipeline_version_from_state(state)
        except (OSError, ValueError) as exc:
            raise ResumeError(f"run state is unreadable: {exc}") from exc
        try:
            options, _ = read_run_options_for_state(run_dir, state)
            self._runtime.run_options = options
            self._runtime.repair_scope = effective_repair_scope_policy(options)
            self._runtime.config = effective_run_config(self._runtime.config, options)
        except RunOptionsError as exc:
            raise ResumeNotAllowedError("run options are missing or invalid") from exc
        self._runtime.secrets = config_secret_values(self._runtime.config, self._runtime.environment)
        self._runtime.observability.begin_trace(run_dir, selected, created=False)
        eligibility = resume_info(run_dir, state)
        if eligibility.operation in {
            CHECKPOINT_INTEGRITY_OPERATION, CHECK_REPAIR_INTEGRITY_OPERATION,
        }:
            # A current checkpoint whose durable evidence no longer proves its
            # identity: fail closed, without any model call.
            failed = store.record_failure(
                "RESUME_INTEGRITY_FAILURE", eligibility.reason,
                resume={"status": "refused", "previous_status": state.get("status"),
                        "previous_failure": state.get("failure")},
                current_step=None,
            )
            return self._runtime.observability.diagnose_result(RunResult.of(run_dir, failed))
        if not eligibility.resumable:
            raise ResumeNotAllowedError(eligibility.reason or "run is not resumable")
        try:
            checkpoint = read_checkpoint(run_dir)
        except ResumeCheckpointError as exc:
            raise ResumeNotAllowedError(str(exc)) from exc
        if checkpoint is None:
            raise ResumeNotAllowedError("no resume checkpoint")
        previous = state.get("resume") if isinstance(state.get("resume"), dict) else {}
        attempts = previous.get("attempts") if isinstance(previous.get("attempts"), int) else 0
        record = {
            "phase": checkpoint.phase.value,
            "label": resume_label(checkpoint),
            "attempts": attempts + 1,
            "previous_status": state.get("status"),
            "previous_failure": state.get("failure"),
            **({"operation": eligibility.operation} if eligibility.operation else {}),
        }
        if checkpoint.phase in {
            ResumePhase.CONTEXT, ResumePhase.PLANNER,
            ResumePhase.PLAN_APPROVAL, ResumePhase.WORKTREE_SETUP,
        }:
            return self._runtime.observability.diagnose_result(self._runtime.bootstrap.resume_pre_execution(
                store, run_dir, selected, state, checkpoint, record,
                on_claimed=on_claimed,
            ))
        try:
            resumed = validate_resume(
                config=self._runtime.config, repair_scope=self._runtime.repair_scope,
                run_dir=run_dir, state=state, checkpoint=checkpoint,
                staging_remote=self._runtime.config.repository.remote,
            )
        except (ResumeIntegrityError, ResumeRequiresOperatorError) as exc:
            failed = store.record_failure(
                exc.code, redact(str(exc), self._runtime.secrets),
                resume={**record, "status": "refused"}, current_step=None,
            )
            return self._runtime.observability.diagnose_result(RunResult.of(run_dir, failed))
        claimed = store.transition_run(
            RunEvent.resume(), expected=run_identity(state, run_dir, checkpoint),
            failure=None, current_step=None,
            resume={**record, "status": "running",
                    "restored_paths": list(resumed.restore_paths)},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            if resumed.restore_paths:
                self._runtime.restore_checkpoint_tree(resumed)
            pipeline = self._runtime.composition.pipeline_context(
                run_dir=run_dir, run_id=selected, spec=resumed.spec,
                context=resumed.context, repo=resumed.info.source_repo,
                info=resumed.info, base_sha=resumed.info.base_sha,
                base_tree_sha=resumed.base_tree_sha,
                repository_reference=resumed.repository_reference,
                plan=resumed.plan, bundle=resumed.bundle, selection=resumed.selection,
            )
            return self._runtime.observability.diagnose_result(
                self._runtime.run_pipeline(store, pipeline, checkpoint, resumed=True)
            )
        except (ResumeIntegrityError, ResumeRequiresOperatorError) as exc:
            failed = store.record_failure(
                exc.code, redact(str(exc), self._runtime.secrets),
                **self._runtime.failure.closing_step_fields(store, "failed"),
            )
            return self._runtime.observability.diagnose_result(RunResult.of(run_dir, failed))
        except KeyboardInterrupt:
            interrupted = store.set_run_state(
                RunMachineState(
                    disposition=RunDisposition.FAILED, reason=INTERRUPTED_REASON,
                ),
                failure={"reason": INTERRUPTED_REASON},
                **self._runtime.failure.closing_step_fields(store, "interrupted"),
            )
            return self._runtime.observability.diagnose_result(RunResult.of(run_dir, interrupted))
        except Exception as exc:
            return self._runtime.observability.diagnose_result(self._runtime.failure.project_exception(store, run_dir, exc))

    def recover_plan(
        self, run_id: str, replacement_raw: str, *,
        on_claimed: Callable[[Path], None] | None = None,
    ) -> RunResult:
        """Replace a failed planner answer with an operator META PLAN v2.

        No model is called.  The replacement is validated exactly like a
        planner answer, published as the run's plan authority, and the
        checkpoint moves to PLAN_APPROVAL.  The run then continues through the
        normal resume workflow: plan approval, worktree setup, the first step...
        A refusal raises :class:`PlanRecoveryError` and changes nothing.
        """

        self._runtime.persist_recovered_plan(run_id, replacement_raw)
        return self.resume(run_id, on_claimed=on_claimed)


def run_orchestrator(
    config: HarnessConfig | str | Path,
    spec: str | Path,
    *,
    run_id: str | None = None,
) -> RunResult:
    """Functional entry point for CLI and embedding callers."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).run(spec, run_id=run_id)


def resume_run(config: HarnessConfig | str | Path, run_id: str) -> RunResult:
    """Resume *run_id* at its durable checkpoint (same run id, same worktree)."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).resume(run_id)


def recover_plan_run(config: HarnessConfig | str | Path, run_id: str, replacement_raw: str) -> RunResult:
    """Recover a failed planner run with an operator META PLAN v2 (no model call)."""

    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return Orchestrator(loaded).recover_plan(run_id, replacement_raw)


__all__ = [
    "CommitBoundaryError",
    "OrchestrationError",
    "Orchestrator",
    "ResumeError",
    "ResumeNotAllowedError",
    "generate_run_id",
    "recover_plan_run",
    "resume_run",
    "run_orchestrator",
]
