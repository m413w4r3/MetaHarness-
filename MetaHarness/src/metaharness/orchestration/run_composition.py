"""The composition of one run: context, operations and cycle authority.

``RunComposition`` builds the immutable :class:`PipelineV2Context`, wires every
:class:`PipelineV2Operations` entry to the service that owns it, loads the plan
authority of each durable cycle and composes the effective mutable scope a
cycle is allowed to touch.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from ..agent.base import AgentRunRequest
from ..agent.execution import ExecutorRuntimeConfig, executor_for_profile
from ..execution_selection import (
    ExecutionSelectionError, ensure_cycle_execution_selection,
    read_cycle_execution_selection,
    resolve_cycle_execution_selection,
    validate_cycle_execution_selection,
)
from ..gitops import RepositoryReference, WorktreeInfo, candidate_tree_sha, current_head
from ..llm.chat import LLMError
from ..models import CycleKind, ExecutionRole, ExecutionSelection, RunCycle, is_replan_cycle
from ..planning.check_replan import check_replan_dir
from ..planning.protocol import TaskPlanV2
from ..profiles import build_llm_endpoint, profile_for_role, profiles_for_config
from ..recommendation import ExecutionRecommender, RecommendationError, write_recommendation_error
from ..redaction import redact
from ..result import RunResult
from ..resume import ResumeIntegrityError
from ..review import Reviewer
from ..state import RunStateStore
from .candidate import CandidateLifecycle
from .check_failure import (
    hard_integrity_failures,
    soft_check_failures,
)
from .check_scope import gate_mutable_authority
from .gate_acceptance import GateAcceptanceService
from .gate_recovery import CheckRepairLadder
from .pipeline_v2 import (
    CyclePlan, PipelineFailure, PipelineV2Context, PipelineV2Operations,
    correction_dir, gate_dir,
    step_dir as cycle_step_dir,
)
from .resume_validation import (
    completed_step_records, load_correction_plan, load_evidence,
    load_revision,
    read_candidate_record, read_cycle_record, semantic_revision_scope,
    verify_correction_scope,
)
from .run_bootstrap import PreparedV2Run
from .shared import CycleArtifactService, OrchestrationError, bounded_parse_detail, chat_client
from .step_authority import approved_step_contract

if TYPE_CHECKING:
    from .runtime import RunRuntime

class RunComposition:
    """The composition root of one run: context, operations and authority."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def reviewer_for_profile(self, profile_id: str) -> Reviewer:
        profile = profile_for_role(self.runtime.config, profile_id, ExecutionRole.REVIEWER)
        client = self.runtime.reviewer_client
        if client is None:
            client = chat_client(
                build_llm_endpoint(profile), self.runtime.environment, self.runtime.observability.trace_transport
            )
        return Reviewer(client, allow_format_repair=True)

    def _recommender_for_profile(self, profile_id: str) -> ExecutionRecommender:
        profile = profile_for_role(self.runtime.config, profile_id, ExecutionRole.PLANNER)
        client = self.runtime.recommender_client
        if client is None:
            # This is deliberately a new client: the recommender has no
            # planner conversation/history, while using the same profile
            # endpoint and transport policy.
            client = chat_client(
                build_llm_endpoint(profile), self.runtime.environment, self.runtime.observability.trace_transport
            )
        return ExecutionRecommender(client)

    def _maybe_recommend_profiles(
        self,
        store: RunStateStore,
        run_dir: Path,
        planner_profile_id: str,
    ) -> None:
        if not self.runtime.config.ui.enable_profile_recommendation:
            return
        profiles = profiles_for_config(self.runtime.config)
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
            message = redact(" ".join(str(exc).split()), self.runtime.secrets)
            warning = f"{type(exc).__name__}: {message}"[:1000]
            try:
                write_recommendation_error(run_dir, warning)
            except (OSError, UnicodeError):
                pass
            store.update_metadata(
                recommendation={"status": "FAILED", "warning": warning},
            )
            return
        store.update_metadata(
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

        profile = profile_for_role(self.runtime.config, profile_id, role)
        runtime = ExecutorRuntimeConfig(
            config=self.runtime.config,
            environment=self.runtime.environment,
            codex_home=self.runtime.config.codex_runtime.home,
            claude_home=self.runtime.config.claude_runtime.home,
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

        profile = profile_for_role(self.runtime.config, profile_id, role)
        executor = self.executor_for_profile(profile.id, role)
        return executor.run(
            AgentRunRequest(
                role=role,
                profile_id=profile.id,
                prompt=redact(prompt, self.runtime.secrets),
                worktree=Path(worktree),
                artifact_dir=Path(artifact_dir),
                mutable_paths=mutable_paths,
                prompt_mode="revision",
            )
        )

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
                self.runtime.config, store.load()["execution"]["planner"]["profile_id"],
                ExecutionRole.PLANNER,
            )
            prepared = self.runtime.bootstrap.prepare_v2_run(
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
            options=self.runtime.run_options,
        )
        return pipeline, prepared.checkpoint

    def pipeline_operations(self, store: RunStateStore) -> PipelineV2Operations:
        """Bind every operation the coordinator sequences to this run."""

        bind = functools.partial
        candidate_lifecycle = CandidateLifecycle(
            staging_remote=self.runtime.config.repository.remote,
            authorize_tree=self.runtime.publication.authorize_candidate_tree,
            gate_mutable_authority=lambda ctx, cycle_plan, stage: gate_mutable_authority(
                ctx.run_dir, cycle_plan.cycle.number, stage,
                base_paths=self.effective_cycle_scope(ctx, cycle_plan),
                policy_config=self.runtime.repair_scope,
                require_attempt_records=True,
            ),
            push_tree=self.runtime.publication.push_candidate,
            cycle_update=self.runtime.cycle_update,
        )
        gate_acceptance = GateAcceptanceService(
            secrets=self.runtime.secrets,
            repair_scope_policy=self.runtime.repair_scope,
            authorize_candidate_tree=self.runtime.publication.authorize_candidate_tree,
            check_repair_attempts=self.runtime.gates.check_repair_attempt_records,
            load_revision=load_revision,
            trace_emit=self.runtime.observability.trace_emit,
            bounded_detail=bounded_parse_detail,
        )
        cycle_artifacts = CycleArtifactService(
            cycle_update=self.runtime.cycle_update,
            trace_emit=self.runtime.observability.trace_emit,
            set_trace_cycle=lambda number: setattr(self.runtime, "trace_cycle", number),
        )
        return PipelineV2Operations(
            checkpoint=lambda ctx, phase, **fields: self.runtime.write_checkpoint(
                ctx.run_dir, phase, **fields
            ),
            current_head=lambda ctx: current_head(ctx.info.worktree),
            candidate_tree=lambda ctx: candidate_tree_sha(ctx.info.worktree),
            begin_cycle=bind(cycle_artifacts.begin, store),
            load_cycle=lambda ctx, number: read_cycle_record(ctx.run_dir, number),
            initial_plan=self._initial_cycle_plan,
            review_implementation_correction=self.runtime.reviews.review_implementation_correction,
            plan_correction=bind(self.runtime.reviews.plan_correction, store),
            load_correction=self._load_correction,
            completed_steps=self.completed_steps,
            execute_step=bind(self.runtime.step_execution.execute_cycle_step, store),
            accept_step=bind(self.runtime.step_acceptance.resume_step_acceptance, store),
            semantic_revision=bind(self.runtime.reviews.semantic_revision, store),
            semantic_review_correction=bind(self.runtime.reviews.semantic_review_correction, store),
            run_gate=bind(self.runtime.gates.run_gate, store),
            load_gate_evidence=lambda ctx, number, stage: load_evidence(
                gate_dir(ctx.run_dir, number, stage)
            ),
            load_accepted_gate_evidence=self.runtime.gates.load_accepted_gate_evidence,
            accept_gate_state=lambda ctx, cycle_plan, stage, evidence: gate_acceptance.accept(
                store, ctx, cycle_plan, stage, evidence,
                base_paths=self.effective_cycle_scope(ctx, cycle_plan),
            ),
            check_repair_attempts=lambda ctx, number, stage: self.runtime.gates.check_repair_attempt_records(
                ctx.run_dir, number, stage
            ),
            check_repair_attempt=bind(self.runtime.gates.run_check_repair_attempt, store),
            hard_failures=hard_integrity_failures,
            soft_failures=soft_check_failures,
            create_candidate=bind(candidate_lifecycle.create, store),
            load_candidate=lambda ctx, number: read_candidate_record(ctx.run_dir, number),
            push_candidate=bind(candidate_lifecycle.push, store),
            review_candidate=bind(self.runtime.reviews.review_candidate, store),
            record_review=bind(self.runtime.reviews.record_review, store),
            request_human=bind(self.runtime.reviews.request_human, store),
            review_repair_exhausted=bind(self.runtime.reviews.review_repair_exhausted, store),
            publish=bind(self.runtime.publication.publish_candidate, store),
            recovery_operations=CheckRepairLadder(
                replan_steps=functools.partial(self.runtime.gates.replan_responsible_step, store),
                replan_cycles=functools.partial(self.runtime.reviews.replan_cycle, store),
            ),
        )

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
            self.runtime.config, ctx.selection, ctx.run_dir, number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        return self.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )

    def load_plan_correction(self, ctx: PipelineV2Context, cycle: RunCycle) -> CyclePlan:
        """Read back the durable plan authority of one correction cycle.

        A check-replan cycle never plans at its own boundary: the red gate's
        durable transaction already produced this decomposition, so opening the
        cycle reloads and verifies it exactly as a resume does.
        """

        plan, bundle, bundle_sha = load_correction_plan(
            self.runtime.config, ctx.selection, ctx.run_dir, cycle.number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha, self.runtime.repair_scope)
        return self.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )

    def correction_cycle_plan(
        self, ctx: PipelineV2Context, cycle: RunCycle, plan: TaskPlanV2,
        bundle: Mapping[str, Any], bundle_sha: str, *, creating: bool,
    ) -> CyclePlan:
        if not is_replan_cycle(cycle.kind):
            raise PipelineFailure("REPLAN_CYCLE_REQUIRED")
        # A red-gate replan authored its plan in the executing cycle's own
        # check-replan directory; a review replan in its correction one.
        contracts_dir = (
            check_replan_dir(ctx.run_dir, cycle.number)
            if cycle.kind is CycleKind.CHECK_REPLAN else correction_dir(ctx.run_dir, cycle)
        )
        planned_profile_ids = {
            step.id: self.runtime.config.routing.profile_for(step.execution_class) for step in plan.steps
        }
        try:
            if creating:
                cycle_selection = ensure_cycle_execution_selection(
                    ctx.run_dir,
                    resolve_cycle_execution_selection(
                        self.runtime.config, cycle=cycle.number,
                        plan_steps=plan.steps,
                        step_profile_ids=planned_profile_ids,
                        fallback_authority=self.runtime.run_options.recovery.execution_fallbacks,
                    ),
                )
            else:
                cycle_selection = read_cycle_execution_selection(ctx.run_dir, cycle.number)
                validate_cycle_execution_selection(self.runtime.config, cycle_selection)
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
            cycle=cycle, plan=plan, bundle=bundle, contracts_dir=contracts_dir,
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
            self.runtime.config, ctx.selection, ctx.run_dir, cycle.number,
            inherited_check_ids=ctx.plan.required_checks,
        )
        if expected_sha is None or bundle_sha != expected_sha:
            raise ResumeIntegrityError(f"cycle {cycle.number:03d} correction plan changed")
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha, self.runtime.repair_scope)
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
            authority = self.runtime.contract_recovery.resolve_step_authority(
                artifact_dir, step, approved_step_contract(cycle_plan, step),
                expected_tree=None, expected_plan_step_count=count,
            )
            scope.update(authority.mutable_scope)
        scope.update(semantic_revision_scope(
            ctx.repo, ctx.run_dir, cycle_plan.cycle.number,
            self.runtime.repair_scope,
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
                    status="completed",
                    no_change=record.get("no_change", False),
                    usage=record["usage"],
                    input_tokens=record["usage"]["input_tokens"],
                    output_tokens=record["usage"]["output_tokens"],
                )
            elif step.id == running:
                row["status"] = "running"
            rows.append(row)
        return rows

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
            bundle=bundle, selection=selection, options=self.runtime.run_options,
        )
