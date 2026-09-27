"""The composition of one run: context, operations and cycle authority.

``RunComposition`` builds the immutable :class:`PipelineV2Context`, wires every
:class:`PipelineV2Operations` entry to the service that owns it, loads the plan
authority of the initial cycle and composes the effective mutable scope the
cycle is allowed to touch.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from ..agent.execution import ExecutorRuntimeConfig, executor_for_profile
from ..gitops import (
    RepositoryReference,
    WorktreeInfo,
    candidate_tree_sha,
    current_head,
)
from ..models import (
    CycleKind, ExecutionRole, ExecutionSelection, RunCycle, TaskPlanV2,
)
from ..profiles import profile_for_role
from ..result import RunResult
from ..state import RunStateStore
from .candidate import CandidateLifecycle
from .check_failure import hard_integrity_failures
from .gate_acceptance import GateAcceptanceService
from .pipeline_v2 import (
    CyclePlan, PipelineV2Context, PipelineV2Operations,
    gate_dir,
    step_dir as cycle_step_dir,
)
from .durable_readers import (
    completed_step_records, load_evidence, read_candidate_record,
    gate_mutable_authority, settled_step_status,
)
from .run_bootstrap import PreparedV2Run
from .shared import (
    CycleArtifactService,
    OrchestrationError,
)
from .step_authority import approved_step_contract

if TYPE_CHECKING:
    from .runtime import RunRuntime

class RunComposition:
    """The composition root of one run: context, operations and authority."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

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
        """Bind the one implementation, audit and acceptance path."""

        bind = functools.partial
        candidate_lifecycle = CandidateLifecycle(
            staging_remote=self.runtime.config.repository.remote,
            authorize_tree=self.runtime.publication.authorize_candidate_tree,
            gate_mutable_authority=lambda ctx, plan, stage: gate_mutable_authority(
                ctx.run_dir, plan.cycle.number, stage,
                base_paths=self.effective_cycle_scope(ctx, plan),
            ),
            push_tree=self.runtime.publication.push_candidate,
            cycle_update=self.runtime.cycle_update,
        )
        gate_acceptance = GateAcceptanceService(
            authorize_candidate_tree=self.runtime.publication.authorize_candidate_tree,
            trace_emit=self.runtime.observability.trace_emit,
        )
        cycle_artifacts = CycleArtifactService(
            cycle_update=self.runtime.cycle_update,
            trace_emit=self.runtime.observability.trace_emit,
            set_trace_cycle=lambda number: setattr(self.runtime, "trace_cycle", number),
        )
        return PipelineV2Operations(
            checkpoint=lambda ctx, phase, **fields: self.runtime.write_checkpoint(
                ctx.run_dir, phase, **fields,
            ),
            current_head=lambda ctx: current_head(ctx.info.worktree),
            candidate_tree=lambda ctx: candidate_tree_sha(ctx.info.worktree),
            begin_cycle=bind(cycle_artifacts.begin, store),
            initial_plan=self._initial_cycle_plan,
            completed_steps=self.completed_steps,
            execute_step=bind(self.runtime.step_execution.execute_cycle_step, store),
            accept_step=bind(self.runtime.step_acceptance.resume_step_acceptance, store),
            run_gate=bind(self.runtime.gates.run_gate, store),
            load_gate_evidence=lambda ctx, number, stage: load_evidence(
                gate_dir(ctx.run_dir, number, stage),
            ),
            run_audit=bind(self.runtime.audit.run, store),
            accept_gate_state=lambda ctx, plan, stage, evidence: gate_acceptance.accept(
                store, ctx, plan, stage, evidence,
                base_paths=self.effective_cycle_scope(ctx, plan),
            ),
            hard_failures=hard_integrity_failures,
            create_candidate=bind(candidate_lifecycle.create, store),
            load_candidate=lambda ctx, number: read_candidate_record(ctx.run_dir, number),
            push_candidate=bind(candidate_lifecycle.push, store),
            publish=bind(self.runtime.publication.publish_candidate, store),
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

    def completed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        return completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number, [step.id for step in cycle_plan.plan.steps],
        )

    def effective_cycle_scope(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> tuple[str, ...]:
        """The approved envelope plus the effective authority of every step.

        A contract repair contributes only through its step's effective
        authority (hash- and chain-verified).  Nothing widens the scope merely
        because a repair exists.
        """

        scope = set(cycle_plan.mutable_scope)
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
        # A step whose worker reached one ordinary path beyond its declared
        # sets had that exact path durably recorded as an audit signal; the
        # boundary of this cycle admits it so the accepted diff can be
        # committed.  No step contract and no later step scope changes.
        for record in completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number,
            [step.id for step in cycle_plan.plan.steps],
        ):
            scope.update(record.get("out_of_scope_paths") or ())
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
            else:
                settled = settled_step_status(
                    cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id), step.id,
                )
                if settled is not None:
                    row["status"] = settled.casefold()
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
