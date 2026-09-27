"""The composition of one run: context, operations and cycle authority.

``RunComposition`` builds the immutable :class:`PipelineV2Context`, wires every
:class:`PipelineV2Operations` entry to the service that owns it, loads the plan
authority of the initial cycle and composes the effective mutable scope the
cycle is allowed to touch.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING
from ..agent.execution import ExecutorRuntimeConfig, executor_for_profile
from ..gitops import (
    RepositoryReference,
    WorktreeInfo,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    diffstat_between_commits,
    is_ancestor,
    push_run_branch,
    resolve_tree,
    rewind_worktree,
)
from ..models import (
    CycleKind, ExecutionRole, ExecutionSelection, RunCycle, RunDisposition, RunMachineState,
    RunPhase, TaskPlanV2, DROP_UNKNOWN_REQUIRED_CHECK,
)
from ..profiles import build_llm_endpoint, profile_for_role
from ..result import RunResult, atomic_write_text
from ..state import RunStateStore
from ..context import build_context, render_context
from ..approval import compute_plan_identity_from_run
from ..execution_selection import (
    ensure_execution_selection, resolve_execution_selection, validate_execution_selection,
)
from ..validation import ValidationError, frozen_check_policy
from ..plan_repository_validation import validate_plan_repository_topology
from ..planning.artifacts import persist_iteration_plan, validate_implementation_bundle
from ..planning.continue_request import PlannerContinueFacts
from ..planning.planner_continue import ContinueDecision, PlannerContinue, PlannerContinueResult
from .candidate import CandidateLifecycle
from .check_failure import hard_integrity_failures
from .gate_acceptance import GateAcceptanceService
from .pipeline_v2 import (
    CyclePlan, IterationOutcome, PipelineFailure, PipelineV2Context, PipelineV2Operations,
    cycle_dir,
    gate_dir,
    step_dir as cycle_step_dir,
)
from .durable_readers import (
    completed_step_records, load_evidence, gate_mutable_authority, settled_step_status,
)
from ..planning.artifacts import iteration_plan_dir
from .run_bootstrap import PreparedV2Run
from .shared import (
    CycleArtifactService,
    OrchestrationError,
)

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
            iteration=prepared.checkpoint.iteration,
        )
        return pipeline, prepared.checkpoint

    def pipeline_operations(self, store: RunStateStore) -> PipelineV2Operations:
        """Bind the one implementation, audit and acceptance path."""

        bind = functools.partial
        candidate_lifecycle = CandidateLifecycle(
            staging_remote=self.runtime.config.repository.remote,
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
            checkpoint=self._checkpoint,
            current_head=lambda ctx: current_head(ctx.info.worktree),
            begin_cycle=bind(cycle_artifacts.begin, store),
            initial_plan=self._initial_cycle_plan,
            completed_steps=self.completed_steps,
            execute_step=bind(self.runtime.step_execution.execute_cycle_step, store),
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
            load_candidate=lambda ctx, number: candidate_lifecycle.load_from_git(store, ctx, number),
            push_candidate=bind(candidate_lifecycle.push, store),
            publish=bind(self.runtime.publication.publish_candidate, store),
            load_iteration_outcome=self._load_iteration_outcome,
            planner_continue=bind(self._planner_continue, store),
            prepare_next_iteration=bind(self._prepare_next_iteration, store),
            close_iteration=bind(self._close_iteration, store),
            partial=bind(self._partial, store),
            spec_decision=bind(self._spec_decision, store),
            failed_continued=self._failed_continued,
        )

    def _initial_cycle_plan(self, ctx: PipelineV2Context) -> CyclePlan:
        return CyclePlan(
            cycle=RunCycle(ctx.iteration, CycleKind.INITIAL),
            plan=ctx.plan,
            bundle=ctx.bundle,
            contracts_dir=iteration_plan_dir(ctx.run_dir, ctx.iteration),
            step_profile_ids={
                item.step_id: item.implementer.profile_id for item in ctx.selection.steps
            },
            step_fallback_profile_ids={
                item.step_id: tuple(profile.profile_id for profile in item.fallbacks)
                for item in ctx.selection.steps
            },
        )

    def _checkpoint(self, ctx: PipelineV2Context, phase: RunPhase, **fields: Any) -> None:
        plan_path = iteration_plan_dir(ctx.run_dir, ctx.iteration) / "task_plan.json"
        try:
            import hashlib
            plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "effective iteration plan is missing") from exc
        self.runtime.write_checkpoint(
            ctx.run_dir, phase, plan_sha256=plan_sha, **fields,
        )

    def _load_iteration_outcome(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> IterationOutcome:
        evidence = load_evidence(gate_dir(ctx.run_dir, cycle_plan.cycle.number, "POST_IMPLEMENTATION"))
        if evidence is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "continuation gate evidence is missing")
        reports = sorted((cycle_dir(ctx.run_dir, cycle_plan.cycle) / "audit").glob("*/report.json"))
        summary: dict[str, Any] = {}
        if reports:
            try:
                payload = json.loads(reports[-1].read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "audit report is unreadable") from exc
            summary = {
                "audit_status": payload.get("status", "NOT_RUN"),
                "audit_remaining": tuple(payload.get("remaining", ())),
                "audit_fixed": tuple(payload.get("fixed", ())),
                "audit_refactored": tuple(payload.get("refactored", ())),
                "audit_risks": tuple(payload.get("risks", ())),
            }
        return IterationOutcome(evidence, **summary)

    def _planner_continue(
        self, store: RunStateStore, ctx: PipelineV2Context,
        cycle_plan: CyclePlan, outcome: IterationOutcome,
        validate: Callable[[PlannerContinueResult], PlannerContinueResult],
    ) -> PlannerContinueResult:
        head = current_head(ctx.info.worktree)
        repository_context = render_context(build_context(
            ctx.info.worktree, head, ctx.spec, self.runtime.config.context,
        ))
        context_path = ctx.run_dir / "iterations" / f"{ctx.iteration:02d}" / "repository-context.txt"
        atomic_write_text(context_path, repository_context)
        state = store.load()
        baseline = state.get("baseline") if isinstance(state.get("baseline"), Mapping) else {}
        baseline_warnings = tuple(
            f"{item.get('id', 'check')}:{item.get('verdict')}"
            for item in baseline.get("checks", ())
            if isinstance(item, Mapping) and item.get("verdict") not in {"PASS", "IMPROVED"}
        )
        normalizations = tuple(
            " ".join(part for part in (
                item.code, item.step_id or "", item.path or "", item.detail or "",
            ) if part)
            for item in cycle_plan.plan.normalizations
        )
        prior = state.get("iteration_remaining")
        prior_remaining = tuple(item for item in prior if isinstance(item, str)) if isinstance(prior, list) else ()
        completed = state.get("completed_iterations")
        milestones = tuple(
            (item["milestone_id"], item.get("status", "CLOSED"))
            for item in completed or ()
            if isinstance(item, Mapping) and isinstance(item.get("milestone_id"), str)
        )
        failed = self._failed_continued(ctx, cycle_plan)
        tree = resolve_tree(ctx.info.worktree, head)
        facts = PlannerContinueFacts(
            spec=ctx.spec,
            plan=cycle_plan.plan,
            iteration=ctx.iteration,
            milestone_id=cycle_plan.plan.milestone_id,
            milestone_title=cycle_plan.plan.milestone_title,
            milestone_goal=cycle_plan.plan.milestone_goal,
            audit_status=outcome.audit_status,
            milestones=milestones,
            normalizations=normalizations,
            failed_steps=failed,
            audit_remaining=outcome.audit_remaining,
            audit_risks=outcome.audit_risks,
            audit_fixed=outcome.audit_fixed,
            audit_refactored=outcome.audit_refactored,
            gate_failures=outcome.evidence.failures,
            gate_warnings=outcome.evidence.warnings,
            gate_baseline_warnings=baseline_warnings,
            diffstat=diffstat_between_commits(ctx.info.worktree, ctx.base_sha, head),
            modified_paths=changed_paths_between_trees(ctx.info.worktree, ctx.base_tree_sha, tree),
            current_repository_context=repository_context,
            continuation_remaining=tuple(state.get("iteration_remaining") or ()),
            prior_iteration_remaining=prior_remaining,
        )
        planner_profile = profile_for_role(
            self.runtime.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER,
        )
        # M02+ consumes the run's frozen check policy: a TOML edited after the
        # freeze can only veto an ID, never change an argv or a default.
        policy = frozen_check_policy(
            self.runtime.config, ctx.run_dir,
            expected_sha256=self.runtime.approved_check_authority_sha256(ctx.run_dir),
        )
        if policy is None:
            raise ValidationError("the run has no check authority")
        service = PlannerContinue(
            client=self.runtime.planner_client or self.runtime.chat(build_llm_endpoint(planner_profile)),
            planning=self.runtime.config.planning,
            check_catalog=policy.checks,
            default_check_ids=policy.default_check_ids,
            check_authority_sha256=policy.sha256,
            prompt_budget_bytes=self.runtime.config.prompt_budget.planner_max_bytes,
        )
        return service.decide(
            facts, iterations_dir=ctx.run_dir / "iterations", validate=validate,
        )

    def _prepare_next_iteration(
        self, store: RunStateStore, ctx: PipelineV2Context,
        decision: PlannerContinueResult,
    ) -> PipelineV2Context:
        if decision.next_plan is None or decision.next_milestone is None:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", "NEXT has no complete plan")
        iteration = ctx.iteration + 1
        head = current_head(ctx.info.worktree)
        start_tree = resolve_tree(ctx.info.worktree, head)
        plan = validate_plan_repository_topology(ctx.info.worktree, start_tree, decision.next_plan)
        if any(item.code == DROP_UNKNOWN_REQUIRED_CHECK for item in plan.normalizations):
            # Unknown checks are a parser normalization for bootstrapping; a
            # continuation authority must send a fully trusted selection.
            unknown = [item.detail for item in plan.normalizations if item.code == DROP_UNKNOWN_REQUIRED_CHECK]
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", {"untrusted_checks": unknown})
        plan_sha = persist_iteration_plan(ctx.run_dir, iteration, plan)
        bundle, _bundle_sha = validate_implementation_bundle(
            iteration_plan_dir(ctx.run_dir, iteration), expected_step_ids=[step.id for step in plan.steps],
        )
        selection = resolve_execution_selection(
            self.runtime.config,
            planner_profile_id=ctx.selection.planner.profile_id,
            plan_steps=plan.steps,
            audit_profile_id=ctx.selection.audit.profile_id,
            fallback_authority=self.runtime.run_options.recovery.execution_fallbacks,
        )
        selection = ensure_execution_selection(ctx.run_dir, selection, iteration=iteration)
        validate_execution_selection(self.runtime.config, selection)
        identity = compute_plan_identity_from_run(ctx.run_dir, iteration=iteration)
        store.update_metadata(
            iteration=iteration,
            current_milestone={"id": plan.milestone_id, "title": plan.milestone_title},
            planner={
                **dict(store.load().get("planner") or {}),
                "decision": plan.decision.value, "title": plan.title,
                "required_checks": list(plan.required_checks),
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
            },
            plan_identity=asdict(identity),
            execution={
                "planner": asdict(selection.planner),
                "steps": [
                    {"step_id": item.step_id, "implementer": asdict(item.implementer)}
                    for item in selection.steps
                ],
                "audit": asdict(selection.audit),
            },
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": selection.steps[index].implementer.profile_id}
                for index, step in enumerate(plan.steps)
            ],
        )
        self.runtime.last_selection = selection
        return replace(
            ctx, plan=plan, bundle=bundle, selection=selection, iteration=iteration,
        )

    def _failed_continued(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> tuple[str, ...]:
        rows = []
        for step in cycle_plan.plan.steps:
            path = cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id) / "step.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("status") == "FAILED_CONTINUED":
                rows.append(f"{step.id}: {record.get('reason', 'worker failure')}")
        return tuple(rows)

    def _close_iteration(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        outcome: IterationOutcome, decision: PlannerContinueResult, start_commit: str,
        fingerprint: str | None, status: str,
    ) -> None:
        number = ctx.iteration
        if number > 1:
            previous_path = ctx.run_dir / "iterations" / f"{number - 1:02d}" / "iteration.json"
            try:
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                start_commit = previous.get("end_commit", start_commit)
            except (OSError, ValueError):
                pass
        plan_path = iteration_plan_dir(ctx.run_dir, number) / "task_plan.json"
        try:
            import hashlib
            plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "iteration plan bytes are missing") from exc
        end_commit = current_head(ctx.info.worktree)
        remaining = tuple(dict.fromkeys((*outcome.audit_remaining, *decision.remaining)))
        record = {
            "iteration": number,
            "milestone_id": cycle_plan.plan.milestone_id,
            "plan_sha256": plan_sha,
            "start_commit": start_commit,
            "end_commit": end_commit,
            "gate_green": outcome.gate_green,
            "audit_status": outcome.audit_status,
            "remaining": list(remaining),
            "stagnation_fingerprint": fingerprint,
            "status": status,
        }
        path = ctx.run_dir / "iterations" / f"{number:02d}" / "iteration.json"
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        state = store.load()
        completed = [item for item in state.get("completed_iterations", []) if isinstance(item, Mapping)]
        compact = {
            "iteration": number, "milestone_id": record["milestone_id"], "status": status,
            "gate_green": record["gate_green"], "remaining": list(remaining),
            "failures": list(outcome.evidence.failures),
        }
        completed = [item for item in completed if item.get("iteration") != number]
        completed.append(compact)
        store.update_metadata(
            current_iteration=number, iteration_remaining=list(remaining),
            completed_iterations=completed,
            **({"completion_kind": "COMPLETE"} if status == "COMPLETE" else {}),
        )

    def _partial(
        self, store: RunStateStore, ctx: PipelineV2Context, reason: str,
        remaining: tuple[str, ...], failures: tuple[str, ...],
    ) -> RunResult:
        candidate_head = current_head(ctx.info.worktree)
        iteration_path = ctx.run_dir / "iterations" / f"{ctx.iteration:02d}" / "iteration.json"
        try:
            iteration_record = json.loads(iteration_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            iteration_record = {}
        if isinstance(iteration_record, dict):
            iteration_record.update(status="PARTIAL", partial_reason=reason)
            atomic_write_text(iteration_path, json.dumps(iteration_record, ensure_ascii=False, indent=2) + "\n")
        accepted_head = ctx.base_sha
        for number in range(ctx.iteration, 0, -1):
            path = ctx.run_dir / "iterations" / f"{number:02d}" / "iteration.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidate = record.get("end_commit") if isinstance(record, dict) else None
            if (isinstance(record, dict) and record.get("gate_green") is True
                    and isinstance(candidate, str)
                    and is_ancestor(ctx.info.worktree, ctx.base_sha, candidate)
                    and is_ancestor(ctx.info.worktree, candidate, candidate_head)):
                accepted_head = candidate
                break
        if candidate_head != accepted_head:
            rewind_worktree(ctx.info.worktree, accepted_head)
        head = accepted_head
        state_before = store.load()
        compact_iterations = [
            {**item, "status": "PARTIAL"}
            if isinstance(item, Mapping) and item.get("iteration") == ctx.iteration else item
            for item in state_before.get("completed_iterations", [])
        ]
        branch_push: dict[str, Any] = {"status": "not_attempted"}
        try:
            pushed = push_run_branch(
                ctx.info.worktree, remote=self.runtime.config.repository.remote,
                branch=ctx.info.branch, commit_sha=head,
            )
            branch_push = {"status": "pushed", "remote": pushed.remote, "branch": pushed.branch,
                           "commit_sha": pushed.commit_sha}
        except Exception:
            branch_push = {"status": "warning", "branch": ctx.info.branch}
        milestones = [
            {"iteration": item.get("iteration"), "milestone_id": item.get("milestone_id"),
             "status": item.get("status")}
            for item in store.load().get("completed_iterations", [])
            if isinstance(item, Mapping) and item.get("status") in {"NEXT", "COMPLETE"}
        ]
        report = {
            "completion_kind": "PARTIAL", "reason": reason,
            "last_accepted_commit": head, "discarded_unaccepted_head": candidate_head,
            "branch": ctx.info.branch,
            "milestones_completed": milestones, "remaining": list(remaining),
            "failures": list(failures), "branch_push": branch_push,
        }
        atomic_write_text(ctx.run_dir / "partial.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        state = store.set_run_state(
            RunMachineState(RunPhase.PLANNER, RunDisposition.COMPLETED),
            completion_kind="PARTIAL", iteration=ctx.iteration,
            current_iteration=ctx.iteration, completed_iterations=compact_iterations,
            partial=report, commit_sha=head, current_step=None,
        )
        return RunResult.of(ctx.run_dir, state)

    def _spec_decision(
        self, store: RunStateStore, ctx: PipelineV2Context,
        decision: PlannerContinueResult, outcome: IterationOutcome,
    ) -> RunResult:
        state = store.set_run_state(
            RunMachineState(RunPhase.PLANNER, RunDisposition.WAIT_HUMAN, "SPEC_DECISION_REQUIRED"),
            failure={"reason": "SPEC_DECISION_REQUIRED", "detail": decision.spec_question},
            current_step=None,
        )
        return RunResult.of(ctx.run_dir, state)

    def completed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        return completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number, [step.id for step in cycle_plan.plan.steps],
        )

    def effective_cycle_scope(
        self, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> tuple[str, ...]:
        """The approved envelope plus paths admitted by completed steps."""

        scope = set(cycle_plan.mutable_scope)
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
        iteration: int = 1,
    ) -> PipelineV2Context:
        """The immutable facts of this run, as the coordinator reads them."""

        return PipelineV2Context(
            run_dir=run_dir, run_id=run_id, spec=spec, context=context, repo=repo,
            base_sha=base_sha, base_tree_sha=base_tree_sha,
            repository_reference=repository_reference, info=info, plan=plan,
            bundle=bundle, selection=selection, options=self.runtime.run_options,
            iteration=iteration,
        )
