"""The red-gate cycle replan: bounded facts, one transaction, one cycle.

A deterministic check that stays red after the bounded repair pass and the
single-step replan proves its decomposition wrong rather than one of its steps,
and this service answers it with a new decomposition.  Every fact handed to
the planner is read from durable artifacts, so a resume rebuilds byte for byte
the same request and the same anti-loop fingerprint, and the answer is written
before the cycle that executes it exists.  The service never depends on the
reviewer: a check replan is answered by the planner alone, and reuse of the
already approved paths goes through the run's own scope policy.
"""

from __future__ import annotations

import time
from typing import (
    Sequence,
    TYPE_CHECKING,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    changed_paths_between_trees,
    path_exists_in_tree,
    render_repository_reference,
)
from ..models import (
    CycleKind,
    ExecutionRole,
    GateStage,
    PlanDecision,
    RunCycle,
    correction_cycles_used,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
)
from ..planning.artifacts import validate_implementation_bundle
from ..planning.check_replan import (
    CheckReplanFacts,
    CheckReplanTransaction,
    bounded_check_proofs,
    bounded_diff_summary,
    check_replan_dir,
    plan_identity,
    render_path_facts,
)
from ..planning.protocol import (
    V2PlanParseError,
    render_repair_plan_summary,
    render_repair_step_index,
)
from ..profiles import (
    build_llm_endpoint,
    profile_for_role,
)
from ..state import RunStateStore
from .check_failure import (
    check_failure_proofs,
    soft_check_failures,
)
from .correction_scope import build_scope_delta
from .gate_recovery import consumed_ladder_strategies
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    gate_dir,
)
from .recovery import (
    GateRecoveryStep,
    RecoveryStepUnavailable,
)
from .shared import (
    OrchestrationError,
    _json_text,
    bounded_parse_detail,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime

# The bounded path facts one red gate may show a planner.
_MAX_REPLAN_PATH_FACTS = 24




def _check_replan_facts(
    *, ctx: PipelineV2Context, cycle_plan: CyclePlan, cycle: int, stage: GateStage,
    evidence: EvidenceBundle, approved_scope: Sequence[str],
) -> CheckReplanFacts:
    """The bounded facts one red deterministic gate hands a re-decomposition.

    Every field is read from durable artifacts -- the gate evidence, its
    archived check logs, the plan in force, the ladder ledger and Git objects
    -- and never from the current worktree content, so a resume rebuilds byte
    for byte the same request and the same anti-loop fingerprint.  The facts
    name the approved envelope without widening it: a plan that needs a path no
    earlier plan approved still asks the run's own scope policy.
    """

    tree = evidence.staged_tree_sha
    declared = {
        path for step in cycle_plan.plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }
    in_question = sorted(set(approved_scope) | declared | set(evidence.changed_files))
    return CheckReplanFacts(
        cycle=cycle, stage=stage.value, candidate_tree_sha=tree,
        failed_check_ids=tuple(
            item.split(":", 1)[1] for item in soft_check_failures(evidence) if ":" in item
        ),
        plan_identity_before=plan_identity(cycle_plan.plan),
        approved_mutable_envelope=tuple(approved_scope),
        repository_reference=render_repository_reference(ctx.repository_reference),
        original_spec=ctx.spec,
        repository_state=_json_text({
            "BASE_SHA": ctx.base_sha,
            "CANDIDATE_TREE_SHA": tree,
            "CHANGED_FILES": changed_paths_between_trees(
                ctx.repo, ctx.base_tree_sha, tree,
            ),
        }),
        approved_plan_summary=render_repair_plan_summary(cycle_plan.plan),
        approved_step_index=render_repair_step_index(cycle_plan.plan),
        check_failure_proofs=bounded_check_proofs(check_failure_proofs(
            evidence=evidence,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=tree,
        )),
        candidate_diff_summary=bounded_diff_summary(evidence.diff),
        repository_path_facts=render_path_facts([
            {"path": path, "exists": path_exists_in_tree(ctx.repo, tree, path)}
            for path in in_question[:_MAX_REPLAN_PATH_FACTS]
        ]),
        consumed_strategies=consumed_ladder_strategies(
            ctx.run_dir, cycle_plan.cycle.number, stage,
        ),
    )


class CheckReplanService:
    """The red-gate cycle replan of one run."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def replan_cycle(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> CyclePlan:
        """Answer one spent red gate with a new decomposition of its cycle.

        The rung that reaches this method is the last one of the episode: the
        bounded repair pass and the single-step replan were durably spent and
        the gate is still red, so the failure proves the decomposition wrong
        rather than one of its steps.  The bounded facts and the one planning
        transaction are the check-replan ones; the answer becomes the plan of
        the cycle that executes it, written before that cycle exists so a crash
        resumes the very same answer instead of paying for a second one.  The
        planner may restructure the steps, correct their contracts and reuse
        every approved path -- it names no new authority, so a path no earlier
        plan approved still goes through the run's own scope policy.
        """

        if correction_cycles_used(cycle_plan.cycle.number) >= ctx.options.max_correction_cycles:
            # The single correction budget is spent: fail closed before the
            # planner transaction, so the ladder advances to its next rung.
            raise RecoveryStepUnavailable(
                step.strategy, "the run's correction budget is spent",
            )
        cycle = RunCycle(cycle_plan.cycle.number + 1, CycleKind.CHECK_REPLAN)
        directory = check_replan_dir(ctx.run_dir, cycle.number)
        facts = _check_replan_facts(
            ctx=ctx, cycle_plan=cycle_plan, cycle=cycle.number, stage=stage,
            evidence=evidence, approved_scope=self.runtime.composition.approved_scope_before(
                ctx, cycle.number,
            ),
        )
        profile = profile_for_role(
            self.runtime.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        store.update_metadata(current_step=None)
        started_at, started_mono = self.runtime.observability.trace_time(), time.perf_counter()
        selected = self.runtime.observability.trace_selected_profile(
            ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        self.runtime.observability.trace_emit(
            "plan.started", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "tree_before": facts.candidate_tree_sha,
                "session": self.runtime.observability.trace_session(
                    profile=profile, selected=selected, role=ExecutionRole.PLANNER,
                    prompt_bytes=None, started_at=started_at, started_mono=started_mono,
                    tree_before=facts.candidate_tree_sha,
                ),
            },
        )
        try:
            plan = CheckReplanTransaction(
                client=self.runtime.planner_client
                or self.runtime.chat(build_llm_endpoint(profile)),
                artifacts_dir=directory,
                planning=self.runtime.config.planning,
                check_catalog=self.runtime.config.check_catalog,
                original_required_check_ids=ctx.plan.required_checks,
                repository_preconditions=RepositoryPreconditions(
                    ctx.repo, facts.candidate_tree_sha,
                ),
            ).plan(facts)
        except PlanRepositoryPreconditionError as exc:
            raise PipelineFailure(exc.code, bounded_parse_detail(exc)) from exc
        except V2PlanParseError as exc:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", bounded_parse_detail(exc)) from exc
        self.runtime.observability.trace_emit(
            "plan.completed", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "decision": plan.decision.value, "title": plan.title,
                "session": self.runtime.observability.trace_finished_model_session(
                    profile=profile, selected=selected, role=ExecutionRole.PLANNER,
                    prompt_bytes=(
                        (directory / "planner.request.txt").stat().st_size
                        if (directory / "planner.request.txt").is_file() else None
                    ),
                    started_at=started_at, started_mono=started_mono,
                    usage=None, tree_before=facts.candidate_tree_sha,
                    tree_after=facts.candidate_tree_sha, final_message=getattr(plan, "raw", None),
                ),
            },
        )
        self.runtime.cycle_update(store, cycle, status="planning", plan_summary=plan.title)
        if plan.decision is PlanDecision.BLOCKED:
            self.runtime.cycle_update(store, cycle, status="blocked", blockers=plan.blockers)
            raise PipelineFailure("REPAIR_PLANNER_BLOCKED", plan.blockers)
        if plan_identity(plan) == facts.plan_identity_before:
            # The planner answered the very decomposition already in force:
            # executing it would replay approved work.  The rung is spent and
            # its durable record keeps these exact facts from being re-planned.
            self.runtime.cycle_update(
                store, cycle, status="failed", failure="CHECK_REPLAN_UNCHANGED",
            )
            raise RecoveryStepUnavailable(
                step.strategy, "the cycle replan re-decomposed nothing",
            )
        bundle, bundle_sha = validate_implementation_bundle(
            directory, expected_step_ids=[step.id for step in plan.steps]
        )
        try:
            delta, content = build_scope_delta(
                directory, original_scope=list(facts.approved_mutable_envelope),
                plan=plan, candidate_commit_sha=facts.candidate_tree_sha,
                repair_bundle_sha=bundle_sha,
                justification=f"{stage.value}: {' '.join(facts.failed_check_ids)}",
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        self.runtime.correction_scope.apply(
            store, ctx, cycle, plan, directory, delta, content,
            list(facts.approved_mutable_envelope), facts.candidate_tree_sha,
        )
        return self.runtime.composition.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=True,
        )

