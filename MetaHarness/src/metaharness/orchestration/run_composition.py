"""The composition of one run: context, operations and cycle authority.

``RunComposition`` builds the immutable :class:`PipelineV2Context`, wires every
:class:`PipelineV2Operations` entry to the service that owns it, loads the plan
authority of each durable cycle and composes the effective mutable scope a
cycle is allowed to touch.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING
from ..agent.base import AgentRunRequest
from ..agent.execution import ExecutorRuntimeConfig, executor_for_profile
from ..baseline import BaselineCache, baseline_payload
from ..execution_selection import (
    ExecutionSelectionError, ensure_cycle_execution_selection,
    read_cycle_execution_selection,
    resolve_cycle_execution_selection,
    validate_cycle_execution_selection,
)
from ..gitops import (
    RepositoryReference,
    WorktreeInfo,
    candidate_tree_sha,
    current_head,
)
from ..llm.chat import LLMError
from ..models import (
    CycleKind, ExecutionRole, ExecutionSelection, RunCycle, TaskPlanV2, is_replan_cycle,
)
from ..planning.check_replan import check_replan_dir
from ..profiles import build_llm_endpoint, profile_for_role, profiles_for_config
from ..recommendation import ExecutionRecommender, RecommendationError, write_recommendation_error
from ..redaction import redact
from ..result import RunResult, atomic_write_text
from ..review import ReviewResult, Reviewer
from ..resume import ResumeIntegrityError
from ..scope import ScopeViolation
from ..state import RunStateStore
from ..validation import config_with_check_authority
from .candidate import CandidateLifecycle
from .check_failure import hard_integrity_failures
from .gate_acceptance import GateAcceptanceService
from .pipeline_v2 import (
    CyclePlan, PipelineFailure, PipelineV2Context, PipelineV2Operations,
    correction_dir, gate_dir,
    step_dir as cycle_step_dir,
)
from .cycle_loader import (
    build_scope_delta, ensure_scope_delta, load_correction_plan, read_cycle_record,
    semantic_revision_scope,
    verify_correction_scope,
)
from .durable_readers import (
    completed_step_records, load_evidence, load_revision, read_candidate_record,
    gate_mutable_authority, settled_step_status,
)
from .run_bootstrap import PreparedV2Run
from .shared import (
    CycleArtifactService,
    OrchestrationError,
    bounded_parse_detail,
    is_object_id,
    json_text,
    read_json_artifact,
)
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
            client = self.runtime.chat(build_llm_endpoint(profile))
        return Reviewer(client, allow_format_repair=True)

    def _recommender_for_profile(self, profile_id: str) -> ExecutionRecommender:
        profile = profile_for_role(self.runtime.config, profile_id, ExecutionRole.PLANNER)
        client = self.runtime.recommender_client
        if client is None:
            # This is deliberately a new client: the recommender has no
            # planner conversation/history, while using the same profile
            # endpoint and transport policy.
            client = self.runtime.chat(build_llm_endpoint(profile))
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
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha)
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
        verify_correction_scope(ctx.run_dir, cycle.number, bundle_sha)
        return self.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=False,
        )

    def completed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        return completed_step_records(
            ctx.run_dir, cycle_plan.cycle.number, [step.id for step in cycle_plan.plan.steps],
        )

    def reviewed_steps(self, ctx: PipelineV2Context, cycle_plan: CyclePlan) -> list[dict[str, Any]]:
        """The step reports a reviewer reads: an abandoned step is a visible deficit."""

        records = {record["id"]: record for record in self.completed_steps(ctx, cycle_plan)}
        for step in cycle_plan.plan.steps:
            status = settled_step_status(cycle_step_dir(ctx.run_dir, cycle_plan.cycle, step.id), step.id)
            if status is not None:
                records[step.id] = {"id": step.id, "status": status, "changed_paths": []}
        return [records[step.id] for step in cycle_plan.plan.steps if step.id in records]

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
            ctx.run_dir, cycle_plan.cycle.number,
        ))
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

    def authorize_review_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, bundle_sha: str, candidate_sha: str, review: ReviewResult,
        approved_scope: list[str],
    ) -> None:
        """Record one review-driven correction's scope delta and admit its paths."""

        repair_dir = correction_dir(ctx.run_dir, cycle)
        try:
            delta, content = build_scope_delta(
                repair_dir, original_scope=approved_scope, plan=plan,
                candidate_commit_sha=candidate_sha, repair_bundle_sha=bundle_sha,
                justification=review.required_fixes.strip() or review.findings.strip(),
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        self.apply_correction_scope(
            store, ctx, cycle, plan, repair_dir, delta, content, approved_scope, candidate_sha,
        )

    def apply_correction_scope(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, repair_dir: Path, delta: Mapping[str, Any], content: str,
        approved_scope: list[str], candidate_sha: str,
    ) -> None:
        """Record one correction scope delta and admit its added paths.

        A review-driven correction and a red-gate cycle replan widen the
        approved envelope through this one path.  The delta is derived from the
        parsed plan alone and created once; the paths it adds are admitted for
        this correction and audited, and the only refusal is the shared scope
        authority's.
        """

        # Created once; on every later pass (resume included) the persisted
        # bytes are only compared, never repaired.
        delta_sha = ensure_scope_delta(repair_dir, content, expected_sha256=None)
        requested = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        try:
            self.runtime.config.scope.check(
                (*requested, *delta["added_paths"]), worktree=ctx.info.worktree,
            )
        except ScopeViolation as violation:
            raise PipelineFailure(violation.code, violation.detail) from None
        atomic_write_text(repair_dir / "scope.json", json_text({
            "repair_mutable_scope": requested,
            "approved_mutable_scope_before": approved_scope,
            "scope_delta_sha256": delta_sha,
        }))
        if delta["added_paths"]:
            self.runtime.cycle_update(
                store, cycle, status="scope_recorded", scope_delta=delta,
            )
        # The correction plan may require trusted checks the initial plan did
        # not; their config-only preflights run before any expensive worker.
        check_config, check_ids = config_with_check_authority(
            self.runtime.config, ctx.run_dir, requested_check_ids=plan.required_checks,
            expected_sha256=self.runtime.approved_check_authority_sha256(ctx.run_dir),
        )
        selected_check_ids = tuple(check_ids or plan.required_checks)
        skipped_checks = self.runtime.gates.run_check_preflights_recoverably(
            store=store, run_dir=ctx.run_dir, worktree=ctx.info.worktree,
            check_config=check_config, check_ids=selected_check_ids,
            phase="planning", cycle=cycle.number,
        )
        # A check this cycle only now requires still gets its own baseline on
        # the unchanged base commit, before any worker runs for it.
        baseline = BaselineCache(self.runtime.config.runs_root).ensure(
            repo=ctx.info.worktree, base_sha=ctx.base_sha, config=check_config,
            check_ids=tuple(dict.fromkeys(
                (*selected_check_ids, *self.runtime.config.gate.per_step)
            )),
            environment=self.runtime.environment,
            setup_commands=self.runtime.config.workspace_setup,
            secrets=self.runtime.secrets,
            skipped={check_id: "PREFLIGHT_FAILED" for check_id in skipped_checks},
        )
        store.update_metadata(
            skipped_checks=list(skipped_checks), baseline=baseline_payload(baseline),
        )

    def admit_scope_request(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        artifact_dir: Path, current_scope: list[str],
    ) -> tuple[str, list[str]]:
        """Admit the paths of one durable ``META SCOPE REQUEST`` automatically.

        The request is parsed, its paths normalized and checked against the one
        scope authority, and then admitted for this correction: no operator
        decision, no second model call, no wait.  The durable record stays so a
        resume re-derives the same scope.
        """

        report = read_json_artifact(artifact_dir / "report.json", 256 * 1024)
        request = report.get("scope_request") if isinstance(report, dict) else None
        paths = request.get("paths") if isinstance(request, dict) else None
        reason = request.get("reason") if isinstance(request, dict) else None
        evidence = request.get("evidence") if isinstance(request, dict) else None
        tree_sha = report.get("tree_before") if isinstance(report, dict) else None
        if (
            not isinstance(paths, list) or not paths or any(
                not isinstance(path, str) for path in paths
            ) or len(paths) != len(set(paths))
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence)
            or not isinstance(tree_sha, str) or not is_object_id(tree_sha)
        ):
            raise PipelineFailure("AGENT_SCOPE_VIOLATION", "semantic scope request is malformed")
        try:
            requested = self.runtime.config.scope.check(paths, worktree=ctx.info.worktree)
        except ScopeViolation as violation:
            raise PipelineFailure(violation.code, violation.detail) from None
        source_report = artifact_dir / "report.json"
        try:
            source_report_sha = hashlib.sha256(source_report.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_REQUIRES_OPERATOR", "semantic scope request report is unreadable",
            ) from exc
        base = tuple(sorted(set(current_scope)))
        requested = tuple(sorted(set(requested)))
        added = tuple(path for path in requested if path not in base)
        root = artifact_dir / "scope_requests"
        root.mkdir(parents=True, exist_ok=True)
        next_number = 1
        prior_request_dir: Path | None = None
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.isdigit():
                continue
            next_number = max(next_number, int(path.name) + 1)
            saved = read_json_artifact(path / "authority.json", 64 * 1024)
            if isinstance(saved, dict) and (
                saved.get("tree_sha") == tree_sha
                and saved.get("base_mutable_scope") == list(base)
                and saved.get("requested_paths") == list(requested)
                and saved.get("reason") == reason
                and saved.get("evidence") == [item[:1000] for item in evidence[:16]]
            ):
                prior_request_dir = path
        target = prior_request_dir or root / f"{next_number:03d}"
        target.mkdir(parents=True, exist_ok=True)
        authority = {
            "schema_version": 1,
            "cycle": cycle_plan.cycle.number,
            "tree_sha": tree_sha,
            "source_report_sha256": source_report_sha,
            "base_mutable_scope": list(base),
            "requested_paths": list(requested),
            "added_paths": list(added),
            "reason": reason[:2000],
            "evidence": [item[:1000] for item in evidence[:16]],
        }
        authority_path = target / "authority.json"
        if authority_path.exists():
            saved_authority = read_json_artifact(authority_path, 64 * 1024)
            saved_semantics = (
                {key: value for key, value in saved_authority.items()
                 if key != "source_report_sha256"}
                if isinstance(saved_authority, dict) else None
            )
            current_semantics = {
                key: value for key, value in authority.items()
                if key != "source_report_sha256"
            }
            if saved_semantics != current_semantics:
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", "semantic scope request authority changed",
                )
            authority = saved_authority
        else:
            atomic_write_text(authority_path, json_text(authority))
        atomic_write_text(target / "decision.json", json_text({
            **authority, "decision": "auto-admitted",
        }))
        if not added:
            atomic_write_text(artifact_dir / "status.json", json_text({
                "status": "SCOPE_REQUEST_RECORDED",
                "reason": reason[:2000],
                "requested_paths": list(requested),
            }))
            return "recorded", current_scope
        self.runtime.cycle_update(
            store, cycle_plan.cycle,
            status="scope_recorded", semantic_revision_status="SCOPE_REQUEST_RECORDED",
        )
        return "expanded", sorted(set(current_scope) | set(added))

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
