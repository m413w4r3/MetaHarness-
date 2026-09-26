"""The run bootstrap: planning, approval, worktree and setup.

``RunBootstrap`` owns everything that happens before the first step of a run:
the initial context and planning, the plan approval transaction, the execution
selection freeze, the worktree creation, the workspace setup, the initial
checkpoint and the pre-execution resume that re-enters those boundaries.
"""

from __future__ import annotations

import dataclasses, json, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, TYPE_CHECKING
from ..approval import (
    ApprovalDecision, ApprovalError, PlanIdentity,
    compute_plan_identity_from_run,
    read_plan_approval, wait_for_plan_approval,
    write_check_authority,
)
from ..context import build_context, render_context
from ..execution_selection import (
    ExecutionSelectionError,
    SCHEMA_VERSION as EXECUTION_SELECTION_SCHEMA,
    ensure_execution_selection,
    read_execution_selection_with_sha256, resolve_execution_selection,
    validate_execution_selection,
)
from ..gitops import (
    GitError, RepositoryReference, WorktreeInfo, branch_exists,
    build_repository_reference,
    build_run_branch, candidate_tree_sha, create_run_worktree,
    current_head, git_root,
    index_tree_sha, registered_worktrees, repository_reference_dict,
    resolve_commit, resolve_tree,
    stage_all, status_porcelain, symbolic_head,
)
from ..models import (
    BlockerKind, ExecutionRole, ExecutionSelection, ModelProfile, PlanDecision,
    INTERRUPTED_REASON,
    PLAN_REJECTED_REASON,
    RunDisposition,
    RunEvent,
    RunMachineState,
)
from ..plan_recovery import plan_source
from ..plan_repository_validation import RepositoryPreconditions, validate_plan_repository_topology
from ..planning.artifacts import validate_implementation_bundle
from ..planning.planner import PlannerV2
from ..planning.protocol import TaskPlanV2, V2PlanParseError, parse_task_plan_v2
from ..profiles import ProfileError, build_llm_endpoint, profile_for_role
from ..redaction import redact
from ..result import RunResult, atomic_write_text
from ..resume import (
    ResumeCheckpoint, ResumeCheckpointError, ResumeError,
    ResumeIntegrityError,
    ResumePhase, ResumeRequiresOperatorError, plan_identity_from_mapping,
    run_identity,
    write_checkpoint,
)
from ..state import RunStateStore
from ..usage import read_usage_artifact
from ..validation import config_with_check_authority
from ..workspace import prepare_workspace
from .resume_validation import persist_planner_conversation, read_repository_reference
from .shared import (
    GitOwnership, OrchestrationError, archive_attempt_tree, chat_client,
    git_ownership,
    git_ownership_payload, is_object_id, read_json_artifact,
)

if TYPE_CHECKING:
    from .runtime import RunRuntime

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


class RunBootstrap:
    """The planning, approval, worktree and setup of one new run."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def prepare_v2_run(
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

        revision_enabled = self.runtime.run_options.semantic_revision_enabled
        repair_enabled = self.runtime.run_options.max_review_repair_cycles > 0
        check_repair_enabled = self.runtime.run_options.max_check_repair_attempts > 0
        planner_profile_id = planner_profile.id
        if existing_plan is None:
            planner = PlannerV2(
                self.runtime.planner_client or chat_client(build_llm_endpoint(planner_profile), self.runtime.environment, self.runtime.observability.trace_transport),
                repository_reference=repository_reference,
                planning=self.runtime.config.planning,
                check_catalog=self.runtime.config.check_catalog,
                default_check_ids=self.runtime.config.default_check_ids,
                prompt_budget_bytes=self.runtime.config.prompt_budget.planner_max_bytes,
                repository_preconditions=RepositoryPreconditions(
                    repo, resolve_tree(repo, base_sha),
                ),
                on_event=lambda name, data: self.runtime.observability.trace_emit(name, phase="planning", cycle=1, data=data),
            )
            plan_started_at = self.runtime.observability.trace_time()
            plan_started_mono = time.perf_counter()
            self.runtime.observability.trace_emit(
                "plan.started",
                phase="planning",
                cycle=1,
                data={
                    "session": self.runtime.observability.trace_session(
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
                self.runtime.write_checkpoint(
                    run_dir, ResumePhase.PLANNER, head=base_sha,
                    tree=resolve_tree(repo, base_sha),
                )
                raise
            persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            self.runtime.observability.trace_emit(
                "plan.completed",
                phase="planning",
                cycle=1,
                data={
                    "decision": plan.decision.value,
                    "title": plan.title,
                    "session": self.runtime.observability.trace_finished_model_session(
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
        store.update_metadata(
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
                     "recommended_profile": self.runtime.config.routing.profile_for(step.execution_class),
                     "status": "waiting"}
                    for step in plan.steps
                ],
                "reviewer_recommendation": self.runtime.run_options.final_reviewer_profile,
            },
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": self.runtime.config.routing.profile_for(step.execution_class)}
                for step in plan.steps
            ],
            current_step=None,
        )
        self.runtime.cycle_update(
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
                        "max_steps_per_plan": self.runtime.config.planning.max_steps_per_plan,
                        "single_step_max_mutable_paths": self.runtime.config.planning.single_step_max_mutable_paths,
                        "staged_step_max_mutable_paths": self.runtime.config.planning.staged_step_max_mutable_paths,
                    },
                    "action": "operator must authorize a higher planning limit or split the requested work",
                }
            else:
                reason = "PLANNER_BLOCKED_REQUIRES_OPERATOR"
                detail = {"blocker_kind": None, "blockers": plan.blockers}
            state = store.set_run_state(
                RunMachineState(
                    disposition=RunDisposition.WAIT_HUMAN, reason=reason,
                ),
                failure={"reason": reason, "detail": detail},
            )
            return RunResult.of(run_dir, state)

        # Every source of this plan (planner, resumed planner answer, operator
        # recovery) must be possible against the base tree before it can be
        # offered for approval.  Resume re-enters here, so it is checked again.
        validate_plan_repository_topology(repo, resolve_tree(repo, base_sha), plan)
        try:
            # REQUIRED_CHECKS has already been parsed against the trusted
            # catalogue.  Materialize those exact trusted definitions before
            # the plan can become approval authority.
            selected_checks = self.runtime.config.select_checks(plan.required_checks)
            # Freeze the whole trusted catalogue, not just this selection: a
            # correction plan may legitimately require another approved check,
            # and it must still run the argv approved at this boundary.
            write_check_authority(
                run_dir, tuple(self.runtime.config.trusted_checks()),
                required_check_ids=tuple(check.id for check in selected_checks),
            )
            _bundle, _bundle_sha = validate_implementation_bundle(run_dir)
            plan_identity = compute_plan_identity_from_run(run_dir)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update_metadata(plan_identity=asdict(plan_identity))
        self.runtime.write_checkpoint(
            run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha,
            tree=resolve_tree(repo, base_sha), plan_identity=plan_identity,
        )

        if self.runtime.config.approval.require_plan_approval:
            store.update_metadata()
            approval = wait_for_plan_approval(
                run_dir, identity=plan_identity,
                poll_interval_seconds=self.runtime.config.approval.poll_interval_seconds,
            )
            if approval.decision is ApprovalDecision.REJECT:
                state = store.set_run_state(RunMachineState(
                    disposition=RunDisposition.WAIT_HUMAN, reason=PLAN_REJECTED_REASON,
                ))
                return RunResult.of(run_dir, state)
            try:
                selection, execution_sha = read_execution_selection_with_sha256(run_dir)
                validate_execution_selection(self.runtime.config, selection)
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
                self.runtime.config,
                planner_profile_id=planner_profile_id,
                plan_steps=plan.steps,
                semantic_reviser_profile_id=(
                    self.runtime.run_options.semantic_reviser_profile
                    if revision_enabled or repair_enabled else None
                ),
                check_repair_profile_id=(
                    self.runtime.run_options.check_repair_profile
                    if check_repair_enabled else None
                ),
                final_reviewer_profile_id=self.runtime.run_options.final_reviewer_profile,
                fallback_authority=self.runtime.run_options.recovery.execution_fallbacks,
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
        validate_execution_selection(self.runtime.config, selection)
        self.runtime.last_selection = selection
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
        store.update_metadata(execution=execution_state,
                              plan_identity=asdict(durable_identity))
        self.runtime.observability.trace_emit(
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
        self.runtime.write_checkpoint(
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
                worktree_path=self.runtime.config.worktrees_root / run_id,
                require_clean_base=self.runtime.config.require_clean_base,
            )
        else:
            info = existing_info
        store.update_metadata(branch=info.branch,
                              worktree=str(info.worktree), base_sha=info.base_sha)
        self.runtime.publication.ensure_github_issue_metadata(
            store=store, run_id=run_id, plan_title=plan.title, info=info,
        )
        self.runtime.observability.trace_emit(
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
        ownership_before = git_ownership(repo, info.worktree)
        # Persisted with an explicit status so a later resume has the stronger
        # durable ownership proof of the pre-execution boundary.
        store.update_metadata(git_ownership=git_ownership_payload(ownership_before))
        setup_results = self.runtime.check_recovery(store).prepare_workspace(
            worktree=info.worktree, run_dir=run_dir,
            run_setup=lambda: prepare_workspace(
                info.worktree, self.runtime.config.workspace_setup,
                environment=self.runtime.environment, artifacts_dir=run_dir,
                secrets=self.runtime.secrets,
            ),
        )
        store.update_metadata(
            workspace_setup=[asdict(result) for result in setup_results],
        )
        # The candidate starts as the base tree and all later gates use the
        # actual index identity, never an inferred file list.
        stage_all(info.worktree)
        candidate_tree = index_tree_sha(info.worktree)
        if candidate_tree != base_tree_sha:
            raise OrchestrationError("initial candidate tree does not match base")
        # The workspace preflight is the first live use of the authority; the
        # expected hash is the one this run's identity already durably bound.
        check_config, check_ids = config_with_check_authority(
            self.runtime.config, run_dir, expected_sha256=durable_identity.checks_sha256,
        )
        preflight_failures = self.runtime.gates.run_check_preflights_recoverably(
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

        claimed = store.transition_run(
            RunEvent.resume(), expected=run_identity(state, run_dir, checkpoint),
            failure=None, current_step=None,
            resume={**record, "status": "running"},
        )
        if claimed is None:
            raise ResumeError("run state changed while the resume was validated")
        if on_claimed is not None:
            on_claimed(run_dir)
        try:
            repo = git_root(self.runtime.config.repo)
            if state.get("repo") not in {None, str(repo), str(self.runtime.config.repo)}:
                refuse("the configured repository is not the run repository")
            spec = (run_dir / "spec.md").read_text(encoding="utf-8")
            base_sha = state.get("base_sha")
            if base_sha is None:
                base_sha = resolve_commit(repo, self.runtime.config.base_ref)
                base_tree = resolve_tree(repo, base_sha)
                store.update_metadata(repo=str(repo), base_sha=base_sha)
            else:
                if not is_object_id(base_sha):
                    refuse("run base SHA is invalid")
                base_tree = resolve_tree(repo, base_sha)
            if checkpoint.expected_head_sha is not None and checkpoint.expected_head_sha != base_sha:
                refuse("the checkpoint base SHA changed")
            if checkpoint.expected_tree_sha is not None and checkpoint.expected_tree_sha != base_tree:
                refuse("the checkpoint base tree changed")
            reference = read_repository_reference(run_dir)
            if (run_dir / "repository_reference.json").exists() and reference is None:
                refuse("repository reference artifact is corrupted")
            if reference is None:
                if checkpoint.phase is not ResumePhase.CONTEXT:
                    refuse("repository reference artifact is missing")
                try:
                    reference = build_repository_reference(repo, base_sha=base_sha, config=self.runtime.config.repository)
                except GitError:
                    reference = RepositoryReference(self.runtime.config.repository.remote, None, base_sha, None)
                atomic_write_text(run_dir / "repository_reference.json", json.dumps(
                    repository_reference_dict(reference), indent=2
                ) + "\n")
            if reference.base_sha != base_sha:
                refuse("the base SHA changed for this run")

            context_path = run_dir / "context.txt"
            if checkpoint.phase is ResumePhase.CONTEXT:
                context_bundle = build_context(repo, base_sha, spec, self.runtime.config.context)
                context = render_context(context_bundle)
                atomic_write_text(context_path, context)
                store.update_metadata(context={
                    "base_sha": context_bundle.base_sha,
                    "locator_used": context_bundle.locator_used,
                    "locator_warning": context_bundle.locator_warning,
                    "omitted": list(context_bundle.omitted),
                    "total_bytes": context_bundle.total_bytes,
                })
                self.runtime.write_checkpoint(run_dir, ResumePhase.PLANNER,
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
            planner_profile = profile_for_role(self.runtime.config, planner_profile_id, ExecutionRole.PLANNER)
            if checkpoint.phase is ResumePhase.PLANNER:
                planner = PlannerV2(
                    self.runtime.planner_client or chat_client(
                        build_llm_endpoint(planner_profile), self.runtime.environment, self.runtime.observability.trace_transport
                    ),
                    repository_reference=reference, planning=self.runtime.config.planning,
                    check_catalog=self.runtime.config.check_catalog,
                    default_check_ids=self.runtime.config.default_check_ids,
                    prompt_budget_bytes=self.runtime.config.prompt_budget.planner_max_bytes,
                    repository_preconditions=RepositoryPreconditions(repo, base_tree),
                    on_event=lambda name, data: self.runtime.observability.trace_emit(name, phase="planning", cycle=1, data=data),
                )
                plan = planner.plan(spec, context, artifacts_dir=run_dir)
                persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            else:
                raw = (run_dir / "planner.raw.md").read_text(encoding="utf-8")
                plan = parse_task_plan_v2(
                    raw,
                    planning=self.runtime.config.planning,
                    check_catalog=self.runtime.config.check_catalog,
                    default_check_ids=self.runtime.config.default_check_ids,
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
                        validate_execution_selection(self.runtime.config, _selection)
                    except (ExecutionSelectionError, ProfileError, OSError, UnicodeError) as exc:
                        refuse(f"execution selection is invalid: {exc}")
                    if selection_sha != checkpoint.execution_selection_sha256:
                        refuse("execution selection hash changed")
                    if self.runtime.config.approval.require_plan_approval:
                        try:
                            approval = read_plan_approval(run_dir, expected_identity=identity)
                        except ApprovalError as exc:
                            refuse(f"approval artifact is invalid: {exc}")
                        if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                            refuse("approval was changed or is no longer APPROVE")
            existing_info = self._existing_setup_worktree(
                repo, run_dir, run_id, plan, base_sha, self.runtime.config.worktrees_root,
                self.runtime.config.base_ref,
            ) if checkpoint.phase is ResumePhase.WORKTREE_SETUP else None
            if existing_info is not None:
                archive_attempt_tree(run_dir / "setup")
            prepared = self.prepare_v2_run(
                store, run_dir, run_id, spec, repo, base_sha, context,
                reference, planner_profile, existing_plan=plan,
                existing_info=existing_info,
            )
            if isinstance(prepared, RunResult):
                return prepared
            return self.runtime.composition.execute_v2(
                store, run_dir, run_id, spec, repo, base_sha, context, reference,
                prepared=prepared,
            )
        except ResumeRequiresOperatorError as exc:
            failed = store.record_failure(exc.code, redact(str(exc), self.runtime.secrets),
                                          **self.runtime.failure.closing_step_fields(store, "failed"))
            return RunResult.of(run_dir, failed)
        except KeyboardInterrupt:
            interrupted = store.set_run_state(
                RunMachineState(
                    disposition=RunDisposition.FAILED, reason=INTERRUPTED_REASON,
                ),
                failure={"reason": INTERRUPTED_REASON},
                **self.runtime.failure.closing_step_fields(store, "interrupted"),
            )
            return RunResult.of(run_dir, interrupted)
        except Exception as exc:
            return self.runtime.failure.project_exception(store, run_dir, exc)

    @staticmethod
    def _existing_setup_worktree(
        repo: Path, run_dir: Path, run_id: str, plan: TaskPlanV2, base_sha: str,
        worktrees_root: Path, base_ref: str,
    ) -> WorktreeInfo | None:
        """Return only an exactly recognizable partial setup; never delete/repair it."""

        expected_branch = build_run_branch(plan.title, run_id)
        raw_path = None
        try:
            raw_state = read_json_artifact(run_dir / "state.json", 256 * 1024)
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
