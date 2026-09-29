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
from ..baseline import BaselineCache, baseline_payload
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
from ..llm.chat import LLMConversationHandle
from ..models import (
    BlockerKind, CheckConfig, ExecutionRole, ExecutionSelection, ModelProfile, PlanDecision,
    INTERRUPTED_REASON,
    PLAN_REJECTED_REASON,
    RunDisposition,
    RunEvent,
    RunMachineState,
)
from ..plan_repository_validation import RepositoryPreconditions, validate_plan_repository_topology
from ..planning.artifacts import (
    iteration_plan_dir, persist_iteration_plan, read_iteration_plan,
    validate_implementation_bundle,
)
from ..planning.planner import PlannerV2
from ..planning.protocol import TaskPlanV2, V2PlanParseError
from ..profiles import ProfileError, build_llm_endpoint, profile_for_role
from ..redaction import redact
from ..result import RunResult, atomic_write_text
from ..resume import (
    ResumeCheckpoint, ResumeError,
    ResumeIntegrityError,
    ResumePhase, ResumeRequiresOperatorError,
    run_identity,
    write_checkpoint,
)
from ..state import RunStateStore
from ..usage import read_usage_artifact
from ..validation import config_with_check_authority, frozen_check_policy
from ..workspace import prepare_workspace
from .durable_readers import read_repository_reference
from .pipeline_v2 import BudgetExhausted
from .per_step_gate import per_step_check_ids
from .shared import (
    GitOwnership, OrchestrationError, PLANNER_CONVERSATION, archive_attempt_tree,
    git_ownership, git_ownership_payload, is_object_id, json_text,
    read_json_artifact,
)

if TYPE_CHECKING:
    from .runtime import RunRuntime


def persist_planner_conversation(run_dir: Path, handle: Any) -> None:
    """Persist a driver-provided planner conversation handle, never a guess."""

    if isinstance(handle, LLMConversationHandle):
        path = run_dir / PLANNER_CONVERSATION
        atomic_write_text(path, json_text({
            "provider_id": handle.provider_id, "conversation_id": handle.conversation_id,
        }))
        path.chmod(0o600)


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

    def _check_policy(
        self, run_dir: Path,
    ) -> tuple[tuple[CheckConfig, ...], tuple[str, ...]]:
        """The durable check policy of a run that already froze one.

        Before the freeze the current TOML *is* the run's policy; afterwards it
        can only remove an ID, so a modified TOML never changes the catalogue
        or the default check IDs a resumed planner consumes.
        """

        policy = frozen_check_policy(self.runtime.config, run_dir)
        if policy is not None:
            return policy.checks, policy.default_check_ids
        return (
            tuple(self.runtime.config.trusted_checks()),
            tuple(self.runtime.config.required_check_ids()),
        )

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
        iteration: int = 1,
    ) -> "PreparedV2Run | RunResult":
        """Plan, obtain the approved selection, create the worktree and set up.

        Returns a terminal :class:`RunResult` for BLOCKED/REJECTED plans.  None
        of this is ever replayed by a resume.
        """

        planner_profile_id = planner_profile.id
        if planner_profile_id != self.runtime.run_options.planner_profile:
            raise ExecutionSelectionError("planner profile differs from the frozen run options")
        check_catalog, default_check_ids = self._check_policy(run_dir)
        if existing_plan is None:
            planner = PlannerV2(
                self.runtime.planner_client
                or self.runtime.chat(build_llm_endpoint(planner_profile)),
                repository_reference=repository_reference,
                planning=self.runtime.config.planning,
                check_catalog=check_catalog,
                default_check_ids=default_check_ids,
                prompt_budget_bytes=self.runtime.config.prompt_budget.planner_max_bytes,
                repository_preconditions=RepositoryPreconditions(
                    repo, resolve_tree(repo, base_sha),
                ),
                attempt_budget=self.runtime.run_options.budget.step_attempts,
                on_event=lambda name, data: self.runtime.observability.trace_emit(name, phase="planning", cycle=iteration, data=data),
            )
            plan_started_at = self.runtime.observability.trace_time()
            plan_started_mono = time.perf_counter()
            self.runtime.observability.trace_emit(
                "plan.started",
                phase="planning",
                cycle=iteration,
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
            if (exhausted := self.runtime.budget_exhausted(store)) is not None:
                raise BudgetExhausted(exhausted)
            try:
                plan = planner.plan(spec, context, artifacts_dir=run_dir, iteration=iteration)
            except Exception:
                # The failed planner answer is deliberately left in place and
                # will be archived by the next planner attempt.
                self.runtime.write_checkpoint(
                    run_dir, ResumePhase.PLANNER, head=base_sha,
                )
                raise
            persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            self.runtime.observability.trace_emit(
                "plan.completed",
                phase="planning",
                cycle=iteration,
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
                "source": "planner",
                "execution_mode": plan.execution_mode.value if plan.execution_mode else None,
                "required_checks": list(plan.required_checks),
            },
            iteration=iteration,
            current_milestone={"id": plan.milestone_id, "title": plan.milestone_title},
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": self.runtime.config.routing.profile_for(step.execution_class)}
                for step in plan.steps
            ],
            current_step=None,
        )
        self.runtime.cycle_update(
            store, iteration, status="running", plan_summary=plan.title,
            steps_summary=[{"id": step.id, "title": step.title} for step in plan.steps],
        )
        if plan.decision is PlanDecision.BLOCKED:
            # ``SPEC_DECISION`` is the only blocker a plan can carry: every
            # other obstacle is MetaHarness' own to resolve.
            kind = plan.blocker_kind
            reason = (
                "SPEC_DECISION_REQUIRED" if kind is BlockerKind.SPEC_DECISION
                else "PLANNER_BLOCKED_REQUIRES_OPERATOR"
            )
            detail: dict[str, Any] = {
                "blocker_kind": kind.value if kind else None,
                "blockers": plan.blockers,
                "action": "operator must resolve the product choice left open by the SPEC",
            }
            state = store.set_run_state(
                RunMachineState(
                    disposition=RunDisposition.WAIT_HUMAN, reason=reason,
                ),
                failure={"reason": reason, "detail": detail},
            )
            return RunResult.of(run_dir, state)

        # Every planner answer must be possible against the base tree before it
        # can be offered for approval. Resume re-enters here, so it is checked again,
        # and the effective plan it returns is the one approval binds.
        plan = validate_plan_repository_topology(repo, resolve_tree(repo, base_sha), plan)
        try:
            # REQUIRED_CHECKS has already been parsed against the trusted
            # catalogue.  Materialize those exact trusted definitions before
            # the plan can become approval authority.
            selected_checks = self.runtime.config.select_checks(plan.required_checks)
            # Freeze the whole trusted catalogue *and* the run's default check
            # policy, not just this selection: a later milestone may require
            # another approved check, and every run must keep running the argv
            # and the defaults approved at this boundary.
            new_check_authority = not (run_dir / "check_authority.json").exists()
            write_check_authority(
                run_dir, tuple(self.runtime.config.trusted_checks()),
                default_check_ids=tuple(self.runtime.config.required_check_ids()),
            )
            if new_check_authority:
                store.update_metadata(per_step_check_ids=list(self.runtime.config.gate.per_step))
            plan_sha = persist_iteration_plan(run_dir, iteration, plan)
            _bundle, _bundle_sha = validate_implementation_bundle(
                iteration_plan_dir(run_dir, iteration), expected_step_ids=[step.id for step in plan.steps],
            )
            plan_identity = compute_plan_identity_from_run(run_dir, iteration=iteration)
        except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"invalid v2 plan artifacts: {exc}") from exc
        store.update_metadata(plan_identity=asdict(plan_identity))
        self.runtime.write_checkpoint(
            run_dir, ResumePhase.PLAN_APPROVAL, head=base_sha, plan_sha256=plan_sha,
        )

        if self.runtime.config.approval.require_plan_approval:
            store.update_metadata()
            approval = wait_for_plan_approval(
                run_dir, identity=plan_identity,
                poll_interval_seconds=self.runtime.config.approval.poll_interval_seconds,
                iteration=iteration,
            )
            if approval.decision is ApprovalDecision.REJECT:
                state = store.set_run_state(RunMachineState(
                    disposition=RunDisposition.WAIT_HUMAN, reason=PLAN_REJECTED_REASON,
                ))
                return RunResult.of(run_dir, state)
            try:
                selection, execution_sha = read_execution_selection_with_sha256(run_dir, iteration=iteration)
                validate_execution_selection(self.runtime.config, selection)
                durable_identity = compute_plan_identity_from_run(run_dir, iteration=iteration)
                if durable_identity.execution_sha256 != execution_sha:
                    raise ApprovalError("execution selection hash mismatch")
                read = compute_plan_identity_from_run(run_dir, iteration=iteration)
                bound = read_plan_approval(run_dir, expected_identity=read, iteration=iteration)
                if bound is None or bound.bundle_sha256 != durable_identity.bundle_sha256:
                    raise ApprovalError("v2 approval is not bound to the exact bundle")
            except (ExecutionSelectionError, ApprovalError, OSError, UnicodeError) as exc:
                raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        else:
            requested = resolve_execution_selection(
                self.runtime.config,
                planner_profile_id=planner_profile_id,
                plan_steps=plan.steps,
                audit_profile_id=self.runtime.run_options.audit_profile,
                fallback_authority=self.runtime.run_options.execution_fallbacks,
            )
            selection = ensure_execution_selection(run_dir, requested, iteration=iteration)
            durable_identity = compute_plan_identity_from_run(run_dir, iteration=iteration)

        # Re-read the complete manifest after the approval transaction.  The
        # first validation protects the approval surface; this one closes the
        # race between approval and worktree creation.  The bundle bytes must
        # still be the ones hashed at planning time and bound by the approval.
        try:
            bundle, bundle_sha = validate_implementation_bundle(
                iteration_plan_dir(run_dir, iteration), expected_step_ids=[step.id for step in plan.steps]
            )
        except (V2PlanParseError, OSError, UnicodeError) as exc:
            raise ApprovalError(f"PLAN_APPROVAL_INVALID: {exc}") from exc
        if bundle_sha != plan_identity.bundle_sha256:
            raise ApprovalError("PLAN_APPROVAL_INVALID: implementation bundle changed after planning")

        if selection.planner.profile_id != planner_profile_id:
            raise ExecutionSelectionError("execution selection planner is not the run planner")
        if selection.audit.profile_id != self.runtime.run_options.audit_profile:
            raise ExecutionSelectionError("execution selection auditor differs from the frozen run options")
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
        execution_state["audit"] = asdict(selection.audit)
        store.update_metadata(execution=execution_state,
                              plan_identity=asdict(durable_identity))
        self.runtime.observability.trace_emit(
            "plan.approved",
            phase="planning",
            cycle=iteration,
            data={
                "plan_identity": asdict(durable_identity),
                "execution_selection_sha256": durable_identity.execution_sha256,
            },
            once=True,
        )
        base_tree_sha = resolve_tree(repo, base_sha)
        self.runtime.write_checkpoint(
            run_dir, ResumePhase.WORKTREE_SETUP, head=base_sha, plan_sha256=plan_sha,
        )
        # Plan approval complete: the next operation is the first step.
        checkpoint = self._initial_checkpoint(plan, base_sha, plan_sha, iteration=iteration)

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
            cycle=iteration,
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
            self.runtime.config, run_dir, requested_check_ids=plan.required_checks,
            expected_sha256=durable_identity.checks_sha256,
        )
        selected_check_ids = tuple(check_ids or ())
        skipped_checks = self.runtime.gates.run_check_preflights_recoverably(
            store=store, run_dir=run_dir, worktree=info.worktree,
            check_config=check_config, check_ids=selected_check_ids, phase="preparing",
        )
        store.update_metadata(skipped_checks=list(skipped_checks))
        # The baseline of the base commit comes before the first implementation
        # step: no later gate ever has to guess whether a failure is new.
        baseline_ids = tuple(dict.fromkeys(
            (*selected_check_ids, *per_step_check_ids(self.runtime, run_dir))
        ))
        baseline = BaselineCache(self.runtime.config.runs_root).ensure(
            repo=info.worktree, base_sha=info.base_sha, config=check_config,
            check_ids=baseline_ids, environment=self.runtime.environment,
            setup_commands=self.runtime.config.workspace_setup,
            secrets=self.runtime.secrets,
            skipped={check_id: "PREFLIGHT_FAILED" for check_id in skipped_checks},
        )
        store.update_metadata(baseline=baseline_payload(baseline))
        # Worktree and setup complete: still the first step.
        if checkpoint is not None:
            write_checkpoint(run_dir, checkpoint)
        return PreparedV2Run(plan, bundle, selection, info, ownership_before, base_tree_sha, checkpoint)

    @staticmethod
    def _initial_checkpoint(
        plan: TaskPlanV2, base_sha: str, plan_sha256: str, *, iteration: int = 1,
    ) -> ResumeCheckpoint | None:
        if not plan.steps:
            return None
        return ResumeCheckpoint(
            phase=ResumePhase.IMPLEMENT_STEP, iteration=iteration, step_index=0,
            last_green_commit=base_sha, plan_sha256=plan_sha256,
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
            if checkpoint.last_green_commit is not None and checkpoint.last_green_commit != base_sha:
                refuse("the pre-worktree checkpoint does not match the run base commit")
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
                                             head=base_sha)
                checkpoint = ResumeCheckpoint(
                    phase=ResumePhase.PLANNER, last_green_commit=base_sha,
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
                # A resumed initial planner replays its paid answer against the
                # policy the run froze, not against a TOML edited meanwhile.
                check_catalog, default_check_ids = self._check_policy(run_dir)
                planner = PlannerV2(
                    self.runtime.planner_client
                    or self.runtime.chat(build_llm_endpoint(planner_profile)),
                    repository_reference=reference, planning=self.runtime.config.planning,
                    check_catalog=check_catalog,
                    default_check_ids=default_check_ids,
                    prompt_budget_bytes=self.runtime.config.prompt_budget.planner_max_bytes,
                    attempt_budget=self.runtime.run_options.budget.step_attempts,
                    repository_preconditions=RepositoryPreconditions(repo, base_tree),
                    on_event=lambda name, data: self.runtime.observability.trace_emit(name, phase="planning", cycle=checkpoint.iteration, data=data),
                )
                if (exhausted := self.runtime.budget_exhausted(store)) is not None:
                    raise BudgetExhausted(exhausted)
                plan = planner.plan(spec, context, artifacts_dir=run_dir, iteration=checkpoint.iteration)
                persist_planner_conversation(run_dir, getattr(planner, "last_conversation", None))
            else:
                if checkpoint.plan_sha256 is None:
                    refuse("checkpoint has no effective plan hash")
                try:
                    plan = read_iteration_plan(run_dir, checkpoint.iteration, checkpoint.plan_sha256)
                except V2PlanParseError as exc:
                    refuse(f"effective plan is invalid: {exc}")
            if checkpoint.phase in {ResumePhase.PLAN_APPROVAL, ResumePhase.WORKTREE_SETUP}:
                try:
                    _bundle, bundle_sha = validate_implementation_bundle(
                        iteration_plan_dir(run_dir, checkpoint.iteration),
                        expected_step_ids=[step.id for step in plan.steps]
                    )
                    identity = compute_plan_identity_from_run(run_dir, iteration=checkpoint.iteration)
                except (ApprovalError, V2PlanParseError, OSError, UnicodeError) as exc:
                    refuse(f"plan artifacts are invalid: {exc}")
                recorded_identity = state.get("plan_identity")
                if isinstance(recorded_identity, Mapping):
                    try:
                        stored_identity = PlanIdentity(
                            raw_sha256=recorded_identity["raw_sha256"],
                            contract_sha256=recorded_identity["contract_sha256"],
                            bundle_sha256=recorded_identity.get("bundle_sha256"),
                            execution_sha256=recorded_identity.get("execution_sha256"),
                            checks_sha256=recorded_identity.get("checks_sha256"),
                        )
                        if identity != stored_identity:
                            refuse("plan identity no longer matches state")
                    except (KeyError, TypeError, ApprovalError) as exc:
                        refuse(str(exc))
                if checkpoint.phase is ResumePhase.WORKTREE_SETUP:
                    try:
                        _selection, _selection_sha = read_execution_selection_with_sha256(
                            run_dir, iteration=checkpoint.iteration,
                        )
                        validate_execution_selection(self.runtime.config, _selection)
                    except (ExecutionSelectionError, ProfileError, OSError, UnicodeError) as exc:
                        refuse(f"execution selection is invalid: {exc}")
                    if self.runtime.config.approval.require_plan_approval:
                        try:
                            approval = read_plan_approval(
                                run_dir, expected_identity=identity, iteration=checkpoint.iteration,
                            )
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
                iteration=checkpoint.iteration,
            )
            if isinstance(prepared, RunResult):
                return prepared
            outcome = self.runtime.composition.execute_v2(
                store, run_dir, run_id, spec, repo, base_sha, context, reference,
                prepared=prepared,
            )
            if isinstance(outcome, RunResult):
                return outcome
            # The prepared run hands back its pipeline and the boundary it
            # entered: the resumed execution is driven here, exactly as a new
            # run drives it, so a successful resume returns a RunResult.
            pipeline, start = outcome
            return self.runtime.run_pipeline(store, pipeline, start, resumed=True)
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
