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

import hashlib, os, re, uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn
from ..approval import (
    ApprovalError, PlanIdentity, compute_plan_identity_from_run,
    write_check_authority,
)
from ..gitops import (
    GitError, candidate_tree_sha, git_root, index_tree_sha, local_branches,
    registered_worktrees,
    resolve_commit, resolve_tree, restore_paths_from_tree, stage_all,
    status_porcelain,
)
from ..integrations.github import GitHubWorkstreamClient, NullGitHubWorkstreamClient
from ..models import (
    GateStage, HarnessConfig, PlanDecision, RunCycle, RunDisposition,
    RunMachineState, RunPhase,
)
from ..plan_recovery import (
    PLAN_SOURCE_OPERATOR, PlanRecoveryError, plan_recovery_info,
    recoverable_plan_failure,
    validate_replacement_text, write_plan_recovery_record,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    validate_plan_repository_topology,
)
from ..planning.artifacts import persist_recovered_plan_artifacts, validate_implementation_bundle
from ..planning.protocol import V2PlanParseError, parse_task_plan_v2
from ..planning.validation import validate_decomposition_policy, validate_execution_mode_policy
from ..profiles import ProfileError, profiles_for_config
from ..redaction import redact
from ..resume import (
    ResumeCheckpoint, ResumeCheckpointError, ResumePhase,
    ResumeRequiresOperatorError,
    read_checkpoint, read_checkpoint_record, write_checkpoint,
    run_identity,
)
from ..run_options import (
    EffectiveRepairScopePolicy, RunOptions, RunOptionsError,
    effective_run_config,
    read_run_options_for_state,
)
from ..state import RunStateStore
from ..trace import TraceSink, TraceStream
from ..validation import ValidationError
from .check_recovery import CheckInfrastructureRecovery
from .contract_recovery import ContractRecoveryService
from .gates import GateService
from .publication import PublicationService
from .recovery import RecoveryCoordinator
from .resume_validation import ResumedRun, read_repository_reference
from .review_recovery import ReviewRecovery
from .review_service import ReviewService
from .run_bootstrap import RunBootstrap
from .run_composition import RunComposition
from .run_failure import RunFailure
from .run_observability import RunObservability
from .shared import (
    OrchestrationError, RECOVERY_ATTEMPT_ARTIFACTS, archive_attempt,
    archive_attempt_target,
    is_object_id, status_has_unstaged_or_untracked,
)
from .step_acceptance import StepAcceptanceService
from .step_execution import StepExecutionService
from .step_replan import StepReplanService
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
        # The step services of this run: the step execution ladder, the one
        # worker attempt it drives, the durable acceptance boundary, the
        # semantic contract repair transaction and the red-gate step replan.
        self.step_execution = StepExecutionService(self)
        self.worker_attempt = WorkerAttemptService(self)
        self.step_acceptance = StepAcceptanceService(self)
        self.contract_recovery = ContractRecoveryService(self)
        self.step_replan = StepReplanService(self)
        self.gates = GateService(self)
        self.reviews = ReviewService(self)
        self.publication = PublicationService(self)

        # The four run authorities; each one owns its own module and reads the
        # runtime's frozen facts and live dependencies through this reference.
        self.observability = RunObservability(self)
        self.failure = RunFailure(self)
        self.composition = RunComposition(self)
        self.bootstrap = RunBootstrap(self)

    def recovery(self, store: RunStateStore) -> RecoveryCoordinator:
        """The recovery coordinator bound to this run's durable state."""

        return RecoveryCoordinator(store, emit=self.observability.trace_emit)

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
                and not status_has_unstaged_or_untracked(status_porcelain(worktree))
            )
        except GitError:
            restored = False
        if not restored:
            raise ResumeRequiresOperatorError(
                "the checkpoint tree could not be restored exactly"
            )

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
        if not is_object_id(base_sha):
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
            reference = read_repository_reference(run_dir)
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
        if not recoverable_plan_failure(state):
            refuse("run is not in an exact recoverable planner state")
        # The recovered plan replaces a durable failure at the same boundary:
        # the claim observes the canonical identity and moves no machine state.
        claimed = store.update_metadata(
            expected=run_identity(state, run_dir, checkpoint),
            plan_recovery={"status": "persisting", "replacement_raw_sha256": replacement_sha},
        )
        if claimed is None:
            refuse("run state changed while the plan recovery was validated")
        try:
            raw_path = run_dir / "planner.raw.md"
            previous_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None
            archived = archive_attempt(run_dir, names=RECOVERY_ATTEMPT_ARTIFACTS)
            if (run_dir / "steps").is_dir():
                # Only planning-time contracts can be there (eligibility):
                # retire them with the plan they belong to.
                if archived is None:
                    archived = archive_attempt_target(run_dir)
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
            store.update_metadata(
                plan_recovery={"status": "failed", "replacement_raw_sha256": replacement_sha,
                               "detail": redact(str(exc), self._secrets_or_empty())},
            )
            raise
        planner_state = state.get("planner") if isinstance(state.get("planner"), dict) else {}
        failed = state.get("failure") if isinstance(state.get("failure"), Mapping) else {}
        store.set_run_state(
            RunMachineState(RunPhase.PLAN_APPROVAL, RunDisposition.FAILED, failed.get("reason")),
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
