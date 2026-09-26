"""The resume integrity gate and the proofs it derives from durable artifacts.

:func:`validate_resume` is the single fail-closed gate in front of every
resumed execution checkpoint: it rebuilds a :class:`ResumedRun` from persisted
artifacts only, proves the checkpoint against Git, the durable cycle bindings
and the approved scope, and derives the bounded restore path of a failed
attempt.  It never calls a model or a check, and never writes Git.
"""

from __future__ import annotations

import dataclasses
import hashlib

from pathlib import Path
from typing import (
    Any,
    Mapping,
    NoReturn,
)
from .check_scope import gate_mutable_authority
from .cycle_loader import (
    check_replan_binding,
    load_correction_plan,
    read_cycle_record,
    semantic_revision_scope,
    validate_correction_bindings,
    verify_correction_scope,
)
from .durable_readers import (
    accepted_review,
    candidate_evidence,
    load_evidence,
    read_candidate_record,
    read_repository_reference,
)
from .pipeline_v2 import (
    check_repair_attempt_dir,
    cycle_record_path,
    final_gate_stage,
    gate_acceptance_path,
    gate_dir,
    implementation_steps_dir,
    pre_semantic_gate_stage,
    review_dir,
    semantic_revision_dir,
    step_dir,
)
from .shared import (
    is_object_id,
    read_json_artifact,
    read_tree_file,
)
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    compute_plan_identity_from_run,
    read_check_authority,
    read_plan_approval,
    read_scope_approval,
)
from ..attempt_transaction import status_has_unstaged_or_untracked
from ..evidence import required_checks_passed
from ..execution_selection import (
    ExecutionSelectionError,
    read_execution_selection_with_sha256,
    validate_execution_selection,
)
from ..gitops import (
    GitError,
    RepositoryReference,
    WorktreeInfo,
    branch_exists,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_parents,
    current_head,
    git_root,
    index_tree_sha,
    is_ancestor,
    registered_worktrees,
    remote_run_branch_tip,
    resolve_commit,
    resolve_tree,
    status_porcelain,
    symbolic_head,
)
from ..models import (
    CycleKind,
    ExecutionSelection,
    GateStage,
    HarnessConfig,
    PlanDecision,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    TaskPlanV2,
)
from ..planning.artifacts import validate_implementation_bundle
from ..planning.check_replan import (
    PLAN_ARTIFACT as CHECK_REPLAN_PLAN_ARTIFACT,
    check_replan_dir,
    plan_identity,
)
from ..planning.protocol import V2PlanParseError, parse_task_plan_v2
from ..profiles import ProfileError
from ..resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeIntegrityError,
    ResumePhase,
    ResumeRequiresOperatorError,
    plan_identity_from_mapping,
)
from ..run_options import EffectiveRepairScopePolicy
from ..validation import ValidationError, config_with_check_authority, resolve_check_cwd


@dataclasses.dataclass(frozen=True)
class ResumedRun:
    """Everything a resume needs, rebuilt from persisted artifacts only."""

    checkpoint: ResumeCheckpoint
    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: ExecutionSelection
    info: WorktreeInfo
    repository_reference: RepositoryReference
    spec: str
    context: str
    base_tree_sha: str
    restore_paths: tuple[str, ...] = ()


# -- the resume integrity gate --------------------------------------------------

_STEP_PHASES = frozenset({ResumePhase.IMPLEMENT_STEP, ResumePhase.REVIEW_IMPLEMENTATION})
_CANDIDATE_PHASES = frozenset({
    ResumePhase.CANDIDATE_PUSH, ResumePhase.FINAL_REVIEW, ResumePhase.PUBLISH,
})


def _refuse(message: str) -> NoReturn:
    raise ResumeIntegrityError(message)


def _plan_scope(plan: TaskPlanV2) -> set[str]:
    return {
        path for step in plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }


def _contract_repair_scope(
    run_dir: Path, number: int, policy: EffectiveRepairScopePolicy,
) -> set[str]:
    """Read durable mutable paths approved by step-contract repairs."""

    scope: set[str] = set()
    root = implementation_steps_dir(run_dir, number)
    if not root.is_dir():
        return scope
    for step_root in root.iterdir():
        repairs = step_root / "contract_repairs"
        if not repairs.is_dir():
            continue
        for repair in repairs.iterdir():
            validation = read_json_artifact(repair / "validation.json", 64 * 1024)
            contract = repair / "contract.md"
            if not isinstance(validation, dict) or validation.get("status") != "validated":
                continue
            added = validation.get("added_mutable_paths")
            digest = validation.get("repaired_contract_sha256")
            if (
                not isinstance(added, list) or any(
                    not isinstance(path, str) or not path or path.startswith("/")
                    or ".." in Path(path).parts for path in added
                )
                or not isinstance(digest, str) or not contract.is_file()
            ):
                raise ResumeIntegrityError("step contract repair scope artifact is malformed")
            try:
                actual = hashlib.sha256(contract.read_bytes()).hexdigest()
            except OSError as exc:
                raise ResumeIntegrityError("step contract repair contract is unreadable") from exc
            if actual != digest:
                raise ResumeIntegrityError("step contract repair contract hash changed")
            if added:
                if policy.policy == "deny-expansion":
                    raise ResumeIntegrityError("step contract repair scope expansion is denied")
                if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
                    delta_path = repair / "scope_delta.json"
                    delta = read_json_artifact(delta_path, 64 * 1024)
                    if not isinstance(delta, dict) or delta.get("added_paths") != added:
                        raise ResumeIntegrityError("step contract repair scope delta is malformed")
                    try:
                        delta_sha = hashlib.sha256(delta_path.read_bytes()).hexdigest()
                        approval = read_scope_approval(repair, expected_sha256=delta_sha)
                    except (OSError, ApprovalError) as exc:
                        raise ResumeIntegrityError("step contract repair scope approval is invalid") from exc
                    if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                        raise ResumeIntegrityError("step contract repair scope was not approved")
            scope.update(added)
    return scope


def _validate_gate_acceptance(
    run_dir: Path, number: int, stage: Any, *, tree: str | None, head: str | None,
    base_scope: tuple[str, ...], policy: EffectiveRepairScopePolicy,
) -> None:
    """Require the durable accepted state for a completed gate boundary."""

    payload = read_json_artifact(gate_acceptance_path(run_dir, number, stage))
    evidence_path = gate_dir(run_dir, number, stage) / "evidence.json"
    try:
        evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    except OSError:
        evidence_sha256 = None
    authority = gate_mutable_authority(
        run_dir, number, stage,
        base_paths=base_scope,
        policy_config=policy,
        require_attempt_records=True,
    )
    no_change = payload.get("no_change", False) if isinstance(payload, dict) else False
    parent_sha = payload.get("parent_sha") if isinstance(payload, dict) else None
    evidence = load_evidence(gate_dir(run_dir, number, stage))
    parent_valid = is_object_id(parent_sha) or (
        no_change is True
        and parent_sha is None
        and evidence is not None
        and not evidence.changed_files
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 2
        or payload.get("review_cycle") != number
        or payload.get("stage") != getattr(stage, "value", stage)
        or not is_object_id(payload.get("tree_sha"))
        or not is_object_id(payload.get("commit_sha"))
        or not isinstance(no_change, bool)
        or not parent_valid
        or (
            evidence is not None
            and no_change is not (not evidence.changed_files)
        )
        or payload.get("tree_sha") != tree
        or payload.get("commit_sha") != head
        or payload.get("acceptance_kind") not in {"existing-head", "repair", "semantic-revision"}
        or not isinstance(payload.get("commit_created"), bool)
        or (
            no_change
            and (
                parent_sha is not None
                or payload.get("commit_created") is not False
                or payload.get("acceptance_kind") != "existing-head"
            )
        )
        or payload.get("mutable_scope") != list(authority.effective_paths)
        or payload.get("mutable_scope_sha256") != authority.sha256
        or (
            no_change
            and (
                evidence is None or evidence.diff != ""
                or not evidence.deterministic_passed
                or not required_checks_passed(evidence)
            )
        )
        or payload.get("evidence_sha256") != evidence_sha256
    ):
        _refuse("the gate acceptance artifact is missing or invalid")


def _approved_scope(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path,
    plan: TaskPlanV2, checkpoint: ResumeCheckpoint, policy: EffectiveRepairScopePolicy,
) -> set[str]:
    """Every path any durable authority of cycles ``1..n`` approved."""

    scope = set(_plan_scope(plan))
    scope |= _contract_repair_scope(run_dir, 1, policy)
    cycle_scopes: dict[int, tuple[str, ...]] = {1: tuple(sorted(scope))}
    cycle_base_scopes: dict[int, tuple[str, ...]] = {1: tuple(sorted(scope))}
    cycle_kinds: dict[int, CycleKind] = {1: CycleKind.INITIAL}
    cycle_semantic_scopes: dict[int, set[str]] = {
        1: semantic_revision_scope(config.repo, run_dir, 1, policy),
    }
    scope |= cycle_semantic_scopes[1]
    for number in range(2, checkpoint.review_cycle + 1):
        cycle = read_cycle_record(run_dir, number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            # Direct semantic corrections reuse the preceding approved plan,
            # while their final gate may still have its own check-repair
            # authority.
            cycle_semantic_scopes[number] = semantic_revision_scope(
                config.repo, run_dir, number, policy,
            )
            scope |= cycle_semantic_scopes[number]
            cycle_base_scopes[number] = tuple(sorted(set(cycle_scopes[number - 1])))
            cycle_scopes[number] = tuple(sorted(
                set(cycle_base_scopes[number]) | cycle_semantic_scopes[number]
            ))
            cycle_kinds[number] = cycle.kind
            continue
        if number == checkpoint.review_cycle and checkpoint.phase is ResumePhase.REVIEW_REPLAN:
            # The correction plan of this cycle is not an authority yet: it
            # may still await its scope approval.
            continue
        correction, _bundle, bundle_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        verify_correction_scope(run_dir, number, bundle_sha, policy)
        cycle_base_scopes[number] = tuple(sorted(_plan_scope(correction)))
        cycle_base_scopes[number] = tuple(sorted(
            set(cycle_base_scopes[number]) | _contract_repair_scope(run_dir, number, policy)
        ))
        semantic_added = semantic_revision_scope(config.repo, run_dir, number, policy)
        cycle_semantic_scopes[number] = semantic_added
        cycle_scopes[number] = tuple(sorted(set(cycle_base_scopes[number]) | semantic_added))
        cycle_kinds[number] = cycle.kind
        scope |= _plan_scope(correction)
        scope |= _contract_repair_scope(run_dir, number, policy)
        scope |= semantic_added
    for number, base in cycle_base_scopes.items():
        kind = cycle_kinds[number]
        stages = (
            (final_gate_stage(kind),)
            if kind is CycleKind.REVIEW_IMPLEMENTATION
            else (pre_semantic_gate_stage(kind), final_gate_stage(kind))
        )
        for stage in dict.fromkeys(stages):
            stage_scope = set(base)
            if stage == final_gate_stage(kind):
                stage_scope |= cycle_semantic_scopes.get(number, set())
            authority = gate_mutable_authority(
                run_dir, number, stage,
                base_paths=stage_scope,
                policy_config=policy,
            )
            scope |= set(authority.effective_paths)
    return scope


def _cycle_base_scope(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path,
    plan: TaskPlanV2, number: int, policy: EffectiveRepairScopePolicy,
    *, include_current_semantic: bool = True,
) -> tuple[str, ...]:
    """Return the approved plan scope which is the base for one cycle."""

    current = tuple(sorted(
        set(_plan_scope(plan)) | _contract_repair_scope(run_dir, 1, policy)
    ))
    if number > 1 or include_current_semantic:
        current = tuple(sorted(
            set(current) | semantic_revision_scope(config.repo, run_dir, 1, policy)
        ))
    for cycle_number in range(2, number + 1):
        cycle = read_cycle_record(run_dir, cycle_number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            if cycle_number != number or include_current_semantic:
                current = tuple(sorted(
                    set(current) | semantic_revision_scope(config.repo, run_dir, cycle_number, policy)
                ))
            continue
        correction, _bundle, _bundle_sha = load_correction_plan(
            config, selection, run_dir, cycle_number,
            inherited_check_ids=plan.required_checks,
        )
        current = tuple(sorted(_plan_scope(correction)))
        current = tuple(sorted(set(current) | _contract_repair_scope(run_dir, cycle_number, policy)))
        if cycle_number != number or include_current_semantic:
            current = tuple(sorted(
                set(current) | semantic_revision_scope(config.repo, run_dir, cycle_number, policy)
            ))
    return current


def _failure_tree_for(
    run_dir: Path, checkpoint: ResumeCheckpoint,
) -> str | None:
    """The tree a failed attempt at *checkpoint* durably recorded, if any."""

    number = checkpoint.review_cycle
    if checkpoint.phase in _STEP_PHASES:
        if checkpoint.phase is ResumePhase.REVIEW_IMPLEMENTATION:
            try:
                if read_cycle_record(run_dir, number).kind is CycleKind.REVIEW_IMPLEMENTATION:
                    return read_tree_file(
                        semantic_revision_dir(run_dir, number) / "tree_after_failure.txt"
                    )
            except ResumeIntegrityError:
                return None
        record = read_json_artifact(
            step_dir(run_dir, number, str(checkpoint.step_id)) / "step.json",
            128 * 1024,
        )
        if isinstance(record, dict) and record.get("status") == "FAILED" and is_object_id(record.get("tree_after")):
            return record["tree_after"]
        if (
            isinstance(record, dict) and record.get("status") == "COMPLETED"
            and is_object_id(record.get("tree_after"))
            and record.get("tree_before") == checkpoint.expected_tree_sha
            and record.get("tree_after") != record.get("tree_before")
            and record.get("commit_sha") is None
        ):
            # A worker success interrupted before its candidate became
            # durable: only an exact in-scope rollback is offered.
            return record["tree_after"]
        return None
    if checkpoint.phase is ResumePhase.SEMANTIC_REVISION:
        return read_tree_file(semantic_revision_dir(run_dir, number) / "tree_after_failure.txt")
    if checkpoint.phase is ResumePhase.CHECK_REPAIR and checkpoint.stage is not None:
        return read_tree_file(
            check_repair_attempt_dir(
                run_dir, number, checkpoint.stage, int(checkpoint.check_repair_attempt or 1),
            ) / "tree_after_failure.txt"
        )
    return None


def _replaced_by_replan(run_dir: Path, number: int) -> bool:
    """True when cycle *number* ended on a red gate that re-decomposed it."""

    try:
        following = read_cycle_record(run_dir, number + 1)
    except (ResumeIntegrityError, OSError):
        return False
    return following.kind is CycleKind.CHECK_REPLAN


def _validate_check_replan_authority(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path,
    number: int, plan: TaskPlanV2,
) -> None:
    """Prove the durable re-decomposition of cycle *number* is intact.

    The record is the one durable answer a red gate paid for; it must still
    name the cycle record's own digest, must differ from the plan it replaced
    and must bind exactly the bundle its steps read.
    """

    # The record is the durability proof: the cycle record names its digest and
    # the re-parsed answer must still render it.
    check_replan_binding(run_dir, number)
    record = read_json_artifact(
        check_replan_dir(run_dir, number) / CHECK_REPLAN_PLAN_ARTIFACT, 64 * 1024,
    )
    plan_value, _bundle, bundle_sha = load_correction_plan(
        config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
    )
    if (
        not isinstance(record, dict)
        or record.get("implementation_bundle_sha256") != bundle_sha
        or record.get("plan_identity_after") is None
        or record.get("plan_identity_after") == record.get("plan_identity_before")
        or record.get("plan_identity_after") != plan_identity(plan_value)
    ):
        _refuse("the check-replan plan changed")


def validate_resume(
    *,
    config: HarnessConfig,
    repair_scope: EffectiveRepairScopePolicy,
    run_dir: Path,
    state: Mapping[str, Any],
    checkpoint: ResumeCheckpoint,
    staging_remote: str,
) -> ResumedRun:
    """Fail-closed integrity gate in front of every resumed execution phase."""

    if state.get("planning_protocol") != "v2":
        _refuse("only pipeline v2 runs can be resumed")
    try:
        repo = git_root(config.repo)
    except GitError as exc:
        _refuse(f"repository is unavailable: {exc}")
    if str(repo) != str(state.get("repo")):
        _refuse("the configured repository is not the run repository")
    base_sha = state.get("base_sha")
    if not is_object_id(base_sha):
        _refuse("run base SHA is invalid")
    reference = read_repository_reference(run_dir)
    if reference is None or reference.base_sha != base_sha:
        _refuse("the base SHA changed for this run")
    try:
        spec = (run_dir / "spec.md").read_text(encoding="utf-8")
        context = (run_dir / "context.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _refuse("run SPEC or planner context is unreadable")

    # Plan identity, the human approval bound to it and the frozen checks.
    try:
        identity = compute_plan_identity_from_run(run_dir)
        recorded = plan_identity_from_mapping(state.get("plan_identity"))
        authority = read_check_authority(
            run_dir, expected_sha256=identity.checks_sha256,
            trusted_check_ids=tuple(check.id for check in config.trusted_checks()),
        )
        approval = read_plan_approval(run_dir, expected_identity=identity)
    except (ApprovalError, ResumeCheckpointError) as exc:
        _refuse(f"plan artifacts are invalid: {exc}")
    if identity != checkpoint.plan_identity or identity != recorded:
        _refuse("plan identity no longer matches")
    if identity.checks_sha256 is not None and authority is None:
        _refuse("check authority is missing for this run")
    if config.approval.require_plan_approval:
        if approval is None or approval.decision is not ApprovalDecision.APPROVE:
            _refuse("plan approval was not APPROVE")
        if (
            approval.execution_sha256 != checkpoint.execution_selection_sha256
            or approval.bundle_sha256 != identity.bundle_sha256
        ):
            _refuse("the approval does not bind the checkpoint execution selection")
    elif approval is not None and approval.decision is not ApprovalDecision.APPROVE:
        _refuse("plan approval artifact is not APPROVE")

    # The approved execution selection, against today's configuration.
    try:
        selection, execution_sha = read_execution_selection_with_sha256(run_dir)
        validate_execution_selection(config, selection)
    except (ExecutionSelectionError, ProfileError) as exc:
        _refuse(f"execution selection is invalid: {exc}")
    if execution_sha != checkpoint.execution_selection_sha256 or execution_sha != identity.execution_sha256:
        _refuse("execution selection hash changed")
    execution = state.get("execution") if isinstance(state.get("execution"), Mapping) else {}
    planner_state = execution.get("planner") if isinstance(execution.get("planner"), Mapping) else {}
    if selection.planner.profile_id != planner_state.get("profile_id"):
        _refuse("execution selection planner is not the run planner")

    # The approved plan and its exact bundle.
    try:
        plan = parse_task_plan_v2(
            (run_dir / "planner.raw.md").read_text(encoding="utf-8"),
            planning=config.planning,
            check_catalog=config.check_catalog,
            default_check_ids=config.default_check_ids,
        )
        bundle, bundle_sha = validate_implementation_bundle(
            run_dir, expected_step_ids=[step.id for step in plan.steps]
        )
    except (V2PlanParseError, OSError, UnicodeError) as exc:
        _refuse(f"approved plan is unreadable: {exc}")
    if plan.decision is not PlanDecision.READY or bundle_sha != identity.bundle_sha256:
        _refuse("approved bundle changed")
    if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
        _refuse("execution selection steps do not match the plan")

    # Worktree and branch.
    worktree_value, branch = state.get("worktree"), state.get("branch")
    if not isinstance(worktree_value, str) or not isinstance(branch, str):
        _refuse("run has no worktree or branch")
    worktree = Path(worktree_value).expanduser().resolve()
    if not worktree.is_dir():
        _refuse("run worktree is missing")
    try:
        frozen_config, frozen_ids = config_with_check_authority(
            config, run_dir, expected_sha256=identity.checks_sha256,
        )
        if frozen_ids is not None:
            for frozen_check in frozen_config.select_checks(frozen_ids):
                resolve_check_cwd(worktree, frozen_check)
    except (ValidationError, ValueError) as exc:
        _refuse(f"check authority is invalid: {exc}")

    # Cycle identity and every authority of cycles 1..n.
    number = checkpoint.review_cycle
    if number > 1 or cycle_record_path(run_dir, 1).exists():
        if read_cycle_record(run_dir, 1).kind is not CycleKind.INITIAL:
            _refuse("cycle 001 is not the initial cycle")
    for earlier in range(2, number + 1):
        read_cycle_record(run_dir, earlier)
    for earlier in range(1, number):
        if _replaced_by_replan(run_dir, earlier):
            continue
        read_candidate_record(run_dir, earlier)
    validate_correction_bindings(run_dir, number)
    current_cycle = read_cycle_record(run_dir, number) if number > 1 else RunCycle(1, CycleKind.INITIAL)
    scope = _approved_scope(config, selection, run_dir, plan, checkpoint, repair_scope)
    if current_cycle.kind is CycleKind.CHECK_REPLAN:
        # A cycle one red gate re-decomposed plans before it opens, so its
        # checkpoint never has to carry the plan: the durable answer does.
        _validate_check_replan_authority(config, selection, run_dir, number, plan)
    elif (
        checkpoint.phase not in {ResumePhase.REVIEW_REPLAN, *_STEP_PHASES}
        and number > 1
        and current_cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION
    ):
        if checkpoint.correction_bundle_sha256 is None:
            _refuse("the correction checkpoint is not bound to its plan")
        _correction, _bundle, correction_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        if correction_sha != checkpoint.correction_bundle_sha256:
            _refuse("the correction plan changed")
    if checkpoint.phase is ResumePhase.REVIEW_IMPLEMENTATION and current_cycle.kind is CycleKind.REVIEW_REPLAN:
        _correction, _bundle, correction_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        if correction_sha != checkpoint.correction_bundle_sha256:
            _refuse("the correction plan changed")

    restore: tuple[str, ...] = ()
    try:
        if str(worktree) not in registered_worktrees(repo):
            _refuse("run worktree is not registered in the repository")
        if not branch_exists(repo, branch):
            _refuse("run branch is missing")
        if symbolic_head(worktree) != f"refs/heads/{branch}":
            _refuse("worktree HEAD is not the run branch")
        head = current_head(worktree)
        if resolve_commit(repo, f"refs/heads/{branch}") != head:
            _refuse("run branch does not point to the worktree HEAD")
        expected_tree = checkpoint.expected_tree_sha
        if head != checkpoint.expected_head_sha:
            # A crash may land between an accepted commit and the next
            # checkpoint; only that exact commit (expected parent, expected
            # tree) is kept.  At a gate boundary it must also be the commit
            # its durable gate acceptance recorded.
            advanced = (
                commit_parents(repo, head) == (checkpoint.expected_head_sha,)
                and resolve_tree(repo, head) == expected_tree
            )
            if advanced and checkpoint.phase is ResumePhase.DETERMINISTIC_GATE and checkpoint.stage is not None:
                acceptance = read_json_artifact(
                    gate_acceptance_path(run_dir, number, checkpoint.stage)
                )
                advanced = (
                    isinstance(acceptance, dict)
                    and acceptance.get("commit_created") is True
                    and acceptance.get("commit_sha") == head
                    and acceptance.get("parent_sha") == checkpoint.expected_head_sha
                )
                if advanced:
                    _validate_gate_acceptance(
                        run_dir, number, checkpoint.stage, tree=expected_tree, head=head,
                        base_scope=_cycle_base_scope(
                            config, selection, run_dir, plan, number, repair_scope,
                            include_current_semantic=(
                                current_cycle.kind is CycleKind.REVIEW_IMPLEMENTATION
                                or checkpoint.stage != pre_semantic_gate_stage(current_cycle.kind)
                            ),
                        ),
                        policy=repair_scope,
                    )
            elif checkpoint.phase is ResumePhase.STEP_ACCEPTANCE:
                # Committed before the next boundary: the step acceptance
                # re-proves that commit against its durable candidate.
                pass
            elif checkpoint.phase is not ResumePhase.CANDIDATE_READY:
                advanced = False
            if not advanced:
                _refuse("HEAD moved since the checkpoint")
        if (
            checkpoint.expected_parent_sha is not None
            and commit_parents(repo, head) != (checkpoint.expected_parent_sha,)
        ):
            _refuse("HEAD parent differs from the checkpoint parent")
        # Every earlier reviewed candidate is a real commit with its recorded
        # tree and parent, and the run branch still descends from it.  A cycle
        # a red gate re-decomposed never reached one: its successor owns that
        # authority instead.
        for earlier in range(1, number):
            if _replaced_by_replan(run_dir, earlier):
                continue
            record = read_candidate_record(run_dir, earlier)
            earlier_parents = (
                (record["parent_sha"],) if record.get("parent_sha") is not None else ()
            )
            if (
                resolve_tree(repo, record["commit_sha"]) != record["tree_sha"]
                or (
                    record.get("no_change") is not True
                    and commit_parents(repo, record["commit_sha"]) != earlier_parents
                )
                or not is_ancestor(repo, record["commit_sha"], head)
            ):
                _refuse(f"cycle {earlier:03d} candidate record is not in the run history")

        if checkpoint.phase in _CANDIDATE_PHASES:
            candidate = read_candidate_record(run_dir, number)
            if candidate["commit_sha"] != head or candidate["tree_sha"] != expected_tree:
                _refuse("the run branch is not the recorded candidate commit")
            candidate_parents = commit_parents(repo, candidate["commit_sha"])
            expected_candidate_parents = (
                (candidate["parent_sha"],)
                if candidate.get("parent_sha") is not None else ()
            )
            evidence = candidate_evidence(run_dir, number)
            if (
                (
                    candidate.get("no_change") is not True
                    and candidate_parents != expected_candidate_parents
                )
                or (
                    candidate.get("parent_sha") is None
                    and (
                        candidate.get("no_change") is not True
                        or evidence is None
                        or bool(evidence.changed_files)
                    )
                )
                or (
                    candidate.get("no_change") is True
                    and (
                        candidate.get("parent_sha") is not None
                        or evidence is None or evidence.diff != ""
                        or evidence.base_sha != base_sha
                        or not evidence.deterministic_passed
                        or not required_checks_passed(evidence)
                    )
                )
            ):
                _refuse("the candidate parent identity is invalid")
            try:
                candidate_stage = GateStage(candidate["gate_stage"])
            except (KeyError, TypeError, ValueError):
                _refuse("the candidate gate stage is invalid")
            cycle_base_scope = _cycle_base_scope(
                config, selection, run_dir, plan, number, repair_scope,
                include_current_semantic=(
                    candidate_stage is final_gate_stage(current_cycle.kind)
                ),
            )
            _validate_gate_acceptance(
                run_dir, number, candidate_stage, tree=expected_tree, head=head,
                base_scope=cycle_base_scope, policy=repair_scope,
            )
            remote_required = config.publish.enabled or (
                config.github.enabled and config.github.pull_request_mode == "create"
            )
            if checkpoint.phase is ResumePhase.PUBLISH and remote_required:
                try:
                    remote_tip = remote_run_branch_tip(
                        repo, remote=staging_remote, branch=branch
                    )
                except (GitError, OSError):
                    _refuse("required remote candidate could not be verified")
                if remote_tip != head:
                    _refuse("remote run branch does not point to the candidate commit")
                if (
                    candidate.get("remote") != staging_remote
                    or candidate.get("remote_branch") != branch
                    or candidate.get("remote_sha") != head
                    or candidate.get("remote_status") != "available"
                    or not isinstance(candidate.get("pushed_at"), str)
                    or not candidate.get("pushed_at")
                ):
                    _refuse("candidate record does not prove the exact remote authority")
        if checkpoint.phase is ResumePhase.PUBLISH:
            evidence = candidate_evidence(run_dir, number)
            if evidence is None or evidence.staged_tree_sha != expected_tree:
                _refuse("the candidate evidence is missing or not for the approved tree")
            review = accepted_review(review_dir(run_dir, number), evidence, head)
            if review is None or review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
                _refuse("the reviewer PASS is missing for the candidate")
        if current_cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION and (
            checkpoint.phase is ResumePhase.SEMANTIC_REVISION
            or (
                checkpoint.phase in {ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR}
                and checkpoint.stage is final_gate_stage(current_cycle.kind)
            )
        ):
            # Semantic revision never precedes a green gate: its pre-semantic
            # acceptance must durably name the HEAD the revision started from.
            pre_head = checkpoint.expected_head_sha
            _validate_gate_acceptance(
                run_dir, number, pre_semantic_gate_stage(current_cycle.kind),
                tree=resolve_tree(repo, pre_head), head=pre_head,
                base_scope=_cycle_base_scope(
                    config, selection, run_dir, plan, number, repair_scope,
                    include_current_semantic=False,
                ),
                policy=repair_scope,
            )
        if checkpoint.phase is ResumePhase.CHECK_REPAIR:
            evidence = load_evidence(gate_dir(run_dir, number, checkpoint.stage))
            if evidence is None or evidence.staged_tree_sha != expected_tree:
                _refuse("the red gate evidence is missing or not for the checkpoint tree")
        if checkpoint.phase is ResumePhase.DETERMINISTIC_GATE and checkpoint.check_repair_attempt is not None:
            record = read_json_artifact(
                check_repair_attempt_dir(
                    run_dir, number, checkpoint.stage, checkpoint.check_repair_attempt,
                ) / "attempt.json"
            )
            if not isinstance(record, dict) or record.get("tree_after") != expected_tree:
                _refuse("the check-repair attempt record is not for the checkpoint tree")

        candidate_tree = candidate_tree_sha(worktree)
        index_tree = index_tree_sha(worktree)
        dirty = status_has_unstaged_or_untracked(status_porcelain(worktree))
        if candidate_tree != expected_tree or index_tree != expected_tree or dirty:
            failure_tree = _failure_tree_for(run_dir, checkpoint)
            if failure_tree is None or candidate_tree != failure_tree:
                _refuse("the worktree differs from the checkpoint tree")
            changed = changed_paths_between_trees(repo, expected_tree, candidate_tree)
            if not changed or any(path not in scope for path in changed):
                raise ResumeRequiresOperatorError(
                    "the failed attempt changed paths outside the approved scope"
                )
            restore = tuple(changed)
        base_tree = resolve_tree(repo, base_sha)
        unapproved = [
            path for path in changed_paths_between_trees(repo, base_tree, expected_tree)
            if path not in scope
        ]
    except GitError as exc:
        _refuse(f"Git state is unreadable: {exc}")
    if unapproved:
        _refuse("the candidate contains a path outside the approved scope")
    return ResumedRun(
        checkpoint=checkpoint, plan=plan, bundle=bundle, selection=selection,
        info=WorktreeInfo(
            source_repo=repo, worktree=worktree, branch=branch,
            base_ref=config.base_ref, base_sha=base_sha,
        ),
        repository_reference=reference, spec=spec, context=context,
        base_tree_sha=base_tree, restore_paths=restore,
    )


__all__ = ["ResumedRun", "validate_resume"]
