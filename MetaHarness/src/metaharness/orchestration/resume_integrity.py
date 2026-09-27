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
from .durable_readers import (
    gate_mutable_authority,
    candidate_evidence,
    completed_step_records,
    load_evidence,
    read_candidate_record,
    read_repository_reference,
)
from .pipeline_v2 import (
    cycle_record_path,
    gate_acceptance_path,
    gate_dir,
    implementation_steps_dir,
    step_dir,
)
from .shared import (
    is_object_id,
    read_json_artifact,
)
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    compute_plan_identity_from_run,
    read_check_authority,
    read_plan_approval,
)
from ..attempt_transaction import status_has_unstaged_or_untracked
from ..evidence import required_checks_passed
from ..execution_selection import (
    ExecutionSelectionError,
    read_execution_selection_with_sha256,
    validate_execution_selection,
)
from ..scope import ScopeViolation
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
    TaskPlanV2,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    validate_plan_repository_topology,
)
from ..planning.artifacts import PLAN_NORMALIZATIONS_NAME, validate_implementation_bundle
from ..planning.normalization import normalizations_payload
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

_STEP_PHASES = frozenset({ResumePhase.IMPLEMENT_STEP, ResumePhase.STEP_ACCEPTANCE})
_CANDIDATE_PHASES = frozenset({ResumePhase.CANDIDATE_PUSH, ResumePhase.PUBLISH})


def _refuse(message: str) -> NoReturn:
    raise ResumeIntegrityError(message)


def _plan_scope(plan: TaskPlanV2) -> set[str]:
    return {
        path for step in plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }


def _recorded_soft_scope(run_dir: Path, number: int, plan: TaskPlanV2) -> set[str]:
    """The out-of-scope paths durably admitted for one cycle's accepted steps."""

    scope: set[str] = set()
    for record in completed_step_records(
        run_dir, number, [step.id for step in plan.steps],
    ):
        scope.update(record.get("out_of_scope_paths") or ())
    return scope


def _contract_repair_scope(run_dir: Path, number: int) -> set[str]:
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
            scope.update(added)
    return scope


def _validate_gate_acceptance(
    run_dir: Path, number: int, stage: Any, *, tree: str | None, head: str | None,
    base_scope: tuple[str, ...],
) -> None:
    """Require the durable accepted state for a completed gate boundary."""

    payload = read_json_artifact(gate_acceptance_path(run_dir, number, stage))
    evidence_path = gate_dir(run_dir, number, stage) / "evidence.json"
    try:
        evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    except OSError:
        evidence_sha256 = None
    authority = gate_mutable_authority(run_dir, number, stage, base_paths=base_scope)
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
        or payload.get("acceptance_kind") != "existing-head"
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


def _failure_tree_for(
    run_dir: Path, checkpoint: ResumeCheckpoint,
) -> str | None:
    """The tree a failed attempt at *checkpoint* durably recorded, if any."""

    if checkpoint.phase not in _STEP_PHASES:
        return None
    record = read_json_artifact(
        step_dir(run_dir, 1, str(checkpoint.step_id)) / "step.json", 128 * 1024,
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
        # A worker success interrupted before its candidate became durable:
        # only an exact in-scope rollback is offered.
        return record["tree_after"]
    return None


def _cycle_base_scope(run_dir: Path, plan: TaskPlanV2) -> tuple[str, ...]:
    """The approved mutable scope of the run's single cycle."""

    return tuple(sorted(
        set(_plan_scope(plan)) | _contract_repair_scope(run_dir, 1)
        | _recorded_soft_scope(run_dir, 1, plan)
    ))


def _approved_scope(run_dir: Path, plan: TaskPlanV2) -> set[str]:
    """Every path a durable authority of the cycle approved."""

    scope = set(_cycle_base_scope(run_dir, plan))
    authority = gate_mutable_authority(
        run_dir, 1, GateStage.POST_IMPLEMENTATION, base_paths=tuple(scope),
    )
    scope |= set(authority.effective_paths)
    return scope


def validate_resume(
    *,
    config: HarnessConfig,
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

    # The approved plan and its exact bundle.  The run approved the *effective*
    # plan, so the very same deterministic normalization is replayed against
    # the very same immutable start tree before any scope, authority or step
    # identity is read from it.  What stays impossible after normalization is
    # an irreducible contradiction, and no resume may invent around it.
    try:
        plan = parse_task_plan_v2(
            (run_dir / "planner.raw.md").read_text(encoding="utf-8"),
            planning=config.planning,
            check_catalog=config.check_catalog,
            default_check_ids=config.default_check_ids,
        )
        plan = validate_plan_repository_topology(
            repo, resolve_tree(repo, base_sha), plan,
        )
        bundle, bundle_sha = validate_implementation_bundle(
            run_dir, expected_step_ids=[step.id for step in plan.steps]
        )
        normalizations = normalizations_payload(plan)
    except (
        V2PlanParseError, PlanRepositoryPreconditionError, GitError, OSError, UnicodeError,
    ) as exc:
        _refuse(f"approved plan is invalid: {exc}")
    if plan.decision is not PlanDecision.READY or bundle_sha != identity.bundle_sha256:
        _refuse("approved bundle changed")
    if read_json_artifact(run_dir / PLAN_NORMALIZATIONS_NAME) != normalizations:
        _refuse("the plan normalization record does not match the approved plan")
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

    # The run has one cycle; its durable record must still say exactly that.
    number = checkpoint.review_cycle
    if number != 1:
        _refuse("only the initial cycle can resume")
    if cycle_record_path(run_dir, 1).exists():
        record = read_json_artifact(cycle_record_path(run_dir, 1))
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("number") != 1
            or record.get("kind") != CycleKind.INITIAL.value
        ):
            _refuse("cycle 001 is not the initial cycle")
    scope = _approved_scope(run_dir, plan)
    for report_path in sorted((run_dir / "cycles" / f"{number:03d}" / "audit").glob("*/report.json")):
        audit = read_json_artifact(report_path)
        paths = audit.get("changed_paths") if isinstance(audit, dict) else None
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            _refuse("audit path evidence is malformed")
        try:
            scope.update(config.scope.check(paths, worktree=worktree))
        except ScopeViolation as exc:
            _refuse(exc.detail)
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
            if checkpoint.phase is ResumePhase.AUDIT:
                reports = sorted((run_dir / "cycles" / f"{number:03d}" / "audit").glob("*/report.json"))
                audit = read_json_artifact(reports[-1]) if reports else None
                if (
                    isinstance(audit, dict)
                    and audit.get("commit_sha") == head
                    and audit.get("input_tree") == expected_tree
                    and audit.get("tree_after") == resolve_tree(repo, head)
                    and commit_parents(repo, head) == (checkpoint.expected_head_sha,)
                ):
                    advanced = True
                    expected_tree = audit["tree_after"]
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
                        base_scope=_cycle_base_scope(run_dir, plan),
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
            _validate_gate_acceptance(
                run_dir, number, candidate_stage, tree=expected_tree, head=head,
                base_scope=_cycle_base_scope(run_dir, plan),
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
            if evidence.changed_files:
                reports = sorted((run_dir / "cycles" / f"{number:03d}" / "audit").glob("*/report.json"))
                audit = read_json_artifact(reports[-1]) if reports else None
                if (
                    not isinstance(audit, dict)
                    or audit.get("status") not in {"DONE", "NEEDS_WORK"}
                    or audit.get("tree_after") != expected_tree
                ):
                    _refuse("the audit is missing for the candidate")
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
