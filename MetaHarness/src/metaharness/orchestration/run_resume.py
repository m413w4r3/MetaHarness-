"""Git-first restoration of one v2 run to its last accepted commit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..approval import ApprovalDecision, PlanIdentity, read_plan_approval
from ..execution_selection import read_execution_selection_with_sha256, validate_execution_selection
from ..gitops import (
    GitError, WorktreeInfo, all_refs, branch_exists, build_run_branch, commit_message,
    commit_parents, current_head, git_root, is_ancestor, local_branches,
    registered_worktrees, resolve_commit, resolve_tree, rewind_worktree,
    status_porcelain, symbolic_head, validate_run_branch,
)
from ..models import HarnessConfig, RunPhase
from ..planning.artifacts import implementation_bundle_payload, read_effective_plan, write_implementation_bundle
from ..profiles import ProfileError
from ..scope import ScopeViolation
from ..resume import ResumeCheckpoint, ResumeIntegrityError, write_checkpoint
from ..validation import ValidationError, config_with_check_authority, resolve_check_cwd
from .durable_readers import read_repository_reference


@dataclass(frozen=True)
class ResumedRun:
    checkpoint: ResumeCheckpoint
    plan: Any
    bundle: dict[str, Any]
    selection: Any
    info: WorktreeInfo
    repository_reference: Any
    spec: str
    context: str
    base_tree_sha: str


def _fail(message: str) -> None:
    raise ResumeIntegrityError(message)


def _check_foreign_refs(repo: Path, state: Mapping[str, Any], branch: str, head: str, remote: str) -> None:
    baseline = state.get("git_ownership")
    refs = baseline.get("refs") if isinstance(baseline, Mapping) else None
    if not isinstance(refs, list):
        _fail("run has no Git ref ownership snapshot")
    original = {item[0]: item[1] for item in refs if isinstance(item, list) and len(item) == 2}
    run_ref = f"refs/heads/{branch}"
    current = dict(all_refs(repo))
    expected = {name: value for name, value in original.items() if name != run_ref}
    observed = {name: value for name, value in current.items() if name != run_ref}
    remote_run_ref = f"refs/remotes/{remote}/{branch}"
    if remote_run_ref not in original and observed.get(remote_run_ref) == head:
        observed.pop(remote_run_ref)
    if expected != observed:
        _fail("a foreign Git ref changed during this run")
    old_branches = set(baseline.get("branches") or ())
    if set(local_branches(repo)) != old_branches:
        _fail("a foreign branch was created or removed during this run")
    if current.get(run_ref) != head:
        _fail("run branch does not point to the worktree HEAD")


def _adopt_completed_commit(
    repo: Path, checkpoint: ResumeCheckpoint, *, head: str, plan: Any, run_id: str,
) -> ResumeCheckpoint:
    if commit_parents(repo, head) != (checkpoint.last_green_commit,):
        _fail("run branch moved beyond the single current operation")
    message = commit_message(repo, head)
    if "MetaHarness-Run: " + run_id not in message:
        _fail("run branch contains a commit from another authority")
    if checkpoint.phase is RunPhase.IMPLEMENT_STEP:
        index = checkpoint.step_index
        if index is None or index >= len(plan.steps):
            _fail("checkpoint step_index is outside the effective plan")
        if not message.splitlines()[0].startswith(f"metaharness({plan.steps[index].id}):"):
            _fail("run branch advanced with a commit for another step")
        next_index = index + 1
        phase = RunPhase.IMPLEMENT_STEP if next_index < len(plan.steps) else RunPhase.DETERMINISTIC_GATE
        adopted = ResumeCheckpoint(
            phase=phase, iteration=checkpoint.iteration,
            step_index=next_index if phase is RunPhase.IMPLEMENT_STEP else None,
            last_green_commit=head, plan_sha256=checkpoint.plan_sha256,
        )
    elif checkpoint.phase is RunPhase.AUDIT and message.splitlines()[0] == "metaharness(audit): cycle 1":
        adopted = ResumeCheckpoint(
            phase=RunPhase.DETERMINISTIC_GATE, iteration=checkpoint.iteration,
            last_green_commit=head, plan_sha256=checkpoint.plan_sha256,
        )
    else:
        _fail("run branch moved since the current operation checkpoint")
    return adopted


def prepare_resume(
    *, config: HarnessConfig, run_dir: Path, run_id: str,
    state: Mapping[str, Any], checkpoint: ResumeCheckpoint,
    restore_worktree: bool = True,
) -> ResumedRun:
    """Validate Git ownership, restore last green, and load the canonical plan."""

    if checkpoint.iteration != 1:
        _fail("this runtime supports iteration 1 only")
    if checkpoint.last_green_commit is None or checkpoint.plan_sha256 is None:
        _fail("execution checkpoint is missing its Git or plan authority")
    try:
        repo = git_root(config.repo)
        if str(repo) != str(state.get("repo")):
            _fail("configured repository is not the run repository")
        base_sha = state.get("base_sha")
        if not isinstance(base_sha, str) or resolve_commit(repo, base_sha) != base_sha:
            _fail("run base commit is missing or invalid")
        green = checkpoint.last_green_commit
        if resolve_commit(repo, green) != green or not is_ancestor(repo, base_sha, green):
            _fail("last_green_commit is missing or outside the run base history")
        reference = read_repository_reference(run_dir)
        if reference is None or reference.base_sha != base_sha:
            _fail("repository reference does not match the run base commit")
        worktree_value, branch = state.get("worktree"), state.get("branch")
        if not isinstance(worktree_value, str) or not isinstance(branch, str):
            _fail("run has no worktree or branch identity")
        worktree = Path(worktree_value).expanduser().resolve()
        expected_worktree = (config.worktrees_root / run_id).expanduser().resolve()
        if worktree != expected_worktree:
            _fail("run worktree path does not match its configured identity")
        if not worktree.is_dir() or str(worktree) not in registered_worktrees(repo):
            _fail("run worktree is missing or not registered")
        branch = validate_run_branch(branch, base_ref=config.base_ref)
        if not branch_exists(repo, branch):
            _fail("run branch identity does not match this run")
        if symbolic_head(worktree) != f"refs/heads/{branch}":
            _fail("worktree HEAD is on a foreign branch")
        head = current_head(worktree)
        _check_foreign_refs(repo, state, branch, head, config.repository.remote)
        if not is_ancestor(repo, base_sha, head) or not is_ancestor(repo, green, head):
            _fail("run branch is outside the authorized commit history")
        if restore_worktree:
            dirty_paths = [line[3:] for line in status_porcelain(worktree, include_ignored=True) if len(line) >= 4]
            try:
                config.scope.check(dirty_paths, worktree=worktree)
            except ScopeViolation as exc:
                error = ResumeIntegrityError(str(exc))
                error.code = exc.code
                raise error from exc
        plan = read_effective_plan(run_dir, checkpoint.plan_sha256)
        if branch != build_run_branch(plan.title, run_id):
            _fail("run branch identity does not match the effective plan")
        if head != green:
            checkpoint = _adopt_completed_commit(repo, checkpoint, head=head, plan=plan, run_id=run_id)
            if restore_worktree:
                write_checkpoint(run_dir, checkpoint)
            green = checkpoint.last_green_commit
        base_tree_sha = resolve_tree(repo, base_sha)
        if restore_worktree:
            rewind_worktree(worktree, green)
            if current_head(worktree) != green or symbolic_head(worktree) != f"refs/heads/{branch}":
                _fail("worktree could not be restored to last_green_commit")
        bundle = (
            write_implementation_bundle(run_dir, plan)
            if restore_worktree else implementation_bundle_payload(plan)
        )
        selection, _ = read_execution_selection_with_sha256(run_dir)
        validate_execution_selection(config, selection)
        if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
            _fail("execution selection steps do not match the effective plan")
        execution = state.get("execution") if isinstance(state.get("execution"), Mapping) else {}
        planner = execution.get("planner") if isinstance(execution.get("planner"), Mapping) else {}
        audit = execution.get("audit") if isinstance(execution.get("audit"), Mapping) else {}
        step_rows = execution.get("steps") if isinstance(execution.get("steps"), list) else []
        if (
            selection.planner.profile_id != planner.get("profile_id")
            or selection.audit.profile_id != audit.get("profile_id")
            or [item.implementer.profile_id for item in selection.steps]
            != [row.get("implementer", {}).get("profile_id") for row in step_rows if isinstance(row, Mapping)]
        ):
            _fail("execution selection does not match the run's frozen profiles")
        spec = (run_dir / "spec.md").read_text(encoding="utf-8")
        context = (run_dir / "context.txt").read_text(encoding="utf-8")
        durable_identity = state.get("plan_identity")
        checks_sha = durable_identity.get("checks_sha256") if isinstance(durable_identity, Mapping) else None
        frozen_config, frozen_ids = config_with_check_authority(
            config, run_dir, expected_sha256=checks_sha,
        )
        if frozen_ids is not None:
            for check in frozen_config.select_checks(frozen_ids):
                resolve_check_cwd(worktree, check)
        if config.approval.require_plan_approval:
            if not isinstance(durable_identity, Mapping):
                _fail("run plan approval identity is missing")
            identity = PlanIdentity(
                raw_sha256=durable_identity.get("raw_sha256"),
                contract_sha256=durable_identity.get("contract_sha256"),
                bundle_sha256=durable_identity.get("bundle_sha256"),
                execution_sha256=durable_identity.get("execution_sha256"),
                checks_sha256=durable_identity.get("checks_sha256"),
            )
            approval = read_plan_approval(run_dir, expected_identity=identity)
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                _fail("plan approval is missing or is not APPROVE")
    except ResumeIntegrityError:
        raise
    except (GitError, OSError, UnicodeError, ValueError, ProfileError, ValidationError) as exc:
        _fail(f"run worktree could not be restored safely: {exc}")

    info = WorktreeInfo(repo, worktree, branch, config.base_ref, base_sha)
    return ResumedRun(
        checkpoint, plan, bundle, selection, info, reference, spec, context, base_tree_sha,
    )


__all__ = ["ResumedRun", "prepare_resume"]
