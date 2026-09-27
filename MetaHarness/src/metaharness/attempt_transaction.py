"""The Git transaction around one process that may touch the candidate.

Every recoverable attempt follows the same mechanical boundary::

    snapshot -> process -> inspect ownership/tree -> changed paths
             -> enforce scope -> security scan -> exact rollback -> prove

This module owns that boundary and nothing else.  It never decides whether a
failure is retried, falls back or waits: the recovery coordinator does.  An
:class:`AttemptViolation` is always an authority failure of the attempt.

Trusted processes (workspace setup, check preflights, deterministic checks)
use :func:`observe_side_effects` and :func:`restore_exact`; untrusted workers
(implementer) uses
:class:`CandidateAttemptTransaction`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Collection, Sequence

from .gitops import (
    CandidateState,
    GitError,
    all_refs,
    candidate_ownership_matches,
    candidate_state_changed_paths,
    candidate_tree_sha,
    changed_paths_between_trees,
    checkout_branch,
    current_head,
    delete_ref,
    index_tree_sha,
    is_ancestor,
    local_branches,
    registered_worktrees,
    reset_worktree_soft,
    restore_candidate_state,
    restore_paths_from_tree,
    snapshot_candidate_state,
    stage_all,
    status_porcelain,
    symbolic_head,
)


class AttemptViolation(Exception):
    """An attempt crossed an authority boundary; ``code`` is a stable reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclasses.dataclass(frozen=True)
class GitOwnership:
    """Git state the implementation agent is not allowed to change."""

    head_ref: str | None
    head: str
    branches: frozenset[str]
    worktrees: frozenset[str]
    # Every ref with its object ID.  It is what proves a push: a push writes
    # ``refs/remotes/**``, which no branch snapshot can see.
    refs: frozenset[tuple[str, str]] = frozenset()


def git_ownership(repo: Path, worktree: Path) -> GitOwnership:
    return GitOwnership(
        head_ref=symbolic_head(worktree),
        head=current_head(worktree),
        branches=local_branches(repo),
        worktrees=registered_worktrees(repo),
        refs=all_refs(repo),
    )


@dataclasses.dataclass(frozen=True)
class GitMutationAudit:
    """What one untrusted process did to the Git boundary of its attempt.

    ``fatal`` is the stable code and detail of a mutation the harness may not
    recover from, or ``None``.  The remaining fields describe the mutations the
    harness takes back: the commits of the run branch, a branch the worker
    created -- and possibly checked out -- and every new local ref.
    """

    fatal_code: str | None = None
    fatal_detail: str = ""
    worker_commits: bool = False
    created_branches: tuple[str, ...] = ()
    adopted_branch: str | None = None

    @property
    def recoverable(self) -> bool:
        return self.fatal_code is None and bool(
            self.worker_commits or self.created_branches or self.adopted_branch
        )


# The run branch moved forward; the harness rewinds it with ``reset --soft``.
REMOTE_AUTHORITY_MISMATCH = "REMOTE_AUTHORITY_MISMATCH"
BRANCH_MODIFIED_OUTSIDE_AUTHORITY = "BRANCH_MODIFIED_OUTSIDE_AUTHORITY"
HEAD_MODIFIED_OUTSIDE_AUTHORITY = "HEAD_MODIFIED_OUTSIDE_AUTHORITY"


def audit_git_mutation(
    before: GitOwnership, after: GitOwnership, *, branch_ref: str, base_sha: str,
    repo: Path | None = None,
) -> GitMutationAudit:
    """Classify one attempt's Git mutations: recoverable or fatal.

    A local commit on the run branch, or on a new branch the worker created and
    checked out, is recoverable: the change stays, the commit does not.  A
    push, a foreign ref, a deleted ref, a rewritten history and a worktree
    change are not.
    """

    created = tuple(sorted(after.branches - before.branches))
    deleted = tuple(sorted(before.branches - after.branches))
    if deleted:
        return GitMutationAudit(
            BRANCH_MODIFIED_OUTSIDE_AUTHORITY,
            "branch(es) deleted: " + ", ".join(deleted),
        )
    if _remote_refs(after) != _remote_refs(before):
        return GitMutationAudit(
            REMOTE_AUTHORITY_MISMATCH,
            "remote-tracking refs changed: the worker pushed or fetched",
        )
    foreign = _foreign_ref_changes(before, after, branch_ref=branch_ref, created=created)
    if foreign:
        return GitMutationAudit(
            BRANCH_MODIFIED_OUTSIDE_AUTHORITY,
            "ref(s) modified outside the run branch: " + ", ".join(foreign),
        )
    roots = {
        name for name, _value in before.refs
    }
    adopted = (
        after.head_ref
        if after.head_ref is not None
        and after.head_ref != branch_ref
        and after.head_ref in created
        and after.head_ref not in roots
        else None
    )
    if after.head_ref not in {branch_ref, adopted}:
        return GitMutationAudit(
            HEAD_MODIFIED_OUTSIDE_AUTHORITY,
            f"worktree HEAD switched to {after.head_ref or 'a detached HEAD'}",
        )
    if after.head != base_sha:
        moved = _moves_forward(before.head, after.head, base_sha, repo=repo)
        if adopted is None and after.head_ref != branch_ref:
            return GitMutationAudit(
                HEAD_MODIFIED_OUTSIDE_AUTHORITY, "worktree HEAD commit changed",
            )
        if not moved:
            return GitMutationAudit(
                HEAD_MODIFIED_OUTSIDE_AUTHORITY,
                "the run branch no longer descends from the expected commit",
            )
        return GitMutationAudit(
            worker_commits=True, created_branches=created, adopted_branch=adopted,
        )
    return GitMutationAudit(created_branches=created, adopted_branch=adopted)


def _remote_refs(ownership: GitOwnership) -> frozenset[tuple[str, str]]:
    return frozenset(
        item for item in ownership.refs if item[0].startswith("refs/remotes/")
    )


def _foreign_ref_changes(
    before: GitOwnership, after: GitOwnership, *, branch_ref: str, created: tuple[str, ...],
) -> list[str]:
    """Ref changes no recovery may absorb: everything but the run branch.

    A branch the worker created is recoverable and deleted after the fact; a
    pre-existing ref that changed is not.
    """

    previous = dict(before.refs)
    current = dict(after.refs)
    changes: list[str] = []
    for name, value in sorted(current.items()):
        if name == branch_ref or name in created or name.startswith("refs/remotes/"):
            continue
        if name not in previous:
            changes.append(f"created {name}")
        elif previous[name] != value:
            changes.append(f"moved {name}")
    for name in sorted(set(previous) - set(current)):
        if name == branch_ref or name in created or name.startswith("refs/remotes/"):
            continue
        changes.append(f"deleted {name}")
    return changes


def _moves_forward(
    previous_head: str, head: str, base_sha: str, *, repo: Path | None,
) -> bool:
    if previous_head != base_sha or head == base_sha or repo is None:
        return False
    try:
        return is_ancestor(repo, base_sha, head)
    except GitError:
        return False


def ownership_violations(
    before: GitOwnership, after: GitOwnership, *, branch_ref: str, base_sha: str
) -> list[str]:
    problems: list[str] = []
    if after.head_ref != branch_ref:
        problems.append(
            f"worktree HEAD switched from {branch_ref} to {after.head_ref or 'a detached HEAD'}"
        )
    if after.head != base_sha:
        problems.append("worktree HEAD commit changed (commit, merge, reset or rewrite)")
    created = sorted(after.branches - before.branches)
    if created:
        problems.append("branch(es) created: " + ", ".join(created))
    deleted = sorted(before.branches - after.branches)
    if deleted:
        problems.append("branch(es) deleted: " + ", ".join(deleted))
    added_worktrees = sorted(after.worktrees - before.worktrees)
    if added_worktrees:
        problems.append("worktree(s) created: " + ", ".join(added_worktrees))
    removed_worktrees = sorted(before.worktrees - after.worktrees)
    if removed_worktrees:
        problems.append("worktree(s) removed: " + ", ".join(removed_worktrees))
    return problems


def status_has_unstaged_or_untracked(status: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for line in status:
        if line.startswith("?? "):
            problems.append(f"new untracked file: {line[3:]}")
        elif len(line) >= 2 and line[1] != " ":
            problems.append(f"unstaged change: {line}")
    return problems


MAX_REPORTED_PATHS = 20


def safe_path_label(path: str) -> str:
    """A printable rendering of one repository path for failure details."""

    return "".join(
        character if character.isprintable() else f"\\x{ord(character) & 0xFF:02x}"
        for character in path
    )[:300]


def paths_detail(paths: Sequence[str]) -> str:
    shown = [safe_path_label(path) for path in paths[:MAX_REPORTED_PATHS]]
    extra = len(paths) - len(shown)
    return ",".join(shown) + (f" (+{extra} more)" if extra > 0 else "")


def recover_worker_git_state(
    repo: Path, worktree: Path, audit: GitMutationAudit, *, branch_ref: str, base_sha: str,
) -> None:
    """Take back one recoverable worker Git mutation, content preserved.

    The worker's commits are removed from the branch (`reset --soft`), HEAD is
    re-attached to the run branch, and every branch the attempt created is
    deleted.  The index and the files keep the worker's content, so the change
    stays exploitable; only the history changes.
    """

    try:
        if current_head(worktree) != base_sha:
            reset_worktree_soft(worktree, base_sha)
        if symbolic_head(worktree) != branch_ref:
            checkout_branch(worktree, branch_ref)
        for branch in audit.created_branches:
            delete_ref(repo, branch)
    except GitError as exc:
        raise AttemptViolation(
            "ROLLBACK_FAILED", f"worker Git mutation is not recoverable: {exc}",
        ) from None
    try:
        restored = (
            current_head(worktree) == base_sha
            and symbolic_head(worktree) == branch_ref
            and not (local_branches(repo) & set(audit.created_branches))
        )
    except GitError as exc:
        raise AttemptViolation(
            "ROLLBACK_FAILED", f"worker Git recovery could not be read back: {exc}",
        ) from None
    if not restored:
        raise AttemptViolation(
            "ROLLBACK_FAILED", "worker Git recovery was not proven",
        )


# -- trusted processes ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SideEffect:
    """A candidate mutation observed around one trusted process."""

    after: CandidateState
    changed_paths: tuple[str, ...]
    ownership_preserved: bool

    @property
    def signature(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
        return (
            self.after.index_tree, self.after.candidate_tree,
            self.changed_paths, self.after.status,
        )


def observe_side_effects(worktree: Path, before: CandidateState) -> SideEffect | None:
    """Compare the candidate with ``before``; ``None`` when it is unchanged."""

    after = snapshot_candidate_state(worktree)
    if after == before:
        return None
    return SideEffect(
        after=after,
        changed_paths=tuple(candidate_state_changed_paths(worktree, before, after)),
        ownership_preserved=candidate_ownership_matches(before, after),
    )


def restore_exact(worktree: Path, before: CandidateState, *, label: str) -> None:
    """Restore the candidate and index to ``before`` and prove the result."""

    try:
        restored = restore_candidate_state(worktree, before)
    except GitError as exc:
        raise AttemptViolation(
            "ROLLBACK_FAILED", f"{label} rollback failed: {type(exc).__name__}",
        ) from None
    if restored != before:
        raise AttemptViolation("ROLLBACK_TREE_MISMATCH", f"{label} rollback was not exact")


def contain_trusted_process(
    worktree: Path, before: CandidateState, *, label: str,
) -> SideEffect | None:
    """Reject ownership changes and exactly undo any candidate mutation."""

    effect = observe_side_effects(worktree, before)
    if effect is None:
        return None
    if not effect.ownership_preserved:
        raise AttemptViolation("AGENT_GIT_VIOLATION", f"{label} changed Git ownership")
    restore_exact(worktree, before, label=label)
    return effect


# -- untrusted workers ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AttemptBoundary:
    """The exact pre-attempt candidate a failed worker must be rolled back to."""

    tree: str
    status: tuple[str, ...]
    ownership: GitOwnership


@dataclasses.dataclass(frozen=True)
class AttemptRollback:
    """Proof that a failed attempt was inspected and exactly undone."""

    changed_paths: tuple[str, ...]
    tree_after: str

    @property
    def changed(self) -> bool:
        return bool(self.changed_paths)


class CandidateAttemptTransaction:
    """Inspect and exactly undo one failed untrusted worker attempt."""

    def __init__(
        self,
        repo: Path,
        worktree: Path,
        boundary: AttemptBoundary,
        *,
        branch_ref: str,
        base_sha: str | None = None,
        secrets: Sequence[str] = (),
    ) -> None:
        self.repo = Path(repo)
        self.worktree = Path(worktree)
        self.boundary = boundary
        self.branch_ref = branch_ref
        self.base_sha = base_sha or boundary.ownership.head
        self._secrets = tuple(secrets)

    @classmethod
    def begin(
        cls, repo: Path, worktree: Path, *, branch_ref: str,
        base_sha: str | None = None, secrets: Sequence[str] = (),
    ) -> "CandidateAttemptTransaction":
        """Snapshot an exact, fully staged pre-attempt boundary."""

        try:
            tree = candidate_tree_sha(worktree)
            index = index_tree_sha(worktree)
            status = status_porcelain(worktree)
            ownership = git_ownership(repo, worktree)
        except GitError as exc:
            raise AttemptViolation(
                "RESUME_REQUIRES_OPERATOR", "pre-attempt tree is unreadable",
            ) from exc
        if index != tree:
            raise AttemptViolation(
                "RESUME_REQUIRES_OPERATOR",
                "worker checkpoint does not have an exact index tree "
                f"(tree={tree}, index={index}, status={list(status)[:8]})",
            )
        return cls(
            repo, worktree, AttemptBoundary(tree, status, ownership),
            branch_ref=branch_ref, base_sha=base_sha, secrets=secrets,
        )

    def audit_ownership(self) -> None:
        try:
            violations = ownership_violations(
                self.boundary.ownership, git_ownership(self.repo, self.worktree),
                branch_ref=self.branch_ref, base_sha=self.base_sha,
            )
        except GitError as exc:
            violations = [f"Git ownership could not be read: {type(exc).__name__}"]
        if violations:
            raise AttemptViolation("AGENT_GIT_VIOLATION", "; ".join(violations))

    def freeze(self) -> tuple[str, tuple[str, ...]]:
        """Stage the failed attempt and return its tree and changed paths."""

        before = self.boundary.tree
        try:
            tree = candidate_tree_sha(self.worktree)
            index = index_tree_sha(self.worktree)
            status = status_porcelain(self.worktree)
            if tree != before or index != before or status_has_unstaged_or_untracked(status):
                stage_all(self.worktree)
                tree = candidate_tree_sha(self.worktree)
            changed = changed_paths_between_trees(self.repo, before, tree)
        except (GitError, OSError, ValueError) as exc:
            raise AttemptViolation(
                "RESUME_REQUIRES_OPERATOR",
                f"failed attempt tree could not be frozen: {type(exc).__name__}",
            ) from exc
        return tree, tuple(changed)

    @staticmethod
    def enforce_scope(changed: Sequence[str], allowed: Collection[str]) -> None:
        outside = [path for path in changed if path not in allowed]
        if outside:
            raise AttemptViolation(
                "AGENT_SCOPE_VIOLATION",
                "worker changed paths outside scope: " + paths_detail(outside),
            )

    def scan(self) -> None:
        # Imported here: evidence depends on validation, which depends on
        # this module for trusted-process containment.
        from .evidence import scan_staged_security

        try:
            failures = scan_staged_security(self.worktree, secrets=self._secrets)
        except (GitError, OSError, ValueError) as exc:
            raise AttemptViolation(
                "STAGED_BLOB_SCAN_FAILED",
                f"staged changes could not be scanned: {type(exc).__name__}",
            ) from exc
        if failures:
            raise AttemptViolation(failures[0].split(":", 1)[0], "; ".join(failures))

    def rollback(self, changed: Sequence[str]) -> None:
        """Restore ``changed`` from the boundary tree and prove exactness."""

        before = self.boundary
        try:
            if changed:
                restore_paths_from_tree(self.worktree, before.tree, sorted(changed))
                stage_all(self.worktree)
            status = status_porcelain(self.worktree)
            if (
                candidate_tree_sha(self.worktree) != before.tree
                or index_tree_sha(self.worktree) != before.tree
                or status != before.status
                or status_has_unstaged_or_untracked(status)
            ):
                raise GitError("rollback did not restore the exact tree")
        except (GitError, OSError) as exc:
            raise AttemptViolation(
                "RESUME_REQUIRES_OPERATOR", "failed attempt rollback was not exact",
            ) from exc
        self.audit_ownership()

    def abort(
        self, allowed: Collection[str], *, requested: Collection[str] = (),
    ) -> AttemptRollback:
        """Undo a failed attempt that stayed inside ``allowed`` + ``requested``."""

        self.audit_ownership()
        tree_after, changed = self.freeze()
        self.enforce_scope(changed, set(allowed) | set(requested))
        if tree_after != self.boundary.tree:
            self.scan()
        self.rollback(changed)
        return AttemptRollback(changed, tree_after)

    def discard(self) -> AttemptRollback:
        """Restore every candidate path touched by a failed worker attempt.

        Callers that use this cleanup path must separately reject any
        out-of-scope paths after rollback. Unlike :meth:`abort`, discard
        never lets a scope violation or failed security scan leave mutations
        in the candidate.
        """

        self.audit_ownership()
        tree_after, changed = self.freeze()
        self.rollback(changed)
        return AttemptRollback(changed, tree_after)


__all__ = [
    "AttemptBoundary", "AttemptRollback", "AttemptViolation",
    "BRANCH_MODIFIED_OUTSIDE_AUTHORITY", "GitMutationAudit",
    "HEAD_MODIFIED_OUTSIDE_AUTHORITY", "REMOTE_AUTHORITY_MISMATCH",
    "audit_git_mutation", "recover_worker_git_state",
    "CandidateAttemptTransaction", "GitOwnership", "SideEffect",
    "MAX_REPORTED_PATHS", "contain_trusted_process", "git_ownership",
    "observe_side_effects", "ownership_violations", "paths_detail", "restore_exact",
    "safe_path_label",
    "status_has_unstaged_or_untracked",
]
