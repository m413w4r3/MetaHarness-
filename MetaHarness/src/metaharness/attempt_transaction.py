"""The Git transaction around one process that may touch the candidate.

Every recoverable attempt follows the same mechanical boundary::

    snapshot -> process -> inspect ownership/tree -> changed paths
             -> enforce scope -> security scan -> exact rollback -> prove

This module owns that boundary and nothing else.  It never decides whether a
failure is retried, falls back or waits: the recovery coordinator does.  An
:class:`AttemptViolation` is always an authority failure of the attempt.

Trusted processes (workspace setup, check preflights, deterministic checks)
use :func:`observe_side_effects` and :func:`restore_exact`; untrusted workers
(implementer, semantic reviser, check repair) use
:class:`CandidateAttemptTransaction`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Collection, Sequence

from .gitops import (
    CandidateState,
    GitError,
    candidate_ownership_matches,
    candidate_state_changed_paths,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    local_branches,
    registered_worktrees,
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


def git_ownership(repo: Path, worktree: Path) -> GitOwnership:
    return GitOwnership(
        head_ref=symbolic_head(worktree),
        head=current_head(worktree),
        branches=local_branches(repo),
        worktrees=registered_worktrees(repo),
    )


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


__all__ = [
    "AttemptBoundary", "AttemptRollback", "AttemptViolation",
    "CandidateAttemptTransaction", "GitOwnership", "SideEffect",
    "MAX_REPORTED_PATHS", "contain_trusted_process", "git_ownership",
    "observe_side_effects", "ownership_violations", "paths_detail", "restore_exact",
    "safe_path_label",
    "status_has_unstaged_or_untracked",
]
