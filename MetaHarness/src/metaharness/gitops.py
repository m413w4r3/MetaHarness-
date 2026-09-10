"""Primitives Git strictes utilisées par MetaHarness V0."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from subprocess import CompletedProcess


class GitError(RuntimeError):
    """Raised when a Git operation cannot be completed safely."""


@dataclass(frozen=True)
class WorktreeInfo:
    source_repo: Path
    worktree: Path
    branch: str
    base_ref: str
    base_sha: str


def _git(repo: Path, *args: str, timeout: int = 60) -> CompletedProcess[str]:
    """Run one Git command without invoking a shell."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise GitError(f"git command failed: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise GitError(f"git command exited with {result.returncode}{suffix}")
    return result


def git_root(path: Path) -> Path:
    """Return the absolute top-level directory of the Git repository."""

    result = _git(path, "rev-parse", "--show-toplevel")
    root_text = result.stdout.strip()
    if not root_text:
        raise GitError("git did not return a repository root")
    return Path(root_text).resolve()


def resolve_commit(repo: Path, ref: str) -> str:
    """Resolve *ref* to a complete commit object ID."""

    if not isinstance(ref, str) or not ref.strip():
        raise GitError("commit reference must be non-empty")
    result = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    commit_sha = result.stdout.strip()
    if not commit_sha:
        raise GitError(f"could not resolve commit reference {ref!r}")
    return commit_sha


def current_head(repo: Path) -> str:
    """Return the complete commit object ID currently checked out."""

    return resolve_commit(repo, "HEAD")


def status_porcelain(repo: Path) -> tuple[str, ...]:
    """Return Git's porcelain status lines, including all untracked files."""

    result = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return tuple(result.stdout.splitlines())


def assert_clean(repo: Path) -> None:
    """Fail unless the repository has no staged, unstaged, or untracked files."""

    status = status_porcelain(repo)
    if status:
        raise GitError("repository is not clean")


def branch_exists(repo: Path, branch: str) -> bool:
    """Return whether *branch* exists as a local branch."""

    git_root(repo)
    try:
        _git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    except GitError:
        return False
    return True


def create_run_worktree(
    repo: Path,
    *,
    base_ref: str,
    branch: str,
    worktree_path: Path,
    require_clean_base: bool,
) -> WorktreeInfo:
    """Create an isolated run worktree at the exact resolved base commit."""

    source_repo = git_root(repo)
    if require_clean_base:
        assert_clean(source_repo)

    base_sha = resolve_commit(source_repo, base_ref)
    worktree = worktree_path.expanduser().resolve()
    if os.path.lexists(worktree):
        raise GitError(f"worktree path already exists: {worktree}")
    if branch_exists(source_repo, branch):
        raise GitError(f"local branch already exists: {branch}")

    try:
        worktree.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GitError(f"could not create worktree parent: {exc}") from exc

    _git(
        source_repo,
        "worktree",
        "add",
        "-b",
        branch,
        str(worktree),
        base_sha,
    )

    if current_head(worktree) != base_sha:
        raise GitError("new worktree HEAD does not match the resolved base commit")

    return WorktreeInfo(
        source_repo=source_repo,
        worktree=worktree,
        branch=branch,
        base_ref=base_ref,
        base_sha=base_sha,
    )


def assert_agent_did_not_commit(info: WorktreeInfo) -> None:
    """Fail if the agent moved the run worktree away from its base commit."""

    if current_head(info.worktree) != info.base_sha:
        raise GitError("agent committed or otherwise moved worktree HEAD")


def stage_all(worktree: Path) -> None:
    """Stage additions, modifications, deletions, and renames."""

    _git(worktree, "add", "--all")


def staged_diff(worktree: Path) -> str:
    """Return the complete staged diff."""

    return _git(worktree, "diff", "--cached", "--binary").stdout


def staged_changed_files(worktree: Path) -> tuple[str, ...]:
    """Return staged paths, safely preserving spaces and other characters."""

    output = _git(worktree, "diff", "--cached", "--name-only", "-z").stdout
    return tuple(path for path in output.split("\0") if path)


def index_tree_sha(worktree: Path) -> str:
    """Return the tree object ID represented by the current index."""

    tree_sha = _git(worktree, "write-tree").stdout.strip()
    if not tree_sha:
        raise GitError("git write-tree returned no tree object ID")
    return tree_sha


def commit_staged(worktree: Path, *, subject: str, body: str) -> str:
    """Create one commit from the index and return its complete object ID."""

    if not staged_changed_files(worktree):
        raise GitError("no staged changes to commit")

    clean_subject = subject.strip()
    if not clean_subject:
        raise GitError("commit subject must be non-empty")
    if len(clean_subject) > 72:
        raise GitError("commit subject must be at most 72 characters")

    args = ["commit", "-m", clean_subject]
    if body:
        args.extend(("-m", body))
    _git(worktree, *args)
    return current_head(worktree)


def _validate_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise GitError("relative path must be non-empty")
    if "\x00" in relative_path:
        raise GitError("relative path must not contain NUL")

    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise GitError("relative path must not be absolute")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise GitError("relative path must not contain ..")
    return relative_path


def read_file_at_commit(
    repo: Path,
    *,
    commit_sha: str,
    relative_path: str,
) -> str:
    """Read one repository file from an exact commit object."""

    path = _validate_relative_path(relative_path)
    if not isinstance(commit_sha, str) or not commit_sha or "\x00" in commit_sha:
        raise GitError("commit SHA must be non-empty and must not contain NUL")
    if commit_sha.startswith("-"):
        raise GitError("commit SHA must not start with -")

    return _git(repo, "show", f"{commit_sha}:{path}").stdout
