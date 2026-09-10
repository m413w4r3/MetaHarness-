"""Primitives Git strictes utilisées par MetaHarness V0."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from subprocess import CompletedProcess

_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class GitError(RuntimeError):
    """Raised when a Git operation cannot be completed safely."""


@dataclass(frozen=True)
class WorktreeInfo:
    source_repo: Path
    worktree: Path
    branch: str
    base_ref: str
    base_sha: str


def _git(
    repo: Path,
    *args: str,
    timeout: int = 60,
    env: dict[str, str] | None = None,
    errors: str = "strict",
) -> CompletedProcess[str]:
    """Run one Git command without invoking a shell."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            encoding="utf-8",
            errors=errors,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            shell=False,
            env=env,
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
    if ref.startswith("-") or "\x00" in ref:
        raise GitError("commit reference must not start with - or contain NUL")
    result = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    commit_sha = result.stdout.strip()
    if not commit_sha:
        raise GitError(f"could not resolve commit reference {ref!r}")
    return commit_sha


def current_head(repo: Path) -> str:
    """Return the complete commit object ID currently checked out."""

    return resolve_commit(repo, "HEAD")


def symbolic_head(repo: Path) -> str | None:
    """Return the ref HEAD points to (``refs/heads/...``), or None if detached."""

    try:
        value = _git(repo, "symbolic-ref", "-q", "HEAD").stdout.strip()
    except GitError:
        return None
    return value or None


def local_branches(repo: Path) -> frozenset[str]:
    """Return every local branch ref name of the repository."""

    output = _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads").stdout
    return frozenset(line for line in output.splitlines() if line)


def registered_worktrees(repo: Path) -> frozenset[str]:
    """Return the paths of every worktree registered in the repository."""

    output = _git(repo, "worktree", "list", "--porcelain").stdout
    return frozenset(
        line[len("worktree ") :] for line in output.splitlines() if line.startswith("worktree ")
    )


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
        timeout=600,
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

    _git(worktree, "add", "--all", timeout=600)


def staged_diff(worktree: Path) -> str:
    """Return the complete staged diff, independent of user diff settings.

    External diff drivers, textconv filters and colors are disabled so the
    reviewer sees the raw content change.  Undecodable bytes are replaced in
    this review text only; the staged tree SHA stays the commit identity.
    """

    return _git(
        worktree,
        "diff",
        "--cached",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        timeout=600,
        errors="replace",
    ).stdout


def staged_changed_files(worktree: Path) -> tuple[str, ...]:
    """Return staged paths, safely preserving spaces and other characters.

    Rename detection is disabled so both sides of a rename are listed.
    """

    output = _git(
        worktree, "diff", "--cached", "--name-only", "--no-renames", "-z", errors="replace"
    ).stdout
    return tuple(path for path in output.split("\0") if path)


def index_tree_sha(worktree: Path) -> str:
    """Return the tree object ID represented by the current index."""

    tree_sha = _git(worktree, "write-tree").stdout.strip()
    if not tree_sha:
        raise GitError("git write-tree returned no tree object ID")
    return tree_sha


def candidate_tree_sha(worktree: Path) -> str:
    """Return the tree ``git add --all`` would record, without touching the index.

    A copy of the worktree index is updated through ``GIT_INDEX_FILE``.  The
    result covers tracked changes, deletions, mode changes and untracked
    non-ignored files, i.e. exactly the candidate submitted for review.
    """

    index_path = Path(
        _git(worktree, "rev-parse", "--path-format=absolute", "--git-path", "index").stdout.strip()
    )
    with tempfile.TemporaryDirectory(prefix="metaharness-index-") as directory:
        temporary_index = Path(directory) / "index"
        if index_path.is_file():
            shutil.copyfile(index_path, temporary_index)
        env = {**os.environ, "GIT_INDEX_FILE": str(temporary_index)}
        _git(worktree, "add", "--all", env=env, timeout=600)
        tree_sha = _git(worktree, "write-tree", env=env).stdout.strip()
    if not tree_sha:
        raise GitError("git write-tree returned no tree object ID")
    return tree_sha


def commit_reviewed_tree(
    worktree: Path,
    *,
    tree_sha: str,
    parent_sha: str,
    subject: str,
    body: str,
) -> str:
    """Commit exactly *tree_sha* on top of *parent_sha* and return the commit.

    This is the only commit primitive of MetaHarness.  It never reads the
    index: the commit object is built from the reviewed tree identity, so a
    late index change cannot enter it.  ``git commit-tree`` runs no commit
    hook that could restage content.  The branch is then advanced with a
    compare-and-swap ``update-ref``: if HEAD is no longer *parent_sha*, Git
    refuses the update and no commit becomes reachable.
    """

    for label, value in (("tree_sha", tree_sha), ("parent_sha", parent_sha)):
        if not isinstance(value, str) or not _OBJECT_ID.fullmatch(value):
            raise GitError(f"{label} must be a complete object ID")
    clean_subject = subject.strip() if isinstance(subject, str) else ""
    if not clean_subject:
        raise GitError("commit subject must be non-empty")
    if "\n" in clean_subject or len(clean_subject) > 72:
        raise GitError("commit subject must be one line of at most 72 characters")
    if _git(worktree, "cat-file", "-t", tree_sha).stdout.strip() != "tree":
        raise GitError("reviewed tree object does not exist")
    if _git(worktree, "rev-parse", "--verify", f"{parent_sha}^{{tree}}").stdout.strip() == tree_sha:
        raise GitError("no changes to commit")

    args = ["commit-tree", tree_sha, "-p", parent_sha, "-m", clean_subject]
    if body:
        args.extend(("-m", body))
    commit_sha = _git(worktree, *args).stdout.strip()
    if not _OBJECT_ID.fullmatch(commit_sha):
        raise GitError("git commit-tree returned no commit object ID")
    _git(
        worktree,
        "update-ref",
        "-m",
        "metaharness: commit reviewed tree",
        "HEAD",
        commit_sha,
        parent_sha,
    )
    if current_head(worktree) != commit_sha:
        raise GitError("HEAD does not point to the harness commit")
    committed_tree = _git(worktree, "rev-parse", "--verify", f"{commit_sha}^{{tree}}").stdout.strip()
    if committed_tree != tree_sha:
        raise GitError("committed tree differs from the reviewed tree")
    return commit_sha


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

    # ``cat-file blob`` returns raw blob bytes only: a directory (tree) is an
    # error instead of a listing, and no textconv/filter is applied.
    return _git(repo, "cat-file", "blob", f"{commit_sha}:{path}").stdout
