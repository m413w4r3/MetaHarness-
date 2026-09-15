"""Primitives Git strictes utilisées par MetaHarness V0."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class StagedBlob:
    """One regular blob currently present in the staged tree."""

    path: str
    object_id: str
    size: int


@dataclass(frozen=True)
class RepositoryReference:
    remote_name: str
    web_url: str | None
    base_sha: str
    immutable_url: str | None


@dataclass(frozen=True)
class PushResult:
    remote: str
    branch: str
    commit_sha: str
    status: str = "pushed"


@dataclass(frozen=True)
class RunBranchCleanupResult:
    remote: str
    branch: str
    commit_sha: str
    status: str


_RUN_BRANCH = re.compile(
    r"harness/[A-Za-z0-9][A-Za-z0-9_.-]{0,59}/[A-Za-z0-9][A-Za-z0-9_.-]*\Z"
)


def validate_run_branch(branch: str, *, base_ref: str | None = None) -> str:
    """Validate the only branch namespace MetaHarness may publish."""

    if not isinstance(branch, str) or _RUN_BRANCH.fullmatch(branch) is None:
        raise GitError("run branch is outside the MetaHarness namespace")
    forbidden = {"main", "master", "tag"}
    if base_ref:
        forbidden.add(base_ref)
    if branch in forbidden:
        raise GitError("run branch is a protected branch name")
    return branch


def run_branch_web_url(reference: RepositoryReference, branch: str) -> str | None:
    """Return a credential-free GitHub branch URL when the repository is known."""

    validate_run_branch(branch)
    if reference.web_url is None or normalize_github_web_url(reference.web_url) is None:
        return None
    # Every component was validated by _RUN_BRANCH; "/" separates components
    # and GitHub expects it literally in /tree/<branch>.
    return f"{reference.web_url}/tree/{urllib.parse.quote(branch, safe='/')}"


def _git(
    repo: Path,
    *args: str,
    timeout: int = 60,
    env: dict[str, str] | None = None,
    errors: str = "strict",
    input: str | None = None,
) -> CompletedProcess[str]:
    """Run one Git command without invoking a shell."""

    stdin_kwargs: dict[str, object] = (
        {"input": input} if input is not None else {"stdin": subprocess.DEVNULL}
    )
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            encoding="utf-8",
            errors=errors,
            capture_output=True,
            timeout=timeout,
            shell=False,
            env=env,
            **stdin_kwargs,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise GitError(f"git command failed: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise GitError(f"git command exited with {result.returncode}{suffix}")
    return result


def repository_remote_url(repo: Path, remote_name: str) -> str:
    """Return a configured remote URL using argv-only Git invocation."""

    if (
        not isinstance(remote_name, str)
        or not remote_name.strip()
        or "\x00" in remote_name
        or remote_name.startswith("-")
        or any(char.isspace() for char in remote_name)
    ):
        raise GitError("remote name is invalid")
    result = _git(repo, "remote", "get-url", "--", remote_name)
    url = result.stdout.strip()
    if not url:
        raise GitError(f"remote {remote_name!r} has no URL")
    return url


def _github_path(path: str) -> str | None:
    if path.startswith("/"):
        path = path[1:]
    if path.endswith("/") or "//" in path:
        return None
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        return None
    if any(char in path for char in "?#\\"):
        return None
    return f"https://github.com/{parts[0]}/{parts[1]}"


def normalize_github_web_url(remote_url: str) -> str | None:
    """Normalize one of the supported GitHub transport URL forms."""

    if not isinstance(remote_url, str) or not remote_url.strip():
        return None
    value = remote_url.strip()
    if "?" in value or "#" in value:
        raise ValueError("repository URL must not contain query or fragment")

    scp = re.fullmatch(r"git@github\.com:([^:]+)", value, re.IGNORECASE)
    if scp:
        return _github_path(scp.group(1))

    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("repository URL is invalid") from exc
    if parsed.username is not None or parsed.password is not None:
        if not (parsed.scheme.lower() == "ssh" and parsed.username == "git" and parsed.password is None):
            raise ValueError("repository URL contains credentials")
    if parsed.scheme.lower() not in {"ssh", "https"}:
        return None
    if hostname is None or hostname.lower() != "github.com" or port is not None:
        return None
    if parsed.scheme.lower() == "ssh" and parsed.username != "git":
        raise ValueError("repository URL contains credentials")
    return _github_path(parsed.path)


def _validate_explicit_web_url(web_url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(web_url)
        hostname = parsed.hostname
        port = parsed.port
    except (AttributeError, ValueError) as exc:
        raise ValueError("repository web_url is invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
    ):
        raise ValueError("repository web_url must be HTTPS without credentials, query, or fragment")
    if parsed.path.startswith("/") and parsed.path.endswith("/"):
        raise ValueError("repository web_url path is ambiguous")
    return web_url


def build_repository_reference(
    repo: Path,
    *,
    base_sha: str,
    config: "RepositoryConfig",
) -> RepositoryReference:
    """Build the secret-free repository identity visible to planners."""

    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_sha):
        raise GitError("base SHA is invalid")
    explicit = config.web_url
    if not config.planner_remote_exploration:
        web_url = None
    elif explicit is not None:
        # Validate that the configured remote exists even when its display URL
        # is explicitly overridden.
        remote_web_url = normalize_github_web_url(repository_remote_url(repo, config.remote))
        web_url = _validate_explicit_web_url(explicit)
        # A remote that is itself a GitHub URL is the local repository's
        # identity: the planner must never be sent to a different repository.
        if remote_web_url is not None and (
            normalize_github_web_url(web_url) or web_url
        ) != remote_web_url:
            raise ValueError("repository.web_url does not match configured Git remote")
    elif config.planner_remote_exploration:
        raw = repository_remote_url(repo, config.remote)
        web_url = normalize_github_web_url(raw)
    else:
        web_url = None
    github_url = normalize_github_web_url(web_url) if web_url is not None else None
    if web_url is not None:
        hostname = urllib.parse.urlsplit(web_url).hostname
        if hostname and hostname.lower() == "github.com" and github_url is None:
            raise ValueError("repository web_url path is ambiguous")
        if github_url is not None:
            web_url = github_url
    immutable_url = f"{github_url}/tree/{base_sha}" if github_url is not None else None
    return RepositoryReference(config.remote, web_url, base_sha, immutable_url)


def render_repository_reference(reference: RepositoryReference) -> str:
    """Render only the non-secret repository identity for a planner prompt."""

    if not isinstance(reference, RepositoryReference):
        raise TypeError("reference must be a RepositoryReference")
    exploration = "ALLOWED" if reference.web_url and reference.immutable_url else "UNAVAILABLE"
    return "\n".join(
        (
            "WEB URL:",
            reference.web_url or "UNAVAILABLE",
            "",
            "BASE SHA:",
            reference.base_sha,
            "",
            "IMMUTABLE BASE URL:",
            reference.immutable_url or "UNAVAILABLE",
            "",
            "REMOTE EXPLORATION:",
            exploration,
        )
    )


def repository_reference_dict(reference: RepositoryReference) -> dict[str, str | None]:
    """Return the exact secret-free artifact shape."""

    return asdict(reference)


def _git_bytes(repo: Path, *args: str, timeout: int = 60) -> bytes:
    """Run Git and return stdout without decoding blob contents."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise GitError(f"git command failed: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise GitError(f"git command exited with {result.returncode}{suffix}")
    return result.stdout


@dataclass(frozen=True)
class StagedChange:
    """One path changed between HEAD and the index (``git diff --cached``).

    ``path`` is decoded exactly like :func:`staged_changed_files`.  ``mode``
    and ``object_id`` describe the staged (new) side; a deletion has
    ``deleted=True`` and no staged object.
    """

    path: str
    mode: str
    object_id: str | None
    deleted: bool

    @property
    def is_gitlink(self) -> bool:
        return self.mode == "160000"


_ZERO_OBJECT_ID = re.compile(r"0{40}|0{64}")


def staged_changes(worktree: Path) -> tuple[StagedChange, ...]:
    """Return the staged changes vs HEAD with their new index mode and object.

    One ``git diff --cached --raw`` call: the cost is proportional to the
    number of changed paths, not to the size of the index.  ``-z`` keeps
    spaces, quotes and non-ASCII names verbatim; renames are split into a
    deletion and an addition like :func:`staged_changed_files`.
    """

    output = _git(
        worktree,
        "diff",
        "--cached",
        "--raw",
        "-z",
        "--no-renames",
        "--no-abbrev",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        timeout=600,
        errors="replace",
    ).stdout
    records = output.split("\0")
    if records and records[-1] == "":
        records.pop()
    if len(records) % 2:
        raise GitError("git diff returned malformed raw output")
    changes: list[StagedChange] = []
    for metadata, path in zip(records[0::2], records[1::2]):
        fields = metadata.split(" ")
        if len(fields) != 5 or not fields[0].startswith(":") or not path:
            raise GitError("git diff returned malformed raw metadata")
        _old_mode, new_mode, _old_id, new_id, status = fields
        if _OBJECT_ID.fullmatch(new_id) is None:
            raise GitError("git diff returned an invalid object ID")
        if status == "D":
            changes.append(StagedChange(path=path, mode=new_mode, object_id=None, deleted=True))
            continue
        if _ZERO_OBJECT_ID.fullmatch(new_id) is not None:
            # Unmerged or otherwise unstaged content has no scannable object.
            raise GitError(f"staged path has no object: {path}")
        changes.append(StagedChange(path=path, mode=new_mode, object_id=new_id, deleted=False))
    return tuple(changes)


def staged_submodule_paths(worktree: Path) -> frozenset[str]:
    """Return changed staged gitlink paths, whose objects are commits, not blobs."""

    return frozenset(change.path for change in staged_changes(worktree) if change.is_gitlink)


def _blob_sizes(worktree: Path, object_ids: tuple[str, ...]) -> dict[str, int]:
    """Size every object with one ``git cat-file --batch-check`` process."""

    if not object_ids:
        return {}
    unique = tuple(dict.fromkeys(object_ids))
    output = _git(
        worktree,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input="".join(f"{object_id}\n" for object_id in unique),
        timeout=600,
    ).stdout
    lines = output.splitlines()
    if len(lines) != len(unique):
        raise GitError("git cat-file returned an unexpected number of records")
    sizes: dict[str, int] = {}
    for expected, line in zip(unique, lines):
        fields = line.split(" ")
        if len(fields) != 3 or fields[0] != expected:
            raise GitError("staged blob object is missing or malformed")
        _object_id, object_type, size_text = fields
        if object_type != "blob":
            raise GitError("staged object is not a blob")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise GitError("git cat-file returned an invalid blob size") from exc
        if size < 0:
            raise GitError("git cat-file returned a negative blob size")
        sizes[expected] = size
    return sizes


def staged_changed_blobs(
    worktree: Path, changes: tuple[StagedChange, ...] | None = None
) -> tuple[StagedBlob, ...]:
    """Describe the staged blobs of changed paths only, without reading them.

    Deletions and gitlinks have no staged blob.  Regular files, executables
    and symlinks (whose blob is the link target) are all returned.
    """

    if changes is None:
        changes = staged_changes(worktree)
    candidates = [
        change
        for change in changes
        if not change.deleted and not change.is_gitlink and change.object_id is not None
    ]
    sizes = _blob_sizes(worktree, tuple(change.object_id for change in candidates))
    return tuple(
        StagedBlob(path=change.path, object_id=change.object_id, size=sizes[change.object_id])
        for change in candidates
    )


def read_staged_blob(worktree: Path, object_id: str) -> bytes:
    """Read one validated Git blob as raw bytes."""

    if _OBJECT_ID.fullmatch(object_id) is None:
        raise GitError("staged blob object ID is invalid")
    return _git_bytes(worktree, "cat-file", "blob", object_id, timeout=600)


def staged_binary_files(worktree: Path) -> frozenset[str]:
    """Return staged paths represented as binary by Git's diff machinery."""

    output = _git(
        worktree,
        "diff",
        "--cached",
        "--numstat",
        "--no-renames",
        "-z",
        errors="surrogateescape",
    ).stdout
    paths: set[str] = set()
    for record in output.split("\0"):
        if not record:
            continue
        added, separator, rest = record.partition("\t")
        if not separator:
            raise GitError("git diff returned malformed numstat output")
        deleted, separator2, path = rest.partition("\t")
        if not separator2:
            raise GitError("git diff returned malformed numstat output")
        if added == "-" and deleted == "-":
            paths.add(path)
    return frozenset(paths)


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


def resolve_tree(repo: Path, ref: str) -> str:
    """Resolve a commit-ish to its tree object ID."""

    if not isinstance(ref, str) or not ref.strip() or ref.startswith("-") or "\x00" in ref:
        raise GitError("tree reference is invalid")
    result = _git(repo, "rev-parse", "--verify", f"{ref}^{{tree}}")
    tree_sha = result.stdout.strip()
    if not tree_sha:
        raise GitError("git did not return a tree object ID")
    return tree_sha


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


def _require_object_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
        raise GitError(f"{label} must be a complete object ID")
    return value


def path_exists_in_tree(repo: Path, tree_sha: str, path: str) -> bool:
    """Return whether *path* names an entry of the tree object *tree_sha*.

    One ``git ls-tree`` restricted to that literal path: the repository is
    never walked and the working tree is never consulted.
    """

    tree = _require_object_id(tree_sha, "tree_sha")
    relative = _validate_relative_path(path)
    output = _git(
        repo,
        "--literal-pathspecs",
        "ls-tree",
        "-z",
        "--full-tree",
        tree,
        "--",
        relative,
        errors="surrogateescape",
    ).stdout
    for record in output.split("\0"):
        if not record:
            continue
        _metadata, separator, entry_path = record.partition("\t")
        if not separator:
            raise GitError("git ls-tree returned malformed output")
        if entry_path == relative:
            return True
    return False


def changed_paths_between_trees(
    repo: Path, before_tree: str, after_tree: str
) -> tuple[str, ...]:
    """Return every path that differs between two tree objects, sorted.

    Renames are split into a deletion and an addition, so both sides are
    reported.  Undecodable bytes are kept as surrogates: such a path can
    never equal an authorized path.
    """

    before = _require_object_id(before_tree, "before_tree")
    after = _require_object_id(after_tree, "after_tree")
    output = _git(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        before,
        after,
        "--",
        timeout=600,
        errors="surrogateescape",
    ).stdout
    return tuple(sorted({path for path in output.split("\0") if path}))


def commit_candidate_tree(
    worktree: Path,
    *,
    tree_sha: str,
    parent_sha: str,
    subject: str,
    body: str,
) -> str:
    """Commit exactly *tree_sha* on top of *parent_sha* and return the commit.

    This is the candidate-commit primitive of MetaHarness.  It never reads
    the index: the commit object is built from the candidate tree identity, so a
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
        raise GitError("candidate tree object does not exist")
    if _git(worktree, "rev-parse", "--verify", f"{parent_sha}^{{tree}}").stdout.strip() == tree_sha:
        raise GitError("no changes to commit for candidate tree")

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
        "metaharness: commit candidate tree",
        "HEAD",
        commit_sha,
        parent_sha,
    )
    if current_head(worktree) != commit_sha:
        raise GitError("HEAD does not point to the harness commit")
    committed_tree = _git(worktree, "rev-parse", "--verify", f"{commit_sha}^{{tree}}").stdout.strip()
    if committed_tree != tree_sha:
        raise GitError("committed tree differs from the candidate tree")
    return commit_sha


# Compatibility for integrations that imported the pre-P40 name.  New code
# must use the candidate terminology because this operation precedes review.
commit_reviewed_tree = commit_candidate_tree


def push_run_branch(
    worktree: Path,
    *,
    remote: str,
    branch: str,
    commit_sha: str,
) -> PushResult:
    """Push exactly the current harness run branch, without force or tags."""

    repository_remote_url(worktree, remote)
    validate_run_branch(branch)
    _require_object_id(commit_sha, "commit_sha")
    if symbolic_head(worktree) != f"refs/heads/{branch}":
        raise GitError("current branch does not match the run branch")
    if current_head(worktree) != commit_sha:
        raise GitError("HEAD does not match the commit to publish")
    args = [
        "push",
        "--porcelain",
        "--set-upstream",
        remote,
        f"HEAD:refs/heads/{branch}",
    ]
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise GitError(f"git command failed: {type(exc).__name__}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise GitError(f"git push exited with {result.returncode}{suffix}")
    return PushResult(remote=remote, branch=branch, commit_sha=commit_sha)


def remote_run_branch_tip(repo: Path, *, remote: str, branch: str) -> str | None:
    """Read the exact remote run-branch tip without changing local refs."""

    repository_remote_url(repo, remote)
    validate_run_branch(branch)
    output = _git(repo, "ls-remote", "--heads", remote, f"refs/heads/{branch}").stdout
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == f"refs/heads/{branch}":
            return _require_object_id(fields[0], "remote commit_sha")
    return None


def delete_run_branch(
    repo: Path,
    *,
    remote: str,
    branch: str,
    expected_commit_sha: str,
    base_ref: str | None = None,
) -> RunBranchCleanupResult:
    """Delete exactly one remote run branch, without force or a target branch.

    The remote branch must still point to the published candidate.  This
    compare-before-delete check avoids deleting a branch that was recreated or
    moved after publication.  A missing branch is a successful idempotent
    cleanup result.
    """

    repository_remote_url(repo, remote)
    validate_run_branch(branch, base_ref=base_ref)
    _require_object_id(expected_commit_sha, "expected_commit_sha")
    tip = remote_run_branch_tip(repo, remote=remote, branch=branch)
    if tip is None:
        return RunBranchCleanupResult(remote, branch, expected_commit_sha, "already_absent")
    if tip != expected_commit_sha:
        raise GitError("remote run branch no longer points to the published candidate")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "push", "--porcelain", "--delete", remote, branch],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise GitError(f"git command failed: {type(exc).__name__}") from exc
    if result.returncode != 0:
        # Transport diagnostics may contain credential-bearing URLs.
        raise GitError(f"git push --delete exited with {result.returncode}")
    if remote_run_branch_tip(repo, remote=remote, branch=branch) is not None:
        raise GitError("remote run branch was not deleted")
    return RunBranchCleanupResult(remote, branch, expected_commit_sha, "success")


class BaseMovedError(GitError):
    """The base branch moved since the run started.

    MetaHarness never merges, rebases or forces: the run stays unpublished.
    """

    code = "BASE_MOVED_SINCE_RUN"


class BasePushError(GitError):
    """The exact reviewed commit could not be pushed to the remote base."""

    def __init__(self, message: str, *, local_base_updated: bool) -> None:
        super().__init__(message)
        self.local_base_updated = local_base_updated


@dataclass(frozen=True)
class FastForwardResult:
    remote: str
    base_branch: str
    commit_sha: str
    local_base_updated: bool
    pushed: bool
    # Worktrees (typically the user's checkout) that have the base branch
    # checked out.  Their index and files are deliberately not touched.
    base_checked_out_in: tuple[str, ...]


def validate_base_branch(repo: Path, branch: str) -> str:
    """Validate a plain local branch name usable as ``refs/heads/<branch>``."""

    if (
        not isinstance(branch, str)
        or not branch
        or branch == "HEAD"
        or branch.startswith(("-", "refs/"))
        or "\x00" in branch
        or any(char.isspace() for char in branch)
    ):
        raise GitError("base branch name is invalid")
    _git(repo, "check-ref-format", "--branch", branch)
    return branch


def _ref_commit(repo: Path, ref: str) -> str | None:
    try:
        value = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").stdout.strip()
    except GitError:
        return None
    return value or None


def commit_parents(repo: Path, commit_sha: str) -> tuple[str, ...]:
    """The exact parent list of one commit object."""

    commit = _require_object_id(commit_sha, "commit_sha")
    fields = _git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.split()
    if not fields or fields[0] != commit:
        raise GitError("commit object is unreadable")
    return tuple(fields[1:])


def branch_checkouts(repo: Path, branch: str) -> tuple[str, ...]:
    """Paths of every registered worktree that has *branch* checked out."""

    output = _git(repo, "worktree", "list", "--porcelain").stdout
    paths: list[str] = []
    current: str | None = None
    for line in output.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree "):]
        elif line == f"branch refs/heads/{branch}" and current is not None:
            paths.append(current)
    return tuple(paths)


def publish_fast_forward_base(
    repo: Path,
    *,
    remote: str,
    base_branch: str,
    base_sha: str,
    commit_sha: str,
    approved_tree: str,
    run_branch: str,
    expected_parent: str | None = None,
) -> FastForwardResult:
    """Fast-forward ``refs/heads/<base>`` to the reviewed commit, then push it.

    No checkout, merge, rebase, force, lease, tag or delete, and no implicit
    fetch: the remote state is the local remote-tracking ref.  Preconditions:
    local base == remote-tracking base == ``base_sha``; the commit's only
    parent is *expected_parent* (``base_sha`` when omitted, the historical
    contract); an explicit *expected_parent* other than the base must itself
    have ``base_sha`` as its only parent (BASE -> C01 -> C02); the commit's
    tree is ``approved_tree``; the run branch points to it.  The local ref
    moves with a compare-and-swap ``update-ref``; a failed swap is
    :class:`BaseMovedError`.  A retry after a failed push accepts a local
    base that already points to the commit.
    """

    repository_remote_url(repo, remote)
    validate_base_branch(repo, base_branch)
    validate_run_branch(run_branch, base_ref=base_branch)
    for label, value in (
        ("base_sha", base_sha), ("commit_sha", commit_sha), ("approved_tree", approved_tree),
    ):
        _require_object_id(value, label)
    parent = base_sha if expected_parent is None else _require_object_id(expected_parent, "expected_parent")
    local_ref = f"refs/heads/{base_branch}"
    local = _ref_commit(repo, local_ref)
    if local is None:
        raise GitError("local base branch does not exist")
    tracking = _ref_commit(repo, f"refs/remotes/{remote}/{base_branch}")
    if tracking is None:
        raise GitError("remote-tracking base branch is unavailable")
    if commit_parents(repo, commit_sha) != (parent,):
        raise GitError(
            "run commit parent is not the run base" if parent == base_sha
            else "run commit parent is not the expected candidate parent"
        )
    if parent != base_sha and commit_parents(repo, parent) != (base_sha,):
        raise GitError("expected candidate parent is not a direct child of the run base")
    if resolve_tree(repo, commit_sha) != approved_tree:
        raise GitError("run commit tree is not the approved tree")
    if _ref_commit(repo, f"refs/heads/{run_branch}") != commit_sha:
        raise GitError("run branch does not point to the run commit")
    already_local = local == commit_sha
    if not already_local and local != base_sha:
        raise BaseMovedError("local base branch moved since the run started")
    checkouts = branch_checkouts(repo, base_branch)
    if already_local and tracking == commit_sha:
        # A previous attempt completed the push; nothing left to publish.
        return FastForwardResult(remote, base_branch, commit_sha, True, False, checkouts)
    if tracking != base_sha:
        raise BaseMovedError("remote base branch moved since the run started")
    if not already_local:
        try:
            _git(
                repo, "update-ref", "-m",
                "metaharness: fast-forward base to the reviewed commit",
                local_ref, commit_sha, base_sha,
            )
        except GitError:
            raise BaseMovedError("local base branch moved since the run started") from None
        if _ref_commit(repo, local_ref) != commit_sha:
            raise BaseMovedError("local base branch moved since the run started")
    args = ["push", "--porcelain", remote, f"{commit_sha}:{local_ref}"]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise BasePushError(
            f"git command failed: {type(exc).__name__}", local_base_updated=True
        ) from exc
    if result.returncode != 0:
        # Transport diagnostics may contain credential-bearing URLs.
        raise BasePushError(
            f"git push exited with {result.returncode}", local_base_updated=True
        )
    return FastForwardResult(remote, base_branch, commit_sha, True, True, checkouts)


def restore_paths_from_tree(worktree: Path, tree_sha: str, paths: tuple[str, ...] | list[str]) -> None:
    """Restore exactly *paths* (index and files) to their state in *tree_sha*.

    Bounded by construction: only the listed repo-relative paths are touched.
    A path present in the tree is restored from the tree object; a path
    absent from it (a file created later) is removed.  Nothing else in the
    worktree is reset.
    """

    tree = _require_object_id(tree_sha, "tree_sha")
    root = Path(worktree).expanduser().resolve()
    checked = [_validate_relative_path(path) for path in paths]
    present = [path for path in checked if path_exists_in_tree(root, tree, path)]
    absent = [path for path in checked if path not in present]
    for path in absent:
        target = root / path
        parent = target.parent.resolve()
        if parent != root and root not in parent.parents:
            raise GitError("restore path escapes the worktree")
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif os.path.lexists(target):
            raise GitError("restore path is not a file")
    if absent:
        _git(root, "--literal-pathspecs", "rm", "-q", "--cached", "--ignore-unmatch", "--", *absent, timeout=600)
    if present:
        _git(
            root, "--literal-pathspecs", "restore", f"--source={tree}",
            "--staged", "--worktree", "--", *present, timeout=600,
        )


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
