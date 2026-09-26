"""The one authority for repository path safety and forbidden paths.

A worker's directory tree is untrusted input, and two different questions are
asked about every path it produces: does the path stay inside the physical
authority of the repository, and does it name a forbidden path.  Both answers
are computed here, once, and every caller -- the worker attempt, the commit
boundary, a correction scope request -- reasons on the same
:class:`ScopeViolation` code instead of re-implementing a matcher.

The mode is a signal, not a barrier: ``soft`` admits an out-of-scope path and
records it, ``strict`` restores it.  Only a path this module refuses is fatal,
and the refusal never depends on the mode.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import re

from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Iterable,
    Sequence,
)
from .gitops import GitError, validate_repository_relative_path


SCOPE_MODES = ("soft", "strict")

# The one hard-deny list.  A mutation of one of these paths is fatal, whatever
# the scope mode is: the harness never lets a worker rewrite its own Git
# metadata, its CI workflows, its environment secrets or a file whose name
# claims to hold a secret.  Build files (Makefile, pyproject.toml, lockfiles,
# requirements) are deliberately absent: they are ordinary, audited paths.
DEFAULT_HARD_DENY_PATTERNS: tuple[str, ...] = (
    ".git/**",
    ".github/workflows/**",
    ".env",
    ".env.*",
    ".env*",
    "**/*secret*",
)

# Stable refusal codes: both are FATAL, and both already exist in the single
# failure-classification table.  ``HARD_DENY_PATH_MUTATION`` is the only code
# this work adds, and it is registered as FATAL there.
UNSAFE_PATH_MUTATION = "TREE_MODIFIED_OUTSIDE_AUTHORITY"
HARD_DENY_PATH_MUTATION = "HARD_DENY_PATH_MUTATION"

_UNSAFE_PATH_CHARS = re.compile(r"[*?\x00]")


class ScopeViolation(ValueError):
    """A path crossed the physical or the forbidden-path authority.

    ``code`` is the stable failure code the caller must project; the class is
    FATAL in the single classification table, so no caller decides it again.
    """

    def __init__(self, code: str, detail: str, paths: Sequence[str] = ()) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.paths = tuple(paths)


def normalize_repo_path(path: object) -> str:
    """One repository-relative POSIX path, or :class:`ScopeViolation`.

    Absolute paths, ``..`` escapes, Windows separators, glob metacharacters and
    empty paths are refused here; the caller projects
    :data:`UNSAFE_PATH_MUTATION`.
    """

    if not isinstance(path, str) or not path or path != path.strip():
        raise ScopeViolation(
            UNSAFE_PATH_MUTATION, "scope path must be a non-empty repository path",
        )
    if "\\" in path or _UNSAFE_PATH_CHARS.search(path):
        raise ScopeViolation(
            UNSAFE_PATH_MUTATION, "scope path must be a plain repository-relative path",
        )
    try:
        relative = validate_repository_relative_path(path)
    except GitError as exc:
        raise ScopeViolation(UNSAFE_PATH_MUTATION, str(exc)) from None
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts:
        raise ScopeViolation(UNSAFE_PATH_MUTATION, "scope path must be repository-relative")
    return posix.as_posix()


def normalize_repo_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Normalize one path list, refusing the first path that escapes."""

    return tuple(normalize_repo_path(path) for path in paths)


def assert_worktree_containment(worktree: Path, paths: Iterable[str]) -> None:
    """Refuse a path whose physical resolution leaves the worktree.

    A relative symlink that stays inside the repository is an ordinary file;
    a link -- or a parent directory link -- pointing outside it is an escape.
    """

    root = Path(worktree).resolve()
    for path in paths:
        target = root / normalize_repo_path(path)
        resolved = Path(os.path.realpath(target))
        if resolved != root and root not in resolved.parents:
            raise ScopeViolation(
                UNSAFE_PATH_MUTATION,
                f"{path} resolves outside the worktree",
                (path,),
            )


@dataclasses.dataclass(frozen=True)
class ScopePolicy:
    """The frozen scope configuration of one run.

    ``hard_deny`` holds the default list plus any configured addition, and
    ``config_path`` is the repository-relative path of the active MetaHarness
    configuration file when that file lives inside the repository.
    """

    mode: str = "soft"
    hard_deny: tuple[str, ...] = DEFAULT_HARD_DENY_PATTERNS
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in SCOPE_MODES:
            raise ValueError("scope mode must be 'soft' or 'strict'")
        if not isinstance(self.hard_deny, tuple) or any(
            not isinstance(pattern, str) or not pattern or pattern != pattern.strip()
            for pattern in self.hard_deny
        ):
            raise ValueError("scope hard_deny must be an array of path patterns")
        if self.config_path is not None:
            normalize_repo_path(self.config_path)
        if not self.hard_deny and self.config_path is None:
            raise ValueError("scope hard_deny may not be empty")

    @property
    def strict(self) -> bool:
        """Whether an admitted out-of-scope path is restored instead of kept."""

        return self.mode == "strict"

    def patterns(self) -> tuple[str, ...]:
        """Every refusal pattern, the active configuration file included."""

        if self.config_path is None:
            return self.hard_deny
        return (*self.hard_deny, self.config_path)

    def is_hard_denied(self, path: str) -> bool:
        """Whether *path* (already normalized) is forbidden."""

        return any(matches_pattern(path, pattern) for pattern in self.patterns())

    def check(self, paths: Iterable[str], *, worktree: Path | None = None) -> tuple[str, ...]:
        """Normalize and authorize one path list, or raise.

        The physical authority comes first: a path that escapes is refused
        before any matching happens.  Then a forbidden path is refused.
        """

        normalized = normalize_repo_paths(paths)
        if worktree is not None:
            assert_worktree_containment(worktree, normalized)
        denied = sorted({path for path in normalized if self.is_hard_denied(path)})
        if denied:
            raise ScopeViolation(
                HARD_DENY_PATH_MUTATION,
                "forbidden path(s) mutated: " + ", ".join(denied[:20]),
                denied,
            )
        return normalized


def matches_pattern(path: str, pattern: str) -> bool:
    """Whether one normalized path matches one hard-deny pattern.

    ``.git/**`` also denies ``.git`` itself, ``**/*secret*`` also denies a
    top-level ``secret.py``, and a pattern without a slash matches at any
    depth, the way a repository ignore rule does.
    """

    if fnmatch.fnmatchcase(path, pattern):
        return True
    if pattern.endswith("/**") and path == pattern[: -len("/**")]:
        return True
    if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
        return True
    if "/" not in pattern:
        return any(fnmatch.fnmatchcase(part, pattern) for part in path.split("/"))
    return False


def config_path_in_repo(repo: Path, config_path: Path) -> str | None:
    """The repository-relative path of *config_path*, when it is inside *repo*.

    The active configuration file is part of the run's authority: a worker may
    not rewrite the file that defines its own gates.
    """

    try:
        root = Path(repo).resolve()
        candidate = Path(config_path).resolve()
    except OSError:  # pragma: no cover - defensive
        return None
    if candidate == root or root not in candidate.parents:
        return None
    return candidate.relative_to(root).as_posix()


__all__ = [
    "DEFAULT_HARD_DENY_PATTERNS", "HARD_DENY_PATH_MUTATION", "SCOPE_MODES",
    "ScopePolicy", "ScopeViolation", "UNSAFE_PATH_MUTATION",
    "assert_worktree_containment", "config_path_in_repo", "matches_pattern",
    "normalize_repo_path", "normalize_repo_paths",
]
