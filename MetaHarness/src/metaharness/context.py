"""Construction of planner context from an exact Git commit."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .gitops import GitError, current_head, git_root, read_file_at_commit, resolve_commit
from .models import ContextConfig


@dataclass(frozen=True)
class ContextExcerpt:
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    content: str


@dataclass(frozen=True)
class ContextBundle:
    base_sha: str
    instruction_files: tuple[tuple[str, str], ...]
    excerpts: tuple[ContextExcerpt, ...]
    locator_used: bool
    locator_warning: str | None
    omitted: tuple[str, ...]
    total_bytes: int


_INSTRUCTION_NAMES = frozenset({"AGENTS.md", "CLAUDE.md"})
_MAX_EXCERPT_LINES = 200


def _read_optional(repo: Path, base_sha: str, path: str) -> str | None:
    """Read a file from Git, treating a missing path as an empty result."""

    try:
        return read_file_at_commit(
            repo,
            commit_sha=base_sha,
            relative_path=path,
        )
    except (GitError, UnicodeError):
        return None


def _configured_instruction_names(config: ContextConfig) -> frozenset[str]:
    return frozenset(
        PurePosixPath(path).name
        for path in config.always_files
        if PurePosixPath(path).name in _INSTRUCTION_NAMES
    )


def _canonical_path(path: str) -> str:
    """Canonicalize harmless leading ``./`` in Git paths."""

    while path.startswith("./"):
        path = path[2:]
    return path


def _applicable_instruction_paths(
    locator_path: str, instruction_names: frozenset[str]
) -> tuple[str, ...]:
    path = PurePosixPath(_canonical_path(locator_path))
    parent_parts = path.parent.parts
    prefixes: list[str] = []
    for index in range(len(parent_parts) + 1):
        directory = PurePosixPath(*parent_parts[:index])
        for name in ("AGENTS.md", "CLAUDE.md"):
            if name in instruction_names:
                prefixes.append(str(directory / name) if str(directory) != "." else name)
    return tuple(prefixes)


def _safe_locator_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        return None
    return _canonical_path(value)


def _valid_line_range(start: Any, end: Any) -> bool:
    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and start >= 1
        and end >= start
        and end - start + 1 <= _MAX_EXCERPT_LINES
    )


def _locator_command(config: ContextConfig, spec: str) -> list[str]:
    return [argument.replace("{query}", spec) for argument in config.locator_argv]


def _run_locator(
    repo: Path, spec: str, config: ContextConfig
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Run the locator and return (was_started, hits, warnings)."""

    try:
        result = subprocess.run(
            _locator_command(config, spec),
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=config.locator_timeout_seconds,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True, [], ["locator timed out"]
    except (OSError, ValueError):
        return True, [], ["locator could not be started"]

    if result.returncode != 0:
        return True, [], [f"locator failed (exit status {result.returncode})"]
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return True, [], ["locator returned invalid JSON"]
    if not isinstance(payload, list):
        return True, [], ["locator JSON must be an array"]
    return True, [item for item in payload if isinstance(item, dict)], []


def _excerpt_from_hit(
    repo: Path, base_sha: str, hit: dict[str, Any]
) -> tuple[ContextExcerpt | None, str | None]:
    path = _safe_locator_path(hit.get("path"))
    if path is None:
        return None, "locator returned an unsafe path"
    start = hit.get("start")
    end = hit.get("end")
    if not _valid_line_range(start, end):
        return None, f"locator returned an invalid range for {path}"

    content = _read_optional(repo, base_sha, path)
    if content is None:
        return None, f"locator path does not exist at base commit: {path}"
    lines = content.splitlines(keepends=True)
    if start > len(lines) or end > len(lines):
        return None, f"locator range is outside file at base commit: {path}"

    symbol = hit.get("symbol")
    if symbol is not None and not isinstance(symbol, str):
        symbol = None
    return (
        ContextExcerpt(
            path=path,
            start_line=start,
            end_line=end,
            symbol=symbol,
            content="".join(lines[start - 1 : end]),
        ),
        None,
    )


def build_context(
    source_repo: str | Path,
    base_ref: str,
    spec: str,
    config: ContextConfig,
) -> ContextBundle:
    """Build planner context whose every file read is pinned to ``base_ref``.

    The locator is advisory: its JSON only supplies paths and line ranges.  The
    source text and instruction files are always read with ``git show`` at the
    resolved base commit.
    """

    if not isinstance(config, ContextConfig):
        raise TypeError("config must be a ContextConfig")
    repo = git_root(Path(source_repo).expanduser())
    base_sha = resolve_commit(repo, base_ref)
    instruction_names = _configured_instruction_names(config)

    warnings: list[str] = []
    instructions: list[tuple[str, str]] = []
    seen_instruction_paths: set[str] = set()

    def add_instruction(path: str) -> None:
        canonical = _canonical_path(path)
        if canonical in seen_instruction_paths:
            return
        content = _read_optional(repo, base_sha, canonical)
        if content is None:
            return
        seen_instruction_paths.add(canonical)
        instructions.append((canonical, content))

    # Root AGENTS is the first instruction whenever that instruction family is
    # configured.  The applicable hierarchy is then added root-to-leaf.
    if "AGENTS.md" in instruction_names:
        add_instruction("AGENTS.md")

    # Root instruction files configured in always_files are always useful,
    # including when the locator is disabled or returns no hits.  Nested
    # instruction files are added below as hits establish their applicability.
    # Keep direct instruction files aside so applicable files keep priority.
    direct_instruction_paths = [
        path
        for path in config.always_files
        if PurePosixPath(path).name in _INSTRUCTION_NAMES
        and _canonical_path(path) != "AGENTS.md"
    ]

    # Read all directly configured files before consulting the locator.  The
    # contents are cached locally so the later ordering of the bundle does not
    # cause a second working-tree read.
    direct_instructions: list[tuple[str, str]] = []
    seen_direct_instruction_paths: set[str] = set()
    for path in direct_instruction_paths:
        canonical = _canonical_path(path)
        if canonical in seen_direct_instruction_paths:
            continue
        content = _read_optional(repo, base_sha, canonical)
        if content is not None:
            seen_direct_instruction_paths.add(canonical)
            direct_instructions.append((canonical, content))

    always_tail: list[tuple[str, str]] = []
    seen_tail_paths: set[str] = set()
    for path in config.always_files:
        canonical = _canonical_path(path)
        if PurePosixPath(path).name in _INSTRUCTION_NAMES:
            continue
        if canonical in seen_tail_paths:
            continue
        content = _read_optional(repo, base_sha, canonical)
        if content is not None:
            seen_tail_paths.add(canonical)
            always_tail.append((canonical, content))

    locator_used = False
    raw_hits: list[dict[str, Any]] = []
    if config.locator_argv:
        if config.require_locator_head_at_base:
            try:
                head = current_head(repo)
            except GitError:
                head = None
            if head != base_sha:
                warnings.append("locator skipped because repository HEAD differs from base commit")
            else:
                locator_used, raw_hits, locator_warnings = _run_locator(repo, spec, config)
                warnings.extend(locator_warnings)
        else:
            locator_used, raw_hits, locator_warnings = _run_locator(repo, spec, config)
            warnings.extend(locator_warnings)

    valid_excerpts: list[ContextExcerpt] = []
    for hit in raw_hits[: config.max_hits]:
        excerpt, warning = _excerpt_from_hit(repo, base_sha, hit)
        if excerpt is not None:
            valid_excerpts.append(excerpt)
            for instruction_path in _applicable_instruction_paths(
                excerpt.path, instruction_names
            ):
                add_instruction(instruction_path)
        if warning is not None:
            warnings.append(warning)

    for path, content in direct_instructions:
        if path not in seen_instruction_paths:
            seen_instruction_paths.add(path)
            instructions.append((path, content))

    omitted: list[str] = []
    omitted_set: set[str] = set()
    selected_instructions: list[tuple[str, str]] = []
    selected_excerpts: list[ContextExcerpt] = []
    selected_tail: list[tuple[str, str]] = []
    total_bytes = 0

    def include(path: str, content: str) -> bool:
        nonlocal total_bytes
        size = len(content.encode("utf-8"))
        if total_bytes + size > config.max_bytes:
            if path not in omitted_set:
                omitted.append(path)
                omitted_set.add(path)
            return False
        total_bytes += size
        return True

    for path, content in instructions:
        if include(path, content):
            selected_instructions.append((path, content))
    for excerpt in valid_excerpts:
        if include(excerpt.path, excerpt.content):
            selected_excerpts.append(excerpt)
    for path, content in always_tail:
        if include(path, content):
            selected_tail.append((path, content))

    return ContextBundle(
        base_sha=base_sha,
        instruction_files=tuple(selected_instructions + selected_tail),
        excerpts=tuple(selected_excerpts),
        locator_used=locator_used,
        locator_warning="; ".join(warnings) if warnings else None,
        omitted=tuple(omitted),
        total_bytes=total_bytes,
    )


build_context_bundle = build_context


def render_context(bundle: ContextBundle) -> str:
    """Render a compact, explicitly data-labelled planner context."""

    parts = [f"BASE SHA: {bundle.base_sha}"]
    if bundle.locator_warning:
        parts.extend(("", "### CONTEXT WARNING", bundle.locator_warning))

    # Keep README/other always-files at the low-priority end of the rendered
    # data, just as they were considered by the byte budget.
    primary_instructions = [
        (path, content)
        for path, content in bundle.instruction_files
        if PurePosixPath(path).name in _INSTRUCTION_NAMES
    ]
    trailing_files = [
        (path, content)
        for path, content in bundle.instruction_files
        if PurePosixPath(path).name not in _INSTRUCTION_NAMES
    ]

    for path, content in primary_instructions:
        parts.extend(("", f"### PROJECT INSTRUCTION: {path}", content.rstrip("\n")))
    for excerpt in bundle.excerpts:
        parts.extend(
            (
                "",
                f"### SOURCE: {excerpt.path}:{excerpt.start_line}-{excerpt.end_line}",
            )
        )
        if excerpt.symbol is not None:
            parts.append(f"symbol: {excerpt.symbol}")
        parts.append(excerpt.content.rstrip("\n"))
    for path, content in trailing_files:
        parts.extend(("", f"### PROJECT INSTRUCTION: {path}", content.rstrip("\n")))
    return "\n".join(parts) + "\n"


__all__ = [
    "ContextBundle",
    "ContextExcerpt",
    "build_context",
    "build_context_bundle",
    "render_context",
]
