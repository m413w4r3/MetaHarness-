"""Backend-neutral worker protocol messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ScopeRequest:
    """Advisory request for operator-reviewed mutable-scope expansion."""

    reason: str
    paths: tuple[str, ...]
    evidence: tuple[str, ...]


_SCOPE_REQUEST_HEADER = "META SCOPE REQUEST v1"
_SCOPE_REQUEST_FOOTER = "END META SCOPE REQUEST"
_MAX_SCOPE_REQUEST_PATHS = 32
_SCOPE_GLOB_CHARS = frozenset("*?[]{}")


def _valid_scope_request_path(path: str) -> bool:
    if (
        not isinstance(path, str)
        or not path
        or path != path.strip()
        or "\x00" in path
        or "\\" in path
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or any(character in _SCOPE_GLOB_CHARS for character in path)
    ):
        return False
    try:
        parsed = PurePosixPath(path)
    except (TypeError, ValueError):
        return False
    return (
        parsed.as_posix() == path
        and path not in {".", ".."}
        and all(part not in {".", ".."} for part in parsed.parts)
    )


def parse_scope_request(final_message: str) -> ScopeRequest | None:
    """Parse the strict advisory scope-request block."""

    if not isinstance(final_message, str):
        return None
    lines = final_message.splitlines()
    if _SCOPE_REQUEST_HEADER not in lines:
        return None
    if lines.count(_SCOPE_REQUEST_HEADER) != 1 or lines.count(_SCOPE_REQUEST_FOOTER) != 1:
        return None
    start = lines.index(_SCOPE_REQUEST_HEADER)
    end = lines.index(_SCOPE_REQUEST_FOOTER)
    if end <= start:
        return None
    block = lines[start : end + 1]
    if len(block) < 11 or block[0] != _SCOPE_REQUEST_HEADER or block[-1] != _SCOPE_REQUEST_FOOTER:
        return None
    if block[1] != "" or block[-2] != "" or block[2] != "REASON":
        return None
    try:
        paths_index = block.index("PATHS", 3)
        evidence_index = block.index("EVIDENCE", paths_index + 1)
    except ValueError:
        return None
    if (
        paths_index <= 3
        or evidence_index <= paths_index + 1
        or block[paths_index - 1] != ""
        or block[evidence_index - 1] != ""
        or evidence_index + 1 >= len(block)
    ):
        return None
    reason_lines = block[3 : paths_index - 1]
    path_lines = block[paths_index + 1 : evidence_index - 1]
    evidence_lines = block[evidence_index + 1 : -2]
    if (
        not reason_lines
        or any(not line.strip() for line in reason_lines)
        or not path_lines
        or not evidence_lines
        or any(not line.startswith("- ") or not line[2:].strip() for line in path_lines)
        or any(not line.startswith("- ") or not line[2:].strip() for line in evidence_lines)
    ):
        return None
    paths = tuple(line[2:] for line in path_lines)
    evidence = tuple(line[2:] for line in evidence_lines)
    if len(paths) > _MAX_SCOPE_REQUEST_PATHS or len(set(paths)) != len(paths):
        return None
    if any(not _valid_scope_request_path(path) for path in paths):
        return None
    reason = "\n".join(reason_lines).strip()
    if not reason:
        return None
    return ScopeRequest(reason=reason, paths=paths, evidence=evidence)


__all__ = ["ScopeRequest", "parse_scope_request"]
