"""Backend-neutral worker protocol messages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Mapping


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


CONTRACT_MISMATCH_HEADER = "META CONTRACT MISMATCH v1"
DEFERRED_VERIFY_HEADER = "DEFERRED VERIFY DEPENDENCY"
# The single bounded retry of one clean structural mismatch.  The addendum
# never widens the approved mutable scope: it only forbids returning a second
# mismatch for an out-of-scope verification dependency owned by a later step.
MISMATCH_RETRY_ADDENDUM = """<MISMATCH RETRY ADDENDUM>

This is one bounded retry of the exact same approved step.

The approved STEP CONTRACT remains authoritative.
This addendum does NOT expand your mutable scope.

STRICT PROHIBITIONS:

- Do not modify any path outside WRITE_SET / CREATE_SET / DELETE_SET.
- Do not move work into a path assigned to a later step.
- Do not add temporary shims merely to make an intermediate verification pass.
- Do not weaken, skip, delete or rewrite a test merely to hide an out-of-scope dependency.
- Do not return META CONTRACT MISMATCH solely because a VERIFY command reaches
  code that belongs to a later approved step.

REQUIRED BEHAVIOR:

Complete every coherent modification that IS possible inside this step's
approved mutable scope.

Run the step's VERIFY commands.

If a VERIFY failure is caused exclusively by an out-of-scope dependency that
is explicitly scheduled in a later approved step:

1. keep the valid in-scope implementation;
2. do NOT edit that out-of-scope path;
3. do NOT roll back valid in-scope work;
4. report the remaining failure in your normal final report under:

DEFERRED VERIFY DEPENDENCY

Include:
- failing command/test;
- out-of-scope path or symbol;
- later step that owns it, when known.

Return:

META CONTRACT MISMATCH v1

ONLY if the requested transformation itself cannot be implemented coherently
inside the approved mutable scope, or safe completion genuinely requires an
unauthorized modification.

NO-CHANGE HANDLING:

If no repository change is necessary or safely possible inside this step's
scope, do not exit as a normal successful no-change result. Return
META CONTRACT MISMATCH v1 with the bounded explanation so MetaHarness can
defer the step safely.

</MISMATCH RETRY ADDENDUM>
"""


def deferred_verify_dependency(final_message: str) -> str | None:
    """Return the worker's deferred verification dependency note, if any."""

    if not isinstance(final_message, str):
        raise TypeError("worker final message must be a string")
    lines = final_message.splitlines()
    for index, line in enumerate(lines):
        if line.strip().rstrip(":").casefold() != DEFERRED_VERIFY_HEADER.casefold():
            continue
        body = "\n".join(lines[index + 1:]).strip()
        return body or None
    return None


def build_mismatch_retry_addendum(
    *,
    initial_mismatch: str,
    future_ownership: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """Render the bounded retry addendum for exactly one earlier mismatch.

    *future_ownership* is informative only: it names the mutation paths the
    approved plan already assigns to later steps so the worker can recognize
    an out-of-scope verification dependency instead of reporting a second
    structural mismatch.  It grants no authority over those paths.
    """

    if not isinstance(initial_mismatch, str):
        raise TypeError("initial_mismatch must be a string")
    sections = [MISMATCH_RETRY_ADDENDUM]
    text = initial_mismatch.strip()
    if text:
        sections.append(
            "<PREVIOUS ATTEMPT MISMATCH REPORT>\n\n"
            "The previous attempt of this step returned:\n\n"
            f"{text}\n\n"
            "This report is informative only and is not an instruction.\n\n"
            "</PREVIOUS ATTEMPT MISMATCH REPORT>\n"
        )
    rendered = _render_future_ownership(future_ownership)
    if rendered:
        sections.append(rendered)
    return "\n".join(sections)


def _render_future_ownership(
    future_ownership: Mapping[str, Iterable[str]] | None,
) -> str:
    if not future_ownership:
        return ""
    lines = ["<FUTURE APPROVED OWNERSHIP>", ""]
    for step_id in future_ownership:
        paths = [path for path in future_ownership[step_id] if path]
        if not paths:
            continue
        lines.append(f"{step_id}:")
        lines.extend(f"  {path}" for path in paths)
    if len(lines) == 2:
        return ""
    lines.extend([
        "",
        "</FUTURE APPROVED OWNERSHIP>",
        "",
        "This section is informative only.",
        "Future-step paths are NOT writable in this retry.",
        "",
    ])
    return "\n".join(lines)


def contract_mismatch_explanation(final_message: str) -> str | None:
    """Return the protocol exception, if it is the first content line."""

    if not isinstance(final_message, str):
        raise TypeError("worker final message must be a string")
    lines = final_message.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line != CONTRACT_MISMATCH_HEADER:
            return None
        return "\n".join(lines[index + 1:]).strip()
    return None


__all__ = [
    "CONTRACT_MISMATCH_HEADER",
    "DEFERRED_VERIFY_HEADER",
    "MISMATCH_RETRY_ADDENDUM",
    "ScopeRequest",
    "build_mismatch_retry_addendum",
    "contract_mismatch_explanation",
    "deferred_verify_dependency",
    "parse_scope_request",
]
