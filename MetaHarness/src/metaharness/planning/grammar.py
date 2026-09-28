"""The labeled-body grammar of META PLAN v2.

One lexical authority: line normalization, the strict ``FIELD: value`` inline
form, the section forms, and the repository path sets the plan parser uses.
It performs no I/O, calls no model and never touches Git.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Sequence

# One step contract is one bounded unit of work: numbered operations, a short
# verification and a short pitfall list.  These are protocol limits, shared by
# the plan parser.
MAX_STEP_INSTRUCTIONS = 12
MAX_STEP_VERIFY_LINES = 6
MAX_STEP_PITFALL_LINES = 6

# The strict ``FIELD: value`` form the labeled bodies are built from.
INLINE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$")

_INSTRUCTION_MARKER = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>\d+[.)]|[-*+])"
    r"(?:[ \t]+(?P<content>\S(?:.*?\S)?))?[ \t]*$"
)


class PlanParseError(ValueError):
    """A planner answer was received but cannot be interpreted unambiguously."""


class V2PlanParseError(PlanParseError):
    """A META PLAN v2 response is not safe to execute."""


def lines(raw: str) -> list[str]:
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def nonempty(value: str, name: str) -> str:
    value = value.strip()
    if not value or value.casefold() in {"none", "n/a", "na", "-", "—", "nil", "tbd"}:
        raise V2PlanParseError(f"{name} is missing or a placeholder")
    return value


def _split_instruction_items(
    value: str,
) -> list[tuple[re.Match[str], list[str]]] | None:
    """Split an unambiguous top-level instruction list into its items.

    The first list marker establishes the harmless indentation of the list.
    Markers at that same indentation start items; everything else after the
    first item is continuation text.  This deliberately does not interpret
    prose before the first marker or indented nested lists.
    """

    raw_lines = value.splitlines()
    first_index = next((index for index, line in enumerate(raw_lines) if line.strip()), None)
    if first_index is None:
        return None
    first_match = _INSTRUCTION_MARKER.fullmatch(raw_lines[first_index])
    if first_match is None:
        return None
    base_indent = first_match.group("indent")
    items: list[tuple[re.Match[str], list[str]]] = []
    continuation: list[str] | None = None
    for raw_line in raw_lines[first_index:]:
        match = _INSTRUCTION_MARKER.fullmatch(raw_line)
        if match is not None and match.group("indent") == base_indent:
            continuation = []
            items.append((match, continuation))
            continue
        if match is not None:
            # A nested-looking marker could be either continuation content or
            # a misindented operation.  Refuse to guess at that boundary.
            return None
        if continuation is None:
            return None
        continuation.append(raw_line)
    return items


def _canonical_instruction_continuation(raw_line: str, base_indent: str) -> str:
    if not raw_line.strip():
        return ""
    remainder = raw_line
    if base_indent and remainder.startswith(base_indent):
        remainder = remainder[len(base_indent):]
    remainder = remainder.rstrip()
    if remainder[:1] not in {" ", "\t"}:
        return "  " + remainder.lstrip()
    return remainder


def _instruction_marker_kind(items: Sequence[tuple[re.Match[str], list[str]]]) -> str:
    markers = [match.group("marker") for match, _ in items]
    if all(marker in {"-", "*", "+"} for marker in markers):
        return "unordered"
    if all(marker.endswith(")") for marker in markers):
        return "numbered_parenthesized"
    if all(marker.endswith(".") for marker in markers):
        return "numbered"
    return "mixed"


def normalize_instruction_list(value: str) -> tuple[str, str | None]:
    """Return canonical numbered instructions and the source marker kind.

    ``None`` means the value was not a mechanically recognizable list and is
    left for the strict validator to reject.  Continuation text remains in its
    item and is never split into additional operations.
    """

    items = _split_instruction_items(value)
    if not items:
        return value, None
    base_indent = items[0][0].group("indent")
    canonical_lines: list[str] = []
    for number, (match, continuation) in enumerate(items, start=1):
        content = (match.group("content") or "").strip()
        canonical_lines.append(f"{number}. {content}" if content else f"{number}.")
        canonical_lines.extend(
            _canonical_instruction_continuation(line, base_indent)
            for line in continuation
        )
    canonical = "\n".join(canonical_lines).strip()
    return canonical, _instruction_marker_kind(items) if canonical != value else None


def validate_step_text_limits(step_id: str, values: dict[str, str]) -> None:
    title = values["TITLE"].strip()
    if len(title) > 100:
        raise V2PlanParseError(f"step {step_id} TITLE exceeds 100 characters")
    instruction_items = _split_instruction_items(values["INSTRUCTIONS"])
    if not instruction_items:
        raise V2PlanParseError(
            f"step {step_id} INSTRUCTIONS must contain 1 to {MAX_STEP_INSTRUCTIONS} "
            "numbered concrete operations"
        )
    if len(instruction_items) > MAX_STEP_INSTRUCTIONS:
        raise V2PlanParseError(
            f"step {step_id} INSTRUCTIONS exceeds {MAX_STEP_INSTRUCTIONS} operations"
        )
    for match, continuation in instruction_items:
        if not (match.group("content") or "").strip() and not any(
            line.strip() for line in continuation
        ):
            raise V2PlanParseError(
                f"step {step_id} INSTRUCTIONS must contain 1 to {MAX_STEP_INSTRUCTIONS} "
                "numbered concrete operations"
            )
    verify_lines = [line for line in values["VERIFY"].splitlines() if line.strip()]
    if len(verify_lines) > MAX_STEP_VERIFY_LINES:
        raise V2PlanParseError(f"step {step_id} VERIFY exceeds {MAX_STEP_VERIFY_LINES} lines")
    pitfall_lines = [line for line in values["PITFALLS"].splitlines() if line.strip()]
    if len(pitfall_lines) > MAX_STEP_PITFALL_LINES:
        raise V2PlanParseError(
            f"step {step_id} PITFALLS exceeds {MAX_STEP_PITFALL_LINES} entries"
        )


def section_name(line: str, allowed: frozenset[str]) -> str | None:
    candidate = line.strip()
    candidate = re.sub(r"^#{1,6}\s+", "", candidate)
    if candidate.endswith(":"):
        candidate = candidate[:-1].rstrip()
    for name in allowed:
        if candidate.casefold() == name.casefold():
            return name
    return None


def parse_labeled_body(
    body: Sequence[str],
    *,
    inline_names: frozenset[str],
    section_names: frozenset[str],
    where: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse one strict labeled body, tolerating Markdown headings for prose."""

    inline: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in body:
        stripped = raw_line.strip()
        if not stripped:
            if current is not None:
                sections[current].append("")
            continue

        match = INLINE.fullmatch(stripped)
        if match and match.group(1) in inline_names:
            name, value = match.groups()
            if name in inline:
                raise V2PlanParseError(f"duplicate {where} field {name}")
            inline[name] = value.strip()
            current = None
            continue

        if match and match.group(1) in section_names:
            name, value = match.groups()
            if name in sections:
                raise V2PlanParseError(f"duplicate {where} section {name}")
            sections[name] = [value] if value.strip() else []
            current = name if not value.strip() else None
            continue

        section = section_name(stripped, section_names)
        if section is not None:
            if section in sections:
                raise V2PlanParseError(f"duplicate {where} section {section}")
            sections[section] = []
            current = section
            continue

        if current is None:
            raise V2PlanParseError(f"unexpected content in {where}: {stripped[:80]}")
        sections[current].append(raw_line)

    values = {name: value.strip() for name, value in inline.items()}
    values.update({name: "\n".join(lines).strip() for name, lines in sections.items()})
    return values, {name: value for name, value in values.items() if name in section_names}


def repo_path(value: str, *, kind: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise V2PlanParseError(f"unsafe {kind} path")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise V2PlanParseError(f"unsafe {kind} path")
    path = PurePosixPath(value)
    if not value or path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise V2PlanParseError(f"unsafe {kind} path")
    if any(char in value for char in "*?["):
        raise V2PlanParseError(f"wildcard {kind} path")
    return value


def read_set(value: str, *, max_paths: int) -> tuple[str, ...]:
    if not value.strip():
        raise V2PlanParseError("READ_SET is missing")
    # Exactly ``NONE`` is the explicit empty set: a CREATE-only step may need
    # no read at all, and the normalizer adds every other mutation itself.
    if value.strip() == "NONE":
        return ()
    anchors_by_path: dict[str, list[str]] = {}
    for line in value.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- ") or " :: " not in line:
            raise V2PlanParseError("each READ_SET line must be '- path :: anchor', or exactly NONE")
        path, anchor = line[2:].split(" :: ", 1)
        path = repo_path(path.strip(), kind="READ_SET")
        anchor = anchor.strip()
        if not anchor:
            raise V2PlanParseError("READ_SET anchor must not be empty")
        # Repeated paths merge their anchors instead of failing: the scope is
        # unchanged, only the serialisation was split across lines.
        anchors = anchors_by_path.setdefault(path, [])
        if anchor not in anchors:
            anchors.append(anchor)
    if not anchors_by_path:
        raise V2PlanParseError("READ_SET is missing")
    if len(anchors_by_path) > max_paths:
        raise V2PlanParseError(
            f"READ_SET contains {len(anchors_by_path)} unique paths; "
            f"maximum is {max_paths}"
        )
    return tuple(path + " :: " + "; ".join(anchors) for path, anchors in anchors_by_path.items())


def read_set_paths(read_set: Sequence[str]) -> tuple[str, ...]:
    """The repo-relative paths of ``'path :: anchor'`` READ_SET entries."""

    return tuple(item.split(" :: ", 1)[0] for item in read_set)


def path_set(value: str, *, name: str) -> tuple[str, ...]:
    """Parse a ``- path`` list; exactly ``NONE`` is the explicit empty set."""

    if not value.strip():
        raise V2PlanParseError(f"{name} is missing")
    if value.strip() == "NONE":
        return ()
    result: list[str] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        if not line.startswith("- ") or " :: " in line:
            raise V2PlanParseError(f"each {name} line must be '- path'")
        path = repo_path(line[2:].strip(), kind=name)
        if path in result:
            raise V2PlanParseError(f"duplicate {name} path")
        result.append(path)
    if not result:
        raise V2PlanParseError(f"{name} is missing")
    return tuple(result)


def change_sets(
    values: dict[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Parse WRITE/CREATE/DELETE sets.

    A path listed in several sets, or a mutation left out of READ_SET, is a
    mechanical declaration defect the deterministic normalizer resolves against
    the tree; the parser only refuses a step that declares no mutation at all.
    """

    write_set = path_set(values["WRITE_SET"], name="WRITE_SET")
    create_set = path_set(values["CREATE_SET"], name="CREATE_SET")
    delete_set = path_set(values["DELETE_SET"], name="DELETE_SET")
    if not (write_set or create_set or delete_set):
        raise V2PlanParseError("a step must write, create or delete at least one path")
    return write_set, create_set, delete_set
