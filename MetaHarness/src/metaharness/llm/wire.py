"""Parseur tolérant de documents textuels étiquetés produits par un LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


class WireParseError(ValueError):
    """Document texte impossible à interpréter sans perdre une information."""


class AmbiguousFieldError(WireParseError):
    """Une même valeur de champ a été trouvée avec des contenus contradictoires."""


@dataclass(frozen=True)
class ParsedTextDocument:
    fields: dict[str, str]
    sections: dict[str, str]
    preamble: str
    postamble: str
    raw: str


@dataclass(frozen=True)
class _SectionStart:
    canonical: str
    line_index: int
    delimiter: str
    close_line_index: int | None = None


def parse_labeled_document(
    text: str,
    *,
    field_aliases: dict[str, tuple[str, ...]],
    section_aliases: dict[str, tuple[str, ...]],
) -> ParsedTextDocument:
    """Parse des champs et sections sans interpréter le contenu des code fences.

    Les clés des mappings sont les noms canoniques qui apparaîtront dans le
    résultat; leurs alias sont comparés sans tenir compte de la casse.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_fields = _normalize_aliases(field_aliases)
    normalized_sections = _normalize_aliases(section_aliases)
    source_lines = text.splitlines()
    lines = _unwrap_markdown_document(source_lines)

    fields: dict[str, str] = {}
    field_values: dict[str, str] = {}
    section_starts: list[_SectionStart] = []
    fence: tuple[str, int] | None = None

    for index, line in enumerate(lines):
        if fence is not None:
            if _is_fence_close(line, fence):
                fence = None
            continue
        opening = _fence_opening(line)
        if opening is not None:
            fence = opening
            continue

        field = _match_field(line, normalized_fields)
        if field is not None:
            canonical, value = field
            previous = field_values.get(canonical)
            if previous is not None and _comparison_value(previous) != _comparison_value(value):
                raise AmbiguousFieldError(
                    f"ambiguous value for field {canonical!r}: "
                    f"{previous!r} versus {value!r}"
                )
            field_values[canonical] = value
            fields[canonical] = value

        section = _match_section(line, normalized_sections)
        if section is not None:
            canonical, delimiter = section
            section_starts.append(_SectionStart(canonical, index, delimiter))

    # Bracket/equals/colon markers may be used as an opening and closing
    # delimiter.  A repeated marker for the active section is therefore not a
    # second section with an empty body.
    section_starts = _pair_explicit_delimiters(section_starts)

    sections: dict[str, str] = {}
    if section_starts:
        first = section_starts[0].line_index
        preamble = _clean_outer_text(lines[:first])
        postamble = ""
        for position, start in enumerate(section_starts):
            if start.close_line_index is not None:
                end = start.close_line_index
            elif position + 1 < len(section_starts):
                end = section_starts[position + 1].line_index
            else:
                end = len(lines)
            content_lines = lines[start.line_index + 1 : end]
            if start.close_line_index is not None:
                section_postamble = _clean_outer_text(
                    lines[start.close_line_index + 1 :]
                )
            else:
                conclusion_index = _conclusion_heading_index(content_lines)
                if conclusion_index is not None:
                    section_postamble = _clean_outer_text(
                        content_lines[conclusion_index + 1 :]
                    )
                    content_lines = content_lines[:conclusion_index]
                else:
                    section_postamble = ""
            if section_postamble:
                postamble = section_postamble
            content = _clean_section_text(content_lines)
            if start.canonical in sections and content:
                sections[start.canonical] = f"{sections[start.canonical]}\n\n{content}"
            else:
                sections.setdefault(start.canonical, content)
        if not postamble:
            postamble = _postamble_after_conclusion_heading(lines, section_starts[-1])
    else:
        preamble = _clean_outer_text(lines)
        postamble = ""

    return ParsedTextDocument(
        fields=fields,
        sections=sections,
        preamble=preamble,
        postamble=postamble,
        raw=text,
    )


def _normalize_aliases(aliases: dict[str, tuple[str, ...]]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for canonical, values in aliases.items():
        if not isinstance(canonical, str):
            raise TypeError("document aliases must use string canonical names")
        for alias in values:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("document aliases must be non-empty strings")
            result.append((canonical, _normalize_label(alias)))
    return tuple(sorted(result, key=lambda item: len(item[1]), reverse=True))


def _normalize_label(value: str) -> str:
    value = value.strip().strip("[]")
    value = value.strip("*=:# ")
    value = re.sub(r"[*_`]+", "", value)
    return " ".join(value.split()).casefold()


def _match_field(
    line: str, aliases: Iterable[tuple[str, str]]
) -> tuple[str, str] | None:
    candidate = line.strip()
    candidate = re.sub(r"^(?:[-*+]\s+)", "", candidate)
    candidate = re.sub(r"^>\s*", "", candidate)
    candidate = re.sub(
        r"^\*\*(?P<label>[^*]+?)\s*:\s*\*\*\s*(?P<value>.*)$",
        r"\g<label>: \g<value>",
        candidate,
    )
    candidate = re.sub(
        r"^__(?P<label>[^_]+?)\s*:\s*__\s*(?P<value>.*)$",
        r"\g<label>: \g<value>",
        candidate,
    )
    candidate = re.sub(r"^(?:\*\*|__)(.+?)(?:\*\*|__)(\s*[:=])", r"\1\2", candidate)
    candidate = re.sub(r"^`([^`]+)`(\s*[:=])", r"\1\2", candidate)
    match = re.match(r"^(?P<label>[^:=]+?)\s*[:=]\s*(?P<value>.*)$", candidate)
    if match is None:
        return None
    label = _normalize_label(match.group("label"))
    for canonical, alias in aliases:
        if label == alias:
            return canonical, match.group("value").strip()
    return None


def _match_section(
    line: str, aliases: Iterable[tuple[str, str]]
) -> tuple[str, str] | None:
    candidate = line.strip()
    candidate = re.sub(r"^[-*+]\s+", "", candidate)
    heading = re.match(r"^#{1,6}\s+(?P<label>.+?)\s*#*\s*$", candidate)
    if heading is not None:
        label = _normalize_label(heading.group("label").rstrip(":="))
        for canonical, alias in aliases:
            if label == alias:
                return canonical, "heading"
        return None

    bracket = re.fullmatch(r"\[\s*(?P<label>[^\]]+)\s*\]", candidate)
    if bracket is not None:
        label = _normalize_label(bracket.group("label"))
        for canonical, alias in aliases:
            if label == alias:
                return canonical, "bracket"

    equals = re.fullmatch(r"={3,}\s*(?P<label>.+?)\s*={3,}", candidate)
    if equals is not None:
        label = _normalize_label(equals.group("label"))
        for canonical, alias in aliases:
            if label == alias:
                return canonical, "equals"

    marker = re.fullmatch(r"(?P<label>[^:=]+?)\s*[:=]", candidate)
    if marker is not None:
        label = _normalize_label(marker.group("label"))
        for canonical, alias in aliases:
            if label == alias:
                return canonical, "colon"
    return None


def _fence_opening(line: str) -> tuple[str, int] | None:
    match = re.match(r"^\s*(`{3,}|~{3,})(?P<info>[^`]*)$", line)
    if match is None:
        return None
    return match.group(1)[0], len(match.group(1))


def _is_fence_close(line: str, fence: tuple[str, int]) -> bool:
    character, length = fence
    return re.match(rf"^\s*{re.escape(character)}{{{length},}}\s*$", line) is not None


def _unwrap_markdown_document(lines: list[str]) -> list[str]:
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if len(nonempty) < 2:
        return lines
    first = nonempty[0]
    last = nonempty[-1]
    opening = re.match(r"^\s*(`{3,}|~{3,})(?P<info>[^`]*)$", lines[first])
    if opening is None or not _is_fence_close(
        lines[last], (opening.group(1)[0], len(opening.group(1)))
    ):
        return lines
    info = opening.group("info").strip().casefold()
    if info not in {"", "markdown", "md", "text", "txt"}:
        return lines
    return lines[first + 1 : last]


def _comparison_value(value: str) -> str:
    return " ".join(value.split()).casefold()


def _pair_explicit_delimiters(starts: list[_SectionStart]) -> list[_SectionStart]:
    paired: list[_SectionStart] = []
    for start in starts:
        if (
            paired
            and start.delimiter in {"bracket", "equals", "colon"}
            and paired[-1].canonical == start.canonical
            and paired[-1].delimiter == start.delimiter
            and paired[-1].close_line_index is None
        ):
            previous = paired[-1]
            paired[-1] = _SectionStart(
                previous.canonical,
                previous.line_index,
                previous.delimiter,
                start.line_index,
            )
        else:
            paired.append(start)
    return paired


def _postamble_after_conclusion_heading(
    lines: list[str], last_section: _SectionStart
) -> str:
    for index in range(last_section.line_index + 1, len(lines)):
        if _is_conclusion_heading(lines[index]):
            return _clean_outer_text(lines[index + 1 :])
    return ""


def _conclusion_heading_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if _is_conclusion_heading(line):
            return index
    return None


def _is_conclusion_heading(line: str) -> bool:
    return re.match(
        r"^\s*#{1,6}\s+(?:conclusion|postamble)\s*:?\s*$", line, re.I
    ) is not None


def _clean_outer_text(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def _clean_section_text(lines: list[str]) -> str:
    return "\n".join(lines).strip()
