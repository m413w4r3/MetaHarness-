"""Parseur tolérant de documents textuels étiquetés produits par un LLM.

Le parseur est tolérant sur la présentation (Markdown, casse, gras, titres,
réponse entière dans un bloc ``markdown``) mais ne lit jamais de métadonnée
dans un bloc de code. Les valeurs de contrôle (STATUS, VERDICT, ROUTE) sont
lues strictement par :func:`control_tokens`.
"""

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


_WRAPPER_INFOS = frozenset({"", "markdown", "md", "text", "txt"})
_CONTROL_DECORATION = "`*_[]()\"' "


def parse_labeled_document(
    text: str,
    *,
    field_aliases: dict[str, tuple[str, ...]],
    section_aliases: dict[str, tuple[str, ...]],
    bare_labels: bool = False,
    control_fields: Iterable[str] = (),
) -> ParsedTextDocument:
    """Parse des champs et sections sans interpréter le contenu des code fences.

    Les clés des mappings sont les noms canoniques qui apparaîtront dans le
    résultat; leurs alias sont comparés sans tenir compte de la casse.

    ``bare_labels`` accepte aussi un alias seul sur sa ligne (``OBJECTIVE``)
    comme titre de section.  Un titre ``## Label: valeur`` devient alors un
    champ; pour un champ de ``control_fields`` il n'ouvre pas de section, afin
    que la prose qui suit ne soit jamais lue comme valeur de contrôle.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized_fields = _normalize_aliases(field_aliases)
    normalized_sections = _normalize_aliases(section_aliases)
    lines = unwrap_document_fence(text.splitlines())
    if bare_labels:
        lines = _expose_bare_labels(lines, normalized_sections, frozenset(control_fields))

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
                    f"{previous[:80]!r} versus {value[:80]!r}"
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


def control_tokens(value: str, known: Iterable[str]) -> tuple[str, ...]:
    """Read a control value strictly.

    Every non-empty line must be exactly one known token, optionally decorated
    with Markdown emphasis, code ticks, brackets or a final period.  Prose such
    as ``not PASS`` is rejected instead of being searched for a keyword.
    """

    allowed = frozenset(item.upper() for item in known)
    tokens: list[str] = []
    for line in value.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^[-*+]\s+", "", candidate)
        candidate = candidate.strip(_CONTROL_DECORATION).rstrip(".").strip(_CONTROL_DECORATION)
        token = candidate.upper()
        if token not in allowed:
            raise WireParseError(
                f"control value {line.strip()[:80]!r} is not exactly one of "
                + ", ".join(sorted(allowed))
            )
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def unwrap_document_fence(lines: list[str]) -> list[str]:
    """Remove a fence wrapping the whole response, and only such a fence.

    A response that merely starts with one code block and ends with another is
    not a wrapped document: unwrapping it would expose the content of those
    code blocks as metadata.  The wrapper is therefore accepted only when no
    line inside it could close it and every inner fence is balanced.
    """

    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if len(nonempty) < 2:
        return lines
    first, last = nonempty[0], nonempty[-1]
    opening = _fence_opening(lines[first])
    if opening is None or _fence_info(lines[first]) not in _WRAPPER_INFOS:
        return lines
    if not _is_fence_close(lines[last], opening):
        return lines
    inner: tuple[str, int] | None = None
    for line in lines[first + 1 : last]:
        if inner is not None:
            if _is_fence_close(line, inner):
                inner = None
            continue
        candidate = _fence_opening(line)
        if candidate is None:
            continue
        if _is_fence_close(line, opening):
            return lines
        inner = candidate
    if inner is not None:
        return lines
    return lines[first + 1 : last]


def _expose_bare_labels(
    lines: list[str],
    sections: tuple[tuple[str, str], ...],
    control_fields: frozenset[str],
) -> list[str]:
    labels = {alias: canonical for canonical, alias in sections}
    transformed: list[str] = []
    fence: tuple[str, int] | None = None
    for line in lines:
        if fence is not None:
            transformed.append(line)
            if _is_fence_close(line, fence):
                fence = None
            continue
        opening = _fence_opening(line)
        if opening is not None:
            transformed.append(line)
            fence = opening
            continue
        heading_inline = re.match(
            r"^\s*#{1,6}\s+(?P<label>[^:=]+?)\s*[:=]\s*(?P<value>.+?)\s*#*\s*$",
            line,
        )
        if heading_inline is not None:
            label = heading_inline.group("label").strip()
            canonical = labels.get(_normalize_label(label))
            if canonical is not None:
                transformed.append(f"{label}: {heading_inline.group('value').strip()}")
                if canonical not in control_fields:
                    transformed.append(f"## {label}")
                continue
        candidate = re.sub(r"^[-*+]\s+", "", line.strip())
        emphasized = re.fullmatch(r"(\*\*|__)(?P<label>[^*_]+?)\1", candidate)
        if emphasized is not None:
            candidate = emphasized.group("label")
        if " ".join(candidate.split()).casefold() in labels:
            transformed.append(f"## {candidate}")
        else:
            transformed.append(line)
    return transformed


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


def _fence_info(line: str) -> str:
    match = re.match(r"^\s*(?:`{3,}|~{3,})(?P<info>[^`]*)$", line)
    return match.group("info").strip().casefold() if match else ""


def _is_fence_close(line: str, fence: tuple[str, int]) -> bool:
    character, length = fence
    return re.match(rf"^\s*{re.escape(character)}{{{length},}}\s*$", line) is not None


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


def fence_lines(text: str) -> list[tuple[str, bool]]:
    """Return ``(line, inside_code_fence)`` pairs after unwrapping the document."""

    result: list[tuple[str, bool]] = []
    fence: tuple[str, int] | None = None
    for line in unwrap_document_fence(text.splitlines()):
        if fence is not None:
            result.append((line, True))
            if _is_fence_close(line, fence):
                fence = None
            continue
        opening = _fence_opening(line)
        if opening is not None:
            result.append((line, True))
            fence = opening
            continue
        result.append((line, False))
    return result


__all__ = [
    "AmbiguousFieldError",
    "ParsedTextDocument",
    "WireParseError",
    "control_tokens",
    "fence_lines",
    "parse_labeled_document",
    "unwrap_document_fence",
]
