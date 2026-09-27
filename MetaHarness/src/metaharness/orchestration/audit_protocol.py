"""The sole machine-readable completion contract for an active audit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditReport:
    status: str
    fixed: tuple[str, ...]
    refactored: tuple[str, ...]
    remaining: tuple[str, ...]
    risks: tuple[str, ...]


def parse_audit_report(message: str) -> AuditReport | None:
    if not isinstance(message, str):
        return None
    lines = message.splitlines()
    header, footer = "META AUDIT v1", "END META AUDIT"
    if lines.count(header) != 1 or lines.count(footer) != 1:
        return None
    start, end = lines.index(header), lines.index(footer)
    if any(line.strip() for line in lines[end + 1:]):
        return None
    block = lines[start + 1:end]
    if not block or block[0] != "" or len(block) < 2 or block[1] != "STATUS":
        return None
    if len(block) < 3 or block[2] not in {"DONE", "NEEDS_WORK", "SPEC_DECISION"}:
        return None
    sections: dict[str, tuple[str, ...]] = {}
    index = 3
    for name in ("FIXED", "REFACTORED", "REMAINING", "RISKS"):
        if index >= len(block) or block[index] != "" or index + 1 >= len(block) or block[index + 1] != name:
            return None
        index += 2
        items: list[str] = []
        while index < len(block) and block[index].startswith("- "):
            item = block[index][2:].strip()
            if not item:
                return None
            if item != "none":
                items.append(item)
            index += 1
        if not items and (index == 0 or block[index - 1] != "- none"):
            return None
        sections[name] = tuple(items)
    if index != len(block):
        return None
    if block[2] in {"NEEDS_WORK", "SPEC_DECISION"} and not sections["REMAINING"]:
        return None
    return AuditReport(block[2], sections["FIXED"], sections["REFACTORED"], sections["REMAINING"], sections["RISKS"])
