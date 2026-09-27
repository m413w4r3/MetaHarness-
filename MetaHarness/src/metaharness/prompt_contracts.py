"""Role-specific, deterministic prompt payloads.

The builders in this module are deliberately narrow.  They accept the
contract for one model role instead of a run-shaped bag of evidence, and they
keep the bytes used for rendering next to the metadata used for diagnostics.
No section content is written to diagnostics: only sizes, hashes and the
authority/truncation decisions are persisted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .result import atomic_write_text


_TRUNCATION_MARKER = "\n[TRUNCATED: secondary prompt evidence]\n"


def _default_template(name: str) -> str:
    """Load one shipped role template for direct builder callers."""

    return (Path(__file__).with_name("prompts") / name).read_text(encoding="utf-8")


def _encoded(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def _utf8_prefix(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return _encoded(text)[:limit].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class PromptSection:
    """One named prompt section and its byte-level identity."""

    name: str
    text: str
    authority: bool
    sha256: str
    byte_count: int
    truncated: bool = False

    @classmethod
    def create(
        cls, name: str, text: str, authority: bool, *, truncated: bool = False
    ) -> "PromptSection":
        if not isinstance(name, str) or not name.strip():
            raise ValueError("prompt section name must be non-empty")
        if not isinstance(text, str):
            raise TypeError("prompt section text must be a string")
        data = _encoded(text)
        return cls(
            name=name,
            text=text,
            authority=bool(authority),
            sha256=hashlib.sha256(data).hexdigest(),
            byte_count=len(data),
            truncated=truncated,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("prompt section name must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("prompt section text must be a string")
        expected_bytes = len(_encoded(self.text))
        expected_sha = hashlib.sha256(_encoded(self.text)).hexdigest()
        if self.byte_count != expected_bytes:
            raise ValueError("prompt section byte_count does not match text")
        if self.sha256 != expected_sha:
            raise ValueError("prompt section sha256 does not match text")


@dataclass(frozen=True)
class PromptPayload:
    """The exact request sent to one role-specific model call."""

    role: str
    sections: tuple[PromptSection, ...]
    rendered: str
    total_bytes: int
    static_prompt_bytes: int = 0
    dynamic_payload_bytes: int = 0
    budget_bytes: int = 0
    budget_overrun: bool = False
    omitted_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        actual = len(_encoded(self.rendered))
        if actual != self.total_bytes:
            raise ValueError("prompt payload total_bytes does not match rendered")
        if self.static_prompt_bytes < 0 or self.dynamic_payload_bytes < 0:
            raise ValueError("prompt payload byte counts must not be negative")

    def diagnostics(self) -> dict[str, object]:
        """Return secret-free, stable diagnostics for this payload."""

        return {
            "role": self.role,
            "prompt_bytes": self.total_bytes,
            "static_prompt_bytes": self.static_prompt_bytes,
            "dynamic_payload_bytes": self.dynamic_payload_bytes,
            "budget_bytes": self.budget_bytes,
            "budget_overrun": self.budget_overrun,
            "sections": [
                {
                    "name": section.name,
                    "bytes": section.byte_count,
                    "sha256": section.sha256,
                    "authority": section.authority,
                    "truncated": section.truncated,
                }
                for section in self.sections
            ],
            "omitted_sections": list(self.omitted_sections),
        }


def write_prompt_diagnostics(
    directory: str | Path,
    payload: PromptPayload,
    *,
    filename: str = "prompt.diagnostics.json",
) -> Path:
    """Persist one payload's secret-free diagnostics atomically."""

    target = Path(directory) / filename
    atomic_write_text(
        target,
        json.dumps(payload.diagnostics(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return target


def payload_for_rendered_request(
    role: str, rendered: str, *, budget_bytes: int = 0
) -> PromptPayload:
    """Wrap an older renderer so its model call still has diagnostics."""

    section = PromptSection.create("rendered_request", rendered, True)
    total = len(_encoded(rendered))
    return PromptPayload(
        role=role,
        sections=(section,),
        rendered=rendered,
        total_bytes=total,
        static_prompt_bytes=0,
        dynamic_payload_bytes=total,
        budget_bytes=budget_bytes,
        budget_overrun=bool(budget_bytes and total > budget_bytes),
    )


def _render_template(template: str, values: Mapping[str, str]) -> str:
    """Single, non-recursive placeholder substitution."""

    import re

    return re.sub(
        r"\{\{[A-Z0-9_]+\}\}",
        lambda match: values.get(match.group(0), match.group(0)),
        template,
    )


def _payload_from_template(
    *,
    role: str,
    template: str,
    sections: Sequence[PromptSection],
    placeholders: Mapping[str, str],
    budget_bytes: int,
    secondary_order: Sequence[str],
) -> PromptPayload:
    """Apply a deterministic budget without ever truncating authority."""

    if not isinstance(template, str):
        raise TypeError("prompt template must be a string")
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise TypeError("prompt budget must be an integer")
    if budget_bytes < 0:
        raise ValueError("prompt budget must not be negative")

    by_name = {section.name: section for section in sections}
    if len(by_name) != len(sections):
        raise ValueError("prompt section names must be unique")

    def render(current: Mapping[str, PromptSection]) -> str:
        values = dict(placeholders)
        for name, placeholder in placeholders.items():
            # ``placeholders`` is placeholder -> section name.
            section = current.get(placeholder)
            if section is not None:
                values[name] = section.text
        return _render_template(template, values)

    current = dict(by_name)
    omitted: list[str] = []
    rendered = render(current)
    # The static scaffold is the request with every dynamic value empty.  It
    # includes labels/tags and fixed policy text, but no user/run data.
    empty_values = dict(placeholders)
    for placeholder in empty_values:
        empty_values[placeholder] = ""
    static_rendered = _render_template(template, empty_values)
    static_bytes = len(_encoded(static_rendered))

    if budget_bytes:
        # Spend the available bytes on all authority first, regardless of
        # where a section occurs in the template.  Secondary evidence is then
        # truncated in declared order and finally omitted.
        secondary_names = [
            name for name in secondary_order
            if name in current and not current[name].authority
        ]
        secondary_names.extend(
            section.name
            for section in sections
            if not section.authority
            and section.name not in secondary_names
        )

        # Remove secondary sections one at a time only after shrinking their
        # text.  Authority sections are never changed.
        for name in secondary_names:
            rendered = render(current)
            if len(_encoded(rendered)) <= budget_bytes:
                break
            section = current[name]
            before = len(_encoded(rendered))
            # Estimate the bytes available by removing this section's current
            # value from the rendered request.  A second pass verifies the
            # result, so placeholder/tag differences cannot cause overflow.
            excess = before - budget_bytes
            keep = max(0, section.byte_count - excess)
            if keep < section.byte_count:
                marker_budget = max(0, keep - len(_encoded(_TRUNCATION_MARKER)))
                shortened = _utf8_prefix(section.text, marker_budget) + (
                    _TRUNCATION_MARKER if marker_budget < section.byte_count else ""
                )
                current[name] = PromptSection.create(
                    name, shortened, section.authority, truncated=True
                )
            rendered = render(current)
            if len(_encoded(rendered)) > budget_bytes:
                current[name] = PromptSection.create(
                    name, "", section.authority, truncated=True
                )
                omitted.append(name)

        # If several secondary sections remain and the first pass still
        # exceeds the budget, omit the remaining secondary sections from the
        # end in a stable order.  This is never applied to authority.
        for name in reversed(secondary_names):
            rendered = render(current)
            if len(_encoded(rendered)) <= budget_bytes:
                break
            if current[name].text:
                current[name] = PromptSection.create(
                    name, "", current[name].authority, truncated=True
                )
                if name not in omitted:
                    omitted.append(name)

        rendered = render(current)

    # Keep sections in the declaration order, with their actual injected
    # contents and hashes after bounding decisions.
    final_sections = tuple(current[section.name] for section in sections)
    total = len(_encoded(rendered))
    # This is the byte delta between the actual request and the same fixed
    # scaffold with dynamic values empty.  It remains correct when one
    # catalogue is intentionally rendered into two role-labelled slots.
    dynamic = max(0, total - static_bytes)
    return PromptPayload(
        role=role,
        sections=final_sections,
        rendered=rendered,
        total_bytes=total,
        static_prompt_bytes=static_bytes,
        dynamic_payload_bytes=dynamic,
        budget_bytes=budget_bytes,
        budget_overrun=bool(budget_bytes and total > budget_bytes),
        omitted_sections=tuple(dict.fromkeys(omitted)),
    )


def _section(name: str, text: str, authority: bool) -> PromptSection:
    return PromptSection.create(name, text, authority)


def build_planner_payload(
    *,
    spec: str,
    repository_identity: str,
    discovery_context: str,
    trusted_check_catalogue: str,
    planning_constraints: str = "NONE",
    template: str | None = None,
    budget_bytes: int = 0,
) -> PromptPayload:
    """Build the planner contract from current-run material only."""

    using_default_template = template is None
    if using_default_template:
        template = _default_template("planner_v2.txt")
    # These protocol constants are harness-owned and are not dynamic role
    # context.  Keep direct callers deterministic without requiring them to
    # know the v2 parser limits.
    template = (
        template.replace("{{DEFAULT_CHECK_IDS}}", "NONE")
        .replace("{{MAX_STEPS}}", "8")
        .replace("{{LAST_STEP_ID}}", "S08")
        .replace("{{MAX_STEP_CONTRACT_CHARS}}", "5000")
        .replace("{{MAX_READ_PATHS_PER_STEP}}", "8")
    )
    if using_default_template:
        template = template.replace("{{PLANNING_CONSTRAINTS}}", "NONE")
    sections = (
        _section("spec", spec, True),
        _section("repository_identity", repository_identity, True),
        _section("discovery_context", discovery_context, False),
        _section("trusted_check_catalogue", trusted_check_catalogue, True),
        _section("planning_constraints", planning_constraints, False),
    )
    placeholders = {
        "{{SPEC}}": "spec",
        "{{REPOSITORY}}": "repository_identity",
        "{{CONTEXT}}": "discovery_context",
        "{{CHECK_CATALOG}}": "trusted_check_catalogue",
        "{{PLANNING_CONSTRAINTS}}": "planning_constraints",
    }
    return _payload_from_template(
        role="planner", template=template, sections=sections,
        placeholders=placeholders, budget_bytes=budget_bytes,
        secondary_order=("discovery_context", "planning_constraints"),
    )


def build_implementer_payload(
    *,
    original_spec: str = "",
    step_objective: str,
    read_set: str,
    mutable_scope: str,
    repository_instructions: str = "",
    verify_instructions: str = "",
    step_title: str = "",
    step_identity: str | None = None,
    write_set: str = "NONE",
    create_set: str = "NONE",
    delete_set: str = "NONE",
    instructions: str | None = None,
    verify_contract: str | None = None,
    forbidden_contract: str | None = None,
    template: str | None = None,
    budget_bytes: int = 0,
    retry_addendum: str = "",
) -> PromptPayload:
    """Build one worker contract; no full plan or run history is accepted."""

    if template is None:
        template = _default_template("implementer.txt")
    effective_identity = step_identity if step_identity is not None else step_title
    effective_instructions = (
        instructions if instructions is not None else repository_instructions
    )
    effective_verify = verify_contract if verify_contract is not None else verify_instructions
    effective_forbidden = forbidden_contract if forbidden_contract is not None else ""
    sections = (
        _section("original_spec", original_spec, True),
        _section("step_identity", effective_identity, True),
        _section("step_objective", step_objective, True),
        _section("read_set", read_set, True),
        _section("write_set", write_set, True),
        _section("create_set", create_set, True),
        _section("delete_set", delete_set, True),
        _section("mutable_scope", mutable_scope, True),
        _section("instructions", effective_instructions, True),
        _section("verify_contract", effective_verify, True),
        _section("forbidden_contract", effective_forbidden, True),
        _section("retry_addendum", retry_addendum, False),
    )
    placeholders = {
        "{{SPEC}}": "original_spec",
        "{{STEP_IDENTITY}}": "step_identity",
        "{{STEP_OBJECTIVE}}": "step_objective",
        "{{READ_SET}}": "read_set",
        "{{WRITE_SET}}": "write_set",
        "{{CREATE_SET}}": "create_set",
        "{{DELETE_SET}}": "delete_set",
        "{{MUTABLE_SCOPE}}": "mutable_scope",
        "{{INSTRUCTIONS}}": "instructions",
        "{{VERIFY_CONTRACT}}": "verify_contract",
        "{{FORBIDDEN_CONTRACT}}": "forbidden_contract",
        "{{RETRY_ADDENDUM}}": "retry_addendum",
    }
    return _payload_from_template(
        role="implementer", template=template, sections=sections,
        placeholders=placeholders, budget_bytes=budget_bytes,
        secondary_order=("retry_addendum",),
    )


__all__ = [
    "PromptSection",
    "PromptPayload",
    "build_planner_payload",
    "build_implementer_payload",
    "write_prompt_diagnostics",
    "payload_for_rendered_request",
]
