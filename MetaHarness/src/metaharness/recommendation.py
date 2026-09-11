"""Advisory execution-profile recommendation for an approved plan."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .llm.chat import TextLLMResult
from .models import ModelProfile


class TextCompletionClient(Protocol):
    def complete(self, prompt: str) -> TextLLMResult | str:
        """Complete exactly one independent user message."""


class RecommendationError(ValueError):
    pass


class RecommendationParseError(RecommendationError):
    pass


@dataclass(frozen=True)
class ExecutionRecommendation:
    implementer_profile: str
    reviewer_profile: str
    rationale: str
    raw: str


_PROMPT_PATH = Path(__file__).with_name("prompts") / "recommender.txt"
_IMPLEMENTER_LABEL = "IMPLEMENTER_PROFILE"
_REVIEWER_LABEL = "REVIEWER_PROFILE"
_END = "END META EXECUTION RECOMMENDATION"
_HEADER = "META EXECUTION RECOMMENDATION v1"
_PROFILE_LINE = re.compile(r"^\s*(IMPLEMENTER_PROFILE|REVIEWER_PROFILE)\s*:(.*)$")
_RATIONALE_HEADING = re.compile(r"^\s*(?:#{1,6}\s+)?RATIONALE\s*$")


def render_profile_catalogue(profiles: Sequence[ModelProfile]) -> str:
    """Render only allowlisted, non-credential profile metadata."""

    rendered: list[str] = []
    for profile in profiles:
        if not isinstance(profile, ModelProfile):
            raise TypeError("profiles must contain ModelProfile values")
        lines = [
            "PROFILE",
            f"ID: {profile.id}",
            f"DISPLAY_NAME: {profile.display_name}",
            f"DRIVER: {profile.driver.value}",
            f"MODEL_LABEL: {profile.model}",
            f"SELECTION_MODE: {profile.selection_mode.value}",
            f"EFFORT: {profile.effort if profile.effort is not None else 'NONE'}",
            f"DESCRIPTION: {profile.description}",
            "STRENGTHS:",
            *(f"- {strength}" for strength in profile.strengths),
            f"COST_TIER: {profile.cost_tier}",
            f"LATENCY_TIER: {profile.latency_tier}",
            "END PROFILE",
        ]
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _completion_text(result: TextLLMResult | str) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if not isinstance(text, str):
        raise RecommendationError("recommender client did not return text")
    return text


def _replace_prompt_values(template: str, contract: str, implementers: str, reviewers: str) -> str:
    values = {
        "{{CONTRACT}}": contract,
        "{{IMPLEMENTERS}}": implementers,
        "{{REVIEWERS}}": reviewers,
    }
    return re.sub(r"\{\{CONTRACT\}\}|\{\{IMPLEMENTERS\}\}|\{\{REVIEWERS\}\}", lambda match: values[match.group(0)], template)


def _build_prompt(contract: str, implementers: str, reviewers: str) -> str:
    if not isinstance(contract, str):
        raise TypeError("contract must be a string")
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return _replace_prompt_values(template, contract, implementers, reviewers)


def _normalise_lines(raw: str) -> list[str]:
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def parse_execution_recommendation(
    raw: str,
    *,
    implementer_ids: frozenset[str],
    reviewer_ids: frozenset[str],
) -> ExecutionRecommendation:
    if not isinstance(raw, str) or not raw.strip():
        raise RecommendationParseError("recommendation response is empty")
    if not isinstance(implementer_ids, frozenset) or not isinstance(reviewer_ids, frozenset):
        raise TypeError("profile IDs must be frozensets")

    lines = _normalise_lines(raw)
    first = 0
    while first < len(lines) and not lines[first].strip():
        first += 1
    last = len(lines) - 1
    while last >= first and not lines[last].strip():
        last -= 1
    if first > last or lines[first].strip() != _HEADER:
        raise RecommendationParseError("missing recommendation header")

    end_positions = [index for index, line in enumerate(lines) if line.strip() == _END]
    if len(end_positions) != 1 or end_positions[0] <= first:
        raise RecommendationParseError("missing or duplicate recommendation end marker")
    end = end_positions[0]
    if any(line.strip() for line in lines[end + 1:]):
        raise RecommendationParseError("content after recommendation end marker")

    values: dict[str, str] = {}
    label_counts = {_IMPLEMENTER_LABEL: 0, _REVIEWER_LABEL: 0}
    rationale_positions: list[int] = []
    for index in range(first + 1, end):
        line = lines[index]
        match = _PROFILE_LINE.fullmatch(line)
        if match is not None:
            label = match.group(1)
            label_counts[label] += 1
            values[label] = match.group(2).strip()
        if _RATIONALE_HEADING.fullmatch(line):
            rationale_positions.append(index)
    if any(count != 1 for count in label_counts.values()):
        raise RecommendationParseError("recommendation must contain exactly one profile value of each kind")
    if len(rationale_positions) != 1:
        raise RecommendationParseError("recommendation rationale is missing or duplicated")
    rationale_start = rationale_positions[0]
    rationale = "\n".join(lines[rationale_start + 1:end]).strip()
    if not rationale:
        raise RecommendationParseError("recommendation rationale is empty")
    if len(rationale) > 2000:
        raise RecommendationParseError("recommendation rationale is too long")

    implementer = values[_IMPLEMENTER_LABEL]
    reviewer = values[_REVIEWER_LABEL]
    if not implementer or "\n" in implementer or implementer not in implementer_ids:
        raise RecommendationParseError("unknown implementer profile")
    if not reviewer or "\n" in reviewer or reviewer not in reviewer_ids:
        raise RecommendationParseError("unknown reviewer profile")
    if any(
        line.strip() and not _PROFILE_LINE.fullmatch(line) and not _RATIONALE_HEADING.fullmatch(line)
        and index not in rationale_positions and line.strip() != _END
        for index, line in enumerate(lines[first + 1:end], start=first + 1)
        if index < rationale_start
    ):
        raise RecommendationParseError("unexpected content in recommendation header")
    return ExecutionRecommendation(implementer, reviewer, rationale, raw)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class ExecutionRecommender:
    def __init__(self, client: TextCompletionClient) -> None:
        self.client = client

    def recommend(
        self,
        contract: str,
        implementer_profiles: Sequence[ModelProfile],
        reviewer_profiles: Sequence[ModelProfile],
        *,
        artifacts_dir: Path,
    ) -> ExecutionRecommendation:
        implementers = render_profile_catalogue(implementer_profiles)
        reviewers = render_profile_catalogue(reviewer_profiles)
        prompt = _build_prompt(contract, implementers, reviewers)
        directory = Path(artifacts_dir)
        _atomic_write_text(directory / "execution_recommendation.request.txt", prompt)
        raw = _completion_text(self.client.complete(prompt))
        _atomic_write_text(directory / "execution_recommendation.raw.md", raw)
        recommendation = parse_execution_recommendation(
            raw,
            implementer_ids=frozenset(profile.id for profile in implementer_profiles),
            reviewer_ids=frozenset(profile.id for profile in reviewer_profiles),
        )
        _atomic_write_text(
            directory / "execution_recommendation.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "implementer_profile": recommendation.implementer_profile,
                    "reviewer_profile": recommendation.reviewer_profile,
                    "rationale": recommendation.rationale,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        return recommendation


def write_recommendation_error(artifacts_dir: Path, error: str) -> None:
    _atomic_write_text(Path(artifacts_dir) / "execution_recommendation.error.txt", error)


__all__ = [
    "ExecutionRecommendation",
    "ExecutionRecommender",
    "RecommendationError",
    "RecommendationParseError",
    "parse_execution_recommendation",
    "render_profile_catalogue",
    "write_recommendation_error",
]
