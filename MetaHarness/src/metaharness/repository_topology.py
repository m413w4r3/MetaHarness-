"""Bounded, deterministic repository topology evidence for planner repairs.

MetaHarness knows the exact tracked paths of a tree.  When a planner answer
or a worker mismatch names a path that does not exist, this module lists the
tracked paths sharing its exact basename or its exact path suffix.  It never
chooses one: the planner stays the only author of a contract path, and a
wrong path is still rejected by the contract validator.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MAX_TOPOLOGY_REFERENCES = 16
MAX_TOPOLOGY_CANDIDATES = 5
MAX_TOPOLOGY_SCORE_PATHS = 50_000
_MAX_SCANNED_CHARS = 64 * 1024
# A path-like token with an extension: ``a/b/C.test.tsx`` or ``C.test.tsx``.
_PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_@./-])"
    r"((?:[A-Za-z0-9_@.-]+/)*[A-Za-z0-9_@-][A-Za-z0-9_@.-]*\.[A-Za-z][A-Za-z0-9]{1,9})"
    r"(?![A-Za-z0-9_@/-])"
)
# Contract preconditions a validator reports as ``label=path,path``.
_INVALID_PATHS = re.compile(r"\b(read_missing|write_missing|delete_missing)=([^\s]+)")


@dataclass(frozen=True)
class RepositoryTopology:
    """The exact tracked paths of one tree."""

    tree_sha: str
    paths: frozenset[str]

    @classmethod
    def from_tree(cls, repo: Path, tree_sha: str) -> "RepositoryTopology":
        from .gitops import tracked_files_in_tree

        return cls(tree_sha, frozenset(tracked_files_in_tree(repo, tree_sha)))

    def candidates(self, reference: str) -> tuple[str, ...]:
        """Tracked paths with the exact suffix, then the exact basename."""

        reference = reference.strip().strip("/")
        if not reference or reference in self.paths:
            return ()
        basename = reference.rsplit("/", 1)[-1]
        suffix = [
            path for path in sorted(self.paths)
            if "/" in reference and path.endswith("/" + reference)
        ]
        named = [
            path for path in sorted(self.paths)
            if path.rsplit("/", 1)[-1] == basename and path not in suffix
        ]
        return tuple((suffix + named)[:MAX_TOPOLOGY_CANDIDATES])

    def nearest(self, reference: str) -> tuple[str, ...]:
        """Return a small deterministic set of similar tracked paths.

        Exact-basename matches are ranked first. If the repository has no
        namesake, all tracked paths are considered. Scoring is bounded so a
        malformed planner answer cannot trigger unbounded prompt work.
        """

        reference = reference.strip().strip("/")[:512]
        if not reference or reference in self.paths:
            return ()
        basename = reference.rsplit("/", 1)[-1]
        same_name = [
            path for path in sorted(self.paths)
            if path.rsplit("/", 1)[-1] == basename
        ]
        pool = same_name or sorted(self.paths)
        pool = pool[:MAX_TOPOLOGY_SCORE_PATHS]
        ranked = sorted(
            pool,
            key=lambda path: (
                -SequenceMatcher(None, reference.casefold(), path.casefold(), autojunk=False).ratio(),
                path,
            ),
        )
        return tuple(ranked[:MAX_TOPOLOGY_CANDIDATES])

    def evidence(self, *texts: str, references: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Bounded entries for every path-like reference absent from the tree."""

        seen: list[str] = []
        for reference in (*references, *_references(*texts)):
            reference = reference.strip().strip("/")
            if reference and reference not in seen and reference not in self.paths:
                seen.append(reference)
            if len(seen) >= MAX_TOPOLOGY_REFERENCES:
                break
        entries = []
        explicit = set(references)
        for reference in seen:
            candidates = list(self.candidates(reference))
            # A bare word with an extension and no tracked namesake is prose.
            if not candidates and "/" not in reference and reference not in explicit:
                continue
            entries.append({
                "reference": reference,
                "basename": reference.rsplit("/", 1)[-1],
                "candidates": candidates,
            })
        return entries


def _references(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for match in _PATH_TOKEN.finditer((text or "")[:_MAX_SCANNED_CHARS]):
            token = match.group(1).rstrip(".")
            if token.startswith(".") or ".." in token.split("/") or token in found:
                continue
            found.append(token)
    return found


def invalid_paths(detail: str) -> list[str]:
    """The paths a contract precondition error reports as missing."""

    paths: list[str] = []
    for _label, values in _INVALID_PATHS.findall(detail or ""):
        for value in values.split(","):
            value = value.strip()
            if value and value not in paths:
                paths.append(value)
    return paths[:MAX_TOPOLOGY_REFERENCES]


def render_path_candidates(entries: Sequence[Mapping[str, Any]]) -> str:
    """``<basename>:`` then its tracked candidates; ``NONE`` when empty."""

    if not entries:
        return "NONE"
    blocks = []
    for entry in entries:
        candidates = entry.get("candidates") or []
        lines = [f"{entry['reference']}:"]
        lines += [f"  - {path}" for path in candidates] or ["  - NONE (no tracked path has this basename)"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_invalid_path_candidates(
    detail: str, topology: RepositoryTopology | None,
) -> str:
    """One ``INVALID PATH`` block per missing path of a rejected answer."""

    if topology is None:
        return ""
    blocks = []
    for path in invalid_paths(detail):
        candidates = topology.candidates(path)
        listed = "\n".join(f"  - {item}" for item in candidates) or "  - NONE"
        blocks.append(
            f"INVALID PATH:\n  {path}\n\nTRACKED CANDIDATES FOR BASENAME:\n{listed}"
        )
    return "\n\n".join(blocks)


def render_repository_path_facts(
    detail: str, topology: RepositoryTopology | None,
) -> str:
    """Render authoritative, bounded facts for invalid contract paths."""

    if topology is None:
        return ""
    invalid = invalid_paths(detail)[:8]
    if not invalid:
        return ""
    bad_paths = "\n".join(invalid)
    nearest: list[str] = []
    seen: set[str] = set()
    for path in invalid:
        for candidate in topology.nearest(path):
            if candidate not in seen:
                seen.add(candidate)
                nearest.append(candidate)
    nearest_text = "\n".join(nearest[:MAX_TOPOLOGY_CANDIDATES * 8]) or "NONE"
    return (
        "<REPOSITORY PATH FACTS>\n\n"
        f"INVALID\n{bad_paths}\n\n"
        f"NEAREST TRACKED PATHS\n{nearest_text}\n\n"
        "RULE\n"
        "READ/WRITE paths must come from tracked repository paths.\n"
        "Only paths explicitly authorized by CREATE_SET may be new.\n\n"
        "</REPOSITORY PATH FACTS>"
    )


def topology_payload(
    topology: RepositoryTopology, entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The bounded ``topology_evidence.json`` diagnostic; no file content."""

    return {
        "schema_version": 1,
        "tree_sha": topology.tree_sha,
        "tracked_path_count": len(topology.paths),
        "references": [dict(entry) for entry in entries],
    }


__all__ = [
    "MAX_TOPOLOGY_CANDIDATES", "MAX_TOPOLOGY_REFERENCES", "MAX_TOPOLOGY_SCORE_PATHS",
    "RepositoryTopology",
    "invalid_paths", "render_invalid_path_candidates", "render_path_candidates",
    "render_repository_path_facts", "topology_payload",
]
