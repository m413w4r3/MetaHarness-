"""The repository facts a META PLAN v2 is bound to before approval.

Mechanical path classification -- a CREATE_SET entry for a path the tree
already holds, a WRITE_SET entry for a path that is not there, a DELETE_SET
entry for a path that never existed -- is not a planning decision: Git answers
it, and :mod:`metaharness.planning.normalization` normalizes it.  What remains
here is the repository boundary itself: the Git-backed facts table the
normalizer reads, the contradictions no deterministic rule can settle, and the
evidence a planner is allowed to see.

This module never reads the working tree, never scans the repository and never
decides architecture: it reads one immutable tree object and hands the planner
bounded facts so it can re-plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .gitops import path_exists_in_tree
from .models import NO_MUTATION_REMAINS, PlanDecision, TaskPlanV2
from .planning.normalization import (
    TreeFacts,
    normalize_plan_contracts as normalize_plan_contracts_for_tree,
    plan_contradictions,
)
from .result import atomic_write_text
from .usage import PLANNER_ATTEMPTS_DIR

PLAN_REPOSITORY_PRECONDITION_INVALID = "PLAN_REPOSITORY_PRECONDITION_INVALID"
PRECONDITION_ARTIFACT = "repository_preconditions.json"
MAX_PREVIOUS_PLAN_CHARS = 64 * 1024
MAX_BLOCKERS_CHARS = 4 * 1024
_MAX_PATHS_PER_GROUP = 8
_MAX_GROUPS = 32
# The one remaining contradiction kind, in report order, and the stable
# lowercase token each contradiction code is reported under.  Every mechanical
# path classification is normalized before this module is consulted.
_KINDS = ("no_mutation",)
_KIND_OF_CONTRADICTION = {NO_MUTATION_REMAINS: _KINDS[0]}
# Never read, even when a plan or SPEC names them explicitly.
_SENSITIVE_PARTS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".cache",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".gradle", ".terraform", "secrets", ".secrets", ".ssh", ".gnupg", ".aws",
})
_SENSITIVE_NAMES = frozenset({
    ".npmrc", ".pypirc", ".netrc", ".htpasswd", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "credentials", "credentials.json",
})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


class PlanRepositoryPreconditionError(ValueError):
    """A plan's change sets are impossible against its start tree."""

    code = PLAN_REPOSITORY_PRECONDITION_INVALID

    def __init__(self, violations: Sequence["PathPreconditionViolation"]):
        self.violations = tuple(violations)
        super().__init__(render_violations(self.violations, separator="; "))


@dataclass(frozen=True)
class PathPreconditionViolation:
    step_id: str
    kind: str
    path: str


@dataclass(frozen=True)
class RepositoryPreconditions:
    """The repository and immutable tree object a plan starts from."""

    repo: Path
    start_tree_sha: str


def is_sensitive_repository_path(path: str) -> bool:
    """True for paths whose content must never be put in a prompt."""

    parts = PurePosixPath(path).parts
    if any(part.lower() in _SENSITIVE_PARTS for part in parts):
        return True
    name = parts[-1].lower() if parts else ""
    return (
        name == ".env" or name.startswith(".env.")
        or name in _SENSITIVE_NAMES
        or "secret" in name
        or name.endswith(_SENSITIVE_SUFFIXES)
    )


def repository_tree_facts(repo: Path, tree_sha: str) -> TreeFacts:
    """The ``path -> exists`` facts of one immutable tree object.

    Exactly the ``path_exists_in_tree`` semantics the runtime already uses, so
    planning and execution can never disagree on what "exists" means.
    """

    cache: dict[str, bool] = {}

    def exists(path: str) -> bool:
        if path not in cache:
            cache[path] = path_exists_in_tree(repo, tree_sha, path)
        return cache[path]

    return TreeFacts(exists)


def normalize_plan_contracts(
    repo: Path, start_tree_sha: str, plan: TaskPlanV2,
) -> TaskPlanV2:
    """Apply every deterministic contract normalization to *plan*.

    The steps are normalized in order against the tree each of them really
    starts from: the start tree for the first one, then the logical tree the
    earlier CREATE/WRITE/DELETE sections authorize.  The returned plan carries
    the record of every rule applied; no model, no Git mutation and no
    architecture decision is involved.
    """

    if plan.decision is not PlanDecision.READY:
        return plan
    normalized = normalize_plan_contracts_for_tree(plan, repository_tree_facts(repo, start_tree_sha))
    return normalized.plan


def plan_repository_violations(plan: TaskPlanV2) -> tuple[PathPreconditionViolation, ...]:
    """The contradictions no deterministic normalization can resolve.

    A step whose declared mutations are all impossible against its start tree
    -- a DELETE of an absent path, for instance -- has no effect left.  The
    harness cannot invent one: that decision belongs to the planner.
    """

    if plan.decision is not PlanDecision.READY:
        return ()
    return tuple(
        PathPreconditionViolation(step_id, _KIND_OF_CONTRADICTION.get(code, code), "")
        for step_id, code in plan_contradictions(plan)
    )


def validate_plan_repository_topology(
    repo: Path, start_tree_sha: str, plan: TaskPlanV2,
) -> TaskPlanV2:
    """Normalize *plan* against its start tree and return the effective plan.

    Raises :class:`PlanRepositoryPreconditionError` only for what stays
    impossible after normalization.
    """

    effective = normalize_plan_contracts(repo, start_tree_sha, plan)
    violations = plan_repository_violations(effective)
    if violations:
        raise PlanRepositoryPreconditionError(violations)
    return effective


def render_violations(
    violations: Sequence[PathPreconditionViolation], *, separator: str = "\n",
) -> str:
    """``step=S01 no_mutation`` lines, bounded and deterministic."""

    groups: dict[tuple[str, str], list[str]] = {}
    for item in violations:
        groups.setdefault((item.step_id, item.kind), []).append(item.path)
    def rank(kind: str) -> tuple[int, str]:
        return (_KINDS.index(kind), "") if kind in _KINDS else (len(_KINDS), kind)

    ordered = sorted(groups.items(), key=lambda entry: (entry[0][0], rank(entry[0][1])))
    lines = []
    for (step_id, kind), paths in ordered[:_MAX_GROUPS]:
        if not any(paths):
            lines.append(f"step={step_id} {kind}")
            continue
        shown = ",".join(paths[:_MAX_PATHS_PER_GROUP])
        extra = len(paths) - _MAX_PATHS_PER_GROUP
        lines.append(f"step={step_id} {kind}={shown}" + (f" (+{extra} more)" if extra > 0 else ""))
    if len(ordered) > _MAX_GROUPS:
        lines.append(f"(+{len(ordered) - _MAX_GROUPS} more violation groups)")
    return separator.join(lines)


def violations_payload(
    start_tree_sha: str, violations: Sequence[PathPreconditionViolation],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reason": PLAN_REPOSITORY_PRECONDITION_INVALID,
        "start_tree_sha": start_tree_sha,
        "violations": [
            {"step_id": item.step_id, "kind": item.kind, "path": item.path}
            for item in violations
        ],
    }


def render_precondition_correction(
    violations: Sequence[PathPreconditionViolation],
    *,
    previous_raw: str,
) -> str:
    """The MetaHarness authority section appended to the correction request."""

    previous = previous_raw
    if len(previous) > MAX_PREVIOUS_PLAN_CHARS:
        previous = previous[:MAX_PREVIOUS_PLAN_CHARS] + "\n[previous plan truncated]"
    parts = [
        "PLAN REPOSITORY CONTRACT ERRORS",
        "Authority: MetaHarness deterministic validation of the previous META PLAN",
        "against the repository tree the plan starts from. These are facts, not",
        "suggestions.",
        "",
        render_violations(violations),
        "",
        "CORRECTION RULES",
        "- Re-emit one COMPLETE META PLAN v2 using exactly the protocol above.",
        "  Never answer with a patch, a diff or a partial plan.",
        "- Keep the same SPEC; do not change its requirements.",
        "- Give every step at least one mutation that is possible against the",
        "  tree that step starts from: it must write, create or delete a path",
        "  the earlier steps really leave in the state the step expects.",
        "- Do not work around an error by widening scope; add no path the SPEC",
        "  does not need.",
        "- If no coherent plan satisfies these facts, return BLOCKED.",
        "",
        "PREVIOUS REJECTED META PLAN (reference only; do not patch)",
        previous.rstrip("\n"),
        "END PREVIOUS REJECTED META PLAN",
    ]
    return "\n".join(parts) + "\n"


def archive_rejected_planner_attempt(
    directory: Path,
    names: Sequence[str],
    *,
    start_tree_sha: str,
    violations: Sequence[PathPreconditionViolation],
) -> Path:
    """Move one rejected planner exchange to ``planner-attempts/NN/``.

    The rejected answer keeps its request, raw text and usage for audit, and
    can never be mistaken for the current plan authority.
    """

    root = directory / PLANNER_ATTEMPTS_DIR
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    target = root / f"{index:02d}"
    target.mkdir(parents=True)
    for name in names:
        source = directory / name
        if source.is_file():
            source.replace(target / name)
    atomic_write_text(
        target / PRECONDITION_ARTIFACT,
        json.dumps(violations_payload(start_tree_sha, violations), ensure_ascii=False, indent=2) + "\n",
    )
    return target


__all__ = [
    "MAX_PREVIOUS_PLAN_CHARS",
    "PLANNER_ATTEMPTS_DIR",
    "PLAN_REPOSITORY_PRECONDITION_INVALID",
    "PRECONDITION_ARTIFACT",
    "PathPreconditionViolation",
    "PlanRepositoryPreconditionError",
    "RepositoryPreconditions",
    "archive_rejected_planner_attempt",
    "is_sensitive_repository_path",
    "normalize_plan_contracts",
    "plan_repository_violations",
    "render_precondition_correction",
    "render_violations",
    "repository_tree_facts",
    "validate_plan_repository_topology",
    "violations_payload",
]
