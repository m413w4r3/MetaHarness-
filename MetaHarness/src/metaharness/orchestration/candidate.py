"""The candidate sub-domain: candidate identity, paths and commit gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from .shared import (
    CommitBoundaryError,
    _is_object_id,
    _read_json_artifact,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    GitError,
    RepositoryReference,
    candidate_tree_sha,
    commit_parents,
    current_head,
    index_tree_sha,
    normalize_github_web_url,
    resolve_tree,
    status_porcelain,
    symbolic_head,
)
from ..models import (
    ReviewRoute,
    ReviewVerdict,
)
from ..planning import (
    PlanDecision,
    TaskPlan,
)
from ..review import (
    ReviewParseError,
    ReviewResult,
    blocking_finding_lines,
    parse_review,
)
from ..agent.base import AgentResult


def _status_has_unstaged_or_untracked(status: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for line in status:
        if line.startswith("?? "):
            problems.append(f"new untracked file: {line[3:]}")
        elif len(line) >= 2 and line[1] != " ":
            problems.append(f"unstaged change: {line}")
    return problems


def authorize_commit(
    *,
    plan: TaskPlan,
    agent_result: AgentResult,
    evidence: EvidenceBundle,
    review: ReviewResult,
    worktree: Path,
    base_sha: str,
    branch_ref: str,
) -> str:
    """Single gate in front of the only commit call; return the tree to commit.

    Every precondition is re-derived here, from primary evidence where
    possible (the reviewer's raw answer is parsed again), immediately before
    committing.  Any failure raises :class:`CommitBoundaryError`.
    """

    if plan.decision is not PlanDecision.READY:
        raise CommitBoundaryError("planner decision is not READY")
    if agent_result.timed_out or agent_result.exit_code != 0:
        raise CommitBoundaryError("implementation agent did not exit successfully")
    if not evidence.deterministic_passed or evidence.failures:
        raise CommitBoundaryError("deterministic gate did not pass")
    approved_tree = evidence.staged_tree_sha
    if not approved_tree:
        raise CommitBoundaryError("no reviewed tree identity was recorded")
    try:
        reparsed = parse_review(review.raw, deterministic_passed=True)
    except ReviewParseError as exc:
        raise CommitBoundaryError(f"reviewer answer does not authorize a commit: {exc}") from exc
    if review.verdict is not ReviewVerdict.PASS or reparsed.verdict is not ReviewVerdict.PASS:
        raise CommitBoundaryError("reviewer verdict is not PASS")
    if review.route is not ReviewRoute.NONE or reparsed.route is not ReviewRoute.NONE:
        raise CommitBoundaryError("reviewer route is not NONE")
    if blocking_finding_lines(review.raw):
        raise CommitBoundaryError("reviewer reported a MAJOR or BLOCKER finding")

    if symbolic_head(worktree) != branch_ref:
        raise CommitBoundaryError("worktree HEAD no longer points to the run branch")
    if current_head(worktree) != base_sha:
        raise CommitBoundaryError("HEAD changed after review")
    if index_tree_sha(worktree) != approved_tree:
        raise CommitBoundaryError("index changed after review")
    problems = _status_has_unstaged_or_untracked(status_porcelain(worktree))
    if problems:
        raise CommitBoundaryError("; ".join(problems))
    if candidate_tree_sha(worktree) != approved_tree:
        raise CommitBoundaryError("working tree differs from the reviewed tree")
    return approved_tree


def _commit_web_url(reference: RepositoryReference, commit_sha: str) -> str | None:
    if reference.web_url is None:
        return None
    try:
        normalized = normalize_github_web_url(reference.web_url)
    except ValueError:
        return None
    return f"{normalized}/commit/{commit_sha}" if normalized is not None else None


def _candidate_commit_path(run_dir: Path, cycle: int) -> Path:
    return run_dir / "candidate" / f"C{cycle:02d}" / "commit.json"


def _candidate_chain_parent(
    worktree: Path, run_dir: Path, base_sha: str, cycle: int, commit_sha: str,
) -> str:
    """The exact direct parent the published candidate of *cycle* must have.

    C01: ``BASE``.  C02: the persisted C01 candidate, itself exactly parented
    to ``BASE`` with its recorded tree.  Raises :class:`GitError` otherwise.
    """

    record = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
    if not isinstance(record, dict) or record.get("commit_sha") != commit_sha:
        raise GitError("candidate commit artifact does not match publication")
    expected_parent = base_sha
    if cycle == 2:
        c01 = _read_json_artifact(_candidate_commit_path(run_dir, 1))
        if not isinstance(c01, dict) or not _is_object_id(c01.get("commit_sha")):
            raise GitError("C01 candidate commit artifact is missing")
        if (
            c01.get("parent_sha") != base_sha
            or commit_parents(worktree, c01["commit_sha"]) != (base_sha,)
            or resolve_tree(worktree, c01["commit_sha"]) != c01.get("tree_sha")
        ):
            raise GitError("C01 candidate identity is not exact")
        expected_parent = c01["commit_sha"]
    if record.get("parent_sha") != expected_parent or commit_parents(worktree, commit_sha) != (expected_parent,):
        raise GitError("candidate commit parent is not the expected parent")
    return expected_parent


def _candidate_commit_payload(
    *, commit_sha: str, tree_sha: str, parent_sha: str, branch: str,
    remote: str, immutable_url: str | None, pushed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "parent_sha": parent_sha,
        "branch": branch,
        "remote": remote,
        "immutable_commit_url": immutable_url,
        "pushed_at": pushed_at,
    }
