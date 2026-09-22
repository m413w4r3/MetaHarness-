"""The candidate sub-domain: candidate identity, paths and the accepted chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from .pipeline_v2 import candidate_dir
from .shared import _read_json_artifact
from ..gitops import (
    GitError,
    RepositoryReference,
    normalize_github_web_url,
    validate_linear_commit_chain,
)


def _commit_web_url(reference: RepositoryReference, commit_sha: str) -> str | None:
    if reference.web_url is None:
        return None
    try:
        normalized = normalize_github_web_url(reference.web_url)
    except ValueError:
        return None
    return f"{normalized}/commit/{commit_sha}" if normalized is not None else None


def _candidate_commit_path(run_dir: Path, cycle: int) -> Path:
    return candidate_dir(run_dir, cycle) / "commit.json"


def _candidate_commit_payload(
    *, commit_sha: str, tree_sha: str, parent_sha: str, branch: str,
    remote: str, immutable_url: str | None, gate_stage: str,
    pushed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "parent_sha": parent_sha,
        "gate_stage": gate_stage,
        "branch": branch,
        "remote_branch": branch,
        "remote": remote,
        "remote_sha": commit_sha if pushed_at is not None else None,
        "immutable_commit_url": immutable_url,
        "pushed_at": pushed_at,
    }


def accepted_chain_records(run_dir: Path) -> tuple[dict[str, Any], ...]:
    """Read the durable accepted commit chain of a run (empty before any)."""

    chain_path = run_dir / "accepted-chain.json"
    if not chain_path.is_file():
        return ()
    chain = _read_json_artifact(chain_path)
    if isinstance(chain, dict):
        chain = chain.get("commits")
    if not isinstance(chain, list) or not chain or not all(
        isinstance(item, dict) for item in chain
    ):
        raise GitError("accepted commit chain artifact is malformed")
    return tuple(chain)


def validate_accepted_chain(
    worktree: Path,
    *,
    run_dir: Path,
    base_sha: str,
    tip_sha: str,
    approved_tree_sha: str,
) -> tuple[str, ...]:
    """Validate the exact durable chain used by candidate publication."""

    return validate_linear_commit_chain(
        worktree,
        base_sha=base_sha,
        tip_sha=tip_sha,
        accepted_commits=accepted_chain_records(run_dir),
        approved_tree_sha=approved_tree_sha,
    )
