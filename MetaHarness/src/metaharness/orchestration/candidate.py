"""The candidate sub-domain: candidate identity, paths and the accepted chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from .pipeline_v2 import PipelineFailure, candidate_dir, gate_acceptance_path
from .shared import CandidatePushError, _json_text, _read_json_artifact
from ..gitops import (
    GitError,
    RepositoryReference,
    commit_parents,
    current_head,
    normalize_github_web_url,
    resolve_tree,
    validate_linear_commit_chain,
)
from ..result import atomic_write_text
from ..models import GateStage, RunStatus


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


class CandidateLifecycle:
    """Own candidate identity, durable candidate state and candidate push."""

    def __init__(
        self,
        *,
        publish_remote: str,
        authorize_tree: Callable[..., None],
        push_tree: Callable[..., dict[str, Any]],
        cycle_update: Callable[..., None],
    ) -> None:
        self._publish_remote = publish_remote
        self._authorize_tree = authorize_tree
        self._push_tree = push_tree
        self._cycle_update = cycle_update

    def create(
        self, store: Any, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: Any,
    ) -> dict[str, Any]:
        if not evidence.deterministic_passed:
            raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
        worktree = ctx.info.worktree
        head = current_head(worktree)
        self._authorize_tree(evidence, worktree, head, ctx.branch_ref)
        if resolve_tree(worktree, head) != evidence.staged_tree_sha:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                "candidate HEAD tree differs from gate evidence",
            )
        acceptance = _read_json_artifact(
            gate_acceptance_path(ctx.run_dir, cycle_plan.cycle, stage)
        )
        if (
            not isinstance(acceptance, dict)
            or acceptance.get("stage") != stage.value
            or acceptance.get("tree_sha") != evidence.staged_tree_sha
            or acceptance.get("commit_sha") != head
        ):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                "final gate acceptance is missing or stale",
            )
        parents = commit_parents(worktree, head)
        if len(parents) != 1:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "candidate HEAD has no single parent")
        path = _candidate_commit_path(ctx.run_dir, cycle_plan.cycle.number)
        stored = _read_json_artifact(path)
        if stored is not None and (
            not isinstance(stored, dict)
            or stored.get("commit_sha") != head
            or stored.get("tree_sha") != evidence.staged_tree_sha
        ):
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "candidate artifact does not match accepted HEAD")
        payload = _candidate_commit_payload(
            commit_sha=head,
            tree_sha=evidence.staged_tree_sha,
            parent_sha=parents[0],
            branch=ctx.info.branch,
            remote=self._publish_remote,
            immutable_url=_commit_web_url(ctx.repository_reference, head),
            gate_stage=stage.value,
            pushed_at=stored.get("pushed_at") if isinstance(stored, dict) else None,
        )
        atomic_write_text(path, _json_text(payload))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"{cycle_plan.cycle.number:03d}"] = payload
        store.update(
            status=RunStatus.APPROVED, candidate=candidate_state,
            candidate_commit_sha=head, approved_tree_sha=evidence.staged_tree_sha,
            expected_head_sha=head, expected_parent_sha=parents[0],
            expected_tree_sha=evidence.staged_tree_sha, next_step_id=None,
        )
        return payload

    def push(
        self, store: Any, ctx: Any, number: int, candidate: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            pushed = self._push_tree(
                run_dir=ctx.run_dir, info=ctx.info, cycle=number,
                candidate=dict(candidate), store=store,
            )
        except CandidatePushError as exc:
            raise PipelineFailure("PUSH_FAILED", "candidate push did not complete") from exc
        self._cycle_update(store, number, status="candidate_pushed")
        return pushed
