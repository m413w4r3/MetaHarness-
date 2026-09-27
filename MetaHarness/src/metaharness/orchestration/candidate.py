"""The candidate sub-domain: candidate identity, paths and the accepted chain."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from .pipeline_v2 import PipelineFailure, candidate_dir
from .shared import CandidatePushError, _json_text, _read_json_artifact
from ..gitops import (
    GitError,
    RepositoryReference,
    commit_parents,
    current_head,
    normalize_github_web_url,
    resolve_tree,
)
from ..result import atomic_write_text
from ..models import GateStage


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
    *, commit_sha: str, tree_sha: str, parent_sha: str | None, branch: str,
    remote: str, immutable_url: str | None, gate_stage: str,
    remote_sha: str | None = None, pushed_at: str | None = None,
    remote_status: str = "pending", no_change: bool = False,
) -> dict[str, Any]:
    return {
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "parent_sha": parent_sha,
        "gate_stage": gate_stage,
        "branch": branch,
        "remote_branch": branch,
        "remote": remote,
        "remote_sha": remote_sha,
        "immutable_commit_url": immutable_url,
        "pushed_at": pushed_at,
        "remote_status": remote_status,
        "no_change": no_change,
    }


def accepted_chain_records(run_dir: Path) -> tuple[dict[str, Any], ...]:
    """Read the durable accepted commit chain of a run (empty before any)."""

    chain_path = run_dir / "accepted-chain.json"
    if not chain_path.is_file():
        return ()
    chain = _read_json_artifact(chain_path)
    if isinstance(chain, dict):
        chain = chain.get("commits")
    if not isinstance(chain, list) or not all(
        isinstance(item, dict) for item in chain
    ):
        return ()
    return tuple(chain)


class CandidateLifecycle:
    """Own candidate identity, durable candidate state and candidate push."""

    def __init__(
        self,
        *,
        staging_remote: str,
        push_tree: Callable[..., dict[str, Any]],
        cycle_update: Callable[..., None],
    ) -> None:
        self._staging_remote = staging_remote
        self._push_tree = push_tree
        self._cycle_update = cycle_update

    def create(
        self, store: Any, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: Any,
    ) -> dict[str, Any]:
        if evidence is None or not evidence.deterministic_passed:
            raise PipelineFailure(
                "DETERMINISTIC_GATE_FAILED", ", ".join(getattr(evidence, "failures", ())),
            )
        return self._from_git(store, ctx, cycle_plan.cycle.number, stage)

    def load_from_git(self, store: Any, ctx: Any, number: int) -> dict[str, Any]:
        """Rebuild candidate metadata from the branch commit, never its report."""

        return self._from_git(store, ctx, number, GateStage.POST_IMPLEMENTATION)

    def _from_git(self, store: Any, ctx: Any, number: int, stage: GateStage) -> dict[str, Any]:
        worktree = ctx.info.worktree
        head = current_head(worktree)
        tree = resolve_tree(worktree, head)
        no_change = head == ctx.info.base_sha
        parents = () if no_change else commit_parents(worktree, head)
        if not no_change and len(parents) != 1:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "candidate HEAD has no single parent")
        payload = _candidate_commit_payload(
            commit_sha=head,
            tree_sha=tree,
            parent_sha=None if no_change else parents[0],
            branch=ctx.info.branch,
            remote=self._staging_remote,
            immutable_url=_commit_web_url(ctx.repository_reference, head),
            gate_stage=stage.value,
            no_change=no_change,
        )
        atomic_write_text(_candidate_commit_path(ctx.run_dir, number), _json_text(payload))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"{number:03d}"] = payload
        store.update_metadata(
            candidate=candidate_state,
            candidate_commit_sha=head, approved_tree_sha=tree,
        )
        return payload

    def push(
        self, store: Any, ctx: Any, number: int, candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if candidate.get("no_change") is True:
            skipped = {**candidate, "remote_sha": None, "pushed_at": None, "remote_status": "not_required"}
            atomic_write_text(_candidate_commit_path(ctx.run_dir, number), _json_text(skipped))
            candidates = dict(store.load().get("candidate") or {})
            candidates[f"{number:03d}"] = skipped
            store.update_metadata(candidate=candidates)
            self._cycle_update(store, number, status="candidate_no_change")
            return skipped
        try:
            pushed = self._push_tree(
                run_dir=ctx.run_dir, info=ctx.info, cycle=number,
                candidate=dict(candidate), store=store,
            )
        except CandidatePushError as exc:
            raise PipelineFailure("PUSH_FAILED", "candidate push did not complete") from exc
        self._cycle_update(
            store,
            number,
            status=(
                "candidate_pushed"
                if pushed.get("remote_status") == "available"
                else "candidate_ready"
            ),
        )
        return pushed


class CandidateRemoteStaging:
    """Stage one exact candidate on the remote under a bounded retry budget.

    Remote proof never replaces local identity: the remote tip must equal the
    candidate SHA exactly.  An optional remote degrades to a warning; a
    required remote that stays unavailable is a waiting condition.
    """

    def __init__(
        self,
        recovery: Any,
        *,
        store: Any,
        attempts: int,
        emit: Callable[..., None],
        remote: str,
        remote_required: bool,
        remote_tip: Callable[..., str | None],
        push: Callable[..., None],
    ) -> None:
        self._recovery = recovery
        self._store = store
        self._attempts = attempts
        self._emit = emit
        self._remote = remote
        self._remote_required = remote_required
        self._remote_tip = remote_tip
        self._push = push

    def stage(
        self, *, run_dir: Path, info: Any, cycle: int, candidate: dict[str, Any],
    ) -> dict[str, Any]:
        store = self._store
        previous_candidate_sha = None
        if cycle > 1:
            previous = _read_json_artifact(_candidate_commit_path(run_dir, cycle - 1))
            if isinstance(previous, dict):
                previous_candidate_sha = previous.get("commit_sha")
        key = self._recovery.budget_key("candidate-push", f"{cycle:03d}")
        tree = candidate.get("tree_sha")
        last_failure: Exception | None = None
        while True:
            try:
                remote_tip = self._remote_tip(
                    info.source_repo, remote=self._remote, branch=info.branch,
                )
                if remote_tip not in {None, candidate["commit_sha"], previous_candidate_sha}:
                    return self._unavailable(
                        run_dir, info, cycle, candidate, "remote branch identity mismatch",
                    )
                if remote_tip != candidate["commit_sha"]:
                    self._push(
                        info.worktree, remote=self._remote,
                        branch=info.branch, commit_sha=candidate["commit_sha"],
                    )
                verified_tip = self._remote_tip(
                    info.source_repo, remote=self._remote, branch=info.branch,
                )
                if verified_tip != candidate["commit_sha"]:
                    return self._unavailable(
                        run_dir, info, cycle, candidate, "remote candidate SHA mismatch",
                    )
                break
            except (GitError, OSError, ValueError) as exc:
                last_failure = exc
                admission = self._recovery.admit(
                    key, reason="PUSH_FAILED", budget=self._attempts,
                    phase="candidate_push", cycle=cycle, tree_before=tree, tree_after=tree,
                )
                if not admission.admitted:
                    unavailable = self._unavailable(
                        run_dir, info, cycle, candidate, "transport unavailable",
                    )
                    if self._remote_required:
                        raise CandidatePushError(
                            "PUSH_FAILED: candidate push did not complete"
                        ) from last_failure
                    return unavailable

        candidate = {
            **candidate,
            "remote": self._remote,
            "remote_branch": info.branch,
            "remote_sha": candidate["commit_sha"],
            "pushed_at": candidate.get("pushed_at") or datetime.now(timezone.utc).isoformat(),
            "remote_status": "available",
        }
        atomic_write_text(_candidate_commit_path(run_dir, cycle), _json_text(candidate))
        candidate_state = dict(store.load().get("candidate") or {})
        candidate_state[f"{cycle:03d}"] = candidate
        store.update_metadata(
            candidate=candidate_state,
            remote_branch=info.branch,
            remote_sha=candidate["commit_sha"],
        )
        self._emit(
            "candidate.pushed", phase="publication", cycle=cycle,
            data={
                "parent_sha": candidate.get("parent_sha"),
                "commit_sha": candidate.get("commit_sha"),
                "tree_sha": candidate.get("tree_sha"),
                "remote": candidate.get("remote"),
                "branch": candidate.get("remote_branch") or info.branch,
                "remote_sha": candidate.get("remote_sha"),
                "pushed_at": candidate.get("pushed_at"),
                "remote_status": candidate.get("remote_status"),
            },
            once=True,
        )
        return candidate

    def _unavailable(
        self, run_dir: Path, info: Any, cycle: int, candidate: dict[str, Any], reason: str,
    ) -> dict[str, Any]:
        unavailable = {
            **candidate,
            "remote": self._remote,
            "remote_branch": info.branch,
            "remote_sha": None,
            "pushed_at": None,
            "remote_status": "unavailable",
        }
        atomic_write_text(_candidate_commit_path(run_dir, cycle), _json_text(unavailable))
        state = self._store.load()
        candidate_state = dict(state.get("candidate") or {})
        candidate_state[f"{cycle:03d}"] = unavailable
        self._store.update_metadata(
            candidate=candidate_state,
            remote_branch=info.branch,
            remote_sha=None,
        )
        self._emit(
            "candidate.push_unavailable",
            phase="publication",
            cycle=cycle,
            data={
                "warning": "candidate staging remote is unavailable",
                "reason": reason,
                "required": self._remote_required,
                "commit_sha": unavailable.get("commit_sha"),
                "tree_sha": unavailable.get("tree_sha"),
                "remote": self._remote,
                "branch": info.branch,
                "remote_status": "unavailable",
            },
        )
        return unavailable
