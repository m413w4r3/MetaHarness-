"""The green-tree acceptance of one deterministic gate episode.

A gate only accepts the tree its own evidence measured: the accepted-gate
artifact binds the candidate tree, the commit that carries it, the mutable
authority in force and the durable evidence hash, so a resume re-derives every
one of those facts instead of trusting the file.  The module records the
accepted chain entry, updates the run metadata the next stage reads and emits
the acceptance trace; it never runs a check, a worker or a revision.
"""

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)
from ..commit_gate import CommitSafetyError, commit_safety_gate
from ..evidence import (
    EvidenceBundle,
    required_checks_passed,
)
from ..gitops import (
    commit_parents,
    commit_repair_tree,
    commit_revision_tree,
    current_head,
    resolve_tree,
)
from ..models import GateStage
from ..result import atomic_write_text
from ..run_options import EffectiveRepairScopePolicy
from .candidate import accepted_chain_records
from .check_failure import CheckRepairAttempt
from .check_scope import gate_mutable_authority
from .pipeline_v2 import (
    PipelineFailure,
    gate_acceptance_path,
    gate_dir,
    semantic_revision_dir,
)
from .shared import (
    is_object_id,
    json_text,
    read_json_artifact,
)


class GateAcceptanceService:
    """Persist and validate the green tree accepted by one gate episode."""

    def __init__(
        self,
        *,
        secrets: tuple[str, ...],
        repair_scope_policy: EffectiveRepairScopePolicy,
        authorize_candidate_tree: Callable[..., None],
        check_repair_attempts: Callable[..., tuple[CheckRepairAttempt, ...]],
        load_revision: Callable[[Path], Any],
        trace_emit: Callable[..., None],
        bounded_detail: Callable[[Exception], str],
    ) -> None:
        self._secrets = secrets
        self._repair_scope_policy = repair_scope_policy
        self._authorize_candidate_tree = authorize_candidate_tree
        self._check_repair_attempts = check_repair_attempts
        self._load_revision = load_revision
        self._trace_emit = trace_emit
        self._bounded_detail = bounded_detail

    def accept(
        self, store: Any, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: EvidenceBundle, *, base_paths: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if (
            not evidence.deterministic_passed
            or not required_checks_passed(evidence)
            or evidence.staged_tree_sha is None
        ):
            raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
        no_change = not evidence.changed_files
        if no_change and (evidence.diff != "" or evidence.base_sha != ctx.base_sha):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change evidence is not bound to the run base",
            )
        worktree = ctx.info.worktree
        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = gate_acceptance_path(ctx.run_dir, cycle_plan.cycle, stage)
        authority = gate_mutable_authority(
            ctx.run_dir, cycle_plan.cycle.number, stage,
            base_paths=(
                cycle_plan.mutable_scope if base_paths is None else base_paths
            ),
            policy_config=self._repair_scope_policy,
            require_attempt_records=True,
        )
        stored = read_json_artifact(path) if path.is_file() else None
        if path.is_file() and stored is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance is corrupted")
        if stored is not None:
            stored_no_change = stored.get("no_change", False) if isinstance(stored, dict) else False
            stored_parent = stored.get("parent_sha") if isinstance(stored, dict) else None
            parent_valid = is_object_id(stored_parent) or (
                stored_no_change is True
                and stored_parent is None
                and not evidence.changed_files
            )
            evidence_sha256 = self._durable_evidence_sha256(directory)
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 2
                or stored.get("review_cycle") != cycle_plan.cycle.number
                or not all(is_object_id(stored.get(key)) for key in ("tree_sha", "commit_sha"))
                or not isinstance(stored_no_change, bool)
                or not parent_valid
                or stored_no_change is not (not evidence.changed_files)
                or (
                    stored_no_change
                    and (
                        stored_parent is not None
                        or stored.get("commit_created") is not False
                        or stored.get("acceptance_kind") != "existing-head"
                    )
                )
                or stored.get("stage") != stage.value
                or stored.get("acceptance_kind") not in {"existing-head", "repair", "semantic-revision"}
                or not isinstance(stored.get("commit_created"), bool)
                or stored.get("tree_sha") != evidence.staged_tree_sha
                or stored.get("mutable_scope") != list(authority.effective_paths)
                or stored.get("mutable_scope_sha256") != authority.sha256
                or stored.get("evidence_sha256") != evidence_sha256
            ):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance does not match evidence")
            if current_head(worktree) != stored["commit_sha"]:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted gate HEAD moved")
            if (
                resolve_tree(worktree, stored["commit_sha"]) != stored["tree_sha"]
                or (
                    not stored_no_change
                    and commit_parents(worktree, stored["commit_sha"]) != (stored["parent_sha"],)
                )
            ):
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", "gate acceptance does not match its commit",
                )
            if stored.get("commit_created"):
                chain = list(accepted_chain_records(ctx.run_dir))
                if not any(
                    isinstance(item, dict) and item.get("commit_sha") == stored["commit_sha"]
                    for item in chain
                ):
                    chain.append({
                        "commit_sha": stored["commit_sha"],
                        "tree_sha": stored["tree_sha"],
                        "parent_sha": stored["parent_sha"],
                    })
                    atomic_write_text(ctx.run_dir / "accepted-chain.json", json_text({"commits": chain}))
            self._emit_acceptance(ctx, cycle_plan, stage, stored)
            return stored

        self._authorize_candidate_tree(
            evidence, worktree, current_head(worktree), ctx.branch_ref,
        )
        head = current_head(worktree)
        current_tree = resolve_tree(worktree, head)
        if no_change and current_tree != evidence.staged_tree_sha:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change HEAD tree differs from gate evidence",
            )
        if current_tree == evidence.staged_tree_sha:
            parents = () if no_change else commit_parents(worktree, head)
            if not no_change and len(parents) != 1:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted HEAD has no single parent")
            commit_sha, parent_sha = head, None if no_change else parents[0]
            parent_tree = resolve_tree(worktree, parent_sha) if parent_sha else None
            attempts = self._check_repair_attempts(
                ctx.run_dir, cycle_plan.cycle.number, stage,
            )
            revision = self._load_revision(
                semantic_revision_dir(ctx.run_dir, cycle_plan.cycle.number)
            )
            last_attempt = attempts[-1] if attempts else None
            recovered_repair = bool(
                last_attempt is not None
                and parent_tree is not None
                and last_attempt.tree_before == parent_tree
                and last_attempt.tree_after == evidence.staged_tree_sha
                and last_attempt.tree_before != last_attempt.tree_after
            )
            recovered_revision = bool(
                stage in {GateStage.POST_SEMANTIC_REVISION, GateStage.POST_REVIEW_IMPLEMENTATION}
                and revision is not None
                and revision.tree_before != revision.tree_after
                and revision.tree_after == evidence.staged_tree_sha
                and parent_tree is not None
                and parent_tree == revision.tree_before
            )
            acceptance_kind = (
                "repair" if recovered_repair else
                "semantic-revision" if recovered_revision else "existing-head"
            )
            commit_created = recovered_repair or recovered_revision
        else:
            parent_sha = head
            try:
                commit_safety_gate(
                    worktree,
                    tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha,
                    mutable_scope=authority.effective_paths,
                    verification_status="passed",
                    secrets=self._secrets,
                    max_diff_bytes=None,
                )
            except CommitSafetyError as exc:
                raise PipelineFailure("COMMIT_GATE_FAILED", self._bounded_detail(exc)) from exc
            attempts = self._check_repair_attempts(
                ctx.run_dir, cycle_plan.cycle.number, stage,
            )
            if attempts:
                commit_sha = commit_repair_tree(
                    worktree, tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha, cycle=cycle_plan.cycle.number,
                    body=f"MetaHarness-Run: {ctx.run_id}",
                )
                acceptance_kind = "repair"
            elif stage in {GateStage.POST_SEMANTIC_REVISION, GateStage.POST_REVIEW_IMPLEMENTATION}:
                commit_sha = commit_revision_tree(
                    worktree, tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha, body=f"MetaHarness-Run: {ctx.run_id}",
                )
                acceptance_kind = "semantic-revision"
            else:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate changed the tree without a repair")
            commit_created = True

        acceptance = {
            "schema_version": 2,
            "review_cycle": cycle_plan.cycle.number,
            "stage": stage.value,
            "tree_sha": evidence.staged_tree_sha,
            "commit_sha": commit_sha,
            "parent_sha": parent_sha,
            "no_change": not evidence.changed_files,
            "commit_created": commit_created,
            "acceptance_kind": acceptance_kind,
            "mutable_scope": list(authority.effective_paths),
            "mutable_scope_sha256": authority.sha256,
            "evidence_sha256": self._durable_evidence_sha256(directory),
        }
        atomic_write_text(path, json_text(acceptance))
        if commit_created:
            chain = list(accepted_chain_records(ctx.run_dir))
            if not any(item.get("commit_sha") == commit_sha for item in chain if isinstance(item, dict)):
                chain.append({"commit_sha": commit_sha, "tree_sha": evidence.staged_tree_sha, "parent_sha": parent_sha})
                atomic_write_text(ctx.run_dir / "accepted-chain.json", json_text({"commits": chain}))
        store.update_metadata(
            approved_tree_sha=evidence.staged_tree_sha,
            expected_head_sha=commit_sha,
            expected_parent_sha=parent_sha,
            expected_tree_sha=evidence.staged_tree_sha,
        )
        self._emit_acceptance(ctx, cycle_plan, stage, acceptance)
        return acceptance

    @staticmethod
    def _durable_evidence_sha256(directory: Path) -> str:
        try:
            return hashlib.sha256((directory / "evidence.json").read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "accepted gate evidence is unreadable",
            ) from exc

    def _emit_acceptance(self, ctx: Any, cycle_plan: Any, stage: GateStage, payload: Mapping[str, Any]) -> None:
        self._trace_emit(
            "gate.accepted",
            phase="validation",
            cycle=cycle_plan.cycle.number,
            data={
                "stage": stage.value,
                "parent_sha": payload["parent_sha"],
                "commit_sha": payload["commit_sha"],
                "tree_sha": payload["tree_sha"],
                "commit_created": payload.get("commit_created"),
                "acceptance_kind": payload.get("acceptance_kind"),
            },
        )
