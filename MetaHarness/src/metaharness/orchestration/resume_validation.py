"""The resume sub-domain: durable artifact readers and their validation.

This module is a safety boundary: every reader here is fail-closed and
returns ``None`` (never a partially trusted value) for an artifact that
does not prove exactly what the caller needs.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path
from typing import (
    Any,
    Mapping,
)
from .check_repair import _hard_failure_items
from .shared import (
    _MAX_AGENT_REPORT_BYTES,
    _MAX_STEP_REPORT_BYTES,
    _PLANNER_CONVERSATION,
    _bounded_v2_report,
    _is_object_id,
    _json_text,
    _read_bounded_text,
    _read_json_artifact,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    RepositoryReference,
    WorktreeInfo,
)
from ..planning_v2 import TaskPlanV2
from ..result import atomic_write_text
from ..resume import (
    ResumeCheckpoint,
    ResumePhase,
)
from ..review import (
    ReviewParseError,
    ReviewResult,
    parse_review,
)
from ..usage import (
    normalize_usage,
    read_usage_artifact,
)
from ..llm.chat import LLMConversationHandle


@dataclasses.dataclass(frozen=True)
class _PersistedRevision:
    """A completed Claude revision read back from its durable artifacts."""

    final_message: str
    usage: dict[str, int]
    tree_before: str
    tree_after: str
    exit_code: int = 0
    timed_out: bool = False
    stderr_tail: str = ""


@dataclasses.dataclass
class _ResumedRun:
    """Everything a resume needs, rebuilt from persisted artifacts only."""

    checkpoint: ResumeCheckpoint
    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: Any
    info: WorktreeInfo
    repository_reference: RepositoryReference
    spec: str
    context: str
    restore_paths: tuple[str, ...] = ()
    mismatch_recovery: dict[str, Any] | None = None
    mismatch_recovery_path: Path | None = None
    scope_violation_recovery: dict[str, Any] | None = None
    c01_steps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # step id -> the clean mismatch its first attempt returned, for the single
    # bounded retry this resume owes that step.
    mismatch_retries: dict[str, str] = dataclasses.field(default_factory=dict)
    c01_revision: _PersistedRevision | None = None
    c01_check_repair_revision: _PersistedRevision | None = None
    c01_expanded_check_repair_revision: _PersistedRevision | None = None
    c01_evidence: EvidenceBundle | None = None
    c01_review: ReviewResult | None = None
    repair_plan: TaskPlanV2 | None = None
    repair_bundle: dict[str, Any] | None = None
    repair_bundle_sha: str | None = None
    c02_steps: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    c02_revision: _PersistedRevision | None = None
    c02_check_repair_revision: _PersistedRevision | None = None
    c02_expanded_check_repair_revision: _PersistedRevision | None = None
    c02_evidence: EvidenceBundle | None = None
    c02_review: ReviewResult | None = None
    existing_commit_sha: str | None = None


def _reusable_pre_checks(artifact_dir: Path, tree: str) -> dict[str, Any] | None:
    """Durable pre-revision evidence frozen for exactly *tree*, if any."""

    payload = _read_json_artifact(artifact_dir / "pre_checks.json")
    if not isinstance(payload, dict) or payload.get("staged_tree_sha") != tree:
        return None
    failures = payload.get("failures")
    if not isinstance(failures, list) or any(not isinstance(item, str) for item in failures):
        return None
    if _hard_failure_items(failures):
        return None
    evidence = _read_json_artifact(artifact_dir / "evidence.json")
    if not isinstance(evidence, dict) or evidence.get("staged_tree_sha") != tree:
        return None
    try:
        # Keep the durable evidence existence check, but never return its
        # contents to a Claude prompt.
        (artifact_dir / "diff.patch").read_bytes()
    except OSError:
        return None
    return payload


def _load_evidence(directory: Path) -> EvidenceBundle | None:
    """Rebuild a frozen evidence bundle from ``evidence.json``."""

    payload = _read_json_artifact(directory / "evidence.json")
    if not isinstance(payload, dict):
        return None
    changed = payload.get("changed_files")
    checks = payload.get("checks")
    failures = payload.get("failures")
    if (
        not _is_object_id(payload.get("base_sha"))
        or not _is_object_id(payload.get("staged_tree_sha"))
        or not isinstance(payload.get("diff"), str)
        or not isinstance(payload.get("deterministic_passed"), bool)
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
        or not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks)
        or not isinstance(failures, list) or any(not isinstance(item, str) for item in failures)
    ):
        return None
    return EvidenceBundle(
        base_sha=payload["base_sha"],
        staged_tree_sha=payload["staged_tree_sha"],
        changed_files=tuple(changed),
        diff=payload["diff"],
        checks=tuple(checks),
        deterministic_passed=payload["deterministic_passed"],
        failures=tuple(failures),
        required_check_ids=tuple(
            item for item in payload.get("required_check_ids", [])
            if isinstance(item, str)
        ),
    )


def _retry_checks_evidence(
    checks_dir: Path, *, cycle: str, initial_tree: str, repaired_tree: str,
    legacy_dir: Path | None = None,
) -> tuple[EvidenceBundle | None, str | None]:
    """Resolve the evidence authority of a ``FINAL_CHECKS_RETRY`` resume.

    This phase is reached only after the automatic check-repair Claude
    *succeeded*, so the checkpoint tree is the repaired tree, not the red tree
    the repair answered to.  Two durable states are legitimate:

    * the first retry crashed before writing its evidence -- the red
      first-pass bundle has already been moved to ``attempts/01/`` and there
      is no current bundle;
    * a complete retry already ran and stayed red -- the current bundle is for
      the repaired tree.

    The red first-pass evidence stays the authority the repair answers to, so
    it is preferred when present; the retry checks are re-executed in both
    cases, which is why a red current bundle is never read as proof that the
    new checks are green.  Returns ``(evidence, refusal)``.
    """

    prior = _load_evidence(checks_dir / "attempts" / "01")
    if prior is not None and prior.staged_tree_sha != initial_tree:
        return None, (
            f"the archived {cycle} first-pass evidence is not for the pre-repair checks tree"
        )
    # Only a run without the per-cycle directory may fall back to the root
    # aliases: those aliases still name the *first* pass once the retry
    # archived it, so they are never the authority for the repaired tree.
    current = (
        _load_evidence(checks_dir) if checks_dir.is_dir() or legacy_dir is None
        else _load_evidence(legacy_dir)
    )
    if current is not None and current.staged_tree_sha != repaired_tree:
        return None, f"the {cycle} retry evidence is not for the repaired checks tree"
    evidence = prior or current
    if evidence is None:
        return None, f"the {cycle} final evidence is missing or not for the expected checks tree"
    return evidence, None


def _accepted_review(
    directory: Path, evidence: EvidenceBundle, candidate_sha: str | None = None,
) -> ReviewResult | None:
    """A reviewer answer already accepted for exactly this candidate tree."""

    if not (directory / "review.json").is_file():
        return None
    try:
        request = (directory / "reviewer.request.txt").read_text(encoding="utf-8")
        raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if f'"CANDIDATE_TREE_SHA": "{evidence.staged_tree_sha}"' not in request:
        return None
    if candidate_sha is not None and candidate_sha not in request:
        return None
    try:
        return parse_review(raw, deterministic_passed=evidence.deterministic_passed)
    except ReviewParseError:
        return None


def _load_c01_review(run_dir: Path, evidence: EvidenceBundle) -> ReviewResult | None:
    for directory in (run_dir / "review" / "C01", run_dir):
        try:
            raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        try:
            return parse_review(raw, deterministic_passed=evidence.deterministic_passed)
        except ReviewParseError:
            return None
    return None


def _load_accepted_c01_review(
    run_dir: Path, evidence: EvidenceBundle, candidate_sha: str,
) -> ReviewResult | None:
    """Reviewer #1's accepted answer for exactly the pushed C01 candidate."""

    for directory in (run_dir / "review" / "C01", run_dir):
        review = _accepted_review(directory, evidence, candidate_sha)
        if review is not None:
            return review
    return None


def _load_completed_step(step_dir: Path, step_id: str) -> dict[str, Any] | None:
    """One completed or cleanly deferred step record."""

    record = _read_json_artifact(step_dir / "step.json", 128 * 1024)
    if not isinstance(record, dict) or record.get("id") != step_id:
        return None
    status = record.get("status")
    if status not in {"COMPLETED", "DEFERRED_CONTRACT_MISMATCH"}:
        return None
    changed = record.get("changed_paths")
    if (
        not _is_object_id(record.get("tree_before"))
        or not _is_object_id(record.get("tree_after"))
        or not isinstance(changed, list) or any(not isinstance(item, str) for item in changed)
    ):
        return None
    if status == "DEFERRED_CONTRACT_MISMATCH" and (
        record["tree_before"] != record["tree_after"]
        or changed
        or not isinstance(record.get("mismatch"), str)
        or not record["mismatch"].strip()
        or len(record["mismatch"].encode("utf-8", errors="replace")) > _MAX_STEP_REPORT_BYTES
    ):
        return None
    return {
        "id": step_id, "status": status, "profile_id": record.get("profile_id"),
        "tree_before": record["tree_before"], "tree_after": record["tree_after"],
        "changed_paths": list(changed),
        "usage": normalize_usage(record.get("usage")),
        "final": _bounded_v2_report(_read_bounded_text(step_dir / "agent.final.md")),
        **({"mismatch": _bounded_v2_report(record["mismatch"])}
           if status == "DEFERRED_CONTRACT_MISMATCH" else {}),
        **({"initial_mismatch": _bounded_v2_report(str(record["initial_mismatch"]))}
           if isinstance(record.get("initial_mismatch"), str) and record["initial_mismatch"].strip()
           else {}),
        **({"mismatch_retry_count": record["mismatch_retry_count"]}
           if isinstance(record.get("mismatch_retry_count"), int) else {}),
        **({"deferred_verify": _bounded_v2_report(str(record["deferred_verify"]))}
           if isinstance(record.get("deferred_verify"), str) and record["deferred_verify"].strip()
           else {}),
    }


def _verify_step_chain(records: list[dict[str, Any] | None], start_tree: str) -> str | None:
    """The last tree of an unbroken step chain starting at *start_tree*."""

    tree = start_tree
    for record in records:
        if record is None or record["tree_before"] != tree:
            return None
        tree = record["tree_after"]
    return tree


def _load_revision(directory: Path) -> _PersistedRevision | None:
    report = _read_json_artifact(directory / "report.json", 1024 * 1024)
    if not isinstance(report, dict) or report.get("status") not in {"COMPLETED", "NO_CHANGE"}:
        return None
    if not _is_object_id(report.get("tree_before")) or not _is_object_id(report.get("tree_after")):
        return None
    final = _read_bounded_text(directory / "agent.final.md", _MAX_AGENT_REPORT_BYTES * 2)
    if not final and isinstance(report.get("final"), str):
        final = report["final"]
    usage = read_usage_artifact(directory / "usage.json") or normalize_usage(report.get("usage"))
    return _PersistedRevision(final, usage, report["tree_before"], report["tree_after"])


# The scope-repair checkpoint authority, by phase.  A scope-repair cycle has
# its own durable chain, so a checkpoint taken inside it is never proved by
# the historical C01/C02 Claude tree -- see
# :func:`_scope_repair_checkpoint_tree`.
_SCOPE_REPAIR_RECOVERY_TREE_PHASES = frozenset({
    ResumePhase.CHECK_SCOPE_PLANNER_C01, ResumePhase.CHECK_SCOPE_APPROVAL_C01,
    ResumePhase.CHECK_SCOPE_PLANNER_C02, ResumePhase.CHECK_SCOPE_APPROVAL_C02,
})
_SCOPE_REPAIR_STEP_PHASES = frozenset({
    ResumePhase.CHECK_SCOPE_REPAIR_STEP_C01, ResumePhase.CHECK_SCOPE_REPAIR_STEP_C02,
})
_SCOPE_REPAIR_RESIDUAL_PHASES = frozenset({
    ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C01,
    ResumePhase.CHECK_SCOPE_REPAIR_FINAL_CHECKS_C02,
})

def _scope_repair_checkpoint_tree(
    *,
    directory: Path,
    checkpoint: ResumeCheckpoint,
    recovery_tree: str,
    step_ids: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """The tree a scope-repair checkpoint must carry, from durable artifacts.

    A scope-repair cycle starts from *recovery_tree* -- the failed-check tree
    the rollback restored before the strong planner ran -- and advances
    through its own durable chain: the bounded Luna steps in *directory*,
    then the residual Claude pass.  Each phase has exactly one authority:

    * planner/approval: nothing has run yet, so *recovery_tree* itself;
    * step: the completed steps strictly before ``checkpoint.step_id``;
    * checks and residual Claude: the whole completed Luna chain -- the
      residual pass has not acquired the checkpoint authority yet;
    * final checks: the residual Claude ``tree_after``, and only when its
      ``tree_before`` is exactly the completed Luna chain end.

    The function is cycle-agnostic: C01 and C02 differ only by *directory*.
    Returns ``(tree, refusal)`` and never a partially trusted tree.
    """

    phase = checkpoint.phase
    if phase in _SCOPE_REPAIR_RECOVERY_TREE_PHASES:
        return recovery_tree, None
    if phase in _SCOPE_REPAIR_STEP_PHASES:
        step_id = checkpoint.step_id
        if step_id is None or step_id not in step_ids:
            return None, "the scope-repair checkpoint step is not in the scope-repair plan"
        prior = step_ids[: step_ids.index(step_id)]
    else:
        prior = step_ids
    records = [_load_completed_step(directory / "steps" / item, item) for item in prior]
    if any(record is None for record in records):
        return None, "a completed scope-repair Luna step record is missing or invalid"
    chain_end = _verify_step_chain(records, recovery_tree)
    if chain_end is None:
        return None, "scope-repair Luna step trees do not form an unbroken chain"
    if phase not in _SCOPE_REPAIR_RESIDUAL_PHASES:
        return chain_end, None
    residual = _load_revision(directory / "residual-claude")
    if residual is None:
        return None, "the scope-repair residual Claude record is missing"
    if residual.tree_before != chain_end:
        return None, "scope-repair residual Claude is not based on the completed Luna tree"
    return residual.tree_after, None


def _read_repository_reference(run_dir: Path) -> RepositoryReference | None:
    payload = _read_json_artifact(run_dir / "repository_reference.json", 16 * 1024)
    if not isinstance(payload, dict) or set(payload) != {"remote_name", "web_url", "base_sha", "immutable_url"}:
        return None
    if not isinstance(payload["remote_name"], str) or not _is_object_id(payload["base_sha"]):
        return None
    if any(payload[key] is not None and not isinstance(payload[key], str) for key in ("web_url", "immutable_url")):
        return None
    return RepositoryReference(**payload)


def _persist_planner_conversation(run_dir: Path, handle: Any) -> None:
    """Persist a driver-provided planner conversation handle, never a guess."""

    if isinstance(handle, LLMConversationHandle):
        atomic_write_text(run_dir / _PLANNER_CONVERSATION, _json_text({
            "provider_id": handle.provider_id, "conversation_id": handle.conversation_id,
        }))


def _read_planner_conversation(run_dir: Path) -> LLMConversationHandle | None:
    payload = _read_json_artifact(run_dir / _PLANNER_CONVERSATION, 4096)
    if not isinstance(payload, dict):
        return None
    try:
        return LLMConversationHandle(payload.get("provider_id"), payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


def _state_cycle_value(state: Mapping[str, Any]) -> int:
    value = state.get("cycle")
    return value if value in (1, 2) and not isinstance(value, bool) else 1
