"""The resume sub-domain: durable artifact readers and their validation.

This module is a safety boundary: every reader here is fail-closed and
returns ``None`` (never a partially trusted value) for an artifact that
does not prove exactly what the caller needs.  :func:`validate_resume` is the
single integrity gate in front of every resumed execution checkpoint; it is
cycle-agnostic and never calls a model, a check or a Git write.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

from pathlib import Path
from typing import (
    Any,
    Mapping,
    NoReturn,
)
from .check_repair import (
    _hard_failure_items,
    gate_mutable_authority,
)
from .pipeline_v2 import (
    candidate_dir,
    check_repair_attempt_dir,
    correction_dir,
    cycle_record_path,
    gate_acceptance_path,
    gate_dir,
    implementation_steps_dir,
    review_dir,
    semantic_revision_dir,
    step_dir,
    final_gate_stage,
    pre_semantic_gate_stage,
)
from .shared import (
    _MAX_AGENT_REPORT_BYTES,
    _MAX_STEP_REPORT_BYTES,
    _PLANNER_CONVERSATION,
    bounded_v2_report,
    _is_object_id,
    _json_text,
    _read_bounded_text,
    _read_json_artifact,
    _read_tree_file,
    _status_has_unstaged_or_untracked,
)
from ..approval import (
    ApprovalDecision,
    ApprovalError,
    compute_plan_identity_from_run,
    read_check_authority,
    read_plan_approval,
    read_scope_approval,
)
from ..evidence import EvidenceBundle, required_checks_passed
from ..execution_selection import (
    ExecutionSelectionError,
    read_cycle_execution_selection,
    read_execution_selection_with_sha256,
    validate_cycle_execution_selection,
    validate_execution_selection,
)
from ..gitops import (
    GitError,
    RepositoryReference,
    WorktreeInfo,
    branch_exists,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_parents,
    current_head,
    git_root,
    index_tree_sha,
    is_ancestor,
    path_exists_in_tree,
    registered_worktrees,
    remote_run_branch_tip,
    resolve_commit,
    resolve_tree,
    status_porcelain,
    symbolic_head,
)
from ..models import (
    CycleKind,
    ExecutionSelection,
    GateStage,
    HarnessConfig,
    PlanDecision,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
    TaskPlanV2,
)
from ..planning.artifacts import validate_implementation_bundle
from ..profiles import ProfileError
from ..planning.protocol import V2PlanParseError, parse_task_plan_v2
from ..result import atomic_write_text
from ..resume import (
    ResumeCheckpoint,
    ResumeCheckpointError,
    ResumeIntegrityError,
    ResumePhase,
    ResumeRequiresOperatorError,
    plan_identity_from_mapping,
)
from ..review import (
    ReviewParseError,
    ReviewResult,
    parse_review,
)
from ..run_options import EffectiveRepairScopePolicy
from ..usage import (
    normalize_usage,
    read_usage_artifact,
)
from ..validation import ValidationError, config_with_check_authority, resolve_check_cwd
from ..llm.chat import LLMConversationHandle


@dataclasses.dataclass(frozen=True)
class _PersistedRevision:
    """A completed revision or check-repair pass read back from its artifacts."""

    final_message: str
    usage: dict[str, int]
    tree_before: str
    tree_after: str
    exit_code: int = 0
    timed_out: bool = False
    stderr_tail: str = ""


@dataclasses.dataclass(frozen=True)
class ResumedRun:
    """Everything a resume needs, rebuilt from persisted artifacts only."""

    checkpoint: ResumeCheckpoint
    plan: TaskPlanV2
    bundle: dict[str, Any]
    selection: ExecutionSelection
    info: WorktreeInfo
    repository_reference: RepositoryReference
    spec: str
    context: str
    base_tree_sha: str
    restore_paths: tuple[str, ...] = ()


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
        # contents to a worker prompt.
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


def _accepted_review(
    directory: Path, evidence: EvidenceBundle, candidate_sha: str | None = None,
) -> ReviewResult | None:
    """A reviewer answer already accepted for exactly this candidate tree."""

    if not (directory / "review.json").is_file():
        return None
    try:
        request = (directory / "reviewer.request.txt").read_text(encoding="utf-8")
        raw = (directory / "reviewer.raw.md").read_text(encoding="utf-8")
        persisted = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return None
    except json.JSONDecodeError:
        return None
    # The request is the exact evidence the answer was given: it must name
    # both the reviewed tree and the immutable candidate commit.
    if evidence.staged_tree_sha not in request:
        return None
    if candidate_sha is not None and candidate_sha not in request:
        return None
    try:
        review = parse_review(raw, deterministic_passed=evidence.deterministic_passed)
    except ReviewParseError:
        return None
    normalized = dataclasses.asdict(review)
    normalized["verdict"] = review.verdict.value
    normalized["route"] = review.route.value
    if persisted != normalized:
        return None
    return review


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
    no_change = record.get("no_change", False)
    if not isinstance(no_change, bool) or (no_change and (
        record["tree_before"] != record["tree_after"] or changed
    )):
        return None
    if record["tree_before"] != record["tree_after"] and not _is_object_id(record.get("commit_sha")):
        # A successful worker whose candidate was not accepted yet: the step
        # is complete only once its commit crossed the acceptance boundary.
        return None
    return {
        "id": step_id, "status": status, "profile_id": record.get("profile_id"),
        "tree_before": record["tree_before"], "tree_after": record["tree_after"],
        "changed_paths": list(changed),
        **({"no_change": True} if no_change else {}),
        "usage": normalize_usage(record.get("usage")),
        "final": bounded_v2_report(_read_bounded_text(step_dir / "agent.final.md")),
        **({"mismatch": bounded_v2_report(record["mismatch"])}
           if status == "DEFERRED_CONTRACT_MISMATCH" else {}),
        **({"initial_mismatch": bounded_v2_report(str(record["initial_mismatch"]))}
           if isinstance(record.get("initial_mismatch"), str) and record["initial_mismatch"].strip()
           else {}),
        **({"mismatch_retry_count": record["mismatch_retry_count"]}
           if isinstance(record.get("mismatch_retry_count"), int) else {}),
        **({"deferred_verify": bounded_v2_report(str(record["deferred_verify"]))}
           if isinstance(record.get("deferred_verify"), str) and record["deferred_verify"].strip()
           else {}),
    }


def completed_step_records(
    run_dir: Path, cycle: int, step_ids: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """The durable completed prefix of one cycle's approved steps."""

    records: list[dict[str, Any]] = []
    for step_id in step_ids:
        record = _load_completed_step(step_dir(run_dir, cycle, step_id), step_id)
        if record is None:
            break
        records.append(record)
    return records


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


def read_cycle_record(run_dir: Path, number: int) -> RunCycle:
    """The durable identity of one cycle, written when the cycle started."""

    payload = _read_json_artifact(cycle_record_path(run_dir, number), 4096)
    expected_schema = 1 if number == 1 else 2
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != expected_schema
        or payload.get("number") != number
    ):
        raise ResumeIntegrityError(f"cycle {number:03d} record is missing or invalid")
    try:
        return RunCycle(number, CycleKind(payload.get("kind")))
    except ValueError as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} kind is invalid") from exc


def _accepted_review_binding(run_dir: Path, number: int) -> tuple[dict[str, Any], ReviewResult, str]:
    """Read the accepted review and hash the exact bytes of review.json."""

    candidate = read_candidate_record(run_dir, number)
    evidence = candidate_evidence(run_dir, number)
    if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
        raise ResumeIntegrityError(f"cycle {number:03d} candidate evidence is missing")
    directory = review_dir(run_dir, number)
    review = _accepted_review(directory, evidence, candidate["commit_sha"])
    if review is None or review.verdict is not ReviewVerdict.REVISE:
        raise ResumeIntegrityError(f"cycle {number:03d} accepted correction review is invalid")
    if review.route not in {ReviewRoute.IMPLEMENTATION, ReviewRoute.REPLAN}:
        raise ResumeIntegrityError(f"cycle {number:03d} accepted correction route is invalid")
    try:
        digest = hashlib.sha256((directory / "review.json").read_bytes()).hexdigest()
    except OSError as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} review artifact is unreadable") from exc
    return candidate, review, digest


def correction_binding(run_dir: Path, number: int) -> dict[str, Any]:
    """Return the exact durable binding required by correction cycle number."""

    if number < 2:
        raise ValueError("correction cycle number must be greater than one")
    candidate, review, review_sha256 = _accepted_review_binding(run_dir, number - 1)
    kind = (
        CycleKind.REVIEW_IMPLEMENTATION
        if review.route is ReviewRoute.IMPLEMENTATION
        else CycleKind.REVIEW_REPLAN
    )
    return {
        "source_review_cycle": number - 1,
        "source_route": review.route.value,
        "source_candidate_sha": candidate["commit_sha"],
        "source_review_sha256": review_sha256,
        "kind": kind.value,
    }


def validate_correction_bindings(run_dir: Path, through_cycle: int) -> None:
    """Validate every persisted correction binding through through_cycle."""

    if through_cycle < 2:
        return
    for number in range(2, through_cycle + 1):
        payload = _read_json_artifact(cycle_record_path(run_dir, number), 4096)
        expected = correction_binding(run_dir, number)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 2
            or payload.get("number") != number
            or any(payload.get(key) != value for key, value in expected.items())
        ):
            raise ResumeIntegrityError(f"cycle {number:03d} correction binding diverges")


def read_candidate_record(run_dir: Path, number: int) -> dict[str, Any]:
    """One cycle's immutable candidate commit record."""

    payload = _read_json_artifact(candidate_dir(run_dir, number) / "commit.json")
    no_change = payload.get("no_change", False) if isinstance(payload, dict) else False
    if (
        not isinstance(payload, dict)
        or not _is_object_id(payload.get("commit_sha"))
        or not _is_object_id(payload.get("tree_sha"))
        or not isinstance(no_change, bool)
        or (
            not _is_object_id(payload.get("parent_sha"))
            and not (no_change and payload.get("parent_sha") is None)
        )
    ):
        raise ResumeIntegrityError(f"cycle {number:03d} candidate commit record is missing")
    return payload


def candidate_evidence(run_dir: Path, number: int) -> EvidenceBundle | None:
    """The gate evidence one cycle's candidate commit answers for."""

    stage = read_candidate_record(run_dir, number).get("gate_stage")
    try:
        return _load_evidence(gate_dir(run_dir, number, stage))
    except ValueError:
        return None


def _validate_gate_acceptance(
    run_dir: Path, number: int, stage: Any, *, tree: str | None, head: str | None,
    base_scope: tuple[str, ...], policy: EffectiveRepairScopePolicy,
) -> None:
    """Require the durable accepted state for a completed gate boundary."""

    payload = _read_json_artifact(gate_acceptance_path(run_dir, number, stage))
    evidence_path = gate_dir(run_dir, number, stage) / "evidence.json"
    try:
        evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    except OSError:
        evidence_sha256 = None
    authority = gate_mutable_authority(
        run_dir, number, stage,
        base_paths=base_scope,
        policy_config=policy,
        require_attempt_records=True,
    )
    no_change = payload.get("no_change", False) if isinstance(payload, dict) else False
    parent_sha = payload.get("parent_sha") if isinstance(payload, dict) else None
    evidence = _load_evidence(gate_dir(run_dir, number, stage))
    parent_valid = _is_object_id(parent_sha) or (
        no_change is True
        and parent_sha is None
        and evidence is not None
        and not evidence.changed_files
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in {1, 2}
        or payload.get("review_cycle") != number
        or payload.get("stage") != getattr(stage, "value", stage)
        or not _is_object_id(payload.get("tree_sha"))
        or not _is_object_id(payload.get("commit_sha"))
        or not isinstance(no_change, bool)
        or not parent_valid
        or (
            evidence is not None
            and no_change is not (not evidence.changed_files)
        )
        or payload.get("tree_sha") != tree
        or payload.get("commit_sha") != head
        or payload.get("acceptance_kind") not in {"existing-head", "repair", "semantic-revision"}
        or not isinstance(payload.get("commit_created"), bool)
        or (
            no_change
            and (
                parent_sha is not None
                or payload.get("commit_created") is not False
                or payload.get("acceptance_kind") != "existing-head"
            )
        )
        or payload.get("mutable_scope") != list(authority.effective_paths)
        or payload.get("mutable_scope_sha256") != authority.sha256
        or (
            no_change
            and (
                evidence is None or evidence.diff != ""
                or not evidence.deterministic_passed
                or not required_checks_passed(evidence)
            )
        )
        or (
            payload.get("schema_version") == 2
            and payload.get("evidence_sha256") != evidence_sha256
        )
    ):
        _refuse("the gate acceptance artifact is missing or invalid")


def load_correction_plan(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path, number: int,
    *, inherited_check_ids: tuple[str, ...],
) -> tuple[TaskPlanV2, dict[str, Any], str]:
    """Parse the durable correction plan of cycle *number* (> 1)."""

    directory = correction_dir(run_dir, number)
    try:
        plan = parse_task_plan_v2(
            (directory / "planner.raw.md").read_text(encoding="utf-8"),
            planning=config.planning,
            check_catalog=config.check_catalog,
            inherited_check_ids=inherited_check_ids,
        )
        bundle, bundle_sha = validate_implementation_bundle(
            directory, expected_step_ids=[step.id for step in plan.steps]
        )
    except (OSError, UnicodeError, V2PlanParseError, ValueError, AttributeError) as exc:
        raise ResumeIntegrityError(f"cycle {number:03d} correction plan is unreadable: {exc}") from exc
    if plan.decision is not PlanDecision.READY:
        raise ResumeIntegrityError(f"cycle {number:03d} correction plan is not READY")
    try:
        cycle_selection = read_cycle_execution_selection(run_dir, number)
        validate_cycle_execution_selection(config, cycle_selection)
    except (ExecutionSelectionError, OSError) as exc:
        raise ResumeIntegrityError(
            f"cycle {number:03d} execution selection is unreadable: {exc}"
        ) from exc
    if [item.step_id for item in cycle_selection.steps] != [step.id for step in plan.steps]:
        raise ResumeIntegrityError(
            f"cycle {number:03d} execution selection does not match the plan"
        )
    return plan, bundle, bundle_sha


def verify_correction_scope(
    run_dir: Path, number: int, bundle_sha: str, policy: EffectiveRepairScopePolicy,
) -> list[str]:
    """The added paths of one correction scope delta, with their authority."""

    directory = correction_dir(run_dir, number)
    delta = _read_json_artifact(directory / "scope_delta.json", 256 * 1024)
    if not isinstance(delta, dict) or delta.get("correction_bundle_sha256") != bundle_sha:
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta is missing or not bound to its plan")
    added = delta.get("added_paths")
    if not isinstance(added, list) or any(not isinstance(path, str) for path in added):
        raise ResumeIntegrityError(f"cycle {number:03d} scope delta is malformed")
    if added:
        if policy.policy == "deny-expansion":
            raise ResumeIntegrityError(f"cycle {number:03d} scope expansion is denied by the run policy")
        if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
            digest = hashlib.sha256((directory / "scope_delta.json").read_bytes()).hexdigest()
            try:
                approval = read_scope_approval(directory, expected_sha256=digest)
            except ApprovalError as exc:
                raise ResumeIntegrityError(f"cycle {number:03d} scope approval is invalid: {exc}") from exc
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                raise ResumeIntegrityError(f"cycle {number:03d} scope expansion was not approved")
    return list(added)


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
        path = run_dir / _PLANNER_CONVERSATION
        atomic_write_text(path, _json_text({
            "provider_id": handle.provider_id, "conversation_id": handle.conversation_id,
        }))
        path.chmod(0o600)


def _read_planner_conversation(run_dir: Path) -> LLMConversationHandle | None:
    payload = _read_json_artifact(run_dir / _PLANNER_CONVERSATION, 4096)
    if not isinstance(payload, dict):
        return None
    try:
        return LLMConversationHandle(payload.get("provider_id"), payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


# -- the resume integrity gate --------------------------------------------------

_STEP_PHASES = frozenset({ResumePhase.IMPLEMENT_STEP, ResumePhase.REVIEW_IMPLEMENTATION})
_CANDIDATE_PHASES = frozenset({
    ResumePhase.CANDIDATE_PUSH, ResumePhase.FINAL_REVIEW, ResumePhase.PUBLISH,
})


def _refuse(message: str) -> NoReturn:
    raise ResumeIntegrityError(message)


def _plan_scope(plan: TaskPlanV2) -> set[str]:
    return {
        path for step in plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }


def _contract_repair_scope(
    run_dir: Path, number: int, policy: EffectiveRepairScopePolicy,
) -> set[str]:
    """Read durable mutable paths approved by step-contract repairs."""

    scope: set[str] = set()
    root = implementation_steps_dir(run_dir, number)
    if not root.is_dir():
        return scope
    for step_root in root.iterdir():
        repairs = step_root / "contract_repairs"
        if not repairs.is_dir():
            continue
        for repair in repairs.iterdir():
            validation = _read_json_artifact(repair / "validation.json", 64 * 1024)
            contract = repair / "contract.md"
            if not isinstance(validation, dict) or validation.get("status") != "validated":
                continue
            added = validation.get("added_mutable_paths")
            digest = validation.get("repaired_contract_sha256")
            if (
                not isinstance(added, list) or any(
                    not isinstance(path, str) or not path or path.startswith("/")
                    or ".." in Path(path).parts for path in added
                )
                or not isinstance(digest, str) or not contract.is_file()
            ):
                raise ResumeIntegrityError("step contract repair scope artifact is malformed")
            try:
                actual = hashlib.sha256(contract.read_bytes()).hexdigest()
            except OSError as exc:
                raise ResumeIntegrityError("step contract repair contract is unreadable") from exc
            if actual != digest:
                raise ResumeIntegrityError("step contract repair contract hash changed")
            if added:
                if policy.policy == "deny-expansion":
                    raise ResumeIntegrityError("step contract repair scope expansion is denied")
                if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
                    delta_path = repair / "scope_delta.json"
                    delta = _read_json_artifact(delta_path, 64 * 1024)
                    if not isinstance(delta, dict) or delta.get("added_paths") != added:
                        raise ResumeIntegrityError("step contract repair scope delta is malformed")
                    try:
                        delta_sha = hashlib.sha256(delta_path.read_bytes()).hexdigest()
                        approval = read_scope_approval(repair, expected_sha256=delta_sha)
                    except (OSError, ApprovalError) as exc:
                        raise ResumeIntegrityError("step contract repair scope approval is invalid") from exc
                    if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                        raise ResumeIntegrityError("step contract repair scope was not approved")
            scope.update(added)
    return scope


def _semantic_revision_scope(
    repo: Path,
    run_dir: Path,
    number: int,
    policy: EffectiveRepairScopePolicy,
) -> set[str]:
    """Read reviser-requested paths only when durable policy authorized them."""

    root = semantic_revision_dir(run_dir, number) / "scope_requests"
    if not root.is_dir():
        return set()
    revision_dir = semantic_revision_dir(run_dir, number)
    scope: set[str] = set()
    added_total: set[str] = set()
    for request_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not request_dir.is_dir() or not request_dir.name.isdigit():
            continue
        authority = _read_json_artifact(request_dir / "authority.json", 64 * 1024)
        if not isinstance(authority, dict):
            _refuse("semantic scope authority is missing")
        added = authority.get("added_paths")
        requested = authority.get("requested_paths")
        tree_sha = authority.get("tree_sha")
        source_report_sha = authority.get("source_report_sha256")
        existing = authority.get("existing_paths")
        creates = authority.get("create_paths")
        if (
            authority.get("schema_version") != 1
            or authority.get("cycle") != number
            or authority.get("policy") != policy.policy
            or authority.get("bound") != policy.max_added_paths
            or not isinstance(added, list) or not isinstance(requested, list)
            or not isinstance(existing, list) or not isinstance(creates, list)
            or any(not _safe_semantic_scope_path(path) for path in (*added, *requested, *existing, *creates))
            or not isinstance(tree_sha, str) or not _is_object_id(tree_sha)
            or not isinstance(source_report_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", source_report_sha)
        ):
            _refuse("semantic scope authority is malformed")
        if (
            added != sorted(set(added))
            or requested != sorted(set(requested))
            or set(existing) | set(creates) != set(requested)
            or set(existing) & set(creates)
            or set(added) - set(requested)
        ):
            _refuse("semantic scope authority is not canonical")
        source_reports = [revision_dir / "report.json"]
        attempts_root = revision_dir / "attempts"
        if attempts_root.is_dir():
            source_reports.extend(sorted(attempts_root.glob("[0-9][0-9]/report.json")))
        source_matches = False
        for report_path in source_reports:
            try:
                raw_report = report_path.read_bytes()
            except OSError:
                continue
            if hashlib.sha256(raw_report).hexdigest() != source_report_sha:
                continue
            report = _read_json_artifact(report_path, 256 * 1024)
            scope_request = report.get("scope_request") if isinstance(report, dict) else None
            if not isinstance(scope_request, dict):
                continue
            source_paths = scope_request.get("paths")
            evidence = scope_request.get("evidence")
            source_reason = scope_request.get("reason")
            if (
                not isinstance(source_paths, list)
                or any(not isinstance(path, str) for path in source_paths)
                or not isinstance(source_reason, str)
            ):
                continue
            source_matches = bool(
                sorted(source_paths) == requested
                and source_reason[:2000] == authority.get("reason")
                and isinstance(evidence, list)
                and [str(item)[:1000] for item in evidence[:16]] == authority.get("evidence")
                and report.get("tree_before") == tree_sha
            )
            if source_matches:
                break
        if not source_matches:
            _refuse("semantic scope authority is not bound to its reviser report")
        for path in requested:
            if path_exists_in_tree(repo, tree_sha, path) != (path in existing):
                _refuse("semantic scope tree existence semantics changed")
        decision = _read_json_artifact(request_dir / "decision.json", 64 * 1024)
        if isinstance(decision, dict) and decision.get("decision") == "denied-expansion":
            if policy.policy != "deny-expansion" or decision.get("added_paths") != added:
                _refuse("semantic denied-scope record is invalid")
            continue
        if added and policy.policy == "deny-expansion":
            _refuse("semantic scope expansion is denied")
        added_total.update(added)
        needs_approval = (
            bool(added) and (
                policy.policy == "require-approval"
                or (policy.policy == "auto-bounded" and len(added_total) > policy.max_added_paths)
            )
        )
        if needs_approval:
            delta_path = request_dir / "scope_delta.json"
            delta = _read_json_artifact(delta_path, 64 * 1024)
            if not isinstance(delta, dict) or delta.get("added_paths") != added:
                _refuse("semantic scope delta is malformed")
            try:
                approval = read_scope_approval(
                    request_dir, expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
                )
            except (OSError, ApprovalError) as exc:
                _refuse(f"semantic scope approval is invalid: {exc}")
            if approval is None:
                continue
            if approval.decision is not ApprovalDecision.APPROVE:
                _refuse("semantic scope request was rejected")
        scope.update(added)
    return scope


def _safe_semantic_scope_path(path: Any) -> bool:
    return bool(
        isinstance(path, str) and path and path == path.strip()
        and path not in {".", ".."} and not path.startswith("/")
        and "\x00" not in path and "\\" not in path and "//" not in path
        and all(part not in {"", ".", "..", ".git"} for part in Path(path).parts)
        and not any(char in path for char in "*?[]{}")
    )


def _approved_scope(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path,
    plan: TaskPlanV2, checkpoint: ResumeCheckpoint, policy: EffectiveRepairScopePolicy,
) -> set[str]:
    """Every path any durable authority of cycles ``1..n`` approved."""

    scope = set(_plan_scope(plan))
    scope |= _contract_repair_scope(run_dir, 1, policy)
    cycle_scopes: dict[int, tuple[str, ...]] = {1: tuple(sorted(scope))}
    cycle_base_scopes: dict[int, tuple[str, ...]] = {1: tuple(sorted(scope))}
    cycle_kinds: dict[int, CycleKind] = {1: CycleKind.INITIAL}
    cycle_semantic_scopes: dict[int, set[str]] = {
        1: _semantic_revision_scope(config.repo, run_dir, 1, policy),
    }
    scope |= cycle_semantic_scopes[1]
    for number in range(2, checkpoint.review_cycle + 1):
        cycle = read_cycle_record(run_dir, number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            # Direct semantic corrections reuse the preceding approved plan,
            # while their final gate may still have its own check-repair
            # authority.
            cycle_semantic_scopes[number] = _semantic_revision_scope(
                config.repo, run_dir, number, policy,
            )
            scope |= cycle_semantic_scopes[number]
            cycle_base_scopes[number] = tuple(sorted(set(cycle_scopes[number - 1])))
            cycle_scopes[number] = tuple(sorted(
                set(cycle_base_scopes[number]) | cycle_semantic_scopes[number]
            ))
            cycle_kinds[number] = cycle.kind
            continue
        if number == checkpoint.review_cycle and checkpoint.phase is ResumePhase.REVIEW_REPLAN:
            # The correction plan of this cycle is not an authority yet: it
            # may still await its scope approval.
            continue
        correction, _bundle, bundle_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        verify_correction_scope(run_dir, number, bundle_sha, policy)
        cycle_base_scopes[number] = tuple(sorted(_plan_scope(correction)))
        cycle_base_scopes[number] = tuple(sorted(
            set(cycle_base_scopes[number]) | _contract_repair_scope(run_dir, number, policy)
        ))
        semantic_added = _semantic_revision_scope(config.repo, run_dir, number, policy)
        cycle_semantic_scopes[number] = semantic_added
        cycle_scopes[number] = tuple(sorted(set(cycle_base_scopes[number]) | semantic_added))
        cycle_kinds[number] = cycle.kind
        scope |= _plan_scope(correction)
        scope |= _contract_repair_scope(run_dir, number, policy)
        scope |= semantic_added
    for number, base in cycle_base_scopes.items():
        kind = cycle_kinds[number]
        stages = (
            (final_gate_stage(kind),)
            if kind is CycleKind.REVIEW_IMPLEMENTATION
            else (pre_semantic_gate_stage(kind), final_gate_stage(kind))
        )
        for stage in dict.fromkeys(stages):
            stage_scope = set(base)
            if stage == final_gate_stage(kind):
                stage_scope |= cycle_semantic_scopes.get(number, set())
            authority = gate_mutable_authority(
                run_dir, number, stage,
                base_paths=stage_scope,
                policy_config=policy,
            )
            scope |= set(authority.effective_paths)
    return scope


def _cycle_base_scope(
    config: HarnessConfig, selection: ExecutionSelection, run_dir: Path,
    plan: TaskPlanV2, number: int, policy: EffectiveRepairScopePolicy,
    *, include_current_semantic: bool = True,
) -> tuple[str, ...]:
    """Return the approved plan scope which is the base for one cycle."""

    current = tuple(sorted(
        set(_plan_scope(plan)) | _contract_repair_scope(run_dir, 1, policy)
    ))
    if number > 1 or include_current_semantic:
        current = tuple(sorted(
            set(current) | _semantic_revision_scope(config.repo, run_dir, 1, policy)
        ))
    for cycle_number in range(2, number + 1):
        cycle = read_cycle_record(run_dir, cycle_number)
        if cycle.kind is CycleKind.REVIEW_IMPLEMENTATION:
            if cycle_number != number or include_current_semantic:
                current = tuple(sorted(
                    set(current) | _semantic_revision_scope(config.repo, run_dir, cycle_number, policy)
                ))
            continue
        correction, _bundle, _bundle_sha = load_correction_plan(
            config, selection, run_dir, cycle_number,
            inherited_check_ids=plan.required_checks,
        )
        current = tuple(sorted(_plan_scope(correction)))
        current = tuple(sorted(set(current) | _contract_repair_scope(run_dir, cycle_number, policy)))
        if cycle_number != number or include_current_semantic:
            current = tuple(sorted(
                set(current) | _semantic_revision_scope(config.repo, run_dir, cycle_number, policy)
            ))
    return current


def _failure_tree_for(
    run_dir: Path, checkpoint: ResumeCheckpoint,
) -> str | None:
    """The tree a failed attempt at *checkpoint* durably recorded, if any."""

    number = checkpoint.review_cycle
    if checkpoint.phase in _STEP_PHASES:
        if checkpoint.phase is ResumePhase.REVIEW_IMPLEMENTATION:
            try:
                if read_cycle_record(run_dir, number).kind is CycleKind.REVIEW_IMPLEMENTATION:
                    return _read_tree_file(
                        semantic_revision_dir(run_dir, number) / "tree_after_failure.txt"
                    )
            except ResumeIntegrityError:
                return None
        record = _read_json_artifact(
            step_dir(run_dir, number, str(checkpoint.step_id)) / "step.json",
            128 * 1024,
        )
        if isinstance(record, dict) and record.get("status") == "FAILED" and _is_object_id(record.get("tree_after")):
            return record["tree_after"]
        if (
            isinstance(record, dict) and record.get("status") == "COMPLETED"
            and _is_object_id(record.get("tree_after"))
            and record.get("tree_before") == checkpoint.expected_tree_sha
            and record.get("tree_after") != record.get("tree_before")
            and record.get("commit_sha") is None
        ):
            # A worker success interrupted before its candidate became
            # durable: only an exact in-scope rollback is offered.
            return record["tree_after"]
        return None
    if checkpoint.phase is ResumePhase.SEMANTIC_REVISION:
        return _read_tree_file(semantic_revision_dir(run_dir, number) / "tree_after_failure.txt")
    if checkpoint.phase is ResumePhase.CHECK_REPAIR and checkpoint.stage is not None:
        return _read_tree_file(
            check_repair_attempt_dir(
                run_dir, number, checkpoint.stage, int(checkpoint.check_repair_attempt or 1),
            ) / "tree_after_failure.txt"
        )
    return None


def validate_resume(
    *,
    config: HarnessConfig,
    repair_scope: EffectiveRepairScopePolicy,
    run_dir: Path,
    state: Mapping[str, Any],
    checkpoint: ResumeCheckpoint,
    staging_remote: str,
) -> ResumedRun:
    """Fail-closed integrity gate in front of every resumed execution phase."""

    if state.get("planning_protocol") != "v2":
        _refuse("only pipeline v2 runs can be resumed")
    try:
        repo = git_root(config.repo)
    except GitError as exc:
        _refuse(f"repository is unavailable: {exc}")
    if str(repo) != str(state.get("repo")):
        _refuse("the configured repository is not the run repository")
    base_sha = state.get("base_sha")
    if not _is_object_id(base_sha):
        _refuse("run base SHA is invalid")
    reference = _read_repository_reference(run_dir)
    if reference is None or reference.base_sha != base_sha:
        _refuse("the base SHA changed for this run")
    try:
        spec = (run_dir / "spec.md").read_text(encoding="utf-8")
        context = (run_dir / "context.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _refuse("run SPEC or planner context is unreadable")

    # Plan identity, the human approval bound to it and the frozen checks.
    try:
        identity = compute_plan_identity_from_run(run_dir)
        recorded = plan_identity_from_mapping(state.get("plan_identity"))
        authority = read_check_authority(
            run_dir, expected_sha256=identity.checks_sha256,
            trusted_check_ids=tuple(check.id for check in config.trusted_checks()),
        )
        approval = read_plan_approval(run_dir, expected_identity=identity)
    except (ApprovalError, ResumeCheckpointError) as exc:
        _refuse(f"plan artifacts are invalid: {exc}")
    if identity != checkpoint.plan_identity or identity != recorded:
        _refuse("plan identity no longer matches")
    if identity.checks_sha256 is not None and authority is None:
        _refuse("check authority is missing for this run")
    if config.approval.require_plan_approval:
        if approval is None or approval.decision is not ApprovalDecision.APPROVE:
            _refuse("plan approval was not APPROVE")
        if (
            approval.execution_sha256 != checkpoint.execution_selection_sha256
            or approval.bundle_sha256 != identity.bundle_sha256
        ):
            _refuse("the approval does not bind the checkpoint execution selection")
    elif approval is not None and approval.decision is not ApprovalDecision.APPROVE:
        _refuse("plan approval artifact is not APPROVE")

    # The approved execution selection, against today's configuration.
    try:
        selection, execution_sha = read_execution_selection_with_sha256(run_dir)
        validate_execution_selection(config, selection)
    except (ExecutionSelectionError, ProfileError) as exc:
        _refuse(f"execution selection is invalid: {exc}")
    if execution_sha != checkpoint.execution_selection_sha256 or execution_sha != identity.execution_sha256:
        _refuse("execution selection hash changed")
    execution = state.get("execution") if isinstance(state.get("execution"), Mapping) else {}
    planner_state = execution.get("planner") if isinstance(execution.get("planner"), Mapping) else {}
    if selection.planner.profile_id != planner_state.get("profile_id"):
        _refuse("execution selection planner is not the run planner")

    # The approved plan and its exact bundle.
    try:
        plan = parse_task_plan_v2(
            (run_dir / "planner.raw.md").read_text(encoding="utf-8"),
            planning=config.planning,
            check_catalog=config.check_catalog,
            default_check_ids=config.default_check_ids,
        )
        bundle, bundle_sha = validate_implementation_bundle(
            run_dir, expected_step_ids=[step.id for step in plan.steps]
        )
    except (V2PlanParseError, OSError, UnicodeError) as exc:
        _refuse(f"approved plan is unreadable: {exc}")
    if plan.decision is not PlanDecision.READY or bundle_sha != identity.bundle_sha256:
        _refuse("approved bundle changed")
    if [item.step_id for item in selection.steps] != [step.id for step in plan.steps]:
        _refuse("execution selection steps do not match the plan")

    # Worktree and branch.
    worktree_value, branch = state.get("worktree"), state.get("branch")
    if not isinstance(worktree_value, str) or not isinstance(branch, str):
        _refuse("run has no worktree or branch")
    worktree = Path(worktree_value).expanduser().resolve()
    if not worktree.is_dir():
        _refuse("run worktree is missing")
    try:
        frozen_config, frozen_ids = config_with_check_authority(
            config, run_dir, expected_sha256=identity.checks_sha256,
        )
        if frozen_ids is not None:
            for frozen_check in frozen_config.select_checks(frozen_ids):
                resolve_check_cwd(worktree, frozen_check)
    except (ValidationError, ValueError) as exc:
        _refuse(f"check authority is invalid: {exc}")

    # Cycle identity and every authority of cycles 1..n.
    number = checkpoint.review_cycle
    if number > 1 or cycle_record_path(run_dir, 1).exists():
        if read_cycle_record(run_dir, 1).kind is not CycleKind.INITIAL:
            _refuse("cycle 001 is not the initial cycle")
    for earlier in range(2, number + 1):
        read_cycle_record(run_dir, earlier)
    for earlier in range(1, number):
        read_candidate_record(run_dir, earlier)
    validate_correction_bindings(run_dir, number)
    current_cycle = read_cycle_record(run_dir, number) if number > 1 else RunCycle(1, CycleKind.INITIAL)
    scope = _approved_scope(config, selection, run_dir, plan, checkpoint, repair_scope)
    if (
        checkpoint.phase not in {ResumePhase.REVIEW_REPLAN, *_STEP_PHASES}
        and number > 1
        and current_cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION
    ):
        if checkpoint.correction_bundle_sha256 is None:
            _refuse("the correction checkpoint is not bound to its plan")
        _correction, _bundle, correction_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        if correction_sha != checkpoint.correction_bundle_sha256:
            _refuse("the correction plan changed")
    if checkpoint.phase is ResumePhase.REVIEW_IMPLEMENTATION and current_cycle.kind is CycleKind.REVIEW_REPLAN:
        _correction, _bundle, correction_sha = load_correction_plan(
            config, selection, run_dir, number, inherited_check_ids=plan.required_checks,
        )
        if correction_sha != checkpoint.correction_bundle_sha256:
            _refuse("the correction plan changed")

    restore: tuple[str, ...] = ()
    try:
        if str(worktree) not in registered_worktrees(repo):
            _refuse("run worktree is not registered in the repository")
        if not branch_exists(repo, branch):
            _refuse("run branch is missing")
        if symbolic_head(worktree) != f"refs/heads/{branch}":
            _refuse("worktree HEAD is not the run branch")
        head = current_head(worktree)
        if resolve_commit(repo, f"refs/heads/{branch}") != head:
            _refuse("run branch does not point to the worktree HEAD")
        expected_tree = checkpoint.expected_tree_sha
        if head != checkpoint.expected_head_sha:
            # A crash may land between an accepted commit and the next
            # checkpoint; only that exact commit (expected parent, expected
            # tree) is kept.  At a gate boundary it must also be the commit
            # its durable gate acceptance recorded.
            advanced = (
                commit_parents(repo, head) == (checkpoint.expected_head_sha,)
                and resolve_tree(repo, head) == expected_tree
            )
            if advanced and checkpoint.phase is ResumePhase.DETERMINISTIC_GATE and checkpoint.stage is not None:
                acceptance = _read_json_artifact(
                    gate_acceptance_path(run_dir, number, checkpoint.stage)
                )
                advanced = (
                    isinstance(acceptance, dict)
                    and acceptance.get("commit_created") is True
                    and acceptance.get("commit_sha") == head
                    and acceptance.get("parent_sha") == checkpoint.expected_head_sha
                )
                if advanced:
                    _validate_gate_acceptance(
                        run_dir, number, checkpoint.stage, tree=expected_tree, head=head,
                        base_scope=_cycle_base_scope(
                            config, selection, run_dir, plan, number, repair_scope,
                            include_current_semantic=(
                                current_cycle.kind is CycleKind.REVIEW_IMPLEMENTATION
                                or checkpoint.stage != pre_semantic_gate_stage(current_cycle.kind)
                            ),
                        ),
                        policy=repair_scope,
                    )
            elif checkpoint.phase is ResumePhase.STEP_ACCEPTANCE:
                # Committed before the next boundary: the step acceptance
                # re-proves that commit against its durable candidate.
                pass
            elif checkpoint.phase is not ResumePhase.CANDIDATE_READY:
                advanced = False
            if not advanced:
                _refuse("HEAD moved since the checkpoint")
        if (
            checkpoint.expected_parent_sha is not None
            and commit_parents(repo, head) != (checkpoint.expected_parent_sha,)
        ):
            _refuse("HEAD parent differs from the checkpoint parent")
        # Every earlier reviewed candidate is a real commit with its recorded
        # tree and parent, and the run branch still descends from it.
        for earlier in range(1, number):
            record = read_candidate_record(run_dir, earlier)
            earlier_parents = (
                (record["parent_sha"],) if record.get("parent_sha") is not None else ()
            )
            if (
                resolve_tree(repo, record["commit_sha"]) != record["tree_sha"]
                or (
                    record.get("no_change") is not True
                    and commit_parents(repo, record["commit_sha"]) != earlier_parents
                )
                or not is_ancestor(repo, record["commit_sha"], head)
            ):
                _refuse(f"cycle {earlier:03d} candidate record is not in the run history")

        if checkpoint.phase in _CANDIDATE_PHASES:
            candidate = read_candidate_record(run_dir, number)
            if candidate["commit_sha"] != head or candidate["tree_sha"] != expected_tree:
                _refuse("the run branch is not the recorded candidate commit")
            candidate_parents = commit_parents(repo, candidate["commit_sha"])
            expected_candidate_parents = (
                (candidate["parent_sha"],)
                if candidate.get("parent_sha") is not None else ()
            )
            evidence = candidate_evidence(run_dir, number)
            if (
                (
                    candidate.get("no_change") is not True
                    and candidate_parents != expected_candidate_parents
                )
                or (
                    candidate.get("parent_sha") is None
                    and (
                        candidate.get("no_change") is not True
                        or evidence is None
                        or bool(evidence.changed_files)
                    )
                )
                or (
                    candidate.get("no_change") is True
                    and (
                        candidate.get("parent_sha") is not None
                        or evidence is None or evidence.diff != ""
                        or evidence.base_sha != base_sha
                        or not evidence.deterministic_passed
                        or not required_checks_passed(evidence)
                    )
                )
            ):
                _refuse("the candidate parent identity is invalid")
            try:
                candidate_stage = GateStage(candidate["gate_stage"])
            except (KeyError, TypeError, ValueError):
                _refuse("the candidate gate stage is invalid")
            cycle_base_scope = _cycle_base_scope(
                config, selection, run_dir, plan, number, repair_scope,
                include_current_semantic=(
                    candidate_stage is final_gate_stage(current_cycle.kind)
                ),
            )
            _validate_gate_acceptance(
                run_dir, number, candidate_stage, tree=expected_tree, head=head,
                base_scope=cycle_base_scope, policy=repair_scope,
            )
            remote_required = config.publish.enabled or (
                config.github.enabled and config.github.pull_request_mode == "create"
            )
            if checkpoint.phase is ResumePhase.PUBLISH and remote_required:
                try:
                    remote_tip = remote_run_branch_tip(
                        repo, remote=staging_remote, branch=branch
                    )
                except (GitError, OSError):
                    _refuse("required remote candidate could not be verified")
                if remote_tip != head:
                    _refuse("remote run branch does not point to the candidate commit")
                if (
                    candidate.get("remote") != staging_remote
                    or candidate.get("remote_branch") != branch
                    or candidate.get("remote_sha") != head
                    or candidate.get("remote_status") != "available"
                    or not isinstance(candidate.get("pushed_at"), str)
                    or not candidate.get("pushed_at")
                ):
                    _refuse("candidate record does not prove the exact remote authority")
        if checkpoint.phase is ResumePhase.PUBLISH:
            evidence = candidate_evidence(run_dir, number)
            if evidence is None or evidence.staged_tree_sha != expected_tree:
                _refuse("the candidate evidence is missing or not for the approved tree")
            review = _accepted_review(review_dir(run_dir, number), evidence, head)
            if review is None or review.verdict is not ReviewVerdict.PASS or review.route is not ReviewRoute.NONE:
                _refuse("the reviewer PASS is missing for the candidate")
        if current_cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION and (
            checkpoint.phase is ResumePhase.SEMANTIC_REVISION
            or (
                checkpoint.phase in {ResumePhase.DETERMINISTIC_GATE, ResumePhase.CHECK_REPAIR}
                and checkpoint.stage is final_gate_stage(current_cycle.kind)
            )
        ):
            # Semantic revision never precedes a green gate: its pre-semantic
            # acceptance must durably name the HEAD the revision started from.
            pre_head = checkpoint.expected_head_sha
            _validate_gate_acceptance(
                run_dir, number, pre_semantic_gate_stage(current_cycle.kind),
                tree=resolve_tree(repo, pre_head), head=pre_head,
                base_scope=_cycle_base_scope(
                    config, selection, run_dir, plan, number, repair_scope,
                    include_current_semantic=False,
                ),
                policy=repair_scope,
            )
        if checkpoint.phase is ResumePhase.CHECK_REPAIR:
            evidence = _load_evidence(gate_dir(run_dir, number, checkpoint.stage))
            if evidence is None or evidence.staged_tree_sha != expected_tree:
                _refuse("the red gate evidence is missing or not for the checkpoint tree")
        if checkpoint.phase is ResumePhase.DETERMINISTIC_GATE and checkpoint.check_repair_attempt is not None:
            record = _read_json_artifact(
                check_repair_attempt_dir(
                    run_dir, number, checkpoint.stage, checkpoint.check_repair_attempt,
                ) / "attempt.json"
            )
            if not isinstance(record, dict) or record.get("tree_after") != expected_tree:
                _refuse("the check-repair attempt record is not for the checkpoint tree")

        candidate_tree = candidate_tree_sha(worktree)
        index_tree = index_tree_sha(worktree)
        dirty = _status_has_unstaged_or_untracked(status_porcelain(worktree))
        if candidate_tree != expected_tree or index_tree != expected_tree or dirty:
            failure_tree = _failure_tree_for(run_dir, checkpoint)
            if failure_tree is None or candidate_tree != failure_tree:
                _refuse("the worktree differs from the checkpoint tree")
            changed = changed_paths_between_trees(repo, expected_tree, candidate_tree)
            if not changed or any(path not in scope for path in changed):
                raise ResumeRequiresOperatorError(
                    "the failed attempt changed paths outside the approved scope"
                )
            restore = tuple(changed)
        base_tree = resolve_tree(repo, base_sha)
        unapproved = [
            path for path in changed_paths_between_trees(repo, base_tree, expected_tree)
            if path not in scope
        ]
    except GitError as exc:
        _refuse(f"Git state is unreadable: {exc}")
    if unapproved:
        _refuse("the candidate contains a path outside the approved scope")
    return ResumedRun(
        checkpoint=checkpoint, plan=plan, bundle=bundle, selection=selection,
        info=WorktreeInfo(
            source_repo=repo, worktree=worktree, branch=branch,
            base_ref=config.base_ref, base_sha=base_sha,
        ),
        repository_reference=reference, spec=spec, context=context,
        base_tree_sha=base_tree, restore_paths=restore,
    )


__all__ = [
    "ResumedRun", "candidate_evidence", "completed_step_records", "correction_binding",
    "load_correction_plan", "read_candidate_record", "read_cycle_record",
    "validate_correction_bindings", "validate_resume", "verify_correction_scope",
]
