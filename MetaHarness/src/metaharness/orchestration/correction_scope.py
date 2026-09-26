"""The one mutable-scope authority of a correction.

Every path by which a correction may widen an approved envelope goes through
this module: a review-driven correction and a red-gate cycle replan bind their
delta here, the reviser's own scope request is authorised here, and the
trusted-check preflight a correction requires runs here before any expensive
worker is paid.

The delta is always derived from the parsed plan alone and created once -- the
persisted bytes are only ever compared -- and the run's own policy, never the
producer of the plan, decides whether an addition is applied, waits for a
human or is refused.
"""

from __future__ import annotations

import hashlib
from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Any,
    Mapping,
    Sequence,
    TYPE_CHECKING,
)
from ..agent.base import AGENT_SCOPE_VIOLATION
from ..approval import (
    ApprovalDecision,
    read_scope_approval,
)
from ..gitops import (
    GitError,
    path_exists_in_tree,
)
from ..models import (
    RunCycle,
    RunDisposition,
    RunMachineState,
    SCOPE_APPROVAL_REASON,
    TaskPlanV2,
)
from ..result import atomic_write_text
from ..resume import ResumeIntegrityError
from ..review import ReviewResult
from ..state import RunStateStore
from ..validation import config_with_check_authority
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    correction_dir,
)
from .shared import (
    OrchestrationError,
    ScopeApprovalRequired,
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _create_file_once,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    bounded_v2_report,
)
from .worker_recovery import safe_scope_request_path
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime




def _repair_mutation_sets(plan: TaskPlanV2) -> tuple[list[str], list[str], list[str]]:
    """Return canonical correction mutation sets and reject structural ambiguity."""

    writes = sorted({path for step in plan.steps for path in step.write_set})
    creates = sorted({path for step in plan.steps for path in step.create_set})
    deletes = sorted({path for step in plan.steps for path in step.delete_set})
    if (set(writes) & set(creates)) or (set(writes) & set(deletes)) or (set(creates) & set(deletes)):
        raise OrchestrationError("REPAIR_SCOPE_MUTATION_SETS_OVERLAP")
    for path in (*writes, *creates, *deletes):
        posix = PurePosixPath(path)
        if (not path or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in posix.parts)
                or any(char in path for char in "*?[")):
            raise OrchestrationError("REPAIR_SCOPE_UNSAFE_PATH")
    return writes, creates, deletes


def build_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, repair_bundle_sha: str, justification: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta, in memory only: from parsed plan sets.

    The delta is derived from the parsed plan alone; *justification* names the
    durable failure the added paths answer, and is never reviewer or planner
    prose.  Every plan that may widen an approved envelope -- a review
    correction and a red-gate cycle replan alike -- goes through this one
    builder, so the same plan always produces the same delta bytes.
    """

    writes, creates, deletes = _repair_mutation_sets(plan)
    requested = sorted(set(writes) | set(creates) | set(deletes))
    original = sorted(set(original_scope))
    added = sorted(set(requested) - set(original))
    unchanged = sorted(set(requested) & set(original))
    findings = justification.strip()
    reasons: dict[str, Any] = {}
    for path in added:
        steps = [step for step in plan.steps if path in set(step.write_set) | set(step.create_set) | set(step.delete_set)]
        step = steps[0]
        reasons[path] = {
            "reason": f"{step.title}: {step.objective}",
            "source_finding": findings,
        }
    try:
        raw_plan = (repair_dir / "planner.raw.md").read_bytes()
    except OSError as exc:
        raise OrchestrationError("REPAIR_SCOPE_PLAN_UNREADABLE") from exc
    plan_sha = hashlib.sha256(raw_plan).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "original_mutable_paths": original,
        "requested_write_paths": writes,
        "requested_create_paths": creates,
        "requested_delete_paths": deletes,
        "added_paths": added,
        "unchanged_paths": unchanged,
        "added_path_reasons": reasons,
        "source_finding": findings,
        "candidate_commit_sha": candidate_commit_sha,
        "repair_plan_sha256": plan_sha,
        "correction_bundle_sha256": repair_bundle_sha,
    }
    return payload, _json_text(payload)


def review_scope_delta(
    repair_dir: Path, *, original_scope: list[str], plan: TaskPlanV2,
    candidate_commit_sha: str, review: ReviewResult, repair_bundle_sha: str,
) -> tuple[dict[str, Any], str]:
    """The canonical scope delta of one review-driven correction cycle."""

    return build_scope_delta(
        repair_dir, original_scope=original_scope, plan=plan,
        candidate_commit_sha=candidate_commit_sha, repair_bundle_sha=repair_bundle_sha,
        justification=review.required_fixes.strip() or review.findings.strip(),
    )


def ensure_scope_delta(
    repair_dir: Path, content: str, *, expected_sha256: str | None,
) -> str:
    """Persist ``scope_delta.json`` exactly once, then only verify it.

    The first creation writes the canonical bytes atomically.  An existing
    artifact is never rewritten: its bytes must equal the canonical bytes and,
    when the checkpoint binds one, the checkpoint hash.  Any difference is a
    :class:`ResumeIntegrityError` and the file is left as found.
    """

    expected = content.encode("utf-8")
    digest = hashlib.sha256(expected).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ResumeIntegrityError("the correction scope delta changed")
    path = repair_dir / "scope_delta.json"
    if expected_sha256 is None:
        try:
            _create_file_once(path, expected)
            return digest
        except FileExistsError:
            pass
        except OSError as exc:
            raise OrchestrationError("REPAIR_SCOPE_DELTA_UNWRITABLE") from exc
    try:
        if path.stat().st_size > 256 * 1024:
            raise ResumeIntegrityError("the correction scope delta is too large")
        existing = path.read_bytes()
    except OSError as exc:
        raise ResumeIntegrityError(f"the correction scope delta is unreadable: {exc}") from exc
    if existing != expected:
        raise ResumeIntegrityError("the correction scope delta changed")
    return digest

class CorrectionScopeService:
    """The mutable-scope policy of one run, applied to every correction."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def authorize_review_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, bundle_sha: str, candidate_sha: str, review: ReviewResult,
        approved_scope: list[str],
    ) -> None:
        """Bind the correction scope delta and apply the run's scope policy."""

        repair_dir = correction_dir(ctx.run_dir, cycle)
        try:
            delta, content = review_scope_delta(
                repair_dir, original_scope=approved_scope, plan=plan,
                candidate_commit_sha=candidate_sha, review=review,
                repair_bundle_sha=bundle_sha,
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        self.apply(
            store, ctx, cycle, plan, repair_dir, delta, content, approved_scope, candidate_sha,
        )

    def apply(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, repair_dir: Path, delta: Mapping[str, Any], content: str,
        approved_scope: list[str], candidate_sha: str,
    ) -> None:
        """Persist one correction scope delta and let the run's policy decide.

        A review-driven correction and a red-gate cycle replan widen an
        approved envelope through this one path: the delta is derived from the
        parsed plan alone, created once, and never approved by its producer.
        """

        # Created once; on every later pass (resume included) the persisted
        # bytes are only compared, never repaired.
        delta_sha = ensure_scope_delta(repair_dir, content, expected_sha256=None)
        for path in delta["requested_write_paths"] + delta["requested_delete_paths"]:
            if not path_exists_in_tree(ctx.repo, candidate_sha, path):
                raise PipelineFailure("REPAIR_SCOPE_EXISTING_PATH_MISSING", path)
        for path in delta["requested_create_paths"]:
            if path_exists_in_tree(ctx.repo, candidate_sha, path):
                raise PipelineFailure("REPAIR_SCOPE_CREATE_PATH_EXISTS", path)
        # A widening is only ever justified by the durable failure it answers:
        # the delta carries that finding itself, so a producer cannot approve
        # its own expansion and an empty finding can never widen a scope.
        if delta["added_paths"] and not str(delta.get("source_finding") or "").strip():
            raise PipelineFailure("REPAIR_SCOPE_UNJUSTIFIED")
        requested = sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        atomic_write_text(repair_dir / "scope.json", _json_text({
            "repair_mutable_scope": requested,
            "approved_mutable_scope_before": approved_scope,
            "scope_delta_sha256": delta_sha,
        }))
        added = delta["added_paths"]
        policy = self.runtime.repair_scope
        if added and policy.policy == "deny-expansion":
            self.runtime.cycle_update(store, cycle, status="failed", failure="REPAIR_SCOPE_EXPANSION",
                               scope_delta=delta)
            raise PipelineFailure("REPAIR_SCOPE_EXPANSION")
        if added and (
            policy.policy == "require-approval"
            or (policy.policy == "auto-bounded" and len(added) > policy.max_added_paths)
        ):
            approval = read_scope_approval(repair_dir, expected_sha256=delta_sha)
            if approval is None:
                self.runtime.cycle_update(store, cycle, status="waiting_scope_approval", scope_delta=delta)
                store.set_run_state(RunMachineState(
                    disposition=RunDisposition.WAIT_HUMAN, reason=SCOPE_APPROVAL_REASON,
                ), scope_delta=delta, current_step=None)
                raise ScopeApprovalRequired()
            if approval.decision is not ApprovalDecision.APPROVE:
                raise PipelineFailure("HUMAN_REQUIRED", "correction scope rejected")
        elif added:
            self.runtime.cycle_update(store, cycle, status="scope_auto_approved", scope_delta=delta)
        # The correction plan may require trusted checks the initial plan did
        # not; their config-only preflights run before any expensive worker.
        check_config, check_ids = config_with_check_authority(
            self.runtime.config, ctx.run_dir, requested_check_ids=plan.required_checks,
            expected_sha256=self.runtime.approved_check_authority_sha256(ctx.run_dir),
        )
        preflight_failures = self.runtime.gates.run_check_preflights_recoverably(
            store=store, worktree=ctx.info.worktree, check_config=check_config,
            check_ids=check_ids or plan.required_checks,
            counter_key=f"check-preflight:cycle:{cycle.number:03d}",
            phase="planning", cycle=cycle.number,
        )
        if preflight_failures:
            raise PipelineFailure(
                preflight_failures[0].split(":", 1)[0], preflight_failures[0],
            )

    def authorize_semantic_scope_request(
        self,
        store: RunStateStore,
        ctx: PipelineV2Context,
        cycle_plan: CyclePlan,
        artifact_dir: Path,
        current_scope: list[str],
        *,
        approved_scope: Sequence[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Persist and apply a strict, policy-bounded reviser scope request."""

        report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
        request = report.get("scope_request") if isinstance(report, dict) else None
        paths = request.get("paths") if isinstance(request, dict) else None
        reason = request.get("reason") if isinstance(request, dict) else None
        evidence = request.get("evidence") if isinstance(request, dict) else None
        tree_sha = report.get("tree_before") if isinstance(report, dict) else None
        if (
            not isinstance(paths, list) or not paths or any(
                not isinstance(path, str) or not safe_scope_request_path(path)
                for path in paths
            ) or len(paths) != len(set(paths))
            or not isinstance(reason, str) or not reason.strip()
            or not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence)
            or not isinstance(tree_sha, str) or not _is_object_id(tree_sha)
        ):
            raise PipelineFailure(AGENT_SCOPE_VIOLATION, "semantic scope request is malformed")
        source_report = artifact_dir / "report.json"
        try:
            source_report_sha = hashlib.sha256(source_report.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_REQUIRES_OPERATOR", "semantic scope request report is unreadable",
            ) from exc
        try:
            exists = {
                path: path_exists_in_tree(ctx.repo, tree_sha, path) for path in paths
            }
        except GitError as exc:
            raise PipelineFailure("RESUME_REQUIRES_OPERATOR", "scope request tree semantics are unreadable") from exc
        base = tuple(sorted(set(current_scope)))
        requested = tuple(sorted(set(paths)))
        if approved_scope is not None and not set(requested).issubset(set(approved_scope)):
            raise PipelineFailure(
                AGENT_SCOPE_VIOLATION,
                "scope request exceeds the cycle's approved mutable scope",
            )
        added = tuple(path for path in requested if path not in base)
        root = artifact_dir / "scope_requests"
        root.mkdir(parents=True, exist_ok=True)
        existing_added: set[str] = set()
        next_number = 1
        prior_request_dir: Path | None = None
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.isdigit():
                continue
            next_number = max(next_number, int(path.name) + 1)
            saved = _read_json_artifact(path / "authority.json", 64 * 1024)
            if isinstance(saved, dict):
                saved_added = saved.get("added_paths")
                if isinstance(saved_added, list):
                    existing_added.update(item for item in saved_added if isinstance(item, str))
                if (
                    saved.get("tree_sha") == tree_sha
                    and saved.get("base_mutable_scope") == list(base)
                    and saved.get("requested_paths") == list(requested)
                    and saved.get("reason") == reason
                    and saved.get("evidence") == [item[:1000] for item in evidence[:16]]
                ):
                    prior_request_dir = path
        target = prior_request_dir or root / f"{next_number:03d}"
        target.mkdir(parents=True, exist_ok=True)
        added_all = tuple(sorted(existing_added | set(added)))
        policy = self.runtime.repair_scope
        authority = {
            "schema_version": 1,
            "cycle": cycle_plan.cycle.number,
            "tree_sha": tree_sha,
            "source_report_sha256": source_report_sha,
            "base_mutable_scope": list(base),
            "requested_paths": list(requested),
            "added_paths": list(added),
            "existing_paths": [path for path in requested if exists[path]],
            "create_paths": [path for path in requested if not exists[path]],
            "reason": reason[:2000],
            "evidence": [item[:1000] for item in evidence[:16]],
            "policy": policy.policy,
            "bound": policy.max_added_paths,
        }
        authority_path = target / "authority.json"
        authority_content = _json_text(authority)
        if authority_path.exists():
            saved_authority = _read_json_artifact(authority_path, 64 * 1024)
            saved_semantics = (
                {key: value for key, value in saved_authority.items()
                 if key != "source_report_sha256"}
                if isinstance(saved_authority, dict) else None
            )
            current_semantics = {
                key: value for key, value in authority.items()
                if key != "source_report_sha256"
            }
            if saved_semantics != current_semantics:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "semantic scope request authority changed")
            authority = saved_authority
        else:
            atomic_write_text(authority_path, authority_content)

        if not added:
            atomic_write_text(target / "decision.json", _json_text({
                **authority, "decision": "recorded-in-scope",
            }))
            atomic_write_text(artifact_dir / "status.json", _json_text({
                "status": "SCOPE_REQUEST_RECORDED",
                "reason": reason[:2000],
                "requested_paths": list(requested),
            }))
            return "recorded", current_scope
        if policy.policy == "deny-expansion":
            denied = {**authority, "decision": "denied-expansion"}
            atomic_write_text(target / "decision.json", _json_text(denied))
            status = {
                "status": "REPLAN_REQUIRED",
                "reason": "semantic scope expansion denied by recovery policy",
            }
            atomic_write_text(artifact_dir / "status.json", _json_text(status))
            report_text = (
                "SEMANTIC REVISION: SCOPE EXPANSION DENIED\nroute=REPLAN\n"
                f"reason={bounded_v2_report(reason)}"
            )
            self.runtime.cycle_update(
                store, cycle_plan.cycle, status="scope_expansion_denied",
                semantic_revision_status="REPLAN_REQUIRED",
                semantic_revision_report=report_text,
            )
            return "replan", current_scope

        delta_path = target / "scope_delta.json"
        delta = {
            "schema_version": 1, "cycle": cycle_plan.cycle.number,
            "tree_sha": tree_sha, "added_paths": list(added),
            "requested_paths": list(requested), "reason": reason[:2000],
            "evidence": [item[:1000] for item in evidence[:16]],
            "policy": policy.policy, "bound": policy.max_added_paths,
        }
        delta_content = _json_text(delta)
        if delta_path.exists() and delta_path.read_text(encoding="utf-8") != delta_content:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "semantic scope delta changed")
        if not delta_path.exists():
            atomic_write_text(delta_path, delta_content)
        delta_sha = hashlib.sha256(delta_path.read_bytes()).hexdigest()
        approval = read_scope_approval(target, expected_sha256=delta_sha)
        requires_approval = policy.policy == "require-approval" or len(added_all) > policy.max_added_paths
        if requires_approval and approval is None:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            delta["approval_artifact"] = delta_path.relative_to(ctx.run_dir).as_posix()
            store.set_run_state(RunMachineState(
                disposition=RunDisposition.WAIT_HUMAN, reason=SCOPE_APPROVAL_REASON,
            ), scope_delta=delta, current_step=None)
            self.runtime.cycle_update(
                store, cycle_plan.cycle, status="waiting_scope_approval", scope_delta=delta,
            )
            raise ScopeApprovalRequired()
        if approval is not None and approval.decision is not ApprovalDecision.APPROVE:
            raise PipelineFailure("HUMAN_REQUIRED", "semantic scope request was rejected")
        if requires_approval and approval is None:
            raise ScopeApprovalRequired()
        return "expanded", sorted(set(current_scope) | set(added))

