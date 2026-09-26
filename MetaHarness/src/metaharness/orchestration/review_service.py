"""Candidate review and the review-driven correction cycles.

This module owns every review decision: building the review context from
durable artifacts, invoking the reviewer and recovering reviewer-side
failures, recording the verdict, the human routes and the review-repair
budget, as well as the semantic revision and the correction plans a review
drives.  It never runs a deterministic check nor publishes anything.
"""

from __future__ import annotations

import hashlib, json, time
from pathlib import Path
from typing import (
    Any,
    Mapping,
    NoReturn,
    Sequence,
    TYPE_CHECKING,
)
from ..agent.base import (
    AGENT_RUNTIME_FAILED,
    AGENT_PROTOCOL_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentScopeError,
)
from ..prompt_contracts import build_final_review_payload
from ..approval import (
    ApprovalDecision,
    read_scope_approval,
)
from ..evidence import (
    EvidenceBundle,
    bounded_semantic_diff,
    required_checks_passed,
)
from ..validation import (
    check_result_json,
    config_with_check_authority,
)
from ..gitops import (
    GitError,
    commit_parents,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    immutable_commit_web_url,
    compare_commits_web_url,
    render_repository_reference,
    path_exists_in_tree,
    remote_run_branch_tip,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
    status_porcelain,
    symbolic_head,
)
from ..models import (
    CycleKind,
    ExecutionRole,
    GateStage,
    PlanDecision,
    ReviewRoute,
    ReviewVerdict,
    RunCycle, RunDisposition, RunMachineState, SCOPE_APPROVAL_REASON, correction_cycles_used,
)
from ..planning.artifacts import validate_implementation_bundle
from ..planning.check_replan import (
    CheckReplanFacts,
    CheckReplanTransaction,
    bounded_check_proofs,
    bounded_diff_summary,
    check_replan_dir,
    plan_identity,
    render_path_facts,
)
from ..planning.planner import RepairPlannerV2
from ..planning.protocol import (
    V2PlanParseError,
    render_repair_plan_summary,
    render_repair_step_index,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
)
from ..resume import ResumeIntegrityError
from ..usage import read_usage_artifact
from ..redaction import redact
from ..profiles import (
    ProfileError,
    build_llm_endpoint,
    profile_for_role,
)
from ..result import (
    RunResult,
    atomic_write_text,
    write_repair_task,
)
from ..review import (
    Reviewer,
    ReviewParseError,
    ReviewResult,
    structured_review_reason,
)
from ..state import RunStateStore
from .shared import (
    OrchestrationError,
    ScopeApprovalRequired,
    _REVIEW_ATTEMPT_ARTIFACTS,
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _bounded_report,
    bounded_v2_report,
    _check_payload,
    _git_ownership,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    _record_failure_tree,
    _repair_checks_payload,
    _safe_candidate_tree,
    bounded_parse_detail,
    chat_client,
)
from .revision import (
    ReviewContextBuilder,
    ReviewCycleInput,
    RevisionRunner,
    SCOPE_REQUEST_ROUTE,
    _bounded_previous_revision_report,
    _review_payload,
    review_cycle_revision_report,
)
from .check_repair import (
    _check_repair_prompt,
    hard_integrity_failures,
    check_failure_proofs,
    consumed_ladder_strategies,
    gate_mutable_authority,
    soft_check_failures,
)
from .recovery import GateRecoveryStep, RecoveryStepUnavailable
from .scope_repair import (
    _build_scope_delta,
    build_scope_delta,
    ensure_scope_delta,
)
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    correction_dir,
    gate_dir,
    gate_acceptance_path,
    review_dir,
    semantic_revision_dir,
    pre_semantic_gate_stage,
)
from .worker_recovery import safe_scope_request_path
from .resume_validation import (
    _accepted_review,
    load_evidence,
    load_revision,
    _read_planner_conversation,
    _reusable_pre_checks,
    candidate_evidence,
    read_candidate_record,
    read_cycle_record,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime

_MAX_REVIEW_FALLBACK_DIFF_BYTES = 32 * 1024

def _review_candidate_record(run_dir: Path, number: int) -> Mapping[str, Any] | None:
    """The candidate of an earlier cycle, or ``None`` when a replan replaced it.

    A cycle whose red deterministic gate exhausted its ladder was answered by a
    new decomposition in the next cycle and never reached a candidate; the
    reviewer reads that cycle as replaced instead of as a reviewed history.
    """

    following = run_dir / "cycles" / f"{number + 1:03d}" / "cycle.json"
    if following.is_file() and read_cycle_record(run_dir, number + 1).kind is CycleKind.CHECK_REPLAN:
        return None
    return read_candidate_record(run_dir, number)


def _required_checks_summary(evidence: EvidenceBundle) -> str:
    """Keep required-check authority while excluding stdout/stderr blobs."""

    rows: list[dict[str, Any]] = []
    required = set(evidence.required_check_ids)
    for raw in evidence.checks:
        item = dict(raw) if isinstance(raw, Mapping) else check_result_json(raw)
        name = item.get("name")
        if name not in required and required:
            continue
        rows.append({
            "id": name,
            "required": bool(item.get("required", name in required)),
            "exit_code": item.get("exit_code"),
            "timed_out": bool(item.get("timed_out", False)),
            "workspace_mutated": bool(item.get("workspace_mutated", False)),
            "status": (
                "timed_out" if item.get("timed_out") else
                "passed" if item.get("exit_code") == 0 else "failed"
            ),
        })
    return _json_text({
        "required_check_ids": list(evidence.required_check_ids),
        "failures": list(evidence.failures),
        "checks": rows,
        "deterministic_passed": evidence.deterministic_passed,
    })

def _diffstat(diff: str, changed_files: Sequence[str]) -> str:
    additions = sum(
        1 for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return _json_text({
        "files": len(tuple(changed_files)),
        "insertions": additions,
        "deletions": deletions,
    })

def _compact_cycle_summary(text: str) -> str:
    """Remove check output and worker narration from cycle history."""

    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {
                    "stdout", "stderr", "stdout_tail", "stderr_tail",
                    "final", "final_message", "agent_report", "step_reports",
                    "semantic_revision_report",
                }
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return _json_text(clean(value))

def review_code_evidence(
    *,
    repository_reference: RepositoryReference,
    base_sha: str,
    candidate_sha: str,
    evidence: EvidenceBundle,
    remote_sha: str | None = None,
    remote_branch: str | None = None,
    remote_name: str | None = None,
    force_inline_diff: bool = False,
) -> str:
    diff_bytes = evidence.diff.encode("utf-8", errors="replace")

    candidate_url = immutable_commit_web_url(
        repository_reference,
        candidate_sha,
    )
    compare_url = compare_commits_web_url(
        repository_reference,
        base_sha,
        candidate_sha,
    )

    # A web URL describes where a candidate might be inspectable; it does not
    # prove that the exact candidate was pushed. Durable remote authority is
    # required before remote exploration can be offered to a reviewer.
    remote_pushed = (
        remote_sha == candidate_sha
        and isinstance(remote_branch, str)
        and bool(remote_branch)
        and isinstance(remote_name, str)
        and bool(remote_name)
    )
    remote_available = (
        remote_pushed and candidate_url is not None and compare_url is not None
        and not force_inline_diff
    )

    payload: dict[str, Any] = {
        "authority": "immutable_candidate_commit",
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "candidate_tree_sha": evidence.staged_tree_sha,
        "candidate_url": candidate_url,
        "compare_url": compare_url,
        "remote_exploration": "AVAILABLE" if remote_available else "UNAVAILABLE",
        "remote_authority": {
            "remote": remote_name,
            "remote_branch": remote_branch,
            "remote_sha": remote_sha,
            "verified": remote_pushed,
        },
        "full_diff_bytes": len(diff_bytes),
        "full_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "diffstat": json.loads(_diffstat(evidence.diff, evidence.changed_files)),
        "inline_full_diff": False,
    }

    if not remote_available:
        excerpt, truncated, full_bytes = bounded_semantic_diff(
            evidence.diff,
            _MAX_REVIEW_FALLBACK_DIFF_BYTES,
        )
        payload["inline_fallback"] = {
            "truncated": truncated,
            "full_diff_bytes": full_bytes,
            "excerpt": excerpt,
        }

    return _json_text(payload)


# The bounded path facts one red gate may show a planner.
_MAX_REPLAN_PATH_FACTS = 24


def _check_replan_facts(
    *, ctx: PipelineV2Context, cycle_plan: CyclePlan, cycle: int, stage: GateStage,
    evidence: EvidenceBundle, approved_scope: Sequence[str],
) -> CheckReplanFacts:
    """The bounded facts one red deterministic gate hands a re-decomposition.

    Every field is read from durable artifacts -- the gate evidence, its
    archived check logs, the plan in force, the ladder ledger and Git objects
    -- and never from the current worktree content, so a resume rebuilds byte
    for byte the same request and the same anti-loop fingerprint.  The facts
    name the approved envelope without widening it: a plan that needs a path no
    earlier plan approved still asks the run's own scope policy.
    """

    tree = evidence.staged_tree_sha
    declared = {
        path for step in cycle_plan.plan.steps
        for path in (*step.write_set, *step.create_set, *step.delete_set)
    }
    in_question = sorted(set(approved_scope) | declared | set(evidence.changed_files))
    return CheckReplanFacts(
        cycle=cycle, stage=stage.value, candidate_tree_sha=tree,
        failed_check_ids=tuple(
            item.split(":", 1)[1] for item in soft_check_failures(evidence) if ":" in item
        ),
        plan_identity_before=plan_identity(cycle_plan.plan),
        approved_mutable_envelope=tuple(approved_scope),
        repository_reference=render_repository_reference(ctx.repository_reference),
        original_spec=ctx.spec,
        repository_state=_json_text({
            "BASE_SHA": ctx.base_sha,
            "CANDIDATE_TREE_SHA": tree,
            "CHANGED_FILES": changed_paths_between_trees(
                ctx.repo, ctx.base_tree_sha, tree,
            ),
        }),
        approved_plan_summary=render_repair_plan_summary(cycle_plan.plan),
        approved_step_index=render_repair_step_index(cycle_plan.plan),
        check_failure_proofs=bounded_check_proofs(check_failure_proofs(
            evidence=evidence,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=tree,
        )),
        candidate_diff_summary=bounded_diff_summary(evidence.diff),
        repository_path_facts=render_path_facts([
            {"path": path, "exists": path_exists_in_tree(ctx.repo, tree, path)}
            for path in in_question[:_MAX_REPLAN_PATH_FACTS]
        ]),
        consumed_strategies=consumed_ladder_strategies(
            ctx.run_dir, cycle_plan.cycle.number, stage,
        ),
    )


class ReviewService:
    """One owner of the pipeline operations described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def review_implementation_correction(
        self, ctx: PipelineV2Context, cycle: RunCycle, starting: bool,
    ) -> tuple[CyclePlan, ReviewResult]:
        """Load the previous candidate and reviewer report for direct correction."""

        if cycle.kind is not CycleKind.REVIEW_IMPLEMENTATION:
            raise PipelineFailure("IMPLEMENTATION_CORRECTION_CYCLE_REQUIRED")
        previous = cycle.number - 1
        candidate = read_candidate_record(ctx.run_dir, previous)
        evidence = candidate_evidence(ctx.run_dir, previous)
        review = _accepted_review(
            review_dir(ctx.run_dir, previous), evidence, candidate["commit_sha"]
        ) if evidence is not None else None
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route is not ReviewRoute.IMPLEMENTATION:
            raise ResumeIntegrityError(
                f"cycle {previous:03d} review did not route direct implementation correction"
            )
        # The correction starts exactly on the reviewed candidate.  Once its
        # green tree is accepted, HEAD is the single accepted child of it.
        head = current_head(ctx.info.worktree)
        if head != candidate["commit_sha"] and (
            starting or commit_parents(ctx.info.worktree, head) != (candidate["commit_sha"],)
        ):
            raise ResumeIntegrityError(
                f"cycle {cycle.number:03d} does not start from the reviewed candidate"
            )
        previous_plan = self.runtime.composition.cycle_plan(ctx, previous)
        return CyclePlan(
            cycle=cycle,
            plan=previous_plan.plan,
            bundle=previous_plan.bundle,
            contracts_dir=previous_plan.contracts_dir,
            step_profile_ids=previous_plan.step_profile_ids,
            step_fallback_profile_ids=previous_plan.step_fallback_profile_ids,
        ), review
    def plan_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
    ) -> CyclePlan:
        """Plan one review-driven correction cycle from the reviewed candidate."""

        if cycle.kind is CycleKind.CHECK_REPLAN:
            # The red gate's own planning transaction already produced this
            # decomposition; a cycle boundary only reloads and verifies it, so
            # no second planner call can ever happen for one replan.
            return self.runtime.composition.load_plan_correction(ctx, cycle)
        previous = cycle.number - 1
        repair_dir = correction_dir(ctx.run_dir, cycle)
        repair_dir.mkdir(parents=True, exist_ok=True)
        if cycle.kind is not CycleKind.REVIEW_REPLAN:
            raise PipelineFailure("REPLAN_CYCLE_REQUIRED")
        candidate = read_candidate_record(ctx.run_dir, previous)
        evidence = candidate_evidence(ctx.run_dir, previous)
        if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
            raise ResumeIntegrityError(f"cycle {previous:03d} candidate evidence is missing")
        review = _accepted_review(review_dir(ctx.run_dir, previous), evidence, candidate["commit_sha"])
        if review is None or review.verdict is not ReviewVerdict.REVISE or review.route is not ReviewRoute.REPLAN:
            raise ResumeIntegrityError(
                f"cycle {previous:03d} review did not route replan correction"
            )
        head = current_head(ctx.info.worktree)
        if head != candidate["commit_sha"]:
            raise ResumeIntegrityError(
                f"cycle {cycle.number:03d} does not start from the reviewed candidate"
            )
        tree_before = candidate_tree_sha(ctx.info.worktree)
        planner_profile = profile_for_role(
            self.runtime.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        approved_scope = self.runtime.composition.approved_scope_before(ctx, cycle.number)
        store.update_metadata(current_step=None)
        current_state = _json_text({
            "BASE_SHA": ctx.base_sha,
            "HEAD_SHA": head,
            "CANDIDATE_TREE_SHA": tree_before,
            "CHANGED_FILES": changed_paths_between_trees(ctx.repo, ctx.base_tree_sha, tree_before),
            "GIT_STATUS": status_porcelain(ctx.info.worktree),
        })
        planner = RepairPlannerV2(
            self.runtime.planner_client or chat_client(
                build_llm_endpoint(planner_profile), self.runtime.environment, self.runtime.observability.trace_transport
            ),
            planning=self.runtime.config.planning,
            check_catalog=self.runtime.config.check_catalog,
            original_required_check_ids=ctx.plan.required_checks,
            # The correction starts from the reviewed candidate, not the base.
            repository_preconditions=RepositoryPreconditions(ctx.repo, candidate["tree_sha"]),
        )
        # The reviewed candidate commit is the code authority: the planner
        # gets its immutable candidate/compare URLs instead of an inline diff.
        remote_available = (
            candidate.get("remote_sha") == head
            and candidate.get("remote_branch") == ctx.info.branch
            and candidate.get("remote") == self.runtime.config.repository.remote
            and immutable_commit_web_url(ctx.repository_reference, head) is not None
            and compare_commits_web_url(ctx.repository_reference, ctx.base_sha, head) is not None
        )
        started_at, started_mono = self.runtime.observability.trace_time(), time.perf_counter()
        planner_selected = self.runtime.observability.trace_selected_profile(
            ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        self.runtime.observability.trace_emit(
            "plan.started", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "tree_before": tree_before,
                "session": self.runtime.observability.trace_session(
                    profile=planner_profile, selected=planner_selected,
                    role=ExecutionRole.PLANNER, prompt_bytes=None,
                    started_at=started_at, started_mono=started_mono,
                    tree_before=tree_before,
                ),
            },
        )
        try:
            plan = planner.plan(
                repository_reference=render_repository_reference(ctx.repository_reference),
                original_spec=ctx.spec,
                original_plan_summary=render_repair_plan_summary(ctx.plan),
                original_step_index=render_repair_step_index(ctx.plan),
                current_repository_state=current_state,
                candidate_code_evidence=review_code_evidence(
                    repository_reference=ctx.repository_reference,
                    base_sha=ctx.base_sha, candidate_sha=head, evidence=evidence,
                    remote_sha=candidate.get("remote_sha"),
                    remote_branch=candidate.get("remote_branch"),
                    remote_name=candidate.get("remote"),
                ),
                previous_cycle_checks=_json_text(_repair_checks_payload(evidence)),
                previous_revision_report=_bounded_previous_revision_report(
                    review_cycle_revision_report(ctx.run_dir, previous, load_revision) or "NONE"
                ),
                original_approved_mutable_scope=_json_text(approved_scope),
                reviewer_result=_json_text(_review_payload(review)),
                artifacts_dir=repair_dir,
                fallback_candidate_diff="" if remote_available else evidence.diff,
            )
        except PlanRepositoryPreconditionError as exc:
            raise PipelineFailure(exc.code, bounded_parse_detail(exc)) from exc
        except V2PlanParseError as exc:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", bounded_parse_detail(exc)) from exc
        self.runtime.observability.trace_emit(
            "plan.completed", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "decision": plan.decision.value, "title": plan.title,
                "session": self.runtime.observability.trace_finished_model_session(
                    profile=planner_profile, selected=planner_selected,
                    role=ExecutionRole.PLANNER,
                    prompt_bytes=(
                        (repair_dir / "planner.request.txt").stat().st_size
                        if (repair_dir / "planner.request.txt").is_file() else None
                    ),
                    started_at=started_at, started_mono=started_mono,
                    usage=getattr(planner, "last_usage", None),
                    tree_before=tree_before, tree_after=tree_before,
                    final_message=getattr(plan, "raw", None),
                ),
            },
        )
        self.runtime.cycle_update(store, cycle, status="planning", plan_summary=plan.title)
        if plan.decision is PlanDecision.BLOCKED:
            self.runtime.cycle_update(store, cycle, status="blocked", blockers=plan.blockers)
            raise PipelineFailure("REPAIR_PLANNER_BLOCKED", plan.blockers)
        bundle, bundle_sha = validate_implementation_bundle(
            repair_dir, expected_step_ids=[step.id for step in plan.steps]
        )
        self._authorize_correction_scope(
            store, ctx, cycle, plan, bundle_sha, candidate["commit_sha"], review, approved_scope,
        )
        return self.runtime.composition.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=True,
        )
    def _authorize_correction_scope(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle: RunCycle,
        plan: TaskPlanV2, bundle_sha: str, candidate_sha: str, review: ReviewResult,
        approved_scope: list[str],
    ) -> None:
        """Bind the correction scope delta and apply the run's scope policy."""

        repair_dir = correction_dir(ctx.run_dir, cycle)
        try:
            delta, content = _build_scope_delta(
                repair_dir, original_scope=approved_scope, plan=plan,
                candidate_commit_sha=candidate_sha, review=review,
                repair_bundle_sha=bundle_sha,
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        self._apply_correction_scope(
            store, ctx, cycle, plan, repair_dir, delta, content, approved_scope, candidate_sha,
        )

    def _apply_correction_scope(
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
    def replan_cycle(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> CyclePlan:
        """Answer one spent red gate with a new decomposition of its cycle.

        The rung that reaches this method is the last one of the episode: the
        bounded repair pass and the single-step replan were durably spent and
        the gate is still red, so the failure proves the decomposition wrong
        rather than one of its steps.  The bounded facts and the one planning
        transaction are the check-replan ones; the answer becomes the plan of
        the cycle that executes it, written before that cycle exists so a crash
        resumes the very same answer instead of paying for a second one.  The
        planner may restructure the steps, correct their contracts and reuse
        every approved path -- it names no new authority, so a path no earlier
        plan approved still goes through the run's own scope policy.
        """

        if correction_cycles_used(cycle_plan.cycle.number) >= ctx.options.max_correction_cycles:
            # The single correction budget is spent: fail closed before the
            # planner transaction, so the ladder advances to its next rung.
            raise RecoveryStepUnavailable(
                step.strategy, "the run's correction budget is spent",
            )
        cycle = RunCycle(cycle_plan.cycle.number + 1, CycleKind.CHECK_REPLAN)
        directory = check_replan_dir(ctx.run_dir, cycle.number)
        facts = _check_replan_facts(
            ctx=ctx, cycle_plan=cycle_plan, cycle=cycle.number, stage=stage,
            evidence=evidence, approved_scope=self.runtime.composition.approved_scope_before(
                ctx, cycle.number,
            ),
        )
        profile = profile_for_role(
            self.runtime.config, ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        store.update_metadata(current_step=None)
        started_at, started_mono = self.runtime.observability.trace_time(), time.perf_counter()
        selected = self.runtime.observability.trace_selected_profile(
            ctx.selection.planner.profile_id, ExecutionRole.PLANNER
        )
        self.runtime.observability.trace_emit(
            "plan.started", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "tree_before": facts.candidate_tree_sha,
                "session": self.runtime.observability.trace_session(
                    profile=profile, selected=selected, role=ExecutionRole.PLANNER,
                    prompt_bytes=None, started_at=started_at, started_mono=started_mono,
                    tree_before=facts.candidate_tree_sha,
                ),
            },
        )
        try:
            plan = CheckReplanTransaction(
                client=self.runtime.planner_client or chat_client(
                    build_llm_endpoint(profile), self.runtime.environment,
                    self.runtime.observability.trace_transport,
                ),
                artifacts_dir=directory,
                planning=self.runtime.config.planning,
                check_catalog=self.runtime.config.check_catalog,
                original_required_check_ids=ctx.plan.required_checks,
                repository_preconditions=RepositoryPreconditions(
                    ctx.repo, facts.candidate_tree_sha,
                ),
            ).plan(facts)
        except PlanRepositoryPreconditionError as exc:
            raise PipelineFailure(exc.code, bounded_parse_detail(exc)) from exc
        except V2PlanParseError as exc:
            raise PipelineFailure("PLANNER_OUTPUT_INVALID", bounded_parse_detail(exc)) from exc
        self.runtime.observability.trace_emit(
            "plan.completed", phase="planning", cycle=cycle.number,
            data={
                "kind": cycle.kind.value, "decision": plan.decision.value, "title": plan.title,
                "session": self.runtime.observability.trace_finished_model_session(
                    profile=profile, selected=selected, role=ExecutionRole.PLANNER,
                    prompt_bytes=(
                        (directory / "planner.request.txt").stat().st_size
                        if (directory / "planner.request.txt").is_file() else None
                    ),
                    started_at=started_at, started_mono=started_mono,
                    usage=None, tree_before=facts.candidate_tree_sha,
                    tree_after=facts.candidate_tree_sha, final_message=getattr(plan, "raw", None),
                ),
            },
        )
        self.runtime.cycle_update(store, cycle, status="planning", plan_summary=plan.title)
        if plan.decision is PlanDecision.BLOCKED:
            self.runtime.cycle_update(store, cycle, status="blocked", blockers=plan.blockers)
            raise PipelineFailure("REPAIR_PLANNER_BLOCKED", plan.blockers)
        if plan_identity(plan) == facts.plan_identity_before:
            # The planner answered the very decomposition already in force:
            # executing it would replay approved work.  The rung is spent and
            # its durable record keeps these exact facts from being re-planned.
            self.runtime.cycle_update(
                store, cycle, status="failed", failure="CHECK_REPLAN_UNCHANGED",
            )
            raise RecoveryStepUnavailable(
                step.strategy, "the cycle replan re-decomposed nothing",
            )
        bundle, bundle_sha = validate_implementation_bundle(
            directory, expected_step_ids=[step.id for step in plan.steps]
        )
        try:
            delta, content = build_scope_delta(
                directory, original_scope=list(facts.approved_mutable_envelope),
                plan=plan, candidate_commit_sha=facts.candidate_tree_sha,
                repair_bundle_sha=bundle_sha,
                justification=f"{stage.value}: {' '.join(facts.failed_check_ids)}",
            )
        except OrchestrationError as exc:
            raise PipelineFailure(str(exc)) from exc
        self._apply_correction_scope(
            store, ctx, cycle, plan, directory, delta, content,
            list(facts.approved_mutable_envelope), facts.candidate_tree_sha,
        )
        return self.runtime.composition.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=True,
        )

    def semantic_revision(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
    ) -> None:
        """One semantic revision pass over the implemented cycle."""

        number = cycle_plan.cycle.number
        artifact_dir = semantic_revision_dir(ctx.run_dir, number)
        steps = self.runtime.composition.completed_steps(ctx, cycle_plan)
        mutable_scope = list(self.runtime.composition.effective_cycle_scope(ctx, cycle_plan))
        pre_stage = pre_semantic_gate_stage(cycle_plan.cycle.kind)
        pre_check_evidence = load_evidence(
            gate_dir(ctx.run_dir, cycle_plan.cycle, pre_stage)
        )
        if pre_check_evidence is None:
            raise PipelineFailure(
                "DURABLE_ARTIFACT_CORRUPTED",
                "pre-semantic deterministic gate evidence is missing",
            )
        while True:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            try:
                result, error = self.run_revision_with_recovery(
                    store=store, cycle=number, is_check_repair=False,
                    request={
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": cycle_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": artifact_dir,
                "mutable_scope": mutable_scope, "step_results": steps,
                "pre_check_evidence": pre_check_evidence,
                    },
                )
            except PipelineFailure:
                raise
            except (AgentScopeError, AgentError) as exc:
                self.runtime.observability.redact_revision_artifacts(artifact_dir)
                _record_failure_tree(artifact_dir, ctx.info.worktree)
                raise PipelineFailure(
                    getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self.runtime.secrets),
                ) from exc
            if error == SCOPE_REQUEST_ROUTE:
                outcome, mutable_scope = self.authorize_semantic_scope_request(
                    store, ctx, cycle_plan, artifact_dir, mutable_scope,
                )
                if outcome == "expanded":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    continue
                if outcome == "replan":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    return
                report = _read_json_artifact(artifact_dir / "report.json", 256 * 1024)
                final = report.get("final", "") if isinstance(report, dict) else ""
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                self.runtime.cycle_update(
                    store, cycle_plan.cycle, status="revised",
                    semantic_revision_status="SCOPE_REQUEST_RECORDED",
                    semantic_revision_report=_bounded_report(str(final)),
                )
                return
            if error is not None:
                if error in {
                    AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                    AGENT_PROTOCOL_FAILED,
                }:
                    unavailable = {"status": "UNAVAILABLE", "reason": error[:120]}
                    atomic_write_text(artifact_dir / "status.json", _json_text(unavailable))
                    self.runtime.cycle_update(
                        store, cycle_plan.cycle, status="revision_unavailable",
                        semantic_revision_status="UNAVAILABLE",
                        semantic_revision_reason=unavailable["reason"],
                        semantic_revision_report=(
                            "SEMANTIC REVISION: UNAVAILABLE\nreason=" + unavailable["reason"]
                        ),
                    )
                    store.update_metadata(semantic_revision=unavailable)
                    return
                if error in {"REVISION_SCOPE_VIOLATION", AGENT_SCOPE_VIOLATION}:
                    raise PipelineFailure(
                        AGENT_SCOPE_VIOLATION,
                        "semantic revision changed a path outside approved authority",
                    )
                raise PipelineFailure(error)
            self.runtime.cycle_update(
                store, cycle_plan.cycle, status="revised",
                semantic_revision_status="COMPLETED",
                semantic_revision_report=_bounded_report(result.final_message) if result else "",
            )
            return
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
    def semantic_review_correction(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        review: ReviewResult,
    ) -> None:
        """Apply one direct semantic correction requested by the reviewer.

        This route deliberately has no new plan or implementation step.  The
        reviewer report is evidence for the reviser, while the approved plan,
        candidate commit and cumulative mutable scope remain the authorities.
        """

        if review is None or review.route is not ReviewRoute.IMPLEMENTATION:
            raise ResumeIntegrityError("direct semantic correction has no implementation review")
        number = cycle_plan.cycle.number
        previous_plan = self.runtime.composition.cycle_plan(ctx, number - 1)
        candidate = read_candidate_record(ctx.run_dir, number - 1)
        evidence = candidate_evidence(ctx.run_dir, number - 1)
        if evidence is None or evidence.staged_tree_sha != candidate["tree_sha"]:
            raise ResumeIntegrityError(f"cycle {number - 1:03d} candidate evidence is missing")
        approved_scope = self.runtime.composition.approved_scope_before(ctx, number)
        code_evidence = review_code_evidence(
            repository_reference=ctx.repository_reference,
            base_sha=ctx.base_sha,
            candidate_sha=candidate["commit_sha"],
            evidence=evidence,
            remote_sha=candidate.get("remote_sha"),
            remote_branch=candidate.get("remote_branch"),
            remote_name=candidate.get("remote"),
        )
        candidate_identity = _json_text({
            "authority": "immutable_candidate_commit",
            "commit_sha": candidate["commit_sha"],
            "tree_sha": candidate["tree_sha"],
            "parent_sha": candidate["parent_sha"],
            "repository_reference": repository_reference_dict(ctx.repository_reference),
        })
        reviewer_evidence = "\n\n".join((
            "SUMMARY\n" + review.summary,
            "FINDINGS\n" + review.findings,
            "REQUIRED FIXES\n" + review.required_fixes,
            "MISSING TESTS\n" + review.missing_tests,
            "CORRECTION EVIDENCE\n" + code_evidence,
        ))
        artifact_dir = semantic_revision_dir(ctx.run_dir, number)
        mutable_scope = list(self.runtime.composition.effective_cycle_scope(ctx, cycle_plan))
        step_results = self.runtime.composition.completed_steps(ctx, previous_plan)
        while True:
            _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            try:
                result, error = self.run_revision_with_recovery(
                    store=store, cycle=number, is_check_repair=False,
                    request={
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": previous_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": artifact_dir,
                "mutable_scope": mutable_scope,
                "step_results": step_results,
                "reviewer_correction_evidence": reviewer_evidence,
                "candidate_identity": candidate_identity,
                "bounded_diff_evidence": bounded_semantic_diff(evidence.diff, 16 * 1024)[0],
                "pre_check_evidence": evidence,
                    },
                )
            except PipelineFailure:
                raise
            except AgentError as exc:
                self.runtime.observability.redact_revision_artifacts(artifact_dir)
                raise PipelineFailure(
                    getattr(exc, "code", AGENT_RUNTIME_FAILED), redact(str(exc), self.runtime.secrets),
                ) from exc
            if error == SCOPE_REQUEST_ROUTE:
                outcome, mutable_scope = self.authorize_semantic_scope_request(
                    store, ctx, cycle_plan, artifact_dir, mutable_scope,
                )
                if outcome == "expanded":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    continue
                if outcome == "replan":
                    _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                    return
                _archive_attempt(artifact_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
                self.runtime.cycle_update(
                    store, cycle_plan.cycle, status="revised",
                    semantic_revision_status="SCOPE_REQUEST_RECORDED",
                    semantic_revision_report="SEMANTIC REVISION: SCOPE REQUEST RECORDED",
                )
                return
            if error in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AGENT_PROTOCOL_FAILED,
            }:
                unavailable = {"status": "UNAVAILABLE", "reason": error[:120]}
                atomic_write_text(artifact_dir / "status.json", _json_text(unavailable))
                self.runtime.cycle_update(
                    store, cycle_plan.cycle, status="revision_unavailable",
                    semantic_revision_status="UNAVAILABLE",
                    semantic_revision_reason=error[:120],
                    semantic_revision_report=(
                        "SEMANTIC REVISION: UNAVAILABLE\nreason=" + error[:120]
                    ),
                )
                return
            if error is not None:
                if error in {"REVISION_SCOPE_VIOLATION", AGENT_SCOPE_VIOLATION}:
                    raise PipelineFailure(AGENT_SCOPE_VIOLATION, "semantic correction changed a path outside approved scope")
                raise PipelineFailure(error)
            self.runtime.cycle_update(
                store, cycle_plan.cycle, status="revised",
                semantic_revision_status="COMPLETED",
                semantic_revision_report=_bounded_report(result.final_message) if result else "",
            )
            return
    def _review_context_builder(self) -> ReviewContextBuilder:
        return ReviewContextBuilder(
            cycle_plan=self.runtime.composition.cycle_plan,
            completed_steps=self.runtime.composition.completed_steps,
            load_revision=load_revision,
            read_candidate=_review_candidate_record,
            candidate_evidence=candidate_evidence,
            accepted_review=_accepted_review,
        )
    def review_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        candidate: Mapping[str, Any], evidence: EvidenceBundle,
    ) -> ReviewResult:
        """Review the immutable candidate, recovering only reviewer-side failures."""

        remote_available = self._assert_candidate_review_authority(ctx, candidate, evidence)
        directory = review_dir(ctx.run_dir, cycle_plan.cycle)
        accepted = _accepted_review(directory, evidence, candidate["commit_sha"])
        if accepted is not None and accepted.verdict is not ReviewVerdict.FAIL:
            store.update_metadata(
                reviewed_candidate_sha=candidate["commit_sha"],
            )
            return accepted
        store.update_metadata(current_step=None)
        if accepted is None:
            _archive_attempt(directory, names=_REVIEW_ATTEMPT_ARTIFACTS)
            review = self._review_with_transport_retries(
                store=store, ctx=ctx, cycle_plan=cycle_plan, candidate=candidate,
                evidence=evidence, artifacts_dir=directory,
                force_inline_diff=not remote_available,
            )
        else:
            # A FAIL response is durable but never an accepted review outcome.
            # Re-enter its deterministic recovery after a crash at FINAL_REVIEW.
            review = accepted

        if review.verdict is ReviewVerdict.FAIL:
            review = self._recover_reviewer_fail(
                store=store, ctx=ctx, cycle_plan=cycle_plan, candidate=candidate,
                supplied_evidence=evidence, first_review=review, artifacts_dir=directory,
            )
        store.update_metadata(
            reviewed_candidate_sha=candidate["commit_sha"],
        )
        return review
    def _review_with_transport_retries(
        self,
        *,
        store: RunStateStore,
        ctx: PipelineV2Context,
        cycle_plan: CyclePlan,
        candidate: Mapping[str, Any],
        evidence: EvidenceBundle,
        artifacts_dir: Path,
        force_inline_diff: bool = False,
        review_input: ReviewCycleInput | None = None,
    ) -> ReviewResult:
        """Retry transient reviewer transport on this candidate only."""

        reviewer = self.runtime.composition.reviewer_for_profile(ctx.selection.final_reviewer.profile_id)
        return self.runtime.review_recovery(store).with_transport_retries(
            cycle=cycle_plan.cycle.number,
            candidate_tree=candidate.get("tree_sha"),
            run_review=lambda: self._run_v2_reviewer(
                reviewer=reviewer, spec=ctx.spec, run_dir=ctx.run_dir,
                repository_reference=ctx.repository_reference, evidence=evidence,
                input=review_input or self._review_context_builder().build(ctx, cycle_plan),
                artifacts_dir=artifacts_dir, worktree=ctx.info.worktree,
                base_sha=ctx.base_sha, candidate_commit=candidate,
                force_inline_diff=force_inline_diff,
            ),
            archive_attempt=lambda: _archive_attempt(
                artifacts_dir, names=_REVIEW_ATTEMPT_ARTIFACTS,
            ),
        )
    def _recover_reviewer_fail(
        self,
        *,
        store: RunStateStore,
        ctx: PipelineV2Context,
        cycle_plan: CyclePlan,
        candidate: Mapping[str, Any],
        supplied_evidence: EvidenceBundle,
        first_review: ReviewResult,
        artifacts_dir: Path,
    ) -> ReviewResult:
        """Rebuild local authority, then allow one reviewer evidence retry."""

        return self.runtime.review_recovery(store).after_reviewer_fail(
            cycle=cycle_plan.cycle.number,
            first_review=first_review,
            rebuild_evidence=lambda: self._assert_local_review_evidence(
                ctx, cycle_plan, candidate, supplied_evidence,
            ),
            rerun=lambda evidence, inline: self._review_with_transport_retries(
                store=store, ctx=ctx, cycle_plan=cycle_plan, candidate=candidate,
                evidence=evidence, artifacts_dir=artifacts_dir,
                force_inline_diff=inline,
                review_input=self._review_context_builder().build(ctx, cycle_plan),
            ),
            archive_attempt=lambda: _archive_attempt(
                artifacts_dir, names=_REVIEW_ATTEMPT_ARTIFACTS,
            ),
        )
    def _assert_local_review_evidence(
        self,
        ctx: PipelineV2Context,
        cycle_plan: CyclePlan,
        candidate: Mapping[str, Any],
        supplied_evidence: EvidenceBundle,
    ) -> EvidenceBundle:
        """Prove candidate and gate artifacts locally before retrying a FAIL."""

        def integrity(detail: str) -> NoReturn:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", detail)

        number = cycle_plan.cycle.number
        try:
            stored_candidate = read_candidate_record(ctx.run_dir, number)
            stage = stored_candidate.get("gate_stage")
            evidence_dir = gate_dir(ctx.run_dir, number, stage)
            evidence_path = evidence_dir / "evidence.json"
            evidence_bytes = evidence_path.read_bytes()
            evidence_payload = json.loads(evidence_bytes.decode("utf-8"))
            durable_evidence = candidate_evidence(ctx.run_dir, number)
            acceptance = _read_json_artifact(gate_acceptance_path(ctx.run_dir, number, stage))
            diff_artifact = (evidence_dir / "diff.patch").read_text(encoding="utf-8")
            changed_artifact = (evidence_dir / "changed-files.txt").read_text(encoding="utf-8")
            checks_artifact = json.loads((evidence_dir / "checks.json").read_text(encoding="utf-8"))
            current = current_head(ctx.info.worktree)
            current_tree = resolve_tree(ctx.info.worktree, stored_candidate["commit_sha"])
            parents = commit_parents(ctx.info.worktree, stored_candidate["commit_sha"])
            authority = gate_mutable_authority(
                ctx.run_dir, number, stage,
                base_paths=self.runtime.composition.effective_cycle_scope(ctx, cycle_plan),
                policy_config=self.runtime.repair_scope,
                require_attempt_records=True,
            )
        except (OSError, UnicodeError, ValueError, GitError, TypeError) as exc:
            integrity(f"candidate or gate evidence is unreadable: {type(exc).__name__}")
        if any(
            candidate.get(key) != stored_candidate.get(key)
            for key in (
                "commit_sha", "tree_sha", "parent_sha", "gate_stage",
            )
        ):
            integrity("candidate identity differs from its durable record")
        if (
            symbolic_head(ctx.info.worktree) != ctx.branch_ref
            or current != stored_candidate.get("commit_sha")
            or current_tree != stored_candidate.get("tree_sha")
            or (
                stored_candidate.get("no_change") is not True
                and parents != (stored_candidate.get("parent_sha"),)
            )
        ):
            integrity("local immutable candidate identity no longer matches")
        if (
            durable_evidence is None
            or not isinstance(evidence_payload, dict)
            or evidence_payload.get("staged_tree_sha") != stored_candidate.get("tree_sha")
            or evidence_payload.get("base_sha") != ctx.base_sha
            or not evidence_payload.get("deterministic_passed")
            or not durable_evidence.deterministic_passed
            or not required_checks_passed(durable_evidence)
            or durable_evidence.staged_tree_sha != stored_candidate.get("tree_sha")
            or durable_evidence.base_sha != supplied_evidence.base_sha
            or durable_evidence.staged_tree_sha != supplied_evidence.staged_tree_sha
            or durable_evidence.changed_files != supplied_evidence.changed_files
            or durable_evidence.diff != supplied_evidence.diff
            or stored_candidate.get("no_change", False) is not (not durable_evidence.changed_files)
            or (
                stored_candidate.get("no_change") is True
                and (
                    stored_candidate.get("parent_sha") is not None
                    or durable_evidence.diff != ""
                )
            )
            or _required_checks_summary(durable_evidence) != _required_checks_summary(supplied_evidence)
        ):
            integrity("durable gate evidence is missing, failed, or differs from the reviewed evidence")
        changed_expected = "".join(f"{path}\n" for path in durable_evidence.changed_files)
        if (
            diff_artifact != durable_evidence.diff
            or changed_artifact != changed_expected
            or checks_artifact != evidence_payload.get("checks")
            or evidence_payload.get("changed_files") != list(durable_evidence.changed_files)
            or evidence_payload.get("required_check_ids") != list(durable_evidence.required_check_ids)
        ):
            integrity("duplicate durable gate evidence artifacts disagree")
        expected_stage = getattr(stage, "value", stage)
        if (
            not isinstance(acceptance, dict)
            or acceptance.get("review_cycle") != number
            or acceptance.get("stage") != expected_stage
            or acceptance.get("tree_sha") != stored_candidate.get("tree_sha")
            or acceptance.get("commit_sha") != stored_candidate.get("commit_sha")
            or acceptance.get("parent_sha") != stored_candidate.get("parent_sha")
            or acceptance.get("schema_version") not in {1, 2}
            or acceptance.get("acceptance_kind") not in {
                "existing-head", "repair", "semantic-revision",
            }
            or not isinstance(acceptance.get("commit_created"), bool)
            or acceptance.get("mutable_scope") != list(authority.effective_paths)
            or acceptance.get("mutable_scope_sha256") != authority.sha256
            or acceptance.get("no_change", False) is not (not durable_evidence.changed_files)
            or (
                acceptance.get("schema_version") == 2
                and acceptance.get("evidence_sha256") != hashlib.sha256(evidence_bytes).hexdigest()
            )
        ):
            integrity("gate acceptance does not bind the durable evidence and candidate")
        return durable_evidence
    def _assert_candidate_review_authority(
        self,
        ctx: PipelineV2Context,
        candidate: Mapping[str, Any],
        evidence: EvidenceBundle,
    ) -> bool:
        """Validate local candidate authority and report remote exploration availability."""

        candidate_sha = candidate.get("commit_sha")
        candidate_tree = candidate.get("tree_sha")
        if not _is_object_id(candidate_sha) or not _is_object_id(candidate_tree):
            raise PipelineFailure(
                "CANDIDATE_PUSH_FAILED", "candidate identity is incomplete before review"
            )
        if (
            not isinstance(candidate.get("no_change", False), bool)
            or candidate.get("no_change", False) is not (len(evidence.changed_files) == 0)
        ):
            raise PipelineFailure(
                "DURABLE_ARTIFACT_CORRUPTED",
                "candidate no-change marker does not match immutable gate evidence",
            )
        try:
            if symbolic_head(ctx.info.worktree) != ctx.branch_ref:
                raise GitError("candidate worktree is not on the run branch")
            if current_head(ctx.info.worktree) != candidate_sha:
                raise GitError("local HEAD does not equal the candidate commit")
            if resolve_tree(ctx.info.worktree, candidate_sha) != candidate_tree:
                raise GitError("candidate tree does not match the candidate commit")
            if evidence.staged_tree_sha != candidate_tree:
                raise GitError("candidate tree does not match accepted gate evidence")
        except (GitError, OSError, ValueError) as exc:
            raise PipelineFailure(
                "CANDIDATE_PUSH_FAILED",
                f"local candidate identity is invalid: {exc}",
            ) from exc
        if candidate.get("no_change") is True:
            return False
        if (
            candidate.get("remote") != self.runtime.config.repository.remote
            or candidate.get("remote_branch") != ctx.info.branch
            or candidate.get("remote_sha") != candidate_sha
            or candidate.get("remote_status") != "available"
            or not isinstance(candidate.get("pushed_at"), str)
            or not candidate.get("pushed_at")
        ):
            return False
        try:
            return remote_run_branch_tip(
                ctx.info.source_repo,
                remote=self.runtime.config.repository.remote,
                branch=ctx.info.branch,
            ) == candidate_sha
        except (GitError, OSError, ValueError):
            return False
    def record_review(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, evidence: EvidenceBundle,
    ) -> None:
        store.update_metadata(
            review=_review_payload(review),
            review_iterations=number,
        )
        self.runtime.cycle_update(
            store, number, status="reviewed",
            checks=_check_payload(evidence), reviewer_conclusion=_review_payload(review),
        )
        self.runtime.observability.update_v2_usage(store, ctx.run_dir)
    def request_human(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, reason: str,
    ) -> RunResult:
        """End the run with the reviewer's request as an operator task."""

        reason_class = structured_review_reason(review.findings)
        authorized_classes = {
            "PRODUCT_SPEC_AMBIGUITY", "SECURITY_POLICY_DECISION", "AUTHORITY_CONFLICT",
        }
        if reason_class == "SCOPE_EXPANSION_REQUIRE_APPROVAL":
            if self.runtime.repair_scope.policy != "require-approval":
                raise PipelineFailure(
                    "REVIEW_FORMAT_INVALID",
                    "HUMAN scope approval was requested outside the configured require-approval policy",
                )
        elif reason_class not in authorized_classes:
            raise PipelineFailure(
                "REVIEW_FORMAT_INVALID", "HUMAN route lacks an authorized structured reason",
            )

        write_repair_task(ctx.run_dir, fields={
            "route": review.route.value,
            "review_summary": review.summary,
            "findings": review.findings,
            "required_fixes": review.required_fixes,
            "missing_tests": review.missing_tests,
            "existing_branch": ctx.info.branch,
            "existing_worktree": str(ctx.info.worktree),
            "run_id": ctx.run_id,
        })
        return self.runtime.failure.v2_failed(store, ctx.run_dir, reason, None)
    def review_repair_exhausted(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        review: ReviewResult, detail: Mapping[str, Any],
    ) -> RunResult:
        """Keep correction exhaustion visible and resumable at FINAL_REVIEW."""

        unchanged: bool | None = None
        if number > 1:
            cycles = store.load().get("cycles")
            previous = next((
                item for item in cycles
                if isinstance(item, dict) and item.get("number") == number - 1
            ), None) if isinstance(cycles, list) else None
            previous_review = (
                previous.get("reviewer_conclusion")
                if isinstance(previous, dict) else None
            )
            previous_findings = (
                previous_review.get("findings")
                if isinstance(previous_review, dict) else None
            )
            if isinstance(previous_findings, str):
                unchanged = " ".join(previous_findings.split()).casefold() == (
                    " ".join(review.findings.split()).casefold()
                )

        return self.runtime.failure.v2_failed(
            store, ctx.run_dir, "WAITING_REPAIR_EXHAUSTED", None,
            {
                **detail,
                "review_summary": review.summary,
                "findings": bounded_v2_report(review.findings),
                "same_findings_as_previous_cycle": unchanged,
            },
        )
    def _run_v2_reviewer(
        self,
        *,
        reviewer: Reviewer,
        spec: str,
        run_dir: Path,
        repository_reference: RepositoryReference,
        evidence: EvidenceBundle,
        input: ReviewCycleInput,
        artifacts_dir: Path,
        worktree: Path,
        base_sha: str,
        candidate_commit: Mapping[str, Any],
        force_inline_diff: bool = False,
    ) -> ReviewResult:
        """The single reviewer evidence assembly of every cycle.

        The required-check summary, the parse argument and the later commit
        gate all use the same actual ``evidence.deterministic_passed``; the
        call is always a fresh conversation.
        """

        candidate_sha = candidate_commit.get("commit_sha")
        if not _is_object_id(candidate_sha):
            raise OrchestrationError(
                "candidate commit SHA is missing before reviewer"
            )

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        code_evidence = review_code_evidence(
            repository_reference=repository_reference,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            evidence=evidence,
            remote_sha=candidate_commit.get("remote_sha"),
            remote_branch=candidate_commit.get("remote_branch"),
            remote_name=candidate_commit.get("remote"),
            force_inline_diff=force_inline_diff,
        )
        try:
            code_evidence_payload = json.loads(code_evidence)
        except (TypeError, ValueError):
            code_evidence_payload = {}
        candidate_identity = _json_text({
            key: candidate_commit.get(key)
            for key in (
                "commit_sha", "tree_sha", "parent_sha", "remote_branch",
                "remote_sha", "candidate_url", "compare_url",
                "immutable_commit_url", "pushed_at",
            )
            if candidate_commit.get(key) is not None
            or key == "immutable_commit_url"
        })
        if candidate_commit.get("no_change") is True:
            candidate_identity = _json_text({
                "candidate": json.loads(candidate_identity),
                "candidate_delta": "candidate delta is empty",
                "no_change_review": "independent reviewer must explicitly confirm SPEC_ALREADY_SATISFIED in SUMMARY",
            })
        if isinstance(code_evidence_payload, Mapping):
            candidate_identity = _json_text({
                "candidate": json.loads(candidate_identity),
                "candidate_url": code_evidence_payload.get("candidate_url"),
                "compare_url": code_evidence_payload.get("compare_url"),
                "candidate_tree_sha": code_evidence_payload.get("candidate_tree_sha"),
                "remote_exploration": code_evidence_payload.get("remote_exploration"),
                **({"review_evidence_mode": "LOCAL_INLINE_ONLY"} if force_inline_diff else {}),
            })
        diff_bytes = evidence.diff.encode("utf-8", errors="replace")
        diff_excerpt = bounded_semantic_diff(evidence.diff, 16 * 1024)[0]
        prompt_payload = build_final_review_payload(
            spec=spec,
            compact_approved_plan=input.plan_text,
            required_checks_summary=_required_checks_summary(evidence),
            immutable_candidate_identity=candidate_identity,
            changed_files="\n".join(evidence.changed_files),
            diff_sha256=hashlib.sha256(diff_bytes).hexdigest(),
            diffstat=_diffstat(evidence.diff, evidence.changed_files),
            bounded_diff_excerpt=diff_excerpt,
            cycle_summary=(
                _compact_cycle_summary(input.cycle_history)
                + (
                    "\nNO-CHANGE REVIEW\ncandidate delta is empty; assess whether the original SPEC is already satisfied.\n"
                    if candidate_commit.get("no_change") is True else ""
                )
                + "\n"
                + "REVISION REPORTS\n"
                + (input.revision_report or "NONE")
                + "\n"
                + input.step_reports
                + "\nDEFERRED VERIFY DEPENDENCIES\n"
                + input.deferred_verifications
            ),
            repository_reference=_json_text(repository_reference_dict(repository_reference)),
            budget_bytes=self.runtime.config.prompt_budget.final_review_max_bytes,
        )
        reviewer_profile_id = getattr(
            getattr(self.runtime.last_selection, "final_reviewer", None),
            "profile_id", None,
        )
        reviewer_profile = None
        reviewer_selected = None
        if reviewer_profile_id is not None:
            reviewer_selected = self.runtime.observability.trace_selected_profile(
                reviewer_profile_id, ExecutionRole.REVIEWER
            )
            try:
                reviewer_profile = profile_for_role(
                    self.runtime.config, reviewer_profile_id, ExecutionRole.REVIEWER
                )
            except ProfileError:
                reviewer_profile = None
        review_started_at = self.runtime.observability.trace_time()
        review_started_mono = time.perf_counter()
        self.runtime.observability.trace_emit(
            "review.started",
            phase="review",
            cycle=self.runtime.trace_cycle,
            data={
                "candidate_sha": candidate_sha,
                "tree_sha": evidence.staged_tree_sha,
                "session": self.runtime.observability.trace_session(
                    profile=reviewer_profile,
                    selected=reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=len(prompt_payload.rendered.encode("utf-8", errors="replace")),
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    tree_before=evidence.staged_tree_sha,
                ),
            },
        )
        review = reviewer.review(
            prompt_payload,
            deterministic_passed=evidence.deterministic_passed,
            artifacts_dir=artifacts_dir,
            require_no_change_confirmation=candidate_commit.get("no_change") is True,
        )
        reviewer_usage = getattr(reviewer, "last_usage", None)
        if reviewer_usage is None:
            reviewer_usage = read_usage_artifact(
                artifacts_dir / "reviewer.usage.json"
            )
        self.runtime.observability.trace_emit(
            "review.completed",
            phase="review",
            cycle=self.runtime.trace_cycle,
            data={
                "candidate_sha": candidate_sha,
                "tree_sha": evidence.staged_tree_sha,
                "verdict": review.verdict.value,
                "route": review.route.value,
                "session": self.runtime.observability.trace_finished_model_session(
                    profile=reviewer_profile,
                    selected=reviewer_selected,
                    role=ExecutionRole.REVIEWER,
                    prompt_bytes=len(prompt_payload.rendered.encode("utf-8", errors="replace")),
                    started_at=review_started_at,
                    started_mono=review_started_mono,
                    usage=reviewer_usage,
                    tree_before=evidence.staged_tree_sha,
                    tree_after=evidence.staged_tree_sha,
                    final_message=review.raw,
                ),
            },
        )
        # planner_thread != reviewer_thread: a driver that reports the
        # planner's own conversation for a review breaks independence.
        planner_thread = _read_planner_conversation(run_dir)
        reviewer_thread = getattr(reviewer, "last_conversation", None)
        if planner_thread is not None and reviewer_thread == planner_thread:
            raise ReviewParseError("reviewer reused the planner conversation")
        return review
    def _revision_runner(self) -> RevisionRunner:
        """Build the revision runner with this run's live dependencies."""

        return RevisionRunner(
            config=self.runtime.config,
            secrets=self.runtime.secrets,
            effective_repair_scope=self.runtime.repair_scope,
            approved_check_authority_sha256=self.runtime.approved_check_authority_sha256,
            run_revision=self.runtime.composition.run_revision,
            ensure_revision_artifacts=self.runtime.observability.ensure_revision_artifacts,
            redact_revision_artifacts=self.runtime.observability.redact_revision_artifacts,
            reusable_pre_checks=_reusable_pre_checks,
            hard_integrity_failures=hard_integrity_failures,
            soft_check_failures=soft_check_failures,
            check_repair_prompt=_check_repair_prompt,
        )
    def _run_v2_revision_cycle(
        self, *, cycle: int, check_repair_attempt: int | None = None, **request: Any,
    ) -> tuple[Any | None, str | None]:
        """Run one revision or check-repair pass -- see ``RevisionRunner.run``."""

        is_check_repair = request.get("check_repair_evidence") is not None
        selection = request["selection"]
        selected = selection.check_repair if is_check_repair else selection.semantic_reviser
        role = ExecutionRole.REPAIR if is_check_repair else ExecutionRole.REVISER
        profile = None
        if selected is not None:
            try:
                profile = profile_for_role(self.runtime.config, selected.profile_id, role)
            except ProfileError:
                profile = None
        tree_before = _safe_candidate_tree(request["info"].worktree)
        started_at = self.runtime.observability.trace_time()
        started_mono = time.perf_counter()
        phase = "repair" if is_check_repair else "revision"
        prefix = "check_repair" if is_check_repair else "revision"

        def session(**extra: Any) -> dict[str, Any]:
            return self.runtime.observability.trace_session(
                profile=profile, selected=selected, role=role,
                started_at=started_at, started_mono=started_mono,
                tree_before=tree_before, **extra,
            )

        self.runtime.observability.trace_emit(
            f"{prefix}.started", phase=phase, cycle=cycle,
            data={
                "tree_before": tree_before,
                "attempt": check_repair_attempt,
                "session": session(prompt_bytes=None),
            },
        )
        try:
            result, error = self._revision_runner().run(**request)
        except Exception as exc:
            self.runtime.observability.trace_emit(
                f"{prefix}.agent.completed", phase=phase, cycle=cycle,
                data={
                    "status": "failed",
                    "error": type(exc).__name__,
                    "session": session(prompt_bytes=None, exit_reason=type(exc).__name__),
                },
            )
            raise
        artifact_path = Path(request["artifact_dir"])
        prompt_path = artifact_path / "agent.prompt.txt"
        self.runtime.observability.trace_emit(
            f"{prefix}.agent.completed", phase=phase, cycle=cycle,
            data={
                "status": "completed" if error is None else "failed",
                "error": error,
                "session": session(
                    prompt_bytes=prompt_path.stat().st_size if prompt_path.is_file() else None,
                    result=result, exit_reason=error,
                ),
            },
        )
        pre_checks = _read_json_artifact(artifact_path / "pre_checks.json")
        if not is_check_repair and isinstance(pre_checks, dict):
            self.runtime.observability.trace_emit(
                "revision.checks.completed", phase=phase, cycle=cycle,
                data={
                    "passed": bool(pre_checks.get("deterministic_passed", False)),
                    "failures": [
                        item for item in pre_checks.get("failures", [])
                        if isinstance(item, str)
                    ] if isinstance(pre_checks.get("failures"), list) else [],
                    "tree_sha": pre_checks.get("staged_tree_sha"),
                },
            )
        return result, error
    def run_revision_with_recovery(
        self,
        *,
        store: RunStateStore,
        request: Mapping[str, Any],
        is_check_repair: bool,
        cycle: int,
        attempt: int | None = None,
    ) -> tuple[Any | None, str | None]:
        """Retry one reviser/repair contract after proving an exact rollback."""

        return self.runtime.worker_recovery(store).run_revision(
            request=request, is_check_repair=is_check_repair, cycle=cycle,
            attempt=attempt,
            fallbacks_limit=self.runtime.run_options.recovery.max_executor_fallbacks,
            run_attempt=self._run_v2_revision_cycle,
        )
