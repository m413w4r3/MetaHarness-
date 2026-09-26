"""The correction routes a review drives, and the reviewer's code evidence.

``ReviewCorrectionService`` owns the two review-driven correction cycles -- a
direct implementation correction and a re-decomposition -- and the direct
semantic correction of a reviewed candidate.  It also renders the immutable
candidate evidence that the reviewer, a correction planner and a corrector all
read, so one candidate is described by one implementation.
"""

from __future__ import annotations

import hashlib, json, time
from typing import (
    Any,
    Mapping,
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
)
from ..evidence import (
    EvidenceBundle,
    bounded_semantic_diff,
)
from ..gitops import (
    RepositoryReference,
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_parents,
    compare_commits_web_url,
    current_head,
    immutable_commit_web_url,
    render_repository_reference,
    repository_reference_dict,
    status_porcelain,
)
from ..models import (
    CycleKind,
    ExecutionRole,
    PlanDecision,
    ReviewRoute,
    ReviewVerdict,
    RunCycle,
)
from ..plan_repository_validation import (
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
)
from ..planning.artifacts import validate_implementation_bundle
from ..planning.planner import RepairPlannerV2
from ..planning.protocol import (
    V2PlanParseError,
    render_repair_plan_summary,
    render_repair_step_index,
)
from ..profiles import (
    build_llm_endpoint,
    profile_for_role,
)
from ..redaction import redact
from ..result import atomic_write_text
from ..resume import ResumeIntegrityError
from ..review import ReviewResult
from ..state import RunStateStore
from ..validation import check_result_json
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    correction_dir,
    review_dir,
    semantic_revision_dir,
)
from .durable_readers import (
    accepted_review,
    candidate_evidence,
    load_revision,
    read_candidate_record,
)
from .revision import (
    SCOPE_REQUEST_ROUTE,
    _bounded_previous_revision_report,
    _review_payload,
    review_cycle_revision_report,
)
from .shared import (
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _bounded_report,
    _git_ownership,
    _json_text,
    _repair_checks_payload,
    bounded_parse_detail,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


# The largest reviewer-facing diff that is inlined when no immutable commit
# URL can be offered: beyond it, only the bounded excerpt travels.
_MAX_REVIEW_FALLBACK_DIFF_BYTES = 32 * 1024


class ReviewCorrectionService:
    """The review-driven corrections of one run."""

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
        review = accepted_review(
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
        review = accepted_review(review_dir(ctx.run_dir, previous), evidence, candidate["commit_sha"])
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
            self.runtime.planner_client
            or self.runtime.chat(build_llm_endpoint(planner_profile)),
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
        self.runtime.correction_scope.authorize_review_correction(
            store, ctx, cycle, plan, bundle_sha, candidate["commit_sha"], review, approved_scope,
        )
        return self.runtime.composition.correction_cycle_plan(
            ctx, cycle, plan, bundle, bundle_sha, creating=True,
        )

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
                result, error = self.runtime.semantic_revision.run_revision_with_recovery(
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
                outcome, mutable_scope = self.runtime.correction_scope.authorize_semantic_scope_request(
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
        "diffstat": json.loads(diffstat(evidence.diff, evidence.changed_files)),
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



def required_checks_summary(evidence: EvidenceBundle) -> str:
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


def diffstat(diff: str, changed_files: Sequence[str]) -> str:
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


def compact_cycle_summary(text: str) -> str:
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
