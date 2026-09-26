"""The review decision of one immutable candidate.

This module owns the whole reviewer decision: it builds the reviewer context
from durable artifacts, invokes the reviewer and recovers reviewer-side
failures, records the accepted verdict, routes PASS/REVISE/FAIL, ends the run
on the reviewer's human route and keeps the review-repair budget visible.  It
never runs a deterministic check, never plans a correction and never publishes.
"""

from __future__ import annotations

import hashlib, json, time
from pathlib import Path
from typing import (
    Any,
    Mapping,
    NoReturn,
    TYPE_CHECKING,
)
from ..evidence import (
    EvidenceBundle,
    bounded_semantic_diff,
    required_checks_passed,
)
from ..gitops import (
    GitError,
    commit_parents,
    current_head,
    remote_run_branch_tip,
    resolve_tree,
    RepositoryReference,
    repository_reference_dict,
    symbolic_head,
)
from ..models import (
    CycleKind,
    ExecutionRole,
    ReviewVerdict,
)
from ..profiles import (
    ProfileError,
    profile_for_role,
)
from ..prompt_contracts import build_final_review_payload
from ..result import (
    RunResult,
    write_repair_task,
)
from ..review import (
    Reviewer,
    ReviewParseError,
    ReviewResult,
    structured_review_reason,
)
from ..state import RunStateStore
from ..usage import read_usage_artifact
from .check_scope import gate_mutable_authority
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    gate_acceptance_path,
    gate_dir,
    review_dir,
)
from .cycle_loader import read_cycle_record
from .durable_readers import (
    accepted_review,
    candidate_evidence,
    load_revision,
    read_candidate_record,
    read_planner_conversation,
)
from .review_correction import (
    compact_cycle_summary,
    diffstat,
    required_checks_summary,
    review_code_evidence,
)
from .revision import (
    ReviewContextBuilder,
    ReviewCycleInput,
    _review_payload,
)
from .shared import (
    OrchestrationError,
    _REVIEW_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _check_payload,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    bounded_v2_report,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime



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


class CandidateReviewService:
    """The review decision of one run, bound to its live dependencies."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def _review_context_builder(self) -> ReviewContextBuilder:
        return ReviewContextBuilder(
            cycle_plan=self.runtime.composition.cycle_plan,
            completed_steps=self.runtime.composition.completed_steps,
            load_revision=load_revision,
            read_candidate=_review_candidate_record,
            candidate_evidence=candidate_evidence,
            accepted_review=accepted_review,
        )

    def review_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        candidate: Mapping[str, Any], evidence: EvidenceBundle,
    ) -> ReviewResult:
        """Review the immutable candidate, recovering only reviewer-side failures."""

        remote_available = self._assert_candidate_review_authority(ctx, candidate, evidence)
        directory = review_dir(ctx.run_dir, cycle_plan.cycle)
        accepted = accepted_review(directory, evidence, candidate["commit_sha"])
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
            or required_checks_summary(durable_evidence) != required_checks_summary(supplied_evidence)
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
            required_checks_summary=required_checks_summary(evidence),
            immutable_candidate_identity=candidate_identity,
            changed_files="\n".join(evidence.changed_files),
            diff_sha256=hashlib.sha256(diff_bytes).hexdigest(),
            diffstat=diffstat(evidence.diff, evidence.changed_files),
            bounded_diff_excerpt=diff_excerpt,
            cycle_summary=(
                compact_cycle_summary(input.cycle_history)
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
        planner_thread = read_planner_conversation(run_dir)
        reviewer_thread = getattr(reviewer, "last_conversation", None)
        if planner_thread is not None and reviewer_thread == planner_thread:
            raise ReviewParseError("reviewer reused the planner conversation")
        return review
