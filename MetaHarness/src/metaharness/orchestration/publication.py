"""Candidate publication: immutable candidate identity and remote facts.

This module owns the candidate boundary and the publication of the accepted
candidate: it authorizes the candidate tree against its evidence before
anything else observes it, stages the run branch best-effort, creates the
optional GitHub workstream metadata and publishes the base branch.  It never
plans, reviews or runs a check.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Mapping,
    TYPE_CHECKING,
)
from ..evidence import (
    EvidenceBundle,
    required_checks_passed,
)
from ..gitops import (
    BaseMovedError,
    BasePushError,
    GitError,
    WorktreeInfo,
    commit_parents,
    publish_fast_forward_base,
    candidate_tree_sha,
    current_head,
    delete_run_branch,
    index_tree_sha,
    push_run_branch,
    remote_run_branch_tip,
    resolve_tree,
    RepositoryReference,
    status_porcelain,
    symbolic_head,
    repository_remote_url,
    validate_run_branch,
)
from ..commit_gate import (
    CommitSafetyError,
    assert_deferred_verifications_resolved,
)
from ..integrations.github import GitHubWorkstreamError
from ..models import (
    PublishMode,
    ReviewRoute,
    ReviewVerdict,
    RunDisposition,
    RunMachineState,
)
from ..resume import (
    ResumePhase,
    mark_checkpoint_completed,
)
from ..result import (
    RunResult,
    atomic_write_text,
)
from ..recovery_policy import classify_failure
from ..state import RunStateStore
from .shared import (
    CommitBoundaryError,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    _status_has_unstaged_or_untracked,
)
from .candidate import (
    CandidateRemoteStaging,
    accepted_chain_records,
    validate_accepted_chain,
    _candidate_commit_path,
    _commit_web_url,
)
from .pipeline_v2 import (
    PipelineFailure,
    PipelineV2Context,
    review_dir,
)
from .recovery import project_exit
from .durable_readers import (
    accepted_review,
    candidate_evidence,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime




class PublicationService:
    """One owner of the pipeline operations described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    @staticmethod
    def _github_result_number(value: Any, kind: str) -> int:
        """Extract only the public numeric identifier from an integration result."""

        number = value if isinstance(value, int) and not isinstance(value, bool) else None
        if number is None and isinstance(value, Mapping):
            candidate = value.get("number")
            number = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
        if number is None:
            candidate = getattr(value, "number", None)
            number = candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else None
        if number is None or number <= 0:
            raise GitHubWorkstreamError(f"GitHub {kind} response did not contain a valid number")
        return number
    @staticmethod
    def _github_issue_title(plan_title: str, run_id: str) -> str:
        first_line = next((line.strip() for line in plan_title.splitlines() if line.strip()), "MetaHarness run")
        title = f"MetaHarness: {first_line} ({run_id})"
        return title[:240]
    @staticmethod
    def _github_metadata_payload(state: Mapping[str, Any]) -> dict[str, int | str]:
        payload: dict[str, int | str] = {}
        for key in (
            "remote_branch", "issue_number", "pull_request_number",
            "reviewed_candidate_sha",
        ):
            value = state.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                payload[key] = value
        return payload
    def ensure_github_issue_metadata(
        self,
        *,
        store: RunStateStore,
        run_id: str,
        plan_title: str,
        info: WorktreeInfo,
    ) -> None:
        """Resolve optional issue metadata without reading issue content into authority."""

        github = self.runtime.config.github
        if not github.enabled or github.issue_mode == "off":
            return
        state = store.load()
        current = state.get("issue_number")
        if isinstance(current, int) and not isinstance(current, bool) and current > 0:
            if github.issue_mode == "link-existing" and github.issue_number not in {None, current}:
                raise GitHubWorkstreamError(
                    "configured GitHub issue does not match the persisted workstream issue",
                    code="GITHUB_CONFIG_INVALID",
                )
            return
        if github.issue_mode == "link-existing":
            requested = github.issue_number
            if requested is None:
                raise GitHubWorkstreamError(
                    "github.issue_number is required for link-existing",
                    code="GITHUB_CONFIG_INVALID",
                )
            try:
                issue = self.runtime.github_client.read_issue(requested)
            except Exception:
                raise GitHubWorkstreamError("GitHub issue operation failed") from None
            if issue is None:
                raise GitHubWorkstreamError(
                    "requested GitHub issue was not found",
                    code="GITHUB_ISSUE_NOT_FOUND",
                )
            number = self._github_result_number(issue, "issue")
            if number != requested:
                raise GitHubWorkstreamError(
                    "GitHub issue response did not match the requested issue",
                    code="GITHUB_ISSUE_NOT_FOUND",
                )
            event = "workstream.issue.linked"
        else:
            title = self._github_issue_title(plan_title, run_id)
            body = (
                "MetaHarness workstream metadata.\n\n"
                f"Run ID: {run_id}\n"
                f"Branch: {info.branch}\n"
                f"Base ref: {self.runtime.config.base_ref}\n"
            )
            try:
                issue = self.runtime.github_client.create_issue(title, body)
            except Exception:
                raise GitHubWorkstreamError("GitHub issue operation failed") from None
            number = self._github_result_number(issue, "issue")
            event = "workstream.issue.created"

        store.update_metadata(issue_number=number)
        state = store.load()
        self.runtime.observability.trace_emit(
            event,
            phase="setup",
            cycle=1,
            data={"issue_number": number},
            once=True,
        )
        self.runtime.observability.trace_emit(
            "workstream.metadata",
            phase="setup",
            cycle=1,
            data=self._github_metadata_payload(state),
        )
    def _ensure_github_pull_request_metadata(
        self,
        *,
        store: RunStateStore,
        run_id: str,
        info: WorktreeInfo,
        commit_sha: str,
        cycle: int | None,
    ) -> None:
        """Create the requested PR from the exact reviewed run branch."""

        github = self.runtime.config.github
        if not github.enabled or github.pull_request_mode == "off":
            return
        state = store.load()
        current = state.get("pull_request_number")
        if isinstance(current, int) and not isinstance(current, bool) and current > 0:
            return
        if (
            not self.runtime.config.publish.enabled
            or self.runtime.config.publish.mode != PublishMode.RUN_BRANCH.value
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request creation requires a published run branch",
                code="GITHUB_PR_REQUIRES_RUN_BRANCH",
            )
        review = state.get("review")
        reviewed_candidate_sha = state.get("reviewed_candidate_sha")
        accepted_candidate_sha = state.get("candidate_commit_sha")
        if (
            not isinstance(review, Mapping)
            or review.get("verdict") != ReviewVerdict.PASS.value
            or review.get("route") not in {None, ReviewRoute.NONE.value}
            or not _is_object_id(reviewed_candidate_sha)
            or not _is_object_id(accepted_candidate_sha)
            or reviewed_candidate_sha != accepted_candidate_sha
            or reviewed_candidate_sha != commit_sha
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request candidate authority is not an exact PASS candidate",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            )
        try:
            if current_head(info.worktree) != accepted_candidate_sha:
                raise GitHubWorkstreamError(
                    "local accepted candidate does not match the reviewed candidate",
                    code="GITHUB_PR_CANDIDATE_MISMATCH",
                )
            remote_tip = remote_run_branch_tip(
                info.source_repo,
                remote=getattr(
                    getattr(self.runtime.config, "repository", None),
                    "remote",
                    self.runtime.config.publish.remote,
                ),
                branch=info.branch,
            )
        except GitHubWorkstreamError:
            raise
        except (GitError, OSError, ValueError):
            raise GitHubWorkstreamError(
                "remote run branch tip could not be verified",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            ) from None
        if remote_tip != reviewed_candidate_sha:
            raise GitHubWorkstreamError(
                "remote run branch tip does not match the reviewed candidate",
                code="GITHUB_PR_CANDIDATE_MISMATCH",
            )
        plan_state = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
        plan_title = plan_state.get("title") if isinstance(plan_state.get("title"), str) else "MetaHarness run"
        title = self._github_issue_title(plan_title, run_id)
        body = (
            "MetaHarness reviewed workstream metadata.\n\n"
            f"Run ID: {run_id}\n"
            f"Reviewed commit: {commit_sha}\n"
            f"Head branch: {info.branch}\n"
            f"Base branch: {self.runtime.config.base_ref}\n"
        )
        try:
            pull_request = self.runtime.github_client.create_pull_request(
                title, body, self.runtime.config.base_ref, info.branch
            )
        except Exception:
            raise GitHubWorkstreamError("GitHub pull-request operation failed") from None
        number = self._github_result_number(pull_request, "pull request")

        store.update_metadata(
            remote_branch=info.branch,
            pull_request_number=number,
        )
        state = store.load()
        self.runtime.observability.trace_emit(
            "workstream.pull_request.created",
            phase="publication",
            cycle=cycle,
            data={
                "remote_branch": info.branch,
                "pull_request_number": number,
                "reviewed_candidate_sha": reviewed_candidate_sha,
            },
            once=True,
        )
        self.runtime.observability.trace_emit(
            "workstream.metadata",
            phase="publication",
            cycle=cycle,
            data=self._github_metadata_payload(state),
        )
    def _validate_github_publication_mode(self) -> None:
        github = self.runtime.config.github
        if (
            github.enabled
            and github.pull_request_mode == "create"
            and (
                not self.runtime.config.publish.enabled
                or self.runtime.config.publish.mode != PublishMode.RUN_BRANCH.value
            )
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request creation requires a published run branch",
                code="GITHUB_PR_REQUIRES_RUN_BRANCH",
            )
    def publish_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        candidate: Mapping[str, Any],
    ) -> RunResult:
        """Publish the reviewed candidate of cycle *number* after its PASS."""

        # Only the exact SHA a durable reviewer PASS names is ever published.
        evidence = candidate_evidence(ctx.run_dir, number)
        review = (
            accepted_review(review_dir(ctx.run_dir, number), evidence, candidate["commit_sha"])
            if evidence is not None and evidence.staged_tree_sha == candidate["tree_sha"]
            else None
        )
        if (
            review is None
            or review.verdict is not ReviewVerdict.PASS
            or review.route is not ReviewRoute.NONE
        ):
            raise PipelineFailure(
                "REVIEW_AUTHORITY_MISSING",
                "no accepted reviewer PASS names the candidate commit",
            )
        if candidate.get("no_change") is True:
            if not review.summary.startswith("SPEC_ALREADY_SATISFIED:"):
                raise PipelineFailure(
                    "REVIEW_AUTHORITY_MISSING",
                    "no-change PASS must explicitly confirm SPEC_ALREADY_SATISFIED",
                )
            self.runtime.cycle_update(store, number, status="completed_no_change")
            state = store.set_run_state(
                RunMachineState(disposition=RunDisposition.COMPLETED),
                no_change=True,
                no_change_candidate_sha=candidate["commit_sha"],
                reviewed_candidate_sha=candidate["commit_sha"],
                commit_sha=None,
                published=False,
                approved_tree_sha=candidate["tree_sha"],
                current_step=None,
            )
            mark_checkpoint_completed(ctx.run_dir)
            self.runtime.observability.trace_emit(
                "run.completed_no_change", phase="run", cycle=number,
                data={"candidate_sha": candidate["commit_sha"], "tree_sha": candidate["tree_sha"]},
                once=True,
            )
            return RunResult.of(ctx.run_dir, state)
        approved_tree = candidate["tree_sha"]
        self.runtime.cycle_update(store, number, status="approved")
        store.update_metadata(approved_tree_sha=approved_tree, current_step=None)
        return self._complete_candidate_publication(
            store=store, run_dir=ctx.run_dir, info=ctx.info,
            approved_tree=approved_tree, commit_sha=candidate["commit_sha"],
            repository_reference=ctx.repository_reference, cycle=number,
        )
    def authorize_candidate_tree(
        self, evidence: EvidenceBundle, worktree: Path, parent_sha: str, branch_ref: str,
    ) -> str:
        """Authorize the immutable candidate tree before semantic review."""

        if not evidence.deterministic_passed or evidence.failures or not evidence.staged_tree_sha:
            raise CommitBoundaryError("deterministic gate did not pass for candidate commit")
        if symbolic_head(worktree) != branch_ref or current_head(worktree) != parent_sha:
            raise CommitBoundaryError("worktree HEAD changed before candidate commit")
        candidate = evidence.staged_tree_sha
        if index_tree_sha(worktree) != candidate or candidate_tree_sha(worktree) != candidate:
            raise CommitBoundaryError("candidate tree changed before candidate commit")
        if _status_has_unstaged_or_untracked(status_porcelain(worktree)):
            raise CommitBoundaryError("worktree has changes before candidate commit")
        return candidate
    def push_candidate(
        self, *, run_dir: Path, info: WorktreeInfo, cycle: int, candidate: dict[str, Any],
        store: RunStateStore,
    ) -> dict[str, Any]:
        """Best-effort stage the candidate; remote proof never replaces local identity."""

        return CandidateRemoteStaging(
            self.runtime.recovery(store), store=store, budgets=self.runtime.run_options.recovery,
            emit=self.runtime.observability.trace_emit, remote=self.runtime.config.repository.remote,
            remote_required=self.runtime.config.publish.enabled or (
                self.runtime.config.github.enabled
                and self.runtime.config.github.pull_request_mode == "create"
            ),
            # Resolved per call: the Git transport is this façade's dependency.
            remote_tip=lambda *args, **kwargs: remote_run_branch_tip(*args, **kwargs),
            push=lambda *args, **kwargs: push_run_branch(*args, **kwargs),
        ).stage(run_dir=run_dir, info=info, cycle=cycle, candidate=candidate)
    def _publication_push_failed(
        self, store: RunStateStore, run_dir: Path, detail: str, *,
        cycle: int, approved_tree: str | None, **fields: Any,
    ) -> RunResult:
        """A required publication remote is unavailable: wait, keep the candidate."""

        decision, terminal = project_exit(
            "PUSH_FAILED", phase=ResumePhase.PUBLISH, remote_required=True,
        )
        self.runtime.recovery(store).trace(
            "recovery.exhausted", reason="PUSH_FAILED", decision=decision,
            attempt=1, tree_before=approved_tree, tree_after=approved_tree,
            budget_remaining=0, phase="publication", cycle=cycle,
            terminal_status=terminal.status, checkpoint_phase=ResumePhase.PUBLISH,
        )
        state = self.runtime.failure.persist_exit(
            store, "PUSH_FAILED", detail, terminal.status,
            recovery_resumable=terminal.resumable, **fields,
        )
        return RunResult.of(run_dir, state)
    def _cleanup_published_run_branch(
        self,
        *,
        store: RunStateStore,
        info: WorktreeInfo,
        commit_sha: str,
    ) -> dict[str, Any]:
        """Best-effort cleanup after a successful fast-forward publication."""

        cleanup: dict[str, Any] = {
            "status": "warning",
            "remote": self.runtime.config.repository.remote,
            "branch": info.branch,
            "commit_sha": commit_sha,
            "warning": "run branch cleanup did not complete; branch retained",
        }
        try:
            persisted_branch = store.load().get("branch")
            if persisted_branch != info.branch:
                raise GitError("persisted run branch does not match the run branch")
            validate_run_branch(persisted_branch, base_ref=self.runtime.config.base_ref)
            result = delete_run_branch(
                info.source_repo,
                remote=self.runtime.config.repository.remote,
                branch=persisted_branch,
                expected_commit_sha=commit_sha,
                base_ref=self.runtime.config.base_ref,
            )
            cleanup["status"] = result.status
            cleanup.pop("warning")
        except Exception:
            # Cleanup is deliberately not a publication failure.  Keep the
            # warning fixed and secret-free; the branch remains for retry or
            # operator cleanup when its identity is not exact.
            pass
        return cleanup
    def _persist_published_run_branch_cleanup(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: WorktreeInfo,
        commit_sha: str,
        publish_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Complete best-effort cleanup after publication is durable.

        The publication state and completed checkpoint are deliberately
        written by the caller before this method is entered.  If interruption
        happens while cleaning up, the already-published state must not be
        downgraded to INTERRUPTED by the outer run boundary.
        """

        try:
            cleanup = self._cleanup_published_run_branch(
                store=store, info=info, commit_sha=commit_sha,
            )
        except KeyboardInterrupt:
            return store.load()
        publish_payload = {
            **publish_payload,
            "run_branch_cleanup": cleanup,
        }
        atomic_write_text(run_dir / "publish.json", _json_text(publish_payload))
        return store.update_metadata(publish=publish_payload)
    def _complete_candidate_publication(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        info: Any,
        approved_tree: str,
        commit_sha: str,
        repository_reference: RepositoryReference,
        cycle: int,
    ) -> RunResult:
        """Publish an already pushed candidate, only after reviewer PASS."""

        self._validate_github_publication_mode()
        fields: dict[str, Any] = {"commit_sha": commit_sha, "current_step": None, "cycle": cycle}
        try:
            state = store.load()
            assert_deferred_verifications_resolved([
                *(state.get("accepted_steps") or []),
                *(state.get("deferred_verifications") or []),
            ])
            chain_records = accepted_chain_records(run_dir)
            if not chain_records:
                raise GitError("the accepted commit chain is missing")
            validate_accepted_chain(
                info.worktree,
                run_dir=run_dir,
                base_sha=info.base_sha,
                tip_sha=commit_sha,
                approved_tree_sha=approved_tree,
            )
        except (CommitSafetyError, GitError) as exc:
            state = store.record_failure("COMMIT_TREE_MISMATCH", str(exc), **fields)
            return RunResult.of(run_dir, state)
        try:
            if state.get("approved_tree_sha") != approved_tree:
                raise GitError("durable approved tree differs from candidate tree")
            if current_head(info.worktree) != commit_sha:
                raise GitError("candidate commit is not the run branch tip")
            if resolve_tree(info.worktree, commit_sha) != approved_tree:
                raise GitError("candidate commit tree differs from approved tree")
            candidate_record = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
            if (
                not isinstance(candidate_record, dict)
                or candidate_record.get("commit_sha") != commit_sha
                or candidate_record.get("tree_sha") != approved_tree
            ):
                raise GitError("candidate local identity is not exact")
            final_evidence = candidate_evidence(run_dir, cycle)
            if (
                final_evidence is None
                or final_evidence.staged_tree_sha != approved_tree
                or not final_evidence.deterministic_passed
                or not required_checks_passed(final_evidence)
            ):
                raise GitError("final deterministic gate evidence is missing or failed")
            review = accepted_review(
                review_dir(run_dir, cycle), final_evidence, commit_sha
            )
            if (
                review is None
                or review.verdict is not ReviewVerdict.PASS
                or review.route is not ReviewRoute.NONE
                or state.get("reviewed_candidate_sha") != commit_sha
            ):
                raise GitError("reviewer PASS does not name the exact candidate commit")
            remote_required = self.runtime.config.publish.enabled or (
                self.runtime.config.github.enabled
                and self.runtime.config.github.pull_request_mode == "create"
            )
            if remote_required and (
                candidate_record.get("remote") != self.runtime.config.repository.remote
                or candidate_record.get("remote_branch") != info.branch
                or candidate_record.get("remote_sha") != commit_sha
                or candidate_record.get("remote_status") != "available"
                or not isinstance(candidate_record.get("pushed_at"), str)
                or not candidate_record.get("pushed_at")
                or remote_run_branch_tip(
                    info.source_repo,
                    remote=self.runtime.config.repository.remote,
                    branch=info.branch,
                ) != commit_sha
            ):
                raise GitError("candidate remote authority is not exact")
            expected_parent = commit_parents(info.worktree, commit_sha)[0]
            validate_run_branch(info.branch, base_ref=self.runtime.config.base_ref)
            if self.runtime.config.publish.enabled:
                repository_remote_url(info.worktree, self.runtime.config.publish.remote)
        except (GitError, OSError, ValueError):
            state = store.record_failure("COMMIT_TREE_MISMATCH", "candidate identity is not exact", **fields)
            return RunResult.of(run_dir, state)

        self.runtime.observability.trace_emit(
            "publish.started",
            phase="publication",
            cycle=cycle,
            data={
                "enabled": self.runtime.config.publish.enabled,
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "remote": self.runtime.config.publish.remote,
                "target": self.runtime.config.publish.mode,
            },
            once=True,
        )
        if not self.runtime.config.publish.enabled:
            self._ensure_github_pull_request_metadata(
                store=store,
                run_id=str(store.load().get("run_id", "")),
                info=info,
                commit_sha=commit_sha,
                cycle=cycle,
            )
            state = store.set_run_state(
                RunMachineState(disposition=RunDisposition.COMPLETED), **fields,
            )
            mark_checkpoint_completed(run_dir)
            self.runtime.observability.trace_emit(
                "publish.completed",
                phase="publication",
                cycle=cycle,
                data={
                    "enabled": False,
                    "status": "committed",
                    "commit_sha": commit_sha,
                    "tree_sha": approved_tree,
                },
                once=True,
            )
            return RunResult.of(run_dir, state)

        # Publication is this run's operation from here on: the checkpoint
        # owns that phase, and the store projects its status.
        self.runtime.write_checkpoint(
            run_dir, ResumePhase.PUBLISH, cycle=cycle,
            head=commit_sha, tree=approved_tree,
            expected_parent_sha=expected_parent,
        )
        store.update_metadata(**fields)
        base_branch = self.runtime.config.base_ref
        try:
            fast_forward = self.runtime.config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
            candidate = _read_json_artifact(_candidate_commit_path(run_dir, cycle))
            if not isinstance(candidate, dict) or candidate.get("commit_sha") != commit_sha:
                raise GitError("candidate commit artifact does not match publication")
            if remote_run_branch_tip(
                info.source_repo, remote=self.runtime.config.repository.remote, branch=info.branch
            ) != commit_sha:
                raise GitError("candidate run branch is not pushed")
            if fast_forward:
                outcome = publish_fast_forward_base(
                    info.source_repo, remote=self.runtime.config.publish.remote,
                    base_branch=base_branch, base_sha=info.base_sha,
                    commit_sha=commit_sha, approved_tree=approved_tree,
                    run_branch=info.branch, expected_parent=expected_parent,
                    accepted_commits=chain_records or None,
                )
                publish_payload = {
                    "mode": PublishMode.FAST_FORWARD_BASE.value,
                    "target": base_branch, "remote": self.runtime.config.publish.remote,
                    "branch": base_branch, "run_branch": info.branch,
                    "base_sha": info.base_sha, "commit_sha": commit_sha,
                    "web_url": _commit_web_url(repository_reference, commit_sha),
                    "status": "pushed", "local_base_updated": outcome.local_base_updated,
                    "base_checked_out_in": list(outcome.base_checked_out_in),
                    "run_branch_cleanup": {
                        "status": "pending",
                        "remote": self.runtime.config.publish.remote,
                        "branch": info.branch,
                        "commit_sha": commit_sha,
                    },
                }
            else:
                publish_payload = {
                    "mode": PublishMode.RUN_BRANCH.value,
                    "target": info.branch, "remote": self.runtime.config.publish.remote,
                    "branch": info.branch, "commit_sha": commit_sha,
                    "web_url": _commit_web_url(repository_reference, commit_sha),
                    "status": "pushed",
                }
        except BaseMovedError as exc:
            decision = classify_failure("BASE_MOVED_SINCE_RUN")
            self.runtime.recovery(store).trace(
                "recovery.classified", reason="BASE_MOVED_SINCE_RUN",
                decision=decision, attempt=1, tree_before=approved_tree,
                tree_after=approved_tree, budget_remaining=0,
                phase="publication", cycle=cycle,
            )
            state = store.record_failure(
                "BASE_MOVED_SINCE_RUN", f"{exc}; candidate remains unpublished",
                publish={"mode": self.runtime.config.publish.mode, "target": base_branch,
                         "remote": self.runtime.config.publish.remote, "commit_sha": commit_sha,
                         "status": "refused", "local_base_updated": False}, **fields,
            )
            return RunResult.of(run_dir, state)
        except BasePushError as exc:
            detail = "push did not complete"
            if exc.local_base_updated:
                detail += f"; local {base_branch} already points to {commit_sha}"
            return self._publication_push_failed(
                store, run_dir, detail, cycle=cycle, approved_tree=approved_tree,
                publish={"mode": self.runtime.config.publish.mode, "target": base_branch,
                         "remote": self.runtime.config.publish.remote, "commit_sha": commit_sha,
                         "status": "push-failed", "local_base_updated": exc.local_base_updated},
                **fields,
            )
        except (GitError, OSError, ValueError):
            return self._publication_push_failed(
                store, run_dir, "publication did not complete", cycle=cycle,
                approved_tree=approved_tree, **fields,
            )
        self._ensure_github_pull_request_metadata(
            store=store,
            run_id=str(store.load().get("run_id", "")),
            info=info,
            commit_sha=commit_sha,
            cycle=cycle,
        )
        atomic_write_text(run_dir / "publish.json", _json_text(publish_payload))
        metadata_fields = {"remote_branch": info.branch} if self.runtime.config.github.enabled else {}
        state = store.set_run_state(
            RunMachineState(disposition=RunDisposition.COMPLETED),
            publish=publish_payload,
            **metadata_fields,
            **fields,
        )
        mark_checkpoint_completed(run_dir)
        self.runtime.observability.trace_emit(
            "publish.completed",
            phase="publication",
            cycle=cycle,
            data={
                "enabled": True,
                "status": publish_payload.get("status"),
                "commit_sha": commit_sha,
                "tree_sha": approved_tree,
                "remote": publish_payload.get("remote"),
                "target": publish_payload.get("target"),
            },
            once=True,
        )
        if fast_forward:
            state = self._persist_published_run_branch_cleanup(
                store=store, run_dir=run_dir, info=info, commit_sha=commit_sha,
                publish_payload=publish_payload,
            )
        return RunResult.of(run_dir, state)

    # -- resume ------------------------------------------------------------
