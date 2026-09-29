"""Candidate publication: immutable candidate identity and remote facts.

This module owns the candidate boundary and the publication of the accepted
candidate: it authorizes the candidate tree against its evidence before
anything else observes it, stages the run branch best-effort, creates the
optional GitHub workstream metadata and publishes the base branch.  It never
plans, audits or runs a check.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ..commit_gate import CommitSafetyError, assert_deferred_verifications_resolved
from ..evidence import EvidenceBundle
from ..attempt_transaction import status_has_unstaged_or_untracked
from ..gitops import (
    BaseMovedError,
    BasePushError,
    GitError,
    RepositoryReference,
    WorktreeInfo,
    candidate_tree_sha,
    commit_message,
    commit_parents,
    current_head,
    delete_run_branch,
    index_tree_sha,
    normalize_github_web_url,
    publish_fast_forward_base,
    push_run_branch,
    remote_run_branch_tip,
    repository_remote_url,
    resolve_tree,
    status_porcelain,
    symbolic_head,
    validate_linear_commit_chain,
    validate_run_branch,
)
from ..integrations.github import GitHubWorkstreamError
from ..models import GateStage, PublishMode, RunDisposition, RunMachineState
from ..recovery_policy import classify_failure
from ..result import RunResult, atomic_write_text
from ..resume import ResumePhase
from ..state import RunStateStore
from .pipeline_v2 import PipelineFailure, PipelineV2Context, candidate_dir
from .recovery import project_exit
from .shared import (
    CandidatePushError,
    CommitBoundaryError,
    is_object_id,
    json_text,
    read_json_artifact,
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
            "candidate_commit_sha",
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
                    code="CONFIGURATION_INVALID",
                )
            return
        if github.issue_mode == "link-existing":
            requested = github.issue_number
            if requested is None:
                raise GitHubWorkstreamError(
                    "github.issue_number is required for link-existing",
                    code="CONFIGURATION_INVALID",
                )
            try:
                issue = self.runtime.github_client.read_issue(requested)
            except Exception:
                raise GitHubWorkstreamError("GitHub issue operation failed") from None
            if issue is None:
                raise GitHubWorkstreamError(
                    "requested GitHub issue was not found",
                    code="CONFIGURATION_INVALID",
                )
            number = self._github_result_number(issue, "issue")
            if number != requested:
                raise GitHubWorkstreamError(
                    "GitHub issue response did not match the requested issue",
                    code="CONFIGURATION_INVALID",
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
        """Create the requested PR from the accepted, gated run branch."""

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
                code="CONFIGURATION_INVALID",
            )
        accepted_candidate_sha = state.get("candidate_commit_sha")
        if (
            not is_object_id(accepted_candidate_sha)
            or accepted_candidate_sha != commit_sha
        ):
            raise GitHubWorkstreamError(
                "GitHub pull-request candidate authority is not the accepted candidate",
                code="CONFIGURATION_INVALID",
            )
        try:
            if current_head(info.worktree) != accepted_candidate_sha:
                raise GitHubWorkstreamError(
                    "local accepted candidate does not match the candidate commit",
                    code="CONFIGURATION_INVALID",
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
                code="CONFIGURATION_INVALID",
            ) from None
        if remote_tip != accepted_candidate_sha:
            raise GitHubWorkstreamError(
                "remote run branch tip does not match the accepted candidate",
                code="CONFIGURATION_INVALID",
            )
        plan_state = state.get("planner") if isinstance(state.get("planner"), Mapping) else {}
        plan_title = plan_state.get("title") if isinstance(plan_state.get("title"), str) else "MetaHarness run"
        title = self._github_issue_title(plan_title, run_id)
        body = (
            "MetaHarness accepted workstream metadata.\n\n"
            f"Run ID: {run_id}\n"
            f"Accepted commit: {commit_sha}\n"
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
                "candidate_commit_sha": accepted_candidate_sha,
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
                code="CONFIGURATION_INVALID",
            )
    def publish_candidate(
        self, store: RunStateStore, ctx: PipelineV2Context, number: int,
        candidate: Mapping[str, Any],
    ) -> RunResult:
        """Publish the candidate accepted by the post-audit deterministic gate."""

        if candidate.get("no_change") is True:
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
        """Authorize the immutable candidate tree before the audit."""

        if not evidence.deterministic_passed or evidence.failures or not evidence.staged_tree_sha:
            raise CommitBoundaryError("deterministic gate did not pass for candidate commit")
        if symbolic_head(worktree) != branch_ref or current_head(worktree) != parent_sha:
            raise CommitBoundaryError("worktree HEAD changed before candidate commit")
        candidate = evidence.staged_tree_sha
        if index_tree_sha(worktree) != candidate or candidate_tree_sha(worktree) != candidate:
            raise CommitBoundaryError("candidate tree changed before candidate commit")
        if status_has_unstaged_or_untracked(status_porcelain(worktree)):
            raise CommitBoundaryError("worktree has changes before candidate commit")
        return candidate
    def push_candidate(
        self, *, run_dir: Path, info: WorktreeInfo, cycle: int, candidate: dict[str, Any],
        store: RunStateStore,
    ) -> dict[str, Any]:
        """Best-effort stage the candidate; remote proof never replaces local identity."""

        return CandidateRemoteStaging(
            self.runtime.recovery(store), store=store,
            attempts=self.runtime.run_options.budget.step_attempts,
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

        decision, terminal = project_exit("PUSH_FAILED", phase=ResumePhase.PUBLISH)
        self.runtime.recovery(store).trace(
            "recovery.exhausted", reason="PUSH_FAILED", decision=decision,
            attempt=1, tree_before=approved_tree, tree_after=approved_tree,
            budget_remaining=0, phase="publication", cycle=cycle,
            terminal_status=terminal.status, checkpoint_phase=ResumePhase.PUBLISH,
        )
        state = self.runtime.failure.persist_exit(
            store, "PUSH_FAILED", detail, terminal.disposition,
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

        The publication state is deliberately written by the caller before
        this method is entered. If interruption
        happens while cleaning up, the outer run boundary must not rewrite the
        already-published state as a failed interruption.
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
        atomic_write_text(run_dir / "publish.json", json_text(publish_payload))
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
        """Publish an already pushed candidate after the authoritative gate."""

        self._validate_github_publication_mode()
        fields: dict[str, Any] = {"commit_sha": commit_sha, "current_step": None, "cycle": cycle}
        try:
            state = store.load()
            assert_deferred_verifications_resolved([
                *(state.get("accepted_steps") or []),
                *(state.get("deferred_verifications") or []),
            ])
            chain = validate_linear_commit_chain(
                info.worktree,
                base_sha=info.base_sha,
                tip_sha=commit_sha,
                approved_tree_sha=approved_tree,
            )
            run_id = state.get("run_id")
            if not isinstance(run_id, str) or any(
                f"MetaHarness-Run: {run_id}" not in commit_message(info.worktree, item)
                for item in chain
            ):
                raise GitError("run history contains a commit from another authority")
        except (CommitSafetyError, GitError) as exc:
            state = store.record_failure("COMMIT_GATE_FAILED", str(exc), **fields)
            return RunResult.of(run_dir, state)
        try:
            if current_head(info.worktree) != commit_sha:
                raise GitError("candidate commit is not the run branch tip")
            if resolve_tree(info.worktree, commit_sha) != approved_tree:
                raise GitError("candidate commit tree differs from approved tree")
            parents = commit_parents(info.worktree, commit_sha)
            if commit_sha != info.base_sha and len(parents) != 1:
                raise GitError("candidate commit has no single parent")
            expected_parent = parents[0] if parents else info.base_sha
            remote_required = self.runtime.config.publish.enabled or (
                self.runtime.config.github.enabled
                and self.runtime.config.github.pull_request_mode == "create"
            )
            if remote_required and remote_run_branch_tip(
                    info.source_repo,
                    remote=self.runtime.config.repository.remote,
                    branch=info.branch,
                ) != commit_sha:
                raise GitError("candidate remote authority is not exact")
            validate_run_branch(info.branch, base_ref=self.runtime.config.base_ref)
            if self.runtime.config.publish.enabled:
                repository_remote_url(info.worktree, self.runtime.config.publish.remote)
        except (GitError, OSError, ValueError):
            state = store.record_failure("COMMIT_GATE_FAILED", "candidate identity is not exact", **fields)
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
            self.runtime.write_checkpoint(
                run_dir, ResumePhase.CANDIDATE_PUSH, iteration=cycle,
                head=commit_sha,
            )
            state = store.set_run_state(
                RunMachineState(
                    ResumePhase.CANDIDATE_PUSH, RunDisposition.COMPLETED,
                ),
                **fields,
            )
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
            run_dir, ResumePhase.PUBLISH, iteration=cycle,
            head=commit_sha,
        )
        store.update_metadata(**fields)
        base_branch = self.runtime.config.base_ref
        try:
            fast_forward = self.runtime.config.publish.mode == PublishMode.FAST_FORWARD_BASE.value
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
        atomic_write_text(run_dir / "publish.json", json_text(publish_payload))
        metadata_fields = {"remote_branch": info.branch} if self.runtime.config.github.enabled else {}
        state = store.set_run_state(
            RunMachineState(disposition=RunDisposition.COMPLETED),
            publish=publish_payload,
            **metadata_fields,
            **fields,
        )
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
    chain = read_json_artifact(chain_path)
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
        atomic_write_text(_candidate_commit_path(ctx.run_dir, number), json_text(payload))
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
            atomic_write_text(_candidate_commit_path(ctx.run_dir, number), json_text(skipped))
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
            previous = read_json_artifact(_candidate_commit_path(run_dir, cycle - 1))
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
        atomic_write_text(_candidate_commit_path(run_dir, cycle), json_text(candidate))
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
        atomic_write_text(_candidate_commit_path(run_dir, cycle), json_text(unavailable))
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
