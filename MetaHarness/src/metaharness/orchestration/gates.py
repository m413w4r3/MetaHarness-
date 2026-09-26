"""The deterministic gate: evidence, acceptance and bounded repair.

This module owns the deterministic gate episode of one cycle stage: running
the trusted checks, reading their evidence back and accepting it under the
authority of the run's approved check hash, and every bounded check-repair
attempt until the gate is green or the recovery ladder is exhausted.
"""

from __future__ import annotations

import hashlib, time
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
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
from ..agent.protocol import parse_check_repair_result
from ..evidence import (
    EvidenceBundle,
    collect_evidence,
    required_checks_passed,
)
from ..validation import config_with_check_authority
from ..gitops import (
    GitError,
    commit_parents,
    candidate_tree_sha,
    current_head,
    index_tree_sha,
    resolve_tree,
)
from ..models import (
    ExecutionRole,
    GateStage,
    HarnessConfig,
)
from ..resume import (
    ResumeIntegrityError,
)
from ..result import atomic_write_text
from ..recovery_policy import (
    RecoveryStrategy,
    classify_failure,
)
from ..state import RunStateStore
from .shared import (
    CheckRepairScope,
    _CHECK_ATTEMPT_ARTIFACTS,
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _archive_attempt_tree,
    _check_payload,
    _git_ownership,
    _is_object_id,
    _json_text,
    _read_json_artifact,
    _record_failure_tree,
    _safe_candidate_tree,
)
from .revision import (
    SCOPE_REQUEST_ROUTE,
)
from .check_repair import (
    CheckRepairAttempt,
    CheckRepairCoordinator,
    _SCOPE_REQUEST_SOURCE,
    gate_mutable_authority,
    soft_check_failures,
)
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    check_repair_attempt_dir,
    check_repair_dir,
    gate_dir,
    gate_acceptance_path,
)
from .recovery import RecoveryAttempt
from .resume_validation import (
    load_evidence,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime




class GateService:
    """One owner of the pipeline operations described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def run_gate(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage,
    ) -> EvidenceBundle:
        """Run the authoritative deterministic checks for the current tree."""

        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        attempts_dir = directory / "attempts"
        archived_gate_attempts = sum(
            1 for item in attempts_dir.iterdir()
            if item.is_dir() and item.name.isdigit()
        ) if attempts_dir.is_dir() else 0
        gate_attempt = archived_gate_attempts + (1 if (directory / "evidence.json").is_file() else 0) + 1
        retry_check = self.runtime.check_recovery(store).gate_retries(
            cycle=cycle_plan.cycle.number, stage=stage.value, worktree=ctx.info.worktree,
        )

        _archive_attempt(directory, names=_CHECK_ATTEMPT_ARTIFACTS)
        store.update_metadata(current_step=None)
        evidence = self._final_evidence(
            ctx.info.worktree, ctx.base_sha, directory,
            check_failures_hard=False, reuse=True, stage=stage,
            expected_head_sha=current_head(ctx.info.worktree),
            required_check_ids=cycle_plan.plan.required_checks or None,
            enforce_diff_size=False,
            retry_check_infrastructure=retry_check,
        )
        gate = {
            "stage": stage.value,
            "attempt": gate_attempt,
            "passed": evidence.deterministic_passed,
            "required_check_ids": list(evidence.required_check_ids),
            "failures": list(evidence.failures),
        }
        store.update_metadata(
            checks=_check_payload(evidence),
            staged_tree_sha=evidence.staged_tree_sha,
            changed_files=list(evidence.changed_files),
            deterministic_gate=gate,
            deterministic_gate_attempt=gate_attempt,
        )
        self.runtime.cycle_update(store, cycle_plan.cycle, deterministic_gate=gate)
        if evidence.deterministic_passed:
            current_repair = store.load().get("check_repair")
            if isinstance(current_repair, Mapping) and "operator_retry_fingerprint" in current_repair:
                cleared = dict(current_repair)
                cleared.pop("operator_retry_fingerprint", None)
                store.update_metadata(check_repair=cleared)
        retry_check.settle(evidence)
        return evidence
    @staticmethod
    def load_accepted_gate_evidence(
        ctx: PipelineV2Context, number: int, stage: GateStage,
    ) -> EvidenceBundle | None:
        """Return a green evidence bundle only when its gate acceptance binds it."""

        directory = gate_dir(ctx.run_dir, number, stage)
        acceptance_path = gate_acceptance_path(ctx.run_dir, number, stage)
        if not acceptance_path.is_file():
            return None
        try:
            payload = _read_json_artifact(acceptance_path)
            evidence_path = directory / "evidence.json"
            evidence_bytes = evidence_path.read_bytes()
            evidence = load_evidence(directory)
            current = current_head(ctx.info.worktree)
            current_tree = resolve_tree(ctx.info.worktree, current)
            parents = commit_parents(ctx.info.worktree, current)
        except (OSError, GitError, ValueError) as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "accepted gate evidence is unreadable",
            ) from exc
        digest = hashlib.sha256(evidence_bytes).hexdigest()
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in {1, 2}
            or payload.get("review_cycle") != number
            or payload.get("stage") != stage.value
            or evidence is None
            or evidence.base_sha != ctx.base_sha
            or not evidence.deterministic_passed
            or not required_checks_passed(evidence)
            or bool(evidence.failures)
            or payload.get("tree_sha") != evidence.staged_tree_sha
            or payload.get("commit_sha") != current
            or current_tree != evidence.staged_tree_sha
            or (
                payload.get("no_change") is not True
                and parents != (payload.get("parent_sha"),)
            )
            or not isinstance(payload.get("no_change", False), bool)
            or (
                payload.get("no_change") is True
                and (
                    payload.get("parent_sha") is not None
                    or evidence.diff != ""
                    or bool(evidence.changed_files)
                    or payload.get("commit_created") is not False
                )
            )
            or (
                payload.get("parent_sha") is None
                and (
                    payload.get("no_change") is not True
                    or bool(evidence.changed_files)
                )
            )
            or (
                payload.get("schema_version") == 2
                and payload.get("evidence_sha256") != digest
            )
        ):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "gate acceptance does not bind its evidence",
            )
        return evidence
    def run_check_preflights_recoverably(
        self,
        *,
        store: RunStateStore,
        worktree: Path,
        check_config: HarnessConfig,
        check_ids: Sequence[str],
        counter_key: str,
        phase: str,
        cycle: int | None = None,
    ) -> tuple[str, ...]:
        """Retry each trusted preflight itself under a durable infra budget."""

        return self.runtime.check_recovery(store).run_preflights(
            worktree=worktree, check_config=check_config, check_ids=check_ids,
            counter_key=counter_key, phase=phase, cycle=cycle,
        )
    @staticmethod
    def check_repair_attempt_records(
        run_dir: Path, cycle: int, stage: GateStage,
    ) -> tuple[CheckRepairAttempt, ...]:
        """The contiguous durable attempts of one gate episode."""

        root = check_repair_dir(run_dir, cycle, stage) / "attempts"
        records: list[CheckRepairAttempt] = []
        if not root.is_dir():
            return ()
        for directory in sorted(root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or not directory.name.isdigit():
                continue
            record_path = directory / "attempt.json"
            if not record_path.is_file():
                # A worker can be interrupted after its prompt was persisted
                # but before the successful attempt record.  That boundary is
                # retryable; a present but malformed record is not.
                continue
            payload = _read_json_artifact(record_path, 128 * 1024)
            if not isinstance(payload, dict):
                raise ResumeIntegrityError("check-repair attempt record is malformed")
            number = payload.get("number")
            failed = payload.get("failed_check_ids_before")
            scope = payload.get("mutable_scope")
            before, after = payload.get("tree_before"), payload.get("tree_after")
            worker_result = payload.get("worker_result")
            targeted_check = payload.get("targeted_check")
            blocked_kind = payload.get("blocked_kind")
            note = payload.get("note")
            if (
                isinstance(number, bool) or not isinstance(number, int) or number < 1
                or number != int(directory.name)
                or not isinstance(failed, list) or any(not isinstance(item, str) for item in failed)
                or not isinstance(scope, list) or any(not isinstance(item, str) for item in scope)
                or not _is_object_id(before) or not _is_object_id(after)
                or payload.get("status") != "completed"
                or worker_result != "DONE"
                or targeted_check not in {"PASS", "FAIL"}
                or blocked_kind != "NONE"
                or not isinstance(note, str) or not note.strip()
            ):
                raise ResumeIntegrityError("check-repair attempt record is invalid")
            records.append(CheckRepairAttempt(
                number=number,
                failed_check_ids_before=tuple(failed),
                tree_before=before,
                tree_after=after,
                mutable_scope=tuple(scope),
                worker_result=worker_result,
                targeted_check=targeted_check,
                blocked_kind=blocked_kind,
                note=note,
            ))
        records.sort(key=lambda item: item.number)
        if any(item.number != index for index, item in enumerate(records, start=1)):
            raise ResumeIntegrityError("check-repair attempt records are not contiguous")
        return tuple(records)
    def run_check_repair_attempt(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, attempt: int, evidence: EvidenceBundle,
    ) -> None:
        """One bounded check-repair worker pass on the red gate evidence."""

        number = cycle_plan.cycle.number
        attempt_dir = check_repair_attempt_dir(ctx.run_dir, number, stage, attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        selected = ctx.selection.check_repair
        if selected is None:
            raise PipelineFailure("CHECK_REPAIR_PROFILE_MISSING")
        records = self.check_repair_attempt_records(ctx.run_dir, number, stage)
        previous_scope = None
        if records:
            previous_authority = gate_mutable_authority(
                ctx.run_dir, number, stage,
                base_paths=self.runtime.composition.effective_cycle_scope(ctx, cycle_plan),
                policy_config=self.runtime.repair_scope,
                through_attempt=len(records),
                require_attempt_records=True,
            )
            previous_scope = CheckRepairScope(
                approved_mutable_scope=previous_authority.base_paths,
                initial_repair_scope=previous_authority.initial_paths,
                added_paths=previous_authority.added_paths,
                effective_repair_scope=previous_authority.effective_paths,
                policy=self.runtime.repair_scope.policy,
                bound=self.runtime.repair_scope.max_added_paths,
                source=previous_authority.source,
            )
        soft = soft_check_failures(evidence)
        failed_ids = tuple(item.split(":", 1)[1] for item in soft if ":" in item)
        scope = CheckRepairCoordinator(
            repair_scope_policy=self.runtime.repair_scope,
        ).resolve_scope(
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=evidence.staged_tree_sha,
            evidence_dir=gate_dir(ctx.run_dir, number, stage), evidence=evidence,
            approved_mutable_scope=self.runtime.composition.effective_cycle_scope(ctx, cycle_plan), previous=previous_scope,
        )

        def updated_scope(paths: Sequence[str]) -> CheckRepairScope:
            effective = tuple(sorted(set(paths)))
            if not set(scope.initial_repair_scope).issubset(effective):
                raise PipelineFailure("HUMAN_REQUIRED", "scope request removed existing authority")
            if not set(effective).issubset(scope.approved_mutable_scope):
                raise PipelineFailure(
                    AGENT_SCOPE_VIOLATION,
                    "scope request exceeds the cycle's approved mutable scope",
                )
            added = tuple(path for path in effective if path not in set(scope.initial_repair_scope))
            if len(added) > scope.bound:
                raise PipelineFailure(
                    "HUMAN_REQUIRED", "scope request exceeds the configured check-repair bound",
            )
            return CheckRepairScope(
                approved_mutable_scope=scope.approved_mutable_scope,
                initial_repair_scope=scope.initial_repair_scope,
                added_paths=added,
                effective_repair_scope=effective,
                policy=scope.policy,
                bound=scope.bound,
                source=(_SCOPE_REQUEST_SOURCE if added else scope.source),
            )

        def persist_scope(value: CheckRepairScope) -> None:
            atomic_write_text(attempt_dir / "scope.json", _json_text({
                "schema_version": 3,
                "approved_mutable_scope": list(value.approved_mutable_scope),
                "initial_repair_scope": list(value.initial_repair_scope),
                "added_paths": list(value.added_paths),
                "effective_repair_scope": list(value.effective_repair_scope),
                "policy": value.policy,
                "bound": value.bound,
                "source": value.source,
            }))

        # A scope approval may suspend this exact, unconsumed attempt. Keep
        # its report reachable across resume and apply the existing authority
        # decision before asking the worker again.
        report_path = attempt_dir / "report.json"
        report = _read_json_artifact(report_path, 256 * 1024) if report_path.is_file() else None
        if not isinstance(report, dict):
            archived_reports = sorted(
                (attempt_dir / "attempts").glob("[0-9][0-9]*/report.json"),
                key=lambda item: item.parent.name,
            )
            if archived_reports:
                archived_scope_report = archived_reports[-1]
                report = _read_json_artifact(archived_scope_report, 256 * 1024)
                if isinstance(report, dict):
                    atomic_write_text(report_path, archived_scope_report.read_text(encoding="utf-8"))
        protocol_record = report.get("check_repair_result") if isinstance(report, dict) else None
        pending_scope_request = (
            isinstance(protocol_record, dict)
            and protocol_record.get("result") == "BLOCKED"
            and protocol_record.get("blocked_kind") == "SCOPE"
            and isinstance(report.get("scope_request"), dict)
        )
        if pending_scope_request:
            outcome, mutable_scope = self.runtime.reviews.authorize_semantic_scope_request(
                store, ctx, cycle_plan, attempt_dir, list(scope.effective_repair_scope),
                approved_scope=scope.approved_mutable_scope,
            )
            if outcome != "expanded":
                raise PipelineFailure(
                    "HUMAN_REQUIRED", "check-repair scope request needs an operator decision",
                )
            scope = updated_scope(mutable_scope)
            persist_scope(scope)
            _archive_attempt(attempt_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
        else:
            # Retire artifacts from a previous unverified/failed try without
            # counting that try against max_check_repair_attempts.
            _archive_attempt_tree(attempt_dir)

        decision = classify_failure(soft[0] if soft else "CHECK_FAILED")
        if decision.strategy is not RecoveryStrategy.REPAIR_TARGETED:
            raise PipelineFailure("CHECK_REPAIR_NOT_AUTHORIZED", "check failure is not repairable")
        atomic_write_text(
            attempt_dir / "failed_check_evidence_before.json",
            _json_text(_check_payload(evidence)),
        )
        atomic_write_text(attempt_dir / "profile.json", _json_text({
            "profile_id": selected.profile_id,
            "profile_fingerprint": selected.config_sha256,
            "selected_profile": asdict(selected),
        }))
        progress = {
            "stage": stage.value,
            "attempt_number": attempt,
            "failure_ids": list(soft),
            "repair_profile_id": selected.profile_id,
            "repair_profile_fingerprint": selected.config_sha256,
            "mutable_scope": list(scope.effective_repair_scope),
        }
        self.runtime.cycle_update(
            store, cycle_plan.cycle,
            check_repair={"status": "running", "attempt_count": len(records), **progress},
        )
        store.update_metadata(
            check_repair={"status": "running", "attempt_count": len(records), **progress},
        )
        self.runtime.recovery(store).trace(
            "recovery.classified", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=evidence.staged_tree_sha,
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt + 1),
            phase="validation", cycle=number,
        )
        self.runtime.recovery(store).trace(
            "recovery.started", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=_safe_candidate_tree(ctx.info.worktree),
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - len(records)),
            phase="validation", cycle=number,
        )
        revision_request = {
                "store": store, "cycle": number, "run_dir": ctx.run_dir, "repo": ctx.repo,
                "base_sha": ctx.base_sha, "base_tree_sha": ctx.base_tree_sha, "spec": ctx.spec,
                "plan": cycle_plan.plan, "repository_reference": ctx.repository_reference,
                "info": ctx.info, "branch_ref": ctx.branch_ref,
                "ownership_before": _git_ownership(ctx.repo, ctx.info.worktree),
                "selection": ctx.selection, "artifact_dir": attempt_dir,
                "mutable_scope": list(scope.effective_repair_scope),
                "check_repair_evidence": evidence, "check_repair_scope": scope,
                "check_evidence_dir": gate_dir(ctx.run_dir, number, stage),
                "check_repair_attempt": attempt,
                "gate_stage": stage.value,
        }
        while True:
            try:
                _result, error = self.runtime.reviews.run_revision_with_recovery(
                    store=store, cycle=number, is_check_repair=True, attempt=attempt,
                    request=revision_request,
                )
            except (AgentError, GitError, OSError) as exc:
                error = getattr(exc, "code", None) or AGENT_RUNTIME_FAILED
                _record_failure_tree(attempt_dir, ctx.info.worktree)
            if error != SCOPE_REQUEST_ROUTE:
                break
            outcome, requested_scope = self.runtime.reviews.authorize_semantic_scope_request(
                store, ctx, cycle_plan, attempt_dir, list(scope.effective_repair_scope),
                approved_scope=scope.approved_mutable_scope,
            )
            if outcome != "expanded" or set(requested_scope) == set(scope.effective_repair_scope):
                error = "HUMAN_REQUIRED"
                break
            scope = updated_scope(requested_scope)
            persist_scope(scope)
            _archive_attempt(attempt_dir, names=_REVISION_ATTEMPT_ARTIFACTS)
            revision_request["mutable_scope"] = list(scope.effective_repair_scope)
            revision_request["check_repair_scope"] = scope
            progress["mutable_scope"] = list(scope.effective_repair_scope)
            store.update_metadata(
                check_repair={"status": "running", "attempt_count": len(records), **progress},
            )
        effective_executor = _read_json_artifact(attempt_dir / "executor.json", 16 * 1024)
        effective_profile_id = (
            effective_executor.get("profile_id")
            if isinstance(effective_executor, dict)
            and isinstance(effective_executor.get("profile_id"), str)
            else selected.profile_id
        )
        effective_selected = self.runtime.observability.trace_selected_profile(effective_profile_id, ExecutionRole.REPAIR)
        effective_fingerprint = getattr(effective_selected, "config_sha256", None)
        if error is not None:
            if error in {
                AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_TIMEOUT,
                AGENT_PROTOCOL_FAILED, "CHECK_REPAIR_UNAVAILABLE",
            }:
                error_detail = f"CHECK_REPAIR_UNAVAILABLE after {error}"
                atomic_write_text(attempt_dir / "failure.json", _json_text({
                    "schema_version": 1,
                    "number": attempt,
                    "failed_check_ids_before": list(failed_ids),
                    "tree_before": evidence.staged_tree_sha,
                    "tree_after": candidate_tree_sha(ctx.info.worktree),
                    "mutable_scope": list(scope.effective_repair_scope),
                    "profile_id": effective_profile_id,
                    "profile_fingerprint": effective_fingerprint,
                    "reason": "CHECK_REPAIR_UNAVAILABLE",
                    "infrastructure_reason": error[:120],
                }))
                store.update_metadata(
                    check_repair={
                        "status": "unavailable", "error": error[:120],
                        "attempt_count": len(records), **progress,
                    },
                )
                raise PipelineFailure("CHECK_REPAIR_UNAVAILABLE", error_detail)
            reason = "REVISION_SCOPE_VIOLATION" if error == SCOPE_REQUEST_ROUTE else error
            self.runtime.recovery(store).trace(
                "recovery.completed", reason="CHECK_FAILED", decision=decision,
                attempt=attempt, tree_before=evidence.staged_tree_sha,
                tree_after=_safe_candidate_tree(ctx.info.worktree),
                budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt),
                phase="validation", cycle=number, recovered=False,
            )
            atomic_write_text(attempt_dir / "failure.json", _json_text({
                "schema_version": 1,
                "number": attempt,
                "failed_check_ids_before": list(failed_ids),
                "tree_before": evidence.staged_tree_sha,
                "tree_after": _safe_candidate_tree(ctx.info.worktree),
                "mutable_scope": list(scope.effective_repair_scope),
                "profile_id": effective_profile_id,
                "profile_fingerprint": effective_fingerprint,
                "reason": reason,
            }))
            store.update_metadata(
                check_repair={"status": "failed", "error": reason, **progress},
            )
            raise PipelineFailure(reason, ", ".join(soft))
        protocol = parse_check_repair_result(
            getattr(_result, "final_message", "") if _result is not None else "",
        )
        if (
            protocol is None or protocol.result != "DONE"
            or protocol.targeted_check not in {"PASS", "FAIL"}
            or protocol.blocked_kind != "NONE"
        ):
            # RevisionRunner owns rollback for this branch. Reaching it means
            # the worker result lost its strict proof between orchestration
            # boundaries, so fail closed before writing an attempt record.
            raise PipelineFailure("CHECK_REPAIR_UNAVAILABLE", "repair result is not verifiable")
        record = CheckRepairAttempt(
            number=attempt,
            failed_check_ids_before=failed_ids,
            tree_before=evidence.staged_tree_sha,
            tree_after=candidate_tree_sha(ctx.info.worktree),
            mutable_scope=tuple(scope.effective_repair_scope),
            worker_result=protocol.result,
            targeted_check=protocol.targeted_check,
            blocked_kind=protocol.blocked_kind,
            note=protocol.note,
        )
        atomic_write_text(attempt_dir / "attempt.json", _json_text({
            "schema_version": 1,
            **asdict(record),
            "profile_id": effective_profile_id,
            "profile_fingerprint": effective_fingerprint,
            "status": "completed",
        }))
        # The check-repair budget is consumed only after a verifiable DONE
        # report and its durable attempt record both exist.
        self.runtime.recovery(store).record(RecoveryAttempt(
            phase="validation", reason="CHECK_FAILED", attempt=attempt,
            budget_key="check_repair_attempts", budget=ctx.options.max_check_repair_attempts,
            budget_consumed=attempt, strategy=decision.strategy.value,
            operation_id=(
                f"check-repair:cycle-{number:03d}:{stage.value}:{attempt:02d}"
            ),
            cycle=number, profile_id=selected.profile_id,
            tree_before=evidence.staged_tree_sha, tree_after=record.tree_after,
        ))
        attempts = [*records, record]
        self.runtime.cycle_update(
            store, cycle_plan.cycle,
            check_repair={
                "status": "completed", "attempt_count": len(attempts),
                "attempts": [asdict(item) for item in attempts], **progress,
            },
        )
        store.update_metadata(
            check_repair={
                "status": "completed", "attempt_count": len(attempts),
                "attempts": [asdict(item) for item in attempts], **progress,
            },
        )
        self.runtime.recovery(store).trace(
            "recovery.completed", reason="CHECK_FAILED", decision=decision,
            attempt=attempt, tree_before=evidence.staged_tree_sha,
            tree_after=record.tree_after,
            budget_remaining=max(0, ctx.options.max_check_repair_attempts - attempt),
            phase="validation", cycle=number, recovered=True,
        )
        self.runtime.observability.update_v2_usage(store, ctx.run_dir)
    def replan_responsible_step(
        self, store: RunStateStore, ctx: PipelineV2Context, cycle_plan: CyclePlan,
        stage: GateStage, step: Any, evidence: EvidenceBundle,
    ) -> str:
        """Replan one responsible approved step and re-execute it.

        The rung is not a replay: the responsible step's contract is rewritten
        through the durable contract repair transaction from this gate's own
        failure evidence, its new authority is proven durably, and only then is
        the step re-executed with the descendants its rewritten commit
        invalidates.  The gate implementation itself owns none of that; it
        delegates to the step execution owner and keeps the ladder contract.
        """

        return self.runtime.step_replan.replan_cycle_step(
            store, ctx, cycle_plan, stage, step, evidence,
        )
    def _final_evidence(
        self,
        worktree: Path,
        base_sha: str,
        evidence_dir: Path,
        *,
        check_failures_hard: bool,
        reuse: bool,
        expected_head_sha: str | None = None,
        required_check_ids: tuple[str, ...] | None = None,
        enforce_diff_size: bool = False,
        stage: GateStage | None = None,
        retry_check_infrastructure: Callable[[str, str], bool] | None = None,
    ) -> EvidenceBundle:
        """Final checks for the exact current candidate.

        With *reuse*, durable evidence already frozen for exactly this index
        tree is reused: checks are never replayed for a tree whose evidence is
        complete.
        """

        if reuse:
            stored = load_evidence(evidence_dir)
            if (
                stored is not None
                and stored.base_sha == base_sha
                and current_head(worktree) == (expected_head_sha or base_sha)
                and stored.staged_tree_sha == index_tree_sha(worktree)
                and stored.staged_tree_sha == candidate_tree_sha(worktree)
            ):
                self.runtime.observability.trace_emit(
                    "checks.completed",
                    phase="validation",
                    cycle=self.runtime.trace_cycle,
                    data={
                        "reused": True,
                        "stage": stage.value if stage is not None else None,
                        "passed": stored.deterministic_passed,
                        "failures": list(stored.failures),
                        "required_check_ids": list(stored.required_check_ids),
                        "tree_sha": stored.staged_tree_sha,
                        "changed_paths": list(stored.changed_files),
                    },
                )
                return stored
        checks_started_at = self.runtime.observability.trace_time()
        checks_started_mono = time.perf_counter()
        checks_tree_before = _safe_candidate_tree(worktree)
        self.runtime.observability.trace_emit(
            "checks.started",
            phase="validation",
            cycle=self.runtime.trace_cycle,
            data={
                "stage": stage.value if stage is not None else None,
                "required_check_ids": list(required_check_ids or ()),
                "tree_before": checks_tree_before,
            },
        )
        check_config, check_ids = config_with_check_authority(
            self.runtime.config, evidence_dir, requested_check_ids=required_check_ids,
            expected_sha256=self.runtime.approved_check_authority_sha256(evidence_dir),
        )
        try:
            evidence = collect_evidence(
                worktree, base_sha, check_config, evidence_dir=evidence_dir,
                secrets=self.runtime.secrets, check_failures_hard=check_failures_hard,
                expected_head_sha=expected_head_sha, required_check_ids=check_ids,
                enforce_diff_size=enforce_diff_size,
                # Accepted step commits make the current HEAD itself the
                # candidate.  There is no staged diff against that HEAD, but the
                # authoritative checks still must run and their tree is exact.
                allow_empty_diff=(
                    stage is not None or current_head(worktree) != base_sha
                ),
                retry_check_infrastructure=retry_check_infrastructure,
            )
        except Exception as exc:
            self.runtime.observability.trace_emit(
                "checks.completed",
                phase="validation",
                cycle=self.runtime.trace_cycle,
                data={
                    "stage": stage.value if stage is not None else None,
                    "passed": False,
                    "failures": [type(exc).__name__],
                    "tree_sha": _safe_candidate_tree(worktree),
                    "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                },
            )
            raise
        self.runtime.observability.trace_emit(
            "checks.completed",
            phase="validation",
            cycle=self.runtime.trace_cycle,
            data={
                "stage": stage.value if stage is not None else None,
                "passed": evidence.deterministic_passed,
                "failures": list(evidence.failures),
                "required_check_ids": list(evidence.required_check_ids),
                "tree_sha": evidence.staged_tree_sha,
                "changed_paths": list(evidence.changed_files),
                "wall_time_ms": round((time.perf_counter() - checks_started_mono) * 1000),
                "started_at": checks_started_at,
            },
        )
        return evidence
