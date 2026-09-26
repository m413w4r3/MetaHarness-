"""The semantic revision pass and the reviser/repair worker transaction.

``SemanticRevisionService`` runs the one semantic revision pass over an
implemented cycle and the single revision worker attempt it drives, including
the exact-rollback recovery of that attempt.  The pass reads the pre-semantic
gate evidence, the completed steps and the effective mutable scope, and the
reviser's own scope request is authorised by the run's scope authority before
any expansion is applied.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import (
    Any,
    Mapping,
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
from ..models import ExecutionRole
from ..profiles import (
    ProfileError,
    profile_for_role,
)
from ..redaction import redact
from ..result import atomic_write_text
from ..state import RunStateStore
from .check_failure import (
    hard_integrity_failures,
    soft_check_failures,
)
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    gate_dir,
    pre_semantic_gate_stage,
    semantic_revision_dir,
)
from .resume_validation import (
    _reusable_pre_checks,
    load_evidence,
)
from .revision import (
    RevisionRunner,
    SCOPE_REQUEST_ROUTE,
    check_repair_prompt,
)
from .shared import (
    _REVISION_ATTEMPT_ARTIFACTS,
    _archive_attempt,
    _bounded_report,
    _git_ownership,
    _json_text,
    _read_json_artifact,
    _record_failure_tree,
    _safe_candidate_tree,
)
if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime


class SemanticRevisionService:
    """The semantic revision pass of one run and its worker transaction."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

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
                outcome, mutable_scope = self.runtime.correction_scope.authorize_semantic_scope_request(
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
            check_repair_prompt=check_repair_prompt,
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

