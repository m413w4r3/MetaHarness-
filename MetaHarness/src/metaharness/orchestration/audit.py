"""Writable high-tier audit between a diagnostic gate and its authoritative rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.base import (
    AGENT_RATE_LIMITED,
    AGENT_RUNTIME_FAILED,
    AGENT_TIMEOUT,
    AgentRunRequest,
)
from ..evidence import EvidenceBundle, extract_failure_evidence, scan_staged_security
from ..gitops import (
    candidate_tree_sha,
    changed_paths_between_trees,
    commit_tree,
    current_head,
    diff_between_trees,
    index_tree_sha,
    stage_all,
    status_porcelain,
)
from ..models import ExecutionRole, GateStage
from ..prompt_contracts import write_prompt_diagnostics
from ..redaction import redact
from ..result import atomic_write_text
from ..scope import ScopeViolation
from ..state import RunStateStore
from .audit_prompt import build_audit_payload
from .audit_protocol import AuditReport, parse_audit_report
from .candidate import accepted_chain_records
from .pipeline_v2 import (
    BudgetExhausted,
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    cycle_dir,
    gate_dir,
)
from .shared import read_json_artifact

if TYPE_CHECKING:
    from .runtime import RunRuntime


def audit_dir(ctx: PipelineV2Context, cycle_plan: CyclePlan) -> Path:
    return cycle_dir(ctx.run_dir, cycle_plan.cycle) / "audit"


def _read_excerpt(path: str, *, limit: int = 10_000) -> str:
    try:
        with Path(path).open("rb") as stream:
            return stream.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _evidence_payload(evidence: EvidenceBundle, directory: Path | None = None) -> dict[str, Any]:
    checks = []
    for check in evidence.checks:
        get = check.get if isinstance(check, dict) else lambda key, default=None: getattr(check, key, default)
        def log_path(name: str, get=get) -> str:
            value = get(name, "") or get(name + "_path", "")
            if value and directory is not None and not Path(value).is_absolute():
                return str(directory / value)
            return value

        stdout_path = log_path("stdout_log")
        stderr_path = log_path("stderr_log")
        failed = bool(get("exit_code") or get("timed_out"))
        stdout = _read_excerpt(stdout_path) if failed else ""
        stderr = _read_excerpt(stderr_path) if failed else ""
        checks.append({
            "id": get("name"),
            "exit_code": get("exit_code"),
            "failure_kind": get("failure_kind"),
            "skipped_reason": get("skipped_reason"),
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
            "failure_excerpt": extract_failure_evidence(
                stdout_log=stdout, stderr_log=stderr,
                stdout_tail=get("stdout_tail", ""), stderr_tail=get("stderr_tail", ""),
                max_bytes=12_000,
            ) if failed else "",
        })
    return {
        "passed": evidence.deterministic_passed,
        "failures": list(evidence.failures),
        "warnings": list(evidence.warnings),
        "checks": checks,
    }


class AuditService:
    def __init__(self, runtime: RunRuntime) -> None:
        self.runtime = runtime

    def run(
        self, store: RunStateStore, ctx: PipelineV2Context,
        cycle_plan: CyclePlan, stage: GateStage, evidence: EvidenceBundle,
        attempt: int,
    ) -> AuditReport:
        directory = audit_dir(ctx, cycle_plan) / f"{attempt:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        worktree = ctx.info.worktree
        report_path = directory / "report.json"
        if report_path.is_file():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise PipelineFailure("DURABLE_ARTIFACT_CORRUPTED", "audit report is unreadable") from exc
            if (
                existing.get("stage") != stage.value
                or existing.get("input_tree") != evidence.staged_tree_sha
                or existing.get("commit_sha", current_head(worktree)) != current_head(worktree)
            ):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "audit report does not match the worktree")
            return AuditReport(
                existing["status"], tuple(existing["fixed"]),
                tuple(existing["refactored"]), tuple(existing["remaining"]),
                tuple(existing["risks"]),
            )
        parent = current_head(worktree)
        tree_before = candidate_tree_sha(worktree)
        state = store.load()
        steps = self.runtime.composition.completed_steps(ctx, cycle_plan)
        first_step = read_json_artifact(
            cycle_dir(ctx.run_dir, cycle_plan.cycle) / "implementation/steps"
            / cycle_plan.plan.steps[0].id / "step.json",
        )
        batch_base_tree = first_step.get("tree_before", ctx.base_tree_sha) if isinstance(first_step, dict) else ctx.base_tree_sha
        batch_changed_paths = changed_paths_between_trees(worktree, batch_base_tree, tree_before)
        baseline = state.get("baseline", {})
        evidence_directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        gate = _evidence_payload(evidence, evidence_directory)
        records = {step["id"]: step for step in state.get("steps", []) if isinstance(step, dict) and "id" in step}
        for step in steps:
            records[step["id"]] = {**records.get(step["id"], {}), **step}
        spec_path = directory / "spec.md"
        atomic_write_text(spec_path, ctx.spec)
        prompt_inputs = {
            "spec": ctx.spec, "plan": cycle_plan.plan, "steps": list(records.values()),
            "baseline": baseline if isinstance(baseline, dict) else {}, "gate": gate,
            "changed_paths": batch_changed_paths, "diff_base_tree": batch_base_tree,
            "candidate_parent": parent, "prior_remaining": state.get("iteration_remaining", []),
            "hard_deny": self.runtime.config.scope.hard_deny,
            "file_refs": {
                "spec": str(spec_path),
                "plan": str(ctx.run_dir / "iterations" / f"{ctx.iteration:02d}" / "plan/task_plan.json"),
                "normalizations": str(ctx.run_dir / "iterations" / f"{ctx.iteration:02d}" / "plan/plan.normalizations.json"),
                "steps": str(cycle_dir(ctx.run_dir, cycle_plan.cycle) / "implementation/steps"),
                "diff": str(directory / "diff.patch"),
                "evidence": str(evidence_directory / "evidence.json"),
                "baseline": str(ctx.run_dir / "state.json"),
            },
            "budget_bytes": self.runtime.config.prompt_budget.audit_max_bytes,
        }
        if (exhausted := self.runtime.budget_exhausted(store)) is not None:
            raise BudgetExhausted(exhausted)
        profiles = (ctx.selection.audit, *ctx.selection.audit_fallbacks)[:ctx.options.budget.step_attempts]
        executions = []
        for index, profile in enumerate(profiles):
            if (exhausted := self.runtime.budget_exhausted(store)) is not None:
                raise BudgetExhausted(exhausted)
            executor = self.runtime.composition.executor_for_profile(profile.profile_id, ExecutionRole.AUDITOR)
            if not executor.capabilities.edits_workspace:
                raise PipelineFailure("AUDIT_PROFILE_NOT_WRITABLE")
            artifact_dir = directory if index == 0 else directory / "executors" / f"{index + 1:03d}"
            artifacts_readable = getattr(executor.capabilities, "reads_external_artifacts", False)
            candidate_diff = redact(
                diff_between_trees(worktree, batch_base_tree, candidate_tree_sha(worktree)), self.runtime.secrets,
            )
            atomic_write_text(
                artifact_dir / "diff.patch",
                candidate_diff,
            )
            call_payload = build_audit_payload(
                    **{**prompt_inputs,
                       "file_refs": {**prompt_inputs["file_refs"], "diff": str(artifact_dir / "diff.patch")},
                       "changed_paths": changed_paths_between_trees(worktree, batch_base_tree, candidate_tree_sha(worktree))},
                    artifacts_readable=artifacts_readable,
                    candidate_diff=candidate_diff if not artifacts_readable else "",
                    handoff=("The preceding auditor hit an execution limit. Its uncommitted edits "
                            "are preserved in this worktree. Inspect and complete them, then produce "
                            "the full audit report for the batch, including the preceding edits.") if index else "",
            )
            write_prompt_diagnostics(artifact_dir, call_payload)
            atomic_write_text(artifact_dir / "prompt.txt", call_payload.rendered)
            result = executor.run(AgentRunRequest(
                role=ExecutionRole.AUDITOR, profile_id=profile.profile_id,
                prompt=call_payload.rendered, worktree=worktree, artifact_dir=artifact_dir,
                mutable_paths=None, prompt_mode="revision",
                read_only_paths=(ctx.run_dir,) if artifacts_readable else (),
            ))
            if current_head(worktree) != parent:
                raise PipelineFailure("AGENT_GIT_VIOLATION", "audit agent moved HEAD")
            try:
                self.runtime.config.scope.check(
                    changed_paths_between_trees(worktree, tree_before, candidate_tree_sha(worktree)),
                    worktree=worktree,
                )
            except ScopeViolation as exc:
                raise PipelineFailure(exc.code, exc.detail) from exc
            failed = (
                result.status != "completed" or result.exit_code not in (None, 0)
                or result.timed_out or result.terminal_is_error is True
            )
            reason = (
                AGENT_TIMEOUT if result.timed_out else
                AGENT_RATE_LIMITED if AGENT_RATE_LIMITED in (result.backend_reason, result.exit_reason) else
                result.exit_reason or AGENT_RUNTIME_FAILED
            )
            executions.append({
                "profile_id": profile.profile_id, "status": result.status,
                "exit_code": result.exit_code, "reason": reason if failed else None,
                "prompt_bytes": call_payload.total_bytes,
            })
            atomic_write_text(directory / "executions.json", json.dumps(executions, indent=2) + "\n")
            if not failed:
                break
            if reason == AGENT_RATE_LIMITED and index + 1 < len(profiles):
                continue
            raise PipelineFailure(reason, {
                "profile_id": profile.profile_id, "exit_code": result.exit_code,
                "report_path": result.report_path,
            })
        report = parse_audit_report(result.final_message)
        if report is None:
            raise PipelineFailure("AGENT_PROTOCOL_FAILED", "audit report is missing or malformed")
        tree_after = candidate_tree_sha(worktree)
        paths = changed_paths_between_trees(worktree, tree_before, tree_after)
        try:
            self.runtime.config.scope.check(paths, worktree=worktree)
        except ScopeViolation as exc:
            raise PipelineFailure(exc.code, exc.detail) from exc
        record = {
            "stage": stage.value,
            "input_tree": evidence.staged_tree_sha,
            "status": report.status,
            "fixed": list(report.fixed),
            "refactored": list(report.refactored),
            "remaining": list(report.remaining),
            "risks": list(report.risks),
            "changed_paths": list(paths),
            "tree_before": tree_before,
            "tree_after": tree_after,
            "failure_ids": list(evidence.failures),
            "profile_id": profile.profile_id,
            "executions": executions,
        }
        if paths:
            stage_all(worktree)
            if (
                current_head(worktree) != parent
                or index_tree_sha(worktree) != tree_after
                or candidate_tree_sha(worktree) != tree_after
                or any(line.startswith("??") or len(line) < 2 or line[1] != " " for line in status_porcelain(worktree))
            ):
                raise PipelineFailure("COMMIT_TREE_MISMATCH", "audit candidate changed before commit")
            security_failures = scan_staged_security(worktree, secrets=self.runtime.secrets)
            if security_failures:
                raise PipelineFailure("COMMIT_SECURITY_FAILURE", ", ".join(security_failures[:20]))
            record["commit_sha"] = commit_tree(
                worktree, tree_sha=tree_after, parent_sha=parent,
                subject=f"metaharness(audit): cycle {cycle_plan.cycle.number}",
                body=f"MetaHarness-Run: {ctx.run_id}",
                reflog_message=f"metaharness: audit cycle {cycle_plan.cycle.number}",
            )
            chain = list(accepted_chain_records(ctx.run_dir))
            chain.append({"commit_sha": record["commit_sha"], "tree_sha": tree_after, "parent_sha": parent})
            atomic_write_text(
                ctx.run_dir / "accepted-chain.json",
                json.dumps({"commits": chain}, ensure_ascii=False, indent=2) + "\n",
            )
        atomic_write_text(report_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        store.update_metadata(audit=record)
        return report
