"""Writable high-tier audit between a diagnostic gate and its authoritative rerun."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.base import AgentRunRequest
from ..evidence import EvidenceBundle, extract_failure_evidence, scan_staged_security
from ..gitops import (
    candidate_tree_sha, changed_paths_between_trees, commit_tree, current_head,
    index_tree_sha, stage_all, status_porcelain,
)
from ..models import ExecutionRole, GateStage
from ..result import atomic_write_text
from ..scope import ScopeViolation
from ..state import RunStateStore
from .audit_protocol import AuditReport, parse_audit_report
from .candidate import accepted_chain_records
from .pipeline_v2 import CyclePlan, PipelineFailure, PipelineV2Context, cycle_dir

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


def _evidence_payload(evidence: EvidenceBundle) -> dict[str, Any]:
    checks = []
    for check in evidence.checks:
        get = check.get if isinstance(check, dict) else lambda key, default=None: getattr(check, key, default)
        stdout = _read_excerpt(get("stdout_log", ""))
        stderr = _read_excerpt(get("stderr_log", ""))
        checks.append({
            "id": get("name"),
            "exit_code": get("exit_code"),
            "failure_kind": get("failure_kind"),
            "skipped_reason": get("skipped_reason"),
            "failure_excerpt": extract_failure_evidence(
                stdout_log=stdout, stderr_log=stderr,
                stdout_tail=get("stdout_tail", ""), stderr_tail=get("stderr_tail", ""),
                max_bytes=12_000,
            ) if get("exit_code") or get("timed_out") else "",
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
        baseline = state.get("baseline", {})
        verdicts = baseline.get("checks", []) if isinstance(baseline, dict) else []
        new_failure_ids = sorted({
            failure_id for verdict in verdicts if isinstance(verdict, dict)
            for failure_id in verdict.get("new_failure_ids", [])
            if isinstance(failure_id, str)
        })
        out_of_scope = sorted({
            path for step in steps if isinstance(step, dict)
            for path in step.get("out_of_scope_paths", [])
            if isinstance(path, str)
        })
        payload = {
            "spec": ctx.spec,
            "milestone": state.get("current_milestone"),
            "prior_iteration_remaining": state.get("iteration_remaining", []),
            "plan": dataclasses.asdict(cycle_plan.plan),
            "normalizations": getattr(cycle_plan.plan, "normalizations", ()),
            "steps": steps,
            "failed_continue_steps": [
                step for step in steps if step.get("status") == "FAILED_CONTINUED"
            ],
            "diff": evidence.diff,
            "changed_paths": list(evidence.changed_files),
            "out_of_scope_paths": out_of_scope,
            "baseline": baseline,
            "gate_verdicts": verdicts,
            "new_failure_ids": new_failure_ids,
            "gate": _evidence_payload(evidence),
            "retry_history": state.get("recovery", {}),
            "escalation_history": state.get("escalations", []),
        }
        prompt = (
            "You are the active high-tier AUDIT authority for this batch. You may edit any "
            "repository path except paths forbidden by the run's hard-deny policy. "
            "Tests and build files are editable. Use the supplied harness check evidence. "
            "Your own sandbox's inability to run Docker, Playwright, pytest, or another "
            "check is not a blocker. The harness will rerun checks after you finish. "
            "Correct new regressions, missing SPEC behavior and bad implementations. "
            "Change obsolete tests when SPEC requires it; never weaken a SPEC invariant. "
            "Remove dead code, unneeded compatibility and temporary adapters, and simplify "
            "overcomplicated code. Keep new refactor paths within 15 when practical.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            + "\n\nEnd your last message with exactly this block, using '- none' for empty lists:\n"
            "META AUDIT v1\n\nSTATUS\nDONE|NEEDS_WORK|SPEC_DECISION\n\nFIXED\n- ...\n\n"
            "REFACTORED\n- ...\n\nREMAINING\n- ...\n\nRISKS\n- ...\nEND META AUDIT\n"
        )
        atomic_write_text(directory / "prompt.txt", prompt)
        profile_id = ctx.selection.audit.profile_id
        executor = self.runtime.composition.executor_for_profile(profile_id, ExecutionRole.AUDITOR)
        if not executor.capabilities.edits_workspace:
            raise PipelineFailure("AUDIT_PROFILE_NOT_WRITABLE")
        result = executor.run(AgentRunRequest(
            role=ExecutionRole.AUDITOR, profile_id=profile_id, prompt=prompt,
            worktree=worktree, artifact_dir=directory, mutable_paths=None,
            prompt_mode="revision",
        ))
        if current_head(worktree) != parent:
            raise PipelineFailure("AGENT_GIT_VIOLATION", "audit agent moved HEAD")
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
