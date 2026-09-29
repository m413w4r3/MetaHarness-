"""Writable high-tier audit between a diagnostic gate and its authoritative rerun."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.base import AGENT_RUNTIME_FAILED, AGENT_TIMEOUT, AgentRunRequest
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
from ..models import ExecutionRole, GateStage, TaskPlanV2
from ..prompt_contracts import (
    PromptPayload,
    PromptSection,
    build_prompt_payload,
    file_authority,
    write_prompt_diagnostics,
)
from ..redaction import redact
from ..result import atomic_write_text
from ..scope import ScopeViolation
from ..state import RunStateStore
from .pipeline_v2 import (
    BudgetExhausted,
    CyclePlan,
    PipelineFailure,
    PipelineV2Context,
    cycle_dir,
    gate_dir,
)
from .publication import accepted_chain_records
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
                "gate": str(evidence_directory / "evidence.json"),
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
            if result.backend_reason == "rate_limited" and not result.timed_out and index + 1 < len(profiles):
                continue
            raise PipelineFailure(reason, {
                "profile_id": profile.profile_id, "exit_code": result.exit_code,
                "report_path": result.report_path,
            })
        report = parse_audit_report(result.final_message)
        if report is None:
            raise PipelineFailure("AGENT_RUNTIME_FAILED", "audit report is missing or malformed")
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
                raise PipelineFailure("COMMIT_GATE_FAILED", "audit candidate changed before commit")
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


def _summary(record: Mapping[str, Any], keys: Sequence[str], *, compact: bool) -> dict[str, Any]:
    """Bound recoverable diagnostics; complete records remain in file_refs."""

    result = {}
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if not compact:
            result[key] = value
            continue
        if isinstance(value, (list, tuple)):
            result[key + "_count"] = len(value)
            result[key] = [str(item)[:500] for item in value[:8]]
            result[key + "_omitted"] = max(0, len(value) - 8)
        elif isinstance(value, str):
            # Artifact pointers and IDs must resolve exactly, never abbreviate them.
            result[key] = value if key in {"id", "stdout_log", "stderr_log"} else value[:500]
        else:
            result[key] = value
    return result


def build_audit_payload(
    *,
    spec: str,
    plan: TaskPlanV2,
    steps: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    gate: Mapping[str, Any],
    changed_paths: Sequence[str],
    diff_base_tree: str,
    candidate_parent: str,
    prior_remaining: Sequence[str],
    file_refs: Mapping[str, str],
    hard_deny: Sequence[str] = (),
    budget_bytes: int = 64_000,
    handoff: str = "",
    artifacts_readable: bool = False,
    candidate_diff: str = "",
) -> PromptPayload:
    """Keep SPEC, milestone boundaries and check verdicts unabridged."""

    step_summaries = [
        {
            key: step[key]
            for key in (
                "step_id",
                "id",
                "title",
                "status",
                "out_of_scope_paths",
                "reason",
            )
            if key in step
        }
        for step in steps
    ]
    batch = {
        "milestone_id": plan.milestone_id,
        "title": plan.milestone_title,
        "objective": plan.objective,
        "milestone_goal": plan.milestone_goal,
        "acceptance": plan.acceptance,
        "constraints": plan.constraints,
        "risks": plan.risks,
        "project_remainder": plan.project_remainder,
        "prior_iteration_remaining": list(prior_remaining),
        "diff_base_tree": diff_base_tree,
        "candidate_parent": candidate_parent,
        "changed_paths": list(changed_paths),
        "file_refs": dict(file_refs) if artifacts_readable else {},
        "hard_deny": list(hard_deny),
        "steps": step_summaries,
        "failed_continue_steps": [
            step for step in step_summaries
            if str(step.get("status", "")).casefold() in {
                "failed_continued", "skipped_dependency",
            }
        ],
    }
    if not artifacts_readable:
        batch["approved_step_contracts"] = [
            {key: getattr(step, key) for key in (
                "id", "title", "context", "read_set", "write_set", "create_set", "delete_set",
                "instructions", "interfaces", "examples", "tests", "pitfalls", "done_when", "verify",
            )}
            for step in plan.steps
        ]
    summary = _summary(gate, ("passed", "failures", "warnings"), compact=artifacts_readable)
    summary["checks"] = [
        _summary(check, ("id", "exit_code", "failure_kind", "skipped_reason", "stdout_log", "stderr_log"), compact=artifacts_readable)
        for check in gate.get("checks", [])
    ]
    summary["baseline"] = [
        _summary(check, (
                "id",
                "verdict",
                "baseline_status",
                "candidate_status",
                "new_failure_ids",
                "warning",
            ), compact=artifacts_readable)
        for check in baseline.get("checks", [])
        if isinstance(check, Mapping)
    ]
    details = "\n\n".join(
        str(check.get("id", "")) + "\n" + check["failure_excerpt"]
        for check in gate.get("checks", [])
        if check.get("failure_excerpt")
    )
    detail_bytes = details.encode("utf-8")
    if artifacts_readable and len(detail_bytes) > 4_000:
        details = detail_bytes[:4_000].decode("utf-8", "ignore") + "\n[read referenced logs for remaining failure details]"

    def encode(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    sections = (
        PromptSection.create("spec", file_authority(spec, file_refs["spec"]) if artifacts_readable and file_refs.get("spec") else spec, True),
        PromptSection.create("batch", encode(batch), True),
        PromptSection.create("handoff", handoff, True),
        PromptSection.create("gate", encode(summary), True),
        PromptSection.create("failure_details", details, not artifacts_readable),
        PromptSection.create("candidate_diff", candidate_diff if not artifacts_readable else "", True),
        PromptSection.create("artifact_access", (
            "Local artifacts are readable. Use the referenced diff.patch if Git is unavailable; "
            "select relevant hunks. Gate diagnostics show counts and bounded samples: read "
            "referenced evidence and baseline fields for omitted failures. Never treat omissions as passes."
            if artifacts_readable else
            "This is an external audit with repository access only. SPEC, approved contracts, "
            "gate and baseline facts and failure excerpts are inline. Local harness artifacts "
            "and log paths are unavailable; do not attempt to read them. Inspect repository source "
            "and the supplied Git object identities directly."
        ), True),
    )
    template = (Path(__file__).parents[1] / "prompts/auditor.txt").read_text(
        encoding="utf-8"
    )
    return build_prompt_payload(
        role="auditor",
        template=template,
        sections=sections,
        placeholders={
            "{{SPEC}}": "spec",
            "{{BATCH}}": "batch",
            "{{GATE}}": "gate",
            "{{HANDOFF}}": "handoff",
            "{{FAILURE_DETAILS}}": "failure_details",
            "{{ARTIFACT_ACCESS}}": "artifact_access",
            "{{CANDIDATE_DIFF}}": "candidate_diff",
        },
        budget_bytes=budget_bytes,
        secondary_order=("failure_details",),
    )


@dataclass(frozen=True)
class AuditReport:
    status: str
    fixed: tuple[str, ...]
    refactored: tuple[str, ...]
    remaining: tuple[str, ...]
    risks: tuple[str, ...]


def parse_audit_report(message: str) -> AuditReport | None:
    if not isinstance(message, str):
        return None
    lines = message.splitlines()
    header, footer = "META AUDIT v1", "END META AUDIT"
    if lines.count(header) != 1 or lines.count(footer) != 1:
        return None
    start, end = lines.index(header), lines.index(footer)
    if any(line.strip() for line in lines[end + 1:]):
        return None
    block = lines[start + 1:end]
    if not block or block[0] != "" or len(block) < 2 or block[1] != "STATUS":
        return None
    if len(block) < 3 or block[2] not in {"DONE", "NEEDS_WORK", "SPEC_DECISION"}:
        return None
    sections: dict[str, tuple[str, ...]] = {}
    index = 3
    for name in ("FIXED", "REFACTORED", "REMAINING", "RISKS"):
        if index >= len(block) or block[index] != "" or index + 1 >= len(block) or block[index + 1] != name:
            return None
        index += 2
        items: list[str] = []
        while index < len(block) and block[index].startswith("- "):
            item = block[index][2:].strip()
            if not item:
                return None
            if item != "none":
                items.append(item)
            index += 1
        if not items and (index == 0 or block[index - 1] != "- none"):
            return None
        sections[name] = tuple(items)
    if index != len(block):
        return None
    if block[2] in {"NEEDS_WORK", "SPEC_DECISION"} and not sections["REMAINING"]:
        return None
    return AuditReport(block[2], sections["FIXED"], sections["REFACTORED"], sections["REMAINING"], sections["RISKS"])
