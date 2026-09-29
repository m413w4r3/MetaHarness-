"""Compact audit authority; repository source and full artifacts stay on disk."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..models import TaskPlanV2
from ..prompt_contracts import (
    PromptPayload,
    PromptSection,
    build_prompt_payload,
    file_authority,
)


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
        "steps": [
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
