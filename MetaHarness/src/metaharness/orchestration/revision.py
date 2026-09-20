"""The Claude revision sub-domain: prompts, reports and scope requests."""

from __future__ import annotations

import json
import re

from pathlib import Path
from typing import (
    Any,
    Mapping,
    Sequence,
)
from .shared import (
    OrchestrationError,
    _PROMPTS_DIR,
    _bounded_report,
    _bounded_v2_report,
    _json_text,
)
from ..gitops import (
    RepositoryReference,
    repository_reference_dict,
)
from ..models import ImplementationStep
from ..planning_v2 import TaskPlanV2
from ..result import atomic_write_text
from ..usage import normalize_usage
from ..claude.agent import ScopeRequest


_MAX_REPAIR_CLAUDE_REPORT_BYTES = 16 * 1024


_SCOPE_REQUEST_HEADER = "META SCOPE REQUEST v1"


_SCOPE_REQUEST_ROUTE = "CLAUDE_SCOPE_REQUEST"


def _bounded_repair_claude_report(text: str) -> str:
    """Bound the advisory Claude C01 report handed to the repair planner.

    The report is consultative; the immutable candidate commit is the code
    authority, so truncating it can never hide a fact the planner must know.
    """

    data = text.encode("utf-8", errors="replace")

    if len(data) <= _MAX_REPAIR_CLAUDE_REPORT_BYTES:
        return text

    marker = (
        "\n[... Claude report truncated for repair planner; "
        "candidate commit is authoritative ...]\n"
    ).encode("utf-8")

    head = data[
        : max(
            0,
            _MAX_REPAIR_CLAUDE_REPORT_BYTES - len(marker),
        )
    ].decode("utf-8", errors="ignore")

    return head + marker.decode("utf-8")


def _scope_request_payload(request: ScopeRequest) -> dict[str, Any]:
    return {
        "reason": request.reason,
        "paths": list(request.paths),
        "evidence": list(request.evidence),
        "authoritative": False,
        "routed_to_bridge_audit": True,
    }


def _scope_request_evidence(request: ScopeRequest | None) -> str:
    if request is None:
        return "NONE\n"
    return "\n".join([
        "REASON",
        request.reason,
        "",
        "PATHS",
        *(f"- {path}" for path in request.paths),
        "",
        "EVIDENCE",
        *(f"- {item}" for item in request.evidence),
        "",
    ])


def _scope_request_diagnostic(request: ScopeRequest) -> str:
    return "\n".join([
        "Claude requested scope expansion:",
        f"  paths: {len(request.paths)}",
        "  authoritative: NO",
        "  routed to bridge audit: YES",
    ])


def _scope_request_from_payload(payload: Any) -> ScopeRequest | None:
    if not isinstance(payload, Mapping):
        return None
    reason = payload.get("reason")
    paths = payload.get("paths")
    evidence = payload.get("evidence")
    if (
        not isinstance(reason, str)
        or not isinstance(paths, list)
        or not isinstance(evidence, list)
        or any(not isinstance(path, str) for path in paths)
        or any(not isinstance(item, str) for item in evidence)
    ):
        return None
    return ScopeRequest(reason=reason, paths=tuple(paths), evidence=tuple(evidence))


def _step_reports_text(results: list[dict[str, Any]]) -> str:
    """Render every step with bounded metadata and a bounded report body."""

    chunks: list[str] = []
    for item in results:
        chunk = "\n".join([
            item["id"], f"status: {item.get('status', 'COMPLETED')}",
            f"profile: {item['profile_id']}",
            f"tree_before: {item['tree_before']}", f"tree_after: {item['tree_after']}",
            f"usage: {json.dumps(item['usage'], sort_keys=True)}", "final report:",
            _bounded_v2_report(item.get("final", "")),
        ]) + "\n"
        chunks.append(chunk)
    return "\n".join(chunks)


def _revision_execution_anomalies(results: list[dict[str, Any]]) -> str:
    """Render only exceptional Luna execution facts for Claude.

    The resulting worktree is the authority for successful implementation
    details.  Reports are retained for reviewer/audit flows, but normal Luna
    narration, usage and tree metadata do not belong in Claude's task prompt.
    """

    records: list[dict[str, Any]] = []
    for item in results:
        status = item.get("status", "COMPLETED")
        exceptional = (
            status != "COMPLETED"
            or bool(item.get("mismatch"))
            or bool(item.get("initial_mismatch"))
            or bool(item.get("deferred_verify"))
            or bool(item.get("mismatch_retry_count"))
        )
        if not exceptional:
            continue

        record: dict[str, Any] = {
            "id": item.get("id"),
            "status": status,
            "changed_paths": list(item.get("changed_paths") or []),
        }
        for key in ("mismatch", "initial_mismatch", "deferred_verify"):
            if item.get(key):
                record[key] = _bounded_v2_report(str(item[key]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        records.append(record)

    return _json_text(records) if records else "NONE\n"


def _revision_contract_index(plan: TaskPlanV2) -> str:
    """Render only the approved behavioral and mutation contract index."""

    return _json_text([
        {
            "id": step.id,
            "title": step.title,
            "depends_on": step.depends_on,
            "objective": step.objective,
            "mutation_scope": {
                "write": list(step.write_set),
                "create": list(step.create_set),
                "delete": list(step.delete_set),
            },
            "verify": step.verify,
            "forbidden": step.forbidden,
        }
        for step in plan.steps
    ])


def _revision_plan_summary(plan: TaskPlanV2) -> str:
    return _json_text({
        "title": plan.title,
        "objective": plan.objective,
        "constraints": plan.constraints,
        "acceptance": plan.acceptance,
        "tests": plan.tests,
        "risks": plan.risks,
    })


def _deferred_contract_mismatches(
    plan: TaskPlanV2, results: list[dict[str, Any]],
) -> str:
    """Render bounded summaries for Claude and the independent reviewer."""

    steps = {step.id: step for step in plan.steps}
    summaries: list[dict[str, Any]] = []
    for item in results:
        deferred = item.get("status") == "DEFERRED_CONTRACT_MISMATCH"
        verify = _bounded_v2_report(str(item.get("deferred_verify") or ""))
        if not deferred and not verify:
            continue
        step = steps.get(item.get("id"))
        if step is None:
            continue
        scope = {
            "write": list(step.write_set),
            "create": list(step.create_set),
            "delete": list(step.delete_set),
        }
        record: dict[str, Any] = {
            "step_id": step.id,
            "step_title": step.title,
            "kind": (
                "DEFERRED_CONTRACT_MISMATCH" if deferred
                else "DEFERRED_VERIFY_DEPENDENCY"
            ),
            "original_scope": scope,
        }
        if deferred:
            record["mismatch"] = _bounded_v2_report(str(item.get("mismatch") or ""))
            record["tree_at_mismatch"] = item.get("tree_before")
        if item.get("initial_mismatch"):
            record["initial_mismatch"] = _bounded_v2_report(str(item["initial_mismatch"]))
        if item.get("mismatch_retry_count"):
            record["mismatch_retry_count"] = item["mismatch_retry_count"]
        if verify:
            # The step completed inside its approved scope but one VERIFY
            # command still fails on a path a later step owns.  Claude and
            # the reviewer must decide; nothing here accepts that failure.
            record["deferred_verify_dependency"] = verify
        summaries.append(record)
    return _json_text(summaries) if summaries else "NONE\n"


def _future_step_ownership(
    steps: Sequence[ImplementationStep], index: int,
) -> dict[str, tuple[str, ...]]:
    """The mutation paths the approved plan assigns to the remaining steps.

    Informative only: a bounded retry uses it to recognize an out-of-scope
    verification dependency.  It never grants write authority.
    """

    ownership: dict[str, tuple[str, ...]] = {}
    for step in steps[index + 1:]:
        paths = tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))
        if paths:
            ownership[step.id] = paths
    return ownership


def _has_deferred_contract_mismatches(results: list[dict[str, Any]]) -> bool:
    return any(item.get("status") == "DEFERRED_CONTRACT_MISMATCH" for item in results)


def _compact_step_history(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep C02 history structural; reports live in ``luna_reports``."""

    compact: list[dict[str, Any]] = []
    for item in results:
        record = {
            "id": item.get("id"),
            "profile_id": item.get("profile_id"),
            "tree_after": item.get("tree_after"),
            "changed_paths": item.get("changed_paths", []),
            "usage": item.get("usage", {}),
        }
        if item.get("status") == "DEFERRED_CONTRACT_MISMATCH":
            record["status"] = item["status"]
            record["mismatch"] = _bounded_v2_report(str(item.get("mismatch") or ""))
        if item.get("deferred_verify"):
            record["deferred_verify"] = _bounded_v2_report(str(item["deferred_verify"]))
        compact.append(record)
    return compact


def _persist_revision_tree(run_dir: Path, name: str, tree: str) -> None:
    atomic_write_text(run_dir / "revision" / name, tree.rstrip() + "\n")


def _render_revision_template(
    template: str, values: Mapping[str, str], *, name: str,
) -> str:
    """Render one Claude template with one non-recursive substitution pass."""

    expected = set(re.findall(r"\{\{[A-Z0-9_]+\}\}", template))
    missing = expected.difference(values)
    if missing:
        raise OrchestrationError(
            f"{name} template contains unresolved placeholders: "
            + ", ".join(sorted(missing))
        )
    return re.sub(
        r"\{\{[A-Z0-9_]+\}\}",
        lambda match: values[match.group(0)],
        template,
    )


def _revision_prompt(
    *,
    repository_reference: RepositoryReference,
    spec: str,
    plan: TaskPlanV2,
    changed_files: str,
    execution_anomalies: str,
    pre_checks: str,
    mutable_scope: str,
    deferred_mismatches: str,
) -> str:
    template = (_PROMPTS_DIR / "reviser.txt").read_text(encoding="utf-8")
    values: dict[str, str] = {
        "{{REPOSITORY_REFERENCE}}": json.dumps(repository_reference_dict(repository_reference), ensure_ascii=False, indent=2),
        "{{SPEC}}": spec,
        "{{PLAN_SUMMARY}}": _revision_plan_summary(plan),
        "{{APPROVED_CONTRACT_INDEX}}": _revision_contract_index(plan),
        "{{EXECUTION_ANOMALIES}}": execution_anomalies,
        "{{CURRENT_CHANGED_FILES}}": changed_files,
        "{{PRE_REVISION_CHECKS}}": pre_checks,
        "{{APPROVED_MUTABLE_SCOPE}}": mutable_scope,
        "{{DEFERRED_LUNA_CONTRACT_MISMATCHES}}": deferred_mismatches,
    }
    return _render_revision_template(template, values, name="reviser")


def _revision_report_text(result: Any, artifact_dir: Path) -> str:
    """Bounded, reviewer-facing record of one Claude revision."""

    def tree(name: str) -> str | None:
        try:
            return (artifact_dir / name).read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeError):
            return None

    return _json_text({
        "final": _bounded_report(result.final_message),
        "tree_before": tree("tree_before.txt"),
        "tree_after": tree("tree_after.txt"),
        "usage": normalize_usage(result.usage),
    })
