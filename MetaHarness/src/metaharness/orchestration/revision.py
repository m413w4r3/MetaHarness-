"""The Claude revision sub-domain: prompts, reports and scope requests."""

from __future__ import annotations

import dataclasses
import json
import re

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)
from .shared import (
    CheckRepairScope,
    OrchestrationError,
    _PROMPTS_DIR,
    _SECOND_CHECK_REPAIR_PHASES,
    _artifact_tail,
    _bounded_report,
    _bounded_v2_report,
    _check_payload,
    _git_ownership,
    _json_text,
    _ownership_violations,
    _read_bounded_text,
    _record_failure_tree,
    _status_has_unstaged_or_untracked,
)
from ..evidence import (
    EvidenceBundle,
    collect_evidence,
)
from ..gitops import (
    GitError,
    RepositoryReference,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    repository_reference_dict,
    restore_paths_from_tree,
    stage_all,
    status_porcelain,
)
from ..models import (
    ExecutionSelection,
    ExecutionRole,
    HarnessConfig,
    ImplementationStep,
    RunStatus,
)
from ..planning_v2 import TaskPlanV2
from ..prompt_contracts import (
    build_semantic_revision_payload,
    write_prompt_diagnostics,
)
from ..redaction import redact
from ..result import atomic_write_text
from ..resume import ResumePhase
from ..run_options import EffectiveRepairScopePolicy
from ..state import RunStateStore
from ..usage import normalize_usage
from ..validation import config_with_check_authority
from ..agent.base import (
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentExecutor,
    AgentRunRequest,
)
from ..agent.protocol import ScopeRequest, parse_scope_request


_MAX_REPAIR_CLAUDE_REPORT_BYTES = 16 * 1024


_SCOPE_REQUEST_HEADER = "META SCOPE REQUEST v1"


_SCOPE_REQUEST_ROUTE = "CLAUDE_SCOPE_REQUEST"


def _bounded_repair_claude_report(text: str) -> str:
    """Bound the advisory Claude 001 report handed to the repair planner.

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
    """Render a compact deterministic index of approved step contracts."""

    lines: list[str] = []
    for step in plan.steps:
        writes = ",".join((*step.write_set, *step.create_set)) or "NONE"
        deletes = ",".join(step.delete_set) or "NONE"
        lines.extend((
            f"{step.id} | title={step.title} | writes={writes} | deletes={deletes}",
            f"  objective={step.objective}",
            f"  invariants={step.forbidden}",
            f"  verify={step.verify}",
        ))
    return "\n".join(lines) + ("\n" if lines else "NONE\n")


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
    """Keep 002 history structural; reports live in ``luna_reports``."""

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
    candidate_identity: str = "",
    bounded_diff_evidence: str = "",
    diagnostics_dir: str | Path | None = None,
    budget_bytes: int = 120_000,
) -> str:
    template = (_PROMPTS_DIR / "reviser.txt").read_text(encoding="utf-8")
    payload = build_semantic_revision_payload(
        spec=spec,
        compact_approved_contract_index=_revision_contract_index(plan),
        candidate_identity=candidate_identity or "UNKNOWN",
        changed_files=changed_files,
        required_checks_summary=pre_checks,
        mutable_scope=mutable_scope,
        bounded_diff_evidence=bounded_diff_evidence or "NONE\n",
        template=template,
        budget_bytes=budget_bytes,
    )
    if diagnostics_dir is not None:
        write_prompt_diagnostics(diagnostics_dir, payload)
    return payload.rendered


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


_REVISION_CHECK_LOG_BYTES = 16 * 1024


def _revision_check_context(payload: Mapping[str, Any]) -> str:
    """Render compact pre-revision check state for Claude.

    Successful checks contribute status only.  Output tails are decision
    evidence only for failed checks and stay bounded for prompt safety; the
    complete logs remain in the durable check artifacts.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("check payload must be a mapping")

    failures = [
        item for item in payload.get("failures", [])
        if isinstance(item, str)
    ]
    failed_names = {
        item.split(":", 1)[1]
        for item in failures
        if item.startswith("CHECK_FAILED:")
    }
    checks: list[dict[str, Any]] = []
    raw_checks = payload.get("checks", [])
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raw_checks = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, Mapping):
            continue
        name = raw_check.get("name")
        failed = name in failed_names
        check: dict[str, Any] = {
            "name": name,
            "required": bool(raw_check.get("required", False)),
            "exit_code": raw_check.get("exit_code"),
            "timed_out": bool(raw_check.get("timed_out", False)),
            "workspace_mutated": bool(raw_check.get("workspace_mutated", False)),
        }
        if failed:
            for key in ("stdout_tail", "stderr_tail"):
                value = raw_check.get(key)
                if isinstance(value, str) and value:
                    data = value.encode("utf-8", errors="replace")
                    if len(data) > _REVISION_CHECK_LOG_BYTES:
                        data = data[-_REVISION_CHECK_LOG_BYTES:]
                    check[key] = data.decode("utf-8", errors="replace")
        checks.append(check)

    return _json_text({
        "deterministic_passed": bool(payload.get("deterministic_passed", False)),
        "failure_ids": failures,
        "checks": checks,
    })


def _agent_auth_failure(
    run_dir: Path, stderr: str, *, revision_dir: Path | None = None
) -> bool:
    events_path = (revision_dir or (run_dir / "revision")) / "agent.events.jsonl"
    haystack = f"{stderr}\n{_artifact_tail(events_path)}".casefold()
    return any(
        marker in haystack
        for marker in (
            "401 unauthorized", "unauthorized", "authentication required",
            "not logged in", "not authenticated", "please log in",
            "authentication failed", "login required",
        )
    )


@dataclasses.dataclass(frozen=True)
class RevisionRunner:
    """Runs one semantic-revision or check-repair executor cycle.

    Every dependency is injected explicitly: the runner never receives the
    ``Orchestrator`` instance.  ``config``/``secrets``/``effective_repair_scope``
    are the live values of the run; the callables are the minimal set of
    operations the runner does not own -- checkpoint writers, the Claude
    executor, artifact redaction, the durable pre-check reader, and the
    check-repair classification and prompt (owned by ``check_repair``, injected
    to keep this module below it in the import order).
    """

    config: HarnessConfig
    secrets: tuple[str, ...]
    effective_repair_scope: EffectiveRepairScopePolicy
    checkpoint: Callable[..., None]
    write_phase_checkpoint: Callable[..., None]
    approved_check_authority_sha256: Callable[[Path], str | None]
    run_revision: Callable[..., Any]
    ensure_revision_artifacts: Callable[[Path, Any], None]
    redact_revision_artifacts: Callable[..., None]
    step_results_for_cycle: Callable[[int], list[dict[str, Any]]]
    reusable_pre_checks: Callable[[Path, str], dict[str, Any] | None]
    hard_integrity_failures: Callable[[EvidenceBundle], list[str]]
    soft_check_failures: Callable[[EvidenceBundle], list[str]]
    check_repair_scope_candidates: Callable[..., list[str]]
    check_repair_prompt: Callable[..., str]
    agent_executor: AgentExecutor | None = None
    legacy_failure_names: bool = False

    def run(
        self,
        *,
        store: RunStateStore,
        run_dir: Path,
        repo: Path,
        base_sha: str,
        base_tree_sha: str,
        spec: str,
        plan: TaskPlanV2,
        repository_reference: RepositoryReference,
        info: Any,
        branch_ref: str,
        ownership_before: Any,
        selection: ExecutionSelection,
        artifact_dir: Path | None = None,
        mutable_scope: list[str] | None = None,
        deferred_mismatches: str | None = None,
        deferred_mismatch_present: bool = False,
        check_repair_evidence: EvidenceBundle | None = None,
        cycle: int = 1,
        check_repair_scope: CheckRepairScope | None = None,
        check_repair_phase_override: ResumePhase | None = None,
        check_repair_next_phase_override: ResumePhase | None = None,
        check_repair_attempt: int | None = None,
    ) -> tuple[Any | None, str | None]:
        """Run one Claude pre-check/revision/scope cycle.

        Pre-revision checks already durable for the exact current tree are
        reused (a resume never replays them); Claude runs once per attempt.
        """

        claude_phase, review_phase = ResumePhase.SEMANTIC_REVISION, ResumePhase.FINAL_REVIEW
        is_check_repair = check_repair_evidence is not None
        artifact_dir = artifact_dir or (
            run_dir / "revision" / "check-repair" / f"C0{cycle}"
            if is_check_repair else run_dir / "revision"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        expected_head = current_head(info.worktree)
        stage_all(info.worktree)
        tree_before = candidate_tree_sha(info.worktree)
        if is_check_repair and check_repair_evidence.staged_tree_sha != tree_before:
            return None, "TOCTOU_FAILURE"
        mutable_scope = mutable_scope or sorted({
            path for step in plan.steps
            for path in (*step.write_set, *step.create_set, *step.delete_set)
        })
        if is_check_repair:
            check_repair_scope = check_repair_scope or CheckRepairScope(
                base_paths=tuple(mutable_scope), added_paths=(),
                effective_paths=tuple(mutable_scope),
                policy=self.effective_repair_scope.policy,
                bound=self.effective_repair_scope.max_added_paths,
                source="human-approved mutable scope",
            )
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "schema_version": 2,
                "base_mutable_scope": list(check_repair_scope.base_paths),
                "added_paths": list(check_repair_scope.added_paths),
                "effective_mutable_scope": list(check_repair_scope.effective_paths),
                "policy": check_repair_scope.policy,
                "bound": check_repair_scope.bound,
                "source": check_repair_scope.source,
                "bound_exceeded": (
                    check_repair_scope.policy == "auto-bounded"
                    and not check_repair_scope.added_paths
                    and len(self.check_repair_scope_candidates(
                        repo=repo, worktree=info.worktree, tree_sha=tree_before,
                        run_dir=run_dir, evidence=check_repair_evidence,
                        base_mutable_scope=check_repair_scope.base_paths,
                    )) > check_repair_scope.bound
                ),
            }))
        else:
            atomic_write_text(artifact_dir / "scope.json", _json_text({
                "approved_mutable_scope": mutable_scope,
                "source": "human-approved mutable scope",
            }))
        if is_check_repair:
            # This boundary is deliberately written before invoking Claude so
            # a timeout/transport failure resumes this exact corrective pass.
            repair_phase = check_repair_phase_override or (
                ResumePhase.CHECK_REPAIR
            )
            self.checkpoint(
                run_dir, repair_phase, cycle=cycle, head=expected_head, tree=tree_before,
                check_repair_attempt=check_repair_attempt,
            )
            pre_payload = {
                "checks": _check_payload(check_repair_evidence),
                "failures": list(check_repair_evidence.failures),
                "deterministic_passed": check_repair_evidence.deterministic_passed,
                "staged_tree_sha": check_repair_evidence.staged_tree_sha,
            }
        else:
            reused = self.reusable_pre_checks(artifact_dir, tree_before)
            if reused is None:
                self.write_phase_checkpoint(
                    run_dir,
                    ResumePhase.DETERMINISTIC_GATE,
                    cycle=cycle, head=expected_head, tree=tree_before,
                )
                store.update(status=RunStatus.PRE_REVISION_VALIDATING, current_step=None)
                check_config, check_ids = config_with_check_authority(
                    self.config, run_dir, requested_check_ids=plan.required_checks or None,
                    expected_sha256=self.approved_check_authority_sha256(run_dir),
                )
                pre_evidence = collect_evidence(
                    info.worktree, base_sha, check_config,
                    required_check_ids=check_ids,
                    evidence_dir=artifact_dir, secrets=self.secrets,
                    check_failures_hard=False,
                    expected_head_sha=expected_head,
                    enforce_diff_size=False,
                    allow_empty_diff=expected_head != base_sha,
                )
                pre_payload = {
                    "checks": _check_payload(pre_evidence),
                    "failures": list(pre_evidence.failures),
                    "deterministic_passed": pre_evidence.deterministic_passed,
                    "staged_tree_sha": pre_evidence.staged_tree_sha,
                }
                atomic_write_text(artifact_dir / "pre_checks.json", _json_text(pre_payload))
                pre_hard = self.hard_integrity_failures(pre_evidence)
                # A clean deferred mismatch intentionally leaves no candidate
                # delta for the pre-revision gate.  Claude is the recovery owner,
                # so EMPTY_DIFF is evidence for Claude here, not a terminal gate.
                # A deferred *verify* dependency is not this case: that step did
                # change the candidate, so the normal gate applies.
                if deferred_mismatch_present:
                    pre_hard = [item for item in pre_hard if item != "EMPTY_DIFF"]
                if pre_hard:
                    return None, pre_hard[0].split(":", 1)[0]
                # Pre-revision checks complete: the next operation is Claude.  The
                # authorized HEAD is the worktree HEAD (base for 001, the 001
                # candidate commit for 002), exactly as _validate_resume expects.
                self.checkpoint(run_dir, claude_phase, cycle=cycle, head=expected_head, tree=tree_before)
            else:
                pre_payload = reused
        if is_check_repair:
            previous_report = ""
            if check_repair_phase_override in _SECOND_CHECK_REPAIR_PHASES:
                # The second pass must read the *first repair's* report, not
                # the initial revision it already superseded.
                previous_dir = run_dir / "revision" / "check-repair" / f"C0{cycle}"
                previous_report = _read_bounded_text(previous_dir / "agent.final.md")
            revision_prompt = self.check_repair_prompt(
                spec=spec,
                plan=plan,
                approved_contract_index=_revision_contract_index(plan),
                changed_files="\n".join(check_repair_evidence.changed_files),
                evidence=check_repair_evidence,
                mutable_scope=mutable_scope,
                previous_report=previous_report,
                added_paths=(check_repair_scope.added_paths if check_repair_scope else ()),
                scope_source=(check_repair_scope.source if check_repair_scope else ""),
                legacy=self.legacy_failure_names,
                candidate_identity=tree_before,
                budget_bytes=self.config.prompt_budget.check_repair_max_bytes,
                diagnostics_dir=artifact_dir,
            )
        else:
            revision_prompt = _revision_prompt(
                repository_reference=repository_reference,
                spec=spec,
                plan=plan,
                changed_files="\n".join(changed_paths_between_trees(repo, base_tree_sha, tree_before)),
                execution_anomalies=_revision_execution_anomalies(
                    self.step_results_for_cycle(cycle)
                ),
                pre_checks=_revision_check_context(pre_payload),
                mutable_scope=_json_text(mutable_scope),
                deferred_mismatches=deferred_mismatches or "NONE\n",
                candidate_identity=tree_before,
                # The semantic reviser can inspect the current worktree
                # directly.  Keep the secondary diff excerpt optional so a
                # large or adversarial diff never becomes the default prompt.
                bounded_diff_evidence="NONE\n",
                diagnostics_dir=artifact_dir,
                budget_bytes=self.config.prompt_budget.semantic_revision_max_bytes,
            )
        store.update(status=RunStatus.REVISING, current_step=None)
        atomic_write_text(artifact_dir / "tree_before.txt", tree_before.rstrip() + "\n")
        legacy_injected_check_repair = is_check_repair and self.legacy_failure_names
        active_selected = (
            getattr(selection, "reviser", None)
            if legacy_injected_check_repair
            else getattr(selection, "check_repair", None)
            if is_check_repair
            else getattr(selection, "semantic_reviser", None)
        )
        # Historical V4 snapshots expose only ``reviser``; V5 uses the
        # role-specific fields above.  This fallback is read-only compatibility
        # for old artifacts and is never used to choose a new V5 role.
        if active_selected is None and not hasattr(selection, "semantic_reviser"):
            active_selected = getattr(selection, "reviser", None)
        if active_selected is None:
            return None, (
                "CHECK_REPAIR_PROFILE_MISSING"
                if is_check_repair else "SEMANTIC_REVISER_PROFILE_MISSING"
            )
        active_role = (
            ExecutionRole.REPAIR
            if is_check_repair and hasattr(selection, "semantic_reviser")
            and not legacy_injected_check_repair
            else ExecutionRole.REVISER
        )
        if self.agent_executor is not None:
            result = self.agent_executor.run(
                AgentRunRequest(
                    role=active_role,
                    profile_id=active_selected.profile_id,
                    prompt=revision_prompt,
                    worktree=Path(info.worktree),
                    artifact_dir=artifact_dir,
                    mutable_paths=tuple(mutable_scope),
                    prompt_mode="revision",
                )
            )
        else:
            result = self.run_revision(
                selection, info.worktree, run_dir, revision_prompt,
                revision_dir=artifact_dir,
                profile_id=active_selected.profile_id,
                role=active_role,
                mutable_paths=tuple(mutable_scope),
            )
        self.ensure_revision_artifacts(artifact_dir, result)
        agent_auth_failure = _agent_auth_failure(
            run_dir, result.stderr_tail, revision_dir=artifact_dir
        )
        self.redact_revision_artifacts(run_dir, revision_dir=artifact_dir)
        result = dataclasses.replace(
            result,
            final_message=redact(result.final_message, self.secrets),
            stderr_tail=redact(result.stderr_tail, self.secrets),
        )
        def failure_reason(generic: str) -> str:
            if not self.legacy_failure_names:
                return generic
            backend = getattr(result, "backend_reason", None)
            if generic == AGENT_TIMEOUT:
                return "CLAUDE_TIMEOUT"
            if backend == "CLAUDE_AUTH_FAILURE" or agent_auth_failure:
                return "CLAUDE_AUTH_FAILURE"
            if generic == AGENT_PROTOCOL_FAILED and getattr(result, "terminal_subtype", None) == "error_max_turns":
                return "CLAUDE_MAX_TURNS"
            if generic == AGENT_SCOPE_VIOLATION:
                return "CLAUDE_COMMITTED"
            return "CLAUDE_FAILED"

        if getattr(result, "timed_out", False) or getattr(result, "exit_reason", None) == AGENT_TIMEOUT:
            _record_failure_tree(artifact_dir, info.worktree)
            return result, failure_reason(AGENT_TIMEOUT)
        terminal_is_error = getattr(result, "terminal_is_error", None) is True
        terminal_subtype = getattr(result, "terminal_subtype", None)
        # A structured terminal marked ``is_error`` is a failure on its own.
        # Some CLI versions and wrappers still exit 0 after one, so exit code
        # is the last signal consulted, never the gate for the others.
        if terminal_is_error or getattr(result, "exit_code", None) not in (None, 0) or getattr(result, "exit_reason", None) in {
            AGENT_START_FAILED, AGENT_RUNTIME_FAILED, AGENT_PROTOCOL_FAILED,
            AGENT_SCOPE_VIOLATION,
        }:
            _record_failure_tree(artifact_dir, info.worktree)
            if agent_auth_failure or getattr(result, "backend_reason", None) == "CLAUDE_AUTH_FAILURE":
                return result, failure_reason(AGENT_RUNTIME_FAILED)
            # Claude's terminal result is authoritative when available.  The
            # textual fallback above remains for older CLI versions and old
            # artifacts that do not expose terminal metadata.
            if terminal_subtype == "error_max_turns":
                return result, failure_reason(AGENT_PROTOCOL_FAILED)
            return result, failure_reason(
                getattr(result, "exit_reason", None) or AGENT_RUNTIME_FAILED
            )
        revision_ownership = _git_ownership(repo, info.worktree)
        if revision_ownership.head != expected_head:
            return result, failure_reason(AGENT_SCOPE_VIOLATION)
        violations = _ownership_violations(
            ownership_before, revision_ownership,
            branch_ref=branch_ref, base_sha=expected_head,
        )
        if violations:
            return result, "AGENT_GIT_VIOLATION"
        stage_all(info.worktree)
        tree_after = candidate_tree_sha(info.worktree)
        atomic_write_text(artifact_dir / "tree_after.txt", tree_after.rstrip() + "\n")
        changed_paths = changed_paths_between_trees(repo, tree_before, tree_after)
        outside_scope = [path for path in changed_paths if path not in set(mutable_scope)]
        # META SCOPE REQUEST is a check-repair-only machine protocol.
        # Semantic revision may report an out-of-scope dependency in prose, but
        # must not enter the bounded check-repair scope state machine.
        scope_request = (
            parse_scope_request(result.final_message)
            if is_check_repair else None
        )
        malformed_scope_request = (
            is_check_repair
            and _SCOPE_REQUEST_HEADER in result.final_message
            and scope_request is None
        )
        usage = normalize_usage(result.usage)
        revision_state = {
            "profile_id": active_selected.profile_id,
            "status": "NO_CHANGE" if tree_after == tree_before else "COMPLETED",
            "tree_before": tree_before,
            "tree_after": tree_after,
            "usage": usage,
            **(
                {
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                }
                if scope_request is not None else {}
            ),
            **(
                {"scope_request_warning": "malformed scope request ignored as authority"}
                if malformed_scope_request else {}
            ),
        }
        atomic_write_text(artifact_dir / "usage.json", _json_text(usage))
        atomic_write_text(artifact_dir / "report.json", _json_text({
            **revision_state,
            "final": _bounded_report(result.final_message),
            "stderr_tail": result.stderr_tail,
            "changed_paths": list(changed_paths),
            "outside_scope_paths": list(outside_scope),
            **({"failure_ids": self.soft_check_failures(check_repair_evidence)}
               if is_check_repair else {}),
        }))
        store.update(status=RunStatus.REVISING, revision=revision_state)
        if scope_request is not None:
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True,
                    "scope_request": _scope_request_payload(scope_request),
                    "scope_request_diagnostic": _scope_request_diagnostic(scope_request),
                },
            )
        elif malformed_scope_request:
            store.update(
                status=RunStatus.REVISING,
                check_repair={
                    "attempted": True,
                    "scope_request_warning": "malformed scope request ignored as authority",
                },
            )
        if outside_scope:
            # This is a successful Claude transport with an unsafe candidate,
            # so the exact failed tree must remain durable for the fail-closed
            # rollback proof used by deterministic check-repair recovery.
            _record_failure_tree(artifact_dir, info.worktree)
            return result, "REVISION_SCOPE_VIOLATION"
        if scope_request is not None:
            # A valid request is advisory evidence, never an authorization
            # delta.  Even an in-scope/no-op attempt is rolled back atomically
            # before the durable bridge path is allowed to inspect it.
            _record_failure_tree(artifact_dir, info.worktree)
            try:
                restore_paths_from_tree(info.worktree, tree_before, list(changed_paths))
                stage_all(info.worktree)
                if (
                    candidate_tree_sha(info.worktree) != tree_before
                    or index_tree_sha(info.worktree) != tree_before
                    or _status_has_unstaged_or_untracked(status_porcelain(info.worktree))
                ):
                    raise GitError("scope-request rollback did not restore the exact tree")
            except (GitError, OSError):
                # Keep the existing dirty-violation recovery as the fail-safe
                # owner of a rollback that could not be proven immediately.
                pass
            return result, _SCOPE_REQUEST_ROUTE
        # Claude complete and durable: the next operation is the final checks
        # followed by candidate commit/push and then the reviewer.
        next_revision_phase = (
            check_repair_next_phase_override or (
                ResumePhase.DETERMINISTIC_GATE
            )
            if is_check_repair
            else ResumePhase.DETERMINISTIC_GATE
        )
        self.checkpoint(
            run_dir, next_revision_phase, cycle=cycle, head=expected_head, tree=tree_after,
            check_repair_attempt=check_repair_attempt,
        )
        return result, None
