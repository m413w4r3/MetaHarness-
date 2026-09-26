"""The check-repair sub-domain: failed-check classification and scope."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re

from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
)
from .candidate import accepted_chain_records
from .pipeline_v2 import (
    CyclePlan,
    PipelineFailure,
    check_repair_attempt_dir,
    check_repair_attempts_dir,
    check_repair_dir,
    gate_acceptance_path,
    gate_dir,
    semantic_revision_dir,
)
from .recovery import GateRecoveryStep, RecoveryStepUnavailable
from .shared import (
    CheckRepairScope,
    GateMutableAuthority,
    _PROMPTS_DIR,
    _is_object_id,
    _json_text,
    _read_json_artifact,
)
from ..evidence import (
    EvidenceBundle,
    extract_failure_evidence,
    required_checks_passed,
    SECRET_IN_DIFF,
    SECRET_IN_STAGED_BLOB,
    UNREVIEWABLE_TEXT_DIFF,
    UNSCANNABLE_STAGED_BLOB,
)
from ..gitops import tracked_files_in_tree
from ..gitops import (
    GitError,
    changed_paths_between_trees,
    commit_parents,
    commit_repair_tree,
    commit_revision_tree,
    current_head,
    path_exists_in_tree,
    read_file_at_commit,
    resolve_tree,
)
from ..commit_gate import CommitSafetyError, commit_safety_gate
from ..result import atomic_write_text
from ..models import GateStage, RunStatus, TaskPlanV2
from ..prompt_contracts import build_check_repair_payload, write_prompt_diagnostics
from ..recovery_policy import (
    FailureClass,
    RecoveryFacts,
    RecoveryFingerprint,
    RecoveryProgression,
    RecoveryStrategy,
    admitted_strategies,
    recovery_ladder,
    terminal_strategy,
)
from ..resume import ResumeIntegrityError
from ..run_options import EffectiveRepairScopePolicy, effective_repair_scope_policy
from ..planning.check_replan import (
    PLAN_ARTIFACT as CHECK_REPLAN_PLAN_ARTIFACT,
    check_replan_dir,
    plan_identity,
)
from ..validation import check_result_json


@dataclasses.dataclass(frozen=True)
class CheckRepairAttempt:
    """Durable summary of one deterministic check-repair worker attempt."""

    number: int
    failed_check_ids_before: tuple[str, ...]
    tree_before: str
    tree_after: str
    mutable_scope: tuple[str, ...]
    worker_result: str
    targeted_check: str
    blocked_kind: str
    note: str


# Gate failures for which a semantic review is pointless or unsafe: the
# candidate is empty, unreviewable, not the agent's output, or leaks a secret.
_DIRECT_FAILURES = frozenset(
    {
        "EMPTY_DIFF",
        "HEAD_MISMATCH",
        SECRET_IN_DIFF,
        SECRET_IN_STAGED_BLOB,
        UNSCANNABLE_STAGED_BLOB,
        UNREVIEWABLE_TEXT_DIFF,
    }
)


# These names are deliberately broader than the evidence enum.
# Evidence produced by a newer check runner must never become a repair task
# merely because this module has not learned its exact spelling yet.
_HARD_FAILURE_CODES = frozenset(
    {
        "UNEXPECTED_HEAD",
        "UNEXPECTED_TREE",
        "TREE_MISMATCH",
        "INTEGRITY_MISMATCH",
        "INTEGRITY_FAILURE",
        "DURABLE_ARTIFACT_CORRUPTED",
        "CORRUPTED_DURABLE_ARTIFACT",
        "RESUME_INTEGRITY_FAILURE",
        "SECRET_SECURITY_VIOLATION",
        "SECURITY_VIOLATION",
        "AGENT_INFRASTRUCTURE_FAILURE",
        "AGENT_START_FAILED",
        "AGENT_RUNTIME_FAILED",
        "AGENT_PROTOCOL_FAILED",
        "AGENT_SCOPE_VIOLATION",
        "AGENT_GIT_VIOLATION",
    }
)
_HARD_FAILURE_PREFIXES = (
    "UNEXPECTED_HEAD:",
    "UNEXPECTED_TREE:",
    "TREE_MISMATCH:",
    "INTEGRITY_MISMATCH:",
    "DURABLE_ARTIFACT_CORRUPTED:",
    "CORRUPTED_DURABLE_ARTIFACT:",
    "RESUME_INTEGRITY_FAILURE:",
    "CHECK_MUTATED_FORBIDDEN_FILES:",
    "SECRET_SECURITY_VIOLATION:",
    "SECURITY_VIOLATION:",
)


def hard_integrity_failures(bundle: EvidenceBundle) -> list[str]:
    """Return the failures that close a gate episode without repair.

    A normal configured check failure is soft: it may open a bounded
    check-repair attempt and never reaches the reviewer. Reversible check
    side effects and transient process failures are recovered before they can
    reach this boundary; secrets, ownership and malformed trees remain hard.
    """

    return _hard_failure_items(bundle.failures)


def soft_check_failures(bundle: EvidenceBundle) -> list[str]:
    """Return ordinary deterministic check failures eligible for one repair.

    This deliberately accepts only the exact ``CHECK_FAILED:<name>`` family.
    Anything else, including a future failure category, fails closed and must
    never be handed to the corrective Claude pass.
    """

    if hard_integrity_failures(bundle) or bundle.deterministic_passed:
        return []
    failures = list(bundle.failures)
    if not failures or any(
        not isinstance(item, str) or not item.startswith("CHECK_FAILED:")
        for item in failures
    ):
        return []
    return failures


_MAX_FAILED_CHECK_LOG_INPUT_BYTES = 4 * 1024 * 1024
_CHECK_REPAIR_PROOF_BYTES = 24 * 1024
_MAX_CHANGED_PATH_FALLBACK = 8


_CHECK_SCOPE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:file://)?"
    r"(?:/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"
    r"\.(?:py|pyi|pyx|ts|tsx|js|jsx|mjs|cjs|java|kt|kts|go|rs|rb|php|c|cc|cpp|h|hpp|cs|fs|fsx|swift|scala|sc|vue|svelte|sql|json|yaml|yml|toml|ini|txt|xml|html|css|scss)"
    r"(?::[0-9]+(?::[0-9]+)?)?(?![A-Za-z0-9_.-])"
)


def _read_failure_log(path: Path) -> str:
    """Read a bounded head and tail so summaries and earlier tracebacks survive."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size <= _MAX_FAILED_CHECK_LOG_INPUT_BYTES:
                stream.seek(0)
                return stream.read().decode("utf-8", errors="replace")
            head_bytes = 256 * 1024
            tail_bytes = _MAX_FAILED_CHECK_LOG_INPUT_BYTES - head_bytes
            stream.seek(0)
            head = stream.read(head_bytes)
            stream.seek(size - tail_bytes)
            tail = stream.read(tail_bytes)
            return (
                head.decode("utf-8", errors="replace")
                + f"\n[... {size - _MAX_FAILED_CHECK_LOG_INPUT_BYTES} middle bytes omitted; see complete log ...]\n"
                + tail.decode("utf-8", errors="replace")
            )
    except OSError:
        return ""


def _bound_failure_log(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_FAILED_CHECK_LOG_INPUT_BYTES:
        return value
    head_bytes = 256 * 1024
    tail_bytes = _MAX_FAILED_CHECK_LOG_INPUT_BYTES - head_bytes
    return (
        encoded[:head_bytes].decode("utf-8", errors="replace")
        + "\n[... middle bytes omitted; see complete log ...]\n"
        + encoded[-tail_bytes:].decode("utf-8", errors="replace")
    )


def _failed_check_logs(*, evidence_dir: Path, evidence: EvidenceBundle) -> list[dict[str, Any]]:
    """Read full-log evidence for failed checks without crossing the artifact root."""

    failed_names = {
        failure.split(":", 1)[1]
        for failure in soft_check_failures(evidence)
        if ":" in failure
    }
    root = evidence_dir.resolve()
    persisted_checks = _read_json_artifact(evidence_dir / "checks.json")
    persisted_by_name = {
        item.get("name"): item
        for item in persisted_checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    } if isinstance(persisted_checks, list) else {}
    records: list[dict[str, Any]] = []
    for raw_check in evidence.checks:
        item = dict(raw_check) if isinstance(raw_check, Mapping) else check_result_json(raw_check)
        name = item.get("name")
        if name not in failed_names:
            continue
        stored = persisted_by_name.get(name)
        if isinstance(stored, Mapping):
            for stream in ("stdout", "stderr"):
                key = f"{stream}_log_path"
                if isinstance(stored.get(key), str):
                    item[key] = stored[key]
        for stream in ("stdout", "stderr"):
            path_key = f"{stream}_log_path"
            candidate: Path | None = None
            raw_path = item.get(path_key)
            if isinstance(raw_path, str) and raw_path:
                try:
                    candidate = (root / raw_path).resolve()
                    candidate.relative_to(root)
                except (OSError, RuntimeError, ValueError):
                    item.pop(path_key, None)
                    candidate = None
                else:
                    item[path_key] = candidate.as_posix()
            log_value = item.get(f"{stream}_log")
            if not isinstance(log_value, str) or not log_value:
                log_value = getattr(raw_check, f"{stream}_log", "")
            if not isinstance(log_value, str) or not log_value:
                log_value = _read_failure_log(candidate) if candidate is not None else ""
            item[f"{stream}_log"] = _bound_failure_log(log_value) if isinstance(log_value, str) else ""
        records.append(item)
    return records


def _failure_log_text(records: Sequence[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    per_check_budget = max(1, _CHECK_REPAIR_PROOF_BYTES // max(1, len(records)))
    for record in records:
        proof = extract_failure_evidence(
            stdout_log=str(record.get("stdout_log") or ""),
            stderr_log=str(record.get("stderr_log") or ""),
            stdout_tail=str(record.get("stdout_tail") or ""),
            stderr_tail=str(record.get("stderr_tail") or ""),
            stdout_log_path=(
                record.get("stdout_log_path")
                if isinstance(record.get("stdout_log_path"), str) else None
            ),
            stderr_log_path=(
                record.get("stderr_log_path")
                if isinstance(record.get("stderr_log_path"), str) else None
            ),
            max_bytes=per_check_budget,
        )
        chunks.append(f"CHECK {record.get('name')}\n{proof}")
    return "\n\n".join(chunks)


def _resolve_check_path_candidate(
    raw_path: str,
    *,
    worktree: Path,
    tracked_files: frozenset[str],
) -> str | None:
    """Normalize one path-shaped check-output token without guessing."""

    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = raw_path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
        path = path[1:-1]
    if path.startswith("file://"):
        path = path[7:]
    path = re.sub(r":\d+(?::\d+)?$", "", path)
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "'\"":
        path = path[1:-1]
    if not path or "\x00" in path or "\\" in path:
        return None
    try:
        candidate_path = Path(path)
        if candidate_path.is_absolute():
            resolved = candidate_path.resolve()
            resolved.relative_to(worktree.resolve())
            candidate = resolved.relative_to(worktree.resolve()).as_posix()
        else:
            if ".." in candidate_path.parts:
                return None
            candidate = candidate_path.as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate in tracked_files:
        return candidate
    matches = sorted(
        tracked for tracked in tracked_files
        if tracked.endswith("/" + candidate)
    )
    return matches[0] if len(matches) == 1 else None


def _is_test_path(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    parts = PurePosixPath(path).parts
    forbidden = {
        ".git", ".venv", "venv", "node_modules", "dist", "build", "coverage",
        "test-results", "playwright-report",
    }
    if any(part in forbidden for part in parts):
        return False
    basename = parts[-1] if parts else ""
    return (
        "tests" in parts
        or "__tests__" in parts
        or (basename.startswith("test_") and basename.endswith((".py", ".pyi")))
        or basename.endswith((
            ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
            ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
        ))
    )


def _check_repair_scope_candidates(
    *,
    repo: Path,
    worktree: Path,
    tree_sha: str,
    evidence_dir: Path,
    evidence: EvidenceBundle,
    approved_mutable_scope: Sequence[str],
) -> list[str]:
    """Find approved, existing non-test files implicated by failure evidence."""

    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    text = _failure_log_text(_failed_check_logs(evidence_dir=evidence_dir, evidence=evidence))
    resolved = [
        _resolve_check_path_candidate(
            match.group(0), worktree=worktree, tracked_files=tracked
        )
        for match in _CHECK_SCOPE_PATH_RE.finditer(text)
    ]
    approved = set(approved_mutable_scope)
    return sorted({
        path for path in resolved
        if path is not None
        and path in approved
        and path in tracked
        and not _is_test_path(path)
    })


def _hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if isinstance(item, str)
        if item in _DIRECT_FAILURES
        or item in _HARD_FAILURE_CODES
        or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
        or any(item.startswith(prefix) for prefix in _HARD_FAILURE_PREFIXES)
    ]


def _check_repair_problem_context(
    evidence: EvidenceBundle,
    *,
    evidence_dir: Path,
    repo: Path,
    worktree: Path,
    tree_sha: str,
) -> str:
    """Render bounded root-cause excerpts plus explicit read-only file paths."""

    failures = soft_check_failures(evidence)
    logs = _failed_check_logs(evidence_dir=evidence_dir, evidence=evidence)
    failed_checks: list[dict[str, Any]] = []
    proof_text = _failure_log_text(logs)
    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    readable: set[str] = set()
    for match in _CHECK_SCOPE_PATH_RE.finditer(proof_text):
        path = _resolve_check_path_candidate(
            match.group(0), worktree=worktree,
            tracked_files=tracked,
        )
        if path is not None:
            readable.add(path)
    proof_budget = max(1, _CHECK_REPAIR_PROOF_BYTES // max(1, len(logs)))
    for raw_check in logs:
        name = raw_check.get("name")
        failure_proof = extract_failure_evidence(
            stdout_log=str(raw_check.get("stdout_log") or ""),
            stderr_log=str(raw_check.get("stderr_log") or ""),
            stdout_tail=str(raw_check.get("stdout_tail") or ""),
            stderr_tail=str(raw_check.get("stderr_tail") or ""),
            stdout_log_path=(raw_check.get("stdout_log_path")
                             if isinstance(raw_check.get("stdout_log_path"), str) else None),
            stderr_log_path=(raw_check.get("stderr_log_path")
                             if isinstance(raw_check.get("stderr_log_path"), str) else None),
            max_bytes=proof_budget,
        )
        check: dict[str, Any] = {
            "name": name,
            "exit_code": raw_check.get("exit_code"),
            "timed_out": bool(raw_check.get("timed_out", False)),
            "workspace_mutated": bool(raw_check.get("workspace_mutated", False)),
            "failure_proof": failure_proof,
        }
        failed_checks.append(check)
    return _json_text({
        "failure_ids": failures,
        "readable_failure_paths": sorted(readable)[:24],
        "checks": failed_checks,
    })


def _check_repair_prompt(
    *,
    spec: str,
    plan: TaskPlanV2,
    approved_contract_index: str,
    changed_files: str,
    evidence: EvidenceBundle,
    evidence_dir: Path,
    repo: Path,
    worktree: Path,
    effective_repair_scope: Sequence[str],
    candidate_identity: str = "",
    budget_bytes: int = 40_000,
    diagnostics_dir: str | Path | None = None,
) -> str:
    """Build the bounded prompt for one automatic check-repair pass."""

    del plan
    template = (_PROMPTS_DIR / "check_repair.txt").read_text(encoding="utf-8")
    failed_ids = soft_check_failures(evidence)
    problem_context = _check_repair_problem_context(
        evidence, evidence_dir=evidence_dir, repo=repo,
        worktree=worktree, tree_sha=evidence.staged_tree_sha or "",
    )
    try:
        context_payload = json.loads(problem_context)
    except (TypeError, json.JSONDecodeError):
        context_payload = {}
    readable_paths = context_payload.get("readable_failure_paths", [])
    read_set = "\n".join(
        f"- {path} :: named by the failing check output"
        for path in readable_paths if isinstance(path, str)
    ) or "Only the bounded failed-check evidence; no repository paths were identified."
    payload = build_check_repair_payload(
        spec=spec,
        failed_check_ids="\n".join(failed_ids) or "NONE",
        failed_check_evidence=problem_context,
        read_set=read_set,
        compact_contract_invariants=approved_contract_index,
        changed_files=changed_files,
        mutable_scope=_json_text(list(effective_repair_scope)),
        candidate_identity=candidate_identity,
        template=template,
        budget_bytes=budget_bytes,
    )
    if diagnostics_dir is not None:
        write_prompt_diagnostics(diagnostics_dir, payload)
    return payload.rendered


_EVIDENCE_SCOPE_SOURCE = "failure evidence within approved mutable scope"
_SCOPE_REQUEST_SOURCE = "explicit META SCOPE REQUEST v1"
_HUMAN_SCOPE_SOURCE = "human-approved mutable scope"
_CYCLE_SCOPE_SOURCE = "cycle mutable scope"



def _read_check_repair_scope(
    directory: Path,
    *,
    fallback_base: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
) -> CheckRepairScope:
    payload = _read_json_artifact(directory / "scope.json", 64 * 1024)
    approved = tuple(sorted(set(fallback_base)))
    if not isinstance(payload, dict):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    version = payload.get("schema_version")
    if version == 3:
        raw_approved = payload.get("approved_mutable_scope")
        raw_initial = payload.get("initial_repair_scope")
        raw_added = payload.get("added_paths")
        raw_effective = payload.get("effective_repair_scope")
    else:
        raise ResumeIntegrityError("check-repair scope artifact has an unsupported schema")
    if not all(isinstance(value, list) for value in (raw_approved, raw_initial, raw_added, raw_effective)):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    if any(
        not isinstance(path, str)
        for paths in (raw_approved, raw_initial, raw_added, raw_effective)
        for path in paths
    ):
        raise ResumeIntegrityError("check-repair scope artifact contains invalid paths")
    if any(
        not path or path.startswith("/") or "\\" in path
        or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
        for paths in (raw_approved, raw_initial, raw_added, raw_effective) for path in paths
    ):
        raise ResumeIntegrityError("check-repair scope artifact contains unsafe paths")
    parsed_approved = tuple(sorted(set(raw_approved)))
    parsed_initial = tuple(sorted(set(raw_initial)))
    parsed_added = tuple(sorted(set(raw_added)))
    parsed_effective = tuple(sorted(set(raw_effective)))
    if (
        raw_approved != list(parsed_approved)
        or raw_initial != list(parsed_initial)
        or raw_added != list(parsed_added)
        or raw_effective != list(parsed_effective)
    ):
        raise ResumeIntegrityError("check-repair scope artifact is not canonical")
    if (
        parsed_approved != approved
        or not set(parsed_initial).issubset(parsed_approved)
        or not set(parsed_added).issubset(parsed_approved)
        or set(parsed_initial) & set(parsed_added)
        or parsed_effective != tuple(sorted(set(parsed_initial) | set(parsed_added)))
    ):
        raise ResumeIntegrityError("check-repair scope artifact does not match its approved envelope")
    policy = payload.get("policy")
    bound = payload.get("bound")
    source = payload.get("source")
    if policy != policy_config.policy or bound != policy_config.max_added_paths or not isinstance(source, str):
        raise ResumeIntegrityError("check-repair scope policy changed")
    valid_sources = {
        _HUMAN_SCOPE_SOURCE, _EVIDENCE_SCOPE_SOURCE, _SCOPE_REQUEST_SOURCE,
    }
    if not isinstance(source, str) or source not in valid_sources:
        raise ResumeIntegrityError("check-repair scope artifact has invalid provenance")
    if parsed_added and source != _SCOPE_REQUEST_SOURCE:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    if not parsed_added and source not in {_HUMAN_SCOPE_SOURCE, _EVIDENCE_SCOPE_SOURCE}:
        raise ResumeIntegrityError("check-repair initial scope has an invalid provenance")
    if len(parsed_added) > policy_config.max_added_paths:
        raise ResumeIntegrityError("check-repair scope bound was exceeded")
    return CheckRepairScope(
        approved_mutable_scope=parsed_approved,
        initial_repair_scope=parsed_initial,
        added_paths=parsed_added,
        effective_repair_scope=parsed_effective,
        policy=policy,
        bound=bound,
        source=source,
    )


def mutable_scope_sha256(paths: Sequence[str]) -> str:
    """Hash the canonical, sorted JSON representation of a mutable scope."""

    canonical = tuple(sorted(set(paths)))
    return hashlib.sha256(_json_text(list(canonical)).encode("utf-8")).hexdigest()


def gate_mutable_authority(
    run_dir: Path,
    cycle: int,
    stage: GateStage | str,
    *,
    base_paths: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
    through_attempt: int | None = None,
    require_attempt_records: bool = False,
) -> GateMutableAuthority:
    """Rebuild and validate the exact mutation authority of one gate episode."""

    base = tuple(sorted(set(base_paths)))
    root = check_repair_attempts_dir(run_dir, cycle, stage)
    if not root.is_dir():
        return GateMutableAuthority(
            base_paths=base,
            added_paths=(),
            effective_paths=base,
            source=_CYCLE_SCOPE_SOURCE,
            sha256=mutable_scope_sha256(base),
            initial_paths=(),
        )

    if through_attempt is not None and (
        isinstance(through_attempt, bool)
        or not isinstance(through_attempt, int)
        or through_attempt < 0
    ):
        raise ResumeIntegrityError("check-repair scope attempt bound is invalid")
    directories = sorted(
        (
            path for path in root.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (through_attempt is None or int(path.name) <= through_attempt)
        ),
        key=lambda path: int(path.name),
    )
    if not directories:
        if through_attempt:
            raise ResumeIntegrityError("check-repair scope attempts are missing")
        return _with_ladder_expansion(GateMutableAuthority(
            base_paths=base,
            added_paths=(),
            effective_paths=base,
            source=_CYCLE_SCOPE_SOURCE,
            sha256=mutable_scope_sha256(base),
            initial_paths=(),
        ), run_dir=run_dir, cycle=cycle, stage=stage, through_attempt=through_attempt,
            base=base, policy_config=policy_config)
    scopes: list[CheckRepairScope] = []
    for expected, directory in enumerate(directories, start=1):
        if int(directory.name) != expected:
            raise ResumeIntegrityError("check-repair scope attempts are not contiguous")
        if not (directory / "scope.json").is_file():
            raise ResumeIntegrityError("check-repair scope attempt artifacts are incomplete")
        scope = _read_check_repair_scope(
            directory, fallback_base=base, policy_config=policy_config,
        )
        if require_attempt_records:
            attempt = _read_json_artifact(directory / "attempt.json", 128 * 1024)
            if (
                not isinstance(attempt, dict)
                or attempt.get("number") != expected
                or attempt.get("mutable_scope") != list(scope.effective_repair_scope)
            ):
                raise ResumeIntegrityError("check-repair attempt is not bound to its scope")
        if scopes and not set(scopes[-1].added_paths).issubset(scope.added_paths):
            raise ResumeIntegrityError("check-repair scope additions are not cumulative")
        if scopes and scopes[-1].initial_repair_scope != scope.initial_repair_scope:
            raise ResumeIntegrityError("check-repair initial scope changed between attempts")
        scopes.append(scope)
    if through_attempt is not None and len(scopes) != through_attempt:
        raise ResumeIntegrityError("check-repair scope attempts are not contiguous")
    final = scopes[-1]
    return _with_ladder_expansion(GateMutableAuthority(
        base_paths=final.approved_mutable_scope,
        added_paths=final.added_paths,
        effective_paths=final.effective_repair_scope,
        source=final.source,
        sha256=mutable_scope_sha256(final.effective_repair_scope),
        initial_paths=final.initial_repair_scope,
    ), run_dir=run_dir, cycle=cycle, stage=stage, through_attempt=through_attempt,
        base=base, policy_config=policy_config)


@dataclasses.dataclass(frozen=True)
class CheckRepairCoordinator:
    """Owns the bounded mutable-scope decision of one check-repair attempt.

    The operator's repair-scope policy is the only state it holds, injected
    explicitly: it never sees the ``Orchestrator`` and never widens a scope
    beyond the injected policy's bound.
    """

    repair_scope_policy: EffectiveRepairScopePolicy

    def resolve_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_sha: str,
        evidence_dir: Path,
        evidence: EvidenceBundle,
        approved_mutable_scope: Sequence[str],
        previous: CheckRepairScope | None = None,
    ) -> CheckRepairScope:
        """The scope of the next attempt; earlier attempts' paths are kept."""

        approved = tuple(sorted(set(approved_mutable_scope)))
        if previous is not None and previous.approved_mutable_scope != approved:
            raise ResumeIntegrityError("check-repair approved mutable scope changed")
        if previous is not None:
            initial = previous.initial_repair_scope
        else:
            initial_candidates = _check_repair_scope_candidates(
                repo=repo, worktree=worktree, tree_sha=tree_sha,
                evidence_dir=evidence_dir, evidence=evidence,
                approved_mutable_scope=approved,
            )
            # When output names no implicated source file, changed paths in
            # this gate's evidence provide a narrow fallback. A traceback
            # match always takes priority and keeps unrelated cycle changes
            # out of the worker's initial WRITE_SET.
            if not initial_candidates:
                tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
                changed_candidates = sorted({
                    path for path in evidence.changed_files
                    if path in approved and path in tracked and not _is_test_path(path)
                })
                if len(changed_candidates) <= _MAX_CHANGED_PATH_FALLBACK:
                    initial_candidates = changed_candidates
            initial = tuple(initial_candidates)
        added = set(previous.added_paths) if previous is not None else set()
        policy = self.repair_scope_policy
        if len(added) > policy.max_added_paths:
            raise ResumeIntegrityError("check-repair scope bound was exceeded")
        return CheckRepairScope(
            approved_mutable_scope=approved,
            initial_repair_scope=tuple(sorted(initial)),
            added_paths=tuple(sorted(added)),
            effective_repair_scope=tuple(sorted(set(initial) | added)),
            policy=policy.policy,
            bound=policy.max_added_paths,
            source=_SCOPE_REQUEST_SOURCE if added else (
                _EVIDENCE_SCOPE_SOURCE if initial else _HUMAN_SCOPE_SOURCE
            ),
        )


# -- the deterministic-gate recovery ladder -----------------------------------
#
# A red deterministic gate walks one ordered ladder of *distinct* strategies
# before any operator wait: the bounded targeted repair pass, one bounded
# evidence-proven scope expansion, a contract replan of the evidence-proven
# responsible step, a replan of the cycle, one final pass under the configured
# executor-fallback authority and only then the operator.  Every step is
# identified by the exact candidate tree, the failed check set, the stage and
# the strategy, and the durable ledger below never proposes the same strategy
# twice for those facts: a new tree or a new failed-check set opens a new
# progression.
#
# The ladder owns no authority of its own.  A repair pass is bounded by the
# frozen ``max_check_repair_attempts`` budget, an expansion is bounded by the
# operator-approved plan scope and the run's repair-scope policy, and every
# replan rewrites one approved step contract through the durable contract
# repair transaction, inside the scope the operator approved.  Nothing here can
# grant a model a scope the operator did not approve.

_LADDER_ARTIFACT = "ladder.json"
_EXPANSION_ARTIFACT = "scope-expansion.json"
_LADDER_SCHEMA_VERSION = 1
_LADDER_CODE = "CHECK_FAILED"
_MAX_LADDER_ENTRIES = 64
_LADDER_STATES = frozenset({"running", "done"})
_SAFE_PATH_PARTS = frozenset({"", ".", ".."})
# Strategies the ladder consumes at most once per gate episode.
_EPISODE_STRATEGIES = frozenset({
    RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
})


@dataclasses.dataclass(frozen=True)
class _LadderEntry:
    """One durably consumed ladder step of one red gate episode."""

    strategy: RecoveryStrategy
    state: str
    tree: str
    failed_check_ids: tuple[str, ...]
    repair_attempt: int | None = None
    step_indices: tuple[int, ...] = ()
    added_paths: tuple[str, ...] = ()
    tree_after: str = ""


@dataclasses.dataclass(frozen=True)
class _LadderLedger:
    """The frozen facts and the consumed steps of one gate episode."""

    proof_required: bool
    fallback_executor_available: bool
    entries: tuple[_LadderEntry, ...] = ()


def _ladder_path(run_dir: Path, cycle: int, stage: GateStage) -> Path:
    return check_repair_dir(run_dir, cycle, stage) / _LADDER_ARTIFACT


def _canonical_ladder_paths(paths: Any, *, what: str) -> tuple[str, ...]:
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise ResumeIntegrityError(f"{what} contains invalid paths")
    canonical = tuple(sorted(set(paths)))
    if list(canonical) != paths:
        raise ResumeIntegrityError(f"{what} is not canonical")
    for item in canonical:
        if (
            not item or item.startswith("/") or "\\" in item
            or any(part in _SAFE_PATH_PARTS for part in PurePosixPath(item).parts)
        ):
            raise ResumeIntegrityError(f"{what} contains unsafe paths")
    return canonical


def _read_ladder(path: Path) -> _LadderLedger | None:
    """Read and validate the ladder ledger; a malformed artifact fails closed."""

    payload = _read_json_artifact(path, 256 * 1024)
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _LADDER_SCHEMA_VERSION:
        raise ResumeIntegrityError("gate recovery ladder artifact is malformed")
    proof = payload.get("proof_required")
    fallback = payload.get("fallback_executor_available")
    if not isinstance(proof, bool) or not isinstance(fallback, bool):
        raise ResumeIntegrityError("gate recovery ladder facts are malformed")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_LADDER_ENTRIES:
        raise ResumeIntegrityError("gate recovery ladder entries are malformed")
    entries: list[_LadderEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ResumeIntegrityError("gate recovery ladder entry is malformed")
        try:
            strategy = RecoveryStrategy(item.get("strategy"))
        except ValueError as exc:
            raise ResumeIntegrityError("gate recovery ladder strategy is unknown") from exc
        state = item.get("state")
        tree = item.get("tree")
        failed = item.get("failed_check_ids")
        attempt = item.get("repair_attempt")
        indices = item.get("step_indices")
        if state not in _LADDER_STATES or not isinstance(tree, str) or len(tree) > 128:
            raise ResumeIntegrityError("gate recovery ladder entry is malformed")
        if not isinstance(failed, list) or any(not isinstance(name, str) or not name for name in failed):
            raise ResumeIntegrityError("gate recovery ladder entry has invalid failed checks")
        if attempt is not None and (
            isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
        ):
            raise ResumeIntegrityError("gate recovery ladder entry has an invalid repair attempt")
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ):
            raise ResumeIntegrityError("gate recovery ladder entry has invalid step indices")
        added = _canonical_ladder_paths(item.get("added_paths"), what="gate recovery ladder expansion")
        tree_after = item.get("tree_after")
        if not isinstance(tree_after, str) or len(tree_after) > 128:
            raise ResumeIntegrityError("gate recovery ladder entry has an invalid produced tree")
        entries.append(_LadderEntry(
            strategy, state, tree, tuple(failed), attempt, tuple(indices), added, tree_after,
        ))
    return _LadderLedger(proof, fallback, tuple(entries))


def _write_ladder(path: Path, ledger: _LadderLedger) -> None:
    atomic_write_text(path, _json_text({
        "schema_version": _LADDER_SCHEMA_VERSION,
        "proof_required": ledger.proof_required,
        "fallback_executor_available": ledger.fallback_executor_available,
        "entries": [
            {
                "strategy": entry.strategy.value, "state": entry.state, "tree": entry.tree,
                "failed_check_ids": list(entry.failed_check_ids),
                "repair_attempt": entry.repair_attempt,
                "step_indices": list(entry.step_indices),
                "added_paths": list(entry.added_paths),
                "tree_after": entry.tree_after,
            }
            for entry in ledger.entries
        ],
    }))


def _pending_scope_expansion(
    run_dir: Path, cycle: int, stage: GateStage, attempt: int, *,
    base: tuple[str, ...], policy_config: EffectiveRepairScopePolicy,
) -> tuple[str, ...]:
    """The ladder expansion authorized for one not-yet-recorded repair pass."""

    directory = check_repair_attempt_dir(run_dir, cycle, stage, attempt)
    if (directory / "scope.json").is_file():
        # The pass is recorded: its own scope artifact is authoritative.
        return ()
    payload = _read_json_artifact(directory / _EXPANSION_ARTIFACT, 64 * 1024)
    if payload is None:
        return ()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ResumeIntegrityError("gate recovery scope expansion artifact is malformed")
    if payload.get("attempt") != attempt:
        raise ResumeIntegrityError("gate recovery scope expansion belongs to another attempt")
    if (
        payload.get("policy") != policy_config.policy
        or payload.get("bound") != policy_config.max_added_paths
    ):
        raise ResumeIntegrityError("gate recovery scope expansion policy changed")
    failed = payload.get("failed_check_ids")
    if not isinstance(failed, list) or any(not isinstance(name, str) or not name for name in failed):
        raise ResumeIntegrityError("gate recovery scope expansion has invalid failed checks")
    added = _canonical_ladder_paths(payload.get("added_paths"), what="gate recovery scope expansion")
    if not added:
        return ()
    if len(added) > policy_config.max_added_paths:
        raise ResumeIntegrityError("gate recovery scope expansion exceeds the configured bound")
    if not set(added).issubset(set(base)):
        raise ResumeIntegrityError("gate recovery scope expansion exceeds the approved envelope")
    return added


def _with_ladder_expansion(
    authority: GateMutableAuthority, *,
    run_dir: Path, cycle: int, stage: GateStage, through_attempt: int | None,
    base: tuple[str, ...], policy_config: EffectiveRepairScopePolicy,
) -> GateMutableAuthority:
    """Fold the ladder expansion of the pending repair pass into its authority."""

    if through_attempt is None:
        return authority
    added = _pending_scope_expansion(
        run_dir, cycle, stage, through_attempt + 1, base=base, policy_config=policy_config,
    )
    if not added:
        return authority
    effective = tuple(sorted(set(authority.effective_paths) | set(added)))
    return GateMutableAuthority(
        base_paths=authority.base_paths,
        added_paths=tuple(sorted(set(authority.added_paths) | set(added))),
        effective_paths=effective,
        source=_SCOPE_REQUEST_SOURCE,
        sha256=mutable_scope_sha256(effective),
        initial_paths=authority.initial_paths,
    )


def _recorded_effective_scope(run_dir: Path, cycle: int, stage: GateStage) -> frozenset[str]:
    """The effective repair scope of the latest recorded pass, best effort.

    The value only narrows the ladder's own expansion candidates: the
    authority of every pass is re-derived and validated by the attempt
    machinery, never by this reader.
    """

    root = check_repair_attempts_dir(run_dir, cycle, stage)
    if not root.is_dir():
        return frozenset()
    found: frozenset[str] = frozenset()
    for directory in sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        payload = _read_json_artifact(directory / "scope.json", 64 * 1024)
        if not isinstance(payload, dict):
            continue
        raw = payload.get("effective_repair_scope", payload.get("effective_mutable_scope"))
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            found = frozenset(raw)
    return found


def _red_gate_identity(evidence: EvidenceBundle) -> tuple[str, tuple[str, ...]]:
    """The exact candidate tree and repairable failed-check set of one red gate."""

    tree = evidence.staged_tree_sha
    if not isinstance(tree, str) or not tree:
        raise ResumeIntegrityError("the red gate evidence has no candidate tree")
    failed = tuple(
        item.split(":", 1)[1] for item in soft_check_failures(evidence) if ":" in item
    )
    if not failed:
        raise ResumeIntegrityError("the red gate evidence has no repairable check failure")
    return tree, failed


def _approved_expansion_scope(
    *, repo: Path, worktree: Path, tree_sha: str, evidence_dir: Path,
    evidence: EvidenceBundle, approved: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The evidence-named approved paths, and the pass scope they imply.

    The initial repair scope mirrors the coordinator exactly: the failure
    evidence's own paths, or the changed approved paths when the output names
    none.  The implicated set is the union of both; the difference is the proof
    that the first pass could not reach the failing check.
    """

    approved_set = set(approved)
    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    initial = tuple(_check_repair_scope_candidates(
        repo=repo, worktree=worktree, tree_sha=tree_sha, evidence_dir=evidence_dir,
        evidence=evidence, approved_mutable_scope=approved,
    ))
    changed = sorted({
        path for path in evidence.changed_files
        if path in approved_set and path in tracked and not _is_test_path(path)
    })
    if not initial and len(changed) <= _MAX_CHANGED_PATH_FALLBACK:
        initial = tuple(changed)
    implicated = sorted(set(initial) | set(changed))
    return tuple(implicated), initial


# The bounded evidence one red gate hands to a step contract replan.  Every
# field is a deterministic fact of this gate episode: the failed check
# identities, the bounded proof the check itself produced, the implicated
# approved paths, the diff the step produced between its own boundary tree and
# the red tree, the Git facts of its declared paths and the bounded summaries
# of its earlier repairs.  Full logs, cycle history and worker narration never
# cross this boundary.
_REPLAN_EVIDENCE_BYTES = 16 * 1024
_REPLAN_DIFF_FILE_BYTES = 2048
_MAX_REPLAN_DIFF_PATHS = 8
_MAX_REPLAN_PATH_FACTS = 16
_MAX_REPLAN_PREVIOUS_REPAIRS = 2
_REPLAN_ORIGIN = "deterministic_gate_replan"


def _bounded_proof(value: str, limit: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="replace") + "\n[TRUNCATED]"


def _file_in_tree(repo: Path, tree_sha: str, path: str) -> str:
    """Read one path from a tree object; a missing path is empty, never an error."""

    try:
        return read_file_at_commit(repo, commit_sha=tree_sha, relative_path=path)
    except (GitError, OSError, UnicodeError):
        return ""


@dataclasses.dataclass(frozen=True)
class ReplanProblem:
    """The durable red-gate facts one contract replan is opened on."""

    red_tree: str
    failed_check_ids: tuple[str, ...]
    failure_proof: str
    implicated_paths: tuple[str, ...]


def replan_problem(
    *, evidence: EvidenceBundle, evidence_dir: Path, repo: Path, worktree: Path,
    approved_scope: Sequence[str],
) -> ReplanProblem:
    """The bounded failure facts of one red gate episode, rebuilt on every resume.

    Everything is read from the durable gate evidence, its archived check logs
    and Git objects, never from the current worktree content, so a resume
    rebuilds exactly the same facts.
    """

    tree, failed = _red_gate_identity(evidence)
    try:
        context = json.loads(_check_repair_problem_context(
            evidence, evidence_dir=evidence_dir, repo=repo, worktree=worktree,
            tree_sha=tree,
        ))
    except (TypeError, ValueError):
        context = {}
    checks = context.get("checks") if isinstance(context, dict) else None
    proofs = [
        str(item.get("failure_proof"))
        for item in (checks or ())
        if isinstance(item, Mapping) and str(item.get("failure_proof") or "").strip()
    ]
    implicated, _initial = _approved_expansion_scope(
        repo=repo, worktree=worktree, tree_sha=tree, evidence_dir=evidence_dir,
        evidence=evidence, approved=approved_scope,
    )
    return ReplanProblem(
        red_tree=tree, failed_check_ids=failed,
        failure_proof=_bounded_proof(proofs[0], 4000) if proofs else "",
        implicated_paths=tuple(implicated),
    )


def check_failure_proofs(
    *, evidence: EvidenceBundle, evidence_dir: Path, repo: Path, worktree: Path,
    tree_sha: str,
) -> tuple[tuple[str, str], ...]:
    """The first useful proof of every failed check of one red gate.

    Read from the durable gate evidence, its archived check logs and Git
    objects, never from the worktree content, so a resume rebuilds exactly the
    same proofs.  A plan that re-decomposes a whole cycle needs them all, not
    only the first: each failing check is one fact the new split must answer.
    """

    try:
        context = json.loads(_check_repair_problem_context(
            evidence, evidence_dir=evidence_dir, repo=repo, worktree=worktree,
            tree_sha=tree_sha,
        ))
    except (TypeError, ValueError):
        return ()
    checks = context.get("checks") if isinstance(context, dict) else None
    return tuple(
        (str(item["name"]), str(item["failure_proof"]))
        for item in (checks or ())
        if isinstance(item, Mapping)
        and isinstance(item.get("name"), str)
        and str(item.get("failure_proof") or "").strip()
    )


def consumed_ladder_strategies(
    run_dir: str | Path, cycle: int, stage: GateStage | str,
) -> tuple[str, ...]:
    """The distinct ladder rungs one gate episode already consumed, in order.

    Read from the episode's own durable ledger, so the same episode always
    reports the same strategies to a planner and to a resume alike.
    """

    ledger = _read_ladder(_ladder_path(Path(run_dir), cycle, GateStage(stage)))
    return tuple(entry.strategy.value for entry in (ledger.entries if ledger else ()))


def replan_mismatch(
    *, problem: ReplanProblem, step_id: str, cycle: int, stage: GateStage,
    anchor_tree: str,
) -> str:
    """The durable identity of one red-gate contract replan.

    The text is the transaction's mismatch identity: it names the exact gate
    episode, the step and the two trees, so a resume finds the slot it opened
    instead of opening a second one.
    """

    return _json_text({
        "origin": _REPLAN_ORIGIN,
        "cycle": cycle,
        "stage": GateStage(stage).value,
        "step_id": step_id,
        "candidate_tree_sha": problem.red_tree,
        "step_boundary_tree_sha": anchor_tree,
        "failed_check_ids": list(problem.failed_check_ids),
        "implicated_paths": list(problem.implicated_paths),
        "note": (
            "The deterministic gate stayed red after this approved step ran. "
            "Repair this step's own contract so its work can satisfy the "
            "failing check; the failure evidence is authoritative."
        ),
    })


def replan_slot_origin(mismatch: str) -> Mapping[str, Any] | None:
    """The parsed origin facts of a contract repair a red gate opened."""

    try:
        payload = json.loads(mismatch)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("origin") != _REPLAN_ORIGIN:
        return None
    return payload


def replan_failure_evidence(
    *, problem: ReplanProblem, repo: Path, step: Any, anchor_tree: str,
    previous_repairs: Sequence[Mapping[str, Any]] = (),
) -> str:
    """The bounded red-gate evidence of one responsible-step contract replan.

    It carries the failed check identities, the first useful failure proof,
    the implicated approved paths, the diff the step produced between its own
    accepted boundary tree and the red candidate tree, the Git facts of its
    declared paths and the bounded summaries of its earlier repairs.  Full
    logs, cycle history and worker narration never cross this boundary.
    """

    red_tree = problem.red_tree
    step_paths = tuple(sorted({
        *getattr(step, "write_set", ()), *getattr(step, "create_set", ()),
        *getattr(step, "delete_set", ()),
    }))
    try:
        changed = changed_paths_between_trees(repo, anchor_tree, red_tree)
    except GitError:
        changed = ()
    diff_paths = sorted(set(changed) & set(step_paths))[:_MAX_REPLAN_DIFF_PATHS]
    return _bounded_proof(_json_text({
        "failed_check_ids": list(problem.failed_check_ids),
        "candidate_tree_sha": red_tree,
        "step_boundary_tree_sha": anchor_tree,
        "first_failure_proof": problem.failure_proof,
        "implicated_paths": list(problem.implicated_paths),
        "step_paths": [
            {
                "path": path,
                "in_step_boundary_tree": path_exists_in_tree(repo, anchor_tree, path),
                "in_candidate_tree": path_exists_in_tree(repo, red_tree, path),
            }
            for path in step_paths[:_MAX_REPLAN_PATH_FACTS]
        ],
        "step_diff": {
            "changed_paths": diff_paths,
            "files": [
                {
                    "path": path,
                    "before": _bounded_proof(
                        _file_in_tree(repo, anchor_tree, path), _REPLAN_DIFF_FILE_BYTES,
                    ),
                    "after": _bounded_proof(
                        _file_in_tree(repo, red_tree, path), _REPLAN_DIFF_FILE_BYTES,
                    ),
                }
                for path in diff_paths
            ],
        },
        "previous_repairs": [
            dict(item) for item in previous_repairs[:_MAX_REPLAN_PREVIOUS_REPAIRS]
        ],
    }), _REPLAN_EVIDENCE_BYTES)


@dataclasses.dataclass(frozen=True)
class CheckRepairLadder:
    """The deterministic-gate ladder of one run.

    It is the single object :class:`PipelineV2Operations` references for the
    red-gate ladder: the machine asks which distinct strategy to try next and
    reports the step as started and finished.  The durable ledger, the frozen
    episode facts and the bounded expansion live here; no model, no planner and
    no worker can widen what the operator already approved.

    ``replan_steps`` is injected by the run: for one ``REPLAN_STEP`` rung it
    identifies the responsible step, produces a new bounded contract from the
    red-gate evidence through the existing durable contract-repair
    transaction, re-executes that step under its new validated authority with
    its necessary descendants, and returns the tree it produced.

    ``replan_cycles`` is the last rung: once a step's contract replan is spent
    too, the whole decomposition of the cycle is replanned through the durable
    check-replan planning transaction and the new cycle it opens is returned.

    Both raise :class:`RecoveryStepUnavailable` when the rung is not
    applicable for these exact facts, so the ladder advances instead of
    looping.
    """

    replan_steps: Callable[..., str] | None = None
    replan_cycles: Callable[..., CyclePlan] | None = None

    def gate_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        evidence: EvidenceBundle, repair_attempt: int, repair_budget: int,
    ) -> GateRecoveryStep:
        """The next distinct ladder step of this red gate, or its terminal."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, failed = _red_gate_identity(evidence)
        fallback_available = bool(ctx.selection.check_repair_fallbacks)
        proof = self._expansion_paths(ctx, cycle_plan, stage_value, evidence)
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            ledger = _LadderLedger(
                proof_required=bool(proof), fallback_executor_available=fallback_available,
            )
        elif ledger.fallback_executor_available != fallback_available:
            raise ResumeIntegrityError(
                "the gate recovery ladder does not match the frozen executor authority"
            )
        elif proof and not ledger.proof_required:
            # The failure evidence has proven that an approved path outside the
            # frozen repair scope is implicated: the proof is a durable fact of
            # this gate episode, never a model decision.
            ledger = dataclasses.replace(ledger, proof_required=True)
            _write_ladder(path, ledger)
        pending = next(
            (entry for entry in reversed(ledger.entries) if entry.state == "running"), None,
        )
        if pending is not None:
            # The strategy of this round is already durable: resume it as such.
            return GateRecoveryStep(
                pending.strategy, tree=tree, failed_check_ids=failed,
                repair_attempt=pending.repair_attempt, step_indices=pending.step_indices,
                added_paths=pending.added_paths, consumed=self._trail(ledger),
            )
        facts = self._facts(ledger, tree=tree, failed=failed, number=number, stage=stage_value)
        progression = RecoveryProgression(self._consumed(ledger, number=number, stage=stage_value))
        for strategy in recovery_ladder(FailureClass.CORRECTNESS):
            if strategy.terminal or not self._policy_admits(strategy, facts):
                continue
            if strategy is RecoveryStrategy.REPAIR_TARGETED and proof:
                # The failure evidence proves an approved path the frozen
                # repair scope cannot reach: the bounded expansion is the next
                # distinct strategy, never another pass on the narrow scope.
                continue
            recorded = strategy in _EPISODE_STRATEGIES and any(
                entry.strategy is strategy for entry in ledger.entries
            )
            # A cycle rung whose durable answer already exists is the *resume*
            # of that one rung, never a second answer: the plan it produced is
            # recovered and judged, so neither the episode rule nor the
            # progression refuses it -- while any other recorded rung stays
            # spent for these exact facts.
            resuming = bool(
                recorded and strategy is RecoveryStrategy.REPLAN_CYCLE
                and self._cycle_replan_open(ctx, cycle_plan, tree, failed)
            )
            if recorded and not resuming:
                continue
            step = self._admit(
                strategy, ctx=ctx, cycle_plan=cycle_plan, stage=stage_value,
                evidence=evidence, ledger=ledger, tree=tree, failed=failed, proof=proof,
                repair_attempt=repair_attempt, repair_budget=repair_budget,
            )
            if step is None:
                # Deterministically inapplicable for these exact facts:
                # advance the ladder without executing anything.
                continue
            if not resuming and progression.is_consumed(progression.fingerprint(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=strategy,
            )):
                # This exact strategy already ran for this tree and failure.
                continue
            return step
        return GateRecoveryStep(
            terminal_strategy(FailureClass.CORRECTNESS, _LADDER_CODE), tree=tree,
            failed_check_ids=failed, consumed=self._trail(ledger), exhausted=True,
        )

    @staticmethod
    def _policy_admits(strategy: RecoveryStrategy, facts: RecoveryFacts) -> bool:
        """Whether the ladder policy itself admits this rung for these facts."""

        return strategy in admitted_strategies(FailureClass.CORRECTNESS, facts)

    def begin_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> None:
        """Consume one ladder step durably before anything runs for it."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, failed = _red_gate_identity(evidence)
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            ledger = _LadderLedger(
                proof_required=bool(self._expansion_paths(
                    ctx, cycle_plan, stage_value, evidence,
                )),
                fallback_executor_available=bool(ctx.selection.check_repair_fallbacks),
            )
        entry = _LadderEntry(
            step.strategy, "running", tree, failed,
            step.repair_attempt, step.step_indices, step.added_paths,
        )
        def same(item: _LadderEntry) -> bool:
            return (
                item.strategy is entry.strategy and item.tree == entry.tree
                and item.failed_check_ids == entry.failed_check_ids
                and item.repair_attempt == entry.repair_attempt
                and item.step_indices == entry.step_indices
            )

        existing = next((item for item in ledger.entries if same(item)), None)
        if existing is None:
            if len(ledger.entries) >= _MAX_LADDER_ENTRIES:
                raise ResumeIntegrityError("the gate recovery ladder is full")
            ledger = dataclasses.replace(ledger, entries=(*ledger.entries, entry))
        elif existing.state != "running":
            # A rung that opens a whole cycle is re-admitted after a crash
            # between its durable answer and that cycle: it is running again,
            # and the ledger is the one place that says so.
            ledger = dataclasses.replace(ledger, entries=tuple(
                entry if same(item) else item for item in ledger.entries
            ))
        _write_ladder(path, ledger)
        if (
            step.strategy is RecoveryStrategy.EXPAND_SCOPE
            and step.repair_attempt is not None and step.added_paths
        ):
            self._authorize_expansion(ctx, cycle_plan, stage_value, step, failed)

    def replan_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> str:
        """Rewrite and re-execute the responsible approved step of one rung.

        The ladder owns no scope of its own.  The rung must name exactly the
        evidence-proven responsible step and the descendants its rewritten
        commit invalidates, and the injected replan owns the durable contract
        repair, the new validated authority and the re-execution; the returned
        tree is what the next gate episode observes.  A rung whose facts no
        longer hold is refused without touching the repository.
        """

        if step.exhausted or step.is_repair_pass or not step.step_indices:
            raise RecoveryStepUnavailable(step.strategy, "the ladder step is not a replan rung")
        if step.strategy is not RecoveryStrategy.REPLAN_STEP:
            raise RecoveryStepUnavailable(
                step.strategy, "only a single responsible step is replanned for now",
            )
        if self.replan_steps is None:
            raise RecoveryStepUnavailable(
                step.strategy, "this run admits no contract replan of approved work",
            )
        stage_value = GateStage(stage)
        tree, _failed = _red_gate_identity(evidence)
        count = len(cycle_plan.plan.steps)
        first = step.step_indices[0]
        if step.step_indices != tuple(range(first, count)):
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not replan a responsible step and its descendants",
            )
        if self._responsible_step_index(ctx, cycle_plan, stage_value, evidence) != first:
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not name the evidence-proven responsible step",
            )
        produced = self.replan_steps(ctx, cycle_plan, stage_value, step, evidence)
        if not isinstance(produced, str) or not produced:
            raise ResumeIntegrityError("the gate recovery replan produced no candidate tree")
        return produced

    def replan_cycle(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> CyclePlan:
        """Re-decompose the whole cycle of one rung and open its new cycle.

        The rung is the last one of the episode: the bounded repair pass and
        the contract replan of the responsible step were both durably spent
        and the gate is still red, so the decomposition itself is what failed.
        The injected transaction produces the new plan from this gate's own
        bounded failure evidence, inside the approved envelope, and returns the
        durable plan of the cycle that executes it -- a plan whose authority is
        never the one already in force.  A rung these facts do not admit is
        refused before anything runs, so the ladder advances without looping.
        """

        if step.exhausted or step.is_repair_pass or step.step_indices:
            raise RecoveryStepUnavailable(
                step.strategy, "the ladder step is not a cycle replan",
            )
        if step.strategy is not RecoveryStrategy.REPLAN_CYCLE:
            raise RecoveryStepUnavailable(
                step.strategy, "only a whole cycle is re-decomposed by this rung",
            )
        if self.replan_cycles is None:
            raise RecoveryStepUnavailable(
                step.strategy, "this run admits no cycle replan of approved work",
            )
        tree, failed = _red_gate_identity(evidence)
        if step.tree != tree or step.failed_check_ids != failed:
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not describe the red gate evidence",
            )
        planned = self.replan_cycles(ctx, cycle_plan, GateStage(stage), step, evidence)
        if planned is None:
            raise RecoveryStepUnavailable(step.strategy, "the cycle replan produced no plan")
        return planned

    def finish_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle, tree_after: str,
    ) -> None:
        """Mark a consumed step as executed; the ledger itself never repeats it."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, _failed = _red_gate_identity(evidence)
        if not isinstance(tree_after, str) or len(tree_after) > 128:
            raise ResumeIntegrityError("the gate recovery ladder produced an invalid tree")
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            raise ResumeIntegrityError("the gate recovery ladder artifact is missing")
        entries: list[_LadderEntry] = []
        marked = False
        for entry in ledger.entries:
            if not marked and (
                entry.state == "running" and entry.strategy is step.strategy
                and entry.repair_attempt == step.repair_attempt and entry.tree == tree
            ):
                entries.append(dataclasses.replace(entry, state="done", tree_after=tree_after))
                marked = True
            else:
                entries.append(entry)
        if not marked:
            raise ResumeIntegrityError("the gate recovery ladder step is not pending")
        _write_ladder(path, dataclasses.replace(ledger, entries=tuple(entries)))

    # -- deterministic facts -------------------------------------------------

    @staticmethod
    def _facts(
        ledger: _LadderLedger, *, tree: str, failed: tuple[str, ...],
        number: int, stage: GateStage,
    ) -> RecoveryFacts:
        return RecoveryFacts(
            candidate_tree=tree,
            observed_facts=(
                ("cycle", f"{number:03d}"),
                ("failed_checks", ",".join(failed)),
                ("stage", stage.value),
            ),
            proof_required=ledger.proof_required,
            fallback_executor_available=ledger.fallback_executor_available,
            # The ladder's first step is a bounded repair pass; the frozen
            # attempt budget is enforced by this ladder, never by the policy.
            retry_allowed=True,
        )

    @classmethod
    def _consumed(
        cls, ledger: _LadderLedger, *, number: int, stage: GateStage,
    ) -> tuple[RecoveryFingerprint, ...]:
        return tuple(
            RecoveryFingerprint(
                entry.tree, FailureClass.CORRECTNESS,
                cls._facts(
                    ledger, tree=entry.tree, failed=entry.failed_check_ids,
                    number=number, stage=stage,
                ).stable_items(),
                entry.strategy,
            )
            for entry in ledger.entries
        )

    @staticmethod
    def _trail(ledger: _LadderLedger) -> tuple[RecoveryStrategy, ...]:
        return tuple(entry.strategy for entry in ledger.entries)

    @staticmethod
    def _cycle_replan_record(ctx: Any, cycle: int) -> dict[str, Any] | None:
        """The durable check-replan answer one cycle holds, if any."""

        record = _read_json_artifact(
            check_replan_dir(ctx.run_dir, cycle) / CHECK_REPLAN_PLAN_ARTIFACT
        )
        return record if isinstance(record, dict) else None

    @staticmethod
    def _cycle_replan_answers(
        record: Mapping[str, Any], tree: str, failed: tuple[str, ...], identity: str,
    ) -> bool:
        """Whether one durable record answers exactly these red-gate facts."""

        return (
            record.get("candidate_tree_sha") == tree
            and tuple(record.get("failed_check_ids") or ()) == tuple(sorted(failed))
            and record.get("plan_identity_before") == identity
        )

    @classmethod
    def _cycle_replan_consumed(
        cls, ctx: Any, cycle_plan: Any, tree: str, failed: tuple[str, ...],
    ) -> bool:
        """Whether these exact facts already opened a cycle replan.

        The fingerprint is this gate's candidate tree, its failed checks and
        the identity of the plan already in force: the same plan produced again
        for the same facts never opens a second cycle.  Every earlier answer is
        read from its own durable record, so a resume reaches the same decision.
        """

        identity = plan_identity(cycle_plan.plan)
        for earlier in range(2, cycle_plan.cycle.number + 1):
            record = cls._cycle_replan_record(ctx, earlier)
            if record is not None and cls._cycle_replan_answers(
                record, tree, failed, identity,
            ):
                return True
        return False

    @classmethod
    def _cycle_replan_open(
        cls, ctx: Any, cycle_plan: Any, tree: str, failed: tuple[str, ...],
    ) -> bool:
        """Whether this rung already produced the plan of the cycle it opens.

        The rung ends the gate episode and the cycle only starts afterwards, so
        a crash in between leaves a durable answer without a running ladder
        entry.  That answer is replayed -- recovered, never re-planned -- while
        one that re-decomposed nothing leaves the rung spent.
        """

        record = cls._cycle_replan_record(ctx, cycle_plan.cycle.number + 1)
        return record is not None and (
            record.get("plan_identity_after") != record.get("plan_identity_before")
            and cls._cycle_replan_answers(
                record, tree, failed, plan_identity(cycle_plan.plan),
            )
        )

    def _expansion_paths(
        self, ctx: Any, cycle_plan: Any, stage: GateStage, evidence: EvidenceBundle,
    ) -> tuple[str, ...]:
        """Bounded, evidence-proven additions to the repair scope."""

        tree, _failed = _red_gate_identity(evidence)
        approved_scope = tuple(cycle_plan.mutable_scope)
        if not approved_scope:
            return ()
        policy = effective_repair_scope_policy(ctx.options)
        _implicated, evidenced = _approved_expansion_scope(
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=tree,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            evidence=evidence, approved=approved_scope,
        )
        recorded = _recorded_effective_scope(ctx.run_dir, cycle_plan.cycle.number, stage)
        current = set(recorded) if recorded else set(evidenced)
        return tuple(sorted(set(evidenced) - current))[: policy.max_added_paths]

    @staticmethod
    def _implicated_paths(
        ctx: Any, cycle_plan: Any, stage: GateStage, evidence: EvidenceBundle,
    ) -> tuple[str, ...]:
        tree, _failed = _red_gate_identity(evidence)
        implicated, _initial = _approved_expansion_scope(
            repo=ctx.repo, worktree=ctx.info.worktree, tree_sha=tree,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            evidence=evidence, approved=tuple(cycle_plan.mutable_scope),
        )
        return implicated

    def _responsible_step_index(
        self, ctx: Any, cycle_plan: Any, stage: GateStage, evidence: EvidenceBundle,
    ) -> int | None:
        """The first approved step whose mutable paths the failure implicates."""

        implicated = set(self._implicated_paths(ctx, cycle_plan, stage, evidence))
        if not implicated:
            return None
        for index, step in enumerate(cycle_plan.plan.steps):
            if implicated & set((*step.write_set, *step.create_set, *step.delete_set)):
                return index
        return None

    def _admit(
        self, strategy: RecoveryStrategy, *, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: EvidenceBundle, ledger: _LadderLedger, tree: str,
        failed: tuple[str, ...], proof: tuple[str, ...],
        repair_attempt: int, repair_budget: int,
    ) -> GateRecoveryStep | None:
        """Materialize the step, or refuse it for these exact facts.

        A refusal is not an outcome: the rung consumes no attempt, produces no
        report and leaves the ladder free to propose the next distinct
        strategy.  The frozen ``repair_budget`` is consulted here and only for
        the rungs that execute a check-repair worker pass; the replan rungs are
        never bounded by it.
        """

        trail = self._trail(ledger)
        common = {"tree": tree, "failed_check_ids": failed, "consumed": trail}
        if strategy in {
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
            RecoveryStrategy.FALLBACK_EXECUTOR,
        } and any(entry.repair_attempt == repair_attempt for entry in ledger.entries):
            # This bounded pass number is already durably consumed: one pass is
            # never proposed, executed or counted twice for these facts.
            return None
        if strategy is RecoveryStrategy.FALLBACK_EXECUTOR:
            if not ledger.fallback_executor_available:
                return None
            if repair_attempt > repair_budget:
                return None
            return GateRecoveryStep(strategy, repair_attempt=repair_attempt, **common)
        if strategy in {RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE}:
            if repair_attempt > repair_budget:
                return None
            if strategy is RecoveryStrategy.REPAIR_TARGETED:
                return GateRecoveryStep(strategy, repair_attempt=repair_attempt, **common)
            if not proof:
                return None
            return GateRecoveryStep(
                strategy, repair_attempt=repair_attempt, added_paths=proof, **common,
            )
        if strategy in {RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE}:
            if strategy is RecoveryStrategy.REPLAN_CYCLE:
                if self.replan_cycles is None:
                    return None
                if self._cycle_replan_consumed(ctx, cycle_plan, tree, failed):
                    # These exact facts already re-decomposed a cycle: the same
                    # failure under the same plan never opens a second one.
                    return None
                return GateRecoveryStep(strategy, **common)
            if self.replan_steps is None:
                return None
            index = self._responsible_step_index(ctx, cycle_plan, stage, evidence)
            if index is None:
                # No step is proven responsible by the failure evidence: a
                # replan would guess, so this rung is refused for these exact
                # facts and the ladder proposes its next distinct strategy.
                return None
            # A replan rewrites the responsible approved step and re-executes
            # it with the descendants its rewritten commit invalidates.
            return GateRecoveryStep(
                strategy,
                step_indices=tuple(range(index, len(cycle_plan.plan.steps))), **common,
            )
        return None

    @staticmethod
    def _authorize_expansion(
        ctx: Any, cycle_plan: Any, stage: GateStage,
        step: GateRecoveryStep, failed: tuple[str, ...],
    ) -> None:
        """Persist the bounded expansion of the pending repair pass."""

        policy = effective_repair_scope_policy(ctx.options)
        approved = set(cycle_plan.mutable_scope)
        added = tuple(sorted(set(step.added_paths)))
        if not added or not set(added).issubset(approved):
            raise ResumeIntegrityError(
                "the gate recovery scope expansion exceeds the approved cycle scope"
            )
        if len(added) > policy.max_added_paths:
            raise ResumeIntegrityError(
                "the gate recovery scope expansion exceeds the configured bound"
            )
        directory = check_repair_attempt_dir(
            ctx.run_dir, cycle_plan.cycle.number, stage, step.repair_attempt,
        )
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(directory / _EXPANSION_ARTIFACT, _json_text({
            "schema_version": 1,
            "attempt": step.repair_attempt,
            "tree": step.tree,
            "failed_check_ids": list(failed),
            "added_paths": list(added),
            "policy": policy.policy,
            "bound": policy.max_added_paths,
        }))


class GateAcceptanceService:
    """Persist and validate the green tree accepted by one gate episode."""

    def __init__(
        self,
        *,
        secrets: tuple[str, ...],
        repair_scope_policy: EffectiveRepairScopePolicy,
        authorize_candidate_tree: Callable[..., None],
        check_repair_attempts: Callable[..., tuple[CheckRepairAttempt, ...]],
        load_revision: Callable[[Path], Any],
        trace_emit: Callable[..., None],
        bounded_detail: Callable[[Exception], str],
    ) -> None:
        self._secrets = secrets
        self._repair_scope_policy = repair_scope_policy
        self._authorize_candidate_tree = authorize_candidate_tree
        self._check_repair_attempts = check_repair_attempts
        self._load_revision = load_revision
        self._trace_emit = trace_emit
        self._bounded_detail = bounded_detail

    def accept(
        self, store: Any, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: EvidenceBundle, *, base_paths: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if (
            not evidence.deterministic_passed
            or not required_checks_passed(evidence)
            or evidence.staged_tree_sha is None
        ):
            raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
        no_change = not evidence.changed_files
        if no_change and (evidence.diff != "" or evidence.base_sha != ctx.base_sha):
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change evidence is not bound to the run base",
            )
        worktree = ctx.info.worktree
        directory = gate_dir(ctx.run_dir, cycle_plan.cycle, stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = gate_acceptance_path(ctx.run_dir, cycle_plan.cycle, stage)
        authority = gate_mutable_authority(
            ctx.run_dir, cycle_plan.cycle.number, stage,
            base_paths=(
                cycle_plan.mutable_scope if base_paths is None else base_paths
            ),
            policy_config=self._repair_scope_policy,
            require_attempt_records=True,
        )
        stored = _read_json_artifact(path) if path.is_file() else None
        if path.is_file() and stored is None:
            raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance is corrupted")
        if stored is not None:
            stored_no_change = stored.get("no_change", False) if isinstance(stored, dict) else False
            stored_parent = stored.get("parent_sha") if isinstance(stored, dict) else None
            parent_valid = _is_object_id(stored_parent) or (
                stored_no_change is True
                and stored_parent is None
                and not evidence.changed_files
            )
            evidence_sha256 = self._durable_evidence_sha256(directory)
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 2
                or stored.get("review_cycle") != cycle_plan.cycle.number
                or not all(_is_object_id(stored.get(key)) for key in ("tree_sha", "commit_sha"))
                or not isinstance(stored_no_change, bool)
                or not parent_valid
                or stored_no_change is not (not evidence.changed_files)
                or (
                    stored_no_change
                    and (
                        stored_parent is not None
                        or stored.get("commit_created") is not False
                        or stored.get("acceptance_kind") != "existing-head"
                    )
                )
                or stored.get("stage") != stage.value
                or stored.get("acceptance_kind") not in {"existing-head", "repair", "semantic-revision"}
                or not isinstance(stored.get("commit_created"), bool)
                or stored.get("tree_sha") != evidence.staged_tree_sha
                or stored.get("mutable_scope") != list(authority.effective_paths)
                or stored.get("mutable_scope_sha256") != authority.sha256
                or stored.get("evidence_sha256") != evidence_sha256
            ):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance does not match evidence")
            if current_head(worktree) != stored["commit_sha"]:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted gate HEAD moved")
            if (
                resolve_tree(worktree, stored["commit_sha"]) != stored["tree_sha"]
                or (
                    not stored_no_change
                    and commit_parents(worktree, stored["commit_sha"]) != (stored["parent_sha"],)
                )
            ):
                raise PipelineFailure(
                    "RESUME_INTEGRITY_FAILURE", "gate acceptance does not match its commit",
                )
            if stored.get("commit_created"):
                chain = list(accepted_chain_records(ctx.run_dir))
                if not any(
                    isinstance(item, dict) and item.get("commit_sha") == stored["commit_sha"]
                    for item in chain
                ):
                    chain.append({
                        "commit_sha": stored["commit_sha"],
                        "tree_sha": stored["tree_sha"],
                        "parent_sha": stored["parent_sha"],
                    })
                    atomic_write_text(ctx.run_dir / "accepted-chain.json", _json_text({"commits": chain}))
            self._emit_acceptance(ctx, cycle_plan, stage, stored)
            return stored

        self._authorize_candidate_tree(
            evidence, worktree, current_head(worktree), ctx.branch_ref,
        )
        head = current_head(worktree)
        current_tree = resolve_tree(worktree, head)
        if no_change and current_tree != evidence.staged_tree_sha:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "no-change HEAD tree differs from gate evidence",
            )
        if current_tree == evidence.staged_tree_sha:
            parents = () if no_change else commit_parents(worktree, head)
            if not no_change and len(parents) != 1:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted HEAD has no single parent")
            commit_sha, parent_sha = head, None if no_change else parents[0]
            parent_tree = resolve_tree(worktree, parent_sha) if parent_sha else None
            attempts = self._check_repair_attempts(
                ctx.run_dir, cycle_plan.cycle.number, stage,
            )
            revision = self._load_revision(
                semantic_revision_dir(ctx.run_dir, cycle_plan.cycle.number)
            )
            last_attempt = attempts[-1] if attempts else None
            recovered_repair = bool(
                last_attempt is not None
                and parent_tree is not None
                and last_attempt.tree_before == parent_tree
                and last_attempt.tree_after == evidence.staged_tree_sha
                and last_attempt.tree_before != last_attempt.tree_after
            )
            recovered_revision = bool(
                stage in {GateStage.POST_SEMANTIC_REVISION, GateStage.POST_REVIEW_IMPLEMENTATION}
                and revision is not None
                and revision.tree_before != revision.tree_after
                and revision.tree_after == evidence.staged_tree_sha
                and parent_tree is not None
                and parent_tree == revision.tree_before
            )
            acceptance_kind = (
                "repair" if recovered_repair else
                "semantic-revision" if recovered_revision else "existing-head"
            )
            commit_created = recovered_repair or recovered_revision
        else:
            parent_sha = head
            try:
                commit_safety_gate(
                    worktree,
                    tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha,
                    mutable_scope=authority.effective_paths,
                    verification_status="passed",
                    secrets=self._secrets,
                    max_diff_bytes=None,
                )
            except CommitSafetyError as exc:
                raise PipelineFailure("COMMIT_GATE_FAILED", self._bounded_detail(exc)) from exc
            attempts = self._check_repair_attempts(
                ctx.run_dir, cycle_plan.cycle.number, stage,
            )
            if attempts:
                commit_sha = commit_repair_tree(
                    worktree, tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha, cycle=cycle_plan.cycle.number,
                    body=f"MetaHarness-Run: {ctx.run_id}",
                )
                acceptance_kind = "repair"
            elif stage in {GateStage.POST_SEMANTIC_REVISION, GateStage.POST_REVIEW_IMPLEMENTATION}:
                commit_sha = commit_revision_tree(
                    worktree, tree_sha=evidence.staged_tree_sha,
                    parent_sha=parent_sha, body=f"MetaHarness-Run: {ctx.run_id}",
                )
                acceptance_kind = "semantic-revision"
            else:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate changed the tree without a repair")
            commit_created = True

        acceptance = {
            "schema_version": 2,
            "review_cycle": cycle_plan.cycle.number,
            "stage": stage.value,
            "tree_sha": evidence.staged_tree_sha,
            "commit_sha": commit_sha,
            "parent_sha": parent_sha,
            "no_change": not evidence.changed_files,
            "commit_created": commit_created,
            "acceptance_kind": acceptance_kind,
            "mutable_scope": list(authority.effective_paths),
            "mutable_scope_sha256": authority.sha256,
            "evidence_sha256": self._durable_evidence_sha256(directory),
        }
        atomic_write_text(path, _json_text(acceptance))
        if commit_created:
            chain = list(accepted_chain_records(ctx.run_dir))
            if not any(item.get("commit_sha") == commit_sha for item in chain if isinstance(item, dict)):
                chain.append({"commit_sha": commit_sha, "tree_sha": evidence.staged_tree_sha, "parent_sha": parent_sha})
                atomic_write_text(ctx.run_dir / "accepted-chain.json", _json_text({"commits": chain}))
        store.update(
            status=store.load().get("status", RunStatus.VALIDATING),
            approved_tree_sha=evidence.staged_tree_sha,
            expected_head_sha=commit_sha,
            expected_parent_sha=parent_sha,
            expected_tree_sha=evidence.staged_tree_sha,
        )
        self._emit_acceptance(ctx, cycle_plan, stage, acceptance)
        return acceptance

    @staticmethod
    def _durable_evidence_sha256(directory: Path) -> str:
        try:
            return hashlib.sha256((directory / "evidence.json").read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE", "accepted gate evidence is unreadable",
            ) from exc

    def _emit_acceptance(self, ctx: Any, cycle_plan: Any, stage: GateStage, payload: Mapping[str, Any]) -> None:
        self._trace_emit(
            "gate.accepted",
            phase="validation",
            cycle=cycle_plan.cycle.number,
            data={
                "stage": stage.value,
                "parent_sha": payload["parent_sha"],
                "commit_sha": payload["commit_sha"],
                "tree_sha": payload["tree_sha"],
                "commit_created": payload.get("commit_created"),
                "acceptance_kind": payload.get("acceptance_kind"),
            },
        )
