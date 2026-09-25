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
    PipelineFailure,
    check_repair_attempts_dir,
    gate_acceptance_path,
    gate_dir,
    semantic_revision_dir,
)
from .shared import (
    CheckRepairScope,
    GateMutableAuthority,
    _PROMPTS_DIR,
    _check_payload,
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
    commit_parents,
    commit_repair_tree,
    commit_revision_tree,
    current_head,
    resolve_tree,
)
from ..commit_gate import CommitSafetyError, commit_safety_gate
from ..result import atomic_write_text
from ..models import GateStage, RunStatus
from ..planning_v2 import TaskPlanV2
from ..prompt_contracts import build_check_repair_payload, write_prompt_diagnostics
from ..resume import ResumeIntegrityError
from ..run_options import EffectiveRepairScopePolicy
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


def _hard_integrity_failures(bundle: EvidenceBundle) -> list[str]:
    """Return the failures that close a gate episode without repair.

    A normal configured check failure is soft: it may open a bounded
    check-repair attempt and never reaches the reviewer. Reversible check
    side effects and transient process failures are recovered before they can
    reach this boundary; secrets, ownership and malformed trees remain hard.
    """

    return _hard_failure_items(bundle.failures)


def _soft_check_failures(bundle: EvidenceBundle) -> list[str]:
    """Return ordinary deterministic check failures eligible for one repair.

    This deliberately accepts only the exact ``CHECK_FAILED:<name>`` family.
    Anything else, including a future failure category, fails closed and must
    never be handed to the corrective Claude pass.
    """

    if _hard_integrity_failures(bundle) or bundle.deterministic_passed:
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
        for failure in _soft_check_failures(evidence)
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

    failures = _soft_check_failures(evidence)
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
    failed_ids = _soft_check_failures(evidence)
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
    if version == 2:
        # Preserve resume compatibility for prior attempts whose base scope
        # was the entire approved cycle envelope.
        raw_approved = payload.get("base_mutable_scope")
        raw_initial = raw_approved
        raw_added = payload.get("added_paths")
        raw_effective = payload.get("effective_mutable_scope")
        legacy = True
    elif version == 3:
        raw_approved = payload.get("approved_mutable_scope")
        raw_initial = payload.get("initial_repair_scope")
        raw_added = payload.get("added_paths")
        raw_effective = payload.get("effective_repair_scope")
        legacy = False
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
        or (not legacy and not set(parsed_added).issubset(parsed_approved))
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
        "auto-bounded failing-test evidence",  # schema v2 resume compatibility
    }
    if not isinstance(source, str) or source not in valid_sources:
        raise ResumeIntegrityError("check-repair scope artifact has invalid provenance")
    if parsed_added and source not in {
        _SCOPE_REQUEST_SOURCE, "auto-bounded failing-test evidence",
    }:
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
        return GateMutableAuthority(
            base_paths=base,
            added_paths=(),
            effective_paths=base,
            source=_CYCLE_SCOPE_SOURCE,
            sha256=mutable_scope_sha256(base),
            initial_paths=(),
        )
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
                or attempt.get("mutable_scope") != list(scope.effective_paths)
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
    return GateMutableAuthority(
        base_paths=final.base_paths,
        added_paths=final.added_paths,
        effective_paths=final.effective_paths,
        source=final.source,
        sha256=mutable_scope_sha256(final.effective_paths),
        initial_paths=final.initial_repair_scope,
    )


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
                or stored.get("schema_version") not in {1, 2}
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
                or (
                    stored.get("schema_version") == 2
                    and stored.get("evidence_sha256") != evidence_sha256
                )
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
