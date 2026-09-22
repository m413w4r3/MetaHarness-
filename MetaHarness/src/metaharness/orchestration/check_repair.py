"""The check-repair sub-domain: failed-check classification and scope."""

from __future__ import annotations

import dataclasses
import os
import re

from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Any,
    Mapping,
    Sequence,
)
from .shared import (
    CheckRepairScope,
    _PROMPTS_DIR,
    _check_payload,
    _json_text,
    _read_json_artifact,
)
from ..evidence import (
    EvidenceBundle,
    SECRET_IN_DIFF,
    SECRET_IN_STAGED_BLOB,
    UNREVIEWABLE_TEXT_DIFF,
    UNSCANNABLE_STAGED_BLOB,
)
from ..gitops import tracked_files_in_tree
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


# These names are deliberately broader than the historical evidence enum.
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
    """Return failures that make semantic review unsafe.

    A normal configured check failure is evidence for the reviewer in P25;
    mutations, timeouts, secrets, ownership and malformed/oversized trees are
    still terminal integrity failures.
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


_MAX_CHECK_SCOPE_LOG_BYTES = 256 * 1024


_CHECK_SCOPE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:file://)?"
    r"(?:/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+)"
    r"\.(?:py|pyi|ts|tsx|js|jsx|java|go|rs|rb|php|c|cc|cpp|h|hpp|json|yaml|yml|toml|ini)"
    r"(?::[0-9]+(?::[0-9]+)?)?(?![A-Za-z0-9_.-])"
)


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_CHECK_SCOPE_LOG_BYTES))
            return stream.read(_MAX_CHECK_SCOPE_LOG_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _failed_check_text(
    *,
    run_dir: Path,
    evidence: EvidenceBundle,
) -> str:
    """Return only bounded output belonging to failed ordinary checks."""

    failed_names = {
        failure.split(":", 1)[1]
        for failure in _soft_check_failures(evidence)
        if ":" in failure
    }
    root = run_dir.resolve()
    chunks: list[str] = []
    for check in evidence.checks:
        payload = dict(check) if isinstance(check, Mapping) else check_result_json(check)
        if payload.get("name") not in failed_names:
            continue
        if not isinstance(check, Mapping):
            for key in ("stdout_log", "stderr_log"):
                value = getattr(check, key, None)
                if isinstance(value, str):
                    chunks.append(value[-_MAX_CHECK_SCOPE_LOG_BYTES:])
        for key in ("stdout_tail", "stderr_tail"):
            value = payload.get(key)
            if isinstance(value, str):
                chunks.append(value)
        for key in ("stdout_log_path", "stderr_log_path"):
            raw_path = payload.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                candidate = (root / raw_path).resolve()
                candidate.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            chunks.append(_read_log_tail(candidate))
    return "\n".join(chunk for chunk in chunks if chunk)


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


def _is_auto_expandable_test_path(path: str) -> bool:
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
    run_dir: Path,
    evidence: EvidenceBundle,
    base_mutable_scope: Sequence[str],
) -> list[str]:
    """Find tracked test paths named by failing-check evidence only."""

    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    text = _failed_check_text(run_dir=run_dir, evidence=evidence)
    resolved = [
        _resolve_check_path_candidate(
            match.group(0), worktree=worktree, tracked_files=tracked
        )
        for match in _CHECK_SCOPE_PATH_RE.finditer(text)
    ]
    base = set(base_mutable_scope)
    return sorted({
        path for path in resolved
        if path is not None
        and path not in base
        and _is_auto_expandable_test_path(path)
    })


def _hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if isinstance(item, str)
        if item in _DIRECT_FAILURES
        or item in _HARD_FAILURE_CODES
        or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
        or item.startswith("CHECK_MUTATED:")
        or item.startswith("CHECK_TIMEOUT:")
        or any(item.startswith(prefix) for prefix in _HARD_FAILURE_PREFIXES)
    ]


_CHECK_REPAIR_LOG_BYTES = 4 * 1024


def _check_repair_problem_context(evidence: EvidenceBundle) -> str:
    """Render only the failed checks for Claude's corrective prompt."""

    failures = _soft_check_failures(evidence)
    failed_names = {
        item.split(":", 1)[1]
        for item in failures
        if ":" in item
    }
    failed_checks: list[dict[str, Any]] = []
    for raw_check in _check_payload(evidence):
        if not isinstance(raw_check, Mapping):
            continue
        name = raw_check.get("name")
        if name not in failed_names:
            continue
        check: dict[str, Any] = {
            "name": name,
            "exit_code": raw_check.get("exit_code"),
            "timed_out": bool(raw_check.get("timed_out", False)),
            "workspace_mutated": bool(raw_check.get("workspace_mutated", False)),
        }
        for key in ("stdout_tail", "stderr_tail"):
            value = raw_check.get(key)
            if not isinstance(value, str) or not value:
                continue
            data = value.encode("utf-8", errors="replace")
            if len(data) > _CHECK_REPAIR_LOG_BYTES:
                data = data[-_CHECK_REPAIR_LOG_BYTES:]
            check[key] = data.decode("utf-8", errors="replace")
        failed_checks.append(check)
    return _json_text({"failure_ids": failures, "checks": failed_checks})


def _check_repair_prompt(
    *,
    spec: str,
    plan: TaskPlanV2,
    approved_contract_index: str,
    changed_files: str,
    evidence: EvidenceBundle,
    mutable_scope: list[str],
    candidate_identity: str = "",
    budget_bytes: int = 40_000,
    diagnostics_dir: str | Path | None = None,
) -> str:
    """Build the bounded prompt for one automatic check-repair pass."""

    del plan
    template = (_PROMPTS_DIR / "check_repair.txt").read_text(encoding="utf-8")
    failed_ids = _soft_check_failures(evidence)
    payload = build_check_repair_payload(
        spec=spec,
        failed_check_ids="\n".join(failed_ids) or "NONE",
        failed_check_evidence=_check_repair_problem_context(evidence),
        compact_contract_invariants=approved_contract_index,
        changed_files=changed_files,
        mutable_scope=_json_text(mutable_scope),
        candidate_identity=candidate_identity,
        template=template,
        budget_bytes=budget_bytes,
    )
    if diagnostics_dir is not None:
        write_prompt_diagnostics(diagnostics_dir, payload)
    return payload.rendered


_AUTO_BOUNDED_SOURCE = "auto-bounded failing-test evidence"



def _read_check_repair_scope(
    directory: Path,
    *,
    fallback_base: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
) -> CheckRepairScope:
    payload = _read_json_artifact(directory / "scope.json", 64 * 1024)
    base = tuple(sorted(set(fallback_base)))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        return CheckRepairScope(
            base_paths=base, added_paths=(), effective_paths=base,
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source="human-approved mutable scope",
        )
    raw_base = payload.get("base_mutable_scope")
    raw_added = payload.get("added_paths")
    raw_effective = payload.get("effective_mutable_scope")
    if not all(isinstance(value, list) for value in (raw_base, raw_added, raw_effective)):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    if any(not isinstance(path, str) for paths in (raw_base, raw_added, raw_effective) for path in paths):
        raise ResumeIntegrityError("check-repair scope artifact contains invalid paths")
    parsed_base = tuple(sorted(set(raw_base)))
    parsed_added = tuple(sorted(set(raw_added)))
    parsed_effective = tuple(sorted(set(raw_effective)))
    if parsed_base != base or parsed_effective != tuple(sorted(set(parsed_base) | set(parsed_added))):
        raise ResumeIntegrityError("check-repair scope artifact does not match its base scope")
    policy = payload.get("policy")
    bound = payload.get("bound")
    source = payload.get("source")
    if policy != policy_config.policy or bound != policy_config.max_added_paths or not isinstance(source, str):
        raise ResumeIntegrityError("check-repair scope policy changed")
    if parsed_added and source != _AUTO_BOUNDED_SOURCE:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    return CheckRepairScope(parsed_base, parsed_added, parsed_effective, policy, bound, source)


@dataclasses.dataclass(frozen=True)
class CheckRepairCoordinator:
    """Owns the bounded mutable-scope decision of one check-repair attempt.

    The operator's repair-scope policy is the only state it holds, injected
    explicitly: it never sees the ``Orchestrator`` and never widens a scope
    beyond the injected policy's bound.
    """

    effective_repair_scope: EffectiveRepairScopePolicy

    def resolve_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_sha: str,
        run_dir: Path,
        evidence: EvidenceBundle,
        base_mutable_scope: Sequence[str],
        previous: CheckRepairScope | None = None,
    ) -> CheckRepairScope:
        """The scope of the next attempt; earlier attempts' paths are kept."""

        base_paths = tuple(sorted(set(
            previous.base_paths if previous is not None else base_mutable_scope
        )))
        added = set(previous.added_paths) if previous is not None else set()
        policy = self.effective_repair_scope
        if policy.policy == "auto-bounded":
            candidates = _check_repair_scope_candidates(
                repo=repo, worktree=worktree, tree_sha=tree_sha, run_dir=run_dir,
                evidence=evidence, base_mutable_scope=tuple(sorted(set(base_paths) | added)),
            )
            proposed = added | set(candidates)
            if len(proposed) <= policy.max_added_paths:
                added = proposed
        return CheckRepairScope(
            base_paths=base_paths,
            added_paths=tuple(sorted(added)),
            effective_paths=tuple(sorted(set(base_paths) | added)),
            policy=policy.policy,
            bound=policy.max_added_paths,
            source=_AUTO_BOUNDED_SOURCE if added else "human-approved mutable scope",
        )
