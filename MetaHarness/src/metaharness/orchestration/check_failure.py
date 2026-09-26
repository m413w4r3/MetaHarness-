"""Failed-check classification and its bounded deterministic evidence.

The red-gate failures split in two families: the hard integrity failures that
close a gate episode without repair, and the ordinary ``CHECK_FAILED:<name>``
failures one bounded repair pass may answer.  Everything a later authority
reads about them -- the archived check logs, the bounded failure proofs, the
approved paths the evidence implicates, the durable record of each bounded
repair attempt and the durable identity of a step contract replan -- is a
deterministic fact of the gate episode, re-read from its artifacts and Git
objects so a resume rebuilds exactly the same evidence.  Nothing here writes
state, and nothing here decides a scope or a rung.
"""

from __future__ import annotations

import dataclasses
import json
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
    json_text,
    read_json_artifact,
)
from ..evidence import (
    EvidenceBundle,
    extract_failure_evidence,
    SECRET_IN_DIFF,
    SECRET_IN_STAGED_BLOB,
    UNREVIEWABLE_TEXT_DIFF,
    UNSCANNABLE_STAGED_BLOB,
)
from ..gitops import (
    GitError,
    changed_paths_between_trees,
    path_exists_in_tree,
    read_file_at_commit,
    tracked_files_in_tree,
)
from ..models import GateStage
from ..resume import ResumeIntegrityError
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

    return hard_failure_items(bundle.failures)


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
    persisted_checks = read_json_artifact(evidence_dir / "checks.json")
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


def _evidence_named_paths(
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


def hard_failure_items(failures: Any) -> list[str]:
    return [
        item for item in failures
        if isinstance(item, str)
        if item in _DIRECT_FAILURES
        or item in _HARD_FAILURE_CODES
        or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
        or any(item.startswith(prefix) for prefix in _HARD_FAILURE_PREFIXES)
    ]


def implicated_repair_paths(
    *,
    repo: Path,
    worktree: Path,
    tree_sha: str,
    evidence_dir: Path,
    evidence: EvidenceBundle,
    approved: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The approved paths the bounded failure evidence implicates.

    The first repair scope mirrors the scope resolver exactly: the failure
    evidence's own paths, or the changed approved paths when the output names
    none.  The implicated set is the union of both; the difference between them
    is the proof that the first pass could not reach the failing check.
    """

    approved_set = set(approved)
    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    initial = tuple(_evidence_named_paths(
        repo=repo, worktree=worktree, tree_sha=tree_sha, evidence_dir=evidence_dir,
        evidence=evidence, approved_mutable_scope=approved,
    ))
    changed = tuple(sorted({
        path for path in evidence.changed_files
        if path in approved_set and path in tracked and not _is_test_path(path)
    }))
    if not initial and len(changed) <= _MAX_CHANGED_PATH_FALLBACK:
        initial = changed
    return tuple(sorted(set(initial) | set(changed))), initial


def check_repair_problem_context(
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
    return json_text({
        "failure_ids": failures,
        "readable_failure_paths": sorted(readable)[:24],
        "checks": failed_checks,
    })


def red_gate_identity(evidence: EvidenceBundle) -> tuple[str, tuple[str, ...]]:
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

    tree, failed = red_gate_identity(evidence)
    try:
        context = json.loads(check_repair_problem_context(
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
    implicated, _initial = implicated_repair_paths(
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
        context = json.loads(check_repair_problem_context(
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


def replan_mismatch(
    *, problem: ReplanProblem, step_id: str, cycle: int, stage: GateStage,
    anchor_tree: str,
) -> str:
    """The durable identity of one red-gate contract replan.

    The text is the transaction's mismatch identity: it names the exact gate
    episode, the step and the two trees, so a resume finds the slot it opened
    instead of opening a second one.
    """

    return json_text({
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
    return _bounded_proof(json_text({
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
