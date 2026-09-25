"""Reusable safety and authority gates for accepted MetaHarness commits.

The worker may leave a traceable tree after a failed attempt.  This module is
the small, shared boundary that decides whether that tree is allowed to enter
the durable harness history.  It deliberately delegates blob/secret/binary
checks to :mod:`metaharness.evidence`; there is one scanner, not a second
security implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evidence import scan_staged_security
from .gitops import (
    GitError,
    candidate_tree_sha,
    changed_paths_between_trees,
    current_head,
    index_tree_sha,
    resolve_tree,
    status_porcelain,
)


# Stable reasons of one refused commit.  The human message stays detailed;
# recovery and diagnostics reason on the code, never on the message text.
COMMIT_GATE_FAILED = "COMMIT_GATE_FAILED"
COMMIT_SCOPE_VIOLATION = "COMMIT_SCOPE_VIOLATION"
COMMIT_PARENT_MISMATCH = "COMMIT_PARENT_MISMATCH"
COMMIT_TREE_MISMATCH = "COMMIT_TREE_MISMATCH"
COMMIT_WORKTREE_DRIFT = "COMMIT_WORKTREE_DRIFT"
COMMIT_SECURITY_FAILURE = "COMMIT_SECURITY_FAILURE"
COMMIT_VERIFICATION_FAILURE = "COMMIT_VERIFICATION_FAILURE"


class CommitSafetyError(GitError):
    """A tree has not passed every pre-commit safety gate."""

    def __init__(
        self, message: str, *, code: str = COMMIT_GATE_FAILED, paths: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.paths = tuple(str(path) for path in paths)


_STEP_ID = re.compile(r"\bS(?:0[1-9]|[1-9][0-9])\b")


@dataclass(frozen=True)
class DeferredVerification:
    """An explicit verification dependency that is safe to defer."""

    reason: str
    dependent_step_ids: tuple[str, ...]
    command_or_contract: str


@dataclass(frozen=True)
class CommitSafetyResult:
    """The immutable facts re-derived immediately before one commit."""

    parent_sha: str
    tree_sha: str
    changed_paths: tuple[str, ...]
    verification_status: str
    deferred: DeferredVerification | None = None
    security_failures: tuple[str, ...] = ()


def parse_deferred_verification(
    report: str,
    *,
    current_step_id: str,
    future_step_ids: Iterable[str],
) -> DeferredVerification | None:
    """Parse the existing explicit ``DEFERRED VERIFY DEPENDENCY`` contract.

    A marker without a body, a command/contract, or a future dependent step is
    not a deferred verification; it is an invalid/failed verification.
    """

    if not isinstance(report, str):
        raise TypeError("report must be a string")
    lines = report.splitlines()
    marker = "DEFERRED VERIFY DEPENDENCY"
    start = next(
        (index for index, line in enumerate(lines)
         if line.strip().rstrip(":").casefold() == marker.casefold()),
        None,
    )
    if start is None:
        return None
    body = "\n".join(lines[start + 1:]).strip()
    if not body:
        raise CommitSafetyError("deferred verification has no reason or contract")
    future = tuple(dict.fromkeys(str(item) for item in future_step_ids))
    ids = tuple(dict.fromkeys(_STEP_ID.findall(body)))
    dependents = tuple(item for item in ids if item in future and item != current_step_id)
    if not dependents:
        raise CommitSafetyError("deferred verification does not name a future dependent step")
    command = "\n".join(
        line.strip()[2:].strip() if line.strip().startswith("-") else line.strip()
        for line in body.splitlines()
        if line.strip()
    )
    if not command:
        raise CommitSafetyError("deferred verification command/contract is empty")
    return DeferredVerification(
        reason=body[:2048],
        dependent_step_ids=dependents,
        command_or_contract=command[:4096],
    )


_VERIFY_FAILED = re.compile(
    r"^\s*(?:VERIFY|VERIFICATION)\s*(?::|=)\s*(?:FAIL|FAILED|RED)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_ENVIRONMENT_VERIFY_FAILED = re.compile(
    r"^\s*(?:VERIFY|VERIFICATION)\s*[:=].*\b(?:FAIL|FAILED|RED)\b\s*\(\s*environment\s*:",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class StepVerification:
    """The verification decision of one successful worker attempt.

    It is a pure function of the durable worker report and the approved
    future step IDs, so a resumed step acceptance re-derives or re-reads it
    without any worker or model call.
    """

    status: str
    deferred: DeferredVerification | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "deferred_reason": self.deferred.reason if self.deferred else None,
            "dependent_step_ids": list(self.deferred.dependent_step_ids) if self.deferred else [],
            "deferred_command_or_contract": (
                self.deferred.command_or_contract if self.deferred else None
            ),
        }

    @classmethod
    def from_payload(cls, value: Any) -> "StepVerification":
        if not isinstance(value, Mapping) or value.get("status") not in {"passed", "deferred"}:
            raise ValueError("step verification record is invalid")
        if value["status"] == "passed":
            if value.get("deferred_reason") is not None or value.get("dependent_step_ids"):
                raise ValueError("passed verification carries deferred metadata")
            return cls("passed")
        reason = value.get("deferred_reason")
        dependents = value.get("dependent_step_ids")
        command = value.get("deferred_command_or_contract")
        if (
            not isinstance(reason, str) or not reason
            or not isinstance(dependents, list) or not dependents
            or any(not isinstance(item, str) for item in dependents)
            or not isinstance(command, str) or not command
        ):
            raise ValueError("deferred verification record is incomplete")
        return cls("deferred", DeferredVerification(reason, tuple(dependents), command))


def step_verification(
    final_report: str,
    *,
    step_id: str,
    future_step_ids: Iterable[str],
    reported_status: str | None = None,
    deferred_requested: bool = False,
) -> StepVerification:
    """Decide ``passed`` or ``deferred``; a red VERIFY is refused.

    A failure reported as an environment limitation is not a red VERIFY: the
    deterministic gate stays the authority over it.
    """

    if isinstance(reported_status, str) and reported_status.casefold() in {"failed", "fail", "red"}:
        raise CommitSafetyError("step VERIFY did not pass", code=COMMIT_VERIFICATION_FAILURE)
    if _VERIFY_FAILED.search(final_report) and not _ENVIRONMENT_VERIFY_FAILED.search(final_report):
        raise CommitSafetyError("step VERIFY did not pass", code=COMMIT_VERIFICATION_FAILURE)
    if not deferred_requested:
        return StepVerification("passed")
    try:
        deferred = parse_deferred_verification(
            final_report, current_step_id=step_id, future_step_ids=future_step_ids,
        )
    except CommitSafetyError as exc:
        raise CommitSafetyError(str(exc), code=COMMIT_VERIFICATION_FAILURE) from exc
    if deferred is None:
        raise CommitSafetyError(
            "a deferred step must provide the explicit DEFERRED VERIFY DEPENDENCY contract",
            code=COMMIT_VERIFICATION_FAILURE,
        )
    return StepVerification("deferred", deferred)


def commit_safety_gate(
    worktree: str | Path,
    *,
    tree_sha: str,
    parent_sha: str,
    mutable_scope: Iterable[str] = (),
    verification_status: str = "passed",
    deferred_reason: str | None = None,
    dependent_step_ids: Iterable[str] = (),
    deferred_command_or_contract: str | None = None,
    security_passed: bool = True,
    secrets: tuple[str, ...] = (),
    max_diff_bytes: int | None = None,
) -> CommitSafetyResult:
    """Re-derive all safety facts immediately before an accepted commit.

    The caller must have run the authoritative VERIFY/check gate.  This
    function independently checks the mutable scope, exact Git boundary and
    the existing evidence security policies.  It never creates a commit.
    """

    root = Path(worktree).expanduser().resolve()
    if verification_status not in {"passed", "deferred"}:
        raise CommitSafetyError("verification did not pass", code=COMMIT_VERIFICATION_FAILURE)
    if not security_passed:
        raise CommitSafetyError("security gate did not pass", code=COMMIT_SECURITY_FAILURE)
    if current_head(root) != parent_sha:
        raise CommitSafetyError("parent HEAD changed before commit", code=COMMIT_PARENT_MISMATCH)
    if index_tree_sha(root) != tree_sha or candidate_tree_sha(root) != tree_sha:
        raise CommitSafetyError(
            "working tree or index differs from the accepted tree", code=COMMIT_TREE_MISMATCH,
        )
    # A candidate is normally staged before this gate.  Staged changes are
    # intentional; only an unstaged worktree or an untracked path means that
    # the tree the worker produced is no longer the tree being authorized.
    dirty = tuple(
        line for line in status_porcelain(root)
        if len(line) < 2 or line[1] != " " or line.startswith("??")
    )
    if dirty:
        raise CommitSafetyError(
            "worktree has unstaged or untracked changes", code=COMMIT_WORKTREE_DRIFT,
        )
    parent_tree = resolve_tree(root, parent_sha)
    changed = tuple(changed_paths_between_trees(root, parent_tree, tree_sha))
    allowed = frozenset(str(path) for path in mutable_scope)
    if allowed and any(path not in allowed for path in changed):
        unexpected = [path for path in changed if path not in allowed]
        raise CommitSafetyError(
            "mutable scope violation: " + ", ".join(unexpected[:20]),
            code=COMMIT_SCOPE_VIOLATION, paths=unexpected[:20],
        )
    failures = scan_staged_security(
        root, secrets=secrets, max_diff_bytes=max_diff_bytes,
    )
    if failures:
        raise CommitSafetyError(
            "security/integrity gate failed: " + ", ".join(failures[:20]),
            code=COMMIT_SECURITY_FAILURE,
        )

    deferred: DeferredVerification | None = None
    if verification_status == "deferred":
        reason = (deferred_reason or "").strip()
        dependents = tuple(dict.fromkeys(str(item) for item in dependent_step_ids))
        contract = (deferred_command_or_contract or "").strip()
        if not reason or not dependents or not contract:
            raise CommitSafetyError(
                "deferred verification requires reason, dependent_step_ids and command/contract",
                code=COMMIT_VERIFICATION_FAILURE,
            )
        deferred = DeferredVerification(reason[:2048], dependents, contract[:4096])
    elif any((deferred_reason, tuple(dependent_step_ids), deferred_command_or_contract)):
        raise CommitSafetyError(
            "deferred metadata is present for a passed verification",
            code=COMMIT_VERIFICATION_FAILURE,
        )
    return CommitSafetyResult(
        parent_sha=parent_sha,
        tree_sha=tree_sha,
        changed_paths=changed,
        verification_status=verification_status,
        deferred=deferred,
        security_failures=(),
    )


def accepted_step_record(
    *,
    step_id: str,
    verification_status: str,
    parent_sha: str,
    commit_sha: str,
    tree_before: str,
    tree_after: str,
    changed_paths: Iterable[str],
    deferred: DeferredVerification | None = None,
    authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the durable, reviewer-safe metadata for one accepted step.

    ``authority`` names the effective step authority the commit gate used;
    its hashes are re-derived from their artifacts on every resume.
    """

    payload: dict[str, Any] = {
        "step_id": step_id,
        "verification_status": verification_status,
        "parent_sha": parent_sha,
        "commit_sha": commit_sha,
        "tree_before": tree_before,
        "tree_after": tree_after,
        "changed_paths": sorted(set(str(path) for path in changed_paths)),
    }
    if deferred is not None:
        payload.update({
            "deferred_reason": deferred.reason,
            "dependent_step_ids": list(deferred.dependent_step_ids),
            "deferred_verify_command_or_contract": deferred.command_or_contract,
        })
    else:
        payload["dependent_step_ids"] = []
    if authority is not None:
        for key in (
            "effective_authority_sha256", "effective_contract_sha256",
            "approved_contract_sha256", "authority_source", "repair_slot",
        ):
            payload[key] = authority.get(key)
    return payload


def unresolved_deferred_verifications(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return deferred records whose future dependencies are not accepted."""

    rows = tuple(record for record in records if isinstance(record, Mapping))
    passed = {
        record.get("step_id")
        for record in rows
        if record.get("verification_status") == "passed"
    }
    unresolved: list[dict[str, Any]] = []
    for record in rows:
        if record.get("verification_status") != "deferred":
            continue
        dependencies = record.get("dependent_step_ids")
        if not isinstance(dependencies, list) or any(item not in passed for item in dependencies):
            unresolved.append(dict(record))
    return tuple(unresolved)


def assert_deferred_verifications_resolved(records: Iterable[Mapping[str, Any]]) -> None:
    unresolved = unresolved_deferred_verifications(records)
    if unresolved:
        ids = ", ".join(str(item.get("step_id")) for item in unresolved)
        raise CommitSafetyError("candidate is forbidden while deferred verifications remain: " + ids)


# Explicit aliases for callers that prefer the noun used in the pipeline
# documentation.
validate_commit_safety = commit_safety_gate
commit_gate = commit_safety_gate
