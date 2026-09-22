"""Low-level primitives shared by the orchestration components.

Nothing here knows about the state machine: bounded text helpers,
artifact readers, git-state snapshots, the error hierarchy and the
durable artifact name tuples every component archives.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    GitError,
    candidate_tree_sha,
    current_head,
    index_tree_sha,
    local_branches,
    registered_worktrees,
    status_porcelain,
    symbolic_head,
)
from ..plan_recovery import PLAN_RECOVERY_ARTIFACT
from ..result import (
    ResultArtifactError,
    atomic_write_text,
)
from ..models import RunStatus, RunCycle
from ..resume import ResumeIntegrityError
from .pipeline_v2 import cycle_record_path
from ..validation import check_result_json
from ..agent.diagnostics import TOKEN_DIAGNOSTICS_NAME


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """A commit precondition does not hold immediately before the commit."""


class ScopeApprovalRequired(OrchestrationError):
    """A scope expansion is durably paused until its exact delta is approved."""

    code = "WAITING_SCOPE_APPROVAL"


@dataclasses.dataclass(frozen=True)
class CycleArtifactService:
    """Own the durable identity and state boundary of one pipeline cycle."""

    cycle_update: Callable[..., None]
    trace_emit: Callable[..., None]
    set_trace_cycle: Callable[[int], None]

    def begin(self, store: Any, ctx: Any, cycle: RunCycle, fresh: bool) -> None:
        self.set_trace_cycle(cycle.number)
        path = cycle_record_path(ctx.run_dir, cycle)
        record = _json_text({
            "schema_version": 1,
            "number": cycle.number,
            "kind": cycle.kind.value,
        })
        if path.exists():
            if path.read_text(encoding="utf-8") != record:
                raise ResumeIntegrityError(f"cycle {cycle.number:03d} record diverges")
        else:
            atomic_write_text(path, record)
        state = store.load()
        store.update(status=state.get("status", RunStatus.IMPLEMENTING), cycle=cycle.number)
        self.cycle_update(store, cycle, status="running")
        if fresh and cycle.number > 1:
            store.update(
                status=RunStatus.PLANNING,
                git_ownership=_git_ownership_payload(
                    _git_ownership(ctx.repo, ctx.info.worktree)
                ),
            )
            self.trace_emit(
                "cycle.started", phase="cycle", cycle=cycle.number,
                data={"kind": cycle.kind.value}, once=True,
            )


_COMMIT_SUBJECT_LIMIT = 72


_MAX_AGENT_REPORT_BYTES = 32_000


_MAX_STEP_REPORT_BYTES = 2_048


_AGENT_ARTIFACTS = (
    "prompt.diagnostics.json",
    "agent.events.jsonl",
    "agent.stderr.log",
    "agent.final.md",
    "agent.result.json",
)


_REVISION_ARTIFACTS = (
    "agent.prompt.txt",
    "prompt.diagnostics.json",
    "agent.events.jsonl",
    "agent.stderr.log",
    "agent.final.md",
    "agent.result.json",
)


def _commit_subject(title: str) -> str:
    """One Git subject line of at most 72 characters from the plan title."""

    first = next((line for line in title.splitlines() if line.strip()), "")
    subject = " ".join(first.split()).strip("#*_` ") or "MetaHarness change"
    if len(subject) > _COMMIT_SUBJECT_LIMIT:
        subject = subject[: _COMMIT_SUBJECT_LIMIT - 3].rstrip() + "..."
    return subject


def _bounded_report(text: str) -> str:
    """Bound the non-authoritative agent report sent to the reviewer."""

    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_AGENT_REPORT_BYTES:
        return text
    head = encoded[:_MAX_AGENT_REPORT_BYTES].decode("utf-8", errors="ignore")
    omitted = len(encoded) - _MAX_AGENT_REPORT_BYTES
    return f"{head}\n[... {omitted} bytes truncated; full report in agent.final.md ...]"


def _bounded_v2_report(text: str) -> str:
    """Bound an individual staged-step report for every semantic prompt."""

    limit = _MAX_STEP_REPORT_BYTES
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    marker = b"\n[... report truncated ...]"
    head = data[: max(0, limit - len(marker))].decode("utf-8", errors="ignore")
    return head + marker.decode()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


_MAX_REPORTED_PATHS = 20


def _safe_path_label(path: str) -> str:
    """A printable rendering of one repository path for failure details."""

    return "".join(
        character if character.isprintable() else f"\\x{ord(character) & 0xFF:02x}"
        for character in path
    )[:300]


def _paths_detail(paths: list[str]) -> str:
    shown = [_safe_path_label(path) for path in paths[:_MAX_REPORTED_PATHS]]
    extra = len(paths) - len(shown)
    return ",".join(shown) + (f" (+{extra} more)" if extra > 0 else "")


@dataclasses.dataclass(frozen=True)
class GitOwnership:
    """Git state the implementation agent is not allowed to change."""

    head_ref: str | None
    head: str
    branches: frozenset[str]
    worktrees: frozenset[str]



def _git_ownership(repo: Path, worktree: Path) -> GitOwnership:
    return GitOwnership(
        head_ref=symbolic_head(worktree),
        head=current_head(worktree),
        branches=local_branches(repo),
        worktrees=registered_worktrees(repo),
    )


def _git_ownership_payload(ownership: GitOwnership) -> dict[str, Any]:
    return {
        "head_ref": ownership.head_ref,
        "head": ownership.head,
        "branches": sorted(ownership.branches),
        "worktrees": sorted(ownership.worktrees),
    }


def _ownership_violations(
    before: GitOwnership, after: GitOwnership, *, branch_ref: str, base_sha: str
) -> list[str]:
    problems: list[str] = []
    if after.head_ref != branch_ref:
        problems.append(
            f"worktree HEAD switched from {branch_ref} to {after.head_ref or 'a detached HEAD'}"
        )
    if after.head != base_sha:
        problems.append("worktree HEAD commit changed (commit, merge, reset or rewrite)")
    created = sorted(after.branches - before.branches)
    if created:
        problems.append("branch(es) created: " + ", ".join(created))
    deleted = sorted(before.branches - after.branches)
    if deleted:
        problems.append("branch(es) deleted: " + ", ".join(deleted))
    added_worktrees = sorted(after.worktrees - before.worktrees)
    if added_worktrees:
        problems.append("worktree(s) created: " + ", ".join(added_worktrees))
    removed_worktrees = sorted(before.worktrees - after.worktrees)
    if removed_worktrees:
        problems.append("worktree(s) removed: " + ", ".join(removed_worktrees))
    return problems


@dataclasses.dataclass(frozen=True)
class StepExecutionOutcome:
    """The durable result of one successful implementation step."""

    step_id: str
    profile_id: str
    tree_before: str
    tree_after: str
    changed_paths: tuple[str, ...]
    usage: dict[str, int]
    final_report: str
    # A verification the worker could not complete because an out-of-scope
    # path owned by a later approved step still fails.  Data only: the
    # deterministic gate and the reviewer remain the authority.
    deferred_verify: str = ""
    mismatch_retry_count: int = 0


class DeferredStepExecutionOutcome:
    """A clean mismatch result without widening the normal outcome schema."""

    status = "DEFERRED_CONTRACT_MISMATCH"

    def __init__(
        self, *, step_id: str, profile_id: str, tree_before: str,
        tree_after: str, changed_paths: tuple[str, ...], usage: dict[str, int],
        final_report: str, mismatch: str, initial_mismatch: str = "",
        mismatch_retry_count: int = 0, deferred_verify: str = "",
    ) -> None:
        self.step_id = step_id
        self.profile_id = profile_id
        self.tree_before = tree_before
        self.tree_after = tree_after
        self.changed_paths = changed_paths
        self.usage = usage
        self.final_report = final_report
        self.mismatch = mismatch
        self.initial_mismatch = initial_mismatch
        self.mismatch_retry_count = mismatch_retry_count
        self.deferred_verify = deferred_verify


_SYNTHETIC_NO_CHANGE_MISMATCH = (
    "Worker completed successfully without producing an in-scope candidate "
    "change. Retry once to distinguish an already-satisfied step from a stale "
    "contract."
)


_BOUNDED_NO_CHANGE_MISMATCH = (
    "No in-scope change remained necessary after bounded retry; the step is "
    "deferred until the contract is revisited."
)


class StepExecutionFailure(OrchestrationError):
    """One implementation step failed a gate; the caller owns the run status."""

    def __init__(
        self,
        reason: str,
        step_id: str,
        detail: str | None = None,
        *,
        profile_id: str | None,
        tree_before: str | None,
        tree_after: str | None = None,
        usage: dict[str, int] | None = None,
        mismatch: str | None = None,
        clean_contract_mismatch: bool = False,
        mismatch_retry_count: int = 0,
        initial_mismatch: str | None = None,
        index_tree_after: str | None = None,
        step_dir: Path | None = None,
    ) -> None:
        super().__init__(f"{reason}: step={step_id}")
        self.reason = reason
        self.step_id = step_id
        self.detail = detail
        self.profile_id = profile_id
        self.tree_before = tree_before
        self.tree_after = tree_after
        self.usage = usage
        self.mismatch = mismatch
        self.clean_contract_mismatch = clean_contract_mismatch
        self.mismatch_retry_count = mismatch_retry_count
        self.initial_mismatch = initial_mismatch
        self.index_tree_after = index_tree_after
        self.step_dir = step_dir


_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


# Artifacts of one failed attempt, moved to ``attempts/NN/`` before the same
# operation is retried so a stale report is never read as the new one.
_ATTEMPT_ARTIFACTS = (
    "agent.prompt.txt", "prompt.diagnostics.json", "agent.events.jsonl", "agent.stderr.log", "agent.final.md",
    "agent.result.json", "step.json", TOKEN_DIAGNOSTICS_NAME, "tree_after_failure.txt",
    "usage.json", "results.json",
)


_PLANNER_ATTEMPT_ARTIFACTS = (
    "planner.request.txt", "planner.repair.request.txt", "prompt.diagnostics.json", "prompt.diagnostics.repair.json", "planner.raw.md", "task_plan_v2.json", "task_plan.json",
    "implementation_bundle.json", "planner.usage.json",
)


_REVIEW_ATTEMPT_ARTIFACTS = (
    "reviewer.request.txt", "reviewer.request.meta.json", "prompt.diagnostics.json", "prompt.diagnostics.repair.json", "reviewer.raw.md",
    "reviewer.usage.json", "review.json",
)


_CHECK_ATTEMPT_ARTIFACTS = ("checks.json", "changed-files.txt", "diff.patch", "evidence.json")




_REVISION_ATTEMPT_ARTIFACTS = _AGENT_ARTIFACTS + ("tree_after_failure.txt",)


_PLANNER_CONVERSATION = "planner.conversation.json"


# An operator recovery also retires the previous plan summary, the planner
# conversation (the repair planner must never continue a conversation whose
# answer was replaced) and any earlier recovery record.
_RECOVERY_ATTEMPT_ARTIFACTS = _PLANNER_ATTEMPT_ARTIFACTS + (
    "implementation_contract.md", _PLANNER_CONVERSATION, PLAN_RECOVERY_ARTIFACT,
)


class CandidatePushError(OrchestrationError):
    """The immutable candidate could not be pushed to the run branch."""

    code = "PUSH_FAILED"


def _read_json_artifact(path: Path, limit: int = 16 * 1024 * 1024) -> Any:
    try:
        if path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def _read_bounded_text(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_tree_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value if _GIT_OBJECT_ID.fullmatch(value) else None


def _is_object_id(value: Any) -> bool:
    return isinstance(value, str) and _GIT_OBJECT_ID.fullmatch(value) is not None


def _safe_candidate_tree(worktree: Path) -> str | None:
    try:
        return candidate_tree_sha(worktree)
    except GitError:
        return None


def _safe_index_tree(worktree: Path) -> str | None:
    try:
        return index_tree_sha(worktree)
    except GitError:
        return None


def _safe_status(worktree: Path) -> tuple[str, ...] | None:
    try:
        return status_porcelain(worktree)
    except GitError:
        return None


def _new_status_lines(
    before: tuple[str, ...], after: tuple[str, ...] | None,
) -> list[str]:
    """The porcelain status lines an attempt added or removed."""

    if after is None:
        return ["the Git status could not be read"]
    kept = set(before)
    seen = set(after)
    return [line for line in after if line not in kept] + [
        line for line in before if line not in seen
    ]


def _record_failure_tree(artifact_dir: Path, worktree: Path) -> None:
    """Record the tree a failed attempt left, so a resume can recognize it."""

    tree = _safe_candidate_tree(worktree)
    if tree is None:
        return
    try:
        atomic_write_text(artifact_dir / "tree_after_failure.txt", tree + "\n")
    except ResultArtifactError:
        pass


def _archive_attempt(directory: Path, *, names: tuple[str, ...] = _ATTEMPT_ARTIFACTS) -> Path | None:
    """Move a failed attempt's artifacts aside before retrying that operation."""

    present = [name for name in names if (directory / name).exists()]
    if not present:
        return None
    target = _archive_attempt_target(directory)
    for name in present:
        os.replace(directory / name, target / name)
    return target


def _archive_attempt_target(directory: Path) -> Path:
    """Create and return the next free ``attempts/NN/`` directory."""

    root = directory / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    target = root / f"{index:02d}"
    target.mkdir(parents=True)
    return target


def _archive_attempt_tree(directory: Path) -> None:
    """Archive every file of a retryable operation, including dynamic logs."""

    if not directory.is_dir():
        return
    files = [
        path for path in directory.rglob("*")
        if path.is_file() and "attempts" not in path.relative_to(directory).parts
    ]
    if not files:
        return
    root = directory / "attempts"
    index = 1
    while (root / f"{index:02d}").exists():
        index += 1
    target = root / f"{index:02d}"
    for source in files:
        relative = source.relative_to(directory)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def _create_file_once(path: Path, data: bytes) -> None:
    """Atomically create *path* with *data*; never replace an existing file.

    Raises :class:`FileExistsError` when *path* already exists.
    """

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


# The prompt templates ship in the ``metaharness`` package directory, one
# level above this sub-package.  Resolved exactly as
# ``orchestrator.py`` used to resolve it, so the template files read are
# byte-for-byte the same ones.
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _repair_checks_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    """Summarize a cycle's accepted deterministic gate for the correction planner.

    The reviewed candidate only exists because its gate was accepted, so argv,
    cwd, durations and log tails add no decision value here; they stay in the
    durable check artifacts.
    """

    checks: list[dict[str, Any]] = []

    for check in bundle.checks:
        payload = (
            dict(check)
            if isinstance(check, Mapping)
            else check_result_json(check)
        )

        checks.append(
            {
                "name": payload.get("name"),
                "exit_code": payload.get("exit_code"),
                "timed_out": bool(payload.get("timed_out", False)),
                "workspace_mutated": bool(
                    payload.get("workspace_mutated", False)
                ),
            }
        )

    return {
        "deterministic_passed": bundle.deterministic_passed,
        "failures": list(bundle.failures),
        "checks": checks,
    }


def _check_payload(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    # A bundle rebuilt from ``evidence.json`` on resume carries the persisted
    # reviewer-safe payloads instead of CheckResult objects.
    payload: list[dict[str, Any]] = []
    for check in bundle.checks:
        item = dict(check) if isinstance(check, Mapping) else check_result_json(check)
        if bundle.required_check_ids:
            item["required"] = item.get("name") in bundle.required_check_ids
        payload.append(item)
    return payload


@dataclasses.dataclass(frozen=True)
class CheckRepairScope:
    base_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    policy: str
    bound: int
    source: str


@dataclasses.dataclass(frozen=True)
class GateMutableAuthority:
    """The one durable mutation authority of a gate episode."""

    base_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    source: str
    sha256: str



def _status_has_unstaged_or_untracked(status: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for line in status:
        if line.startswith("?? "):
            problems.append(f"new untracked file: {line[3:]}")
        elif len(line) >= 2 and line[1] != " ":
            problems.append(f"unstaged change: {line}")
    return problems
