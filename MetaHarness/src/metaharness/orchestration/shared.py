"""Low-level primitives shared by the orchestration components.

Nothing here knows about the state machine: bounded text helpers,
artifact readers, git-state snapshots, the error hierarchy and the
durable artifact name tuples every component archives.
"""

from __future__ import annotations

import dataclasses
import inspect
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
from ..attempt_transaction import (
    GitOwnership,
    git_ownership,
    ownership_violations,
    status_has_unstaged_or_untracked,
)
from ..evidence import EvidenceBundle
from ..gitops import (
    GitError,
    candidate_tree_sha,
    index_tree_sha,
    status_porcelain,
)
from ..result import (
    ResultArtifactError,
    atomic_write_text,
)
from ..models import CycleKind, RunCycle
from ..resume import ResumeIntegrityError
from .pipeline_v2 import cycle_record_path
from ..validation import check_result_json
from ..agent.diagnostics import TOKEN_DIAGNOSTICS_NAME
from ..llm.chat import OpenAIChatTextClient


class OrchestrationError(RuntimeError):
    """A run could not be started or completed safely."""


class CommitBoundaryError(OrchestrationError):
    """A commit precondition does not hold immediately before the commit."""


@dataclasses.dataclass(frozen=True)
class CycleArtifactService:
    """Own the durable identity and state boundary of one pipeline cycle."""

    cycle_update: Callable[..., None]
    trace_emit: Callable[..., None]
    set_trace_cycle: Callable[[int], None]

    def begin(self, store: Any, ctx: Any, cycle: RunCycle, fresh: bool) -> None:
        self.set_trace_cycle(cycle.number)
        if cycle.number != ctx.iteration or cycle.kind is not CycleKind.INITIAL:
            raise ResumeIntegrityError("cycle number does not match the current iteration")
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
        store.load()
        execution = store.load().get("execution")
        execution = dict(execution) if isinstance(execution, Mapping) else {}
        execution["steps"] = [
            {
                "step_id": step.id,
                "implementer": dataclasses.asdict(selected.implementer),
            }
            for step, selected in zip(ctx.plan.steps, ctx.selection.steps, strict=True)
        ]
        planner = store.load().get("planner")
        planner = dict(planner) if isinstance(planner, Mapping) else {}
        planner["required_checks"] = list(ctx.plan.required_checks)
        planner["execution_mode"] = (
            ctx.plan.execution_mode.value if ctx.plan.execution_mode else None
        )
        store.update_metadata(
            cycle=cycle.number,
            iteration=cycle.number,
            current_milestone={"id": ctx.plan.milestone_id, "title": ctx.plan.milestone_title},
            planner=planner,
            steps=[
                {"id": step.id, "title": step.title, "status": "waiting",
                 "execution_class": step.execution_class.value,
                 "profile_id": selected.implementer.profile_id}
                for step, selected in zip(ctx.plan.steps, ctx.selection.steps, strict=True)
            ],
            execution=execution,
        )
        self.cycle_update(store, cycle, status="running")


_MAX_AGENT_REPORT_BYTES = 32_000


_MAX_STEP_REPORT_BYTES = 2_048


_AGENT_ARTIFACTS = (
    "prompt.diagnostics.json",
    "agent.events.jsonl",
    "agent.stderr.log",
    "agent.final.md",
    "agent.result.json",
    "executor.json",
)


def bounded_v2_report(text: str) -> str:
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


def bounded_parse_detail(exc: Exception) -> str:
    """Keep a parser failure readable without persisting an unbounded message."""

    return " ".join(str(exc).split())[:500]


def chat_client(
    endpoint: Any, environment: Mapping[str, str],
    on_transport: Callable[[dict[str, Any]], None] | None = None,
) -> OpenAIChatTextClient:
    """Construct the production chat client with the runtime mapping.

    The constructor is inspected once so embedded clients can receive the
    runtime environment or the transport observer when they declare them.
    """

    constructor = OpenAIChatTextClient
    try:
        parameters = inspect.signature(constructor).parameters.values()
        accepts_environment = any(
            parameter.name == "environment"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        accepts_transport = any(
            parameter.name == "on_transport"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_environment = True
        accepts_transport = True
    kwargs: dict[str, Any] = {}
    if accepts_environment:
        kwargs["environment"] = environment
    if accepts_transport:
        kwargs["on_transport"] = on_transport
    return constructor(endpoint, **kwargs)



_git_ownership = git_ownership
_ownership_violations = ownership_violations


def _git_ownership_payload(ownership: GitOwnership) -> dict[str, Any]:
    return {
        "head_ref": ownership.head_ref,
        "head": ownership.head,
        "branches": sorted(ownership.branches),
        "worktrees": sorted(ownership.worktrees),
        "refs": [list(item) for item in sorted(ownership.refs)],
    }


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
    # deterministic gate and the audit remain the authority.
    deferred_verify: str = ""
    no_change: bool = False
    # Paths this attempt changed outside its declared mutable scope and the
    # harness admitted.  They authorize this attempt's diff only: they never
    # widen the durable contract nor the authority of a later step.
    out_of_scope_paths: tuple[str, ...] = ()
    # The exact porcelain status this attempt started from, so a fast-gate
    # refusal can be rolled back on the same boundary as any other failure.
    status_before: tuple[str, ...] = ()


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
        index_tree_after: str | None = None,
        status_before: tuple[str, ...] | None = None,
        step_dir: Path | None = None,
        retry_feedback: str | None = None,
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
        self.index_tree_after = index_tree_after
        self.status_before = status_before
        self.step_dir = step_dir
        # The bounded feedback the next same-executor attempt receives when
        # this failure discarded something the worker must not produce again.
        self.retry_feedback = retry_feedback


_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


# Artifacts of one failed attempt, moved to ``attempts/NN/`` before the same
# operation is retried so a stale report is never read as the new one.
_ATTEMPT_ARTIFACTS = (
    "agent.prompt.txt", "prompt.diagnostics.json", "agent.events.jsonl", "agent.stderr.log", "agent.final.md",
    "agent.result.json", "step.json", TOKEN_DIAGNOSTICS_NAME, "tree_after_failure.txt",
    "usage.json", "results.json", "executor.json", "failure.json",
    # The authority, candidate and acceptance records of one worker attempt.
    "step_authority.json", "step_candidate.json", "step_acceptance.json",
)


_PLANNER_ATTEMPT_ARTIFACTS = (
    "planner.request.txt", "planner.repair.request.txt", "prompt.diagnostics.json", "prompt.diagnostics.repair.json", "planner.raw.md", "task_plan.json",
    "implementation_bundle.json", "planner.usage.json",
)


_CHECK_ATTEMPT_ARTIFACTS = ("checks.json", "changed-files.txt", "diff.patch", "evidence.json", "checks")




_PLANNER_CONVERSATION = "planner.conversation.json"


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


def is_object_id(value: Any) -> bool:
    """Whether `value` names one Git object; the siblings' public spelling."""

    return isinstance(value, str) and _GIT_OBJECT_ID.fullmatch(value) is not None


# The refoundation's frozen importers still reach for the private spelling.
_is_object_id = is_object_id


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


def _check_payload(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    # A bundle rebuilt from ``evidence.json`` on resume carries the persisted
    # bounded payloads instead of CheckResult objects.
    payload: list[dict[str, Any]] = []
    for check in bundle.checks:
        item = dict(check) if isinstance(check, Mapping) else check_result_json(check)
        if bundle.required_check_ids:
            item["required"] = item.get("name") in bundle.required_check_ids
        payload.append(item)
    return payload


@dataclasses.dataclass(frozen=True)
class GateMutableAuthority:
    """The one durable mutation authority of a gate episode."""

    base_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    effective_paths: tuple[str, ...]
    source: str
    sha256: str
    initial_paths: tuple[str, ...] = ()



_status_has_unstaged_or_untracked = status_has_unstaged_or_untracked


# The public spelling of the toolbox the run authorities import: the sibling
# services keep reading the private names above, while `run_bootstrap`, the
# composition root, the durable readers, the resume gate, the failure
# projection, the observability stream, the runtime kernel and the step
# services reach the same objects through their public names.
AGENT_ARTIFACTS = _AGENT_ARTIFACTS
MAX_AGENT_REPORT_BYTES = _MAX_AGENT_REPORT_BYTES
PLANNER_CONVERSATION = _PLANNER_CONVERSATION
json_text = _json_text
git_ownership_payload = _git_ownership_payload
read_bounded_text = _read_bounded_text
read_json_artifact = _read_json_artifact
create_file_once = _create_file_once
read_tree_file = _read_tree_file
safe_candidate_tree = _safe_candidate_tree
archive_attempt = _archive_attempt
archive_attempt_target = _archive_attempt_target
archive_attempt_tree = _archive_attempt_tree
record_failure_tree = _record_failure_tree
safe_index_tree = _safe_index_tree
