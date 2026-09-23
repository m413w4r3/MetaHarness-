"""The check-repair sub-domain: failed-check classification and scope."""

from __future__ import annotations

import dataclasses
import hashlib
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
_HUMAN_SCOPE_SOURCE = "human-approved mutable scope"
_CYCLE_SCOPE_SOURCE = "cycle mutable scope"



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
            source=_HUMAN_SCOPE_SOURCE,
        )
    raw_base = payload.get("base_mutable_scope")
    raw_added = payload.get("added_paths")
    raw_effective = payload.get("effective_mutable_scope")
    if not all(isinstance(value, list) for value in (raw_base, raw_added, raw_effective)):
        raise ResumeIntegrityError("check-repair scope artifact is malformed")
    if any(not isinstance(path, str) for paths in (raw_base, raw_added, raw_effective) for path in paths):
        raise ResumeIntegrityError("check-repair scope artifact contains invalid paths")
    if any(
        not path or path.startswith("/") or "\\" in path
        or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
        for paths in (raw_base, raw_added, raw_effective) for path in paths
    ):
        raise ResumeIntegrityError("check-repair scope artifact contains unsafe paths")
    parsed_base = tuple(sorted(set(raw_base)))
    parsed_added = tuple(sorted(set(raw_added)))
    parsed_effective = tuple(sorted(set(raw_effective)))
    if (
        raw_base != list(parsed_base)
        or raw_added != list(parsed_added)
        or raw_effective != list(parsed_effective)
    ):
        raise ResumeIntegrityError("check-repair scope artifact is not canonical")
    if parsed_base != base or parsed_effective != tuple(sorted(set(parsed_base) | set(parsed_added))):
        raise ResumeIntegrityError("check-repair scope artifact does not match its base scope")
    policy = payload.get("policy")
    bound = payload.get("bound")
    source = payload.get("source")
    if policy != policy_config.policy or bound != policy_config.max_added_paths or not isinstance(source, str):
        raise ResumeIntegrityError("check-repair scope policy changed")
    if source not in {_HUMAN_SCOPE_SOURCE, _AUTO_BOUNDED_SOURCE}:
        raise ResumeIntegrityError("check-repair scope artifact has invalid provenance")
    if parsed_added and source != _AUTO_BOUNDED_SOURCE:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    if not parsed_added and source != _HUMAN_SCOPE_SOURCE:
        raise ResumeIntegrityError("check-repair base scope has an invalid provenance")
    if len(parsed_added) > policy_config.max_added_paths:
        raise ResumeIntegrityError("check-repair scope bound was exceeded")
    return CheckRepairScope(parsed_base, parsed_added, parsed_effective, policy, bound, source)


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
    )


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
            source=_AUTO_BOUNDED_SOURCE if added else _HUMAN_SCOPE_SOURCE,
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
        if not evidence.deterministic_passed or evidence.staged_tree_sha is None:
            raise PipelineFailure("DETERMINISTIC_GATE_FAILED", ", ".join(evidence.failures))
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
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 1
                or stored.get("review_cycle") != cycle_plan.cycle.number
                or not all(_is_object_id(stored.get(key)) for key in ("tree_sha", "commit_sha", "parent_sha"))
                or stored.get("stage") != stage.value
                or stored.get("acceptance_kind") not in {"existing-head", "repair", "semantic-revision"}
                or not isinstance(stored.get("commit_created"), bool)
                or stored.get("tree_sha") != evidence.staged_tree_sha
                or stored.get("mutable_scope") != list(authority.effective_paths)
                or stored.get("mutable_scope_sha256") != authority.sha256
            ):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "gate acceptance does not match evidence")
            if current_head(worktree) != stored["commit_sha"]:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted gate HEAD moved")
            if (
                resolve_tree(worktree, stored["commit_sha"]) != stored["tree_sha"]
                or commit_parents(worktree, stored["commit_sha"]) != (stored["parent_sha"],)
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
        if current_tree == evidence.staged_tree_sha:
            parents = commit_parents(worktree, head)
            if len(parents) != 1:
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "accepted HEAD has no single parent")
            commit_sha, parent_sha = head, parents[0]
            parent_tree = resolve_tree(worktree, parent_sha)
            attempts = self._check_repair_attempts(
                ctx.run_dir, cycle_plan.cycle.number, stage,
            )
            revision = self._load_revision(
                semantic_revision_dir(ctx.run_dir, cycle_plan.cycle.number)
            )
            last_attempt = attempts[-1] if attempts else None
            recovered_repair = bool(
                last_attempt is not None
                and last_attempt.tree_before == parent_tree
                and last_attempt.tree_after == evidence.staged_tree_sha
                and last_attempt.tree_before != last_attempt.tree_after
            )
            recovered_revision = bool(
                stage in {GateStage.POST_SEMANTIC_REVISION, GateStage.POST_REVIEW_IMPLEMENTATION}
                and revision is not None
                and revision.tree_before != revision.tree_after
                and revision.tree_after == evidence.staged_tree_sha
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
            "schema_version": 1,
            "review_cycle": cycle_plan.cycle.number,
            "stage": stage.value,
            "tree_sha": evidence.staged_tree_sha,
            "commit_sha": commit_sha,
            "parent_sha": parent_sha,
            "commit_created": commit_created,
            "acceptance_kind": acceptance_kind,
            "mutable_scope": list(authority.effective_paths),
            "mutable_scope_sha256": authority.sha256,
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
