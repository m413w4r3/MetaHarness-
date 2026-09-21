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
from .revision import (
    _render_revision_template,
)
from .shared import (
    CheckRepairScope,
    _PROMPTS_DIR,
    _check_payload,
    _json_text,
    _read_json_artifact,
)
from ..evidence import (
    DIFF_TOO_LARGE,
    EvidenceBundle,
    SECRET_IN_DIFF,
    SECRET_IN_STAGED_BLOB,
    UNREVIEWABLE_TEXT_DIFF,
    UNSCANNABLE_STAGED_BLOB,
)
from ..gitops import tracked_files_in_tree
from ..planning_v2 import TaskPlanV2
from ..resume import (
    ResumeIntegrityError,
    ResumePhase,
    phase_index,
)
from ..run_options import EffectiveRepairScopePolicy
from ..validation import check_result_json


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


_LEGACY_DIRECT_FAILURES = _DIRECT_FAILURES | frozenset({DIFF_TOO_LARGE})


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
        if item in _DIRECT_FAILURES
        or any(item.startswith(f"{prefix}:") for prefix in _DIRECT_FAILURES)
        or item.startswith("CHECK_MUTATED:")
        or item.startswith("CHECK_TIMEOUT:")
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
    previous_report: str,
    added_paths: Sequence[str] = (),
    scope_source: str = "",
) -> str:
    """Build the bounded prompt for one automatic check-repair pass."""

    values = {
        "{{CHECK_DETAILS}}": _check_repair_problem_context(evidence),
        "{{CHANGED_FILES}}": changed_files,
        "{{MUTABLE_SCOPE}}": _json_text(mutable_scope),
        "{{SECOND_PASS_NOTE}}": _SAME_SCOPE_RETRY_NOTE
        if scope_source == _SAME_SCOPE_RETRY_SOURCE else "",
    }
    template = (_PROMPTS_DIR / "check_repair.txt").read_text(
        encoding="utf-8"
    )
    return _render_revision_template(template, values, name="check_repair")


_SAME_SCOPE_RETRY_NOTE = """
This is the second and final bounded automatic check-repair pass.

No mutable-scope expansion was required.

Correct the remaining deterministic failures inside the exact existing
mutable scope.

Do not repeat unrelated changes from the previous repair.
"""


# The two provenances a *second* bounded check-repair scope may carry.  They
# are durable artifact values: never rename them, a stored run depends on the
# exact string.
_AUTO_BOUNDED_SOURCE = "auto-bounded failing-test evidence"


_SAME_SCOPE_RETRY_SOURCE = "bounded same-scope retry"


_SECOND_SCOPE_SOURCES = frozenset({_AUTO_BOUNDED_SOURCE, _SAME_SCOPE_RETRY_SOURCE})


def _check_repair_scope_payload(scope: CheckRepairScope) -> dict[str, Any]:
    return {
        "base_paths": list(scope.base_paths),
        "added_paths": list(scope.added_paths),
        "effective_paths": list(scope.effective_paths),
        "policy": scope.policy,
        "bound": scope.bound,
        "source": scope.source,
    }


def _second_check_repair_state(scope: CheckRepairScope) -> dict[str, Any]:
    """State/diagnostics facts about the second bounded check-repair pass.

    ``expanded_attempted`` is kept for durable compatibility with runs and
    UIs that predate the same-scope retry; ``scope_expanded`` is the fact
    that actually distinguishes a true expansion from a same-scope retry.
    """

    return {
        "expanded_attempted": True,
        "second_check_repair_attempted": True,
        "scope_expanded": scope.source != _SAME_SCOPE_RETRY_SOURCE,
        **_check_repair_scope_payload(scope),
    }


def _read_check_repair_scope(
    directory: Path,
    *,
    fallback_base: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
    allowed_added_sources: frozenset[str] = frozenset({_AUTO_BOUNDED_SOURCE}),
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
    if parsed_added and source not in allowed_added_sources:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    return CheckRepairScope(parsed_base, parsed_added, parsed_effective, policy, bound, source)


def _validate_expanded_check_repair_scope(
    directory: Path,
    *,
    repo: Path,
    tree_sha: str,
    normal_scope: CheckRepairScope,
    policy_config: EffectiveRepairScopePolicy,
) -> CheckRepairScope:
    """Validate the durable scope of the second bounded check-repair pass.

    Two forms are legitimate, and each is validated strictly.

    Form A, a true expansion (``auto-bounded failing-test evidence``): every
    added path must be tracked in the checkpoint tree, must be an
    auto-expandable test path, must not already be in the base scope, and the
    policy and its bound must still hold.

    Form B, a same-scope retry (``bounded same-scope retry``): the scope must
    be *exactly* the one the first repair already held.  One extra path, or
    any other difference, is a resume integrity failure: the second pass
    never creates authority.
    """

    scope = _read_check_repair_scope(
        directory, fallback_base=normal_scope.base_paths,
        policy_config=policy_config, allowed_added_sources=_SECOND_SCOPE_SOURCES,
    )
    if scope.source == _SAME_SCOPE_RETRY_SOURCE:
        if (
            scope.added_paths != tuple(sorted(set(normal_scope.added_paths)))
            or scope.effective_paths != tuple(sorted(set(normal_scope.effective_paths)))
        ):
            raise ResumeIntegrityError(
                "same-scope check-repair retry scope is not the first repair scope"
            )
        return scope
    if not scope.added_paths:
        raise ResumeIntegrityError("expanded check-repair scope has no added paths")
    if scope.policy != "auto-bounded" or len(scope.added_paths) > scope.bound:
        raise ResumeIntegrityError("expanded check-repair scope violates its bound")
    tracked = frozenset(tracked_files_in_tree(repo, tree_sha))
    if any(
        path in scope.base_paths
        or path not in tracked
        or not _is_auto_expandable_test_path(path)
        for path in scope.added_paths
    ):
        raise ResumeIntegrityError("expanded check-repair scope contains an invalid path")
    return scope


def _expanded_scope_is_applicable(
    *,
    checkpoint_phase: ResumePhase,
    expansion_phase: ResumePhase,
    scope_path: Path,
) -> bool:
    """Whether an expanded check-repair scope still carries authority.

    A validated expanded scope is cumulative: it stays part of the candidate
    tree's authority for every phase at or after the expansion, up to
    publication.  Applicability is decided from the canonical phase order and
    the durable artifact, never from a hand-maintained list of downstream
    phases -- such a list silently forgets every phase added later.
    """

    if phase_index(checkpoint_phase) < phase_index(expansion_phase):
        # The expansion has not happened yet: no additional authority.
        return False
    if checkpoint_phase is expansion_phase:
        # At the expansion phase itself the artifact is mandatory; its absence
        # must surface as a strict validation failure, not as a silent skip.
        return True
    # Downstream, only a durable expansion carries authority.  A run that
    # never expanded gains nothing.
    return scope_path.is_file()


@dataclasses.dataclass(frozen=True)
class CheckRepairCoordinator:
    """Owns every bounded check-repair scope decision of one cycle.

    The operator's repair-scope policy is the only state it holds, injected
    explicitly: it never sees the ``Orchestrator``.  It decides the normal
    scope, the one second bounded pass (auto-bounded expansion or the
    same-scope retry) and which durable scope outranks a recomputation.  It
    owns neither candidate publication nor the reviewer, and it never widens
    a scope beyond the injected policy's bound.
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
    ) -> CheckRepairScope:
        base_paths = tuple(sorted(set(base_mutable_scope)))
        scope_policy = self.effective_repair_scope
        policy = scope_policy.policy
        bound = scope_policy.max_added_paths
        candidates = (
            _check_repair_scope_candidates(
                repo=repo,
                worktree=worktree,
                tree_sha=tree_sha,
                run_dir=run_dir,
                evidence=evidence,
                base_mutable_scope=base_paths,
            )
            if policy == "auto-bounded" else []
        )
        added = tuple(candidates) if len(candidates) <= bound else ()
        return CheckRepairScope(
            base_paths=base_paths,
            added_paths=added,
            effective_paths=tuple(sorted(set(base_paths) | set(added))),
            policy=policy,
            bound=bound,
            source=(
                _AUTO_BOUNDED_SOURCE
                if added else "human-approved mutable scope"
            ),
        )

    def second_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        tree_sha: str,
        run_dir: Path,
        evidence: EvidenceBundle,
        normal_scope: CheckRepairScope,
        expanded_dir: Path,
    ) -> CheckRepairScope | None:
        """The scope of the one second bounded check-repair pass, or ``None``.

        This is the single decision point for "may the corrective Claude pass
        run once more", and the retry budget is deliberately independent from
        the mutable scope.  A scope expansion is one possible *outcome* of
        this decision, never its precondition: a soft deterministic failure
        whose fix already lives inside the authorized paths earns a second
        pass with exactly that scope.

        It stays model-free and fail-closed; every condition is a durable
        fact -- the failure family, the scope the first repair already held,
        the operator's policy and its bound, and whether a second pass
        already produced a report.  ``None`` means the deterministic gate is
        final.
        """

        if not _soft_check_failures(evidence):
            return None
        if _hard_integrity_failures(evidence):
            return None
        # One second pass per cycle, even across a crash: never a third.
        if (expanded_dir / "report.json").exists():
            return None
        policy_config = self.effective_repair_scope
        # No new path, no new authority: exactly what the first repair held.
        same_scope = CheckRepairScope(
            base_paths=normal_scope.base_paths,
            added_paths=normal_scope.added_paths,
            effective_paths=normal_scope.effective_paths,
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source=_SAME_SCOPE_RETRY_SOURCE,
        )
        if policy_config.policy != "auto-bounded":
            # ``deny-expansion`` and ``require-approval`` forbid *growing* the
            # mutable scope.  Neither forbids one more bounded correction
            # inside the scope a human already approved.
            return same_scope
        candidates = _check_repair_scope_candidates(
            repo=repo,
            worktree=worktree,
            tree_sha=tree_sha,
            run_dir=run_dir,
            evidence=evidence,
            # Only genuinely new paths are candidates; the paths the first
            # repair already earned are never re-added, and never lost.
            base_mutable_scope=normal_scope.effective_paths,
        )
        if not candidates:
            return same_scope
        added = tuple(sorted(set(normal_scope.added_paths) | set(candidates)))
        if len(added) > policy_config.max_added_paths:
            # The bound caps the expansion, not the retry.
            return same_scope
        return CheckRepairScope(
            base_paths=normal_scope.base_paths,
            added_paths=added,
            effective_paths=tuple(sorted(
                set(normal_scope.base_paths) | set(added)
            )),
            policy=policy_config.policy,
            bound=policy_config.max_added_paths,
            source=_AUTO_BOUNDED_SOURCE,
        )

    def durable_normal_scope(
        self,
        run_dir: Path,
        *,
        cycle: int,
        base_paths: Sequence[str],
        scope: CheckRepairScope | None = None,
    ) -> CheckRepairScope:
        """The first repair's scope, from memory or from its durable artifact.

        A resume that starts at the second repair has no in-band scope, so the
        artifact the first repair published is the authority for what that
        pass was allowed to touch.
        """

        if scope is not None:
            return scope
        return _read_check_repair_scope(
            run_dir / "revision" / "check-repair" / f"C0{cycle}",
            fallback_base=base_paths,
            policy_config=self.effective_repair_scope,
        )

    @staticmethod
    def normal_scope(
        scope: CheckRepairScope | None, base_paths: Sequence[str],
        policy: EffectiveRepairScopePolicy,
    ) -> CheckRepairScope:
        """The normal check-repair scope, defaulted to the approved base."""

        if scope is not None:
            return scope
        base = tuple(sorted(set(base_paths)))
        return CheckRepairScope(
            base_paths=base, added_paths=(), effective_paths=base,
            policy=policy.policy, bound=policy.max_added_paths,
            source="human-approved mutable scope",
        )

    def durable_second_scope(
        self,
        *,
        repo: Path,
        worktree: Path,
        run_dir: Path,
        evidence: EvidenceBundle,
        normal_scope: CheckRepairScope,
        expanded_dir: Path,
    ) -> tuple[CheckRepairScope | None, bool]:
        """Reuse a durable second-pass scope, or decide a new one.

        A crash may have landed between ``scope.json`` and the second
        corrective Claude pass while the checkpoint still names the retry
        checks.  The durable scope then outranks any recomputation: the model
        is never recalled with a scope different from the one already
        published for it.  Returns ``(scope, archive_required)``;
        ``archive_required`` is false for a reused scope, whose attempt was
        already archived.
        """

        if (expanded_dir / "scope.json").is_file():
            scope = _validate_expanded_check_repair_scope(
                expanded_dir, repo=repo, tree_sha=evidence.staged_tree_sha,
                normal_scope=normal_scope,
                policy_config=self.effective_repair_scope,
            )
            # One second pass per cycle, even across a crash.
            if (expanded_dir / "report.json").exists():
                return None, False
            return scope, False
        return self.second_scope(
            repo=repo, worktree=worktree, tree_sha=evidence.staged_tree_sha,
            run_dir=run_dir, evidence=evidence, normal_scope=normal_scope,
            expanded_dir=expanded_dir,
        ), True
