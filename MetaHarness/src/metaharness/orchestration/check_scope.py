"""The mutable-scope authority of one check-repair gate episode.

The operator approves a cycle-level mutable envelope; this module derives the
one durable mutation authority of a gate episode from it: the scope artifact
each recorded repair pass persisted, the bounded ladder expansion of a not yet
recorded pass, and the evidence-proven additions the failure evidence proves
are implicated.  No model, planner or worker can widen what the operator
already approved: every path list is canonicalized, bounded by the run's
repair-scope policy and validated against the approved envelope.
"""

from __future__ import annotations

import dataclasses
import hashlib

from pathlib import (
    Path,
    PurePosixPath,
)
from typing import (
    Any,
    Sequence,
)
from .check_failure import (
    implicated_repair_paths,
    red_gate_identity,
)
from .pipeline_v2 import (
    check_repair_attempt_dir,
    check_repair_attempts_dir,
)
from .shared import (
    CheckRepairScope,
    GateMutableAuthority,
    json_text,
    read_json_artifact,
)
from ..evidence import EvidenceBundle
from ..models import GateStage
from ..result import atomic_write_text
from ..resume import ResumeIntegrityError
from ..run_options import EffectiveRepairScopePolicy


_EXPANSION_ARTIFACT = "scope-expansion.json"
_SAFE_PATH_PARTS = frozenset({"", ".", ".."})


EVIDENCE_SCOPE_SOURCE = "failure evidence within approved mutable scope"
SCOPE_REQUEST_SOURCE = "explicit META SCOPE REQUEST v1"
HUMAN_SCOPE_SOURCE = "human-approved mutable scope"
CYCLE_SCOPE_SOURCE = "cycle mutable scope"


def _read_scope_artifact(
    directory: Path,
    *,
    fallback_base: Sequence[str],
    policy_config: EffectiveRepairScopePolicy,
) -> CheckRepairScope:
    payload = read_json_artifact(directory / "scope.json", 64 * 1024)
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
        HUMAN_SCOPE_SOURCE, EVIDENCE_SCOPE_SOURCE, SCOPE_REQUEST_SOURCE,
    }
    if not isinstance(source, str) or source not in valid_sources:
        raise ResumeIntegrityError("check-repair scope artifact has invalid provenance")
    if parsed_added and source != SCOPE_REQUEST_SOURCE:
        raise ResumeIntegrityError("check-repair added paths have an invalid provenance")
    if not parsed_added and source not in {HUMAN_SCOPE_SOURCE, EVIDENCE_SCOPE_SOURCE}:
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
    return hashlib.sha256(json_text(list(canonical)).encode("utf-8")).hexdigest()


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
            source=CYCLE_SCOPE_SOURCE,
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
            source=CYCLE_SCOPE_SOURCE,
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
        scope = _read_scope_artifact(
            directory, fallback_base=base, policy_config=policy_config,
        )
        if require_attempt_records:
            attempt = read_json_artifact(directory / "attempt.json", 128 * 1024)
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
            # When the output names no implicated source file, the changed
            # paths of this gate's evidence provide a narrow fallback. A
            # traceback match always takes priority and keeps unrelated cycle
            # changes out of the worker's initial WRITE_SET.
            _implicated, initial = implicated_repair_paths(
                repo=repo, worktree=worktree, tree_sha=tree_sha,
                evidence_dir=evidence_dir, evidence=evidence, approved=approved,
            )
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
            source=SCOPE_REQUEST_SOURCE if added else (
                EVIDENCE_SCOPE_SOURCE if initial else HUMAN_SCOPE_SOURCE
            ),
        )


def _pending_scope_expansion(
    run_dir: Path, cycle: int, stage: GateStage, attempt: int, *,
    base: tuple[str, ...], policy_config: EffectiveRepairScopePolicy,
) -> tuple[str, ...]:
    """The ladder expansion authorized for one not-yet-recorded repair pass."""

    directory = check_repair_attempt_dir(run_dir, cycle, stage, attempt)
    if (directory / "scope.json").is_file():
        # The pass is recorded: its own scope artifact is authoritative.
        return ()
    payload = read_json_artifact(directory / _EXPANSION_ARTIFACT, 64 * 1024)
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
    added = canonical_scope_paths(payload.get("added_paths"), what="gate recovery scope expansion")
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
        source=SCOPE_REQUEST_SOURCE,
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
        payload = read_json_artifact(directory / "scope.json", 64 * 1024)
        if not isinstance(payload, dict):
            continue
        raw = payload.get("effective_repair_scope", payload.get("effective_mutable_scope"))
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            found = frozenset(raw)
    return found


def canonical_scope_paths(paths: Any, *, what: str) -> tuple[str, ...]:
    """The canonical, safe path list of one durable scope expansion."""

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


def evidence_proven_expansion(
    *,
    run_dir: Path,
    cycle: int,
    stage: GateStage,
    repo: Path,
    worktree: Path,
    evidence_dir: Path,
    evidence: EvidenceBundle,
    approved: Sequence[str],
    policy: EffectiveRepairScopePolicy,
) -> tuple[str, ...]:
    """Bounded, evidence-proven additions to the repair scope."""

    approved_scope = tuple(approved)
    if not approved_scope:
        return ()
    tree, _failed = red_gate_identity(evidence)
    _implicated, evidenced = implicated_repair_paths(
        repo=repo, worktree=worktree, tree_sha=tree, evidence_dir=evidence_dir,
        evidence=evidence, approved=approved_scope,
    )
    recorded = _recorded_effective_scope(run_dir, cycle, stage)
    current = set(recorded) if recorded else set(evidenced)
    return tuple(sorted(set(evidenced) - current))[: policy.max_added_paths]


def responsible_step_index(
    *,
    repo: Path,
    worktree: Path,
    evidence_dir: Path,
    evidence: EvidenceBundle,
    approved: Sequence[str],
    steps: Sequence[Any],
) -> int | None:
    """The first approved step whose mutable paths the failure implicates."""

    tree, _failed = red_gate_identity(evidence)
    implicated, _initial = implicated_repair_paths(
        repo=repo, worktree=worktree, tree_sha=tree, evidence_dir=evidence_dir,
        evidence=evidence, approved=approved,
    )
    if not implicated:
        return None
    for index, step in enumerate(steps):
        if set(implicated) & set((*step.write_set, *step.create_set, *step.delete_set)):
            return index
    return None


def authorize_scope_expansion(
    *,
    run_dir: Path,
    cycle: int,
    stage: GateStage | str,
    attempt: int,
    tree: str,
    added_paths: Sequence[str],
    approved: Sequence[str],
    failed: Sequence[str],
    policy: EffectiveRepairScopePolicy,
) -> None:
    """Persist the bounded expansion of one pending repair pass."""

    stage_value = GateStage(stage)
    added = tuple(sorted(set(added_paths)))
    if not added or not set(added).issubset(set(approved)):
        raise ResumeIntegrityError(
            "the gate recovery scope expansion exceeds the approved cycle scope"
        )
    if len(added) > policy.max_added_paths:
        raise ResumeIntegrityError(
            "the gate recovery scope expansion exceeds the configured bound"
        )
    directory = check_repair_attempt_dir(run_dir, cycle, stage_value, attempt)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / _EXPANSION_ARTIFACT, json_text({
        "schema_version": 1,
        "attempt": attempt,
        "tree": tree,
        "failed_check_ids": list(failed),
        "added_paths": list(added),
        "policy": policy.policy,
        "bound": policy.max_added_paths,
    }))
