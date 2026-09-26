"""The single effective authority of one approved step.

The approved plan and its step contract are immutable historical evidence.
Validated contract repairs may change what a step is allowed to do; after
them, every operational decision of that step (worker prompt and mutable
paths, rollback, verification, commit gate, accepted record, resume) must use
exactly one :class:`EffectiveStepAuthority`, resolved from durable artifacts
by :func:`resolve_effective_step_authority`.

The resolver never falls back to an older authority: a repair that claims to
be validated but whose artifacts do not prove it is a
``RESUME_INTEGRITY_FAILURE``.  It does not validate a planner answer either;
that stays the contract repair transaction's job.  It only re-proves, from
hashes and the immutable step identity, which validated repair is in force.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..models import ImplementationStep
from ..planning.contract_repair import StepRepairIdentity
from ..planning.protocol import V2PlanParseError, parse_step_contract_repair
from ..result import atomic_write_text
from . import contract_repair
from .shared import StepExecutionOutcome

AUTHORITY_SCHEMA_VERSION = 1
SOURCE_APPROVED = "approved"
SOURCE_CONTRACT_REPAIR = "contract_repair"
STEP_AUTHORITY_NAME = "step_authority.json"
STEP_CANDIDATE_NAME = "step_candidate.json"
STEP_ACCEPTANCE_NAME = "step_acceptance.json"
STEP_CANDIDATE_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 256 * 1024
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StepAuthorityError(Exception):
    """Durable authority evidence does not prove one effective authority."""

    code = "RESUME_INTEGRITY_FAILURE"


def mutable_paths(step: ImplementationStep) -> tuple[str, ...]:
    return tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def canonical_sha256(value: Any) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


@dataclass(frozen=True)
class EffectiveStepAuthority:
    """What one step may do after every validated repair, and why."""

    step_id: str
    title: str
    execution_class: str
    depends_on: str | None
    effective_step: ImplementationStep = field(repr=False)
    effective_contract: str = field(repr=False)
    authority_source: str
    approved_contract_sha256: str
    effective_contract_sha256: str
    approved_mutable_paths: tuple[str, ...]
    repair_slot: int | None = None
    # ``(slot, repaired_contract_sha256, validation_sha256)`` of every repair.
    repair_chain: tuple[tuple[int, str, str], ...] = ()
    # The tree every repair of the chain was validated on.
    tree_sha: str | None = None

    @property
    def read_set(self) -> tuple[str, ...]:
        return self.effective_step.read_set

    @property
    def write_set(self) -> tuple[str, ...]:
        return self.effective_step.write_set

    @property
    def create_set(self) -> tuple[str, ...]:
        return self.effective_step.create_set

    @property
    def delete_set(self) -> tuple[str, ...]:
        return self.effective_step.delete_set

    @property
    def mutable_scope(self) -> tuple[str, ...]:
        return mutable_paths(self.effective_step)

    @property
    def added_mutable_paths(self) -> tuple[str, ...]:
        """Paths the validated repairs added to the approved scope."""

        return tuple(sorted(set(self.mutable_scope) - set(self.approved_mutable_paths)))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "step_id": self.step_id,
            "title": self.title,
            "execution_class": self.execution_class,
            "depends_on": self.depends_on,
            "read_set": list(self.read_set),
            "write_set": list(self.write_set),
            "create_set": list(self.create_set),
            "delete_set": list(self.delete_set),
            "authority_source": self.authority_source,
            "approved_contract_sha256": self.approved_contract_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
            "repair_slot": self.repair_slot,
            "repair_chain": [list(item) for item in self.repair_chain],
            "tree_sha": self.tree_sha,
        }

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def summary(self) -> dict[str, Any]:
        """Bounded diagnostics and durable references; recomputed on resume."""

        return {
            "effective_authority_sha256": self.authority_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
            "approved_contract_sha256": self.approved_contract_sha256,
            "authority_source": self.authority_source,
            "repair_slot": self.repair_slot,
            "repair_chain": [
                {"slot": slot, "repaired_contract_sha256": contract, "validation_sha256": validation}
                for slot, contract, validation in self.repair_chain
            ],
            "tree_sha": self.tree_sha,
            "approved_mutable_paths": list(self.approved_mutable_paths),
            "added_mutable_paths": list(self.added_mutable_paths),
            "effective_mutable_paths": list(self.mutable_scope),
        }


@dataclass(frozen=True)
class EffectiveStepExecution:
    """A worker outcome together with the exact authority it executed under."""

    outcome: StepExecutionOutcome
    authority: EffectiveStepAuthority


def approved_step_authority(
    step: ImplementationStep, approved_contract: str,
) -> EffectiveStepAuthority:
    digest = _sha256_text(approved_contract)
    return EffectiveStepAuthority(
        step_id=step.id, title=step.title,
        execution_class=step.execution_class.value, depends_on=step.depends_on,
        effective_step=step, effective_contract=approved_contract,
        authority_source=SOURCE_APPROVED,
        approved_contract_sha256=digest, effective_contract_sha256=digest,
        approved_mutable_paths=mutable_paths(step),
    )


def _refuse(slot: Path, message: str) -> StepAuthorityError:
    return StepAuthorityError(f"contract repair {slot.name}: {message}")


def resolve_effective_step_authority(
    artifact_dir: Path,
    original_step: ImplementationStep,
    approved_contract: str,
    *,
    max_read_paths_per_step: int,
    expected_plan_step_count: int | None = None,
    expected_tree_sha: str | None = None,
    authorize_added: Callable[[Path, list[str]], None] | None = None,
) -> EffectiveStepAuthority:
    """Replay the chain of validated repairs from the approved authority.

    A slot is authority only when its transaction is ``validated`` or
    ``completed`` (or a pre-transaction slot whose validation says so).
    Each such slot must prove: its repaired contract hash, that it was
    opened on the previous authority's contract, the tree it was validated on,
    the immutable step identity, no removed mutable path, and exactly the
    added paths it recorded.  ``authorize_added`` re-checks the scope policy
    of each addition.  Any failure raises :class:`StepAuthorityError`.
    """

    current = approved_step_authority(original_step, approved_contract)
    identity = StepRepairIdentity.of(original_step, expected_plan_step_count)
    chain: list[tuple[int, str, str]] = []
    tree = expected_tree_sha
    pending_seen: Path | None = None
    for directory in contract_repair.repair_dirs(artifact_dir):
        try:
            transaction = contract_repair.read_transaction(directory)
        except contract_repair.ContractRepairIntegrityError as exc:
            raise StepAuthorityError(str(exc)) from exc
        validation_path = directory / "validation.json"
        validation = _read_json(validation_path)
        validated = isinstance(validation, dict) and validation.get("status") == "validated"
        if transaction is not None:
            if transaction["status"] == contract_repair.SUPERSEDED:
                continue
            if transaction["status"] not in {contract_repair.VALIDATED, contract_repair.COMPLETED}:
                pending_seen = directory
                continue
            if not validated:
                raise _refuse(directory, "transaction is finished without a validated repair")
        elif not validated:
            pending_seen = directory
            continue
        if pending_seen is not None:
            raise _refuse(directory, f"follows the unfinished repair {pending_seen.name}")
        assert isinstance(validation, dict)
        contract_path = directory / "contract.md"
        try:
            contract_bytes = contract_path.read_bytes()
            validation_bytes = validation_path.read_bytes()
            contract_text = contract_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise _refuse(directory, "repaired contract is unreadable") from exc
        contract_sha = _sha256_bytes(contract_bytes)
        if validation.get("repaired_contract_sha256") != contract_sha:
            raise _refuse(directory, "repaired contract hash changed")
        opened_on = [
            value for value in (
                validation.get("original_step_contract_sha256"),
                (transaction or {}).get("original_contract_sha256"),
            ) if value is not None
        ]
        if not opened_on or any(value != current.effective_contract_sha256 for value in opened_on):
            raise _refuse(directory, "was not opened on the previous effective contract")
        trees = [
            value for value in (
                validation.get("current_tree_sha"), (transaction or {}).get("tree_sha"),
            ) if value is not None
        ]
        for value in trees:
            if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
                raise _refuse(directory, "tree binding is malformed")
            if tree is not None and value != tree:
                raise _refuse(directory, "was validated on another tree")
            tree = value
        try:
            repaired = parse_step_contract_repair(
                contract_text,
                max_read_paths_per_step=max_read_paths_per_step,
                expected_step_id=identity.step_id,
                expected_title=identity.title,
                expected_execution_class=identity.execution_class,
                expected_depends_on=identity.depends_on,
                expected_plan_step_count=identity.expected_plan_step_count,
            )
        except V2PlanParseError as exc:
            raise _refuse(directory, f"repaired contract no longer validates: {exc}") from exc
        if repaired.id != original_step.id or identity.violation(repaired) is not None:
            raise _refuse(directory, "repaired contract changed the step identity")
        before = set(current.mutable_scope)
        after = set(mutable_paths(repaired))
        if before - after or set(current.approved_mutable_paths) - after:
            raise _refuse(directory, "removed an authorized mutable path")
        added = sorted(after - before)
        recorded = validation.get("added_mutable_paths")
        if not isinstance(recorded, list) or sorted(recorded) != added:
            raise _refuse(directory, "recorded added paths differ from its contract")
        if added and authorize_added is not None:
            authorize_added(directory, added)
        chain.append((int(directory.name), contract_sha, _sha256_bytes(validation_bytes)))
        current = EffectiveStepAuthority(
            step_id=original_step.id, title=original_step.title,
            execution_class=original_step.execution_class.value,
            depends_on=original_step.depends_on,
            effective_step=repaired, effective_contract=contract_text,
            authority_source=SOURCE_CONTRACT_REPAIR,
            approved_contract_sha256=current.approved_contract_sha256,
            effective_contract_sha256=contract_sha,
            approved_mutable_paths=current.approved_mutable_paths,
            repair_slot=int(directory.name), repair_chain=tuple(chain), tree_sha=tree,
        )
    return current


# -- the approved step, re-read from its hash-bound contract --------------------

_APPROVED_SECTIONS = (
    ("OBJECTIVE", "OBJECTIVE"), ("READ SET", "READ_SET"), ("WRITE SET", "WRITE_SET"),
    ("CREATE SET", "CREATE_SET"), ("DELETE SET", "DELETE_SET"),
    ("INSTRUCTIONS", "INSTRUCTIONS"), ("VERIFY", "VERIFY"), ("FORBIDDEN", "FORBIDDEN"),
)


def approved_step_from_contract(
    contract: str, *, depends_on: str | None, max_read_paths_per_step: int,
) -> tuple[ImplementationStep, int]:
    """Re-read an approved ``META IMPLEMENTATION STEP v1`` contract.

    The contract is MetaHarness' own rendering (hash-bound by the approved
    bundle); its fixed section order is the only structure relied on.  The
    result is parsed by the same strict repair parser, never trusted as text.
    """

    text = contract.replace("\r\n", "\n")
    if not text.startswith("META IMPLEMENTATION STEP v1\n\n"):
        raise V2PlanParseError("approved step contract header is invalid")
    header_step = re.search(r"\n\nSTEP\n(S\d{2}) / (\d{2})\n\nTITLE\n(.+?)\n\nEXECUTION CLASS\n([A-Z]+)\n\n", text)
    if header_step is None:
        raise V2PlanParseError("approved step contract identity is invalid")
    step_id, count, title, execution_class = header_step.groups()
    position = header_step.end() - 2
    bodies: dict[str, str] = {}
    labels = [label for label, _ in _APPROVED_SECTIONS] + ["END META IMPLEMENTATION STEP"]
    for index, (label, _canonical) in enumerate(_APPROVED_SECTIONS):
        opening = f"\n\n{label}\n"
        if not text.startswith(opening, position):
            raise V2PlanParseError(f"approved step contract section {label} is missing")
        start = position + len(opening)
        closing = f"\n\n{labels[index + 1]}\n"
        end = text.find(closing, start)
        if end < 0:
            raise V2PlanParseError(f"approved step contract section {label} is unterminated")
        bodies[label] = text[start:end]
        position = end
    repair = "\n\n".join((
        "META STEP CONTRACT REPAIR v1",
        f"STEP_ID: {step_id}",
        f"TITLE: {title}",
        f"EXECUTION_CLASS: {execution_class}",
        f"DEPENDS_ON: {depends_on or 'NONE'}",
        *(f"{canonical}\n{bodies[label]}" for label, canonical in _APPROVED_SECTIONS),
        "END META STEP CONTRACT REPAIR",
    )) + "\n"
    step = parse_step_contract_repair(
        repair, max_read_paths_per_step=max_read_paths_per_step,
        expected_step_id=step_id, expected_title=title,
        expected_execution_class=execution_class,
        expected_depends_on=depends_on or "NONE",
        expected_plan_step_count=int(count),
    )
    return step, int(count)


# -- the durable step candidate ----------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_step_candidate(
    *, run_id: str, cycle: int, step_id: str, parent_head_sha: str,
    tree_before: str, tree_after: str, changed_paths: Sequence[str], profile_id: str,
    authority: EffectiveStepAuthority, verification: Mapping[str, Any],
    step_record_sha256: str, final_report_sha256: str | None,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STEP_CANDIDATE_SCHEMA_VERSION,
        "run_id": run_id,
        "cycle": cycle,
        "step_id": step_id,
        "parent_head_sha": parent_head_sha,
        "tree_before": tree_before,
        "tree_after": tree_after,
        "changed_paths": sorted(changed_paths),
        "profile_id": profile_id,
        "effective_authority_sha256": authority.authority_sha256,
        "effective_contract_sha256": authority.effective_contract_sha256,
        "approved_contract_sha256": authority.approved_contract_sha256,
        "authority_source": authority.authority_source,
        "repair_slot": authority.repair_slot,
        "effective_mutable_paths": list(authority.mutable_scope),
        "verification": dict(verification),
        "outcome": {
            "step_record": "step.json",
            "step_record_sha256": step_record_sha256,
            "final_report": "agent.final.md",
            "final_report_sha256": final_report_sha256,
        },
        "source": source,
        "created_at": _now(),
    }
    payload["candidate_sha256"] = canonical_sha256(payload)
    return payload


def write_step_candidate(step_dir: Path, payload: Mapping[str, Any]) -> str:
    """Write the candidate, read it back and return the sha of its bytes."""

    path = step_dir / STEP_CANDIDATE_NAME
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)
    if read_step_candidate(step_dir) != dict(payload):
        raise StepAuthorityError("step candidate could not be durably written")
    return _sha256_bytes(path.read_bytes())


def read_step_candidate(step_dir: Path) -> dict[str, Any] | None:
    """The self-hashed candidate, or ``None`` when absent; corruption raises."""

    path = step_dir / STEP_CANDIDATE_NAME
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != STEP_CANDIDATE_SCHEMA_VERSION:
        raise StepAuthorityError("step candidate is unreadable or has an unknown schema")
    body = {key: value for key, value in payload.items() if key != "candidate_sha256"}
    if payload.get("candidate_sha256") != canonical_sha256(body):
        raise StepAuthorityError("step candidate hash changed")
    changed = payload.get("changed_paths")
    outcome = payload.get("outcome")
    if (
        not all(
            isinstance(payload.get(key), str) and _OBJECT_ID.fullmatch(payload[key])
            for key in ("parent_head_sha", "tree_before", "tree_after")
        )
        or not all(
            isinstance(payload.get(key), str) and _SHA256.fullmatch(payload[key])
            for key in ("effective_authority_sha256", "effective_contract_sha256")
        )
        or not isinstance(changed, list) or not changed
        or any(not isinstance(item, str) for item in changed)
        or not isinstance(outcome, dict)
        or not isinstance(payload.get("step_id"), str)
    ):
        raise StepAuthorityError("step candidate is incomplete")
    return payload


def write_authority_diagnostic(step_dir: Path, authority: EffectiveStepAuthority) -> None:
    """Advisory copy of the authority an attempt ran with; never trusted."""

    atomic_write_text(
        step_dir / STEP_AUTHORITY_NAME,
        json.dumps({"step_id": authority.step_id, **authority.summary()},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


# -- scope policy of repair additions ---------------------------------------------


def strict_repair_scope_authorizer(policy: Any) -> Callable[[Path, list[str]], None]:
    """Re-check a validated repair's additions against the frozen scope policy.

    Used where no operator interaction can happen (resume, status): a
    missing approval of a validated repair is an integrity failure.
    """

    from ..approval import ApprovalDecision, ApprovalError, read_scope_approval

    def authorize(directory: Path, added: list[str]) -> None:
        if policy.policy == "deny-expansion":
            raise _refuse(directory, "scope expansion is denied by the run policy")
        if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
            delta_path = directory / "scope_delta.json"
            delta = _read_json(delta_path)
            if not isinstance(delta, dict) or delta.get("added_paths") != sorted(added):
                raise _refuse(directory, "scope delta is malformed")
            try:
                approval = read_scope_approval(
                    directory, expected_sha256=_sha256_bytes(delta_path.read_bytes()),
                )
            except (OSError, ApprovalError) as exc:
                raise _refuse(directory, "scope approval is invalid") from exc
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                raise _refuse(directory, "scope expansion was not approved")

    return authorize


__all__ = [
    "strict_repair_scope_authorizer",
    "EffectiveStepAuthority", "EffectiveStepExecution", "SOURCE_APPROVED", "SOURCE_CONTRACT_REPAIR",
    "STEP_ACCEPTANCE_NAME", "STEP_AUTHORITY_NAME", "STEP_CANDIDATE_NAME",
    "StepAuthorityError", "approved_step_authority", "approved_step_from_contract",
    "build_step_candidate", "canonical_sha256", "mutable_paths", "read_step_candidate",
    "resolve_effective_step_authority", "write_authority_diagnostic", "write_step_candidate",
]
