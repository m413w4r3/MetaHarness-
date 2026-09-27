"""The approved authority and durable candidate record of one plan step."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from ..models import ImplementationStep
from ..planning.artifacts import read_approved_step_contract
from ..planning.protocol import V2PlanParseError
from ..result import atomic_write_text
from .pipeline_v2 import PipelineFailure
from .shared import StepExecutionOutcome

if TYPE_CHECKING:  # pragma: no cover - the plan authority is the coordinator
    from .pipeline_v2 import CyclePlan


STEP_AUTHORITY_NAME = "step_authority.json"
STEP_CANDIDATE_NAME = "step_candidate.json"
STEP_ACCEPTANCE_NAME = "step_acceptance.json"
STEP_CANDIDATE_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 256 * 1024
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StepAuthorityError(Exception):
    """Durable candidate evidence is incomplete or inconsistent."""

    code = "RESUME_INTEGRITY_FAILURE"


def mutable_paths(step: ImplementationStep) -> tuple[str, ...]:
    return tuple(sorted({*step.write_set, *step.create_set, *step.delete_set}))


def future_step_ownership(
    steps: Sequence[ImplementationStep], index: int,
) -> dict[str, tuple[str, ...]]:
    """Paths assigned to later approved steps, for verification dependencies."""

    return {
        step.id: paths
        for step in steps[index + 1:]
        if (paths := mutable_paths(step))
    }


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


@dataclass(frozen=True)
class EffectiveStepAuthority:
    """The plan step and its hash-bound, approved contract."""

    step_id: str
    title: str
    execution_class: str
    depends_on: str | None
    effective_step: ImplementationStep = field(repr=False)
    effective_contract: str = field(repr=False)
    approved_contract_sha256: str
    effective_contract_sha256: str

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

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "step_id": self.step_id,
            "title": self.title,
            "execution_class": self.execution_class,
            "depends_on": self.depends_on,
            "read_set": list(self.read_set),
            "write_set": list(self.write_set),
            "create_set": list(self.create_set),
            "delete_set": list(self.delete_set),
            "approved_contract_sha256": self.approved_contract_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
        }

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def summary(self) -> dict[str, Any]:
        return {
            "effective_authority_sha256": self.authority_sha256,
            "effective_contract_sha256": self.effective_contract_sha256,
            "approved_contract_sha256": self.approved_contract_sha256,
            "effective_mutable_paths": list(self.mutable_scope),
        }


@dataclass(frozen=True)
class EffectiveStepExecution:
    """A worker outcome together with the authority it executed under."""

    outcome: StepExecutionOutcome
    authority: EffectiveStepAuthority


def approved_step_contract(cycle_plan: "CyclePlan", step: ImplementationStep) -> str:
    """The hash-bound approved contract: immutable evidence of the step."""

    try:
        return read_approved_step_contract(
            cycle_plan.contracts_dir, cycle_plan.bundle, step.id,
        )
    except (V2PlanParseError, OSError, UnicodeError) as exc:
        raise PipelineFailure("PLAN_APPROVAL_INVALID", str(exc), step_id=step.id) from exc


def approved_step_authority(
    step: ImplementationStep, approved_contract: str,
) -> EffectiveStepAuthority:
    digest = hashlib.sha256(approved_contract.encode("utf-8")).hexdigest()
    return EffectiveStepAuthority(
        step_id=step.id, title=step.title,
        execution_class=step.execution_class.value, depends_on=step.depends_on,
        effective_step=step, effective_contract=approved_contract,
        approved_contract_sha256=digest, effective_contract_sha256=digest,
    )


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
        "effective_mutable_paths": list(authority.mutable_scope),
        "verification": dict(verification),
        "outcome": {
            "step_record": "step.json",
            "step_record_sha256": step_record_sha256,
            "final_report": "agent.final.md",
            "final_report_sha256": final_report_sha256,
        },
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Advisory copy of the approved authority an attempt ran with."""

    atomic_write_text(
        step_dir / STEP_AUTHORITY_NAME,
        json.dumps({"step_id": authority.step_id, **authority.summary()},
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


__all__ = [
    "EffectiveStepAuthority", "EffectiveStepExecution", "STEP_ACCEPTANCE_NAME",
    "STEP_AUTHORITY_NAME", "STEP_CANDIDATE_NAME", "StepAuthorityError",
    "approved_step_authority", "approved_step_contract", "build_step_candidate",
    "canonical_sha256", "mutable_paths", "read_step_candidate",
    "write_authority_diagnostic", "write_step_candidate",
]
