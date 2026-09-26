"""StepContractRepairPlanner: one bounded, durable repair of a step contract.

Owns the repair identity, the repair request envelopes and the bounded output
correction of one semantic repair slot.  It contains no initial-planner logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..models import ImplementationStep
from ..prompt_contracts import payload_for_rendered_request, write_prompt_diagnostics
from ..repository_topology import (
    invalid_paths,
    render_invalid_path_candidates,
    render_path_candidates,
    render_repository_path_facts,
    topology_payload,
)
from ..result import atomic_write_text
from ..usage import completion_usage, write_usage_artifact
from . import TextCompletionClient
from .artifacts import (
    STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
    StepContractRepairArtifactError,
    StepRepairAttemptFiles,
    read_bounded_json,
    render_json,
    sha256_bytes,
    step_repair_attempt_files,
    step_repair_attempt_state,
)
from .protocol import (
    STEP_REPAIR_END,
    STEP_REPAIR_HEADER,
    V2PlanParseError,
    parse_step_contract_repair,
    render_repaired_step_contract,
)

_REPAIR_PREVIOUS_RAW_CHARS = 24_000
_REPAIR_ERROR_CHARS = 1_000


@dataclass(frozen=True)
class StepRepairIdentity:
    """The immutable identity of the step whose contract is repaired.

    It is handed to the planner explicitly: the approved contract renders its
    step as ``<id> / <count>``, which is display text, never an identity.
    """

    step_id: str
    title: str
    execution_class: str
    depends_on: str
    expected_plan_step_count: int | None = None

    @classmethod
    def of(
        cls, step: ImplementationStep, expected_plan_step_count: int | None = None,
    ) -> "StepRepairIdentity":
        return cls(
            step.id, step.title, step.execution_class.value,
            step.depends_on or "NONE", expected_plan_step_count,
        )

    def fields(self) -> str:
        return (
            f"STEP_ID: {self.step_id}\nTITLE: {self.title}\n"
            f"EXECUTION_CLASS: {self.execution_class}\nDEPENDS_ON: {self.depends_on}"
        )

    def violation(self, step: ImplementationStep) -> str | None:
        """The first identity field the answer changed; never rewritten."""

        for name, expected, actual in (
            ("STEP_ID", self.step_id, step.id),
            ("TITLE", self.title, step.title),
            ("EXECUTION_CLASS", self.execution_class, step.execution_class.value),
            ("DEPENDS_ON", self.depends_on, step.depends_on or "NONE"),
        ):
            if expected != actual:
                return f"step contract repair {name} changed: expected {expected!r}, got {actual!r}"
        return None


class StepContractRepairOutputInvalid(Exception):
    """Every admitted StepContractRepairPlanner answer was invalid.

    This is a planner protocol failure of one semantic repair slot, never a
    worker contract mismatch.
    """

    code = STEP_CONTRACT_REPAIR_OUTPUT_INVALID

    def __init__(self, detail: str, *, output_attempt: int, corrections: int, limit: int):
        super().__init__(detail)
        self.detail = detail
        self.output_attempt = output_attempt
        self.corrections = corrections
        self.limit = limit


def _step_repair_template(identity: StepRepairIdentity) -> str:
    return f"""{STEP_REPAIR_HEADER}
{identity.fields()}

OBJECTIVE
<complete objective>

READ_SET
- relative/path :: exact symbol or anchor

WRITE_SET
- relative/path

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. <concrete instruction>

VERIFY
<at most 3 lines>

FORBIDDEN
- <rule>

{STEP_REPAIR_END}"""


_STEP_REPAIR_FORMAT_LIMITS = """FORMAT LIMITS (checked deterministically):
- INSTRUCTIONS: at most 6 numbered operations;
- VERIFY: at most 3 non-empty lines;
- FORBIDDEN: at most 4 non-empty rules;
- every section is present and non-empty; empty sets are written NONE."""


def _step_repair_identity_rules(identity: StepRepairIdentity) -> str:
    return f"""{_STEP_REPAIR_FORMAT_LIMITS}

<IMMUTABLE STEP IDENTITY>
IMMUTABLE STEP ID: {identity.step_id}
IMMUTABLE TITLE: {identity.title}
IMMUTABLE EXECUTION CLASS: {identity.execution_class}
IMMUTABLE DEPENDS_ON: {identity.depends_on}
</IMMUTABLE STEP IDENTITY>

STEP_ID is the bare MetaHarness step identifier only.
For this repair it is exactly `{identity.step_id}`.
Do NOT append the plan step count: `{identity.step_id} / <step count>` is not
canonical, even though CURRENT STEP CONTRACT displays its STEP that way. MetaHarness
accepts that wire-format variation only when the primary ID matches this
immutable identity and the count matches the approved plan.

The following identity fields are immutable and MUST be copied byte-for-byte:
{identity.fields()}"""


def build_step_contract_repair_prompt(
    *, original_spec: str, current_tree_sha: str, original_plan_identity: str,
    current_contract: str, mismatch_explanation: str,
    read_set: str, write_set: str, create_set: str, delete_set: str,
    identity: StepRepairIdentity,
    future_ownership: str = "NONE",
    repository_evidence: str = "NONE",
    repository_path_candidates: str = "NONE",
    failure_evidence: str = "NONE",
) -> str:
    """Build the bounded planner transaction for one contract repair.

    ``mismatch_explanation`` names why the current contract could not be
    executed; ``failure_evidence`` carries the bounded deterministic-gate
    facts when a red gate, not a worker, opened the repair.  Both are inputs
    of the same single repair protocol.
    """

    return f"""You are the MetaHarness StepContractRepairPlanner.

Repair only the current approved step contract so one implementation worker
can execute it deterministically. ORIGINAL SPEC remains semantic authority.
Do not reinterpret the SPEC, hide the mismatch, move work to another step, or
broaden mutable scope unless a genuinely required path is explicitly requested.
WRITE_SET, CREATE_SET and DELETE_SET are absolute until MetaHarness applies its
scope policy. Preserve step identity, dependency and required checks.

<ORIGINAL SPEC AUTHORITY>
{original_spec}
</ORIGINAL SPEC AUTHORITY>

<CURRENT TREE SHA>
{current_tree_sha}
</CURRENT TREE SHA>

<ORIGINAL PLAN IDENTITY>
{original_plan_identity}
</ORIGINAL PLAN IDENTITY>

<CURRENT STEP CONTRACT>
{current_contract}
</CURRENT STEP CONTRACT>

<WORKER MISMATCH>
{mismatch_explanation}
</WORKER MISMATCH>

<DETERMINISTIC GATE FAILURE EVIDENCE>
{failure_evidence}
</DETERMINISTIC GATE FAILURE EVIDENCE>

<CURRENT READ_SET>
{read_set}
</CURRENT READ_SET>
<CURRENT WRITE_SET>
{write_set}
</CURRENT WRITE_SET>
<CURRENT CREATE_SET>
{create_set}
</CURRENT CREATE_SET>
<CURRENT DELETE_SET>
{delete_set}
</CURRENT DELETE_SET>
<FUTURE STEP OWNERSHIP>
{future_ownership}
</FUTURE STEP OWNERSHIP>
<BOUNDED REPOSITORY EVIDENCE>
{repository_evidence}
</BOUNDED REPOSITORY EVIDENCE>
<REPOSITORY PATH CANDIDATES>
{repository_path_candidates}
</REPOSITORY PATH CANDIDATES>

REPOSITORY PATH CANDIDATES lists, for each path or file name the mismatch
names that is not a tracked path of CURRENT TREE SHA, the tracked paths with
that exact basename or suffix. MetaHarness never picks one for you: a
contract path must be copied exactly from a tracked path you choose.

Return exactly a complete repaired current-step contract. READ_SET may be
clarified and instructions, objective, VERIFY, FORBIDDEN and anchors may be
repaired. Do not change mutation sets unless the requested work truly requires
it; any such change is subject to MetaHarness scope policy. An existing file
that must change belongs to READ_SET and WRITE_SET, never CREATE_SET.

DETERMINISTIC GATE FAILURE EVIDENCE is the bounded, unedited output of the
failing check that ran after this step: it is authoritative about what failed
and must never be paraphrased away. Repair the contract so the step's own work
can satisfy it.

{_step_repair_identity_rules(identity)}

{_step_repair_template(identity)}
"""


def build_step_contract_repair_correction_prompt(
    *, identity: StepRepairIdentity, current_tree_sha: str, current_contract: str,
    mismatch_explanation: str, read_set: str, write_set: str, create_set: str,
    delete_set: str, previous_raw: str, parse_error: str,
    output_attempt: int, max_output_corrections: int,
    repository_path_candidates: str = "NONE", invalid_path_candidates: str = "",
    repository_path_facts: str = "",
) -> str:
    """Ask for a protocol-valid answer of the same semantic repair.

    The request is standalone: it carries every input the planner needs and
    the rejected answer, bounded, so no conversation state is assumed.
    """

    facts_block = repository_path_facts + "\n\n" if repository_path_facts else ""
    invalid_block = invalid_path_candidates + "\n\n" if invalid_path_candidates else ""
    previous = previous_raw
    if len(previous) > _REPAIR_PREVIOUS_RAW_CHARS:
        previous = previous[:_REPAIR_PREVIOUS_RAW_CHARS] + "\n[TRUNCATED]"
    return f"""You are the MetaHarness StepContractRepairPlanner.

Your previous StepContractRepair response was rejected deterministically.
This is output correction {output_attempt - 1} of at most {max_output_corrections}
for the same contract repair; it is not a new repair.

The deterministic repository path facts below are authoritative. They take
priority over every path proposal in the rejected response. Copy an exact
tracked path for READ_SET or WRITE_SET, or remove the invalid path.

{facts_block}{invalid_block}An INVALID PATH is not tracked in CURRENT TREE SHA. Never guess another
directory: use one exact tracked candidate below, or remove the path.

<PARSE ERROR>
{parse_error[:_REPAIR_ERROR_CHARS]}
</PARSE ERROR>

<REPOSITORY PATH CANDIDATES>
{repository_path_candidates}
</REPOSITORY PATH CANDIDATES>

IMMUTABLE STEP ID:
{identity.step_id}

Do not explain the error.
Do not emit Markdown fences.
Return exactly one complete {STEP_REPAIR_HEADER} ... {STEP_REPAIR_END}
envelope and nothing outside it.

Preserve every current mutable path. Add mutable paths only when justified by
the worker mismatch; an existing file that must change belongs to READ_SET and
WRITE_SET, never CREATE_SET. Any mutable-scope change remains subject to
MetaHarness scope policy.

<CURRENT TREE SHA>
{current_tree_sha}
</CURRENT TREE SHA>

<CURRENT STEP CONTRACT>
{current_contract}
</CURRENT STEP CONTRACT>

<WORKER MISMATCH>
{mismatch_explanation}
</WORKER MISMATCH>

<CURRENT READ_SET>
{read_set}
</CURRENT READ_SET>
<CURRENT WRITE_SET>
{write_set}
</CURRENT WRITE_SET>
<CURRENT CREATE_SET>
{create_set}
</CURRENT CREATE_SET>
<CURRENT DELETE_SET>
{delete_set}
</CURRENT DELETE_SET>

<REJECTED RESPONSE>
{previous}
</REJECTED RESPONSE>

{_step_repair_identity_rules(identity)}

{_step_repair_template(identity)}
"""


@dataclass(frozen=True)
class _RepairInputs:
    identity: StepRepairIdentity
    original_plan_identity: str
    current_contract: str
    mismatch_explanation: str
    current_tree_sha: str
    read_set: str
    write_set: str
    create_set: str
    delete_set: str
    # Tracked paths of ``current_tree_sha``; evidence only, never a choice.
    topology: Any = None

    def topology_entries(self, *texts: str, references: Sequence[str] = ()) -> list[dict[str, Any]]:
        if self.topology is None:
            return []
        return self.topology.evidence(self.mismatch_explanation, *texts, references=references)


class StepContractRepairPlanner:
    """One bounded, durable planner transaction for a contract mismatch.

    A deterministically invalid answer is corrected inside the same semantic
    repair slot, within ``max_output_corrections``; the identity fields are
    validated, never rewritten.  Callbacks receive the output attempt number:
    ``on_request`` before any provider call, ``on_response_durable`` once an
    answer is durable, ``on_output_invalid`` once its rejection is durable.
    """

    def __init__(
        self, client: TextCompletionClient, *, max_read_paths_per_step: int,
        max_output_corrections: int = 0,
    ):
        self.client = client
        self.max_read_paths_per_step = max_read_paths_per_step
        self.max_output_corrections = max_output_corrections
        self.last_usage: dict[str, Any] | None = None

    def repair(
        self, *, original_spec: str, current_tree_sha: str,
        original_plan_identity: str, current_contract: str,
        mismatch_explanation: str, read_set: str, write_set: str,
        create_set: str, delete_set: str, future_ownership: str,
        repository_evidence: str, artifacts_dir: str | Path,
        identity: StepRepairIdentity,
        validate: Callable[[ImplementationStep], None] | None = None,
        on_request: Callable[[int], None] | None = None,
        on_response_durable: Callable[[int], None] | None = None,
        on_output_invalid: Callable[[int, str], None] | None = None,
        topology: Any = None,
        failure_evidence: str = "NONE",
    ) -> ImplementationStep:
        inputs = _RepairInputs(
            identity, original_plan_identity, current_contract,
            mismatch_explanation, current_tree_sha,
            read_set, write_set, create_set, delete_set, topology,
        )
        request = build_step_contract_repair_prompt(
            original_spec=original_spec, current_tree_sha=current_tree_sha,
            original_plan_identity=original_plan_identity,
            current_contract=current_contract,
            mismatch_explanation=mismatch_explanation,
            read_set=read_set, write_set=write_set, create_set=create_set,
            delete_set=delete_set, identity=identity,
            future_ownership=future_ownership,
            repository_evidence=repository_evidence,
            repository_path_candidates=render_path_candidates(inputs.topology_entries()),
            failure_evidence=failure_evidence,
        )
        return self._complete(
            Path(artifacts_dir), request, inputs,
            validate=validate, on_request=on_request,
            on_response_durable=on_response_durable, on_output_invalid=on_output_invalid,
        )

    def resume(
        self, *, artifacts_dir: str | Path, original_plan_identity: str,
        current_contract: str, mismatch_explanation: str, current_tree_sha: str,
        identity: StepRepairIdentity, read_set: str, write_set: str,
        create_set: str, delete_set: str,
        validate: Callable[[ImplementationStep], None] | None = None,
        on_request: Callable[[int], None] | None = None,
        on_response_durable: Callable[[int], None] | None = None,
        on_output_invalid: Callable[[int, str], None] | None = None,
        topology: Any = None,
    ) -> ImplementationStep:
        """Complete the exact durable request of an interrupted repair.

        The request is never rebuilt: its bytes are the transaction identity,
        so a resume can only re-send, or re-parse the answer to, that request.
        """

        target = Path(artifacts_dir)
        meta = read_bounded_json(target / "request.meta.json", 64 * 1024)
        try:
            request = (target / "planner.request.txt").read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise StepContractRepairArtifactError("durable contract repair request is unavailable") from exc
        request_sha = hashlib.sha256(request.encode("utf-8")).hexdigest()
        if (
            not isinstance(meta, dict)
            or meta.get("request_sha256") != request_sha
            or meta.get("current_tree_sha") != current_tree_sha
        ):
            raise StepContractRepairArtifactError("durable contract repair request identity changed")
        return self._complete(
            target, request,
            _RepairInputs(
                identity, original_plan_identity, current_contract,
                mismatch_explanation, current_tree_sha,
                read_set, write_set, create_set, delete_set, topology,
            ),
            validate=validate, on_request=on_request,
            on_response_durable=on_response_durable, on_output_invalid=on_output_invalid,
        )

    def _checked(
        self, raw: str, inputs: _RepairInputs,
        validate: Callable[[ImplementationStep], None] | None,
        normalizations: list[dict[str, str]] | None = None,
    ) -> ImplementationStep:
        identity = inputs.identity
        step = parse_step_contract_repair(
            raw,
            max_read_paths_per_step=self.max_read_paths_per_step,
            expected_step_id=identity.step_id,
            expected_title=identity.title,
            expected_execution_class=identity.execution_class,
            expected_depends_on=identity.depends_on,
            expected_plan_step_count=identity.expected_plan_step_count,
            _normalizations=normalizations,
        )
        violation = inputs.identity.violation(step)
        if violation is not None:
            raise V2PlanParseError(violation)
        if validate is not None:
            validate(step)
        return step

    def _validated(
        self, target: Path, request_sha: str, inputs: _RepairInputs,
        validate: Callable[[ImplementationStep], None] | None,
    ) -> ImplementationStep | None:
        contract_path = target / "contract.md"
        validation = read_bounded_json(target / "validation.json", 64 * 1024)
        if (
            not isinstance(validation, dict)
            or validation.get("status") not in {"planner_validated", "validated"}
            or validation.get("request_sha256") != request_sha
            or not contract_path.is_file()
        ):
            return None
        contract = contract_path.read_bytes()
        if sha256_bytes(contract) != validation.get("repaired_contract_sha256"):
            raise StepContractRepairArtifactError("validated repaired contract hash changed")
        try:
            return self._checked(contract.decode("utf-8"), inputs, validate)
        except (V2PlanParseError, UnicodeError) as exc:
            raise StepContractRepairArtifactError(
                f"validated repaired contract no longer validates: {exc}"
            ) from exc

    def _correction_request(self, target: Path, number: int, inputs: _RepairInputs) -> str:
        previous = step_repair_attempt_files(target, number - 1)
        state, raw = step_repair_attempt_state(previous, current_tree_sha=inputs.current_tree_sha)
        error = read_bounded_json(previous.parse_error, 64 * 1024)
        if state != "invalid" or raw is None or not isinstance(error, dict):
            raise StepContractRepairArtifactError(
                f"contract repair output attempt {number - 1:03d} is not a durable rejection"
            )
        detail = str(error.get("detail") or "")
        return build_step_contract_repair_correction_prompt(
            identity=inputs.identity, current_tree_sha=inputs.current_tree_sha,
            current_contract=inputs.current_contract,
            mismatch_explanation=inputs.mismatch_explanation,
            read_set=inputs.read_set, write_set=inputs.write_set,
            create_set=inputs.create_set, delete_set=inputs.delete_set,
            previous_raw=raw, parse_error=detail,
            output_attempt=number, max_output_corrections=self.max_output_corrections,
            repository_path_candidates=render_path_candidates(
                inputs.topology_entries(detail, references=invalid_paths(detail))
            ),
            invalid_path_candidates=render_invalid_path_candidates(detail, inputs.topology),
            repository_path_facts=render_repository_path_facts(detail, inputs.topology),
        )

    @staticmethod
    def _persist_topology(
        files: StepRepairAttemptFiles, target: Path, inputs: _RepairInputs,
    ) -> None:
        """Bounded ``topology_evidence.json`` next to one durable request."""

        if inputs.topology is None:
            return
        detail = ""
        if files.number > 1:
            error = read_bounded_json(
                step_repair_attempt_files(target, files.number - 1).parse_error, 64 * 1024,
            )
            detail = str(error.get("detail") or "") if isinstance(error, dict) else ""
        entries = inputs.topology_entries(detail, references=invalid_paths(detail))
        files.request.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            files.request.parent / "topology_evidence.json",
            render_json(topology_payload(inputs.topology, entries)),
        )

    def _call(
        self, files: StepRepairAttemptFiles, request: str, current_tree_sha: str,
        *, pending: bool,
    ) -> str:
        files.directory.mkdir(parents=True, exist_ok=True)
        request_sha = sha256_bytes(request.encode("utf-8"))
        if not pending:
            atomic_write_text(files.request, request)
            write_prompt_diagnostics(
                files.request.parent,
                payload_for_rendered_request("step-contract-repair", request),
            )
            atomic_write_text(files.meta, render_json({
                "status": "pending", "request_sha256": request_sha,
                "current_tree_sha": current_tree_sha,
            }))
        elif files.raw.exists():
            # An answer written before its metadata is never silently lost.
            orphan = files.raw.read_bytes()
            atomic_write_text(
                files.directory / f"planner.raw.orphan-{sha256_bytes(orphan)[:12]}.md",
                orphan.decode("utf-8", errors="replace"),
            )
        result = self.client.complete(request)
        self.last_usage = completion_usage(result)
        raw = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(raw, str):
            raise V2PlanParseError("step contract repair planner did not return text")
        raw_sha = sha256_bytes(raw.encode("utf-8"))
        atomic_write_text(files.raw, raw)
        write_usage_artifact(files.usage, self.last_usage)
        atomic_write_text(files.meta, render_json({
            "status": "raw", "request_sha256": request_sha, "raw_sha256": raw_sha,
            "current_tree_sha": current_tree_sha,
        }))
        return raw

    def _response_meta(
        self, target: Path, files: StepRepairAttemptFiles, raw: str, status: str,
    ) -> None:
        files.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(files.response_meta, render_json({
            "schema_version": 1, "output_attempt": files.number, "status": status,
            "request_path": files.request.relative_to(target).as_posix(),
            "request_sha256": sha256_bytes(files.request.read_bytes()),
            "raw_path": files.raw.relative_to(target).as_posix(),
            "raw_sha256": sha256_bytes(raw.encode("utf-8")),
        }))

    def _complete(
        self, target: Path, initial_request: str, inputs: _RepairInputs, *,
        validate: Callable[[ImplementationStep], None] | None,
        on_request: Callable[[int], None] | None,
        on_response_durable: Callable[[int], None] | None,
        on_output_invalid: Callable[[int, str], None] | None,
    ) -> ImplementationStep:
        target.mkdir(parents=True, exist_ok=True)
        initial_sha = sha256_bytes(initial_request.encode("utf-8"))
        validated = self._validated(target, initial_sha, inputs, validate)
        if validated is not None:
            return validated
        number = 1
        while True:
            files = step_repair_attempt_files(target, number)
            state, raw = step_repair_attempt_state(files, current_tree_sha=inputs.current_tree_sha)
            if number == 1 and state != "none":
                meta = read_bounded_json(files.meta, 64 * 1024)
                if not isinstance(meta, dict) or meta.get("request_sha256") != initial_sha:
                    raise StepContractRepairArtifactError(
                        "durable contract repair request identity changed"
                    )
            if state == "invalid":
                # A rejected answer is never re-parsed as if it were new: it
                # is the input of the next admitted output correction.
                if number - 1 >= self.max_output_corrections:
                    error = read_bounded_json(files.parse_error, 64 * 1024) or {}
                    raise StepContractRepairOutputInvalid(
                        str(error.get("detail") or "step contract repair output is invalid"),
                        output_attempt=number, corrections=number - 1,
                        limit=self.max_output_corrections,
                    )
                number += 1
                continue
            if raw is None:
                request = (
                    initial_request if number == 1
                    else files.request.read_text(encoding="utf-8") if state == "pending"
                    else self._correction_request(target, number, inputs)
                )
                if on_request is not None:
                    on_request(number)
                if state != "pending":
                    self._persist_topology(files, target, inputs)
                raw = self._call(files, request, inputs.current_tree_sha, pending=state == "pending")
            else:
                # A paid answer is durable: a resume re-parses it, never re-buys it.
                usage = read_bounded_json(files.usage, 64 * 1024)
                self.last_usage = usage if isinstance(usage, dict) else None
            self._response_meta(target, files, raw, "raw")
            if on_response_durable is not None:
                on_response_durable(number)
            normalizations: list[dict[str, str]] = []
            try:
                step = self._checked(raw, inputs, validate, normalizations)
            except V2PlanParseError as exc:
                detail = str(exc)[:_REPAIR_ERROR_CHARS]
                atomic_write_text(files.parse_error, render_json({
                    "schema_version": 1,
                    "code": STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
                    "detail": detail,
                    "output_attempt": number,
                    "request_sha256": sha256_bytes(files.request.read_bytes()),
                    "raw_sha256": sha256_bytes(raw.encode("utf-8")),
                }))
                self._response_meta(target, files, raw, "invalid")
                if on_output_invalid is not None:
                    on_output_invalid(number, detail)
                continue
            return self._record_validated(
                target, files, raw, step, inputs, initial_sha, normalizations,
            )

    def _record_validated(
        self, target: Path, files: StepRepairAttemptFiles, raw: str,
        step: ImplementationStep, inputs: _RepairInputs, initial_sha: str,
        normalizations: list[dict[str, str]],
    ) -> ImplementationStep:
        canonical = render_repaired_step_contract(step)
        raw_sha = sha256_bytes(raw.encode("utf-8"))
        atomic_write_text(target / "contract.md", canonical)
        validation = {
            "status": "planner_validated",
            "request_sha256": initial_sha,
            "output_attempt": files.number,
            "output_corrections": files.number - 1,
            "output_request_sha256": sha256_bytes(files.request.read_bytes()),
            "raw_sha256": raw_sha,
            "original_plan_identity": inputs.original_plan_identity,
            "original_contract_sha256": sha256_bytes(inputs.current_contract.encode("utf-8")),
            "current_tree_sha": inputs.current_tree_sha,
            "mismatch_sha256": sha256_bytes(inputs.mismatch_explanation.encode("utf-8")),
            "repaired_contract_sha256": sha256_bytes(canonical.encode("utf-8")),
            "step_id": step.id,
            "write_set": list(step.write_set),
            "create_set": list(step.create_set),
            "delete_set": list(step.delete_set),
        }
        if normalizations:
            # Diagnostic only; validation authority remains the canonical
            # contract and its hash, never these normalization notes.
            validation["parser_normalization"] = {
                "schema_version": 1, "normalizations": normalizations[:4],
            }
        atomic_write_text(target / "validation.json", render_json(validation))
        atomic_write_text(files.meta, render_json({
            "status": "validated", "request_sha256": sha256_bytes(files.request.read_bytes()),
            "raw_sha256": raw_sha, "current_tree_sha": inputs.current_tree_sha,
        }))
        self._response_meta(target, files, raw, "validated")
        return step


__all__ = [
    "StepContractRepairOutputInvalid",
    "StepContractRepairPlanner",
    "StepRepairIdentity",
    "build_step_contract_repair_correction_prompt",
    "build_step_contract_repair_prompt",
]
