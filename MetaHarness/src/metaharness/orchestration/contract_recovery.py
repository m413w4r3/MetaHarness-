"""The semantic contract repair of one step, and the authority it produces.

A clean ``AGENT_CONTRACT_MISMATCH`` opens exactly one numbered repair slot
(``contract_repairs/NN``); :meth:`ContractRecoveryService.repair_transaction`
drives that durable slot to ``completed`` through the
:class:`~metaharness.planning.contract_repair.StepContractRepairPlanner`, and
``resolve_step_authority`` re-proves from the durable artifacts which repair is
in force.  The repaired contract is validated here; what a step may then do is
still decided by :mod:`~metaharness.orchestration.step_authority` alone.

The slot is transaction state: a transport failure parks it, a durable but
invalid planner answer is corrected inside it, and none of it ever consumes a
worker attempt.
"""

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    TYPE_CHECKING,
)
from ..agent.base import AgentError
from ..approval import (
    ApprovalDecision,
    read_scope_approval,
)
from ..gitops import (
    GitError,
    candidate_tree_sha,
    status_porcelain,
)
from ..llm.chat import LLMError
from ..models import (
    ExecutionRole,
    ImplementationStep,
    RunStatus,
)
from ..planning.artifacts import (
    STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
    StepContractRepairArtifactError,
)
from ..planning.contract_repair import (
    StepContractRepairOutputInvalid,
    StepContractRepairPlanner,
    StepRepairIdentity,
)
from ..planning.protocol import (
    V2PlanParseError,
    read_set_paths,
)
from ..profiles import (
    build_llm_endpoint,
    profile_for_role,
)
from ..redaction import redact
from ..recovery_policy import classify_failure
from ..repository_topology import RepositoryTopology
from ..result import atomic_write_text
from ..state import RunStateStore
from . import contract_repair
from .check_repair import replan_slot_origin
from .contract_repair import ContractRepairIntegrityError
from .pipeline_v2 import PipelineFailure
from .recovery import (
    RecoveryAttempt,
    RecoveryCoordinator,
)
from .shared import (
    ScopeApprovalRequired,
    StepExecutionFailure,
    bounded_v2_report,
    chat_client,
    json_text,
    read_json_artifact,
)
from .step_authority import (
    EffectiveStepAuthority,
    StepAuthorityError,
    resolve_effective_step_authority,
    step_contract_drift,
)


if TYPE_CHECKING:  # pragma: no cover - the composition root is the runtime
    from .runtime import RunRuntime

# The bounded red-gate evidence a contract replan planner was given, kept with
# its slot so an interrupted replan resumes with the same evidence instead of
# a second, weaker request.
GATE_REPLAN_EVIDENCE = "gate_evidence.txt"
_MAX_GATE_REPLAN_EVIDENCE_BYTES = 64 * 1024


class ContractRecoveryService:
    """One owner of the semantic contract repair transactions described in this module."""

    def __init__(self, runtime: "RunRuntime") -> None:
        self.runtime = runtime

    def repair_transaction(
        self, *, store: RunStateStore, recovery: RecoveryCoordinator,
        repo: Path, worktree: Path, run_dir: Path, artifact_dir: Path,
        directory: Path, cycle: int, number: int, step: ImplementationStep,
        current_contract: str, mismatch: str, tree_before: str,
        original_spec: str, original_plan_identity: str,
        future_ownership: Mapping[str, tuple[str, ...]] | None,
        max_repairs: int, profile_id: str, resumed: bool,
        expected_plan_step_count: int | None,
        usage: dict[str, int] | None = None,
        failure_evidence: str = "NONE",
        require_new_contract: bool = False,
    ) -> tuple[ImplementationStep, str]:
        """Drive one durable semantic repair slot to ``completed``.

        A planner transport failure parks the slot in ``waiting_external``
        and propagates; the resume re-enters this same slot.  A durable but
        invalid planner answer is corrected inside the same slot within
        ``recovery.max_contract_repair_output_corrections``; it is never a
        worker contract mismatch.  Only a validated repair consumes the
        semantic ``contract_repairs`` budget.
        """

        max_corrections = self.runtime.run_options.recovery.max_contract_repair_output_corrections
        try:
            semantic_attempt = contract_repair.semantic_repair_count(artifact_dir)
            transaction = contract_repair.read_transaction(directory) or {}
            attempt = contract_repair.output_attempt(transaction)
            if contract_repair.planner_response_durable(directory, attempt):
                transaction = contract_repair.ensure(directory, contract_repair.PLANNER_RESPONSE_DURABLE)
            if "output_correction_limit" not in transaction:
                # A slot opened before output corrections existed.
                transaction = contract_repair.advance(
                    directory, transaction["status"], output_correction_limit=max_corrections,
                )
            if resumed and transaction.get("status") == contract_repair.OUTPUT_CORRECTION_EXHAUSTED:
                # The operator retried the planner: one more bounded round of
                # output corrections, still inside this semantic slot.
                transaction = contract_repair.advance(
                    directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                    output_attempt=attempt + 1, output_correction_attempt=attempt,
                    output_correction_limit=int(transaction["output_correction_limit"]) + max(1, max_corrections),
                    operator_output_retries=int(transaction.get("operator_output_retries") or 0) + 1,
                    planner_transport_attempt=0,
                )
            if contract_repair.is_awaiting_planner(transaction):
                transaction = contract_repair.advance(
                    directory, contract_repair.awaiting_status(transaction),
                    planner_transport_attempt=int(transaction.get("planner_transport_attempt") or 0) + 1,
                )
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc

        def progress_of(current: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "step_id": step.id, "attempt": semantic_attempt,
                "pending_operation": "contract_repair",
                "contract_repair_number": number,
                "repair_id": current.get("repair_id"),
                "planner_transport_attempt": current.get("planner_transport_attempt"),
                "output_attempt": contract_repair.output_attempt(dict(current)),
                "output_correction_attempt": current.get("output_correction_attempt", 0),
                "output_correction_limit": current.get("output_correction_limit"),
            }

        def publish(status: str, current: Mapping[str, Any]) -> dict[str, Any]:
            progress = progress_of(current)
            store.update(
                status=RunStatus.CONTRACT_REPAIRING, current_step=step.id,
                contract_repair={"status": status, **progress},
            )
            return progress

        progress = publish(
            "correcting_output" if contract_repair.output_attempt(transaction) > 1 else "running",
            transaction,
        )
        if resumed:
            self.runtime.observability.trace_emit(
                "recovery.resumed", phase="implementation", cycle=cycle, step_id=step.id,
                data={"operation": "contract_repair", "transaction_status": transaction.get("status"), **progress},
            )

        def on_request(output_attempt: int) -> None:
            current = contract_repair.read_transaction(directory) or {}
            if contract_repair.output_attempt(current) >= output_attempt:
                return
            current = contract_repair.advance(
                directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                output_attempt=output_attempt,
                output_correction_attempt=output_attempt - 1,
                planner_transport_attempt=1,
            )
            data = publish("correcting_output", current)
            self.runtime.observability.trace_emit(
                "contract_repair.output_correction.started", phase="implementation",
                cycle=cycle, step_id=step.id, data=data,
            )

        def on_response_durable(_output_attempt: int) -> None:
            contract_repair.ensure(directory, contract_repair.PLANNER_RESPONSE_DURABLE)

        def on_output_invalid(output_attempt: int, detail: str) -> None:
            error = {
                "code": STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
                "detail": detail[:500], "output_attempt": output_attempt,
            }
            current = contract_repair.ensure(
                directory, contract_repair.PLANNER_OUTPUT_INVALID, last_output_error=error,
            )
            data = publish("output_invalid", current)
            self.runtime.observability.trace_emit(
                "contract_repair.output_invalid", phase="implementation",
                cycle=cycle, step_id=step.id, data={**data, "error": error},
            )

        while True:
            try:
                repaired = self._repair_step_contract(
                    repo=repo, worktree=worktree, run_dir=run_dir,
                    artifact_dir=directory, original_spec=original_spec,
                    original_plan_identity=original_plan_identity,
                    original_step=step, current_contract=current_contract,
                    expected_plan_step_count=expected_plan_step_count,
                    mismatch=mismatch, tree_before=tree_before,
                    future_ownership=future_ownership,
                    resume_request=contract_repair.durable_request_matches(directory, tree_before) is True,
                    max_output_corrections=int(
                        (contract_repair.read_transaction(directory) or {}).get(
                            "output_correction_limit", max_corrections,
                        )
                    ),
                    on_request=on_request, on_response_durable=on_response_durable,
                    on_output_invalid=on_output_invalid,
                    failure_evidence=failure_evidence,
                    require_new_contract=require_new_contract,
                )
                effective_contract = (directory / "contract.md").read_text(encoding="utf-8")
                break
            except LLMError as exc:
                detail = redact(str(exc), self.runtime.secrets)[:500]
                try:
                    current = contract_repair.read_transaction(directory) or {}
                    if contract_repair.is_awaiting_planner(current):
                        current = contract_repair.advance(
                            directory, contract_repair.WAITING_EXTERNAL,
                            last_transport_failure=detail,
                        )
                except ContractRepairIntegrityError as marker:
                    raise PipelineFailure(marker.code, str(marker), step_id=step.id) from exc
                data = publish("waiting_external", current)
                self.runtime.observability.trace_emit(
                    "recovery.waiting_external", phase="implementation", cycle=cycle,
                    step_id=step.id,
                    data={"operation": "contract_repair", "reason": "LLM_FAILURE", **data},
                )
                raise
            except ScopeApprovalRequired:
                delta = read_json_artifact(directory / "scope_delta.json", 64 * 1024)
                store.update(
                    status=RunStatus.WAITING_SCOPE_APPROVAL,
                    current_step=step.id,
                    scope_delta=delta if isinstance(delta, dict) else {},
                )
                raise
            except (ContractRepairIntegrityError, StepContractRepairArtifactError) as exc:
                raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
            except StepContractRepairOutputInvalid as exc:
                # Never a new AGENT_CONTRACT_MISMATCH.  One bounded,
                # self-contained planner restart first, in this same semantic
                # slot; then the slot waits for an operator planner retry.
                if self._restart_contract_repair_planner(directory, cycle, step.id, publish):
                    continue
                try:
                    current = contract_repair.ensure(
                        directory, contract_repair.OUTPUT_CORRECTION_EXHAUSTED,
                        last_output_error={
                            "code": exc.code, "detail": exc.detail[:500],
                            "output_attempt": exc.output_attempt,
                        },
                    )
                except ContractRepairIntegrityError as marker:
                    raise PipelineFailure(marker.code, str(marker), step_id=step.id) from exc
                data = publish("output_correction_exhausted", current)
                self.runtime.observability.trace_emit(
                    "contract_repair.output_correction.exhausted", phase="implementation",
                    cycle=cycle, step_id=step.id, data=data,
                )
                raise PipelineFailure(
                    exc.code,
                    bounded_v2_report(
                        f"contract repair {progress['repair_id']} planner output is invalid after "
                        f"{exc.corrections} of {exc.limit} output corrections: {exc.detail}"
                    ),
                    step_id=step.id,
                ) from exc
            except V2PlanParseError as exc:
                # A planner answer that could not even be made durable.
                raise PipelineFailure(
                    STEP_CONTRACT_REPAIR_OUTPUT_INVALID,
                    bounded_v2_report(f"contract repair planner output is unusable: {exc}"),
                    step_id=step.id,
                ) from exc
            except (AgentError, GitError, OSError) as exc:
                raise StepExecutionFailure(
                    "AGENT_CONTRACT_MISMATCH", step.id,
                    bounded_v2_report(f"contract repair failed: {exc}"),
                    profile_id=profile_id, tree_before=tree_before,
                    tree_after=tree_before, usage=usage, mismatch=mismatch,
                    mismatch_retry_count=semantic_attempt, step_dir=artifact_dir,
                ) from exc
        progress = progress_of(contract_repair.read_transaction(directory) or {})
        decision = classify_failure(
            "AGENT_CONTRACT_MISMATCH", clean_contract_mismatch=True, rollback_succeeded=True,
        )
        # One semantic record per repair, whatever the number of resumes.
        recovery.record(RecoveryAttempt(
            phase="implementation", reason="AGENT_CONTRACT_MISMATCH",
            attempt=semantic_attempt, budget_key="contract_repairs",
            budget=max_repairs, budget_consumed=semantic_attempt,
            strategy=decision.strategy.value,
            cycle=cycle, step_id=step.id, profile_id=profile_id,
            tree_before=tree_before, tree_after=tree_before,
            operation_id=progress["repair_id"],
        ))
        contract_repair.ensure(directory, contract_repair.COMPLETED)
        store.update(
            status=RunStatus.CONTRACT_REPAIRING, current_step=step.id,
            contract_repair={"status": "completed", **progress},
        )
        self.runtime.observability.trace_emit(
            "step.contract_repair.completed", phase="implementation",
            cycle=cycle, step_id=step.id,
            data={
                "repair": number, "repair_id": progress["repair_id"], "tree_sha": tree_before,
                "output_corrections": progress["output_correction_attempt"],
            },
        )
        recovery.trace(
            "recovery.completed", reason="AGENT_CONTRACT_MISMATCH",
            decision=decision, attempt=semantic_attempt,
            tree_before=tree_before, tree_after=candidate_tree_sha(worktree),
            budget_remaining=max(0, max_repairs - semantic_attempt),
            phase="implementation", cycle=cycle,
            step_id=step.id, recovered=True,
        )
        return repaired, effective_contract
    def _restart_contract_repair_planner(
        self, directory: Path, cycle: int, step_id: str,
        publish: Callable[[str, Mapping[str, Any]], dict[str, Any]],
    ) -> bool:
        """Admit one bounded planner restart of an exhausted output budget.

        The restart stays inside the same semantic repair slot: it consumes
        no ``contract_repairs`` budget, replays no worker, keeps every raw
        answer, and re-sends a standalone request built from the durable
        mismatch, tree, identity and repository topology evidence.
        """

        limit = self.runtime.run_options.recovery.max_contract_repair_planner_restarts
        max_corrections = self.runtime.run_options.recovery.max_contract_repair_output_corrections
        try:
            current = contract_repair.read_transaction(directory) or {}
            used = int(current.get("planner_restarts") or 0)
            if used >= limit:
                return False
            attempt = contract_repair.output_attempt(current)
            current = contract_repair.advance(
                directory, contract_repair.AWAITING_OUTPUT_CORRECTION,
                output_attempt=attempt + 1, output_correction_attempt=attempt,
                output_correction_limit=int(current.get("output_correction_limit") or 0)
                + max(1, max_corrections),
                planner_restarts=used + 1, planner_transport_attempt=0,
            )
        except ContractRepairIntegrityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step_id) from exc
        data = publish("correcting_output", current)
        self.runtime.observability.trace_emit(
            "contract_repair.planner_restart", phase="implementation",
            cycle=cycle, step_id=step_id, data={**data, "planner_restarts": used + 1},
        )
        return True
    def _repair_step_contract(
        self, *, repo: Path, worktree: Path, run_dir: Path,
        artifact_dir: Path, original_spec: str, original_plan_identity: str,
        original_step: ImplementationStep, current_contract: str,
        expected_plan_step_count: int | None,
        mismatch: str, tree_before: str,
        future_ownership: Mapping[str, tuple[str, ...]] | None,
        resume_request: bool = False,
        max_output_corrections: int = 0,
        on_request: Callable[[int], None] | None = None,
        on_response_durable: Callable[[int], None] | None = None,
        on_output_invalid: Callable[[int, str], None] | None = None,
        failure_evidence: str = "NONE",
        require_new_contract: bool = False,
    ) -> ImplementationStep:
        """Run and validate one durable StepContractRepairPlanner transaction.

        ``resume_request`` re-sends (or re-parses the answer to) the exact
        durable request of an interrupted transaction instead of rebuilding it.
        Deterministic answer defects (protocol, identity, removed approved
        paths, anchors absent from the tree) are planner output corrections;
        a mutable-scope expansion is then decided by the scope policy only.
        ``require_new_contract`` marks a repair a red deterministic gate
        opened: an answer identical to the contract that gate failed under is
        one more answer defect, never a silent replay of the approved step.
        """

        planner_profile = profile_for_role(
            self.runtime.config, self.runtime.run_options.planner_profile, ExecutionRole.PLANNER
        )
        original_mutable = set((*original_step.write_set, *original_step.create_set, *original_step.delete_set))

        def validate(repaired: ImplementationStep) -> None:
            removed = original_mutable - set((*repaired.write_set, *repaired.create_set, *repaired.delete_set))
            if removed:
                raise V2PlanParseError(
                    "contract repair removed approved mutable paths: " + ", ".join(sorted(removed))[:300]
                )
            if require_new_contract and repaired == original_step:
                raise V2PlanParseError(
                    "contract repair is identical to the contract the deterministic gate "
                    "failed under: a replan repairs the step contract, it never replays it"
                )
            drift = step_contract_drift(repo, tree_before, tree_before, repaired)
            if drift:
                raise V2PlanParseError("repaired contract is not executable on current tree: " + drift)

        planner = StepContractRepairPlanner(
            self.runtime.planner_client or chat_client(
                build_llm_endpoint(planner_profile), self.runtime.environment, self.runtime.observability.trace_transport
            ),
            max_read_paths_per_step=self.runtime.config.planning.max_read_paths_per_step,
            max_output_corrections=max_output_corrections,
        )
        sets = {
            "read_set": "\n".join(f"- {item}" for item in original_step.read_set),
            "write_set": "\n".join(f"- {item}" for item in original_step.write_set) or "NONE",
            "create_set": "\n".join(f"- {item}" for item in original_step.create_set) or "NONE",
            "delete_set": "\n".join(f"- {item}" for item in original_step.delete_set) or "NONE",
        }
        try:
            # Deterministic evidence for the planner, never a path decision.
            topology: RepositoryTopology | None = RepositoryTopology.from_tree(repo, tree_before)
        except GitError:
            topology = None
        hooks = {
            "identity": StepRepairIdentity.of(original_step, expected_plan_step_count),
            "validate": validate,
            "on_request": on_request, "on_response_durable": on_response_durable,
            "on_output_invalid": on_output_invalid,
            "topology": topology,
        }
        if resume_request:
            repaired = planner.resume(
                artifacts_dir=artifact_dir,
                original_plan_identity=original_plan_identity,
                current_contract=current_contract,
                mismatch_explanation=bounded_v2_report(mismatch),
                current_tree_sha=tree_before,
                **sets, **hooks,
            )
        else:
            evidence_parts = [
                "TREE SHA: " + tree_before,
                "STATUS: " + "; ".join(status_porcelain(worktree)[:20]),
            ]
            for path in read_set_paths(original_step.read_set):
                target = worktree / path
                try:
                    data = target.read_bytes()[:8192]
                    evidence_parts.append(
                        f"PATH {path}\n" + data.decode("utf-8", errors="replace")
                    )
                except (OSError, UnicodeError):
                    evidence_parts.append(f"PATH {path}\n<unavailable>")
            repaired = planner.repair(
                original_spec=original_spec, current_tree_sha=tree_before,
                original_plan_identity=original_plan_identity,
                current_contract=current_contract,
                mismatch_explanation=bounded_v2_report(mismatch),
                future_ownership=json_text(future_ownership or {}),
                repository_evidence="\n\n".join(evidence_parts),
                artifacts_dir=artifact_dir,
                failure_evidence=failure_evidence,
                **sets, **hooks,
            )
        contract_repair.ensure(artifact_dir, contract_repair.PLANNER_VALIDATED)
        repaired_mutable = set((*repaired.write_set, *repaired.create_set, *repaired.delete_set))
        added = repaired_mutable - original_mutable
        if added:
            policy = self.runtime.repair_scope
            if len(added) > policy.max_added_paths or policy.policy == "deny-expansion":
                # A valid answer the scope policy does not authorize: an
                # operator decision, never a planner output correction.
                raise PipelineFailure(
                    "CONTRACT_REPAIR_SCOPE_DENIED",
                    f"contract repair requested {len(added)} additional mutable path(s) "
                    f"beyond the {policy.policy} bound of {policy.max_added_paths}: "
                    + ", ".join(sorted(added))[:500],
                    step_id=original_step.id,
                )
            if policy.policy == "require-approval":
                delta = {
                    "schema_version": 1, "step_id": original_step.id,
                    "added_paths": sorted(added), "tree_sha": tree_before,
                    "repair_number": int(artifact_dir.name),
                }
                delta_path = artifact_dir / "scope_delta.json"
                atomic_write_text(delta_path, json_text(delta))
                approval = read_scope_approval(
                    artifact_dir,
                    expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
                )
                if approval is None:
                    contract_repair.ensure(artifact_dir, contract_repair.SCOPE_WAITING)
                    raise ScopeApprovalRequired()
                if approval.decision is not ApprovalDecision.APPROVE:
                    raise PipelineFailure("HUMAN_REQUIRED", "contract repair scope rejected")
        validation_path = artifact_dir / "validation.json"
        validation = read_json_artifact(validation_path, 64 * 1024)
        if isinstance(validation, dict):
            validation.update({
                "status": "validated", "added_mutable_paths": sorted(added),
                "removed_mutable_paths": [],
                "original_step_contract_sha256": hashlib.sha256(current_contract.encode("utf-8")).hexdigest(),
                "repaired_contract_sha256": hashlib.sha256(
                    (artifact_dir / "contract.md").read_bytes()
                ).hexdigest(),
            })
            atomic_write_text(validation_path, json_text(validation))
            contract_repair.ensure(artifact_dir, contract_repair.VALIDATED)
        return repaired
    def resolve_step_authority(
        self, artifact_dir: Path, step: ImplementationStep, contract: str, *,
        expected_tree: str | None, expected_plan_step_count: int | None,
    ) -> EffectiveStepAuthority:
        """The single effective authority of *step*; corruption fails closed."""

        try:
            return resolve_effective_step_authority(
                artifact_dir, step, contract,
                max_read_paths_per_step=self.runtime.config.planning.max_read_paths_per_step,
                expected_plan_step_count=expected_plan_step_count,
                expected_tree_sha=expected_tree,
                authorize_added=self._authorize_repair_additions,
            )
        except StepAuthorityError as exc:
            raise PipelineFailure(exc.code, str(exc), step_id=step.id) from exc
    @staticmethod
    def repaired_authority(authority: EffectiveStepAuthority, number: int) -> EffectiveStepAuthority:
        if authority.repair_slot != number:
            raise PipelineFailure(
                "RESUME_INTEGRITY_FAILURE",
                f"validated contract repair {number:02d} is not the effective step authority",
                step_id=authority.step_id,
            )
        return authority
    def _authorize_repair_additions(self, repair_dir: Path, added: list[str]) -> None:
        """The frozen scope policy of one validated repair's added paths."""

        policy = self.runtime.repair_scope
        if policy.policy == "deny-expansion":
            raise PipelineFailure("REPAIR_SCOPE_EXPANSION")
        if policy.policy == "require-approval" or len(added) > policy.max_added_paths:
            delta_path = repair_dir / "scope_delta.json"
            delta = read_json_artifact(delta_path, 64 * 1024)
            if not isinstance(delta, dict) or delta.get("added_paths") != sorted(added):
                raise PipelineFailure("RESUME_INTEGRITY_FAILURE", "step contract repair scope delta is malformed")
            approval = read_scope_approval(
                repair_dir,
                expected_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
            )
            if approval is None or approval.decision is not ApprovalDecision.APPROVE:
                raise ScopeApprovalRequired()
    @staticmethod
    def repair_failure_evidence(directory: Path) -> str:
        """The durable failure evidence of a slot a red gate opened, if any."""

        try:
            data = (directory / GATE_REPLAN_EVIDENCE).read_bytes()[:_MAX_GATE_REPLAN_EVIDENCE_BYTES]
        except OSError:
            return "NONE"
        return data.decode("utf-8", errors="replace") or "NONE"
    @staticmethod
    def repair_slot_origin(directory: Path) -> Mapping[str, Any] | None:
        """The red-gate replan identity of one contract repair slot, if it is one."""

        archived = read_json_artifact(directory / contract_repair.MISMATCH_NAME, 64 * 1024)
        if not isinstance(archived, dict) or not isinstance(archived.get("mismatch"), str):
            return None
        return replan_slot_origin(archived["mismatch"])
