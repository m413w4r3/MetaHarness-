"""The deterministic-gate recovery ladder of one run.

A red deterministic gate walks one ordered ladder of *distinct* strategies
before any operator wait: the bounded targeted repair pass, one bounded
evidence-proven scope expansion, a contract replan of the evidence-proven
responsible step, a replan of the whole cycle, one final pass under the
configured executor-fallback authority and only then the operator.  Every rung
is identified by the exact candidate tree, the failed check set, the stage and
the strategy, and the durable ledger never proposes the same strategy twice for
those facts: a new tree or a new failed-check set opens a new progression.

The ladder owns no authority of its own, builds no prompt and constructs no
scope: a repair pass is bounded by the frozen ``max_check_repair_attempts``
budget, an expansion by the operator-approved plan scope and the run's
repair-scope policy, and every replan rewrites one approved step contract
through the durable transaction the run injected.  Nothing here can grant a
model a scope the operator did not approve.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
)
from ..evidence import EvidenceBundle
from ..models import (
    GateStage,
    correction_cycles_used,
)
from ..planning.check_replan import (
    PLAN_ARTIFACT as CHECK_REPLAN_PLAN_ARTIFACT,
    check_replan_dir,
    plan_identity,
)
from ..recovery_policy import (
    FailureClass,
    RecoveryFacts,
    RecoveryFingerprint,
    RecoveryProgression,
    RecoveryStrategy,
    admitted_strategies,
    recovery_ladder,
    terminal_strategy,
)
from ..result import atomic_write_text
from ..resume import ResumeIntegrityError
from ..run_options import effective_repair_scope_policy
from .check_failure import red_gate_identity
from .check_scope import (
    authorize_scope_expansion,
    canonical_scope_paths,
    evidence_proven_expansion,
    responsible_step_index,
)
from .pipeline_v2 import (
    CyclePlan,
    check_repair_dir,
    gate_dir,
)
from .recovery import (
    GateRecoveryStep,
    RecoveryStepUnavailable,
)
from .shared import (
    json_text,
    read_json_artifact,
)


_LADDER_ARTIFACT = "ladder.json"
_LADDER_SCHEMA_VERSION = 1
_LADDER_CODE = "CHECK_FAILED"
_MAX_LADDER_ENTRIES = 64
_LADDER_STATES = frozenset({"running", "done"})
# Strategies the ladder consumes at most once per gate episode.
_EPISODE_STRATEGIES = frozenset({
    RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE,
})


@dataclasses.dataclass(frozen=True)
class _LadderEntry:
    """One durably consumed ladder step of one red gate episode."""

    strategy: RecoveryStrategy
    state: str
    tree: str
    failed_check_ids: tuple[str, ...]
    repair_attempt: int | None = None
    step_indices: tuple[int, ...] = ()
    added_paths: tuple[str, ...] = ()
    tree_after: str = ""


@dataclasses.dataclass(frozen=True)
class _LadderLedger:
    """The frozen facts and the consumed steps of one gate episode."""

    proof_required: bool
    fallback_executor_available: bool
    entries: tuple[_LadderEntry, ...] = ()


def _ladder_path(run_dir: Path, cycle: int, stage: GateStage) -> Path:
    return check_repair_dir(run_dir, cycle, stage) / _LADDER_ARTIFACT


def _read_ladder(path: Path) -> _LadderLedger | None:
    """Read and validate the ladder ledger; a malformed artifact fails closed."""

    payload = read_json_artifact(path, 256 * 1024)
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _LADDER_SCHEMA_VERSION:
        raise ResumeIntegrityError("gate recovery ladder artifact is malformed")
    proof = payload.get("proof_required")
    fallback = payload.get("fallback_executor_available")
    if not isinstance(proof, bool) or not isinstance(fallback, bool):
        raise ResumeIntegrityError("gate recovery ladder facts are malformed")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_LADDER_ENTRIES:
        raise ResumeIntegrityError("gate recovery ladder entries are malformed")
    entries: list[_LadderEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ResumeIntegrityError("gate recovery ladder entry is malformed")
        try:
            strategy = RecoveryStrategy(item.get("strategy"))
        except ValueError as exc:
            raise ResumeIntegrityError("gate recovery ladder strategy is unknown") from exc
        state = item.get("state")
        tree = item.get("tree")
        failed = item.get("failed_check_ids")
        attempt = item.get("repair_attempt")
        indices = item.get("step_indices")
        if state not in _LADDER_STATES or not isinstance(tree, str) or len(tree) > 128:
            raise ResumeIntegrityError("gate recovery ladder entry is malformed")
        if not isinstance(failed, list) or any(not isinstance(name, str) or not name for name in failed):
            raise ResumeIntegrityError("gate recovery ladder entry has invalid failed checks")
        if attempt is not None and (
            isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
        ):
            raise ResumeIntegrityError("gate recovery ladder entry has an invalid repair attempt")
        if not isinstance(indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ):
            raise ResumeIntegrityError("gate recovery ladder entry has invalid step indices")
        added = canonical_scope_paths(item.get("added_paths"), what="gate recovery ladder expansion")
        tree_after = item.get("tree_after")
        if not isinstance(tree_after, str) or len(tree_after) > 128:
            raise ResumeIntegrityError("gate recovery ladder entry has an invalid produced tree")
        entries.append(_LadderEntry(
            strategy, state, tree, tuple(failed), attempt, tuple(indices), added, tree_after,
        ))
    return _LadderLedger(proof, fallback, tuple(entries))


def _write_ladder(path: Path, ledger: _LadderLedger) -> None:
    atomic_write_text(path, json_text({
        "schema_version": _LADDER_SCHEMA_VERSION,
        "proof_required": ledger.proof_required,
        "fallback_executor_available": ledger.fallback_executor_available,
        "entries": [
            {
                "strategy": entry.strategy.value, "state": entry.state, "tree": entry.tree,
                "failed_check_ids": list(entry.failed_check_ids),
                "repair_attempt": entry.repair_attempt,
                "step_indices": list(entry.step_indices),
                "added_paths": list(entry.added_paths),
                "tree_after": entry.tree_after,
            }
            for entry in ledger.entries
        ],
    }))


@dataclasses.dataclass(frozen=True)
class CheckRepairLadder:
    """The deterministic-gate ladder of one run.

    It is the single object :class:`PipelineV2Operations` references for the
    red-gate ladder: the machine asks which distinct strategy to try next and
    reports the step as started and finished.  The durable ledger, the frozen
    episode facts and the admission of every rung live here; the bounded
    expansion a rung proposes is derived by the scope authority from the
    failure evidence, never chosen here, so no model, no planner and no worker
    can widen what the operator already approved.

    ``replan_steps`` is injected by the run: for one ``REPLAN_STEP`` rung it
    identifies the responsible step, produces a new bounded contract from the
    red-gate evidence through the existing durable contract-repair
    transaction, re-executes that step under its new validated authority with
    its necessary descendants, and returns the tree it produced.

    ``replan_cycles`` is the last rung: once a step's contract replan is spent
    too, the whole decomposition of the cycle is replanned through the durable
    check-replan planning transaction and the new cycle it opens is returned.

    Both raise :class:`RecoveryStepUnavailable` when the rung is not
    applicable for these exact facts, so the ladder advances instead of
    looping.
    """

    replan_steps: Callable[..., str] | None = None
    replan_cycles: Callable[..., CyclePlan] | None = None

    def gate_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        evidence: EvidenceBundle, repair_attempt: int, repair_budget: int,
        correction_budget: int,
    ) -> GateRecoveryStep:
        """The next distinct ladder step of this red gate, or its terminal."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, failed = red_gate_identity(evidence)
        fallback_available = bool(ctx.selection.check_repair_fallbacks)
        proof = self._expansion_paths(ctx, cycle_plan, stage_value, evidence)
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            ledger = _LadderLedger(
                proof_required=bool(proof), fallback_executor_available=fallback_available,
            )
        elif ledger.fallback_executor_available != fallback_available:
            raise ResumeIntegrityError(
                "the gate recovery ladder does not match the frozen executor authority"
            )
        elif proof and not ledger.proof_required:
            # The failure evidence has proven that an approved path outside the
            # frozen repair scope is implicated: the proof is a durable fact of
            # this gate episode, never a model decision.
            ledger = dataclasses.replace(ledger, proof_required=True)
            _write_ladder(path, ledger)
        pending = next(
            (entry for entry in reversed(ledger.entries) if entry.state == "running"), None,
        )
        if pending is not None:
            # The strategy of this round is already durable: resume it as such.
            return GateRecoveryStep(
                pending.strategy, tree=tree, failed_check_ids=failed,
                repair_attempt=pending.repair_attempt, step_indices=pending.step_indices,
                added_paths=pending.added_paths, consumed=self._trail(ledger),
            )
        facts = self._facts(ledger, tree=tree, failed=failed, number=number, stage=stage_value)
        progression = RecoveryProgression(self._consumed(ledger, number=number, stage=stage_value))
        for strategy in recovery_ladder(FailureClass.CORRECTNESS):
            if strategy.terminal or not self._policy_admits(strategy, facts):
                continue
            if strategy is RecoveryStrategy.REPAIR_TARGETED and proof:
                # The failure evidence proves an approved path the frozen
                # repair scope cannot reach: the bounded expansion is the next
                # distinct strategy, never another pass on the narrow scope.
                continue
            recorded = strategy in _EPISODE_STRATEGIES and any(
                entry.strategy is strategy for entry in ledger.entries
            )
            # A cycle rung whose durable answer already exists is the *resume*
            # of that one rung, never a second answer: the plan it produced is
            # recovered and judged, so neither the episode rule nor the
            # progression refuses it -- while any other recorded rung stays
            # spent for these exact facts.
            resuming = bool(
                recorded and strategy is RecoveryStrategy.REPLAN_CYCLE
                and self._cycle_replan_open(ctx, cycle_plan, tree, failed)
            )
            if recorded and not resuming:
                continue
            step = self._admit(
                strategy, ctx=ctx, cycle_plan=cycle_plan, stage=stage_value,
                evidence=evidence, ledger=ledger, tree=tree, failed=failed, proof=proof,
                repair_attempt=repair_attempt, repair_budget=repair_budget,
                correction_budget=correction_budget,
            )
            if step is None:
                # Deterministically inapplicable for these exact facts:
                # advance the ladder without executing anything.
                continue
            if not resuming and progression.is_consumed(progression.fingerprint(
                candidate_tree=tree, failure_class=FailureClass.CORRECTNESS,
                facts=facts, strategy=strategy,
            )):
                # This exact strategy already ran for this tree and failure.
                continue
            return step
        return GateRecoveryStep(
            terminal_strategy(FailureClass.CORRECTNESS, _LADDER_CODE), tree=tree,
            failed_check_ids=failed, consumed=self._trail(ledger), exhausted=True,
        )

    @staticmethod
    def _policy_admits(strategy: RecoveryStrategy, facts: RecoveryFacts) -> bool:
        """Whether the ladder policy itself admits this rung for these facts."""

        return strategy in admitted_strategies(FailureClass.CORRECTNESS, facts)

    def begin_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> None:
        """Consume one ladder step durably before anything runs for it."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, failed = red_gate_identity(evidence)
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            ledger = _LadderLedger(
                proof_required=bool(self._expansion_paths(
                    ctx, cycle_plan, stage_value, evidence,
                )),
                fallback_executor_available=bool(ctx.selection.check_repair_fallbacks),
            )
        entry = _LadderEntry(
            step.strategy, "running", tree, failed,
            step.repair_attempt, step.step_indices, step.added_paths,
        )
        def same(item: _LadderEntry) -> bool:
            return (
                item.strategy is entry.strategy and item.tree == entry.tree
                and item.failed_check_ids == entry.failed_check_ids
                and item.repair_attempt == entry.repair_attempt
                and item.step_indices == entry.step_indices
            )

        existing = next((item for item in ledger.entries if same(item)), None)
        if existing is None:
            if len(ledger.entries) >= _MAX_LADDER_ENTRIES:
                raise ResumeIntegrityError("the gate recovery ladder is full")
            ledger = dataclasses.replace(ledger, entries=(*ledger.entries, entry))
        elif existing.state != "running":
            # A rung that opens a whole cycle is re-admitted after a crash
            # between its durable answer and that cycle: it is running again,
            # and the ledger is the one place that says so.
            ledger = dataclasses.replace(ledger, entries=tuple(
                entry if same(item) else item for item in ledger.entries
            ))
        _write_ladder(path, ledger)
        if (
            step.strategy is RecoveryStrategy.EXPAND_SCOPE
            and step.repair_attempt is not None and step.added_paths
        ):
            authorize_scope_expansion(
                run_dir=ctx.run_dir, cycle=number, stage=stage_value,
                attempt=step.repair_attempt, tree=step.tree, added_paths=step.added_paths,
                approved=tuple(cycle_plan.mutable_scope), failed=failed,
                policy=effective_repair_scope_policy(ctx.options),
            )

    def replan_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> str:
        """Rewrite and re-execute the responsible approved step of one rung.

        The ladder owns no scope of its own.  The rung must name exactly the
        evidence-proven responsible step and the descendants its rewritten
        commit invalidates, and the injected replan owns the durable contract
        repair, the new validated authority and the re-execution; the returned
        tree is what the next gate episode observes.  A rung whose facts no
        longer hold is refused without touching the repository.
        """

        if step.exhausted or step.is_repair_pass or not step.step_indices:
            raise RecoveryStepUnavailable(step.strategy, "the ladder step is not a replan rung")
        if step.strategy is not RecoveryStrategy.REPLAN_STEP:
            raise RecoveryStepUnavailable(
                step.strategy, "only a single responsible step is replanned for now",
            )
        if self.replan_steps is None:
            raise RecoveryStepUnavailable(
                step.strategy, "this run admits no contract replan of approved work",
            )
        stage_value = GateStage(stage)
        tree, _failed = red_gate_identity(evidence)
        count = len(cycle_plan.plan.steps)
        first = step.step_indices[0]
        if step.step_indices != tuple(range(first, count)):
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not replan a responsible step and its descendants",
            )
        if self._responsible_step_index(ctx, cycle_plan, stage_value, evidence) != first:
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not name the evidence-proven responsible step",
            )
        produced = self.replan_steps(ctx, cycle_plan, stage_value, step, evidence)
        if not isinstance(produced, str) or not produced:
            raise ResumeIntegrityError("the gate recovery replan produced no candidate tree")
        return produced

    def replan_cycle(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle,
    ) -> CyclePlan:
        """Re-decompose the whole cycle of one rung and open its new cycle.

        The rung is the last one of the episode: the bounded repair pass and
        the contract replan of the responsible step were both durably spent
        and the gate is still red, so the decomposition itself is what failed.
        The injected transaction produces the new plan from this gate's own
        bounded failure evidence, inside the approved envelope, and returns the
        durable plan of the cycle that executes it -- a plan whose authority is
        never the one already in force.  A rung these facts do not admit is
        refused before anything runs, so the ladder advances without looping.
        """

        if step.exhausted or step.is_repair_pass or step.step_indices:
            raise RecoveryStepUnavailable(
                step.strategy, "the ladder step is not a cycle replan",
            )
        if step.strategy is not RecoveryStrategy.REPLAN_CYCLE:
            raise RecoveryStepUnavailable(
                step.strategy, "only a whole cycle is re-decomposed by this rung",
            )
        if self.replan_cycles is None:
            raise RecoveryStepUnavailable(
                step.strategy, "this run admits no cycle replan of approved work",
            )
        tree, failed = red_gate_identity(evidence)
        if step.tree != tree or step.failed_check_ids != failed:
            raise RecoveryStepUnavailable(
                step.strategy, "the rung does not describe the red gate evidence",
            )
        planned = self.replan_cycles(ctx, cycle_plan, GateStage(stage), step, evidence)
        if planned is None:
            raise RecoveryStepUnavailable(step.strategy, "the cycle replan produced no plan")
        return planned

    def finish_step(
        self, *, ctx: Any, cycle_plan: Any, stage: GateStage | str,
        step: GateRecoveryStep, evidence: EvidenceBundle, tree_after: str,
    ) -> None:
        """Mark a consumed step as executed; the ledger itself never repeats it."""

        number = cycle_plan.cycle.number
        stage_value = GateStage(stage)
        tree, _failed = red_gate_identity(evidence)
        if not isinstance(tree_after, str) or len(tree_after) > 128:
            raise ResumeIntegrityError("the gate recovery ladder produced an invalid tree")
        path = _ladder_path(ctx.run_dir, number, stage_value)
        ledger = _read_ladder(path)
        if ledger is None:
            raise ResumeIntegrityError("the gate recovery ladder artifact is missing")
        entries: list[_LadderEntry] = []
        marked = False
        for entry in ledger.entries:
            if not marked and (
                entry.state == "running" and entry.strategy is step.strategy
                and entry.repair_attempt == step.repair_attempt and entry.tree == tree
            ):
                entries.append(dataclasses.replace(entry, state="done", tree_after=tree_after))
                marked = True
            else:
                entries.append(entry)
        if not marked:
            raise ResumeIntegrityError("the gate recovery ladder step is not pending")
        _write_ladder(path, dataclasses.replace(ledger, entries=tuple(entries)))

    # -- deterministic facts -------------------------------------------------

    @staticmethod
    def _facts(
        ledger: _LadderLedger, *, tree: str, failed: tuple[str, ...],
        number: int, stage: GateStage,
    ) -> RecoveryFacts:
        return RecoveryFacts(
            candidate_tree=tree,
            observed_facts=(
                ("cycle", f"{number:03d}"),
                ("failed_checks", ",".join(failed)),
                ("stage", stage.value),
            ),
            proof_required=ledger.proof_required,
            fallback_executor_available=ledger.fallback_executor_available,
            # The ladder's first step is a bounded repair pass; the frozen
            # attempt budget is enforced by this ladder, never by the policy.
            retry_allowed=True,
        )

    @classmethod
    def _consumed(
        cls, ledger: _LadderLedger, *, number: int, stage: GateStage,
    ) -> tuple[RecoveryFingerprint, ...]:
        return tuple(
            RecoveryFingerprint(
                entry.tree, FailureClass.CORRECTNESS,
                cls._facts(
                    ledger, tree=entry.tree, failed=entry.failed_check_ids,
                    number=number, stage=stage,
                ).stable_items(),
                entry.strategy,
            )
            for entry in ledger.entries
        )

    @staticmethod
    def _trail(ledger: _LadderLedger) -> tuple[RecoveryStrategy, ...]:
        return tuple(entry.strategy for entry in ledger.entries)

    @staticmethod
    def _cycle_replan_record(ctx: Any, cycle: int) -> dict[str, Any] | None:
        """The durable check-replan answer one cycle holds, if any."""

        record = read_json_artifact(
            check_replan_dir(ctx.run_dir, cycle) / CHECK_REPLAN_PLAN_ARTIFACT
        )
        return record if isinstance(record, dict) else None

    @staticmethod
    def _cycle_replan_answers(
        record: Mapping[str, Any], tree: str, failed: tuple[str, ...], identity: str,
    ) -> bool:
        """Whether one durable record answers exactly these red-gate facts."""

        return (
            record.get("candidate_tree_sha") == tree
            and tuple(record.get("failed_check_ids") or ()) == tuple(sorted(failed))
            and record.get("plan_identity_before") == identity
        )

    @classmethod
    def _cycle_replan_consumed(
        cls, ctx: Any, cycle_plan: Any, tree: str, failed: tuple[str, ...],
    ) -> bool:
        """Whether these exact facts already opened a cycle replan.

        The fingerprint is this gate's candidate tree, its failed checks and
        the identity of the plan already in force: the same plan produced again
        for the same facts never opens a second cycle.  Every earlier answer is
        read from its own durable record, so a resume reaches the same decision.
        """

        identity = plan_identity(cycle_plan.plan)
        for earlier in range(2, cycle_plan.cycle.number + 1):
            record = cls._cycle_replan_record(ctx, earlier)
            if record is not None and cls._cycle_replan_answers(
                record, tree, failed, identity,
            ):
                return True
        return False

    @classmethod
    def _cycle_replan_open(
        cls, ctx: Any, cycle_plan: Any, tree: str, failed: tuple[str, ...],
    ) -> bool:
        """Whether this rung already produced the plan of the cycle it opens.

        The rung ends the gate episode and the cycle only starts afterwards, so
        a crash in between leaves a durable answer without a running ladder
        entry.  That answer is replayed -- recovered, never re-planned -- while
        one that re-decomposed nothing leaves the rung spent.
        """

        record = cls._cycle_replan_record(ctx, cycle_plan.cycle.number + 1)
        return record is not None and (
            record.get("plan_identity_after") != record.get("plan_identity_before")
            and cls._cycle_replan_answers(
                record, tree, failed, plan_identity(cycle_plan.plan),
            )
        )

    def _expansion_paths(
        self, ctx: Any, cycle_plan: Any, stage: GateStage, evidence: EvidenceBundle,
    ) -> tuple[str, ...]:
        """Bounded, evidence-proven additions to the repair scope."""

        return evidence_proven_expansion(
            run_dir=ctx.run_dir, cycle=cycle_plan.cycle.number, stage=stage,
            repo=ctx.repo, worktree=ctx.info.worktree,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            evidence=evidence, approved=tuple(cycle_plan.mutable_scope),
            policy=effective_repair_scope_policy(ctx.options),
        )

    @staticmethod
    def _responsible_step_index(
        ctx: Any, cycle_plan: Any, stage: GateStage, evidence: EvidenceBundle,
    ) -> int | None:
        """The first approved step whose mutable paths the failure implicates."""

        return responsible_step_index(
            repo=ctx.repo, worktree=ctx.info.worktree,
            evidence_dir=gate_dir(ctx.run_dir, cycle_plan.cycle.number, stage),
            evidence=evidence, approved=tuple(cycle_plan.mutable_scope),
            steps=cycle_plan.plan.steps,
        )

    def _admit(
        self, strategy: RecoveryStrategy, *, ctx: Any, cycle_plan: Any, stage: GateStage,
        evidence: EvidenceBundle, ledger: _LadderLedger, tree: str,
        failed: tuple[str, ...], proof: tuple[str, ...],
        repair_attempt: int, repair_budget: int, correction_budget: int,
    ) -> GateRecoveryStep | None:
        """Materialize the step, or refuse it for these exact facts.

        A refusal is not an outcome: the rung consumes no attempt, produces no
        report and leaves the ladder free to propose the next distinct
        strategy.  The frozen ``repair_budget`` is consulted here and only for
        the rungs that execute a check-repair worker pass; the replan rungs are
        never bounded by it.  ``REPLAN_CYCLE`` is bounded by the run's single
        ``correction_budget`` instead, because it opens one more cycle.
        """

        trail = self._trail(ledger)
        common = {"tree": tree, "failed_check_ids": failed, "consumed": trail}
        if strategy in {
            RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE,
            RecoveryStrategy.FALLBACK_EXECUTOR,
        } and any(entry.repair_attempt == repair_attempt for entry in ledger.entries):
            # This bounded pass number is already durably consumed: one pass is
            # never proposed, executed or counted twice for these facts.
            return None
        if strategy is RecoveryStrategy.FALLBACK_EXECUTOR:
            if not ledger.fallback_executor_available:
                return None
            if repair_attempt > repair_budget:
                return None
            return GateRecoveryStep(strategy, repair_attempt=repair_attempt, **common)
        if strategy in {RecoveryStrategy.REPAIR_TARGETED, RecoveryStrategy.EXPAND_SCOPE}:
            if repair_attempt > repair_budget:
                return None
            if strategy is RecoveryStrategy.REPAIR_TARGETED:
                return GateRecoveryStep(strategy, repair_attempt=repair_attempt, **common)
            if not proof:
                return None
            return GateRecoveryStep(
                strategy, repair_attempt=repair_attempt, added_paths=proof, **common,
            )
        if strategy in {RecoveryStrategy.REPLAN_STEP, RecoveryStrategy.REPLAN_CYCLE}:
            if strategy is RecoveryStrategy.REPLAN_CYCLE:
                if self.replan_cycles is None:
                    return None
                if correction_cycles_used(cycle_plan.cycle.number) >= correction_budget:
                    # This rung opens one more cycle after ``INITIAL``, so it
                    # spends the run's single correction budget; a budget these
                    # facts have already used up refuses it before any planner
                    # call, artifact or cycle exists, and the ladder moves on.
                    return None
                if self._cycle_replan_consumed(ctx, cycle_plan, tree, failed):
                    # These exact facts already re-decomposed a cycle: the same
                    # failure under the same plan never opens a second one.
                    return None
                return GateRecoveryStep(strategy, **common)
            if self.replan_steps is None:
                return None
            index = self._responsible_step_index(ctx, cycle_plan, stage, evidence)
            if index is None:
                # No step is proven responsible by the failure evidence: a
                # replan would guess, so this rung is refused for these exact
                # facts and the ladder proposes its next distinct strategy.
                return None
            # A replan rewrites the responsible approved step and re-executes
            # it with the descendants its rewritten commit invalidates.
            return GateRecoveryStep(
                strategy,
                step_indices=tuple(range(index, len(cycle_plan.plan.steps))), **common,
            )
        return None


def consumed_ladder_strategies(
    run_dir: str | Path, cycle: int, stage: GateStage | str,
) -> tuple[str, ...]:
    """The distinct ladder rungs one gate episode already consumed, in order.

    Read from the episode's own durable ledger, so the same episode always
    reports the same strategies to a planner and to a resume alike.
    """

    ledger = _read_ladder(_ladder_path(Path(run_dir), cycle, GateStage(stage)))
    return tuple(entry.strategy.value for entry in (ledger.entries if ledger else ()))
