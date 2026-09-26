"""Deterministic normalization of a step contract against a tree.

A planner can understand a task and still misclassify one path: declare a
``CREATE_SET`` entry for a file the tree already holds, or a ``DELETE_SET``
entry for a path that is not there.  None of those facts needs a semantic
decision -- Git already answers them -- so the harness normalizes the contract
and lets the run continue instead of stopping it.

:func:`normalize_step_contract` is the single authority of that normalization:
pure, deterministic, no model call, no Git call and no global state.  Both
application points share it:

* after parsing, before approval, projecting the logical tree step by step
  (:func:`normalize_plan_contracts`);
* just before one step executes, against the real current tree.

Nothing here invents architecture.  What stays contradictory once every
deterministic rule has been applied -- a step left with no mutation at all --
is reported for the planner to correct.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping

from ..models import (
    ADD_MUTATION_TO_READ,
    CONTRADICTION_CODES,
    CREATE_EXISTING_TO_WRITE,
    DROP_MISSING_DELETE,
    DROP_MISSING_READ,
    DROP_READ_OF_CREATE,
    ContractNormalization,
    ImplementationStep,
    NO_MUTATION_REMAINS,
    PlanDecision,
    RESOLVE_MUTATION_CONFLICT,
    TaskPlanV2,
    WRITE_MISSING_TO_CREATE,
)

# The anchor a READ_SET entry gains when normalization adds the path itself:
# honest, deterministic, and never a claim about the file's content.
NORMALIZED_READ_ANCHOR = "current content"
# One path, one canonical mutation, in decreasing precedence.
_MUTATION_ORDER = ("write", "create", "delete")


class TreeFacts:
    """The ``path -> exists`` facts of the tree one step starts from."""

    __slots__ = ("_lookup", "_overlay")

    def __init__(self, lookup: Callable[[str], bool], overlay: Mapping[str, bool] | None = None) -> None:
        if not callable(lookup):
            raise TypeError("tree facts require a lookup callable")
        self._lookup = lookup
        self._overlay = dict(overlay or {})

    @classmethod
    def from_mapping(cls, table: Mapping[str, bool]) -> "TreeFacts":
        """Facts from a plain ``path -> exists`` table (the unit-test form)."""

        frozen = {str(path): bool(exists) for path, exists in table.items()}
        return cls(lambda path: frozen.get(path, False))

    @classmethod
    def from_paths(cls, paths: Iterable[str]) -> "TreeFacts":
        """Facts where exactly *paths* exist."""

        frozen = {str(path) for path in paths}
        return cls(lambda path: path in frozen)

    def exists(self, path: str) -> bool:
        if path in self._overlay:
            return self._overlay[path]
        return bool(self._lookup(path))

    def after(self, step: ImplementationStep) -> "TreeFacts":
        """The logical tree *step* is authorized to produce."""

        overlay = dict(self._overlay)
        for path in step.write_set:
            overlay[path] = True
        for path in step.create_set:
            overlay[path] = True
        for path in step.delete_set:
            overlay[path] = False
        return TreeFacts(self._lookup, overlay)


@dataclass(frozen=True)
class NormalizedStepContract:
    """The effective contract of one step, and every rule that produced it."""

    step: ImplementationStep
    normalizations: tuple[ContractNormalization, ...] = ()
    # Irreducible contradictions: codes that stay impossible after every
    # deterministic rule was applied, and therefore belong to the planner.
    contradictions: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.normalizations)


@dataclass(frozen=True)
class NormalizedPlanContracts:
    """The plan whose steps are all effective, plus its unresolvable facts."""

    plan: TaskPlanV2
    # ``(step_id, code)`` pairs of the contradictions above.
    contradictions: tuple[tuple[str, str], ...] = ()


def _read_entries(read_set: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """The ordered READ_SET paths and the anchor declared for each of them."""

    paths: list[str] = []
    anchors: dict[str, str] = {}
    for entry in read_set:
        path, _, anchor = entry.partition(" :: ")
        path = path.strip()
        if not path:
            continue
        if path not in anchors:
            paths.append(path)
            anchors[path] = anchor.strip() or NORMALIZED_READ_ANCHOR
    return paths, anchors


def normalize_step_contract(
    step: ImplementationStep, repository_tree_facts: TreeFacts,
) -> NormalizedStepContract:
    """Return the effective contract of *step* against the tree it starts from.

    Duplicates are resolved first (``WRITE > CREATE > DELETE``), then existence
    decides what each remaining mutation really is, and finally READ_SET is
    made coherent with the result.  Idempotent: normalizing an effective
    contract changes nothing and records nothing.
    """

    if not isinstance(step, ImplementationStep):
        raise TypeError("step must be an ImplementationStep")
    if not isinstance(repository_tree_facts, TreeFacts):
        raise TypeError("repository_tree_facts must be TreeFacts")

    records: list[ContractNormalization] = []
    read, anchors = _read_entries(step.read_set)
    mutations = {
        "write": list(step.write_set),
        "create": list(step.create_set),
        "delete": list(step.delete_set),
    }

    def record(code: str, path: str, detail: str | None = None) -> None:
        records.append(ContractNormalization(code=code, step_id=step.id, path=path, detail=detail))

    # 1. One path, one canonical mutation.
    for path in [item for name in _MUTATION_ORDER for item in mutations[name]]:
        owners = [name for name in _MUTATION_ORDER if path in mutations[name]]
        if len(owners) <= 1:
            continue
        kept = owners[0]
        for name in owners[1:]:
            mutations[name].remove(path)
        record(RESOLVE_MUTATION_CONFLICT, path, detail=f"kept={kept}")

    # 2. Existence decides between WRITE and CREATE; a missing DELETE goes away.
    for path in list(mutations["write"]):
        if not repository_tree_facts.exists(path):
            mutations["write"].remove(path)
            mutations["create"].append(path)
            record(WRITE_MISSING_TO_CREATE, path)
    for path in list(mutations["create"]):
        if repository_tree_facts.exists(path):
            mutations["create"].remove(path)
            mutations["write"].append(path)
            record(CREATE_EXISTING_TO_WRITE, path)
    for path in list(mutations["delete"]):
        if not repository_tree_facts.exists(path):
            mutations["delete"].remove(path)
            record(DROP_MISSING_DELETE, path)

    # 3. READ_SET: every surviving mutation of an existing path is readable, a
    #    CREATE path is not, and no read survives a path absent at step start.
    created = set(mutations["create"])
    for path in (*mutations["write"], *mutations["delete"]):
        if path not in read:
            read.append(path)
            anchors[path] = NORMALIZED_READ_ANCHOR
            record(ADD_MUTATION_TO_READ, path)
    for path in list(read):
        if path in created:
            read.remove(path)
            record(DROP_READ_OF_CREATE, path)
        elif not repository_tree_facts.exists(path):
            read.remove(path)
            record(DROP_MISSING_READ, path)

    effective = replace(
        step,
        read_set=tuple(f"{path} :: {anchors[path]}" for path in read),
        write_set=tuple(mutations["write"]),
        create_set=tuple(mutations["create"]),
        delete_set=tuple(mutations["delete"]),
    )
    contradictions = (
        () if any(mutations.values()) else (NO_MUTATION_REMAINS,)
    )
    return NormalizedStepContract(effective, tuple(records), contradictions)


def normalize_plan_contracts(
    plan: TaskPlanV2, repository_tree_facts: TreeFacts,
) -> NormalizedPlanContracts:
    """Normalize every step of *plan*, projecting the logical tree in order.

    Each step is normalized against the tree it really starts from: the start
    tree for the first one, then the tree the earlier CREATE/WRITE/DELETE
    sections authorize.
    """

    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    if plan.decision is not PlanDecision.READY or not plan.steps:
        return NormalizedPlanContracts(plan)
    facts = repository_tree_facts
    records = list(plan.normalizations)
    contradictions: list[tuple[str, str]] = []
    steps: list[ImplementationStep] = []
    for step in plan.steps:
        contract = normalize_step_contract(step, facts)
        records.extend(contract.normalizations)
        for code in contract.contradictions:
            # Recorded once: a second normalization of the same plan adds no
            # duplicate entry.
            entry = ContractNormalization(code=code, step_id=step.id)
            if entry not in records:
                records.append(entry)
            contradictions.append((step.id, code))
        steps.append(contract.step)
        facts = facts.after(contract.step)
    effective = replace(plan, steps=tuple(steps), normalizations=tuple(records))
    return NormalizedPlanContracts(effective, tuple(contradictions))


def plan_contradictions(plan: TaskPlanV2) -> tuple[tuple[str, str], ...]:
    """The ``(step_id, code)`` contradictions a normalized plan still carries."""

    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    return tuple(
        (item.step_id, item.code) for item in plan.normalizations
        if item.step_id and item.code in CONTRADICTION_CODES
    )


def normalization_entries(
    normalizations: Iterable[ContractNormalization],
) -> list[dict[str, str]]:
    """The compact JSON entries of normalization records; no prose, no plan."""

    entries: list[dict[str, str]] = []
    for item in normalizations:
        entry: dict[str, str] = {"code": item.code}
        if item.path:
            entry["path"] = item.path
        if item.detail:
            entry["detail"] = item.detail
        entries.append(entry)
    return entries


def normalizations_payload(plan: TaskPlanV2) -> dict[str, object]:
    """The compact ``plan.normalizations.json`` body; no plan copy, no prose."""

    if not isinstance(plan, TaskPlanV2):
        raise TypeError("plan must be a TaskPlanV2")
    steps: dict[str, list[dict[str, str]]] = {}
    plan_level: list[dict[str, str]] = []
    for item in plan.normalizations:
        entry = normalization_entries((item,))[0]
        if item.step_id:
            steps.setdefault(item.step_id, []).append(entry)
        else:
            plan_level.append(entry)
    payload: dict[str, object] = {"schema": 1, "steps": steps}
    if plan_level:
        payload["plan"] = plan_level
    return payload


__all__ = [
    "NORMALIZED_READ_ANCHOR",
    "normalization_entries",
    "NormalizedPlanContracts",
    "NormalizedStepContract",
    "TreeFacts",
    "normalizations_payload",
    "normalize_plan_contracts",
    "normalize_step_contract",
    "plan_contradictions",
]
