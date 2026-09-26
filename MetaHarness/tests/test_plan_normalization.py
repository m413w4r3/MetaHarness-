"""Pure, deterministic normalization of a step contract against a tree table.

No Git, no model, no run: a step, a ``path -> exists`` table, and the exact
rules the harness applies at planning and at execution.
"""

from __future__ import annotations

import unittest

from metaharness.models import (
    ADD_MUTATION_TO_READ,
    CREATE_EXISTING_TO_WRITE,
    DROP_MISSING_DELETE,
    DROP_MISSING_READ,
    DROP_READ_OF_CREATE,
    ExecutionClass,
    ExecutionMode,
    ImplementationStep,
    PlanDecision,
    RESOLVE_MUTATION_CONFLICT,
    TaskPlanV2,
    WRITE_MISSING_TO_CREATE,
    NO_MUTATION_REMAINS,
)
from metaharness.planning.normalization import (
    NORMALIZED_READ_ANCHOR,
    TreeFacts,
    normalizations_payload,
    normalize_plan_contracts,
    normalize_step_contract,
    plan_contradictions,
)


def step(
    step_id: str = "S01",
    *,
    read: tuple[str, ...] = ("kept.py",),
    write: tuple[str, ...] = (),
    create: tuple[str, ...] = (),
    delete: tuple[str, ...] = (),
) -> ImplementationStep:
    return ImplementationStep(
        step_id, "title", ExecutionClass.MECHANICAL, None, "objective",
        tuple(f"{path} :: anchor" for path in read), write,
        "1. do", "- check", "- none", create_set=create, delete_set=delete,
    )


def tree(*paths: str, absent: tuple[str, ...] = ()) -> TreeFacts:
    return TreeFacts.from_mapping(
        {**{path: True for path in paths}, **{path: False for path in absent}}
    )


def records(value) -> list[tuple[str, str | None]]:
    return [(item.code, item.path) for item in value.normalizations]


def reads(value: ImplementationStep) -> list[str]:
    return [item.split(" :: ", 1)[0] for item in value.read_set]


class StepContractNormalizationTests(unittest.TestCase):
    def test_create_on_an_existing_path_becomes_a_write(self) -> None:
        contract = normalize_step_contract(step(create=("src/a.py",)), tree("kept.py", "src/a.py"))
        self.assertEqual(contract.step.write_set, ("src/a.py",))
        self.assertEqual(contract.step.create_set, ())
        self.assertEqual(reads(contract.step), ["kept.py", "src/a.py"])
        self.assertEqual(
            records(contract),
            [(CREATE_EXISTING_TO_WRITE, "src/a.py"), (ADD_MUTATION_TO_READ, "src/a.py")],
        )
        self.assertEqual(contract.contradictions, ())

    def test_write_on_a_missing_path_becomes_a_create(self) -> None:
        contract = normalize_step_contract(step(write=("src/new.py",)), tree("kept.py"))
        self.assertEqual(contract.step.write_set, ())
        self.assertEqual(contract.step.create_set, ("src/new.py",))
        self.assertEqual(reads(contract.step), ["kept.py"])
        self.assertEqual(records(contract), [(WRITE_MISSING_TO_CREATE, "src/new.py")])

    def test_delete_of_a_missing_path_disappears(self) -> None:
        contract = normalize_step_contract(
            step(write=("kept.py",), delete=("old.py",)), tree("kept.py"),
        )
        self.assertEqual(contract.step.delete_set, ())
        self.assertEqual(reads(contract.step), ["kept.py"])
        self.assertEqual(records(contract), [(DROP_MISSING_DELETE, "old.py")])
        self.assertEqual(contract.contradictions, ())

    def test_read_of_a_missing_path_disappears(self) -> None:
        contract = normalize_step_contract(
            step(read=("kept.py", "gone.py"), write=("kept.py",)), tree("kept.py"),
        )
        self.assertEqual(reads(contract.step), ["kept.py"])
        self.assertEqual(records(contract), [(DROP_MISSING_READ, "gone.py")])

    def test_overlapping_sets_leave_one_canonical_mutation(self) -> None:
        existing = normalize_step_contract(
            step(read=(), write=("src/a.py",), create=("src/a.py",)), tree("src/a.py"),
        )
        self.assertEqual(existing.step.write_set, ("src/a.py",))
        self.assertEqual(existing.step.create_set, ())
        self.assertEqual(records(existing)[0], (RESOLVE_MUTATION_CONFLICT, "src/a.py"))
        missing = normalize_step_contract(
            step(read=(), write=("src/a.py",), create=("src/a.py",)), tree(),
        )
        self.assertEqual(missing.step.write_set, ())
        self.assertEqual(missing.step.create_set, ("src/a.py",))
        self.assertEqual(
            [code for code, _ in records(missing)],
            [RESOLVE_MUTATION_CONFLICT, WRITE_MISSING_TO_CREATE],
        )
        # A DELETE declared on the same path never wins over the mutation.
        both = normalize_step_contract(
            step(read=(), create=("src/a.py",), delete=("src/a.py",)), tree("src/a.py"),
        )
        self.assertEqual(both.step.write_set, ("src/a.py",))
        self.assertEqual(both.step.delete_set, ())

    def test_read_set_gains_mutations_and_drops_created_paths(self) -> None:
        contract = normalize_step_contract(
            step(
                read=("kept.py", "src/new.py"),
                write=("kept.py",), create=("src/new.py",), delete=("removed.py",),
            ),
            tree("kept.py", "removed.py"),
        )
        self.assertEqual(reads(contract.step), ["kept.py", "removed.py"])
        self.assertEqual(
            [code for code, _ in records(contract)],
            [ADD_MUTATION_TO_READ, DROP_READ_OF_CREATE],
        )
        self.assertIn(f"removed.py :: {NORMALIZED_READ_ANCHOR}", contract.step.read_set)

    def test_normalization_is_idempotent(self) -> None:
        facts = tree("kept.py", "src/a.py", "removed.py")
        first = normalize_step_contract(
            step(
                read=("kept.py", "gone.py", "src/new.py"),
                write=("src/a.py", "src/new.py"), create=("kept.py",),
                delete=("removed.py", "old.py"),
            ),
            facts,
        )
        second = normalize_step_contract(first.step, facts)
        self.assertEqual(second.step, first.step)
        self.assertEqual(second.normalizations, ())
        self.assertEqual(second.contradictions, first.contradictions)

    def test_a_step_with_no_possible_mutation_is_a_contradiction(self) -> None:
        contract = normalize_step_contract(step(read=("kept.py",), delete=("old.py",)), tree("kept.py"))
        self.assertEqual(contract.contradictions, (NO_MUTATION_REMAINS,))
        self.assertEqual(contract.step.delete_set, ())

    def test_tree_facts_read_a_plain_path_table(self) -> None:
        facts = TreeFacts.from_mapping({"a.py": True, "b.py": False})
        self.assertTrue(facts.exists("a.py"))
        self.assertFalse(facts.exists("b.py"))
        self.assertFalse(facts.exists("never-listed.py"))
        self.assertTrue(facts.after(step(write=("c.py",))).exists("c.py"))
        self.assertFalse(facts.after(step(write=("a.py",), delete=("a.py",))).exists("a.py"))


class PlanProjectionTests(unittest.TestCase):
    @staticmethod
    def plan(*steps: ImplementationStep) -> TaskPlanV2:
        return TaskPlanV2(
            PlanDecision.READY, "title", "objective", "constraints",
            ExecutionMode.SINGLE if len(steps) == 1 else ExecutionMode.STAGED,
            steps, "acceptance", "tests", "NONE", "NONE", "raw",
        )

    def test_a_path_created_by_an_earlier_step_is_valid_later(self) -> None:
        normalized = normalize_plan_contracts(
            self.plan(
                step("S01", read=("kept.py",), create=("generated/schema.py",)),
                step("S02", read=("generated/schema.py",), write=("generated/schema.py",)),
            ),
            tree("kept.py"),
        )
        first, second = normalized.plan.steps
        self.assertEqual(first.create_set, ("generated/schema.py",))
        self.assertEqual(second.write_set, ("generated/schema.py",))
        self.assertEqual(second.create_set, ())
        self.assertEqual(normalized.contradictions, ())

    def test_a_later_create_of_a_created_path_becomes_a_write(self) -> None:
        normalized = normalize_plan_contracts(
            self.plan(
                step("S01", read=("kept.py",), create=("generated/schema.py",)),
                step("S02", read=("kept.py",), create=("generated/schema.py",)),
            ),
            tree("kept.py"),
        )
        self.assertEqual(normalized.plan.steps[1].write_set, ("generated/schema.py",))
        self.assertEqual(
            [(item.step_id, item.code) for item in normalized.plan.normalizations],
            [("S02", CREATE_EXISTING_TO_WRITE), ("S02", ADD_MUTATION_TO_READ)],
        )

    def test_a_deleted_path_is_absent_for_the_next_step(self) -> None:
        normalized = normalize_plan_contracts(
            self.plan(
                step("S01", read=("kept.py",), delete=("kept.py",)),
                step("S02", read=("kept.py",), write=("kept.py",)),
            ),
            tree("kept.py"),
        )
        second = normalized.plan.steps[1]
        self.assertEqual(second.write_set, ())
        self.assertEqual(second.create_set, ("kept.py",))
        self.assertEqual(second.read_set, ())
        self.assertEqual(normalized.contradictions, ())

    def test_contradictions_are_recorded_once_and_survive_a_second_pass(self) -> None:
        plan = self.plan(step("S01", read=("kept.py",), delete=("old.py",)))
        facts = tree("kept.py")
        first = normalize_plan_contracts(plan, facts)
        self.assertEqual(first.contradictions, (("S01", NO_MUTATION_REMAINS),))
        self.assertEqual(plan_contradictions(first.plan), (("S01", NO_MUTATION_REMAINS),))
        again = normalize_plan_contracts(first.plan, facts)
        self.assertEqual(again.plan, first.plan)

    def test_payload_is_compact_and_names_every_rule(self) -> None:
        normalized = normalize_plan_contracts(
            self.plan(
                step("S01", read=("kept.py", "gone.py"), create=("kept.py",)),
                step("S02", read=("kept.py",), create=("generated/schema.py",)),
            ),
            tree("kept.py"),
        )
        payload = normalizations_payload(normalized.plan)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(
            payload["steps"]["S01"],
            [
                {"code": CREATE_EXISTING_TO_WRITE, "path": "kept.py"},
                {"code": DROP_MISSING_READ, "path": "gone.py"},
            ],
        )
        # S02 creates a path no earlier step produces: nothing to normalize.
        self.assertNotIn("S02", payload["steps"])
        self.assertNotIn("raw", payload)


if __name__ == "__main__":
    unittest.main()
