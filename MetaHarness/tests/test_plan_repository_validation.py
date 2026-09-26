"""The repository facts a META PLAN v2 is bound to, and their normalization.

A mechanical path misclassification -- a ``CREATE_SET`` entry for a path the
tree already holds, a ``WRITE_SET`` entry for a path that is not there -- is
not a planning decision: Git answers it, and
:mod:`metaharness.planning.normalization` keeps the plan.  What is tested here
is the boundary that remains: the immutable tree facts the normalizer reads,
the one contradiction no deterministic rule settles, the evidence a planner may
see, and the drift gate of the execution boundary.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metaharness.context import build_context, render_context
from metaharness.gitops import resolve_tree
from metaharness.llm.chat import TextLLMResult
from metaharness.models import (
    ADD_MUTATION_TO_READ,
    ContractNormalization,
    ContextConfig,
    CREATE_EXISTING_TO_WRITE,
    DROP_MISSING_DELETE,
    DROP_MISSING_READ,
    DROP_READ_OF_CREATE,
    ExecutionClass,
    ExecutionMode,
    ExecutionRole,
    ImplementationStep,
    NO_MUTATION_REMAINS,
    PlanDecision,
    RESOLVE_MUTATION_CONFLICT,
    RunStatus,
    TaskPlanV2,
    WRITE_MISSING_TO_CREATE,
)
from metaharness.plan_recovery import PlanRecoveryError, plan_recovery_info
from metaharness.plan_repository_validation import (
    PathPreconditionViolation,
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
    normalize_plan_contracts,
    plan_repository_violations,
    render_blocker_repository_evidence,
    render_precondition_correction,
    render_violations,
    validate_plan_repository_topology,
    violations_payload,
)
from metaharness.planning.normalization import normalizations_payload, plan_contradictions
from metaharness.planning.planner import PlannerV2
from metaharness.usage import phase_usage_summary
from tests.pipeline_support import PipelineHarness, git, initial_plan, review, write

X = "pkg/x.py"
AW010_PATH = "backend/src/cti_app/domain/reference_corpus.py"
GONE = "pkg/gone.py"


def step(
    step_id: str, *, read: tuple[str, ...] = (), write_set: tuple[str, ...] = (),
    create: tuple[str, ...] = (), delete: tuple[str, ...] = (),
) -> ImplementationStep:
    return ImplementationStep(
        step_id, "title", ExecutionClass.MECHANICAL, None, "objective",
        tuple(f"{path} :: anchor" for path in read), write_set,
        "1. do", "- check", "- none", create_set=create, delete_set=delete,
    )


def task_plan(*steps: ImplementationStep) -> TaskPlanV2:
    return TaskPlanV2(
        PlanDecision.READY, "title", "objective", "constraints",
        ExecutionMode.SINGLE if len(steps) == 1 else ExecutionMode.STAGED,
        tuple(steps), "acceptance", "tests", "NONE", "NONE", "raw",
    )


def meta_plan(*steps: dict[str, tuple[str, ...]], title: str = "Add the feature") -> str:
    """A READY META PLAN v2 text; each step maps set names to paths."""

    def lines(paths: tuple[str, ...]) -> str:
        return "\n".join(f"- {path}" for path in paths) if paths else "NONE"

    blocks = []
    for index, sets in enumerate(steps, 1):
        step_id = f"S{index:02d}"
        reads = "\n".join(f"- {path} :: current content" for path in sets.get("read", ()))
        blocks.append(f"""BEGIN STEP {step_id}
TITLE: Change the feature
EXECUTION_CLASS: MECHANICAL
DEPENDS_ON: NONE

OBJECTIVE
Change the feature.

READ_SET
{reads}

WRITE_SET
{lines(sets.get("write", ()))}

CREATE_SET
{lines(sets.get("create", ()))}

DELETE_SET
{lines(sets.get("delete", ()))}

INSTRUCTIONS
1. Change the feature.

VERIFY
- Run the configured test.

FORBIDDEN
- Do not change paths outside the declared sets.

END STEP {step_id}
""")
    mode = "SINGLE" if len(steps) == 1 else "STAGED"
    return f"""META PLAN v2

STATUS: READY
TITLE: {title}

OBJECTIVE
Implement the requested feature.

CONSTRAINTS
Keep the change local.

EXECUTION_MODE: {mode}
STEP_COUNT: {len(steps)}

{"".join(blocks)}
ACCEPTANCE
The feature file holds the requested content.

REQUIRED_CHECKS
- test

TESTS
The configured test is the final evidence.

RISKS
NONE

BLOCKERS
NONE

END META PLAN
"""


def impossible_plan_message(*, title: str = "Add the feature") -> str:
    """A plan no Git fact can rescue: its only mutation is impossible."""

    return meta_plan({"read": ("feature.txt",), "delete": ("gone.txt",)}, title=title)


class _Repo(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "tests")
        git(self.repo, "config", "user.email", "tests@example.invalid")
        (self.repo / "pkg").mkdir()
        (self.repo / "README.md").write_text("readme\n", encoding="utf-8")
        (self.repo / "pkg/other.py").write_text("other\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.without_x = resolve_tree(self.repo, "HEAD")
        (self.repo / X).write_text("VALUE = 1\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add x")
        self.with_x = resolve_tree(self.repo, "HEAD")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def effective(self, tree: str, *steps: ImplementationStep) -> TaskPlanV2:
        return normalize_plan_contracts(self.repo, tree, task_plan(*steps))

    def records(self, tree: str, *steps: ImplementationStep) -> list[tuple[str, str, str]]:
        plan = self.effective(tree, *steps)
        return [(item.code, item.step_id or "", item.path or "") for item in plan.normalizations]

    def violations(self, tree: str, *steps: ImplementationStep) -> list[tuple[str, str, str]]:
        return [
            (item.step_id, item.kind, item.path)
            for item in plan_repository_violations(self.effective(tree, *steps))
        ]


class PlanContractNormalizationTests(_Repo):
    """Every Git-decidable misclassification becomes the effective contract."""

    def test_a_create_on_an_existing_path_becomes_a_write(self) -> None:
        plan = self.effective(self.with_x, step("S01", read=(X,), create=(X,)))

        (effective,) = plan.steps
        self.assertEqual(effective.write_set, (X,))
        self.assertEqual(effective.create_set, ())
        self.assertEqual(plan.normalizations, (
            ContractNormalization(CREATE_EXISTING_TO_WRITE, "S01", X),
        ))
        # The path is already read: adding it again would be a second record.
        self.assertEqual(plan.steps[0].read_set, (f"{X} :: anchor",))
        self.assertEqual(plan_repository_violations(plan), ())

    def test_a_write_on_a_missing_path_becomes_a_create(self) -> None:
        plan = self.effective(self.without_x, step("S01", read=("README.md",), write_set=(X,)))

        (effective,) = plan.steps
        self.assertEqual(effective.write_set, ())
        self.assertEqual(effective.create_set, (X,))
        self.assertEqual(self.records(
            self.without_x, step("S01", read=("README.md",), write_set=(X,)),
        ), [(WRITE_MISSING_TO_CREATE, "S01", X)])

    def test_a_delete_of_a_missing_path_is_dropped_not_refused(self) -> None:
        plan = self.effective(
            self.without_x, step("S01", read=("README.md",), write_set=(X,), delete=(GONE,)),
        )

        (effective,) = plan.steps
        self.assertEqual(effective.delete_set, ())
        self.assertIn((DROP_MISSING_DELETE, "S01", GONE), self.records(
            self.without_x, step("S01", read=("README.md",), write_set=(X,), delete=(GONE,)),
        ))
        self.assertEqual(plan_repository_violations(plan), ())

    def test_a_read_of_a_missing_path_is_dropped(self) -> None:
        plan = self.effective(
            self.with_x, step("S01", read=(X, GONE, "pkg/other.py"), write_set=(X,)),
        )

        (effective,) = plan.steps
        self.assertEqual(
            [item.split(" :: ")[0] for item in effective.read_set], [X, "pkg/other.py"],
        )
        self.assertIn((DROP_MISSING_READ, "S01", GONE), self.records(
            self.with_x, step("S01", read=(X, GONE, "pkg/other.py"), write_set=(X,)),
        ))

    def test_a_read_of_a_created_path_is_dropped(self) -> None:
        plan = self.effective(
            self.without_x, step("S01", read=("README.md", X), create=(X,)),
        )

        (effective,) = plan.steps
        self.assertEqual([item.split(" :: ")[0] for item in effective.read_set], ["README.md"])
        self.assertIn((DROP_READ_OF_CREATE, "S01", X), self.records(
            self.without_x, step("S01", read=("README.md", X), create=(X,)),
        ))

    def test_a_path_declared_in_several_sets_keeps_one_canonical_mutation(self) -> None:
        plan = self.effective(
            self.with_x,
            step("S01", read=(X,), write_set=(X,), create=(X,), delete=(X,)),
        )

        (effective,) = plan.steps
        self.assertEqual((effective.write_set, effective.create_set, effective.delete_set), ((X,), (), ()))
        self.assertEqual(self.records(
            self.with_x, step("S01", read=(X,), write_set=(X,), create=(X,), delete=(X,)),
        )[0], (RESOLVE_MUTATION_CONFLICT, "S01", X))

    def test_a_path_created_by_an_earlier_step_is_valid_later(self) -> None:
        plan = self.effective(
            self.without_x,
            step("S01", read=("README.md",), create=(X,)),
            step("S02", read=(X,), write_set=(X,)),
        )

        first, second = plan.steps
        self.assertEqual(first.create_set, (X,))
        self.assertEqual(second.write_set, (X,))
        self.assertEqual(second.create_set, ())
        self.assertEqual(plan.normalizations, ())

    def test_a_path_deleted_by_an_earlier_step_is_absent_later(self) -> None:
        plan = self.effective(
            self.with_x,
            step("S01", read=(X,), delete=(X,)),
            step("S02", read=(X,), write_set=(X,)),
        )

        second = plan.steps[1]
        self.assertEqual(second.write_set, ())
        self.assertEqual(second.create_set, (X,))
        self.assertEqual(second.read_set, ())

    def test_the_tree_each_step_really_starts_from_decides_write_or_create(self) -> None:
        # A candidate of a replan has X although the base does not.
        normalizing_writes = self.records(self.with_x, step("S01", read=(X,), write_set=(X,)))
        self.assertEqual(normalizing_writes, [])
        self.assertEqual(
            self.records(self.with_x, step("S01", read=(X,), create=(X,))),
            [(CREATE_EXISTING_TO_WRITE, "S01", X)],
        )
        self.assertEqual(self.records(
            self.with_x, step("S01", read=("README.md",), create=(X,)),
        ), [(CREATE_EXISTING_TO_WRITE, "S01", X), (ADD_MUTATION_TO_READ, "S01", X)])
        self.assertEqual(
            self.records(self.without_x, step("S01", read=("README.md",), create=(X,))), [],
        )

    def test_the_reference_corpus_create_is_one_git_fact_from_a_write(self) -> None:
        """Run 20260923T131351Z-e9a34fd827: the file exists, the planner wrote CREATE."""

        corpus = self.repo / AW010_PATH
        corpus.parent.mkdir(parents=True)
        corpus.write_text("class ReferenceMember:\n    pass\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "reference corpus")
        tree = resolve_tree(self.repo, "HEAD")

        plan = self.effective(tree, step(
            "S01", read=("README.md",), write_set=("README.md",), create=(AW010_PATH,),
        ))

        (effective,) = plan.steps
        self.assertEqual(effective.create_set, ())
        self.assertEqual(effective.write_set, ("README.md", AW010_PATH))
        self.assertIn(
            {"code": CREATE_EXISTING_TO_WRITE, "path": AW010_PATH},
            normalizations_payload(plan)["steps"]["S01"],
        )
        self.assertEqual(plan_repository_violations(plan), ())

    def test_a_step_left_with_no_mutation_is_the_planner_contradiction(self) -> None:
        plan = self.effective(
            self.without_x, step("S01", read=("README.md",), delete=(GONE,)),
        )

        self.assertEqual(plan.steps[0].delete_set, ())
        self.assertEqual(
            plan_repository_violations(plan),
            (PathPreconditionViolation("S01", "no_mutation", ""),),
        )
        # The plan record keeps the contradiction code; the violation reports
        # the stable lowercase token.
        self.assertEqual(plan_contradictions(plan), (("S01", NO_MUTATION_REMAINS),))

    def test_blocked_plan_has_no_preconditions(self) -> None:
        plan = task_plan(step("S01", read=(X,), write_set=(X,)))
        blocked = TaskPlanV2(PlanDecision.BLOCKED, *list(plan.__dict__.values())[1:])
        self.assertEqual(normalize_plan_contracts(self.repo, self.without_x, blocked).steps, blocked.steps)
        self.assertEqual(plan_repository_violations(blocked), ())

    def test_no_mutation_violations_render_and_serialize_without_a_path(self) -> None:
        rendered = render_violations([
            PathPreconditionViolation("S02", "no_mutation", ""),
            PathPreconditionViolation("S01", "no_mutation", ""),
        ])
        self.assertEqual(rendered, "step=S01 no_mutation\nstep=S02 no_mutation")
        self.assertEqual(violations_payload("tree", [
            PathPreconditionViolation("S01", "no_mutation", ""),
        ])["violations"], [{"step_id": "S01", "kind": "no_mutation", "path": ""}])
        correction = render_precondition_correction(
            [PathPreconditionViolation("S01", "no_mutation", "")],
            previous_raw="META PLAN v2\n",
        )
        self.assertIn("PLAN REPOSITORY CONTRACT ERRORS", correction)
        self.assertIn("step=S01 no_mutation", correction)
        self.assertIn("META PLAN v2", correction)


class TopologyValidationTests(_Repo):
    def test_a_normally_classified_plan_is_returned_effective(self) -> None:
        plan = validate_plan_repository_topology(
            self.repo, self.with_x, task_plan(step("S01", read=(X,), create=(X,))),
        )
        self.assertEqual(plan.steps[0].write_set, (X,))

    def test_only_the_irreducible_contradiction_raises(self) -> None:
        with self.assertRaises(PlanRepositoryPreconditionError) as caught:
            validate_plan_repository_topology(
                self.repo, self.without_x, task_plan(
                    step("S01", read=("README.md",), delete=(GONE,)),
                ),
            )
        self.assertEqual(str(caught.exception), "step=S01 no_mutation")
        self.assertEqual(caught.exception.code, "PLAN_REPOSITORY_PRECONDITION_INVALID")

    def test_a_created_path_can_be_read_and_written_later(self) -> None:
        validate_plan_repository_topology(
            self.repo, self.without_x, task_plan(
                step("S01", read=("README.md",), create=(X,)),
                step("S02", read=(X,), write_set=(X,)),
            ),
        )


class BlockerEvidenceTests(_Repo):
    """The bounded, tree-pinned evidence a planner is allowed to see."""

    def test_sensitive_and_binary_paths_are_never_put_in_a_prompt(self) -> None:
        (self.repo / ".env").write_text("TOKEN=hidden\n", encoding="utf-8")
        (self.repo / "pkg/blob.bin").write_bytes(b"\x00\x01binary")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "sensitive")
        tree = resolve_tree(self.repo, "HEAD")

        _, evidence = render_blocker_repository_evidence(
            self.repo, tree, "- path: .env\n- pkg/blob.bin\n- pkg/big.py :: symbol\n- pkg/x.py :: VALUE",
        )

        self.assertIn(".env", evidence)
        self.assertIn("withheld (sensitive path)", evidence)
        self.assertIn("binary file", evidence)
        self.assertIn("absent from the immutable tree", evidence)
        self.assertIn("VALUE = 1", evidence)
        self.assertNotIn("hidden", evidence)

    def test_evidence_is_bounded_and_tree_pinned(self) -> None:
        (self.repo / X).write_text("WORKING TREE ONLY\n", encoding="utf-8")
        big = "pkg/big.py"
        (self.repo / big).write_text("x" * 20000, encoding="utf-8")
        git(self.repo, "add", big)
        git(self.repo, "commit", "-qm", "big")
        tree = resolve_tree(self.repo, "HEAD")

        _, evidence = render_blocker_repository_evidence(self.repo, tree, f"- {X}\n- {big}")

        self.assertIn("VALUE = 1", evidence)
        self.assertNotIn("WORKING TREE ONLY", evidence)
        self.assertIn("SIZE:", evidence.replace("CONTENT (first 8192 bytes of 20000):", "SIZE:"))
        self.assertNotIn("x" * 8193, evidence)


class SpecPathContextTests(_Repo):
    def test_explicit_spec_paths_report_base_existence_with_bounded_excerpts(self) -> None:
        (self.repo / ".env").write_text("TOKEN=hidden\n", encoding="utf-8")
        git(self.repo, "add", ".env")
        git(self.repo, "commit", "-qm", "env")
        base = git(self.repo, "rev-parse", "HEAD")
        (self.repo / X).write_text("MUTABLE WORKING TREE\n", encoding="utf-8")
        spec = f"Change `{X}` and add pkg/new_module.py, not .env; e.g. Python 3.12.\n"
        rendered = render_context(build_context(self.repo, base, spec, ContextConfig(always_files=())))
        section = rendered.split("### SPEC PATH EVIDENCE (UNTRUSTED)", 1)[1]
        self.assertIn(f"PATH: {X}\nEXISTS_AT_BASE: true\nKIND: file", section)
        self.assertIn("VALUE = 1", section)
        self.assertNotIn("MUTABLE WORKING TREE", section)
        self.assertIn("PATH: pkg/new_module.py\nEXISTS_AT_BASE: false", section)
        self.assertNotIn(".env", section)
        self.assertNotIn("hidden", rendered)
        self.assertNotIn("PATH: e.g", section)
        self.assertNotIn("3.12", section)


class _Recording:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.requests: list[str] = []

    def complete(self, request: str) -> TextLLMResult:
        self.requests.append(request)
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return TextLLMResult(self.answers.pop(0), "fake", usage, {})


class PlannerNormalizationTests(PipelineHarness):
    """The planner keeps going when Git can classify the path itself."""

    def setUp(self) -> None:
        super().setUp()
        target = self.repo / AW010_PATH
        target.parent.mkdir(parents=True)
        target.write_text(
            "class ReferenceMember:\n    \"\"\"A member of the malware reference corpus.\"\"\"\n",
            encoding="utf-8",
        )
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "reference corpus")
        git(self.repo, "push", "-q", "origin", "main")

    def reference_corpus_plan(self) -> str:
        return meta_plan({"read": ("feature.txt",), "write": ("feature.txt",), "create": (AW010_PATH,)})

    def test_aw010_create_on_an_existing_file_reaches_execution(self) -> None:
        """Run 20260923T131351Z-e9a34fd827 must not stop on STEP_CONTRACT_DRIFT."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[self.reference_corpus_plan()], reviewer=[review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        # One answer, no correction: the harness owns the deterministic fix.
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(self.trace_names().count("plan.completed"), 1)
        self.assertEqual(self.workers.roles(), ["implementer"])
        contract = (run_dir / "steps/S01/contract.md").read_text(encoding="utf-8")
        self.assertIn(f"- {AW010_PATH}", contract.split("WRITE SET", 1)[1].split("CREATE SET", 1)[0])
        created = contract.split("CREATE SET", 1)[1].split("DELETE SET", 1)[0]
        self.assertNotIn(AW010_PATH, created)
        normalized = json.loads((run_dir / "plan.normalizations.json").read_text(encoding="utf-8"))
        self.assertIn(
            {"code": CREATE_EXISTING_TO_WRITE, "path": AW010_PATH},
            normalized["steps"]["S01"],
        )
        (step_json,) = json.loads(
            (run_dir / "task_plan.json").read_text(encoding="utf-8"),
        )["steps"]
        self.assertEqual(step_json["create_set"], [])
        self.assertIn(AW010_PATH, step_json["write_set"])

    def test_two_impossible_plans_fail_planning_before_any_worker(self) -> None:
        result = self.orchestrator(
            replace(self.config(), planning=replace(self.config().planning, max_preapproval_corrections=1)),
            planner=[impossible_plan_message(), impossible_plan_message()],
            reviewer=[review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        state = self.state()
        self.assertEqual(state["failure"]["reason"], "PLAN_REPOSITORY_PRECONDITION_INVALID")
        self.assertIn("step=S01 no_mutation", state["failure"]["detail"])
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.workers.calls, [])
        run_dir = self.run_dir()
        for name in ("implementation_bundle.json", "task_plan.json", "planner.raw.md", "steps"):
            self.assertFalse((run_dir / name).exists(), name)
        record = json.loads((run_dir / "planner-attempts/02/repository_preconditions.json").read_text())
        self.assertEqual(record["violations"], [{"step_id": "S01", "kind": "no_mutation", "path": ""}])
        # A planning failure: the run is at its PLANNER checkpoint, never at a step.
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertFalse(self.worktree().exists())
        self.assertTrue(plan_recovery_info(run_dir, state).eligible)

    def test_operator_recovery_is_validated_before_approval(self) -> None:
        orchestrator = self.orchestrator(
            replace(self.config(), planning=replace(self.config().planning, max_preapproval_corrections=1)),
            planner=[impossible_plan_message(), impossible_plan_message()],
            reviewer=[review()],
        )
        orchestrator.run_text("Make feature.txt good.\n", run_id="run")
        before = self.state()
        with self.assertRaises(PlanRecoveryError) as caught:
            orchestrator.recover_plan("run", impossible_plan_message())
        self.assertIn("PLAN_REPOSITORY_PRECONDITION_INVALID", str(caught.exception))
        self.assertIn("step=S01 no_mutation", str(caught.exception))
        self.assertEqual(self.state()["updated_at"], before["updated_at"])
        self.assertFalse((self.run_dir() / "implementation_bundle.json").exists())
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertEqual(self.workers.calls, [])

    def test_operator_recovery_normalizes_a_classifiable_plan(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        orchestrator = self.orchestrator(
            self.config(), planner=[impossible_plan_message()], reviewer=[review()],
        )
        orchestrator.run_text("Make feature.txt good.\n", run_id="run")
        self.assertEqual(self.state()["failure"]["reason"], "PLAN_REPOSITORY_PRECONDITION_INVALID")

        recovered = orchestrator.recover_plan("run", self.reference_corpus_plan())

        self.assertEqual(recovered.status, RunStatus.COMMITTED, self.state().get("failure"))
        contract = (self.run_dir() / "steps/S01/contract.md").read_text(encoding="utf-8")
        self.assertIn(AW010_PATH, contract.split("WRITE SET", 1)[1].split("CREATE SET", 1)[0])

    def test_both_planner_calls_stay_in_usage_accounting(self) -> None:
        chat = _Recording([impossible_plan_message(), self.reference_corpus_plan()])
        run_dir = self.root / "planning"
        run_dir.mkdir()
        config = self.config()
        base_tree = resolve_tree(self.repo, "HEAD")
        planner = PlannerV2(
            chat, planning=config.planning, check_catalog=config.check_catalog,
            default_check_ids=config.default_check_ids,
            repository_preconditions=RepositoryPreconditions(self.repo, base_tree),
        )
        planner.plan("spec", "context", artifacts_dir=run_dir)
        self.assertEqual(phase_usage_summary(run_dir)["planner"]["total_tokens"], 30)
        self.assertEqual(planner.last_usage["total_tokens"], 30)


class ReplanNormalizationTests(PipelineHarness):
    def test_replan_is_normalized_against_the_reviewed_candidate_tree(self) -> None:
        # new.txt is absent from the base but present in the reviewed candidate.
        self.workers.on(ExecutionRole.IMPLEMENTER, _write_both, write("new.txt", "second\n"))
        initial = meta_plan({"read": ("feature.txt",), "write": ("feature.txt",), "create": ("new.txt",)})
        replan = meta_plan({"read": ("feature.txt",), "create": ("new.txt",)}, title="Correct")
        result = self.orchestrator(
            self.config(correction_cycles=1),
            planner=[initial, replan],
            reviewer=[review("REVISE", "REPLAN"), review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 2)
        correction = self.run_dir() / "cycles/002/correction"
        contract = (correction / "steps/S01/contract.md").read_text(encoding="utf-8")
        written = contract.split("WRITE SET", 1)[1].split("CREATE SET", 1)[0]
        self.assertIn("- new.txt", written)
        self.assertEqual((correction / "planner.raw.md").read_text(), replan)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    def test_an_impossible_replan_still_stops_before_the_correction_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        initial = meta_plan({"read": ("feature.txt",), "write": ("feature.txt",)})
        result = self.orchestrator(
            self.config(correction_cycles=1),
            planner=[initial, impossible_plan_message(title="Correct"), impossible_plan_message(title="Correct")],
            reviewer=[review("REVISE", "REPLAN")],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "PLAN_REPOSITORY_PRECONDITION_INVALID")
        self.assertIn("step=S01 no_mutation", self.state()["failure"]["detail"])
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertFalse((self.run_dir() / "cycles/002/correction/implementation_bundle.json").exists())


def _write_both(request):  # type: ignore[no-untyped-def]
    (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
    (request.worktree / "new.txt").write_text("first\n", encoding="utf-8")
    return "done\n"


class RuntimeDriftGateTests(PipelineHarness):
    def test_a_worktree_changed_outside_a_step_is_an_integrity_failure(self) -> None:
        """Only a real tree change is a drift; a misclassified path never is."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestration.worker_attempt.candidate_tree_sha",
            return_value="0" * 40,
        ):
            result = self.orchestrator(
                self.config(),
                planner=[initial_plan(("S01", "feature.txt", "Write the feature"))],
                reviewer=[review()],
            ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "REPOSITORY_TREE_DRIFT_UNEXPLAINED")


if __name__ == "__main__":
    unittest.main()
