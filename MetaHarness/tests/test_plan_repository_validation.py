"""Repository-topology preconditions of a META PLAN v2 before approval."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metaharness.context import build_context, render_context
from metaharness.gitops import resolve_tree
from metaharness.llm.chat import TextLLMResult
from metaharness.models import (
    ContextConfig,
    ExecutionClass,
    ExecutionRole,
    ExecutionMode,
    ImplementationStep,
    PlanDecision,
    RunStatus,
    TaskPlanV2,
)
from metaharness.orchestrator import Orchestrator
from metaharness.plan_recovery import PlanRecoveryError, plan_recovery_info
from metaharness.plan_repository_validation import (
    PathPreconditionViolation,
    PlanRepositoryPreconditionError,
    RepositoryPreconditions,
    plan_repository_violations,
    render_conflict_evidence,
    render_violations,
    validate_plan_repository_topology,
)
from metaharness.planning_v2 import PlannerV2
from metaharness.usage import phase_usage_summary
from tests.pipeline_support import PipelineHarness, git, review, write

X = "pkg/x.py"
AW010_PATH = "backend/src/cti_app/domain/reference_corpus.py"


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

    def violations(self, tree: str, *steps: ImplementationStep) -> list[tuple[str, str, str]]:
        return [
            (item.step_id, item.kind, item.path)
            for item in plan_repository_violations(self.repo, tree, task_plan(*steps))
        ]


class TopologyTests(_Repo):
    def test_a_create_existing_path_is_invalid(self) -> None:
        plan = task_plan(step("S01", read=("README.md",), create=(X,)))
        with self.assertRaises(PlanRepositoryPreconditionError) as caught:
            validate_plan_repository_topology(self.repo, self.with_x, plan)
        self.assertEqual(str(caught.exception), f"step=S01 create_exists={X}")
        self.assertEqual(caught.exception.code, "PLAN_REPOSITORY_PRECONDITION_INVALID")

    def test_b_write_missing_path_is_invalid(self) -> None:
        self.assertEqual(
            self.violations(self.without_x, step("S01", read=(X,), write_set=(X,))),
            [("S01", "read_missing", X), ("S01", "write_missing", X)],
        )

    def test_c_delete_missing_path_is_invalid(self) -> None:
        self.assertEqual(
            self.violations(self.without_x, step("S01", read=(X,), delete=(X,))),
            [("S01", "read_missing", X), ("S01", "delete_missing", X)],
        )

    def test_d_read_write_existing_path_is_valid(self) -> None:
        validate_plan_repository_topology(
            self.repo, self.with_x, task_plan(step("S01", read=(X,), write_set=(X,))),
        )

    def test_e_created_path_can_be_read_and_written_later(self) -> None:
        self.assertEqual(self.violations(
            self.without_x,
            step("S01", read=("README.md",), create=(X,)),
            step("S02", read=(X,), write_set=(X,)),
        ), [])

    def test_f_second_create_of_the_same_path_is_invalid(self) -> None:
        self.assertEqual(self.violations(
            self.without_x,
            step("S01", read=("README.md",), create=(X,)),
            step("S02", read=("README.md",), create=(X,)),
        ), [("S02", "create_exists", X)])

    def test_g_deleted_path_cannot_be_read_or_written_later(self) -> None:
        self.assertEqual(self.violations(
            self.with_x,
            step("S01", read=(X,), delete=(X,)),
            step("S02", read=(X,), write_set=(X,)),
        ), [("S02", "read_missing", X), ("S02", "write_missing", X)])

    def test_h_deleted_path_can_be_created_again(self) -> None:
        self.assertEqual(self.violations(
            self.with_x,
            step("S01", read=(X,), delete=(X,)),
            step("S02", read=("README.md",), create=(X,)),
        ), [])

    def test_k_start_tree_decides_write_versus_create(self) -> None:
        writes = step("S01", read=(X,), write_set=(X,))
        creates = step("S01", read=("README.md",), create=(X,))
        # The candidate of a replan has X although the base does not.
        self.assertEqual(self.violations(self.with_x, writes), [])
        self.assertEqual(self.violations(self.with_x, creates), [("S01", "create_exists", X)])
        self.assertEqual(self.violations(self.without_x, creates), [])

    def test_blocked_plan_has_no_preconditions(self) -> None:
        plan = task_plan(step("S01", read=(X,), write_set=(X,)))
        blocked = TaskPlanV2(PlanDecision.BLOCKED, *list(plan.__dict__.values())[1:])
        self.assertEqual(plan_repository_violations(self.repo, self.without_x, blocked), ())

    def test_violations_are_grouped_and_bounded(self) -> None:
        paths = [f"a/{index}.py" for index in range(10)]
        rendered = render_violations(
            [PathPreconditionViolation("S01", "write_missing", path) for path in paths]
            + [PathPreconditionViolation("S01", "read_missing", "b.py")]
        )
        self.assertEqual(rendered.splitlines(), [
            "step=S01 read_missing=b.py",
            "step=S01 write_missing=" + ",".join(paths[:8]) + " (+2 more)",
        ])

    def test_conflict_evidence_reads_the_immutable_tree_bounded(self) -> None:
        (self.repo / X).write_text("WORKING TREE ONLY\n", encoding="utf-8")
        big = "pkg/big.py"
        (self.repo / big).write_text("x" * 20000, encoding="utf-8")
        (self.repo / ".env").write_text("TOKEN=hidden\n", encoding="utf-8")
        git(self.repo, "add", big, ".env")
        git(self.repo, "commit", "-qm", "big")
        tree = resolve_tree(self.repo, "HEAD")
        evidence = render_conflict_evidence(self.repo, tree, [
            PathPreconditionViolation("S01", "create_exists", path)
            for path in (X, big, ".env")
        ])
        self.assertIn("VALUE = 1", evidence)
        self.assertNotIn("WORKING TREE ONLY", evidence)
        self.assertIn("SIZE: 20000 bytes (first 8192 bytes)", evidence)
        self.assertNotIn("x" * 8193, evidence)
        self.assertIn("withheld (sensitive path)", evidence)
        self.assertNotIn("hidden", evidence)


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


class PlannerCorrectionTests(PipelineHarness):
    """The bounded planner correction on a real repository and run."""

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

    def invalid_aw010(self) -> str:
        return meta_plan({"read": ("feature.txt",), "write": ("feature.txt",), "create": (AW010_PATH,)})

    def valid_distinct_module(self) -> str:
        return meta_plan({
            "read": ("feature.txt",), "write": ("feature.txt",),
            "create": ("backend/src/cti_app/domain/benchmark_dataset.py",),
        })

    def test_i_and_aw010_only_the_corrected_plan_becomes_authority(self) -> None:
        def implement(request):  # type: ignore[no-untyped-def]
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            (request.worktree / "backend/src/cti_app/domain/benchmark_dataset.py").write_text(
                "DATASET = ()\n", encoding="utf-8",
            )
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, implement)
        result = self.orchestrator(
            self.config(), planner=[self.invalid_aw010(), self.valid_distinct_module()],
            reviewer=[review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        run_dir = self.run_dir()
        self.assertEqual(len(self.planner.requests), 2)
        correction = self.planner.requests[1]
        self.assertTrue(correction.startswith(self.planner.requests[0].rstrip("\n")))
        self.assertIn("DETERMINISTIC PRE-APPROVAL CORRECTION", correction)
        self.assertIn(f"S01: create_exists: {AW010_PATH}", correction)
        self.assertIn("Re-emit one COMPLETE META PLAN v2", correction)
        self.assertIn("class ReferenceMember", correction)
        # The rejected answer is audited, never authority.
        attempt = run_dir / "planner-attempts/01"
        self.assertEqual((attempt / "planner.raw.md").read_text(), self.invalid_aw010())
        record = json.loads((attempt / "repository_preconditions.json").read_text())
        self.assertEqual(record["violations"], [
            {"step_id": "S01", "kind": "create_exists", "path": AW010_PATH},
        ])
        self.assertEqual((run_dir / "planner.raw.md").read_text(), self.valid_distinct_module())
        self.assertEqual((run_dir / "planner.request.txt").read_text(), correction)
        task_plan_json = json.loads((run_dir / "task_plan.json").read_text())
        self.assertNotIn(AW010_PATH, json.dumps(task_plan_json))
        self.assertNotIn(AW010_PATH, (run_dir / "steps/S01/contract.md").read_text())
        self.assertEqual(self.trace_names().count("plan.completed"), 1)
        self.assertEqual(self.workers.roles(), ["implementer"])

    def test_j_two_invalid_plans_fail_planning_before_any_worker(self) -> None:
        result = self.orchestrator(
            replace(self.config(), planning=replace(self.config().planning, max_preapproval_corrections=1)),
            planner=[self.invalid_aw010(), self.invalid_aw010()],
            reviewer=[review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        state = self.state()
        self.assertEqual(state["failure"]["reason"], "PLAN_REPOSITORY_PRECONDITION_INVALID")
        self.assertIn(f"create_exists={AW010_PATH}", state["failure"]["detail"])
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.workers.calls, [])
        run_dir = self.run_dir()
        for name in ("implementation_bundle.json", "task_plan.json", "planner.raw.md", "steps"):
            self.assertFalse((run_dir / name).exists(), name)
        self.assertTrue((run_dir / "planner-attempts/02/repository_preconditions.json").is_file())
        # A planning failure: the run is at its PLANNER checkpoint, never at a step.
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertFalse(self.worktree().exists())
        self.assertTrue(plan_recovery_info(run_dir, state).eligible)

    def test_l_operator_recovery_is_validated_before_approval(self) -> None:
        orchestrator = self.orchestrator(
            replace(self.config(), planning=replace(self.config().planning, max_preapproval_corrections=1)),
            planner=[self.invalid_aw010(), self.invalid_aw010()],
            reviewer=[review()],
        )
        orchestrator.run_text("Make feature.txt good.\n", run_id="run")
        before = self.state()
        with self.assertRaises(PlanRecoveryError) as caught:
            orchestrator.recover_plan("run", self.invalid_aw010())
        self.assertIn("PLAN_REPOSITORY_PRECONDITION_INVALID", str(caught.exception))
        self.assertIn(f"create_exists={AW010_PATH}", str(caught.exception))
        self.assertEqual(self.state()["updated_at"], before["updated_at"])
        self.assertFalse((self.run_dir() / "implementation_bundle.json").exists())
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertEqual(self.workers.calls, [])

    def test_both_planner_calls_stay_in_usage_accounting(self) -> None:
        chat = _Recording([self.invalid_aw010(), self.valid_distinct_module()])
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


class ReplanPreconditionTests(PipelineHarness):
    def test_k_replan_is_validated_against_the_reviewed_candidate_tree(self) -> None:
        # new.txt is absent from the base but present in the reviewed candidate.
        self.workers.on(ExecutionRole.IMPLEMENTER, _write_both, write("new.txt", "second\n"))
        initial = meta_plan({"read": ("feature.txt",), "write": ("feature.txt",), "create": ("new.txt",)})
        invalid = meta_plan({"read": ("feature.txt",), "create": ("new.txt",)}, title="Correct")
        valid = meta_plan({"read": ("new.txt",), "write": ("new.txt",)}, title="Correct")
        result = self.orchestrator(
            self.config(review_repair=1),
            planner=[initial, invalid, valid],
            reviewer=[review("REVISE", "REPLAN"), review()],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 3)
        self.assertIn("step=S01 create_exists=new.txt", self.planner.requests[2])
        correction = self.run_dir() / "cycles/002/correction"
        self.assertEqual((correction / "planner-attempts/01/planner.raw.md").read_text(), invalid)
        self.assertEqual((correction / "planner.raw.md").read_text(), valid)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])

    def test_invalid_replan_twice_fails_before_the_correction_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        invalid = meta_plan({"read": ("missing.txt",), "write": ("missing.txt",)}, title="Correct")
        result = self.orchestrator(
            self.config(review_repair=1),
            planner=[meta_plan({"read": ("feature.txt",), "write": ("feature.txt",)}), invalid, invalid],
            reviewer=[review("REVISE", "REPLAN")],
        ).run_text("Make feature.txt good.\n", run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "PLAN_REPOSITORY_PRECONDITION_INVALID")
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertFalse((self.run_dir() / "cycles/002/correction/implementation_bundle.json").exists())


def _write_both(request):  # type: ignore[no-untyped-def]
    (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
    (request.worktree / "new.txt").write_text("first\n", encoding="utf-8")
    return "done\n"


class RuntimeDriftGateTests(_Repo):
    def test_m_post_approval_drift_still_fails_the_runtime_gate(self) -> None:
        # Valid at planning time: X is absent from the approved start tree.
        creates = step("S01", read=("README.md",), create=(X,))
        validate_plan_repository_topology(self.repo, self.without_x, task_plan(creates))
        # The worktree then really drifts: X appears after approval.
        drift = Orchestrator._step_contract_drift(
            object(), self.repo, self.with_x, self.with_x, creates,  # type: ignore[arg-type]
        )
        self.assertEqual(drift, f"create_exists={X}")
        self.assertEqual(
            Orchestrator._step_contract_drift(
                object(), self.repo, self.with_x, self.without_x, creates,  # type: ignore[arg-type]
            ),
            "worktree changed outside a step",
        )


if __name__ == "__main__":
    unittest.main()
