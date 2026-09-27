"""The durable v4 loop: plan against the accepted iteration HEAD and continue."""

from __future__ import annotations

import hashlib
import json
import unittest
from unittest import mock

from metaharness.models import ExecutionRole, RunStatus
from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.planning.normalization import CREATE_EXISTING_TO_WRITE, WRITE_MISSING_TO_CREATE
from metaharness.planning.planner_continue import ContinueDecision
from metaharness.planning.continue_request import planner_continue_dir, read_planner_continue_artifacts

from tests.autonomy.support import Step, meta_plan
from tests.pipeline.support import (
    SPEC, STEP, PipelineHarness, audit, continuation_answer, initial_plan, write,
)
from tests.pipeline_support import git


def milestone(raw: str, number: int) -> str:
    return raw.replace("MILESTONE_ID: M01", f"MILESTONE_ID: M{number:02d}")


class MultiIterationTests(PipelineHarness):
    def test_m01_next_m02_complete_publishes_the_last_accepted_head(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "second\n"),
        )
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=milestone(
                    initial_plan(("S01", "other.txt", "Write the companion")), 2)),
                continuation_answer("COMPLETE"),
            ], auditor=[audit(), audit()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        state = self.state()
        self.assertEqual(state["completion_kind"], "COMPLETE")
        self.assertEqual(state["current_iteration"], 2)
        self.assertEqual(state["current_milestone"]["id"], "M02")
        self.assertEqual([item["milestone_id"] for item in state["completed_iterations"]], ["M01", "M02"])
        self.assertEqual(git(self.worktree(), "rev-parse", "HEAD"), state["commit_sha"])
        record = json.loads((self.run_dir() / "iterations/02/iteration.json").read_text())
        first_record = json.loads((self.run_dir() / "iterations/01/iteration.json").read_text())
        self.assertEqual(record["start_commit"], first_record["end_commit"])
        plan_bytes = (self.run_dir() / "iterations/02/plan/task_plan.json").read_bytes()
        self.assertEqual(record["plan_sha256"], hashlib.sha256(plan_bytes).hexdigest())
        self.assertEqual((self.run_dir() / "iterations/02/execution_selection.json").is_file(), True)

    def test_m02_create_of_m01_path_normalizes_to_write(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("new.txt", "one\n"), write("new.txt", "two\n"),
        )
        m01 = meta_plan(Step("S01", "Create new file", read=(), create=("new.txt",)))
        m02 = meta_plan(Step("S01", "Rewrite new file", read=(), create=("new.txt",)))
        result = self.orchestrator(
            self.config(), planner=[m01], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=milestone(m02, 2)),
                continuation_answer("COMPLETE"),
            ], auditor=[audit(), audit()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        plan = json.loads((self.run_dir() / "iterations/02/plan/task_plan.json").read_text())
        self.assertEqual(plan["steps"][0]["write_set"], ["new.txt"])
        self.assertIn(CREATE_EXISTING_TO_WRITE, {item["code"] for item in plan["normalizations"]})

    def test_m02_write_of_missing_path_normalizes_to_create(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("future.txt", "created\n"),
        )
        m02 = meta_plan(Step(
            "S01", "Create missing file", read=("future.txt",), write=("future.txt",),
        ))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=milestone(m02, 2)),
                continuation_answer("COMPLETE"),
            ], auditor=[audit(), audit()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        plan = json.loads((self.run_dir() / "iterations/02/plan/task_plan.json").read_text())
        self.assertEqual(plan["steps"][0]["create_set"], ["future.txt"])
        self.assertIn(WRITE_MISSING_TO_CREATE, {item["code"] for item in plan["normalizations"]})

    def test_step_s01_can_exist_in_both_milestones(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "new\n"),
        )
        m02 = milestone(initial_plan(("S01", "other.txt", "Write other")), 2)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("COMPLETE"),
            ], auditor=[audit(), audit()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertTrue((self.run_dir() / "cycles/001/implementation/steps/S01/step.json").is_file())
        self.assertTrue((self.run_dir() / "cycles/002/implementation/steps/S01/step.json").is_file())

    def test_red_gate_after_two_audits_reaches_planner_continue(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"), write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, audit("NEEDS_WORK", remaining="Fix the check"),
                        audit("NEEDS_WORK", remaining="Fix the check"), audit())
        m02 = milestone(initial_plan(STEP), 2)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", remaining="- Fix the check", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        request = json.loads((planner_continue_dir(self.run_dir() / "iterations", 1) / "request.json").read_text())
        self.assertIn("Fix the check", request["facts"]["audit"])
        self.assertIn("CHECK_FAILED:test", request["facts"]["evidence"])

    def test_false_complete_is_recovered_by_the_continuation_planner(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        self.workers.on(ExecutionRole.AUDITOR, audit("NEEDS_WORK", remaining="still broken"),
                        audit("NEEDS_WORK", remaining="still broken"), audit())
        m02 = milestone(initial_plan(STEP), 2)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("COMPLETE"),
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual(len(self.continuation.requests), 3)
        self.assertIn("deterministic gate is red", self.continuation.requests[1])
        _request, accepted_raw, accepted_result = read_planner_continue_artifacts(
            planner_continue_dir(self.run_dir() / "iterations", 1),
        )
        self.assertIn("\nNEXT\n", accepted_raw)
        self.assertEqual(accepted_result["accepted_attempt"], 1)

    def test_false_complete_with_project_remainder_is_not_published(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        plan = initial_plan(STEP).replace("PROJECT_REMAINDER\nNONE", "PROJECT_REMAINDER\nM02 still remains")
        result = self.orchestrator(
            self.config(), planner=[plan], continuation=[continuation_answer("COMPLETE")],
        ).run_text(SPEC, run_id="run")
        self.assertNotEqual(result.status, RunStatus.PUBLISHED)
        self.assertEqual(result.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

    def test_complete_is_refused_when_project_remainder_is_real(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        plan = initial_plan(STEP).replace("PROJECT_REMAINDER\nNONE", "PROJECT_REMAINDER\nM02 still remains")
        result = self.orchestrator(
            self.config(), planner=[plan], continuation=[continuation_answer("COMPLETE")],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

    def test_spec_decision_is_the_human_wait_path(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("SPEC_DECISION", question="Which output format should M02 use?"),
            ],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(result.state["failure"]["reason"], "SPEC_DECISION_REQUIRED")

    def test_stagnation_terminates_partial_and_pushes_the_run_branch(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        def transient_edit(request):
            (request.worktree / "feature.txt").write_text("temporary regression\n", encoding="utf-8")
            return "done\n"

        def restore_accepted_tree(request):
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "META AUDIT v1\n\nSTATUS\nDONE\n\nFIXED\n- restored accepted content\n\nREFACTORED\n- none\n\nREMAINING\n- none\n\nRISKS\n- none\nEND META AUDIT\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, transient_edit)
        answers = [
            continuation_answer("NEXT", milestone="M02", remaining="- same remaining", plan_text=milestone(initial_plan(STEP), 2)),
            continuation_answer("NEXT", milestone="M03", remaining="- same remaining", plan_text=milestone(initial_plan(STEP), 3)),
        ]
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=answers,
            auditor=[audit(), restore_accepted_tree],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PARTIAL, result.state.get("failure"))
        self.assertEqual(result.state["partial"]["reason"], "stagnation")
        self.assertEqual(self.remote_tip(result.state["branch"]), result.state["partial"]["last_accepted_commit"])
        self.assertTrue((self.run_dir() / "partial.json").is_file())

    def test_iteration_8_terminates_partial(self) -> None:
        worker_actions = [write("feature.txt", "good\n")]
        worker_actions.extend(write("feature.txt", f"regression {number}\n") for number in range(2, 9))
        self.workers.on(ExecutionRole.IMPLEMENTER, *worker_actions)
        answers = []
        for current in range(1, 9):
            next_id = current + 1
            answers.append(continuation_answer(
                "NEXT", milestone=f"M{next_id:02d}", remaining=f"- outstanding item {current}",
                plan_text=milestone(initial_plan(STEP), next_id),
            ))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=answers,
            auditor=[audit() for _ in range(15)],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PARTIAL, (result.state.get("failure"), self.workers.roles()))
        self.assertEqual(result.state["partial"]["reason"], "max_iterations")
        self.assertEqual(result.state["current_iteration"], 8)
        first = json.loads((self.run_dir() / "iterations/01/iteration.json").read_text())
        self.assertTrue(first["gate_green"])
        self.assertEqual(result.state["commit_sha"], first["end_commit"])
        self.assertEqual(self.remote_tip(result.state["branch"]), first["end_commit"])

    def test_resume_after_m02_audit_commit_does_not_commit_audit_twice(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "good\n"), write("other.txt", "new\n"),
        )

        def audited_m02(request):
            (request.worktree / "audit-note.txt").write_text("reviewed\n", encoding="utf-8")
            return audit()(request)

        self.workers.on(ExecutionRole.AUDITOR, audit(), audited_m02)
        m02 = milestone(initial_plan(("S01", "other.txt", "Write other")), 2)
        orchestrator = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        )
        audit_service = orchestrator._runtime.audit
        real_run = audit_service.run
        crashed = False

        def run_then_crash(*args, **kwargs):
            nonlocal crashed
            report = real_run(*args, **kwargs)
            if args[2].cycle.number == 2 and not crashed:
                crashed = True
                raise RuntimeError("crash after durable M02 audit commit")
            return report

        with mock.patch.object(audit_service, "run", side_effect=run_then_crash):
            interrupted = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(interrupted.status, RunStatus.WAITING_EXTERNAL)
        reports = list(self.run_dir().rglob("report.json"))
        self.assertTrue(reports, {
            "failure": interrupted.state.get("failure"), "roles": self.workers.roles(),
            "checkpoint": self.checkpoint(),
        })
        self.assertTrue(any("cycles/002" in str(path) for path in reports), {
            "reports": [str(path.relative_to(self.run_dir())) for path in reports],
            "failure": interrupted.state.get("failure"), "roles": self.workers.roles(),
            "checkpoint": self.checkpoint(),
        })
        audit_commits_before = git(self.worktree(), "rev-list", "--count", "--grep=metaharness(audit): cycle 2", "HEAD")
        report = json.loads(next(path for path in reports if "cycles/002" in str(path)).read_text())
        self.assertEqual(audit_commits_before, "1", {
            "failure": interrupted.state.get("failure"), "roles": self.workers.roles(),
            "report_commit": report.get("commit_sha"), "head": git(self.worktree(), "rev-parse", "HEAD"),
        })
        resumed = orchestrator.resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(self.workers.roles().count("auditor"), 2)
        self.assertEqual(
            git(self.worktree(), "rev-list", "--count", "--grep=metaharness(audit): cycle 2", "HEAD"),
            "1",
        )

    def test_baseline_regression_remains_red_in_m02_until_m03_fixes_it(self) -> None:
        self.check.write_text(
            "import pathlib, sys\n"
            "sys.exit(0 if pathlib.Path('feature.txt').read_text().strip() in {'base', 'good'} else 1)\n",
            encoding="utf-8",
        )
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "regression\n"),
            write("other.txt", "M02\n"),
            write("feature.txt", "good\n"),
        )
        self.workers.on(
            ExecutionRole.AUDITOR,
            audit("NEEDS_WORK", remaining="fix introduced check regression"),
            audit("NEEDS_WORK", remaining="fix introduced check regression"),
            audit("NEEDS_WORK", remaining="fix introduced check regression"),
            audit("NEEDS_WORK", remaining="fix introduced check regression"),
            audit(),
        )
        m02 = milestone(initial_plan(("S01", "other.txt", "Write other")), 2)
        m03 = milestone(initial_plan(STEP), 3)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", remaining="- fix introduced check regression", plan_text=m02),
                continuation_answer("NEXT", milestone="M03", remaining="- fix introduced check regression", plan_text=m03),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        m02_evidence = json.loads((self.run_dir() / "iterations/02/planner-continue/request.json").read_text())
        self.assertIn("CHECK_FAILED:test", m02_evidence["facts"]["evidence"])
        self.assertIn("BASELINE WARNINGS", m02_evidence["facts"]["evidence"])
        self.assertIn("REGRESSION", m02_evidence["facts"]["evidence"])
        self.assertEqual(
            json.loads((self.run_dir() / "iterations/02/plan/task_plan.json").read_text())["milestone_id"],
            "M02",
        )

    def test_failed_continued_is_audited_and_repaired_in_the_next_iteration(self) -> None:
        def mismatch(_request):
            return CONTRACT_MISMATCH_HEADER + "\nThe approved step instructions were not met."

        first_plan = meta_plan(
            Step("S01", "Create the auxiliary output", read=(), create=("aux.txt",)),
            Step("S02", "Write feature", read=(), create=("feature.txt",)),
        )
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            mismatch, mismatch, mismatch, write("feature.txt", "good\n"),
            write("aux.txt", "repaired\n"),
        )
        self.workers.on(
            ExecutionRole.AUDITOR,
            audit("NEEDS_WORK", remaining="M01/S01 auxiliary output was not created"), audit(),
        )
        m02 = milestone(meta_plan(Step("S01", "Create auxiliary output", read=(), create=("aux.txt",))), 2)
        result = self.orchestrator(
            self.config(), planner=[first_plan], continuation=[
                continuation_answer("NEXT", milestone="M02", remaining="- M01/S01 auxiliary output was not created", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        request = json.loads((self.run_dir() / "iterations/01/planner-continue/request.json").read_text())
        self.assertIn("S01:", request["facts"]["plan"])
        audit_prompt = next(call.prompt for call in self.workers.calls if call.role is ExecutionRole.AUDITOR)
        self.assertIn("failed_continue_steps", audit_prompt)

    def test_resume_at_m02_implementation_replays_only_that_step(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"), write("other.txt", "two\n"),
        )
        m02 = milestone(initial_plan(("S01", "other.txt", "Write other")), 2)
        orchestrator = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02), continuation_answer("COMPLETE"),
            ], auditor=[audit(), audit()],
        )
        runtime_type = type(orchestrator._runtime)
        real = runtime_type.write_checkpoint
        seen = 0

        def crash_after_m02_checkpoint(run_dir, phase, **fields):
            nonlocal seen
            result = real(run_dir, phase, **fields)
            if phase.value == "implement_step" and fields.get("iteration") == 2:
                seen += 1
                if seen == 1:
                    raise RuntimeError("crash after M02 checkpoint")
            return result

        with mock.patch.object(runtime_type, "write_checkpoint", staticmethod(crash_after_m02_checkpoint)):
            failed = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["iteration"], 2)
        self.assertEqual(self.workers.roles().count("implementer"), 1)
        resumed = orchestrator.resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 2)

    def test_paid_continuation_raw_is_reparsed_without_another_call(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        orchestrator = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[continuation_answer("COMPLETE")],
        )
        import metaharness.planning.planner_continue as module
        real = module.write_planner_continue_raw
        interrupted = False

        def write_then_crash(directory, raw):
            nonlocal interrupted
            path = real(directory, raw)
            if not interrupted:
                interrupted = True
                raise RuntimeError("crash after paid continuation response")
            return path

        with mock.patch.object(module, "write_planner_continue_raw", write_then_crash):
            failed = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.checkpoint()["phase"], "planner")
        self.assertEqual(len(self.continuation.requests), 1)
        resumed = orchestrator.resume("run")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(len(self.continuation.requests), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
