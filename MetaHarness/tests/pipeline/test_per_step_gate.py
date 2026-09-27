"""The fast per-step gate and the trusted preflight of one run.

``[gate] per_step`` names the checks a step pays *before* its own commit: only
their *new* regressions refuse the step, the bounded evidence of that refusal is
what the worker receives, and the harness -- never the worker -- reruns and
judges the checks.  The same module proves the preflight contract: one verdict
per run, a skip with a durable warning, and the external wait of a check that
declares itself blocking.
"""

from __future__ import annotations

import json
import sys
import unittest

from metaharness.models import ExecutionRole, RunStatus

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    audit_report,
    continuation_answer,
    git,
    initial_plan,
    write,
)

LINT = """import pathlib, sys
if pathlib.Path('feature.txt').read_text().strip() == 'bad':
    sys.stdout.write('x' * 200000 + '\\n')
    print('FAILED tests/test_feature.py::test_behavior - AssertionError')
    raise SystemExit(1)
"""


class PerStepGateTests(PipelineHarness):
    def lint_check(self) -> str:
        """A second check whose only regression is the ``bad`` content."""

        path = self.root / "lint.py"
        path.write_text(LINT, encoding="utf-8")
        return f"""[[check_catalog]]
id = "lint"
argv = [{sys.executable!r}, {str(path)!r}]
timeout_seconds = 30
"""

    def test_per_step_gate_runs_before_step_commit(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(per_step_gate="lint", extra_checks=self.lint_check()),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.workers.roles().count("implementer"), 2)
        record = json.loads((
            self.run_dir()
            / "cycles/001/implementation/steps/S01/per-step-gate/attempt-01.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(record["check_ids"], ["lint"])
        self.assertEqual(record["regressions"], ["lint"])
        # The refused tree was never committed: the step committed only its
        # second, green attempt on top of the base.
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")

    def test_per_step_regression_retries_step_with_feedback(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(per_step_gate="lint", extra_checks=self.lint_check()),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        first, second = self.workers.calls[0], self.workers.calls[1]
        # The retry keeps the profile and receives only the bounded evidence.
        self.assertEqual(second.profile_id, first.profile_id)
        self.assertIn("The fast per-step gate refused this step tree.", second.prompt)
        self.assertIn("CHECK: lint", second.prompt)
        self.assertIn("NEW FAILING TEST IDS:", second.prompt)
        self.assertIn("tests/test_feature.py::test_behavior", second.prompt)
        self.assertIn("FAILURE EXCERPT:", second.prompt)
        self.assertIn("CHANGED PATHS:", second.prompt)
        self.assertNotIn("x" * 5000, second.prompt, "the whole log reached the worker")
        self.assertIsNone(self.state().get("failure"))

    def test_a_worker_claim_never_closes_the_gate(self) -> None:
        """A report is informative: only the rerun decides, and it stays red."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.AUDITOR,
            *(
                lambda _request: audit_report(
                    "DONE", fixed="the check passes now; the tree is already correct",
                )
                for _ in range(2)
            ),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        m02 = initial_plan(STEP).replace("MILESTONE_ID: M01", "MILESTONE_ID: M02")
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP)], continuation=[
                continuation_answer("NEXT", milestone="M02", plan_text=m02),
                continuation_answer("COMPLETE"),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state())
        self.assertEqual(len(self.continuation.requests), 2)
        # The claimed fix never became evidence: the harness reran the check.
        m01_evidence = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/evidence.json").read_text()
        )
        self.assertFalse(m01_evidence["deterministic_passed"])
        reports = sorted((self.run_dir() / "cycles/001/audit").glob("*/report.json"))
        self.assertEqual(len(reports), 2)
        self.assertTrue(json.loads(
            (self.run_dir() / "iterations/01/planner-continue/request.json").read_text()
        )["facts"]["evidence"].find("CHECK_FAILED:test") >= 0)


class PreflightTests(PipelineHarness):
    def preflight_check(self, *, status: int, blocking: bool = False) -> str:
        """A check whose infrastructure is decided by one counter probe."""

        marker = self.root / "preflight-runs.txt"
        probe = self.root / "probe.py"
        probe.write_text(
            "import pathlib, sys\n"
            f"marker = pathlib.Path({str(marker)!r})\n"
            "count = int(marker.read_text()) if marker.exists() else 0\n"
            "marker.write_text(str(count + 1))\n"
            f"raise SystemExit({status})\n",
            encoding="utf-8",
        )
        self.preflight_marker = marker
        return f"""[[check_catalog]]
id = "integration"
argv = [{sys.executable!r}, {str(self.check)!r}]
timeout_seconds = 30
blocking = {'true' if blocking else 'false'}
preflight_argv = [{sys.executable!r}, {str(probe)!r}]
"""

    def test_preflight_runs_once_per_run(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(extra_checks=self.preflight_check(status=0)),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        # Two gate runs answered this run (the gate and its post-audit rerun):
        # the trusted probe was still paid exactly once.
        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.preflight_marker.read_text(), "1")
        verdicts = json.loads(
            (self.run_dir() / "preflights.json").read_text(encoding="utf-8")
        )["checks"]
        self.assertEqual(sorted(verdicts), ["integration"])
        self.assertEqual(verdicts["integration"]["status"], "PASS")

    def test_nonblocking_preflight_failure_is_skipped(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(extra_checks=self.preflight_check(status=3)),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(self.preflight_marker.read_text(), "1")
        state = self.state()
        self.assertEqual(state["skipped_checks"], ["integration"])
        self.assertTrue(any("PREFLIGHT_FAILED" in item for item in state["check_warnings"]))
        evidence = json.loads((
            self.run_dir() / "cycles/001/checks/post-implementation/evidence.json"
        ).read_text(encoding="utf-8"))
        self.assertTrue(evidence["deterministic_passed"], evidence["failures"])

    def test_blocking_preflight_failure_routes_wait_external(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(extra_checks=self.preflight_check(status=3, blocking=True)),
            planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL, self.state())
        self.assertEqual(
            self.state()["failure"]["reason"], "CHECK_INFRASTRUCTURE_UNAVAILABLE",
        )
        self.assertEqual(self.preflight_marker.read_text(), "1")
        self.assertNotEqual(result.status, RunStatus.WAITING_HUMAN)
        # The blocking condition is an external wait, never a new human door,
        # and it is decided before any agent is bought.
        self.assertEqual(self.workers.roles(), [])
        self.assertFalse((self.run_dir() / "cycles/001/audit").exists())


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
