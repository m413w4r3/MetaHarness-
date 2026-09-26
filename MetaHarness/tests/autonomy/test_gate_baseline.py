"""Scenarios 07 and 10 — the gate compares against the baseline, not against red.

07: the fixture repository ships one test that is already red on main.  The
task changes an independent component, so the old failure stays the only
failure; the gate must accept that candidate with a baseline warning instead of
walking its repair ladder.

10: a required check whose infrastructure is unavailable (an impossible
preflight, conceptually "no Docker") must be skipped with a warning, not turn
into a fatal gate failure.  The fake check never runs and never touches Docker.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from metaharness.models import ExecutionRole

from tests.autonomy.support import (
    SPEC,
    AutonomyHarness,
    Step,
    meta_plan,
    repaired_contract,
)
from tests.pipeline_support import review, write

OLD_FAILURE = "FAILED tests/test_component.py::test_old_failure"

# A dependency-free stand-in for a test runner: it reports failures as the
# ``FAILED <path>::<test>`` lines the gate's baseline parser reads, and it
# exits non-zero while any test fails.  The failure is honest: the fixture test
# really asserts something false on main and really keeps failing afterwards.
FAILING_TESTS_CHECK = '''\
import io
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True


def main() -> int:
    suite = unittest.TestLoader().discover("tests", top_level_dir=".")
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    root = Path.cwd().resolve()
    failed = []
    for test, _report in (*result.failures, *result.errors):
        module = sys.modules[test.__class__.__module__]
        path = Path(module.__file__).resolve().relative_to(root).as_posix()
        failed.append(f"FAILED {path}::{test._testMethodName}")
    print("\\n".join(failed))
    return 1 if failed else 0


sys.exit(main())
'''

COMPONENT_TESTS = '''\
import unittest

import component


class ComponentTests(unittest.TestCase):
    def test_component_value(self) -> None:
        self.assertEqual(component.VALUE, 2)

    def test_old_failure(self) -> None:
        self.assertEqual(1, 2)
'''


class GateBaselineTests(AutonomyHarness):
    def failing_tests(self) -> list[str]:
        """The failing-test lines the check reports on the accepted tree."""

        completed = subprocess.run(
            [sys.executable, str(self.check)], cwd=self.worktree(),
            capture_output=True, text=True, check=False,
        )
        return [line for line in completed.stdout.splitlines() if line.startswith("FAILED ")]

    @unittest.expectedFailure
    def test_a_baseline_red_test_is_not_a_regression(self) -> None:
        self.commit_files({
            "component.py": "VALUE = 1\n",
            "tests/__init__.py": "",
            "tests/test_component.py": COMPONENT_TESTS,
        })
        self.check.write_text(FAILING_TESTS_CHECK, encoding="utf-8")
        plan = meta_plan(
            Step(
                id="S01", title="Raise the component value",
                read=("component.py",), write=("component.py",),
            ),
        )
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("component.py", "VALUE = 2\n"),
            write("component.py", "VALUE = 2\n"),
        )

        result = self.orchestrator(
            self.config(),
            planner=[plan, repaired_contract("S01", "Raise the component value", "component.py")],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
        self.assertEqual(self.failing_tests(), [OLD_FAILURE])
        self.assertIn("baseline", self.durable_report().casefold())

    @unittest.expectedFailure
    def test_a_check_without_its_infrastructure_is_skipped_not_fatal(self) -> None:
        self.green_check()
        marker = self.root / "integration-ran.txt"
        integration = self.root / "integration_check.py"
        integration.write_text(
            "import pathlib, sys\n"
            f"pathlib.Path({str(marker)!r}).write_text('ran\\n', encoding='utf-8')\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        extra_checks = f"""[[check_catalog]]
id = "integration"
argv = [{sys.executable!r}, {str(integration)!r}]
timeout_seconds = 30
preflight_argv = [{sys.executable!r}, "-c", "import sys; sys.exit(3)"]
description = "integration check whose infrastructure is unavailable"
"""
        plan = meta_plan(
            Step(id="S01", title="Write the feature", write=("feature.txt",)),
            required_checks=("test", "integration"),
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        result = self.orchestrator(
            self.config(extra_checks=extra_checks), planner=[plan], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
        self.assertFalse(marker.exists(), "the check ran without its infrastructure")
        self.assertIn("SKIPPED_INFRA", self.durable_report())
