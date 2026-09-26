"""Scenario 02 — a CREATE_SET on an existing file must be normalized.

The plan is valid except for one deterministic contradiction: it declares
``CREATE_SET: existing.py`` while that file is already in the start tree.  The
harness owns the correction (CREATE becomes WRITE); the planner is never asked
to re-plan a fact it cannot change.
"""

from __future__ import annotations

import unittest

from metaharness.models import ExecutionRole

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import review, write


class PlannerCreateNormalizationTests(AutonomyHarness):
    @unittest.expectedFailure
    def test_a_create_set_on_an_existing_file_does_not_stop_the_run(self) -> None:
        self.commit_files({"existing.py": "base\n"})
        self.green_check()
        self.workers.on(ExecutionRole.IMPLEMENTER, write("existing.py", "base\nextended\n"))
        plan = meta_plan(
            Step(id="S01", title="Extend the existing module", create=("existing.py",)),
        )

        result = self.orchestrator(
            self.config(), planner=[plan], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assert_not_false_human_stop(result)
        self.assert_not_unrecoverable_hard_stop(result)
        self.assert_run_completed(result)
        self.assertEqual(
            (self.worktree() / "existing.py").read_text(encoding="utf-8"),
            "base\nextended\n",
        )
        # C3 also records the normalization it applied in the plan's durable
        # record; the exact spelling is asserted when that chantier lands.
