"""The autonomy contract of a red deterministic gate, proven on the real pipeline.

The pure v4 policy is proven in ``tests/test_recovery_policy.py`` and
``tests/test_recovery_matrix.py``; this module proves the durable route: a red
gate consumes its ladder without any human wait.
"""

from __future__ import annotations

import unittest

from metaharness.models import ExecutionRole, RunStatus

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    initial_plan,
    ladder_strategies,
    review,
    write,
)


class CorrectnessRouteIsAutonomousTests(PipelineHarness):
    """The durable route: a red gate consumes its ladder before any human wait."""

    def test_a_red_gate_is_repaired_autonomously_without_a_human_wait(self) -> None:
        counter = self.root / "gate-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "feature = pathlib.Path('feature.txt').read_text().strip()\n"
            "raise SystemExit(1 if feature == 'bad' else 0)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))

        result = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertGreaterEqual(int(counter.read_text()), 2, "the gate was never red")
        self.assertEqual(
            ladder_strategies(self), ["repair_targeted"],
            "the red gate consumed its autonomous rung before anything else",
        )
        self.assertEqual(
            human_terminals(self), [],
            "a recovery loop routed this run to a human wait",
        )


def human_terminals(harness: PipelineHarness) -> list[tuple[str, str]]:
    """(reason, terminal status) of every exhausted recovery loop, from the trace."""

    routes: list[tuple[str, str]] = []
    for event in harness.trace_events():
        if event.get("event") != "recovery.exhausted":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if str(data.get("terminal_status")) == RunStatus.WAITING_HUMAN.value:
            routes.append((str(data.get("reason")), str(data.get("terminal_status"))))
    return routes


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
