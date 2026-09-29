"""Worker contract failures use the ordinary retry and fallback ladder."""

from __future__ import annotations

from pathlib import Path

from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.config import load_config
from metaharness.models import ExecutionRole, RunStatus
from tests.pipeline.support import PipelineHarness, SPEC, STEP, git, initial_plan, write


class WorkerRecoveryTests(PipelineHarness):
    def mismatch(self, *, edit: str | None = None):
        def action(request):
            if edit is not None:
                (request.worktree / "feature.txt").write_text(edit, encoding="utf-8")
            return CONTRACT_MISMATCH_HEADER + "\nThe approved step instructions were not met."
        return action

    def run_scripts(self, scripts):
        self.workers.on(ExecutionRole.IMPLEMENTER, *scripts)
        return self.orchestrator(
            self.config(), planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

    def test_contract_mismatch_is_rolled_back_and_sent_to_replanning(self) -> None:
        result = self.run_scripts([self.mismatch(edit="partial\n")])

        self.assertIn(result.status, {RunStatus.COMMITTED, RunStatus.PUBLISHED})
        self.assertEqual([call.role for call in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 1)
        self.assertEqual(len(self.planner.requests), 1)
        self.assertTrue((self.run_dir() / "cycles/001/implementation/steps/S01/attempts/01/step.json").exists())
        import json
        request = json.loads((self.run_dir() / "iterations/01/planner-continue/request.json").read_text())
        self.assertIn("approved step instructions were not met", request["facts"]["plan"])

    def test_no_change_is_sent_to_replanning_without_reexecution(self) -> None:
        result = self.run_scripts([lambda _request: "done\n"])

        self.assertIn(result.status, {RunStatus.COMMITTED, RunStatus.PUBLISHED})
        self.assertEqual(
            [call.role for call in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 1,
        )
        self.assertEqual(len(self.planner.requests), 1)
        import json
        request = json.loads((self.run_dir() / "iterations/01/planner-continue/request.json").read_text())
        self.assertIn("No in-scope candidate change", request["facts"]["plan"])

    def test_mismatch_settles_without_same_executor_retry(self) -> None:
        result = self.run_scripts([self.mismatch(), self.mismatch(), self.mismatch()])

        self.assertNotIn(
            result.status,
            {RunStatus.WAITING_HUMAN, RunStatus.WAITING_EXTERNAL, RunStatus.FAILED},
        )
        step = self.run_dir() / "cycles/001/implementation/steps/S01/step.json"
        import json
        record = json.loads(step.read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["reason"]), ("FAILED_CONTINUED", "AGENT_CONTRACT_MISMATCH"))
        self.assertEqual(
            [call.role for call in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 1,
        )
        self.assertEqual(len(self.planner.requests), 1)
        self.assertFalse(step.parent.joinpath("contract_repairs").exists())
        self.assertEqual(git(self.worktree(), "status", "--porcelain"), "")

    def test_contract_mismatch_does_not_use_fallback_executor(self) -> None:
        """A missing prerequisite is sent to replanning, not another worker."""

        self.config()
        self.config_path.write_text(
            self.config_path.read_text(encoding="utf-8")
            + '''\n[recovery.execution_fallbacks]\nmechanical = ["rescue"]\n\n[model_profiles.rescue]\ndisplay_name = "Rescue"\nroles = ["implementer"]\ndriver = "fake-worker"\nprovider = "test"\nmodel = "fake-rescue"\nselection_mode = "cli"\n''',
            encoding="utf-8",
        )
        config = load_config(self.config_path)
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            self.mismatch(), write("feature.txt", "good\n"),
        )

        result = self.orchestrator(
            config, planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertIn(result.status, {RunStatus.COMMITTED, RunStatus.PUBLISHED})
        implementers = [call.profile_id for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER]
        self.assertEqual(implementers, ["worker"])
        self.assertEqual(len(self.planner.requests), 1)
