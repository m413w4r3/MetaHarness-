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

    def test_contract_mismatch_retries_with_feedback_after_exact_rollback(self) -> None:
        observed: list[str] = []

        def successful_retry(request):
            observed.append((request.worktree / "feature.txt").read_text(encoding="utf-8"))
            self.assertIn("approved step instructions were not met", request.retry_addendum)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "done\n"

        result = self.run_scripts([self.mismatch(edit="partial\n"), successful_retry])

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(observed, ["base\n"])
        self.assertEqual([call.role for call in self.workers.calls].count(ExecutionRole.IMPLEMENTER), 2)
        self.assertEqual(len(self.planner.requests), 1)
        self.assertTrue((self.run_dir() / "cycles/001/implementation/steps/S01/attempts/01/step.json").exists())

    def test_no_change_retries_with_the_worker_addendum(self) -> None:
        observed: list[str] = []

        def successful_retry(request):
            observed.append(request.retry_addendum or "")
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "done\n"

        result = self.run_scripts([lambda _request: "done\n", successful_retry])

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual(len(observed), 1)
        self.assertIn("No in-scope candidate change", observed[0])
        self.assertEqual(len(self.planner.requests), 1)

    def test_repeated_mismatch_settles_without_an_operator_or_external_wait(self) -> None:
        result = self.run_scripts([self.mismatch(), self.mismatch(), self.mismatch()])

        self.assertNotIn(
            result.status,
            {RunStatus.WAITING_HUMAN, RunStatus.WAITING_EXTERNAL, RunStatus.FAILED},
        )
        step = self.run_dir() / "cycles/001/implementation/steps/S01/step.json"
        import json
        record = json.loads(step.read_text(encoding="utf-8"))
        self.assertEqual((record["status"], record["reason"]), ("FAILED_CONTINUED", "AGENT_CONTRACT_MISMATCH"))
        self.assertEqual(len(self.planner.requests), 1)
        self.assertFalse(step.parent.joinpath("contract_repairs").exists())
        self.assertEqual(git(self.worktree(), "status", "--porcelain"), "")

    def test_available_fallback_executor_runs_after_ordinary_retries(self) -> None:
        config = self.config()
        self.config_path.write_text(
            self.config_path.read_text(encoding="utf-8")
            + '''\n[recovery]\nmax_transient_attempts = 1\n\n[recovery.execution_fallbacks]\nmechanical = ["rescue"]\n\n[model_profiles.rescue]\ndisplay_name = "Rescue"\nroles = ["implementer"]\ndriver = "fake-worker"\nprovider = "test"\nmodel = "fake-rescue"\nselection_mode = "cli"\n''',
            encoding="utf-8",
        )
        config = load_config(self.config_path)
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            self.mismatch(), self.mismatch(), write("feature.txt", "good\n"),
        )

        result = self.orchestrator(
            config, planner=[initial_plan(STEP)],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assertEqual([call.profile_id for call in self.workers.calls[:3]], ["worker", "worker", "rescue"])
        self.assertEqual(len(self.planner.requests), 1)
