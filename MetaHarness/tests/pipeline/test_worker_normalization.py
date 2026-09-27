"""C3/C6 path normalizations reach the worker as the approved contract."""

from __future__ import annotations

import json

from metaharness.models import ExecutionRole, RunStatus
from tests.pipeline.support import PipelineHarness, SPEC, STEP, initial_plan


class WorkerNormalizationTests(PipelineHarness):
    def run_plan(self, raw_plan: str, worker):
        self.workers.on(ExecutionRole.IMPLEMENTER, worker)
        return self.orchestrator(
            self.config(), planner=[raw_plan],
        ).run_text(SPEC, run_id="run")

    def assert_plan_normalization(self, code: str) -> None:
        path = self.run_dir() / "plan.normalizations.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(code, {item["code"] for item in payload["steps"]["S01"]})

    def test_create_existing_is_normalized_before_the_worker(self) -> None:
        raw = initial_plan(STEP).replace(
            "WRITE_SET\n- feature.txt\n\nCREATE_SET\nNONE",
            "WRITE_SET\nNONE\n\nCREATE_SET\n- feature.txt",
        )

        def worker(request):
            self.assertEqual(request.mutable_paths, ("feature.txt",))
            self.assertIn("WRITE SET\n- feature.txt", request.contract)
            self.assertIn("CREATE SET\nNONE", request.contract)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "done\n"

        result = self.run_plan(raw, worker)

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assert_plan_normalization("CREATE_EXISTING_TO_WRITE")
        self.assertEqual(len(self.planner.requests), 1)

    def test_write_missing_is_normalized_to_create_before_the_worker(self) -> None:
        raw = initial_plan(STEP).replace("feature.txt", "generated/new.txt")

        def worker(request):
            self.assertEqual(request.mutable_paths, ("generated/new.txt",))
            self.assertIn("WRITE SET\nNONE", request.contract)
            self.assertIn("CREATE SET\n- generated/new.txt", request.contract)
            (request.worktree / "generated").mkdir()
            (request.worktree / "generated/new.txt").write_text("created\n", encoding="utf-8")
            return "done\n"

        result = self.run_plan(raw, worker)

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assert_plan_normalization("WRITE_MISSING_TO_CREATE")
        self.assertEqual(len(self.planner.requests), 1)

    def test_missing_delete_is_dropped_while_the_other_mutation_runs(self) -> None:
        raw = initial_plan(STEP).replace(
            "DELETE_SET\nNONE", "DELETE_SET\n- absent.txt",
        )

        def worker(request):
            self.assertEqual(request.mutable_paths, ("feature.txt",))
            self.assertIn("WRITE SET\n- feature.txt", request.contract)
            self.assertIn("DELETE SET\nNONE", request.contract)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return "done\n"

        result = self.run_plan(raw, worker)

        self.assertEqual(result.status, RunStatus.PUBLISHED, self.state().get("failure"))
        self.assert_plan_normalization("DROP_MISSING_DELETE")
        self.assertEqual(len(self.planner.requests), 1)
