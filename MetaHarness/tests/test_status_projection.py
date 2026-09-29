from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import RunStatus
from metaharness.result import RunResult
from metaharness.web.api import list_runs, run_pipeline
from metaharness.web.pages import refresh_seconds_for_run


class StatusProjectionTests(unittest.TestCase):
    def test_result_and_web_api_project_an_old_phase_status(self) -> None:
        state = {"run_id": "legacy", "status": "implementing", "phase": "implement_step"}
        self.assertIs(RunResult.of("/runs/legacy", state).status, RunStatus.RUNNING)
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            run_dir = runs / "legacy"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(list_runs(runs)[0]["status"], RunStatus.RUNNING.value)

    def test_empty_phase_running_status_still_polls(self) -> None:
        self.assertEqual(refresh_seconds_for_run({"status": "running"}), 2)

    def test_web_pipeline_projects_an_old_phase_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = run_pipeline(
                Path(temporary), {"status": "publishing", "phase": "publish"},
            )
        self.assertEqual(next(row["state"] for row in rows if row["key"] == "publish"), "running")


if __name__ == "__main__":
    unittest.main()
