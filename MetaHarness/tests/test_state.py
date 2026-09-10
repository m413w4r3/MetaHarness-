import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import RunStatus
from metaharness.state import RunStateStore


class StateTests(unittest.TestCase):
    def test_initial_state_and_successive_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            state_path = Path(directory_name) / "nested" / "state.json"
            store = RunStateStore(state_path)
            initial = store.initialize("run-123")

            self.assertEqual(initial["schema_version"], 1)
            self.assertEqual(initial["run_id"], "run-123")
            self.assertEqual(initial["status"], "created")
            self.assertIsNone(initial["failure"])
            self.assertEqual(json.loads(state_path.read_text())["status"], "created")

            store.update(status=RunStatus.PLANNING, planner={"request_id": "p1"})
            store.update(status="validating", checks=[{"name": "test", "ok": True}])
            loaded = store.load()

            self.assertEqual(loaded["status"], "validating")
            self.assertEqual(loaded["planner"]["request_id"], "p1")
            self.assertTrue(loaded["checks"][0]["ok"])
            self.assertGreaterEqual(loaded["updated_at"], loaded["started_at"])
            json.loads(state_path.read_text(encoding="utf-8"))

    def test_failure_is_recorded_and_state_remains_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            store = RunStateStore(Path(directory_name) / "state.json")
            store.initialize("run-failure", started_at="2026-01-01T00:00:00Z")
            store.record_failure("check failed", detail={"exit_code": 1})
            loaded = store.load()

            self.assertEqual(loaded["status"], RunStatus.FAILED.value)
            self.assertEqual(loaded["failure"], {
                "reason": "check failed",
                "detail": {"exit_code": 1},
            })
            self.assertIsInstance(json.loads(store.path.read_text()), dict)


if __name__ == "__main__":
    unittest.main()
