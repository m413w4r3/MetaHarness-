import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import RunStatus
from metaharness.state import RunStateStore, _exclusive_state_lock
from metaharness.web.api import ARTIFACT_ALLOWLIST


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


class StateLockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def store(self, name: str = "run") -> RunStateStore:
        store = RunStateStore(self.root / name / "state.json")
        store.initialize(name)
        return store

    def test_lock_file_is_internal_and_never_served(self) -> None:
        store = self.store()
        store.update(status=RunStatus.PLANNING)
        self.assertEqual(store.lock_path, self.root / "run" / "state.lock")
        self.assertTrue(store.lock_path.exists())
        self.assertNotIn("state.lock", ARTIFACT_ALLOWLIST)

    def test_lock_excludes_other_threads_until_released(self) -> None:
        store = self.store()
        finished = threading.Event()

        def writer() -> None:
            store.update(status=RunStatus.PLANNING, marker=True)
            finished.set()

        with _exclusive_state_lock(store.lock_path):
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertFalse(finished.wait(0.3))
            self.assertEqual(store.load()["status"], "created")
        thread.join(timeout=5)
        self.assertTrue(finished.is_set())
        self.assertTrue(store.load()["marker"])

    def test_update_if_status_is_compare_and_set(self) -> None:
        store = self.store()
        store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
        written = store.update_if_status(
            RunStatus.AWAITING_PLAN_APPROVAL, execution={"owner": "web"}
        )
        self.assertIsNotNone(written)
        self.assertEqual(written["status"], "awaiting_plan_approval")
        store.update(status=RunStatus.PLANNING, execution={"owner": "orchestrator"})
        self.assertIsNone(
            store.update_if_status("awaiting_plan_approval", execution={"owner": "late web"})
        )
        final = store.load()
        self.assertEqual(final["status"], "planning")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        with self.assertRaises(ValueError):
            store.update_if_status(RunStatus.PLANNING, status="failed")

    def test_web_metadata_never_resurrects_the_approval_gate(self) -> None:
        for iteration in range(30):
            store = self.store(f"race-{iteration}")
            store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
            barrier = threading.Barrier(3)
            observed: list[str] = []
            done = threading.Event()

            def orchestrator() -> None:
                barrier.wait()
                store.update(status=RunStatus.PLANNING, execution={"owner": "orchestrator"})
                store.update(status=RunStatus.WORKTREE_READY, branch="harness/x")

            def web() -> None:
                barrier.wait()
                for _ in range(5):
                    store.update_if_status(
                        RunStatus.AWAITING_PLAN_APPROVAL,
                        plan_identity={"web": True},
                        execution={"owner": "web"},
                    )

            def reader() -> None:
                barrier.wait()
                while not done.is_set():
                    observed.append(store.load()["status"])

            threads = [threading.Thread(target=target) for target in (orchestrator, web, reader)]
            for thread in threads:
                thread.start()
            threads[0].join(timeout=10)
            threads[1].join(timeout=10)
            done.set()
            threads[2].join(timeout=10)
            final = store.load()
            self.assertEqual(final["status"], "worktree_ready")
            self.assertEqual(final["branch"], "harness/x")
            self.assertEqual(final["execution"], {"owner": "orchestrator"})
            # Once the orchestrator left the gate, the gate never reappears.
            if "planning" in observed:
                later = observed[observed.index("planning"):]
                self.assertNotIn("awaiting_plan_approval", later)

    def test_concurrent_updates_never_lose_fields(self) -> None:
        store = self.store()

        def writer(thread_index: int) -> None:
            for item in range(20):
                store.update(status=RunStatus.IMPLEMENTING, **{f"k{thread_index}_{item}": item})

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        final = store.load()
        for index in range(6):
            for item in range(20):
                self.assertEqual(final[f"k{index}_{item}"], item)

    def test_record_failure_is_not_lost_to_a_concurrent_update(self) -> None:
        for iteration in range(30):
            store = self.store(f"failure-{iteration}")
            barrier = threading.Barrier(2)

            def fail() -> None:
                barrier.wait()
                store.record_failure("AGENT_FAILED", "exit status 1")

            def stale() -> None:
                barrier.wait()
                store.update(status=RunStatus.IMPLEMENTING, extra=iteration)

            threads = [threading.Thread(target=fail), threading.Thread(target=stale)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            final = store.load()
            self.assertEqual(final["failure"], {"reason": "AGENT_FAILED", "detail": "exit status 1"})
            self.assertEqual(final["extra"], iteration)

    def test_record_failure_merges_fields_in_one_write(self) -> None:
        store = self.store()
        state = store.record_failure(
            "STEP_WRITE_SET_VIOLATION", "step=S02 unexpected=x.py",
            steps=[{"id": "S02", "status": "failed"}], current_step=None,
        )
        self.assertEqual(state["status"], "failed")
        self.assertEqual(store.load()["steps"], [{"id": "S02", "status": "failed"}])
        with self.assertRaises(ValueError):
            store.record_failure("X", status="committed")

    def test_state_is_always_parseable_under_contention(self) -> None:
        store = self.store()
        stop = threading.Event()
        errors: list[BaseException] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    self.assertIsInstance(json.loads(store.path.read_text(encoding="utf-8")), dict)
                except BaseException as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)

        def writer() -> None:
            for item in range(40):
                store.update(status=RunStatus.VALIDATING, payload="x" * (item * 50))

        readers = [threading.Thread(target=reader) for _ in range(2)]
        writers = [threading.Thread(target=writer) for _ in range(3)]
        for thread in readers + writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=30)
        time.sleep(0.01)
        stop.set()
        for thread in readers:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(store.load()["status"], "validating")


if __name__ == "__main__":
    unittest.main()
