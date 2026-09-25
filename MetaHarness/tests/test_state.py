import hashlib
import fcntl
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import RunStatus
from metaharness.resume import (
    CHECKPOINT_INTEGRITY_OPERATION,
    CHECKPOINT_NAME,
    ResumeCheckpoint,
    ResumePhase,
    checkpoint_payload,
    plan_identity_from_mapping,
    resume_info,
)
from metaharness.run_options import (
    RUN_SCHEMA_UNSUPPORTED,
    SCHEMA_VERSION,
    RunOptions,
    RunOptionsError,
    read_run_options_for_state,
)
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


class LockContentionProbe:
    """Record, per thread name, that acquiring the state lock had to block.

    Installed in place of ``fcntl.flock``: an exclusive request is first tried
    without blocking; if the kernel reports the lock as held by another open
    file description, the thread's event is set before the real blocking call.
    A set event is therefore proof that the lock really excluded that thread.
    """

    def __init__(self) -> None:
        self._real_flock = fcntl.flock
        self._events: dict[str, threading.Event] = {}
        self._guard = threading.Lock()

    def blocked(self, thread_name: str) -> threading.Event:
        with self._guard:
            return self._events.setdefault(thread_name, threading.Event())

    def flock(self, fd: int, operation: int) -> None:
        if operation == fcntl.LOCK_EX:
            try:
                return self._real_flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.blocked(threading.current_thread().name).set()
        return self._real_flock(fd, operation)

    def install(self) -> "mock._patch":
        return mock.patch.object(fcntl, "flock", self.flock)


def pause_inside_lock(store: RunStateStore, thread_name: str) -> tuple[threading.Event, threading.Event]:
    """Park *thread_name* inside its first critical section, right after load.

    Every mutation of the store loads the state under the exclusive lock, so
    the parked thread holds the lock with a stale read in hand: exactly the
    interleaving a lost update or a resurrected status would need.
    """

    reached = threading.Event()
    resume = threading.Event()
    real_load = store.load

    def load() -> dict:
        state = real_load()
        if threading.current_thread().name == thread_name and not reached.is_set():
            reached.set()
            resume.wait(10)
        return state

    store.load = load  # type: ignore[method-assign]
    return reached, resume


class StateLockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # These tests exercise locking and atomic replacement, not durability:
        # skipping the disk flush keeps every interleaving and write intact.
        fsync = mock.patch.object(os, "fsync")
        fsync.start()
        self.addCleanup(fsync.stop)

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
        probe = LockContentionProbe()
        finished = threading.Event()

        def writer() -> None:
            store.update(status=RunStatus.PLANNING, marker=True)
            finished.set()

        with _exclusive_state_lock(store.lock_path), probe.install():
            thread = threading.Thread(target=writer, name="writer")
            thread.start()
            # The writer really reached flock and the kernel refused it.
            self.assertTrue(probe.blocked("writer").wait(5))
            self.assertFalse(finished.is_set())
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

    def run_interleaved(self, store: RunStateStore, first: tuple, second: tuple) -> dict:
        """Park *first* inside the lock, prove *second* blocks, then release.

        Each argument is ``(thread_name, target)``; the targets' return values
        are collected by thread name.
        """

        probe = LockContentionProbe()
        reached, resume = pause_inside_lock(store, first[0])
        results: dict[str, object] = {}

        def runner(name: str, target) -> threading.Thread:
            thread = threading.Thread(
                target=lambda: results.__setitem__(name, target()), name=name,
            )
            thread.start()
            return thread

        with probe.install():
            threads = [runner(*first)]
            self.assertTrue(reached.wait(5))
            threads.append(runner(*second))
            self.assertTrue(probe.blocked(second[0]).wait(5))
            self.assertNotIn(second[0], results)
            resume.set()
            for thread in threads:
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
        return results

    def test_web_metadata_never_resurrects_the_approval_gate(self) -> None:
        def orchestrator() -> dict:
            store.update(status=RunStatus.PLANNING, execution={"owner": "orchestrator"})
            return store.update(status=RunStatus.WORKTREE_READY, branch="harness/x")

        def web() -> dict | None:
            return store.update_if_status(
                RunStatus.AWAITING_PLAN_APPROVAL,
                plan_identity={"web": True},
                execution={"owner": "web"},
            )

        # The web holds the lock with the gate still open: its write is legal,
        # and the orchestrator's later writes are merged on top of it.
        store = self.store("web-first")
        store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
        results = self.run_interleaved(store, ("web", web), ("orchestrator", orchestrator))
        self.assertIsNotNone(results["web"])
        final = store.load()
        self.assertEqual(final["status"], "worktree_ready")
        self.assertEqual(final["branch"], "harness/x")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        self.assertEqual(final["plan_identity"], {"web": True})

        # The orchestrator holds the lock with a read of the open gate: the web
        # must then observe the closed gate and write nothing.
        store = self.store("orchestrator-first")
        store.update(status=RunStatus.AWAITING_PLAN_APPROVAL)
        results = self.run_interleaved(store, ("orchestrator", orchestrator), ("web", web))
        self.assertIsNone(results["web"])
        final = store.load()
        self.assertEqual(final["status"], "worktree_ready")
        self.assertEqual(final["branch"], "harness/x")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        self.assertIsNone(final["plan_identity"])
        # Once the orchestrator left the gate, the gate never reappears.
        self.assertIsNone(web())
        self.assertEqual(store.load()["status"], "worktree_ready")

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
        # Both orders of the dangerous interleaving: one writer holds a read of
        # the state while the other must wait instead of overwriting it.
        for order in (("fail", "stale"), ("stale", "fail")):
            with self.subTest(first=order[0]):
                store = self.store(f"failure-{order[0]}-first")
                targets = {
                    "fail": lambda: store.record_failure("AGENT_RUNTIME_FAILED", "exit status 1"),
                    "stale": lambda: store.update(status=RunStatus.IMPLEMENTING, extra=order[0]),
                }
                self.run_interleaved(
                    store, (order[0], targets[order[0]]), (order[1], targets[order[1]]),
                )
                final = store.load()
                self.assertEqual(
                    final["failure"], {"reason": "AGENT_RUNTIME_FAILED", "detail": "exit status 1"}
                )
                self.assertEqual(final["extra"], order[0])

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

class FrozenRunOptionsSchemaTests(unittest.TestCase):
    """An older snapshot fails closed; the run directory is never rewritten."""

    def older_snapshot_bytes(self) -> bytes:
        options = RunOptions(
            schema_version=SCHEMA_VERSION,
            pipeline_version=2,
            protocol="v2",
            decomposition="balanced",
            execution_mode_policy="auto",
            single_step_max_mutable_paths=2,
            staged_step_max_mutable_paths=6,
            semantic_revision_enabled=False,
            max_check_repair_attempts=0,
            max_review_repair_cycles=0,
            planner_profile="planner",
            mechanical_profile="worker",
            reasoning_profile="worker",
            agentic_profile="worker",
            final_reviewer_profile="reviewer",
        )
        snapshot = options.to_dict()
        snapshot["schema_version"] = SCHEMA_VERSION - 1
        return (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

    def test_older_schema_run_fails_closed_and_keeps_its_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            store = RunStateStore(directory / "state.json")
            store.initialize("legacy-run")
            path = directory / "run_options.json"
            data = self.older_snapshot_bytes()
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            state = store.update(status=RunStatus.CREATED, run_options_sha256=digest)

            with self.assertRaises(RunOptionsError) as caught:
                read_run_options_for_state(directory, state)

            self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(store.load()["run_options_sha256"], digest)
            self.assertEqual(store.load()["status"], RunStatus.CREATED.value)


class FrozenCheckpointSchemaTests(unittest.TestCase):
    """Only the current checkpoint schema is resumable; older ones stay read-only."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.state_path = self.run_dir / "state.json"
        self.checkpoint_path = self.run_dir / CHECKPOINT_NAME

    def waiting_state(self) -> dict:
        store = RunStateStore(self.state_path)
        store.initialize("run")
        return store.update(
            status=RunStatus.WAITING_EXTERNAL, planning_protocol="v2",
            failure={"reason": "AGENT_TIMEOUT", "detail": "timed out"},
        )

    def checkpoint_bytes(self) -> bytes:
        checkpoint = ResumeCheckpoint(
            phase=ResumePhase.IMPLEMENT_STEP,
            step_id="S01",
            expected_head_sha="6" * 40,
            expected_tree_sha="7" * 40,
            execution_selection_sha256="8" * 64,
            plan_identity=plan_identity_from_mapping({
                "raw_sha256": "1" * 64, "contract_sha256": "2" * 64,
                "bundle_sha256": "3" * 64, "execution_sha256": "4" * 64,
                "checks_sha256": "5" * 64,
            }),
        )
        return (json.dumps(checkpoint_payload(checkpoint), indent=2, sort_keys=True) + "\n").encode()

    def test_an_older_checkpoint_schema_is_unsupported_and_never_an_incident(self) -> None:
        state = self.waiting_state()
        payload = json.loads(self.checkpoint_bytes())
        payload["schema_version"] -= 1
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        self.checkpoint_path.write_bytes(data)
        state_before = self.state_path.read_bytes()

        info = resume_info(self.run_dir, state)

        self.assertFalse(info.resumable)
        self.assertEqual(info.operation, RUN_SCHEMA_UNSUPPORTED)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, info.reason)
        self.assertNotEqual(info.operation, CHECKPOINT_INTEGRITY_OPERATION)
        self.assertEqual(self.checkpoint_path.read_bytes(), data)
        self.assertEqual(self.state_path.read_bytes(), state_before)

    def test_the_current_checkpoint_schema_is_resumable(self) -> None:
        state = self.waiting_state()
        self.checkpoint_path.write_bytes(self.checkpoint_bytes())

        info = resume_info(self.run_dir, state)

        self.assertTrue(info.resumable, info.reason)
        self.assertEqual(info.phase, ResumePhase.IMPLEMENT_STEP.value)
        self.assertEqual(info.step_id, "S01")
        self.assertEqual(info.expected_tree, "7" * 40)

    def test_a_corrupt_current_checkpoint_is_an_integrity_failure(self) -> None:
        state = self.waiting_state()
        payload = json.loads(self.checkpoint_bytes())
        payload["step_id"] = "not-a-step"
        for name, data in (
            ("incoherent identity", json.dumps(payload, indent=2, sort_keys=True) + "\n"),
            ("unreadable bytes", "{not json"),
        ):
            with self.subTest(corruption=name):
                self.checkpoint_path.write_text(data, encoding="utf-8")
                info = resume_info(self.run_dir, state)
                self.assertFalse(info.resumable)
                self.assertEqual(info.operation, CHECKPOINT_INTEGRITY_OPERATION)
                self.assertNotEqual(info.operation, RUN_SCHEMA_UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
