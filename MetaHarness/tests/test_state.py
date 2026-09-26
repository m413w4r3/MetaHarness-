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

from metaharness.models import (
    GateStage,
    RunDisposition,
    RunEvent,
    RunEventKind,
    RunIdentity,
    RunMachineState,
    RunPhase,
    RunStatus,
    RunTransitionError,
    disposition_for_status,
    phase_successors,
    project_run_outcome,
    transition,
)
from metaharness.resume import (
    CHECKPOINT_INTEGRITY_OPERATION,
    CHECKPOINT_NAME,
    PHASE_STATUS,
    ResumeCheckpoint,
    ResumePhase,
    checkpoint_payload,
    machine_state_for_run,
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

            store.update_metadata(planner={"request_id": "p1"})
            store.set_run_state(
                RunMachineState(RunPhase.DETERMINISTIC_GATE),
                checks=[{"name": "test", "ok": True}],
            )
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
        store.set_run_state(RunMachineState(RunPhase.PLANNER))
        self.assertEqual(store.lock_path, self.root / "run" / "state.lock")
        self.assertTrue(store.lock_path.exists())
        self.assertNotIn("state.lock", ARTIFACT_ALLOWLIST)

    def test_lock_excludes_other_threads_until_released(self) -> None:
        store = self.store()
        probe = LockContentionProbe()
        finished = threading.Event()

        def writer() -> None:
            store.update_metadata(marker=True)
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

    def test_update_metadata_is_compare_and_set_on_the_canonical_identity(self) -> None:
        store = self.store()
        store.set_run_state(RunMachineState(RunPhase.PLAN_APPROVAL))
        gate = store.identity()
        written = store.update_metadata(expected=gate, execution={"owner": "web"})
        self.assertIsNotNone(written)
        self.assertEqual(written["status"], "awaiting_plan_approval")
        store.set_run_state(
            RunMachineState(RunPhase.PLANNER), execution={"owner": "orchestrator"},
        )
        self.assertIsNone(store.update_metadata(expected=gate, execution={"owner": "late web"}))
        final = store.load()
        self.assertEqual(final["status"], "planning")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        with self.assertRaises(ValueError):
            store.update_metadata(expected=store.identity(), status="failed")

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
            store.update_metadata(execution={"owner": "orchestrator"})
            return store.set_run_state(
                RunMachineState(RunPhase.WORKTREE_SETUP), branch="harness/x",
            )

        def web() -> dict | None:
            return store.update_metadata(
                expected=gate,
                plan_identity={"web": True},
                execution={"owner": "web"},
            )

        # The web holds the lock with the gate still open: its write is legal,
        # and the orchestrator's later writes are merged on top of it.
        store = self.store("web-first")
        store.set_run_state(RunMachineState(RunPhase.PLAN_APPROVAL))
        gate = store.identity()
        results = self.run_interleaved(store, ("web", web), ("orchestrator", orchestrator))
        self.assertIsNotNone(results["web"])
        final = store.load()
        self.assertEqual(final["status"], "preparing")
        self.assertEqual(final["branch"], "harness/x")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        self.assertEqual(final["plan_identity"], {"web": True})

        # The orchestrator holds the lock with a read of the open gate: the web
        # must then observe the closed gate and write nothing.
        store = self.store("orchestrator-first")
        store.set_run_state(RunMachineState(RunPhase.PLAN_APPROVAL))
        gate = store.identity()
        results = self.run_interleaved(store, ("orchestrator", orchestrator), ("web", web))
        self.assertIsNone(results["web"])
        final = store.load()
        self.assertEqual(final["status"], "preparing")
        self.assertEqual(final["branch"], "harness/x")
        self.assertEqual(final["execution"], {"owner": "orchestrator"})
        self.assertIsNone(final["plan_identity"])
        # Once the orchestrator left the gate, the gate never reappears.
        self.assertIsNone(web())
        self.assertEqual(store.load()["status"], "preparing")

    def test_concurrent_updates_never_lose_fields(self) -> None:
        store = self.store()

        def writer(thread_index: int) -> None:
            for item in range(20):
                store.update_metadata(**{f"k{thread_index}_{item}": item})

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
                    "stale": lambda: store.update_metadata(extra=order[0]),
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
                store.set_run_state(
                    RunMachineState(RunPhase.DETERMINISTIC_GATE), payload="x" * (item * 50),
                )

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
            state = store.update_metadata(run_options_sha256=digest)

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
        return store.set_run_state(
            RunMachineState(disposition=D.WAIT_EXTERNAL, reason="AGENT_TIMEOUT"),
            planning_protocol="v2",
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


# -- the one durable state machine -------------------------------------------

R = RunPhase
D = RunDisposition

# (phase, event, expected phase, expected disposition).  One row per legal
# hand-over of the pipeline; ``transition`` is the only thing that decides.
TRANSITION_MATRIX = (
    (R.CONTEXT, RunEvent.advance(R.PLANNER), R.PLANNER, D.RUNNING),
    (R.PLANNER, RunEvent.advance(R.PLAN_APPROVAL), R.PLAN_APPROVAL, D.RUNNING),
    (R.PLAN_APPROVAL, RunEvent.advance(R.WORKTREE_SETUP), R.WORKTREE_SETUP, D.RUNNING),
    (R.WORKTREE_SETUP, RunEvent.advance(R.IMPLEMENT_STEP), R.IMPLEMENT_STEP, D.RUNNING),
    # A cycle implements several steps, then accepts and gates the candidate.
    (R.IMPLEMENT_STEP, RunEvent.advance(R.IMPLEMENT_STEP), R.IMPLEMENT_STEP, D.RUNNING),
    (R.IMPLEMENT_STEP, RunEvent.advance(R.STEP_ACCEPTANCE), R.STEP_ACCEPTANCE, D.RUNNING),
    (R.STEP_ACCEPTANCE, RunEvent.advance(R.IMPLEMENT_STEP), R.IMPLEMENT_STEP, D.RUNNING),
    (R.IMPLEMENT_STEP, RunEvent.advance(R.DETERMINISTIC_GATE), R.DETERMINISTIC_GATE, D.RUNNING),
    # A red gate repairs, then re-validates, as often as its budget admits.
    (R.DETERMINISTIC_GATE, RunEvent.advance(R.CHECK_REPAIR), R.CHECK_REPAIR, D.RUNNING),
    (R.CHECK_REPAIR, RunEvent.advance(R.DETERMINISTIC_GATE), R.DETERMINISTIC_GATE, D.RUNNING),
    (R.DETERMINISTIC_GATE, RunEvent.advance(R.SEMANTIC_REVISION), R.SEMANTIC_REVISION, D.RUNNING),
    (R.SEMANTIC_REVISION, RunEvent.advance(R.DETERMINISTIC_GATE), R.DETERMINISTIC_GATE, D.RUNNING),
    (R.DETERMINISTIC_GATE, RunEvent.advance(R.CANDIDATE_READY), R.CANDIDATE_READY, D.RUNNING),
    (R.CANDIDATE_READY, RunEvent.advance(R.CANDIDATE_PUSH), R.CANDIDATE_PUSH, D.RUNNING),
    (R.CANDIDATE_PUSH, RunEvent.advance(R.FINAL_REVIEW), R.FINAL_REVIEW, D.RUNNING),
    # A reviewed candidate is published, or opens one bounded correction cycle.
    (R.FINAL_REVIEW, RunEvent.advance(R.PUBLISH), R.PUBLISH, D.RUNNING),
    (R.FINAL_REVIEW, RunEvent.advance(R.REVIEW_IMPLEMENTATION), R.REVIEW_IMPLEMENTATION, D.RUNNING),
    # A review that sends the candidate back to direct implementation opens a
    # correction cycle whose first operation is the revision itself.
    (R.FINAL_REVIEW, RunEvent.advance(R.SEMANTIC_REVISION), R.SEMANTIC_REVISION, D.RUNNING),
    (R.FINAL_REVIEW, RunEvent.advance(R.REVIEW_REPLAN), R.REVIEW_REPLAN, D.RUNNING),
    (R.REVIEW_IMPLEMENTATION, RunEvent.advance(R.REVIEW_IMPLEMENTATION), R.REVIEW_IMPLEMENTATION, D.RUNNING),
    (R.REVIEW_IMPLEMENTATION, RunEvent.advance(R.DETERMINISTIC_GATE), R.DETERMINISTIC_GATE, D.RUNNING),
    (R.REVIEW_REPLAN, RunEvent.advance(R.REVIEW_IMPLEMENTATION), R.REVIEW_IMPLEMENTATION, D.RUNNING),
    (R.WORKTREE_SETUP, RunEvent.advance(R.REVIEW_IMPLEMENTATION), R.REVIEW_IMPLEMENTATION, D.RUNNING),
    # A run stops without failing: the phase it stopped at is the operation to
    # retry, and the failure reason carries the business detail.
    (R.IMPLEMENT_STEP, RunEvent.wait(D.WAIT_EXTERNAL, reason="AGENT_TIMEOUT"), R.IMPLEMENT_STEP, D.WAIT_EXTERNAL),
    (R.DETERMINISTIC_GATE, RunEvent.wait(D.WAIT_EXTERNAL, reason="CHECK_TIMEOUT"), R.DETERMINISTIC_GATE, D.WAIT_EXTERNAL),
    (R.FINAL_REVIEW, RunEvent.wait(D.WAIT_HUMAN, reason="SPEC_DECISION_REQUIRED"), R.FINAL_REVIEW, D.WAIT_HUMAN),
    (R.DETERMINISTIC_GATE, RunEvent.wait(D.WAIT_HUMAN, reason="CHECK_REPAIR_EXHAUSTED"), R.DETERMINISTIC_GATE, D.WAIT_HUMAN),
    (R.IMPLEMENT_STEP, RunEvent.fail(reason="AGENT_SCOPE_VIOLATION"), R.IMPLEMENT_STEP, D.FAILED),
    (R.PUBLISH, RunEvent.fail(reason="PUSH_REJECTED"), R.PUBLISH, D.FAILED),
    (R.PUBLISH, RunEvent.complete(), R.PUBLISH, D.COMPLETED),
    (R.CANDIDATE_PUSH, RunEvent.complete(), R.CANDIDATE_PUSH, D.COMPLETED),
)

# A waiting run claims its exact phase again; only its posture changes.
RESUME_MATRIX = (
    (R.IMPLEMENT_STEP, D.WAIT_EXTERNAL, RunEvent.resume(), R.IMPLEMENT_STEP, D.RUNNING),
    (R.DETERMINISTIC_GATE, D.WAIT_HUMAN, RunEvent.resume(), R.DETERMINISTIC_GATE, D.RUNNING),
)

# Every illegal pair fails in ``transition`` and nowhere else.
INVALID_TRANSITIONS = (
    (RunMachineState(R.CONTEXT, D.RUNNING), RunEvent.advance(R.DETERMINISTIC_GATE), "never advances"),
    (RunMachineState(R.PUBLISH, D.RUNNING), RunEvent.advance(R.PUBLISH), "never advances"),
    (RunMachineState(R.IMPLEMENT_STEP, D.RUNNING), RunEvent.advance(None), "requires a target"),
    (RunMachineState(R.IMPLEMENT_STEP, D.WAIT_EXTERNAL), RunEvent.advance(R.STEP_ACCEPTANCE), "must be resumed"),
    (RunMachineState(R.IMPLEMENT_STEP, D.WAIT_HUMAN), RunEvent.complete(), "must be resumed"),
    (RunMachineState(R.IMPLEMENT_STEP, D.RUNNING), RunEvent.complete(), "only a reviewed candidate push"),
    (RunMachineState(R.IMPLEMENT_STEP, D.RUNNING), RunEvent.wait(D.RUNNING), "requires the WAIT_EXTERNAL or WAIT_HUMAN"),
    (RunMachineState(R.IMPLEMENT_STEP, D.RUNNING), RunEvent.resume(), "is not waiting"),
    (RunMachineState(R.PUBLISH, D.COMPLETED), RunEvent.fail(), "accepts no further event"),
    (RunMachineState(R.PUBLISH, D.COMPLETED), RunEvent.resume(), "accepts no further event"),
    (RunMachineState(R.IMPLEMENT_STEP, D.FAILED), RunEvent.advance(R.STEP_ACCEPTANCE), "accepts no further event"),
)

# (phase, disposition, reason, status, resumable, resume eligible)
PROJECTION_MATRIX = (
    (None, D.RUNNING, None, RunStatus.CREATED, False, False),
    (R.CONTEXT, D.RUNNING, None, RunStatus.PLANNING, False, False),
    (R.PLAN_APPROVAL, D.RUNNING, None, RunStatus.AWAITING_PLAN_APPROVAL, False, False),
    (R.IMPLEMENT_STEP, D.RUNNING, None, RunStatus.IMPLEMENTING, False, False),
    (R.DETERMINISTIC_GATE, D.RUNNING, None, RunStatus.VALIDATING, False, False),
    (R.SEMANTIC_REVISION, D.RUNNING, None, RunStatus.REVISING, False, False),
    (R.FINAL_REVIEW, D.RUNNING, None, RunStatus.REVIEWING, False, False),
    (R.PUBLISH, D.RUNNING, None, RunStatus.PUBLISHING, False, False),
    (R.CANDIDATE_PUSH, D.COMPLETED, None, RunStatus.COMMITTED, False, False),
    (R.PUBLISH, D.COMPLETED, None, RunStatus.PUBLISHED, False, False),
    (R.IMPLEMENT_STEP, D.FAILED, "AGENT_SCOPE_VIOLATION", RunStatus.FAILED, False, True),
    # The failure reason names the flavour of an external wait; a reason that
    # belongs to another phase stays a plain external wait.
    (R.IMPLEMENT_STEP, D.WAIT_EXTERNAL, "AGENT_TIMEOUT", RunStatus.WAITING_EXTERNAL, True, True),
    (R.DETERMINISTIC_GATE, D.WAIT_EXTERNAL, "CHECK_TIMEOUT", RunStatus.WAITING_CHECK_INFRASTRUCTURE, True, True),
    (R.FINAL_REVIEW, D.WAIT_EXTERNAL, "CHECK_TIMEOUT", RunStatus.WAITING_CHECK_INFRASTRUCTURE, True, True),
    (R.CANDIDATE_PUSH, D.WAIT_EXTERNAL, "PUSH_FAILED", RunStatus.WAITING_REMOTE, True, True),
    (R.PUBLISH, D.WAIT_EXTERNAL, "PUSH_FAILED", RunStatus.WAITING_REMOTE, True, True),
    (R.FINAL_REVIEW, D.WAIT_EXTERNAL, "PUSH_FAILED", RunStatus.WAITING_EXTERNAL, True, True),
    # Only the bounded repair slots a human may retry stay resumable.
    (R.IMPLEMENT_STEP, D.WAIT_HUMAN, "STEP_CONTRACT_REPAIR_OUTPUT_INVALID", RunStatus.WAITING_CONTRACT_REPAIR, True, True),
    (R.DETERMINISTIC_GATE, D.WAIT_HUMAN, "CHECK_REPAIR_EXHAUSTED", RunStatus.WAITING_CHECK_REPAIR, True, True),
    (R.FINAL_REVIEW, D.WAIT_HUMAN, "CHECK_REPAIR_EXHAUSTED", RunStatus.WAITING_HUMAN, False, False),
    # An operator gate owns its durable pending operation; a human wait with
    # no gate and no repair slot owns nothing to resume.
    (R.SEMANTIC_REVISION, D.WAIT_HUMAN, "WAITING_SCOPE_APPROVAL", RunStatus.WAITING_SCOPE_APPROVAL, True, True),
    (R.IMPLEMENT_STEP, D.WAIT_HUMAN, "WAITING_SCOPE_APPROVAL", RunStatus.WAITING_SCOPE_APPROVAL, True, True),
    (R.SEMANTIC_REVISION, D.WAIT_HUMAN, None, RunStatus.WAITING_HUMAN, False, False),
    (R.IMPLEMENT_STEP, D.WAIT_HUMAN, "SPEC_DECISION_REQUIRED", RunStatus.WAITING_HUMAN, False, False),
    (R.FINAL_REVIEW, D.WAIT_HUMAN, "SECURITY_POLICY_DECISION_REQUIRED", RunStatus.WAITING_HUMAN, False, False),
)


class RunMachineTests(unittest.TestCase):
    """The durable phase and disposition, and the one projection of them."""

    @staticmethod
    def checkpoint_at(phase: RunPhase) -> ResumeCheckpoint:
        """A minimal coherent checkpoint for *phase*."""

        phase_fields: dict = {}
        if phase in {RunPhase.DETERMINISTIC_GATE, RunPhase.CHECK_REPAIR}:
            phase_fields["stage"] = GateStage.POST_IMPLEMENTATION
            if phase is RunPhase.CHECK_REPAIR:
                phase_fields["check_repair_attempt"] = 1
        elif phase in {RunPhase.IMPLEMENT_STEP, RunPhase.STEP_ACCEPTANCE, RunPhase.REVIEW_IMPLEMENTATION}:
            phase_fields["step_id"] = "S01"
        return ResumeCheckpoint(
            phase=phase, **phase_fields,
            expected_head_sha="6" * 40, expected_tree_sha="7" * 40,
            execution_selection_sha256="8" * 64,
            plan_identity=plan_identity_from_mapping({
                "raw_sha256": "1" * 64, "contract_sha256": "2" * 64,
                "bundle_sha256": "3" * 64, "execution_sha256": "4" * 64,
                "checks_sha256": "5" * 64,
            }),
        )

    def test_the_transition_table_is_the_only_place_a_hand_over_is_decided(self) -> None:
        for phase, event, expected_phase, expected_disposition in TRANSITION_MATRIX:
            with self.subTest(phase=phase.value, event=event.kind.value):
                nxt = transition(RunMachineState(phase, D.RUNNING), event)
                self.assertIs(nxt.disposition, expected_disposition)
                if event.kind is RunEventKind.ADVANCE:
                    # The graph and the machine agree on every hand-over.
                    self.assertIn(expected_phase, phase_successors(phase))
                    self.assertIsNone(nxt.reason)
                else:
                    # Stopping never moves the phase: it names the operation
                    # that still has to run, and the reason is its detail.
                    self.assertEqual(expected_phase, phase)
                    self.assertEqual(nxt.reason, event.reason)
                self.assertEqual(nxt.phase, expected_phase)

    def test_waiting_durations_keep_the_exact_phase_they_stop_at(self) -> None:
        for phase, disposition, event, expected_phase, expected_disposition in RESUME_MATRIX:
            with self.subTest(phase=phase.value):
                waiting = transition(RunMachineState(phase, D.RUNNING), RunEvent.wait(disposition))
                self.assertEqual((waiting.phase, waiting.disposition), (phase, disposition))
                resumed = transition(waiting, event)
                self.assertEqual((resumed.phase, resumed.disposition), (expected_phase, expected_disposition))
                self.assertIsNone(resumed.reason)

    def test_every_invalid_transition_fails_in_the_same_place(self) -> None:
        for current, event, message in INVALID_TRANSITIONS:
            with self.subTest(phase=current.phase, disposition=current.disposition.value,
                              event=event.kind.value):
                with self.assertRaises(RunTransitionError) as caught:
                    transition(current, event)
                self.assertIn(message, str(caught.exception))
                # A refusal never leaks the other ValueError spelling.
                self.assertIsInstance(caught.exception, ValueError)

    def test_a_terminal_run_never_accepts_another_event(self) -> None:
        for event in (
            RunEvent.advance(R.PUBLISH), RunEvent.wait(D.WAIT_HUMAN),
            RunEvent.fail(), RunEvent.complete(), RunEvent.resume(),
        ):
            with (
                self.subTest(disposition=D.COMPLETED.value, event=event.kind.value),
                self.assertRaises(RunTransitionError),
            ):
                transition(RunMachineState(R.PUBLISH, D.COMPLETED), event)

    def test_only_a_failed_run_may_be_resumed_from_a_terminal_posture(self) -> None:
        # The resume gate has validated the durable retry boundary: RESUME is
        # the one event that re-claims a failure, and every other event stops.
        resumed = transition(
            RunMachineState(R.IMPLEMENT_STEP, D.FAILED, "LLM_FAILURE"), RunEvent.resume(),
        )
        self.assertEqual((resumed.phase, resumed.disposition), (R.IMPLEMENT_STEP, D.RUNNING))
        self.assertIsNone(resumed.reason)
        for event in (
            RunEvent.advance(R.STEP_ACCEPTANCE), RunEvent.wait(D.WAIT_HUMAN),
            RunEvent.fail(), RunEvent.complete(),
        ):
            with (
                self.subTest(event=event.kind.value),
                self.assertRaises(RunTransitionError),
            ):
                transition(RunMachineState(R.IMPLEMENT_STEP, D.FAILED), event)

    def test_the_projection_is_the_only_reader_of_a_phase_and_disposition(self) -> None:
        for phase, disposition, reason, status, resumable, eligible in PROJECTION_MATRIX:
            with self.subTest(phase=phase.value if phase else None, disposition=disposition.value):
                outcome = project_run_outcome(RunMachineState(phase, disposition, reason))
                self.assertIs(outcome.status, status)
                self.assertIs(outcome.resumable, resumable)
                self.assertIs(outcome.resume_eligible, eligible)

    def test_every_phase_has_one_running_projection(self) -> None:
        for phase in RunPhase:
            with self.subTest(phase=phase.value):
                outcome = project_run_outcome(RunMachineState(phase, D.RUNNING))
                self.assertEqual(outcome.status.value, PHASE_STATUS[phase])
                self.assertFalse(outcome.resume_eligible)
                # The status a resume claims is never a waiting or terminal one.
                self.assertNotIn(outcome.status.value, {"failed", "published", "committed"})
                self.assertFalse(outcome.status.value.startswith("waiting_"))

    def test_a_status_names_exactly_one_disposition(self) -> None:
        for status in RunStatus:
            with self.subTest(status=status.value):
                self.assertIsInstance(disposition_for_status(status), RunDisposition)
        for status, disposition in (
            ("waiting_external", D.WAIT_EXTERNAL),
            ("waiting_check_infrastructure", D.WAIT_EXTERNAL),
            ("waiting_remote", D.WAIT_EXTERNAL),
            ("waiting_human", D.WAIT_HUMAN),
            ("waiting_contract_repair", D.WAIT_HUMAN),
            ("waiting_check_repair", D.WAIT_HUMAN),
            ("failed", D.FAILED),
            ("interrupted", D.FAILED),
            ("committed", D.COMPLETED),
            ("published", D.COMPLETED),
            ("validating", D.RUNNING),
        ):
            with self.subTest(status=status):
                self.assertIs(disposition_for_status(status), disposition)
        with self.assertRaises(ValueError):
            disposition_for_status("not-a-status")

    def test_the_checkpoint_phase_is_the_authority(self) -> None:
        # The run state only ever carries the posture: whatever a status
        # string says, the checkpoint names the phase.
        state = {
            "status": "waiting_external",
            "disposition": "WAIT_EXTERNAL",
            "failure": {"reason": "CHECK_TIMEOUT"},
        }
        checkpoint = self.checkpoint_at(RunPhase.DETERMINISTIC_GATE)
        machine = machine_state_for_run(state, checkpoint)
        self.assertIs(machine.phase, RunPhase.DETERMINISTIC_GATE)
        self.assertIs(machine.disposition, D.WAIT_EXTERNAL)
        self.assertIs(project_run_outcome(machine).status, RunStatus.WAITING_CHECK_INFRASTRUCTURE)
        # Without a checkpoint there is no phase authority at all.
        self.assertIsNone(machine_state_for_run(state, None).phase)

    def test_the_failure_reason_and_not_the_stored_status_names_the_wait(self) -> None:
        # An incoherent pair (a repair-slot status with a decision reason) is
        # projected from its reason, so no implicit status/phase matrix is left.
        state = {
            "status": "waiting_contract_repair", "disposition": "WAIT_HUMAN",
            "failure": {"reason": "SPEC_DECISION_REQUIRED"},
        }
        machine = machine_state_for_run(state, self.checkpoint_at(RunPhase.IMPLEMENT_STEP))
        self.assertIs(project_run_outcome(machine).status, RunStatus.WAITING_HUMAN)
        self.assertFalse(project_run_outcome(machine).resumable)

    def test_a_run_state_written_before_the_vocabulary_is_read_through_the_bridge(self) -> None:
        for status, disposition in (
            ("waiting_check_repair", D.WAIT_HUMAN),
            ("waiting_remote", D.WAIT_EXTERNAL),
            ("failed", D.FAILED),
        ):
            with self.subTest(status=status):
                machine = machine_state_for_run({"status": status})
                self.assertIs(machine.disposition, disposition)
        with self.assertRaises(ValueError):
            machine_state_for_run({"status": "not-a-status"})


class RunStateProjectionTests(unittest.TestCase):
    """The store keeps the posture and derives the status from it."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RunStateStore(Path(self.temp.name) / "state.json")
        self.store.initialize("run")

    def test_initialize_seeds_the_canonical_posture(self) -> None:
        state = self.store.load()
        self.assertEqual(state["status"], RunStatus.CREATED.value)
        self.assertEqual(state["disposition"], RunDisposition.RUNNING.value)

    def test_set_run_state_derives_the_status_from_the_posture(self) -> None:
        state = self.store.set_run_state(
            RunMachineState(RunPhase.DETERMINISTIC_GATE, D.WAIT_HUMAN, "CHECK_REPAIR_EXHAUSTED"),
            current_step=None,
        )
        self.assertEqual(state["status"], RunStatus.WAITING_CHECK_REPAIR.value)
        self.assertEqual(state["disposition"], D.WAIT_HUMAN.value)
        self.assertEqual(state["phase"], RunPhase.DETERMINISTIC_GATE.value)
        self.assertEqual(state["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.store.load()["status"], RunStatus.WAITING_CHECK_REPAIR.value)

    def test_set_run_state_owns_the_status(self) -> None:
        with self.assertRaises(ValueError):
            self.store.set_run_state(RunMachineState(RunPhase.PUBLISH), status="published")
        with self.assertRaises(TypeError):
            self.store.set_run_state("publishing")

    def test_every_recorded_status_is_the_projection_of_its_machine_state(self) -> None:
        for phase, disposition, reason, status, _resumable, _eligible in PROJECTION_MATRIX:
            with self.subTest(status=status.value):
                state = self.store.set_run_state(RunMachineState(phase, disposition, reason))
                self.assertEqual(state["status"], status.value)
                self.assertEqual(state["disposition"], disposition.value)
                self.assertEqual(state["phase"], phase.value if phase else None)

    def test_a_recorded_failure_is_a_failed_posture(self) -> None:
        state = self.store.record_failure("AGENT_SCOPE_VIOLATION", "out of scope")
        self.assertEqual(state["disposition"], RunDisposition.FAILED.value)

    def test_a_metadata_update_never_moves_the_machine_state(self) -> None:
        before = self.store.set_run_state(
            RunMachineState(RunPhase.DETERMINISTIC_GATE, D.WAIT_HUMAN, "CHECK_REPAIR_EXHAUSTED"),
        )
        machine_before = self.store.machine_state()
        after = self.store.update_metadata(checks=[{"name": "test", "ok": True}])
        self.assertEqual(after["checks"], [{"name": "test", "ok": True}])
        for field in ("status", "disposition", "phase", "reason"):
            self.assertEqual(after[field], before[field], field)
        self.assertEqual(self.store.machine_state(), machine_before)
        self.assertEqual(
            self.store.outcome().status, RunStatus.WAITING_CHECK_REPAIR,
        )

    def test_no_caller_can_author_the_status_or_the_machine_state(self) -> None:
        machine = RunMachineState(RunPhase.IMPLEMENT_STEP, D.WAIT_EXTERNAL, "AGENT_TIMEOUT")
        for mutate in (
            lambda: self.store.update_metadata(status="committed"),
            lambda: self.store.update_metadata(disposition="COMPLETED"),
            lambda: self.store.update_metadata(phase=RunPhase.PUBLISH),
            lambda: self.store.update_metadata(reason="AGENT_TIMEOUT"),
            lambda: self.store.set_run_state(machine, status="committed"),
            lambda: self.store.record_failure("AGENT_TIMEOUT", disposition="COMPLETED"),
            lambda: self.store.transition_run(
                RunEvent.resume(), expected=self.store.identity(), status="implementing",
            ),
        ):
            with self.assertRaises(ValueError):
                mutate()
        self.assertEqual(self.store.load()["status"], RunStatus.CREATED.value)

    def test_a_recorded_status_is_re_derived_and_never_trusted(self) -> None:
        # A recorded status that contradicts the posture is not authority: the
        # next write re-derives it from the machine state alone.
        payload = self.store.load()
        payload.update(
            status="committed",
            disposition=RunDisposition.RUNNING.value,
            phase=RunPhase.IMPLEMENT_STEP.value,
        )
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        state = self.store.update_metadata(marker=True)
        self.assertEqual(state["status"], RunStatus.IMPLEMENTING.value)
        self.assertEqual(state["disposition"], RunDisposition.RUNNING.value)
        self.assertEqual(state["marker"], True)

    def test_the_checkpoint_phase_is_never_contradicted_by_a_machine_state(self) -> None:
        self.store.path.parent.joinpath(CHECKPOINT_NAME).write_text(
            json.dumps(
                checkpoint_payload(
                    ResumeCheckpoint(
                        phase=ResumePhase.DETERMINISTIC_GATE,
                        stage=GateStage.POST_IMPLEMENTATION,
                        execution_selection_sha256="8" * 64,
                        expected_head_sha="6" * 40, expected_tree_sha="7" * 40,
                        plan_identity=plan_identity_from_mapping({
                            "raw_sha256": "1" * 64, "contract_sha256": "2" * 64,
                            "bundle_sha256": "3" * 64, "execution_sha256": "4" * 64,
                            "checks_sha256": "5" * 64,
                        }),
                    )
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.store.set_run_state(RunMachineState(RunPhase.PUBLISH))
        self.assertEqual(self.store.machine_state().phase, RunPhase.DETERMINISTIC_GATE)

    def test_an_invalid_event_is_refused_by_the_transition_alone(self) -> None:
        self.store.set_run_state(RunMachineState(RunPhase.PUBLISH, D.RUNNING))
        before = self.store.load()
        with self.assertRaises(RunTransitionError):
            self.store.transition_run(RunEvent.resume(), expected=self.store.identity())
        self.assertEqual(self.store.load(), before)

    def test_a_wait_external_gate_reason_still_projects_the_ui_status(self) -> None:
        # The UI keeps reading a status: it is a projection of the machine, so a
        # gate reason keeps naming its operator-facing flavour.
        for reason, status in (
            ("CHECK_TIMEOUT", RunStatus.WAITING_CHECK_INFRASTRUCTURE),
            ("PUSH_FAILED", RunStatus.WAITING_REMOTE),
            ("AGENT_TIMEOUT", RunStatus.WAITING_EXTERNAL),
        ):
            with self.subTest(reason=reason):
                phase = (
                    RunPhase.CANDIDATE_PUSH if reason == "PUSH_FAILED"
                    else RunPhase.DETERMINISTIC_GATE
                )
                state = self.store.set_run_state(
                    RunMachineState(phase, D.WAIT_EXTERNAL, reason),
                )
                self.assertEqual(state["status"], status.value)
                self.assertEqual(state["disposition"], D.WAIT_EXTERNAL.value)

    def test_a_claimed_run_moves_its_posture_and_keeps_its_phase(self) -> None:
        waiting = self.store.set_run_state(
            RunMachineState(RunPhase.IMPLEMENT_STEP, D.WAIT_EXTERNAL, "AGENT_TIMEOUT"),
            failure={"reason": "AGENT_TIMEOUT"},
        )
        claimed = self.store.transition_run(
            RunEvent.resume(), expected=self.store.identity(), failure=None,
        )
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["status"], RunStatus.IMPLEMENTING.value)
        self.assertEqual(claimed["disposition"], RunDisposition.RUNNING.value)
        self.assertEqual(claimed["phase"], waiting["phase"])
        self.assertIsNone(claimed["reason"])


if __name__ == "__main__":
    unittest.main()
