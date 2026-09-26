"""The web run manager and the run-creation API contracts."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import (
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ExecutionRole,
    ProfileDriver,
    RoutingConfig,
    SelectionMode,
    UIConfig,
)
from metaharness.run_options import RunOptions
from metaharness.state import RunStateStore
from metaharness.web.api import WebAPIError, create_run, validate_spec
from metaharness.web.run_manager import RunCapacityError, RunManager


def config_for(root: Path, *, max_active_runs: int = 1) -> HarnessConfig:
    endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
    profiles = {
        "planner": ModelProfile("planner", "Planner", (ExecutionRole.PLANNER,), ProfileDriver.OPENAI_CHAT, "planner", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
        "implementer": ModelProfile("implementer", "Implementer", (ExecutionRole.IMPLEMENTER,), ProfileDriver.EXTERNAL, "worker", SelectionMode.CLI, argv=("true",)),
        "reviewer": ModelProfile("reviewer", "Reviewer", (ExecutionRole.REVIEWER,), ProfileDriver.OPENAI_CHAT, "reviewer", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
    }
    return HarnessConfig(
        repo=root,
        base_ref="HEAD",
        runs_root=root / "runs",
        worktrees_root=root / "worktrees",
        require_clean_base=True,
        context=ContextConfig(),
        check_catalog=(),
        allow_no_required_checks=True,
        ui=UIConfig(
            max_active_runs=max_active_runs,
            default_planner_profile="planner",
            default_reviewer_profile="reviewer",
        ),
        model_profiles=profiles,
        routing=RoutingConfig(
            mechanical_profile="implementer",
            reasoning_profile="implementer",
            agentic_profile="implementer",
        ),
    )


class RunManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = config_for(self.root)
        self.config.runs_root.mkdir()
        root = self.root
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = False

        owner = self

        class FakeOrchestrator:
            def __init__(self, _config: HarnessConfig) -> None:
                pass

            def run_text(self, _spec: str, *, run_id: str, on_created, run_options=None) -> None:
                owner.started.set()
                on_created(owner.root / "runs" / run_id)
                if owner.fail:
                    raise RuntimeError("worker failure")
                owner.release.wait(timeout=2)

        self.factory = FakeOrchestrator

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_start_run_returns_after_created_callback(self) -> None:
        manager = RunManager(self.config, orchestrator_factory=self.factory)
        self.assertEqual(manager.start_run("spec", run_id="one"), "one")
        self.assertTrue(self.started.is_set())
        self.release.set()

    def test_active_capacity_blocks_second_run(self) -> None:
        manager = RunManager(self.config, orchestrator_factory=self.factory)
        manager.start_run("one", run_id="one")
        with self.assertRaises(RunCapacityError):
            manager.start_run("two", run_id="two")
        self.release.set()

    def test_worker_completion_releases_capacity(self) -> None:
        manager = RunManager(self.config, orchestrator_factory=self.factory)
        manager.start_run("one", run_id="one")
        self.release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager._active_run_ids:
            time.sleep(0.01)
        self.assertEqual(manager._active_run_ids, set())
        self.started.clear()
        self.release.clear()
        manager.start_run("two", run_id="two")
        self.release.set()

    def test_worker_failure_releases_capacity(self) -> None:
        self.fail = True
        manager = RunManager(self.config, orchestrator_factory=self.factory)
        manager.start_run("one", run_id="one")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and manager._active_run_ids:
            time.sleep(0.01)
        self.assertEqual(manager._active_run_ids, set())

    def test_run_manager_does_not_store_business_state(self) -> None:
        manager = RunManager(self.config, orchestrator_factory=self.factory)
        self.assertEqual(
            set(vars(manager)),
            {"_config", "_max_active_runs", "_orchestrator_factory", "_lock", "_active_run_ids"},
        )


class CreateRunAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = config_for(self.root)
        self.config.runs_root.mkdir()
        root = self.root

        self.received_run_options: list[object] = []
        received = self.received_run_options

        def factory(_config: HarnessConfig):
            class FakeOrchestrator:
                def run_text(self, spec: str, *, run_id: str, on_created, run_options=None) -> None:
                    received.append(run_options)
                    run_dir = root / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "spec.md").write_text(spec, encoding="utf-8")
                    RunStateStore(run_dir / "state.json").initialize(run_id)
                    on_created(run_dir)

            return FakeOrchestrator()

        self.manager = RunManager(self.config, orchestrator_factory=factory)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_create_run_valid(self) -> None:
        payload = create_run(self.manager, spec="original\n", run_id="api-one")
        self.assertEqual(payload["ok"], True)
        # The frozen UI run options always reach the orchestrator.
        self.assertEqual(len(self.received_run_options), 1)
        self.assertIsInstance(self.received_run_options[0], RunOptions)
        self.assertEqual(payload["run_id"], "api-one")
        self.assertEqual(payload["location"], "/runs/api-one")

    def test_create_run_empty_spec(self) -> None:
        with self.assertRaisesRegex(WebAPIError, "must not be empty"):
            create_run(self.manager, spec=" \n")

    def test_create_run_oversized_spec(self) -> None:
        with self.assertRaisesRegex(WebAPIError, "too large"):
            create_run(self.manager, spec="x" * (48 * 1024 + 1))

    def test_create_run_invalid_run_id(self) -> None:
        with self.assertRaisesRegex(WebAPIError, "invalid run id"):
            create_run(self.manager, spec="ok", run_id="../bad")

    def test_create_run_collision(self) -> None:
        (self.config.runs_root / "taken").mkdir()
        with self.assertRaisesRegex(WebAPIError, "run already exists"):
            create_run(self.manager, spec="ok", run_id="taken")

    def test_create_run_capacity_full(self) -> None:
        blocker = threading.Event()
        root = self.root

        class Blocking:
            def run_text(self, _spec: str, *, run_id: str, on_created, run_options=None) -> None:
                on_created(root / "runs" / run_id)
                blocker.wait(timeout=2)

        manager = RunManager(
            self.config,
            orchestrator_factory=lambda _config: Blocking(),
        )
        manager.start_run("one", run_id="one")
        with self.assertRaisesRegex(WebAPIError, "maximum active runs reached"):
            create_run(manager, spec="two", run_id="two")
        blocker.set()

    def test_validate_spec_preserves_original_text(self) -> None:
        value = "  keep whitespace  \n"
        self.assertEqual(validate_spec(value), value)


if __name__ == "__main__":
    unittest.main()
