from __future__ import annotations

import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
import json
from pathlib import Path
from unittest.mock import Mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.config import ConfigError, load_config
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
from metaharness.orchestrator import Orchestrator
from metaharness.run_options import RunOptions
from metaharness.state import RunStateStore
from metaharness.web.api import WebAPIError, create_run, validate_spec
from metaharness.web.run_manager import RunCapacityError, RunManager
from metaharness.web.server import create_server


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


class CoreP15Tests(unittest.TestCase):
    def test_run_file_delegates_to_run_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "SPEC.md"
            spec_path.write_text("exact\n", encoding="utf-8")
            orchestrator = Orchestrator.__new__(Orchestrator)
            expected = object()
            orchestrator.run_text = Mock(return_value=expected)  # type: ignore[method-assign]
            self.assertIs(orchestrator.run(spec_path, run_id="r"), expected)
            orchestrator.run_text.assert_called_once_with("exact\n", run_id="r")

    def test_run_text_persists_exact_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_for(root)
            config.runs_root.mkdir()
            orchestrator = Orchestrator(config)
            orchestrator._execute = Mock(return_value=object())  # type: ignore[method-assign]
            spec = "# SPEC\n\ntext with trailing spaces  \n"
            orchestrator.run_text(spec, run_id="exact")
            run_dir = config.runs_root / "exact"
            self.assertEqual((run_dir / "spec.md").read_text(encoding="utf-8"), spec)

    def test_on_created_runs_after_initial_state_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_for(root)
            config.runs_root.mkdir()
            orchestrator = Orchestrator(config)
            orchestrator._execute = Mock(return_value=object())  # type: ignore[method-assign]
            observed: list[tuple[Path, str, str]] = []

            def on_created(run_dir: Path) -> None:
                state = RunStateStore(run_dir / "state.json").load()
                observed.append((run_dir, (run_dir / "spec.md").read_text(), state["status"]))

            orchestrator.run_text("body", run_id="created", on_created=on_created)
            self.assertEqual(observed, [(config.runs_root / "created", "body", "created")])


class ManagerP15Tests(unittest.TestCase):
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


class APIP15Tests(unittest.TestCase):
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


class ConfigP15Tests(unittest.TestCase):
    def test_ui_capacity_is_bounded_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = (
                f'repo = "{root}"\nbase_ref = "HEAD"\nruns_root = "{root / "runs"}"\n'
                f'worktrees_root = "{root / "worktrees"}"\nallow_no_required_checks = true\n'
                '[ui]\ndefault_planner_profile = "planner"\ndefault_implementer_profile = "worker"\ndefault_reviewer_profile = "reviewer"\n'
                '[model_profiles.planner]\ndisplay_name = "Planner"\nroles = ["planner"]\ndriver = "openai-chat"\nprovider = "test"\nmodel = "p"\nselection_mode = "request"\nbase_url = "https://example.invalid"\nendpoint_path = "/chat"\n'
                '[model_profiles.worker]\ndisplay_name = "Worker"\nroles = ["implementer"]\ndriver = "external"\nprovider = "test"\nmodel = "worker"\nselection_mode = "cli"\nargv = ["worker"]\n'
                '[model_profiles.reviewer]\ndisplay_name = "Reviewer"\nroles = ["reviewer"]\ndriver = "openai-chat"\nprovider = "test"\nmodel = "r"\nselection_mode = "request"\nbase_url = "https://example.invalid"\nendpoint_path = "/chat"\n'
            )
            path = root / "config.toml"
            path.write_text(content, encoding="utf-8")
            self.assertEqual(load_config(path).ui.max_active_runs, 1)
            path.write_text(content.replace("[ui]\n", "[ui]\nmax_active_runs = 5\n", 1), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "at most 4"):
                load_config(path)


class HTTPSecurityP15Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = config_for(self.root)
        self.config.runs_root.mkdir()
        self.server = create_server(self.config, port=0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, str]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        connection.close()
        return response.status, content

    def test_new_page_and_index_token_rules(self) -> None:
        status, index = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn('href="/new">NEW RUN', index)
        self.assertNotIn(self.server.browser_token, index)
        status, page = self.request("GET", "/new")
        self.assertEqual(status, 200)
        self.assertIn("New Run", page)
        self.assertIn("CREATE RUN", page)
        self.assertIn('<form action="/runs" method="post"', page)
        self.assertNotIn("<script", page)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn(self.server.browser_token, page)
        self.assertNotIn("innerHTML", page)

    def test_post_api_runs_no_token_is_forbidden(self) -> None:
        body = json.dumps({"spec": "body"}).encode()
        status, _ = self.request(
            "POST",
            "/api/runs",
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
            body,
        )
        self.assertEqual(status, 403)

    def test_post_api_runs_bad_host_is_forbidden(self) -> None:
        body = json.dumps({"spec": "body"}).encode()
        status, _ = self.request(
            "POST",
            "/api/runs",
            {
                "Host": "evil.example",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-MetaHarness-Token": self.server.browser_token,
            },
            body,
        )
        self.assertEqual(status, 403)

    def test_post_api_runs_bad_origin_is_forbidden(self) -> None:
        body = json.dumps({"spec": "body"}).encode()
        status, _ = self.request(
            "POST",
            "/api/runs",
            {
                "Origin": "http://evil.example",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-MetaHarness-Token": self.server.browser_token,
            },
            body,
        )
        self.assertEqual(status, 403)

    def test_post_api_runs_rejects_unknown_fields(self) -> None:
        body = json.dumps({"spec": "body", "extra": True}).encode()
        status, _ = self.request(
            "POST",
            "/api/runs",
            {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-MetaHarness-Token": self.server.browser_token,
            },
            body,
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
