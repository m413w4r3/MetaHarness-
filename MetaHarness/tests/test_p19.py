from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.approval import compute_plan_identity
from metaharness.models import (
    ContextConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ProfileDriver,
    RunMachineState,
    RunPhase,
    RoutingConfig,
    SelectionMode,
    UIConfig,
)
from metaharness.state import RunStateStore
from metaharness.web.api import get_run
from metaharness.web.pages import refresh_seconds_for_run
from metaharness.web.server import create_server


class P19WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        runs = root / "runs"
        runs.mkdir()
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
        profiles = {
            "planner": ModelProfile(
                id="planner",
                display_name="Planner",
                roles=(ExecutionRole.PLANNER,),
                driver=ProfileDriver.OPENAI_CHAT,
                model="planner-model",
                selection_mode=SelectionMode.REQUEST,
                base_url=endpoint.base_url,
                endpoint_path=endpoint.endpoint_path,
            ),
            "implementer": ModelProfile(
                id="implementer",
                display_name="Implementer",
                roles=(ExecutionRole.IMPLEMENTER,),
                driver=ProfileDriver.EXTERNAL,
                model="worker",
                selection_mode=SelectionMode.REQUEST,
                provider="test",
                argv=("worker",),
            ),
            "reviewer": ModelProfile(
                id="reviewer",
                display_name="Reviewer",
                roles=(ExecutionRole.REVIEWER,),
                driver=ProfileDriver.OPENAI_CHAT,
                model="reviewer-model",
                selection_mode=SelectionMode.REQUEST,
                base_url=endpoint.base_url,
                endpoint_path=endpoint.endpoint_path,
            ),
        }
        config = HarnessConfig(
            repo=root,
            base_ref="HEAD",
            runs_root=runs,
            worktrees_root=root / "worktrees",
            require_clean_base=True,
            context=ContextConfig(),
            check_catalog=(),
            allow_no_required_checks=True,
            model_profiles=profiles,
            ui=UIConfig(
                default_planner_profile="planner",
                default_reviewer_profile="reviewer",
            ),
            routing=RoutingConfig(
                mechanical_profile="implementer",
                reasoning_profile="implementer",
                agentic_profile="implementer",
            ),
        )
        self.server = create_server(config, port=0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body: str | None = None, **headers: str):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        if body is not None:
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            headers["Content-Length"] = str(len(body.encode()))
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read()
        result = response.status, dict(response.getheaders()), content
        connection.close()
        return result

    def create_waiting(self, run_id: str = "waiting") -> Path:
        run = self.server.config.runs_root / run_id
        store = RunStateStore(run / "state.json")
        store.initialize(run_id)
        raw, contract = "STATUS: READY\n", "# contract\n"
        (run / "planner.raw.md").write_text(raw)
        (run / "implementation_contract.md").write_text(contract)
        store.set_run_state(
            RunMachineState(RunPhase.PLAN_APPROVAL),
            planning_protocol="v2",
            plan_identity=compute_plan_identity(raw, contract).__dict__,
        )
        return run

    def token_from_new(self) -> str:
        _status, _headers, body = self.request("GET", "/new")
        return re.search(rb'name="_token" value="([^"]+)"', body).group(1).decode()

    def test_pages_have_no_script_and_refresh_contract(self) -> None:
        status, headers, index = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"<script", index)
        self.assertIn(b'meta http-equiv="refresh" content="5"', index)
        self.assertNotIn(self.server.browser_token.encode(), index)
        self.assertIn("script-src 'none'", headers["Content-Security-Policy"])
        self.assertIn("form-action 'self'", headers["Content-Security-Policy"])

        status, _headers, new = self.request("GET", "/new")
        self.assertEqual(status, 200)
        self.assertNotIn(b"<script", new)
        self.assertIn(b'name="_token"', new)
        self.assertNotIn(b'http-equiv="refresh"', new)

        run = self.server.config.runs_root / "planning"
        RunStateStore(run / "state.json").initialize("planning")
        status, _headers, planning = self.request("GET", "/runs/planning")
        self.assertEqual(status, 200)
        # P29: the run page never reloads itself; it loads the static script.
        self.assertNotIn(b'http-equiv="refresh"', planning)
        self.assertIn(b"/static/run.js", planning)

    def test_html_create_rejects_bad_form_shape_and_redirects(self) -> None:
        token = self.token_from_new()
        with patch("metaharness.web.server.create_run", return_value={"location": "/runs/new-id"}) as create:
            body = urlencode({"_token": token, "spec": "do it", "run_id": "", "planner_profile": "planner"})
            status, headers, _content = self.request("POST", "/runs", body, Origin=f"http://127.0.0.1:{self.server.server_port}")
            self.assertEqual(status, 303)
            self.assertEqual(headers["Location"], "/runs/new-id")
            create.assert_called_once()

        duplicate = f"_token={token}&spec=do+it&run_id=&planner_profile=planner&spec=again"
        status, _headers, _content = self.request("POST", "/runs", duplicate)
        self.assertEqual(status, 400)
        unknown = urlencode({"_token": token, "spec": "do it", "run_id": "", "planner_profile": "planner", "extra": "x"})
        status, _headers, _content = self.request("POST", "/runs", unknown)
        self.assertEqual(status, 400)

    def test_html_create_accepts_opaque_origin_and_redirects(self) -> None:
        token = self.token_from_new()
        body = urlencode({"_token": token, "spec": "do it", "run_id": "", "planner_profile": ""})
        with patch("metaharness.web.server.create_run", return_value={"location": "/runs/new-id"}) as create:
            status, headers, _content = self.request(
                "POST",
                "/runs",
                body,
                Host=f"127.0.0.1:{self.server.server_port}",
                Origin="null",
            )
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], "/runs/new-id")
        create.assert_called_once()

    def test_html_form_opaque_origin_still_requires_mutation_token(self) -> None:
        self.create_waiting()
        for token in (None, "wrong"):
            with self.subTest(token=token):
                fields = {"decision": "REJECT"}
                if token is not None:
                    fields["_token"] = token
                status, _headers, content = self.request(
                    "POST",
                    "/runs/waiting/approval",
                    urlencode(fields),
                    Host=f"127.0.0.1:{self.server.server_port}",
                    Origin="null",
                )
                self.assertEqual(status, 403)
                self.assertIn(b"mutation token required", content)

    def test_html_approval_uses_form_token_origin_and_redirects(self) -> None:
        run = self.create_waiting()
        token = self.token_from_new()
        form = urlencode({"_token": "wrong", "decision": "REJECT"})
        status, _headers, _content = self.request("POST", "/runs/waiting/approval", form)
        self.assertEqual(status, 403)
        form = urlencode({"_token": token, "decision": "REJECT"})
        status, headers, _content = self.request(
            "POST", "/runs/waiting/approval", form,
            Origin=f"http://evil.example:{self.server.server_port}",
        )
        self.assertEqual(status, 403)
        status, headers, _content = self.request(
            "POST", "/runs/waiting/approval", form,
            Origin=f"http://127.0.0.1:{self.server.server_port}",
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], "/runs/waiting")
        self.assertEqual(json.loads((run / "plan_approval.json").read_text())["decision"], "REJECT")

    def test_html_approval_accepts_opaque_origin_and_redirects(self) -> None:
        run = self.create_waiting()
        token = self.token_from_new()
        form = urlencode({"_token": token, "decision": "REJECT"})
        status, headers, _content = self.request(
            "POST",
            "/runs/waiting/approval",
            form,
            Host=f"127.0.0.1:{self.server.server_port}",
            Origin="null",
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["Location"], "/runs/waiting")
        self.assertEqual(json.loads((run / "plan_approval.json").read_text())["decision"], "REJECT")

    def test_html_create_rejects_foreign_origin(self) -> None:
        token = self.token_from_new()
        body = urlencode({"_token": token, "spec": "do it", "run_id": "", "planner_profile": ""})
        status, _headers, content = self.request(
            "POST",
            "/runs",
            body,
            Host=f"127.0.0.1:{self.server.server_port}",
            Origin="http://evil.example",
        )
        self.assertEqual(status, 403)
        self.assertIn(b"origin not allowed", content)

    def test_bounded_diagnostics_diff_and_progress(self) -> None:
        run = self.create_waiting("artifacts")
        store = RunStateStore(run / "state.json")
        store.record_failure("AGENT_RUNTIME_FAILED")
        step = run / "cycles/001/implementation/steps/S01"
        step.mkdir(parents=True)
        (step / "agent.final.md").write_text("x" * (32 * 1024 + 100))
        (step / "agent.stderr.log").write_text("e" * (32 * 1024 + 100))
        (run / "implementation_bundle.json").write_text(json.dumps({
            "schema_version": 1, "steps": [{"id": "S01", "title": "one"}],
        }))
        gate = run / "cycles/001/checks/post-implementation"
        gate.mkdir(parents=True)
        (gate / "diff.patch").write_text("d" * (64 * 1024 + 100))
        (gate / "changed-files.txt").write_text("src/app.py\n")
        payload = get_run(self.server.config.runs_root, "artifacts")
        step_payload = payload["cycle_artifacts"][0]["steps"][0]
        self.assertLessEqual(len(step_payload["final"].encode()), 8 * 1024)
        self.assertLessEqual(len(step_payload["stderr"].encode()), 8 * 1024)
        self.assertLessEqual(len(payload["candidate"]["diff_tail"].encode()), 64 * 1024)
        self.assertEqual(payload["candidate"]["changed_files"], ["src/app.py"])
        self.assertEqual(payload["approval"], {"recorded": False, "decision": None})

    def test_refresh_rules(self) -> None:
        for status in ("created", "planning", "preparing", "implementing", "validating", "reviewing"):
            self.assertEqual(refresh_seconds_for_run({"status": status}), 2)
        self.assertEqual(refresh_seconds_for_run({"status": "approved"}), 1)
        self.assertIsNone(refresh_seconds_for_run({"status": "awaiting_plan_approval", "approval": {"recorded": False}}))
        self.assertEqual(refresh_seconds_for_run({"status": "awaiting_plan_approval", "approval": {"recorded": True}}), 1)
        self.assertIsNone(refresh_seconds_for_run({"status": "committed"}))


if __name__ == "__main__":
    unittest.main()
