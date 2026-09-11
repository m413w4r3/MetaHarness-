from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.approval import compute_plan_identity
from metaharness.models import (
    AgentConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.state import RunStateStore
from metaharness.web.server import create_server


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runs = root / "runs"
        self.runs.mkdir()
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
        self.config = HarnessConfig(
            repo=root,
            base_ref="HEAD",
            runs_root=self.runs,
            worktrees_root=root / "worktrees",
            require_clean_base=True,
            planner=endpoint,
            reviewer=endpoint,
            context=ContextConfig(),
            agent=AgentConfig(),
            checks=(),
            allow_no_required_checks=True,
        )
        self.server = create_server(self.config, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body: object | None = None, token: str | None = None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Accept": "application/json"}
        if token is not None:
            headers["X-MetaHarness-Token"] = token
        encoded = None
        if body is not None:
            encoded = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(encoded))
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        content = response.read()
        connection.close()
        return response.status, json.loads(content) if content else None, content

    def create_run(self, run_id: str, status: str = "created") -> Path:
        run_dir = self.runs / run_id
        store = RunStateStore(run_dir / "state.json")
        store.initialize(run_id)
        if status != "created":
            store.update(status=status)
        return run_dir

    def test_list_and_run_and_missing_artifacts(self) -> None:
        run_dir = self.create_run("run-1")
        status, payload, _ = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(payload["runs"][0]["run_id"], "run-1")
        status, payload, _ = self.request("GET", "/api/runs/run-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"]["run_id"], "run-1")
        self.assertIsNone(payload["plan"]["raw"])
        self.assertIsNone(payload["checks"])
        self.assertFalse((run_dir / "plan_approval.json").exists())

    def test_unknown_and_traversal_runs_are_rejected(self) -> None:
        for path in (
            "/api/runs/unknown",
            "/api/runs/../x",
            "/api/runs/a%2Fb",
            "/api/runs/%2e%2e",
        ):
            with self.subTest(path=path):
                status, _payload, _ = self.request("GET", path)
                self.assertIn(status, (400, 404))

    def test_approval_requires_token_and_is_exclusive(self) -> None:
        run_dir = self.create_run("waiting", "awaiting_plan_approval")
        raw = "STATUS: READY\n"
        contract = "# contract\n"
        (run_dir / "planner.raw.md").write_text(raw, encoding="utf-8")
        (run_dir / "implementation_contract.md").write_text(contract, encoding="utf-8")
        identity = compute_plan_identity(raw, contract)
        RunStateStore(run_dir / "state.json").update(
            status="awaiting_plan_approval", plan_identity=identity.__dict__
        )
        for token in (None, "wrong"):
            status, _payload, _ = self.request(
                "POST", "/api/runs/waiting/approval", {"decision": "APPROVE"}, token
            )
            self.assertEqual(status, 403)
        status, payload, _ = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {"decision": "APPROVE"},
            self.server.token,
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["decision"], "APPROVE")
        approval = json.loads((run_dir / "plan_approval.json").read_text())
        self.assertEqual(approval["source"], "web-ui")
        status, _payload, _ = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {"decision": "REJECT"},
            self.server.token,
        )
        self.assertEqual(status, 409)

    def test_wrong_state_is_conflict(self) -> None:
        self.create_run("done", "committed")
        status, _payload, _ = self.request(
            "POST", "/api/runs/done/approval", {"decision": "APPROVE"}, self.server.token
        )
        self.assertEqual(status, 409)

    def test_progress_ignores_malformed_lines_and_uses_byte_offsets(self) -> None:
        run_dir = self.create_run("events")
        events = [
            "not json\n",
            json.dumps({"type": "turn.started"}) + "\n",
            json.dumps({"item": {"command": "python -m unittest"}}) + "\n",
        ]
        (run_dir / "agent.events.jsonl").write_text("".join(events), encoding="utf-8")
        status, payload, _ = self.request("GET", "/api/runs/events/progress?offset=0")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], ["turn.started", "$ python -m unittest"])
        status, second, _ = self.request(
            "GET", f"/api/runs/events/progress?offset={payload['next_offset']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(second["events"], [])

    def test_progress_is_capped_to_complete_lines(self) -> None:
        run_dir = self.create_run("large-events")
        line = json.dumps({"type": "turn.completed"}) + "\n"
        (run_dir / "agent.events.jsonl").write_text(line * 40_000, encoding="utf-8")
        status, payload, _ = self.request("GET", "/api/runs/large-events/progress")
        self.assertEqual(status, 200)
        self.assertLessEqual(payload["next_offset"], 256 * 1024)
        self.assertGreater(len(payload["events"]), 0)


if __name__ == "__main__":
    unittest.main()
