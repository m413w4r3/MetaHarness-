from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch

from metaharness.approval import compute_plan_identity
from metaharness.models import (
    AgentConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.state import RunStateStore
from metaharness.web import api
from metaharness.web.pages import RUN_PAGE_DYNAMIC_IDS, STATE_POLL_MS
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

    def post_form(self, path: str, fields: dict[str, str]) -> int:
        from urllib.parse import urlencode

        body = urlencode(fields).encode()
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", path, body=body, headers={
            "Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body)),
        })
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status

    def test_approval_fields_follow_the_actual_12_step_bundle(self) -> None:
        run_dir = self.create_run("form-steps", "committed")
        (run_dir / "implementation_bundle.json").write_text(
            json.dumps({"steps": [{"id": f"S{number:02d}"} for number in range(1, 13)]}),
            encoding="utf-8",
        )
        fields = {"_token": self.server.token, "decision": "APPROVE", "reviewer_profile": "r"}
        twelve = {f"step_profile__S{number:02d}": "luna" for number in range(1, 13)}
        # The fields for the actual bundle are accepted; the gate then refuses
        # this deliberately incomplete synthetic run.
        self.assertEqual(self.post_form("/runs/form-steps/approval", {**fields, **twelve}), 409)
        self.assertEqual(
            self.post_form("/runs/form-steps/approval", {**fields, **twelve, "step_profile__S13": "luna"}), 400
        )

        page_dir = self.create_run("form-page", "awaiting_plan_approval")
        (page_dir / "implementation_bundle.json").write_text(
            json.dumps({"steps": [{"id": f"S{number:02d}"} for number in range(1, 13)]}),
            encoding="utf-8",
        )
        RunStateStore(page_dir / "state.json").update(
            status="awaiting_plan_approval",
            planning_protocol="v2",
            planner={
                "execution_mode": "STAGED",
                "steps": [{"id": f"S{number:02d}", "title": f"Step {number}"} for number in range(1, 13)],
            },
        )
        page = self.get_html("/runs/form-page")
        self.assertEqual(page.count('name="step_profile__'), 12)
        self.assertNotIn('name="step_profile__S13"', page)

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

    def get_html(self, path: str) -> str:
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", path)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        connection.close()
        self.assertEqual(response.status, 200)
        return content

    def assert_poll_shape(self, payload: dict) -> None:
        # Fields read by the run-page polling script.
        for key in (
            "run_id",
            "status",
            "updated_at",
            "state",
            "plan",
            "checks",
            "review",
            "reviewer_raw",
            "reviewer_raw_available",
            "failure",
            "commit_sha",
        ):
            self.assertIn(key, payload)
        for key in ("status", "base_sha", "branch", "worktree", "commit_sha", "failure"):
            self.assertIn(key, payload["state"])
        self.assertIn("raw", payload["plan"])
        self.assertIn("contract", payload["plan"])
        self.assertEqual(payload["status"], payload["state"]["status"])

    def test_run_page_observes_transitions_without_manual_reload(self) -> None:
        run_dir = self.runs / "live"
        store = RunStateStore(run_dir / "state.json")
        store.initialize("live", base_sha="b" * 40, branch="metaharness/live", worktree="/tmp/wt live")
        store.update(status="implementing")

        page = self.get_html("/runs/live")
        self.assertIn("Run <span class=\"mono\">live</span>", page)
        # P29: no full-page refresh; targeted polling through the static
        # same-origin script only (no inline script).
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn('<script src="/static/run.js" defer></script>', page)
        self.assertNotIn("<script>", page)

        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assertEqual(status, 200)
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "implementing")
        self.assertEqual(payload["state"]["worktree"], "/tmp/wt live")

        checks = [{"name": "unit", "exit_code": 0, "timed_out": False, "stdout_tail": "<b>ok</b>"}]
        (run_dir / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
        store.update(status="validating")
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "validating")
        self.assertEqual(payload["checks"], checks)
        self.assertFalse(payload["reviewer_raw_available"])

        (run_dir / "review.json").write_text(
            json.dumps({"verdict": "PASS", "route": "NONE", "summary": "ok"}), encoding="utf-8"
        )
        (run_dir / "reviewer.raw.md").write_text("VERDICT: PASS\n", encoding="utf-8")
        store.update(status="reviewing")
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "reviewing")
        self.assertEqual(payload["review"]["verdict"], "PASS")
        self.assertTrue(payload["reviewer_raw_available"])
        self.assertEqual(payload["reviewer_raw"], "VERDICT: PASS\n")

        store.update(status="committed", commit_sha="c" * 40)
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "committed")
        self.assertEqual(payload["state"]["commit_sha"], "c" * 40)
        self.assertIsNone(payload["failure"])

        failed = self.create_run("broken", "implementing")
        RunStateStore(failed / "state.json").record_failure("CHECK_FAILED", "unit")
        status, payload, _ = self.request("GET", "/api/runs/broken")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["state"]["failure"], {"reason": "CHECK_FAILED", "detail": "unit"})

    def test_approval_controls_follow_status(self) -> None:
        self.create_run("before", "planning")
        page = self.get_html("/runs/before")
        self.assertNotIn('action="/runs/before/approval"', page)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn("/static/run.js", page)
        self.assertNotIn(self.server.token, page)

        self.create_run("gate", "awaiting_plan_approval")
        page = self.get_html("/runs/gate")
        self.assertIn('action="/runs/gate/approval"', page)
        self.assertIn('>APPROVE</button>', page)
        self.assertIn('>REJECT</button>', page)
        self.assertNotIn('http-equiv="refresh"', page)


class ProgressOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runs = Path(self.temp.name)
        (self.runs / "p").mkdir()
        self.path = self.runs / "p" / "agent.events.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, data: bytes, mode: str = "wb") -> None:
        with self.path.open(mode) as stream:
            stream.write(data)

    def poll(self, offset: int) -> dict:
        return api.progress(self.runs, "p", offset)

    def drain(self, offset: int = 0, max_requests: int = 200) -> tuple[int, list[str]]:
        """Poll like the browser until the offset stops, checking progress."""

        events: list[str] = []
        for _ in range(max_requests):
            payload = self.poll(offset)
            events.extend(payload["events"])
            if payload["next_offset"] == offset:
                return offset, events
            self.assertGreater(payload["next_offset"], offset)
            offset = payload["next_offset"]
        self.fail("progress offset did not settle")

    @staticmethod
    def event(kind: str) -> bytes:
        return json.dumps({"type": kind}).encode() + b"\n"

    def test_normal_lines_advance_normally(self) -> None:
        data = self.event("one") + self.event("two") + self.event("three")
        self.write(data)
        payload = self.poll(0)
        self.assertEqual(payload, {"next_offset": len(data), "events": ["one", "two", "three"]})
        self.assertEqual(self.poll(len(data)), {"next_offset": len(data), "events": []})

    def test_partial_final_line_waits_correctly(self) -> None:
        first = self.event("first")
        self.write(first + b'{"type": "sec')
        payload = self.poll(0)
        self.assertEqual(payload, {"next_offset": len(first), "events": ["first"]})
        self.assertEqual(self.poll(len(first)), {"next_offset": len(first), "events": []})
        self.write(b'ond"}\n', "ab")
        self.assertEqual(self.poll(len(first))["events"], ["second"])

    def test_line_longer_than_window_eventually_advances(self) -> None:
        big = json.dumps({"type": "big", "pad": "x" * (300 * 1024)}).encode() + b"\n"
        data = self.event("before") + big + b"not json\n" + self.event("after")
        self.write(data)
        offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, ["before", "big", "after"])

    def test_oversized_line_is_omitted_and_never_repeats_offset(self) -> None:
        oversized = b'{"type": "huge", "pad": "' + b"y" * (api.PROGRESS_MAX_EVENT_BYTES + 4096) + b'"}\n'
        data = self.event("before") + oversized + b"{malformed\n" + self.event("after")
        self.write(data)
        offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, ["before", api.OVERSIZED_EVENT, "after"])

    def test_oversized_line_is_skipped_across_bounded_requests(self) -> None:
        with patch.multiple(
            api,
            PROGRESS_MAX_BYTES=64,
            PROGRESS_MAX_EVENT_BYTES=128,
            PROGRESS_MAX_SKIP_BYTES=100,
            _PROGRESS_SCAN_CHUNK_BYTES=16,
        ):
            data = b"z" * 1000 + b"\n" + b"[1, 2]\n" + b"not json\n" + self.event("after")
            self.write(data)
            offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, [api.OVERSIZED_EVENT, "after"])

    def test_oversized_line_still_being_written_advances(self) -> None:
        with patch.multiple(
            api,
            PROGRESS_MAX_BYTES=64,
            PROGRESS_MAX_EVENT_BYTES=128,
            PROGRESS_MAX_SKIP_BYTES=100,
            _PROGRESS_SCAN_CHUNK_BYTES=16,
        ):
            self.write(b"q" * 500)
            first = self.poll(0)
            self.assertGreater(first["next_offset"], 0)
            self.assertEqual(first["events"], [api.OVERSIZED_EVENT])
            offset, events = self.drain(first["next_offset"])
            self.assertEqual(offset, 500)
            self.assertEqual(events, [])
            self.write(b"q" * 50 + b"\n" + self.event("after"), "ab")
            offset, events = self.drain(offset)
            self.assertEqual(offset, self.path.stat().st_size)
            self.assertEqual(events, ["after"])


if __name__ == "__main__":
    unittest.main()
