from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.approval import compute_plan_identity
from metaharness.models import (
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ExecutionRole,
    ProfileDriver,
    SelectionMode,
    UIConfig,
)
from metaharness.state import RunStateStore
from metaharness.web.server import create_server


def _config(root: Path, runs: Path) -> HarnessConfig:
    endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
    profiles = {
        "planner": ModelProfile("planner", "Planner", (ExecutionRole.PLANNER,), ProfileDriver.OPENAI_CHAT, "planner", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
        "implementer": ModelProfile("implementer", "Implementer", (ExecutionRole.IMPLEMENTER,), ProfileDriver.EXTERNAL, "worker", SelectionMode.CLI, argv=("true",)),
        "reviewer": ModelProfile("reviewer", "Reviewer", (ExecutionRole.REVIEWER,), ProfileDriver.OPENAI_CHAT, "reviewer", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
    }
    return HarnessConfig(
        repo=root,
        base_ref="HEAD",
        runs_root=runs,
        worktrees_root=root / "worktrees",
        require_clean_base=True,
        context=ContextConfig(),
        check_catalog=(),
        allow_no_required_checks=True,
        ui=UIConfig(default_planner_profile="planner", default_implementer_profile="implementer", default_reviewer_profile="reviewer"),
        model_profiles=profiles,
    )


class LocalServerHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runs = root / "runs"
        self.runs.mkdir()
        self.server = create_server(_config(root, self.runs), port=0)
        self.port = self.server.server_port
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, headers: dict[str, str] | None = None, body: object | None = None):
        connection = HTTPConnection("127.0.0.1", self.port)
        all_headers = dict(headers or {})
        encoded = None
        if body is not None:
            encoded = json.dumps(body).encode()
            all_headers["Content-Type"] = "application/json"
            all_headers["Content-Length"] = str(len(encoded))
        connection.request(method, path, body=encoded, headers=all_headers)
        response = connection.getresponse()
        content = response.read()
        headers = {name.lower(): value for name, value in response.getheaders()}
        result = (response.status, headers, content.decode("utf-8"))
        connection.close()
        return result

    def awaiting_run(self, run_id: str = "waiting") -> Path:
        run_dir = self.runs / run_id
        store = RunStateStore(run_dir / "state.json")
        store.initialize(run_id)
        raw, contract = "STATUS: READY\n", "# contract\n"
        (run_dir / "planner.raw.md").write_text(raw, encoding="utf-8")
        (run_dir / "implementation_contract.md").write_text(contract, encoding="utf-8")
        store.update(
            status="awaiting_plan_approval",
            planning_protocol="v2",
            plan_identity=compute_plan_identity(raw, contract).__dict__,
        )
        return run_dir

    def test_foreign_hosts_are_rejected_for_get(self) -> None:
        self.awaiting_run()
        for host in (
            "evil.example",
            f"evil.example:{self.port}",
            f"127.0.0.1.evil.example:{self.port}",
            f"localhost.evil.example:{self.port}",
            "arbitrary-host",
            f"127.0.0.1:{self.port + 1}",
            "127.0.0.1",
            f"localhost.:{self.port}",
            f"[::1]:{self.port}",
            f"x127.0.0.1:{self.port}",
            "",
        ):
            for path in ("/", "/runs/waiting", "/api/runs", "/api/runs/waiting", "/api/v1/health"):
                with self.subTest(host=host, path=path):
                    status, _headers, content = self.request("GET", path, {"Host": host})
                    self.assertEqual(status, 403)
                    self.assertNotIn(self.server.token, content)

    def test_missing_or_duplicated_host_is_rejected(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.putrequest("GET", "/api/runs", skip_host=True)
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 403)
        connection.close()
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.putrequest("GET", "/api/runs", skip_host=True)
        connection.putheader("Host", f"127.0.0.1:{self.port}")
        connection.putheader("Host", "evil.example")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 403)
        connection.close()

    def test_local_hosts_are_accepted(self) -> None:
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"LocalHost:{self.port}"):
            with self.subTest(host=host):
                status, _headers, _content = self.request("GET", "/api/runs", {"Host": host})
                self.assertEqual(status, 200)

    def test_mutation_rejects_foreign_host_even_with_token(self) -> None:
        run_dir = self.awaiting_run()
        status, _headers, _content = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {"Host": f"evil.example:{self.port}", "X-MetaHarness-Token": self.server.token},
            {"decision": "APPROVE"},
        )
        self.assertEqual(status, 403)
        self.assertFalse((run_dir / "plan_approval.json").exists())

    def test_mutation_origin_must_be_exact_local_origin(self) -> None:
        run_dir = self.awaiting_run()
        for origin in (
            "http://evil.example",
            f"http://evil.example:{self.port}",
            f"http://127.0.0.1.evil.example:{self.port}",
            f"https://127.0.0.1:{self.port}",
            f"http://127.0.0.1:{self.port + 1}",
            f"http://127.0.0.1:{self.port}/",
            "null",
        ):
            with self.subTest(origin=origin):
                status, _headers, _content = self.request(
                    "POST",
                    "/api/runs/waiting/approval",
                    {"Origin": origin, "X-MetaHarness-Token": self.server.token},
                    {"decision": "APPROVE"},
                )
                self.assertEqual(status, 403)
                self.assertFalse((run_dir / "plan_approval.json").exists())
        status, _headers, _content = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {
                "Host": f"localhost:{self.port}",
                "Origin": f"http://localhost:{self.port}",
                "X-MetaHarness-Token": self.server.token,
            },
            {"decision": "REJECT"},
        )
        self.assertEqual(status, 200)
        self.assertTrue((run_dir / "plan_approval.json").exists())

    def test_api_create_rejects_opaque_origin(self) -> None:
        status, _headers, content = self.request(
            "POST",
            "/api/runs",
            {
                "Host": f"127.0.0.1:{self.port}",
                "Origin": "null",
                "X-MetaHarness-Token": self.server.token,
            },
            {"spec": "do it", "run_id": "opaque-origin"},
        )
        self.assertEqual(status, 403)
        self.assertIn("origin not allowed", content)

    def test_api_approval_rejects_opaque_origin(self) -> None:
        run_dir = self.awaiting_run()
        status, _headers, content = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {
                "Host": f"127.0.0.1:{self.port}",
                "Origin": "null",
                "X-MetaHarness-Token": self.server.token,
            },
            {"decision": "REJECT"},
        )
        self.assertEqual(status, 403)
        self.assertIn("origin not allowed", content)
        self.assertFalse((run_dir / "plan_approval.json").exists())

    def test_api_foreign_origin_is_rejected(self) -> None:
        status, _headers, content = self.request(
            "POST",
            "/api/runs",
            {
                "Host": f"127.0.0.1:{self.port}",
                "Origin": "http://evil.example",
                "X-MetaHarness-Token": self.server.token,
            },
            {"spec": "do it", "run_id": "foreign-origin"},
        )
        self.assertEqual(status, 403)
        self.assertIn("origin not allowed", content)

    def test_mutation_without_origin_is_accepted_with_host_and_token(self) -> None:
        run_dir = self.awaiting_run()
        status, _headers, _content = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {"X-MetaHarness-Token": self.server.token},
            {"decision": "REJECT"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads((run_dir / "plan_approval.json").read_text())["decision"], "REJECT")

    def test_secret_server_token_not_in_get_index(self) -> None:
        self.awaiting_run()
        status, _headers, content = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("waiting", content)
        self.assertNotIn(self.server.token, content)
        self.assertNotIn("META_TOKEN", content)
        status, _headers, content = self.request("GET", "/api/runs")
        self.assertNotIn(self.server.token, content)

    def test_token_only_on_run_page_that_can_decide(self) -> None:
        self.awaiting_run()
        done = RunStateStore(self.runs / "done" / "state.json")
        done.initialize("done")
        done.update(status="committed", commit_sha="a" * 40)
        status, _headers, awaiting_page = self.request("GET", "/runs/waiting")
        self.assertEqual(status, 200)
        self.assertIn(self.server.token, awaiting_page)
        status, _headers, done_page = self.request("GET", "/runs/done")
        self.assertEqual(status, 200)
        self.assertNotIn(self.server.token, done_page)
        self.assertNotIn('name="_token"', done_page)
        self.assertNotIn("<script", done_page)
        for path in ("/api/runs/waiting", "/api/runs/done", "/api/runs/waiting/progress"):
            _status, _headers, content = self.request("GET", path)
            self.assertNotIn(self.server.token, content)

    def test_defense_headers_and_nonce_csp(self) -> None:
        self.awaiting_run()
        for path in ("/", "/runs/waiting"):
            with self.subTest(path=path):
                status, headers, content = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(headers["x-content-type-options"], "nosniff")
                self.assertEqual(headers["referrer-policy"], "no-referrer")
                self.assertEqual(headers["x-frame-options"], "DENY")
                csp = headers["content-security-policy"]
                for directive in (
                    "default-src 'none'",
                    "connect-src 'self'",
                    "object-src 'none'",
                    "base-uri 'none'",
                    "script-src 'none'",
                    "form-action 'self'",
                    "frame-ancestors 'none'",
                ):
                    self.assertIn(directive, csp)
                self.assertNotIn("unsafe-inline", csp)
                self.assertNotIn("http", csp)
                nonce = re.search(r"style-src 'nonce-([^']+)'", csp).group(1)
                self.assertIn(f'<style nonce="{nonce}">', content)
                self.assertNotIn("<script", content)
                self.assertNotIn("innerHTML", content)
        _status, first, _ = self.request("GET", "/")
        _status, second, _ = self.request("GET", "/")
        self.assertNotEqual(first["content-security-policy"], second["content-security-policy"])
        for path in ("/api/runs", "/api/runs/missing"):
            _status, headers, _content = self.request("GET", path)
            self.assertEqual(headers["x-frame-options"], "DENY")
            self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])

    def test_server_binds_loopback_only(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_health_identifies_service_without_exposing_tokens(self) -> None:
        control_file = Path(self.temp.name) / "private" / "control.token"
        control_server = create_server(
            _config(Path(self.temp.name), self.runs), port=0,
            control_token_file=control_file,
        )
        thread = threading.Thread(
            target=control_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            control = control_server.control_token
            self.assertIsNotNone(control)
            connection = HTTPConnection("127.0.0.1", control_server.server_port)
            connection.request("GET", "/api/v1/health")
            response = connection.getresponse()
            payload = response.read().decode("utf-8")
            connection.close()
            self.assertEqual(response.status, 200)
            data = json.loads(payload)
            self.assertEqual(data, {
                "service": "metaharness", "api_version": 1,
                "status": "ok", "control_api": True,
            })
            self.assertNotIn(control, payload)
            self.assertNotIn(control_server.browser_token, payload)
        finally:
            control_server.shutdown()
            control_server.server_close()
            thread.join(timeout=2)

    def test_control_token_file_is_private_and_reused(self) -> None:
        token_file = Path(self.temp.name) / "nested" / "control.token"
        config = _config(Path(self.temp.name), self.runs)
        first = create_server(config, port=0, control_token_file=token_file)
        try:
            token = first.control_token
            self.assertTrue(token)
            self.assertEqual(token_file.read_text(encoding="utf-8"), token)
            if os.name == "posix":
                self.assertEqual(token_file.stat().st_mode & 0o777, 0o600)
        finally:
            first.server_close()
        second = create_server(config, port=0, control_token_file=token_file)
        try:
            self.assertEqual(second.control_token, token)
            self.assertNotEqual(second.browser_token, token)
        finally:
            second.server_close()

        token_file.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            create_server(config, port=0, control_token_file=token_file)
        token_file.write_bytes(b"x" * 4097)
        with self.assertRaises(ValueError):
            create_server(config, port=0, control_token_file=token_file)

    def test_control_token_authenticates_json_mutation_only(self) -> None:
        run_dir = self.awaiting_run()
        token_file = Path(self.temp.name) / "control.token"
        control_server = create_server(
            _config(Path(self.temp.name), self.runs), port=0,
            control_token_file=token_file,
        )
        thread = threading.Thread(
            target=control_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            control = control_server.control_token
            assert control is not None
            def control_request(token: str | None):
                connection = HTTPConnection("127.0.0.1", control_server.server_port)
                headers = {"Content-Type": "application/json"}
                if token is not None:
                    headers["X-MetaHarness-Token"] = token
                connection.request(
                    "POST", "/api/runs/waiting/approval",
                    json.dumps({"decision": "REJECT"}), headers,
                )
                response = connection.getresponse()
                content = response.read()
                connection.close()
                return response.status, content

            self.assertEqual(control_request(control)[0], 200)
            self.assertEqual(control_request("invalid-token")[0], 403)
            self.assertEqual(control_request(None)[0], 403)
            self.assertTrue((run_dir / "plan_approval.json").exists())

            connection = HTTPConnection("127.0.0.1", control_server.server_port)
            body = "_token=" + control + "&decision=REJECT"
            connection.request(
                "POST", "/runs/waiting/approval", body,
                {"Content-Type": "application/x-www-form-urlencoded"},
            )
            response = connection.getresponse()
            response.read()
            connection.close()
            self.assertEqual(response.status, 403)
        finally:
            control_server.shutdown()
            control_server.server_close()
            thread.join(timeout=2)


class WebSecurityTests(unittest.TestCase):
    def test_untrusted_plan_review_and_paths_are_html_escaped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            run = runs / "xss"
            RunStateStore(run / "state.json").initialize(
                "xss", worktree="<img src=x onerror=alert(1)>"
            )
            (run / "planner.raw.md").write_text(
                "<script>window.PWNED=true</script>", encoding="utf-8"
            )
            review = run / "cycles" / "001" / "review"
            review.mkdir(parents=True)
            (review / "reviewer.raw.md").write_text(
                "<img src=x onerror=alert(1)>", encoding="utf-8"
            )
            (review / "review.json").write_text(
                '{"summary":"<script>window.PWNED=true</script>","findings":"<img src=x onerror=alert(1)>"}',
                encoding="utf-8",
            )
            config = _config(root, runs)
            server = create_server(config, port=0)
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port)
                connection.request("GET", "/runs/xss")
                response = connection.getresponse()
                content = response.read().decode("utf-8")
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertEqual(response.status, 200)
            self.assertIn("&lt;script&gt;window.PWNED=true&lt;/script&gt;", content)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", content)
            self.assertNotIn("<script>window.PWNED", content)
            self.assertNotIn('<img src=x onerror=alert(1)>', content)


if __name__ == "__main__":
    unittest.main()
