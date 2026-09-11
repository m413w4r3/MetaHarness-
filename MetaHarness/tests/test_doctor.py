"""``metaharness doctor``: local gate, sandbox probe and bridge health."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import cli  # noqa: E402
from metaharness.cli import _usable_secret_value, main  # noqa: E402

SECRET = "doctor-secret-value-123"


class FakeBridge:
    def __init__(self, payload: Any = None, status: int = 200):
        owner = self
        self.payload = {"status": "ok"} if payload is None else payload
        self.status = status
        self.requests: list[tuple[str, str, str | None]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                owner.requests.append(("GET", self.path, self.headers.get("Authorization")))
                body = json.dumps(owner.payload).encode()
                self.send_response(owner.status if self.path == "/health" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                owner.requests.append(("POST", self.path, self.headers.get("Authorization")))
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class UsableSecretTests(unittest.TestCase):
    def test_same_rule_as_http_client(self) -> None:
        for value in ("abc", "sk-THIS_is~valid!", "\x21\x7e"):
            self.assertTrue(_usable_secret_value(value), value)
        for value in ("", "has space", "tab\t", "new\nline", "é", "\x7f", None, 5, b"bytes"):
            self.assertFalse(_usable_secret_value(value), repr(value))


class DoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        repo = self.root / "repo"
        repo.mkdir()
        for args in (("init", "-q"), ("config", "user.name", "Doctor"), ("config", "user.email", "d@example.invalid")):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        (repo / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True, capture_output=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.record = self.root / "codex-record.json"
        self.mode = self.root / "codex-mode.txt"
        codex = self.bin / "codex"
        codex.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """
                import json, os, sys
                with open(RECORD, "w") as stream:
                    json.dump({"argv": sys.argv[1:], "codex_home": os.environ.get("CODEX_HOME"),
                               "has_key": "BRIDGE_API_KEY" in os.environ}, stream)
                mode = open(MODE).read().strip() if os.path.exists(MODE) else "ok"
                if mode == "fail":
                    sys.stderr.write("bwrap: setting up uid map: Permission denied " + SECRET + "\\n")
                    sys.exit(1)
                """
            )
            .replace("RECORD", repr(str(self.record)))
            .replace("MODE", repr(str(self.mode)))
            .replace("SECRET", repr(SECRET)),
            encoding="utf-8",
        )
        codex.chmod(0o755)
        self.bridge = FakeBridge()

    def tearDown(self) -> None:
        self.bridge.close()
        self.temp.cleanup()

    def config(self, *, secret: str = SECRET, base_url: str | None = None) -> Path:
        (self.root / ".env.test").write_text(f"BRIDGE_API_KEY={secret}\n", encoding="utf-8")
        path = self.root / "doctor.toml"
        path.write_text(
            textwrap.dedent(
                f"""
                repo = "repo"
                base_ref = "HEAD"
                runs_root = "runs"
                worktrees_root = "worktrees"
                require_clean_base = true

                [environment]
                files = [".env.test"]

                [codex_runtime]
                home = "codex-home"

                [ui]
                default_planner_profile = "bridge"
                default_implementer_profile = "impl"
                default_reviewer_profile = "bridge"

                [model_profiles.bridge]
                display_name = "Bridge"
                roles = ["planner", "reviewer"]
                driver = "openai-chat"
                model = "chatgpt-web"
                selection_mode = "external-ui"
                base_url = {(base_url or self.bridge.base_url)!r}
                endpoint_path = "/v1/chat/completions"
                api_key_env = "BRIDGE_API_KEY"

                [model_profiles.impl]
                display_name = "Impl"
                roles = ["implementer"]
                driver = "codex"
                model = "luna"
                effort = "high"
                sandbox = "workspace-write"
                selection_mode = "cli"

                [[checks]]
                name = "test"
                argv = [{sys.executable!r}, "-c", "pass"]
                """
            ),
            encoding="utf-8",
        )
        return path

    def doctor(self, config: Path) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        environment = {"PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"}
        with mock.patch.dict(os.environ, environment):
            os.environ.pop("BRIDGE_API_KEY", None)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["doctor", "--config", str(config)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_pass_probes_sandbox_and_bridge_without_leaking_or_calling_models(self) -> None:
        code, out, err = self.doctor(self.config())
        self.assertEqual(code, 0, err)
        self.assertIn("OK env BRIDGE_API_KEY: usable", out)
        self.assertIn("OK codex sandbox: usable", out)
        self.assertIn("OK planner bridge: healthy", out)
        self.assertIn("doctor: PASS", out)
        self.assertNotIn(SECRET, out + err)
        record = json.loads(self.record.read_text())
        self.assertEqual(record["argv"], ["sandbox", "--", "/bin/true"])
        self.assertEqual(record["codex_home"], str((self.root / "codex-home").resolve()))
        self.assertFalse(record["has_key"])
        self.assertEqual(self.bridge.requests, [("GET", "/health", None)])

    def test_invalid_secret_is_reported_without_its_value(self) -> None:
        code, out, err = self.doctor(self.config(secret="has space inside"))
        self.assertEqual(code, 1)
        self.assertIn("error: required env BRIDGE_API_KEY is missing or invalid", err)
        self.assertNotIn("has space inside", out + err)

    def test_sandbox_failure_is_reported_bounded_and_redacted(self) -> None:
        self.mode.write_text("fail", encoding="utf-8")
        code, out, err = self.doctor(self.config())
        self.assertEqual(code, 1)
        self.assertIn("error: codex sandbox probe failed", err)
        self.assertIn("bwrap: setting up uid map", err)
        self.assertNotIn(SECRET, out + err)
        self.assertIn("[REDACTED]", err)
        self.assertFalse((self.root / "runs").exists())

    def test_unhealthy_bridge_fails(self) -> None:
        self.bridge.payload = {"status": "starting"}
        code, _out, err = self.doctor(self.config())
        self.assertEqual(code, 1)
        self.assertIn("planner bridge is not healthy", err)
        self.assertEqual([request[:2] for request in self.bridge.requests], [("GET", "/health")])

    def test_non_local_endpoint_is_not_probed(self) -> None:
        with mock.patch.object(cli, "_probe_bridge_health") as probe:
            code, out, err = self.doctor(self.config(base_url="https://bridge.example.invalid"))
        self.assertEqual(code, 0, err)
        probe.assert_not_called()
        self.assertNotIn("planner bridge", out)
        self.assertEqual(self.bridge.requests, [])


if __name__ == "__main__":
    unittest.main()
