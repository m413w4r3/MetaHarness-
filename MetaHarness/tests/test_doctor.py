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
from metaharness.agent import auth as auth_module  # noqa: E402
from metaharness.agent.auth import check_codex_authentication  # noqa: E402
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
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
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
                args = sys.argv[1:]
                if args == ["sandbox", "--", "/bin/true"]:
                    with open(RECORD, "w") as stream:
                        json.dump({"argv": args, "codex_home": os.environ.get("CODEX_HOME"),
                                   "has_key": "BRIDGE_API_KEY" in os.environ}, stream)
                if args == ["login", "--help"]:
                    print("Commands:\\n  status  Show login status")
                    sys.exit(0)
                if args == ["login", "status"]:
                    state = os.path.join(os.environ["CODEX_HOME"], "auth.json")
                    if os.path.isfile(state) and os.path.getsize(state) > 0:
                        print("Logged in")
                        sys.exit(0)
                    sys.stderr.write("Not logged in\\n")
                    sys.exit(1)
                mode = open(MODE).read().strip() if os.path.exists(MODE) else "ok"
                if mode == "parser_fail" and args and args[0] == "exec":
                    sys.stderr.write("unknown option: --strict-config\\n")
                    sys.exit(2)
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

    def config(
        self, *, secret: str = SECRET, base_url: str | None = None,
        authenticated: bool = True,
    ) -> Path:
        (self.root / ".env.test").write_text(f"BRIDGE_API_KEY={secret}\n", encoding="utf-8")
        codex_home = self.root / "codex-home"
        if authenticated:
            codex_home.mkdir(parents=True, exist_ok=True)
            (codex_home / "auth.json").write_text("authenticated-state\n", encoding="utf-8")
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

                [[check_catalog]]
                id = "test"
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
        self.assertIn("OK codex CLI contract: supported", out)
        self.assertIn("OK codex authentication: available", out)
        self.assertIn("OK planner bridge: healthy", out)
        self.assertIn("doctor: PASS", out)
        self.assertNotIn(SECRET, out + err)
        record = json.loads(self.record.read_text())
        self.assertEqual(record["argv"], ["sandbox", "--", "/bin/true"])
        self.assertEqual(record["codex_home"], str((self.root / "codex-home").resolve()))
        self.assertFalse(record["has_key"])
        self.assertEqual(self.bridge.requests, [("GET", "/health", None)])

    def test_no_codex_profile_skips_every_codex_probe(self) -> None:
        path = self.config()
        text = path.read_text(encoding="utf-8").replace(
            'driver = "codex"\nmodel = "luna"\neffort = "high"\nsandbox = "workspace-write"',
            f'driver = "external"\nmodel = "worker"\nargv = [{sys.executable!r}, "-c", "pass"]',
        )
        self.assertIn('driver = "external"', text)
        path.write_text(text, encoding="utf-8")
        (self.bin / "codex").unlink()
        code, out, err = self.doctor(path)
        self.assertEqual(code, 0, err)
        self.assertIn("codex: no codex profile configured", out)
        self.assertNotIn("codex binary", out + err)
        self.assertFalse(self.record.exists())

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

    def test_codex_parser_failure_is_reported_without_a_model_call(self) -> None:
        self.mode.write_text("parser_fail", encoding="utf-8")
        code, _out, err = self.doctor(self.config())
        self.assertEqual(code, 1)
        self.assertIn("error: unsupported Codex CLI for MetaHarness worker", err)
        self.assertNotIn("model", err.casefold())
        self.assertFalse((self.root / "runs").exists())

    def test_unavailable_authentication_fails_closed_with_configured_home(self) -> None:
        code, out, err = self.doctor(self.config(authenticated=False))
        configured_home = str((self.root / "codex-home").resolve())
        self.assertEqual(code, 1)
        self.assertIn("OK codex sandbox: usable", out)
        self.assertIn("error: codex authentication is unavailable for managed CODEX_HOME", err)
        self.assertIn(f'hint: run CODEX_HOME="{configured_home}" codex login', err)
        self.assertNotIn(SECRET, out + err)
        self.assertNotIn("authenticated-state", out + err)

    def test_authentication_status_is_local_and_does_not_make_model_call(self) -> None:
        home = self.root / "auth-check"
        home.mkdir()
        environment = {"PATH": str(self.bin)}
        (home / "auth.json").write_text(SECRET, encoding="utf-8")
        status = check_codex_authentication(home, environment=environment)
        self.assertTrue(status.available)
        self.assertNotIn(SECRET, status.detail)

    def test_cli_authentication_checks_only_nonempty_known_state(self) -> None:
        home = self.root / "auth-check-help"
        home.mkdir()
        (home / "auth.json").write_text(SECRET, encoding="utf-8")
        help_only = subprocess.CompletedProcess(
            ["codex", "login", "--help"], 0,
            stdout="Usage: codex login\nCommands:\n  device-auth\n",
            stderr="",
        )
        with mock.patch.object(auth_module, "_run_local", return_value=help_only):
            status = check_codex_authentication(
                home, environment={"PATH": str(self.bin)}
            )
        self.assertTrue(status.available)
        self.assertNotIn(SECRET, status.detail)

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

    def test_claude_probe_is_parser_only_and_does_not_use_help_text(self) -> None:
        probe = self.root / "claude-probe"
        model_call = self.root / "model-call"
        argv_record = self.root / "claude-argv.json"
        probe.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            f"pathlib.Path({str(argv_record)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            f"if sys.argv[-1] != '--help': pathlib.Path({str(model_call)!r}).write_text('called')\n"
            "print('Usage: claude')\n",
            encoding="utf-8",
        )
        probe.chmod(0o755)
        supported, _detail = cli._probe_claude_capabilities(
            str(probe), {"PATH": str(self.bin)}, self.root
        )
        self.assertTrue(supported)
        recorded = json.loads(argv_record.read_text())
        self.assertEqual(recorded[-1], "--help")
        self.assertIn("--safe-mode", recorded)
        self.assertNotIn("--bare", recorded)
        self.assertIn("--restricted", recorded)
        self.assertIn("--settings", recorded)
        self.assertIn(str(self.root / "settings.json"), recorded)
        self.assertFalse(model_call.exists())

        probe.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "sys.exit(2 if '--no-chrome' in sys.argv else 0)\n",
            encoding="utf-8",
        )
        supported, _detail = cli._probe_claude_capabilities(
            str(probe), {"PATH": str(self.bin)}, self.root
        )
        self.assertFalse(supported)


if __name__ == "__main__":
    unittest.main()
