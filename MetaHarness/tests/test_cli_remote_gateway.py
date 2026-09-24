"""CLI contract of ``metaharness remote-gateway``.

The command starts the loopback gateway only: the parser exposes no host
option, both token files are required and read before anything is bound, the
two tokens must differ, the ports must be valid and distinct, the banner is
printed only for a bound socket, and no token value ever reaches stdout or
stderr.
"""

import argparse
import contextlib
import http.client
import io
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.cli import build_parser, main
from metaharness.remote import server as gateway_module

REMOTE_TOKEN = "remote-gateway-cli-remote-token"
CONTROL_TOKEN = "remote-gateway-cli-control-token"
GATEWAY_OPTIONS = ("--port", "--metaharness-port", "--remote-token-file", "--control-token-file")
LOCAL_HOST = "127.0.0.1"
TEST_TIMEOUT = 10


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def local_sockets_available() -> bool:
    try:
        with socket.socket() as probe:
            probe.bind((LOCAL_HOST, 0))
    except OSError:
        return False
    return True


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LOCAL_HOST, 0))
        return int(probe.getsockname()[1])


class GatewayParserTests(unittest.TestCase):
    """Defaults, required files, typed ports, and the absence of ``--host``."""

    def parse(self, *extra: str) -> argparse.Namespace:
        return build_parser().parse_args(
            [
                "remote-gateway",
                "--remote-token-file",
                "remote.token",
                "--control-token-file",
                "control.token",
                *extra,
            ]
        )

    def test_defaults_are_8770_and_8765(self) -> None:
        args = self.parse()
        self.assertEqual(args.command, "remote-gateway")
        self.assertEqual(args.port, 8770)
        self.assertEqual(args.metaharness_port, 8765)
        self.assertEqual(args.remote_token_file, Path("remote.token"))
        self.assertEqual(args.control_token_file, Path("control.token"))

    def test_explicit_values_are_parsed(self) -> None:
        args = self.parse("--port", "9001", "--metaharness-port", "9002")
        self.assertEqual((args.port, args.metaharness_port), (9001, 9002))

    def test_token_files_are_required(self) -> None:
        for missing in ("--remote-token-file", "--control-token-file"):
            with self.subTest(missing=missing):
                argv = [
                    "remote-gateway",
                    "--remote-token-file",
                    "remote.token",
                    "--control-token-file",
                    "control.token",
                ]
                index = argv.index(missing)
                del argv[index : index + 2]
                stderr = io.StringIO()
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
                    build_parser().parse_args(argv)
                self.assertIn(missing, stderr.getvalue())

    def test_non_integer_port_is_refused(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
            self.parse("--port", "not-a-port")
        self.assertIn("invalid int value", stderr.getvalue())

    def test_host_option_does_not_exist(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(stderr):
            self.parse("--host", "0.0.0.0")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())
        self.assertIn("--host", stderr.getvalue())

    def test_help_lists_exactly_the_documented_options(self) -> None:
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(stdout):
            build_parser().parse_args(["remote-gateway", "--help"])
        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertTrue(text.startswith("usage: metaharness remote-gateway"), text)
        for option in GATEWAY_OPTIONS:
            self.assertIn(option, text)
        self.assertNotIn("--host", text)


class GatewayCommandTests(unittest.TestCase):
    """Startup output, token handling, and port validation of the command."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.remote_token_file = self.root / "remote.token"
        self.control_token_file = self.root / "control.token"
        self.remote_token_file.write_text(f"{REMOTE_TOKEN}\n", encoding="utf-8")
        self.control_token_file.write_text(f"{CONTROL_TOKEN}\n", encoding="utf-8")

    def argv(self, *extra: str) -> list[str]:
        return [
            "remote-gateway",
            "--port",
            "8770",
            "--metaharness-port",
            "8765",
            "--remote-token-file",
            str(self.remote_token_file),
            "--control-token-file",
            str(self.control_token_file),
            *extra,
        ]

    def run_gateway(
        self, *extra: str, create: mock.Mock | None = None
    ) -> tuple[int, str, str, mock.Mock]:
        if create is None:
            server = mock.Mock()
            server.server_port = 8770
            server.metaharness_port = 8765
            create = mock.Mock(return_value=server)
        with mock.patch("metaharness.cli.create_remote_gateway", create):
            code, out, err = run_cli(self.argv(*extra))
        for token in (REMOTE_TOKEN, CONTROL_TOKEN):
            self.assertNotIn(token, out + err)
        return code, out, err, create

    def test_startup_prints_only_the_two_loopback_lines(self) -> None:
        code, out, err, create = self.run_gateway()
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertEqual(
            out,
            "MetaHarness remote gateway listening on http://127.0.0.1:8770\n"
            "Upstream MetaHarness: http://127.0.0.1:8765\n",
        )
        # Exactly these four arguments: no host is ever forwarded.
        create.assert_called_once_with(
            port=8770,
            metaharness_port=8765,
            remote_token_file=self.remote_token_file,
            control_token_file=self.control_token_file,
        )
        server = create.return_value
        server.serve_forever.assert_called_once_with()
        server.server_close.assert_called_once_with()

    def test_failed_bind_never_announces_a_listening_gateway(self) -> None:
        create = mock.Mock(side_effect=OSError("address already in use"))
        code, out, err, _create = self.run_gateway(create=create)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("address already in use", err)

    def test_identical_tokens_are_refused_before_serving(self) -> None:
        self.control_token_file.write_text(f"{REMOTE_TOKEN}\n", encoding="utf-8")
        code, out, err, create = self.run_gateway()
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("remote token and control token must be different", err)
        create.assert_not_called()

    def test_out_of_range_ports_are_refused(self) -> None:
        for option in ("--port", "--metaharness-port"):
            for value in ("0", "65536"):
                with self.subTest(option=option, value=value):
                    code, out, err, create = self.run_gateway(option, value)
                    self.assertEqual(code, 2)
                    self.assertEqual(out, "")
                    self.assertIn("must be between 1 and 65535", err)
                    create.assert_not_called()

    def test_equal_gateway_and_metaharness_ports_are_refused(self) -> None:
        code, out, err, create = self.run_gateway("--metaharness-port", "8770")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("must be different", err)
        create.assert_not_called()

    def test_absent_token_file_is_refused_without_echoing_a_token(self) -> None:
        absent = self.root / "absent.token"
        code, out, err, create = self.run_gateway("--remote-token-file", str(absent))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn(str(absent), err)
        create.assert_not_called()

    def test_empty_token_file_is_refused(self) -> None:
        self.remote_token_file.write_bytes(b"")
        code, out, err, create = self.run_gateway()
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("error:", err)
        create.assert_not_called()

    def test_live_gateway_binds_loopback_and_serves(self) -> None:
        if not local_sockets_available():
            self.skipTest("local sockets are unavailable in this sandbox")
        gateway_port = free_port()
        upstream_port = free_port()
        while upstream_port == gateway_port:
            upstream_port = free_port()
        argv = self.argv(
            "--port", str(gateway_port), "--metaharness-port", str(upstream_port)
        )
        captured: list[ThreadingHTTPServer] = []
        result: list[int] = []
        real_serve_forever = gateway_module._RemoteGatewayServer.serve_forever

        def capture(server, *args, **kwargs):
            captured.append(server)
            return real_serve_forever(server, *args, **kwargs)

        stdout = io.StringIO()
        thread = threading.Thread(target=lambda: result.append(main(argv)), daemon=True)
        with mock.patch.object(
            gateway_module._RemoteGatewayServer, "serve_forever", capture
        ), contextlib.redirect_stdout(stdout):
            thread.start()
            deadline = time.monotonic() + TEST_TIMEOUT
            while not captured and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(captured, "the gateway never started serving")
            self.assertEqual(captured[0].server_address[0], LOCAL_HOST)
            connection = http.client.HTTPConnection(
                LOCAL_HOST, gateway_port, timeout=TEST_TIMEOUT
            )
            try:
                connection.request("GET", "/v1/health")
                self.assertEqual(connection.getresponse().status, 401)
                connection.request(
                    "GET", "/v1/health", headers={"Authorization": f"Bearer {REMOTE_TOKEN}"}
                )
                # Nothing listens upstream: reaching 502 proves the CLI passed
                # its ports through and the request was routed, not refused.
                self.assertEqual(connection.getresponse().status, 502)
            finally:
                connection.close()
            captured[0].shutdown()
        thread.join(TEST_TIMEOUT)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [0])
        self.assertEqual(
            stdout.getvalue(),
            f"MetaHarness remote gateway listening on http://{LOCAL_HOST}:{gateway_port}\n"
            f"Upstream MetaHarness: http://{LOCAL_HOST}:{upstream_port}\n",
        )


if __name__ == "__main__":
    unittest.main()
