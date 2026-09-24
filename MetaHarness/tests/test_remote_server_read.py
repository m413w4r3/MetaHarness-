"""Read-only behaviour of the remote gateway: auth, routing, bounds.

A real loopback server is preferred; when the sandbox forbids local sockets a
memory transport drives the same handler, mirroring ``test_remote_client.py``.
"""

from __future__ import annotations

import http.client
import inspect
import io
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.remote import server as gateway_module
from metaharness.remote import create_remote_gateway, serve_remote_gateway
from metaharness.remote.auth import load_token_file

REMOTE_TOKEN = "remote-gateway-unit-test-token"
CONTROL_TOKEN = "control-gateway-unit-test-token"
RUN_ID = "run-2024-05-06"
UNAUTHORIZED = {"error": "unauthorized", "message": "authentication required"}
UNKNOWN_ROUTE = {"error": "not_found", "message": "unknown route"}
METHOD_REFUSED = {"error": "method_not_allowed", "message": "only GET is supported"}
BAD_OFFSET = {"error": "invalid_offset", "message": "offset must be a non-negative integer"}
TEST_TIMEOUT = 10
LOCAL_HOST = "127.0.0.1"
GATEWAY_MEMORY_PORT = 8770
UPSTREAM_MEMORY_PORT = 8765


# ---------------------------------------------------------------------------
# Local MetaHarness stand-in: one fixed payload per allowlisted target.
# ---------------------------------------------------------------------------


def upstream_response(path: str) -> tuple[int, dict[str, object]]:
    target, _, query = path.partition("?")
    if target == "/api/v1/health":
        return 200, {"service": "metaharness", "api_version": 1, "status": "ok"}
    if target == "/api/v1/config":
        return 200, {"repository": {"base_ref": "HEAD"}}
    if target == "/api/v1/model-profiles":
        return 200, {"profiles": [{"name": "planner"}]}
    if target == "/api/v1/runs":
        return 200, {"runs": [{"run_id": RUN_ID}]}
    if target == f"/api/v1/runs/{RUN_ID}":
        return 200, {"run_id": RUN_ID, "status": "planned"}
    if target == f"/api/v1/runs/{RUN_ID}/progress":
        fields = dict(field.split("=", 1) for field in query.split("&") if "=" in field)
        if not fields.get("offset", "").isdigit():
            return 400, {"error": "offset must be a non-negative integer"}
        return 200, {"run_id": RUN_ID, "offset": int(fields["offset"]), "target": path}
    return 404, {"error": "not found", "message": "not found"}


class UpstreamState:
    def __init__(self) -> None:
        self.down = False
        self._requests: list[tuple[str, dict[str, str]]] = []
        self.lock = threading.Lock()

    def record(self, path: str, headers: dict[str, str]) -> None:
        with self.lock:
            self._requests.append((path, headers))

    def records(self) -> list[tuple[str, dict[str, str]]]:
        with self.lock:
            return list(self._requests)

    def paths(self) -> list[str]:
        return [path for path, _ in self.records()]

    def header_names(self) -> set[str]:
        return {name for _, headers in self.records() for name in headers}

    def header_values(self) -> list[str]:
        return [value for _, headers in self.records() for value in headers.values()]


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        self.server.state.record(  # type: ignore[attr-defined]
            self.path, {name.lower(): value for name, value in self.headers.items()}
        )
        status, payload = upstream_response(self.path)
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class UpstreamServer(ThreadingHTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# Socket-free transport: one HTTPConnection factory serving both legs.
# ---------------------------------------------------------------------------


class _Collector:
    """The ``wfile`` of a handler driven without a socket."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

    def getvalue(self) -> bytes:
        return b"".join(self._chunks)


class _MemorySocket:
    """Socket-shaped buffer: canned input, collected output."""

    def __init__(self, request: bytes = b"") -> None:
        self._reader = io.BytesIO(request)
        self._writer = _Collector()

    def makefile(self, mode: str, buffering: int = -1) -> object:
        return self._reader if "r" in mode else self._writer

    def sendall(self, data: bytes) -> None:
        # socketserver wraps the socket in _SocketWriter, which calls sendall.
        self._writer.write(data)

    def settimeout(self, _timeout: object) -> None:
        return

    def setsockopt(self, *_args: object) -> None:
        return

    def close(self) -> None:
        return

    def getvalue(self) -> bytes:
        return self._writer.getvalue()


def _parse_request(raw: bytes) -> tuple[str, str, dict[str, str]]:
    head, _, _body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return method, path, headers


def _upstream_bytes(state: UpstreamState, raw_request: bytes) -> bytes:
    if state.down:
        raise ConnectionRefusedError("upstream is down")
    _method, path, headers = _parse_request(raw_request)
    state.record(path, headers)
    status, payload = upstream_response(path)
    body = json.dumps(payload).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {http.client.responses.get(status, '')}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    )
    return head.encode("latin-1") + body


def _gateway_bytes(server: object, raw_request: bytes) -> bytes:
    """Drive one request through the real gateway handler, without a socket."""

    memory = _MemorySocket(raw_request)
    gateway_module._GatewayRequestHandler(memory, (LOCAL_HOST, 0), server)
    return memory.getvalue()


class _GatewayStub:
    """The server attributes the gateway handler reads, without a socket."""

    def __init__(self, *, remote_token: str, control_token: str, metaharness_port: int) -> None:
        self.remote_token = remote_token
        self.control_token = control_token
        self.metaharness_port = metaharness_port
        self.server_port = GATEWAY_MEMORY_PORT
        self.server_address = (LOCAL_HOST, GATEWAY_MEMORY_PORT)


class MemoryConnection(http.client.HTTPConnection):
    """An HTTPConnection whose request and response travel through memory."""

    def __init__(
        self,
        host: str,
        port: int | None = None,
        timeout: float | None = None,
        *,
        upstream: UpstreamState,
        server: object,
    ) -> None:
        super().__init__(host, port, timeout=timeout)
        self._upstream = upstream
        self._server = server
        self._request = bytearray()

    def connect(self) -> None:
        return

    def send(self, data: bytes) -> None:
        self._request.extend(data)

    def getresponse(self) -> http.client.HTTPResponse:
        raw = bytes(self._request)
        if self.port == self._server.server_port:
            served = _gateway_bytes(self._server, raw)
        else:
            served = _upstream_bytes(self._upstream, raw)
        self.sock = _MemorySocket(served)
        response = self.response_class(self.sock, method=self._method)
        response.begin()
        return response


class Response:
    """One gateway response, decoded for assertions."""

    def __init__(self, status: int, body: bytes, headers: list[tuple[str, str]]) -> None:
        self.status = status
        self.body = body
        self.text = body.decode("utf-8")
        self.headers = {name.lower(): value for name, value in headers}

    @property
    def payload(self) -> object:
        return json.loads(self.text)


class GatewayCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.remote_token_file = root / "remote.token"
        self.control_token_file = root / "control.token"
        self.remote_token_file.write_text(f"{REMOTE_TOKEN}\n", encoding="utf-8")
        self.control_token_file.write_text(f"{CONTROL_TOKEN}\n", encoding="utf-8")
        self.upstream = UpstreamState()
        self.upstream_server: UpstreamServer | None = None
        self.live = True
        try:
            self.upstream_server = UpstreamServer((LOCAL_HOST, 0), UpstreamHandler)
        except PermissionError:
            self.live = False
            self.upstream_port = UPSTREAM_MEMORY_PORT
            self.gateway: object = _GatewayStub(
                remote_token=load_token_file(self.remote_token_file),
                control_token=load_token_file(self.control_token_file),
                metaharness_port=self.upstream_port,
            )
            patcher = mock.patch.object(
                http.client, "HTTPConnection", self._memory_connection
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        else:
            self.upstream_server.state = self.upstream
            self.addCleanup(self.upstream_server.shutdown)
            self.addCleanup(self.upstream_server.server_close)
            self.start_thread(self.upstream_server)
            self.upstream_port = self.upstream_server.server_port
            self.gateway = create_remote_gateway(
                port=0,
                remote_token_file=self.remote_token_file,
                control_token_file=self.control_token_file,
                metaharness_port=self.upstream_port,
            )
            self.addCleanup(self.gateway.shutdown)
            self.addCleanup(self.gateway.server_close)
            self.start_thread(self.gateway)
        self.gateway_port: int = self.gateway.server_port  # type: ignore[attr-defined]

    def start_thread(self, server: ThreadingHTTPServer) -> None:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        self.addCleanup(thread.join, TEST_TIMEOUT)

    def _memory_connection(
        self, host, port=None, timeout=None, source_address=None, blocksize=8192
    ) -> MemoryConnection:
        return MemoryConnection(
            host, port, timeout=timeout, upstream=self.upstream, server=self.gateway
        )

    # -- helpers ----------------------------------------------------------

    def request(
        self,
        method: str = "GET",
        path: str = "/v1/health",
        *,
        token: str | None = REMOTE_TOKEN,
        extra_headers: tuple[tuple[str, str], ...] = (),
        body: bytes | None = None,
    ) -> Response:
        connection = http.client.HTTPConnection(
            LOCAL_HOST, self.gateway_port, timeout=TEST_TIMEOUT
        )
        try:
            connection.putrequest(method, path, skip_accept_encoding=True)
            if token is not None:
                connection.putheader("Authorization", f"Bearer {token}")
            for name, value in extra_headers:
                connection.putheader(name, value)
            if body is not None:
                connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body)
            response = connection.getresponse()
            return Response(response.status, response.read(), response.getheaders())
        finally:
            connection.close()

    def raw_exchange(self, request: bytes) -> bytes:
        """Send raw bytes and return the raw reply, EOF terminated."""

        if not self.live:
            return _gateway_bytes(self.gateway, request)
        with socket.create_connection(
            (LOCAL_HOST, self.gateway_port), timeout=TEST_TIMEOUT
        ) as connection:
            connection.sendall(request)
            chunks: list[bytes] = []
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)

    def take_upstream_down(self) -> None:
        if self.upstream_server is None:
            self.upstream.down = True
            return
        self.upstream_server.shutdown()
        self.upstream_server.server_close()

    def assert_unauthorized(self, response: Response) -> None:
        self.assertEqual(response.status, 401)
        self.assertEqual(response.payload, UNAUTHORIZED)

    def assert_upstream_untouched(self) -> None:
        self.assertEqual(self.upstream.paths(), [])


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class AuthorizationTests(GatewayCase):
    def test_missing_authorization_is_401(self) -> None:
        for path in ("/v1/health", "/v1/config", "/v1/model-profiles", "/v1/runs", f"/v1/runs/{RUN_ID}"):
            with self.subTest(path=path):
                self.assert_unauthorized(self.request("GET", path, token=None))
                self.assert_upstream_untouched()
        self.assertEqual(self.request("GET", "/v1/health").status, 200)

    def test_every_bad_credential_looks_exactly_like_a_missing_one(self) -> None:
        reference = self.request("GET", "/v1/health", token=None)
        cases: list[tuple[str | None, tuple[tuple[str, str], ...]]] = [
            (None, (("Authorization", f"Token {REMOTE_TOKEN}"),)),
            ("", ()),
            (" ", ()),
            ("wrong-token", ()),
            (f"{REMOTE_TOKEN[:-1]}x", ()),
            (f"{REMOTE_TOKEN}extra", ()),
            (f"{REMOTE_TOKEN}x", ()),
            (REMOTE_TOKEN.upper(), ()),
            (f"Bearer {REMOTE_TOKEN}", ()),
        ]
        for token, headers in cases:
            with self.subTest(token=token, headers=headers):
                response = self.request("GET", "/v1/health", token=token, extra_headers=headers)
                self.assert_unauthorized(response)
                self.assertEqual(response.body, reference.body)
        self.assert_upstream_untouched()

    def test_header_name_and_scheme_are_case_insensitive(self) -> None:
        cases = (
            ("authorization", f"Bearer {REMOTE_TOKEN}"),
            ("Authorization", f"bearer {REMOTE_TOKEN}"),
        )
        for name, value in cases:
            with self.subTest(header=name, value=value):
                response = self.request(
                    "GET", "/v1/health", token=None, extra_headers=((name, value),)
                )
                self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream.paths(), ["/api/v1/health"] * len(cases))

    def test_duplicated_authorization_headers_are_rejected(self) -> None:
        for extra in (
            ("Authorization", f"Bearer {REMOTE_TOKEN}"),
            ("Authorization", "Bearer"),
        ):
            with self.subTest(extra=extra):
                self.assert_unauthorized(
                    self.request("GET", "/v1/health", extra_headers=(extra,))
                )
        self.assert_upstream_untouched()

    def test_authentication_precedes_the_method_check(self) -> None:
        self.assert_unauthorized(self.request("POST", "/v1/health", token=None, body=b"{}"))
        self.assert_upstream_untouched()

    def test_the_upstream_leg_carries_no_credential(self) -> None:
        self.assertEqual(self.request("GET", "/v1/health").status, 200)
        headers = self.upstream.records()[0][1]
        self.assertEqual(headers.get("accept"), "application/json")
        self.assertEqual(headers.get("host"), f"127.0.0.1:{self.upstream_port}")
        self.assertLessEqual(set(headers), {"host", "accept", "accept-encoding"})
        for value in headers.values():
            self.assertNotIn(REMOTE_TOKEN, value)
            self.assertNotIn(CONTROL_TOKEN, value)


# ---------------------------------------------------------------------------
# Read routes
# ---------------------------------------------------------------------------


class ReadRouteTests(GatewayCase):
    def test_allowlisted_routes_map_to_local_targets(self) -> None:
        cases = (
            ("/v1/health", "/api/v1/health"),
            ("/v1/config", "/api/v1/config"),
            ("/v1/model-profiles", "/api/v1/model-profiles"),
            ("/v1/runs", "/api/v1/runs"),
            ("/v1/runs?limit=5", "/api/v1/runs"),
            (f"/v1/runs/{RUN_ID}", f"/api/v1/runs/{RUN_ID}"),
            (f"/v1/runs/{RUN_ID}?limit=5", f"/api/v1/runs/{RUN_ID}"),
            (f"/v1/runs/{RUN_ID}/progress?offset=0", f"/api/v1/runs/{RUN_ID}/progress?offset=0"),
            (f"/v1/runs/{RUN_ID}/progress?offset=7", f"/api/v1/runs/{RUN_ID}/progress?offset=7"),
        )
        for path, target in cases:
            with self.subTest(path=path):
                response = self.request("GET", path)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.payload, upstream_response(target)[1])
        self.assertEqual(self.upstream.paths(), [target for _, target in cases])

    def test_progress_offset_is_canonicalised(self) -> None:
        for query, offset in (("offset=0003", "3"), ("offset=%31", "1")):
            with self.subTest(query=query):
                self.assertEqual(
                    self.request("GET", f"/v1/runs/{RUN_ID}/progress?{query}").status, 200
                )
                self.assertEqual(
                    self.upstream.paths()[-1],
                    f"/api/v1/runs/{RUN_ID}/progress?offset={offset}",
                )

    def test_upstream_status_and_body_are_passed_through(self) -> None:
        response = self.request("GET", "/v1/runs/other-run")
        self.assertEqual(response.status, 404)
        self.assertEqual(response.payload, {"error": "not found", "message": "not found"})
        self.assertEqual(self.upstream.paths(), ["/api/v1/runs/other-run"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class RefusalTests(GatewayCase):
    def test_bad_offsets_are_rejected_before_proxying(self) -> None:
        queries = (
            "",
            "?offset=",
            "?offset=-1",
            "?offset=abc",
            "?offset=1&offset=2",
            "?offset=+1",
            "?offset=1.0",
            "?offset=1%20",
            "?offset=%D9%A1",
            "?offset=" + "9" * 40,
        )
        for query in queries:
            with self.subTest(query=query):
                response = self.request("GET", f"/v1/runs/{RUN_ID}/progress{query}")
                self.assertEqual(response.status, 400)
                self.assertEqual(response.payload, BAD_OFFSET)
        self.assert_upstream_untouched()

    def test_run_ids_are_single_plain_path_components(self) -> None:
        for run_id in ("..", ".", "%2e%2e", "a..b", "a%2Fb", "a%5Cb", "a%00b", "run%20id"):
            with self.subTest(run_id=run_id):
                response = self.request("GET", f"/v1/runs/{run_id}")
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload, {"error": "invalid_run_id", "message": "invalid run id"}
                )
                self.assertEqual(
                    self.request("GET", f"/v1/runs/{run_id}/progress?offset=0").status, 400
                )
        self.assert_upstream_untouched()

    def test_unknown_routes_are_404_and_never_proxied(self) -> None:
        paths = (
            "/",
            "/v1",
            "/v1/",
            "/v1/health/",
            "/v1/unknown",
            "/v2/health",
            "/api/v1/health",
            "/api/v1/runs",
            "/v1/runs/",
            "/v1/runs/../config",
            f"/v1/runs/{RUN_ID}/live",
            f"/v1/runs/{RUN_ID}/artifact?name=review.json",
            f"/v1/runs/{RUN_ID}/progress/extra?offset=0",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.request("GET", path)
                self.assertEqual(response.status, 404)
                self.assertEqual(response.payload, UNKNOWN_ROUTE)
        self.assert_upstream_untouched()

    def test_non_get_methods_are_405(self) -> None:
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE", "FROB"):
            with self.subTest(method=method):
                response = self.request(
                    method, "/v1/health", body=b"{}" if method == "POST" else None
                )
                self.assertEqual(response.status, 405)
                if method == "HEAD":
                    self.assertEqual(response.body, b"")
                    self.assertEqual(
                        response.headers["content-length"],
                        str(len(json.dumps(METHOD_REFUSED).encode("utf-8"))),
                    )
                else:
                    self.assertEqual(response.payload, METHOD_REFUSED)
        self.assert_upstream_untouched()

    def test_unreachable_upstream_is_502(self) -> None:
        self.take_upstream_down()
        response = self.request("GET", "/v1/health")
        self.assertEqual(response.status, 502)
        self.assertEqual(
            response.payload,
            {"error": "upstream_error", "message": "local MetaHarness request failed"},
        )

    def test_malformed_request_line_is_answered_with_json_and_not_echoed(self) -> None:
        raw = self.raw_exchange(b"BOGUS-LINE\r\n\r\n")
        self.assertTrue(raw.startswith(b"HTTP/1.0 400"), raw[:40])
        self.assertIn(b"request_error", raw)
        self.assertNotIn(b"BOGUS-LINE", raw)
        self.assert_upstream_untouched()


# ---------------------------------------------------------------------------
# Response hygiene
# ---------------------------------------------------------------------------


class ResponseHygieneTests(GatewayCase):
    def responses(self) -> list[Response]:
        return [
            self.request("GET", "/v1/health"),
            self.request("GET", "/v1/health", token=None),
            self.request("GET", "/v1/unknown"),
            self.request("POST", "/v1/health"),
            self.request("GET", f"/v1/runs/{RUN_ID}/progress?offset=nope"),
            self.request("GET", "/v1/runs/missing"),
        ]

    def test_every_response_is_json_with_no_store_and_no_cors(self) -> None:
        for response in self.responses():
            with self.subTest(status=response.status):
                self.assertEqual(response.headers["content-type"], "application/json; charset=utf-8")
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertIsInstance(response.payload, dict)
                for name in response.headers:
                    self.assertNotIn("access-control", name)

    def test_neither_token_is_ever_returned(self) -> None:
        for response in self.responses():
            with self.subTest(status=response.status):
                self.assertNotIn(REMOTE_TOKEN, response.text)
                self.assertNotIn(CONTROL_TOKEN, response.text)
                for value in response.headers.values():
                    self.assertNotIn(REMOTE_TOKEN, value)
                    self.assertNotIn(CONTROL_TOKEN, value)
        self.assertNotIn(REMOTE_TOKEN, "".join(self.upstream.paths()))
        self.assertNotIn("authorization", self.upstream.header_names())
        self.assertNotIn("x-metaharness-token", self.upstream.header_names())
        for value in self.upstream.header_values():
            self.assertNotIn(REMOTE_TOKEN, value)
            self.assertNotIn(CONTROL_TOKEN, value)

    def test_requests_are_not_logged(self) -> None:
        captured = StringIO()
        with redirect_stderr(captured):
            self.request("GET", "/v1/health")
            self.request("GET", "/v1/secret-run", token=None)
            self.request("GET", f"/v1/runs/{RUN_ID}/progress?offset=bad")
            self.raw_exchange(b"BOGUS-LINE\r\n\r\n")
        self.assertEqual(captured.getvalue(), "")


# ---------------------------------------------------------------------------
# Gateway surface
# ---------------------------------------------------------------------------


class GatewaySurfaceTests(GatewayCase):
    def test_gateway_takes_no_host_argument(self) -> None:
        for function in (serve_remote_gateway, create_remote_gateway):
            self.assertNotIn("host", inspect.signature(function).parameters)

    def test_gateway_is_always_bound_to_loopback(self) -> None:
        with mock.patch.object(gateway_module, "_RemoteGatewayServer") as server_class:
            server = create_remote_gateway(
                port=8770,
                remote_token_file=self.remote_token_file,
                control_token_file=self.control_token_file,
                metaharness_port=self.upstream_port,
            )
        address, handler = server_class.call_args.args
        self.assertEqual(address, (LOCAL_HOST, 8770))
        self.assertIs(handler, gateway_module._GatewayRequestHandler)
        self.assertEqual(server.remote_token, REMOTE_TOKEN)
        self.assertEqual(server.control_token, CONTROL_TOKEN)
        self.assertEqual(server.metaharness_port, self.upstream_port)

    def test_live_gateway_binds_loopback_only(self) -> None:
        if not self.live:
            self.skipTest("local sockets are unavailable in this sandbox")
        self.assertEqual(self.gateway.server_address[0], LOCAL_HOST)  # type: ignore[attr-defined]

    def test_token_files_are_read_eagerly(self) -> None:
        with self.assertRaises(OSError):
            create_remote_gateway(
                port=0,
                remote_token_file=Path(self.temp.name) / "absent",
                control_token_file=self.control_token_file,
            )
        self.control_token_file.write_bytes(b"")
        with self.assertRaises(ValueError):
            create_remote_gateway(
                port=0,
                remote_token_file=self.remote_token_file,
                control_token_file=self.control_token_file,
            )

    def test_serve_remote_gateway_serves_loopback_and_stops_on_shutdown(self) -> None:
        if not self.live:
            self.skipTest("local sockets are unavailable in this sandbox")
        captured: list[ThreadingHTTPServer] = []
        real_serve_forever = ThreadingHTTPServer.serve_forever

        def capture(server, *args, **kwargs):
            captured.append(server)
            return real_serve_forever(server, *args, **kwargs)

        thread = threading.Thread(
            target=serve_remote_gateway,
            kwargs={
                "port": 0,
                "remote_token_file": self.remote_token_file,
                "control_token_file": self.control_token_file,
                "metaharness_port": self.upstream_port,
            },
            daemon=True,
        )
        with mock.patch.object(gateway_module._RemoteGatewayServer, "serve_forever", capture):
            thread.start()
            deadline = time.monotonic() + TEST_TIMEOUT
            while not captured and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(captured, "serve_remote_gateway never started serving")
            self.assertEqual(captured[0].server_address[0], LOCAL_HOST)
            port = captured[0].server_port
        connection = http.client.HTTPConnection(LOCAL_HOST, port, timeout=TEST_TIMEOUT)
        try:
            connection.request("GET", "/v1/health")
            self.assertEqual(connection.getresponse().status, 401)
        finally:
            connection.close()
        captured[0].shutdown()
        thread.join(timeout=TEST_TIMEOUT)
        self.assertFalse(thread.is_alive())
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((LOCAL_HOST, port))


if __name__ == "__main__":
    unittest.main()
