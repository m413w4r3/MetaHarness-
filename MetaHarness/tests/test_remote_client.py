from __future__ import annotations

import http.client
import inspect
import io
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.remote import (
    LOCAL_HOST,
    MAX_RESPONSE_BYTES,
    LocalMetaHarnessClient,
    LocalMetaHarnessError,
)

TOKEN = "unit-test-control-token"
RUN_ID = "run-2024-05-06"
FALLBACK_PORT = 54321


class RecordedRequest:
    def __init__(self, method: str, path: str, headers: dict[str, str], body: bytes):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body


class ServerState:
    """Requests seen so far plus the responder used for the next one."""

    def __init__(self) -> None:
        self.records: list[RecordedRequest] = []
        self.lock = threading.Lock()
        self.responder = None


def send_json(handler, payload: object, status: int = 200) -> None:
    send_bytes(
        handler,
        json.dumps(payload).encode("utf-8"),
        status=status,
        content_type="application/json; charset=utf-8",
    )


def send_bytes(
    handler,
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "application/json",
) -> None:
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except OSError:
        # A client that already gave up (timeout, oversized response) may
        # close the socket first; the test server stays silent about it.
        pass


def default_response(handler, record) -> None:
    send_json(handler, {"status": "ok"})


# --------------------------------------------------------------------------
# Preferred transport: a real, temporary local HTTP server.
# --------------------------------------------------------------------------


class TestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MetaHarnessTestServer"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        self._dispatch("GET", b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self._dispatch("POST", self.rfile.read(length))

    def _dispatch(self, method: str, body: bytes) -> None:
        state = self.server.state
        record = RecordedRequest(method, self.path, dict(self.headers.items()), body)
        with state.lock:
            state.records.append(record)
        (state.responder or default_response)(self, record)


class TestServer(ThreadingHTTPServer):
    daemon_threads = True


# --------------------------------------------------------------------------
# Fallback transport: real HTTPConnection serialisation over memory, used
# only when the sandbox forbids local sockets.
# --------------------------------------------------------------------------


class MemoryHandler:
    """Collects a response exactly like BaseHTTPRequestHandler would write it."""

    def __init__(self) -> None:
        self._status = 200
        self._headers: list[tuple[str, str]] = []
        self._chunks: list[bytes] = []
        self.wfile = self

    def send_response(self, status, message=None) -> None:
        self._status = int(status)

    def send_header(self, name, value) -> None:
        self._headers.append((str(name), str(value)))

    def end_headers(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        self._chunks.append(bytes(data))

    def to_bytes(self) -> bytes:
        reason = http.client.responses.get(self._status, "")
        lines = [f"HTTP/1.1 {self._status} {reason}\r\n"]
        lines.extend(f"{name}: {value}\r\n" for name, value in self._headers)
        lines.append("\r\n")
        return "".join(lines).encode("latin-1") + b"".join(self._chunks)


class MemorySocket:
    def __init__(self, response: bytes):
        self._response = response

    def makefile(self, mode: str, buffering: int = -1):
        return io.BytesIO(self._response if "r" in mode else b"")

    def settimeout(self, timeout) -> None:
        pass

    def close(self) -> None:
        pass


def parse_request(raw: bytes) -> RecordedRequest:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip()] = value.strip()
    return RecordedRequest(method, path, headers, body)


class MemoryHTTPConnection(http.client.HTTPConnection):
    """An HTTPConnection whose request and response travel through memory."""

    def __init__(self, host, port=None, timeout=None, *, state: ServerState):
        super().__init__(host, port, timeout=timeout)
        self._state = state
        self._request = bytearray()

    def connect(self) -> None:
        pass

    def send(self, data: bytes) -> None:
        self._request.extend(data)

    def getresponse(self):
        started = time.monotonic()
        response = _serve_in_memory(self._state, bytes(self._request))
        if self.timeout is not None and time.monotonic() - started > self.timeout:
            raise TimeoutError("local server exceeded the client timeout")
        self.sock = MemorySocket(response)
        return super().getresponse()


def _serve_in_memory(state: ServerState, raw: bytes) -> bytes:
    record = parse_request(raw)
    with state.lock:
        state.records.append(record)
    handler = MemoryHandler()
    (state.responder or default_response)(handler, record)
    return handler.to_bytes()


def memory_transport(state: ServerState):
    def factory(host, port=None, timeout=None, source_address=None, blocksize=8192):
        return MemoryHTTPConnection(host, port, timeout=timeout, state=state)

    return factory


class RemoteClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ServerState()
        self.ip_transport = True
        try:
            server = TestServer((LOCAL_HOST, 0), TestHandler)
        except PermissionError:
            self.ip_transport = False
            self.port = FALLBACK_PORT
            self.server = None
            self.thread = None
            patcher = mock.patch.object(
                http.client, "HTTPConnection", memory_transport(self.state)
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        else:
            server.state = self.state
            self.server = server
            self.port = server.server_port
            self.thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            self.thread.start()
            self.addCleanup(self._stop_server)
        self.client = LocalMetaHarnessClient(
            port=self.port, control_token=TOKEN, timeout_seconds=5.0
        )

    def _stop_server(self) -> None:
        assert self.server is not None and self.thread is not None
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def requests(self) -> list[RecordedRequest]:
        with self.state.lock:
            return list(self.state.records)

    def respond_with(self, responder) -> None:
        self.state.responder = responder


class HealthTests(RemoteClientTest):
    def test_health_returns_object_over_local_host(self) -> None:
        payload = {"service": "metaharness", "api_version": 1, "status": "ok"}
        self.respond_with(lambda handler, record: send_json(handler, payload))

        self.assertEqual(self.client.health(), payload)

        (record,) = self.requests()
        self.assertEqual(record.method, "GET")
        self.assertEqual(record.path, "/api/v1/health")
        self.assertEqual(record.headers["Host"], f"{LOCAL_HOST}:{self.port}")
        self.assertEqual(record.headers["Accept"], "application/json")
        self.assertNotIn("X-MetaHarness-Token", record.headers)
        self.assertNotIn("Content-Type", record.headers)
        self.assertEqual(record.body, b"")

    def test_host_cannot_be_chosen_by_the_caller(self) -> None:
        self.assertNotIn("host", inspect.signature(LocalMetaHarnessClient).parameters)
        with self.assertRaises(TypeError):
            LocalMetaHarnessClient(host="evil.example", port=self.port, control_token=TOKEN)


class ReadEndpointTests(RemoteClientTest):
    def test_config(self) -> None:
        self.respond_with(lambda handler, record: send_json(handler, {"sections": []}))
        self.assertEqual(self.client.config(), {"sections": []})
        self.assertEqual(self.requests()[-1].path, "/api/v1/config")

    def test_model_profiles(self) -> None:
        self.respond_with(lambda handler, record: send_json(handler, {"profiles": []}))
        self.assertEqual(self.client.model_profiles(), {"profiles": []})
        self.assertEqual(self.requests()[-1].path, "/api/v1/model-profiles")

    def test_list_runs(self) -> None:
        self.respond_with(
            lambda handler, record: send_json(handler, {"runs": [{"run_id": RUN_ID}]})
        )
        self.assertEqual(self.client.list_runs(), {"runs": [{"run_id": RUN_ID}]})
        record = self.requests()[-1]
        self.assertEqual(record.method, "GET")
        self.assertEqual(record.path, "/api/v1/runs")

    def test_get_run(self) -> None:
        self.respond_with(
            lambda handler, record: send_json(handler, {"run_id": RUN_ID, "status": "running"})
        )
        self.assertEqual(self.client.get_run(RUN_ID), {"run_id": RUN_ID, "status": "running"})
        record = self.requests()[-1]
        self.assertEqual(record.path, f"/api/v1/runs/{RUN_ID}")
        self.assertEqual(record.headers["Host"], f"{LOCAL_HOST}:{self.port}")

    def test_progress_offset(self) -> None:
        self.respond_with(lambda handler, record: send_json(handler, {"offset": 42, "events": []}))
        self.assertEqual(self.client.progress(RUN_ID, 42), {"offset": 42, "events": []})
        self.assertEqual(self.requests()[-1].path, f"/api/v1/runs/{RUN_ID}/progress?offset=42")

    def test_progress_offset_zero(self) -> None:
        self.assertEqual(self.client.progress(RUN_ID, 0), {"status": "ok"})
        self.assertEqual(self.requests()[-1].path, f"/api/v1/runs/{RUN_ID}/progress?offset=0")

    def test_unsafe_run_ids_are_refused_before_any_request(self) -> None:
        unsafe = ("", "/", "a/b", "a\\b", "..", "../etc", "a..b", "a\nb", "a b", None, 42)
        for run_id in unsafe:
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    self.client.get_run(run_id)
        self.assertEqual(self.requests(), [])

    def test_bad_offsets_are_refused_before_any_request(self) -> None:
        for offset in (-1, -42, 1.5, "0", None, True):
            with self.subTest(offset=offset):
                with self.assertRaises(ValueError):
                    self.client.progress(RUN_ID, offset)
        self.assertEqual(self.requests(), [])


class PostPrimitiveTests(RemoteClientTest):
    def test_post_sends_token_and_json_body(self) -> None:
        self.respond_with(lambda handler, record: send_json(handler, {"accepted": True}))
        status, payload = self.client._request_json(
            "POST", f"/api/v1/runs/{RUN_ID}/resume", {"reason": "retry"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"accepted": True})

        (record,) = self.requests()
        self.assertEqual(record.method, "POST")
        self.assertEqual(record.headers["X-MetaHarness-Token"], TOKEN)
        self.assertEqual(record.headers["Content-Type"], "application/json")
        self.assertEqual(record.headers["Accept"], "application/json")
        self.assertEqual(json.loads(record.body.decode("utf-8")), {"reason": "retry"})

    def test_get_never_sends_the_token(self) -> None:
        self.client._request_json("GET", "/api/v1/runs")
        self.assertNotIn("X-MetaHarness-Token", self.requests()[-1].headers)

    def test_invalid_requests_are_refused_before_any_request(self) -> None:
        cases = (
            ("DELETE", "/api/v1/runs", None),
            ("GET", "api/v1/runs", None),
            ("GET", "/api/v1/runs", {"nope": True}),
            ("POST", "/api/v1/runs", ["not", "an", "object"]),
            ("POST", "/api/v1/runs", {"value": object()}),
        )
        for method, path, body in cases:
            with self.subTest(method=method, path=path):
                with self.assertRaises(ValueError):
                    self.client._request_json(method, path, body)
        self.assertEqual(self.requests(), [])


class ErrorHandlingTests(RemoteClientTest):
    def test_http_error_carries_status_and_payload(self) -> None:
        self.respond_with(
            lambda handler, record: send_json(handler, {"error": "forbidden"}, status=403)
        )
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertEqual(raised.exception.status, 403)
        self.assertEqual(raised.exception.payload, {"error": "forbidden"})
        self.assertIn("403", str(raised.exception))
        self.assertNotIn(TOKEN, f"{raised.exception!s} {raised.exception!r}")

    def test_redirect_is_not_followed(self) -> None:
        def redirect(handler, record):
            handler.send_response(302)
            handler.send_header("Location", "/api/v1/config")
            handler.send_header("Content-Length", "0")
            handler.end_headers()

        self.respond_with(redirect)
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertEqual(raised.exception.status, 302)
        self.assertIn("302", str(raised.exception))
        self.assertEqual(len(self.requests()), 1)

    def test_invalid_json_is_refused(self) -> None:
        self.respond_with(lambda handler, record: send_bytes(handler, b"{not json"))
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertIn("JSON", str(raised.exception))
        self.assertEqual(raised.exception.status, 200)

    def test_invalid_utf8_is_refused(self) -> None:
        self.respond_with(lambda handler, record: send_bytes(handler, b'{"value": "\xff\xfe"}'))
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertIn("UTF-8", str(raised.exception))

    def test_pathologically_nested_json_is_refused(self) -> None:
        deep = b"[" * 100_000 + b"]" * 100_000
        self.respond_with(lambda handler, record: send_bytes(handler, deep))
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertIn("JSON", str(raised.exception))
        self.assertIsNone(raised.exception.payload)

    def test_non_object_json_is_refused(self) -> None:
        self.respond_with(lambda handler, record: send_json(handler, [1, 2, 3]))
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertIn("object", str(raised.exception))
        self.assertEqual(raised.exception.payload, [1, 2, 3])

    def test_oversized_response_is_refused(self) -> None:
        big = b'"' + b"a" * MAX_RESPONSE_BYTES + b'"'
        self.respond_with(lambda handler, record: send_bytes(handler, big))
        with self.assertRaises(LocalMetaHarnessError) as raised:
            self.client.health()
        self.assertIn(str(MAX_RESPONSE_BYTES), str(raised.exception))
        self.assertIsNone(raised.exception.payload)

    def test_response_at_the_limit_is_accepted(self) -> None:
        length = MAX_RESPONSE_BYTES // 2
        body = json.dumps({"value": "a" * length}).encode("utf-8")
        self.assertLessEqual(len(body), MAX_RESPONSE_BYTES)
        self.respond_with(lambda handler, record: send_bytes(handler, body))
        self.assertEqual(len(self.client.health()["value"]), length)

    def test_timeout_is_refused(self) -> None:
        def slow(handler, record):
            time.sleep(0.4)
            send_json(handler, {"status": "ok"})

        self.respond_with(slow)
        impatient = LocalMetaHarnessClient(
            port=self.port, control_token=TOKEN, timeout_seconds=0.1
        )
        with self.assertRaises(LocalMetaHarnessError) as raised:
            impatient.health()
        self.assertIn("timed out", str(raised.exception))
        self.assertNotIn(TOKEN, f"{raised.exception!s} {raised.exception!r}")

    def test_connection_failure_is_clean(self) -> None:
        if self.ip_transport:
            dead = TestServer((LOCAL_HOST, 0), TestHandler)
            dead.state = ServerState()
            port = dead.server_port
            dead.server_close()
            client = LocalMetaHarnessClient(port=port, control_token=TOKEN, timeout_seconds=1.0)
            with self.assertRaises(LocalMetaHarnessError) as raised:
                client.health()
        else:
            with mock.patch.object(
                http.client, "HTTPConnection", side_effect=ConnectionRefusedError(61, "refused")
            ):
                with self.assertRaises(LocalMetaHarnessError) as raised:
                    self.client.health()
        self.assertIsNone(raised.exception.status)
        self.assertNotIn(TOKEN, f"{raised.exception!s} {raised.exception!r}")

    def test_control_token_never_appears_in_any_failure(self) -> None:
        oversized = b'"' + b"a" * (MAX_RESPONSE_BYTES + 1) + b'"'
        cases = {
            "http-error": (
                lambda handler, record: send_json(handler, {"error": "forbidden"}, status=403),
                5.0,
            ),
            "invalid-json": (lambda handler, record: send_bytes(handler, b"{oops"), 5.0),
            "oversized": (lambda handler, record: send_bytes(handler, oversized), 5.0),
            "timeout": (
                lambda handler, record: (time.sleep(0.4), send_json(handler, {"status": "ok"})),
                0.1,
            ),
        }
        for name, (responder, timeout) in cases.items():
            with self.subTest(case=name):
                self.respond_with(responder)
                client = LocalMetaHarnessClient(
                    port=self.port, control_token=TOKEN, timeout_seconds=timeout
                )
                with self.assertRaises(LocalMetaHarnessError) as raised:
                    client.health()
                self.assertNotIn(TOKEN, f"{raised.exception!s} {raised.exception!r}")


class ConstructorValidationTests(unittest.TestCase):
    def test_invalid_port(self) -> None:
        for port in (0, -1, 65536, True, "8765", None, 8.0):
            with self.subTest(port=port):
                with self.assertRaises(ValueError):
                    LocalMetaHarnessClient(port=port, control_token=TOKEN)

    def test_invalid_token(self) -> None:
        for token in ("", "has space", "line\nbreak", "tab\t", "é", "café", None, 42):
            with self.subTest(token=token):
                with self.assertRaises(ValueError) as raised:
                    LocalMetaHarnessClient(control_token=token)
                rendered = str(token)
                if rendered:
                    self.assertNotIn(rendered, str(raised.exception))

    def test_invalid_timeout(self) -> None:
        for timeout in (0, -1.0, float("inf"), float("nan"), True, "10", None):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    LocalMetaHarnessClient(control_token=TOKEN, timeout_seconds=timeout)

    def test_defaults(self) -> None:
        client = LocalMetaHarnessClient(control_token=TOKEN)
        self.assertEqual(client._port, 8765)
        self.assertEqual(client._timeout_seconds, 10.0)


if __name__ == "__main__":
    unittest.main()
