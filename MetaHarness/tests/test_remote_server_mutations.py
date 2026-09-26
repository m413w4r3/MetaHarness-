"""Mutation behaviour of the remote gateway: contracts, auth, relaying.

The gateway owns the external contract: it validates and translates here,
sends exactly one loopback request carrying the control token, never the
remote bearer token, and relays the local status and JSON body.  A real
loopback server is preferred; when the sandbox forbids local sockets a
memory transport drives the same handler, mirroring
``test_remote_server_read.py``.
"""

from __future__ import annotations

import http.client
import io
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.plan_recovery import MAX_REPLACEMENT_PLAN_BYTES
from metaharness.remote import (
    MAX_RESPONSE_BYTES,
    LocalMetaHarnessClient,
    create_remote_gateway,
)
from metaharness.remote import server as gateway_module
from metaharness.remote.auth import load_token_file

REMOTE_TOKEN = "remote-mutation-unit-test-token"
CONTROL_TOKEN = "control-mutation-unit-test-token"
RUN_ID = "run-2024-05-06"
UNAUTHORIZED = {"error": "unauthorized", "message": "authentication required"}
UNKNOWN_ROUTE = {"error": "not_found", "message": "unknown route"}
METHOD_REFUSED = {
    "error": "method_not_allowed",
    "message": "method not allowed for this route",
}
UPSTREAM_ERROR = {
    "error": "upstream_error",
    "message": "local MetaHarness request failed",
}
BAD_RUN_ID = {"error": "invalid_run_id", "message": "invalid run id"}
TOO_LARGE = {"error": "body_too_large", "message": "request body is too large"}
NOT_JSON = {"error": "unsupported_media_type", "message": "body must be application/json"}
BAD_FRAMING = {
    "error": "invalid_request",
    "message": "body must use Content-Length framing",
}
DUPLICATED_LENGTH = {
    "error": "invalid_request",
    "message": "body must declare one Content-Length",
}
BAD_TARGET = {"error": "invalid_request", "message": "invalid request target"}
BAD_JSON = {"error": "invalid_request", "message": "body must be valid JSON"}
TEST_TIMEOUT = 10
LOCAL_HOST = "127.0.0.1"
GATEWAY_MEMORY_PORT = 8771
UPSTREAM_MEMORY_PORT = 8765

CREATE_PATH = "/v1/runs"
APPROVAL_PATH = f"/v1/runs/{RUN_ID}/approval"
SCOPE_PATH = f"/v1/runs/{RUN_ID}/scope-approval"
RESUME_PATH = f"/v1/runs/{RUN_ID}/resume"
RECOVERY_PATH = f"/v1/runs/{RUN_ID}/recover-plan"
MUTATION_PATHS = (CREATE_PATH, APPROVAL_PATH, SCOPE_PATH, RESUME_PATH, RECOVERY_PATH)

LOCAL_CREATE = "/api/v1/runs"
LOCAL_APPROVAL = f"/api/v1/runs/{RUN_ID}/approval"
LOCAL_SCOPE = f"/api/v1/runs/{RUN_ID}/scope-approval"
LOCAL_RESUME = f"/api/v1/runs/{RUN_ID}/resume"
LOCAL_RECOVERY = f"/api/v1/runs/{RUN_ID}/recover-plan"
LOCAL_TARGETS = {
    CREATE_PATH: LOCAL_CREATE,
    APPROVAL_PATH: LOCAL_APPROVAL,
    SCOPE_PATH: LOCAL_SCOPE,
    RESUME_PATH: LOCAL_RESUME,
    RECOVERY_PATH: LOCAL_RECOVERY,
}

CREATE_RESULT = {
    "ok": True, "run_id": RUN_ID, "location": f"/runs/{RUN_ID}", "accepted": True,
}
APPROVAL_RESULT = {"ok": True, "decision": "APPROVE"}
SCOPE_RESULT = {"ok": True, "decision": "APPROVE", "scope_delta_sha256": "0" * 64}
RESUME_RESULT = {
    "ok": True, "run_id": RUN_ID, "location": f"/runs/{RUN_ID}", "accepted": True,
}
RECOVERY_RESULT = RESUME_RESULT

DEFAULT_RESPONSES = {
    ("POST", LOCAL_CREATE): (202, CREATE_RESULT),
    ("POST", LOCAL_APPROVAL): (200, APPROVAL_RESULT),
    ("POST", LOCAL_SCOPE): (200, SCOPE_RESULT),
    ("POST", LOCAL_RESUME): (202, RESUME_RESULT),
    ("POST", LOCAL_RECOVERY): (202, RECOVERY_RESULT),
}

VALID_BODIES: dict[str, dict[str, object]] = {
    CREATE_PATH: {"spec": "Implement the widget."},
    APPROVAL_PATH: {
        "decision": "APPROVE",
        "final_reviewer_profile": "final-reviewer",
        "step_profiles": {"S01": "implementer"},
    },
    SCOPE_PATH: {"decision": "APPROVE"},
    RESUME_PATH: {},
    RECOVERY_PATH: {"plan": "# META PLAN v2\n"},
}


class RecordedRequest:
    def __init__(
        self, method: str, path: str, headers: dict[str, str], body: bytes
    ) -> None:
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body


class UpstreamState:
    """Requests seen by the local stand-in plus its canned answers."""

    def __init__(self) -> None:
        self.down = False
        # Only the timeout test stalls the local server.
        self.delay = 0.0
        self.responses = dict(DEFAULT_RESPONSES)
        self._requests: list[RecordedRequest] = []
        self.lock = threading.Lock()

    def respond(self, method: str, path: str, status: int, payload: object) -> None:
        with self.lock:
            self.responses[(method, path)] = (status, payload)

    def record(self, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        with self.lock:
            self._requests.append(RecordedRequest(method, path, headers, body))

    def records(self) -> list[RecordedRequest]:
        with self.lock:
            return list(self._requests)

    def reset(self) -> None:
        with self.lock:
            self._requests.clear()
            self.responses = dict(DEFAULT_RESPONSES)


def upstream_response(state: UpstreamState, method: str, path: str) -> tuple[int, object]:
    target, _, _query = path.partition("?")
    with state.lock:
        answer = state.responses.get((method, target))
    if answer is not None:
        return answer
    return 404, {"error": "not found", "message": "not found"}


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length else 0
        body = self.rfile.read(length) if length else b""
        state = self.server.state  # type: ignore[attr-defined]
        state.record(
            method, self.path, {name.lower(): value for name, value in self.headers.items()}, body
        )
        if state.delay:
            time.sleep(state.delay)
        status, payload = upstream_response(state, method, self.path)
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except OSError:
            # A gateway that gave up (oversized body, timeout) closes first.
            pass


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
        self._writer.write(data)

    def settimeout(self, _timeout: object) -> None:
        return

    def setsockopt(self, *_args: object) -> None:
        return

    def close(self) -> None:
        return

    def getvalue(self) -> bytes:
        return self._writer.getvalue()


def _parse_request(raw: bytes) -> tuple[str, str, dict[str, str], bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _version = lines[0].split(" ", 2)
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return method, path, headers, body


def _upstream_bytes(state: UpstreamState, raw_request: bytes) -> bytes:
    if state.down:
        raise ConnectionRefusedError("upstream is down")
    method, path, headers, body = _parse_request(raw_request)
    state.record(method, path, headers, body)
    if state.delay:
        time.sleep(state.delay)
    status, payload = upstream_response(state, method, path)
    encoded = json.dumps(payload).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {http.client.responses.get(status, '')}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(encoded)}\r\n\r\n"
    )
    return head.encode("latin-1") + encoded


def _gateway_bytes(server: object, raw_request: bytes) -> bytes:
    """Drive one request through the real gateway handler, without a socket."""

    memory = _MemorySocket(raw_request)
    gateway_module._GatewayRequestHandler(memory, (LOCAL_HOST, 0), server)
    return memory.getvalue()


class _GatewayStub:
    """The server attributes the gateway handler reads, without a socket."""

    def __init__(
        self,
        *,
        remote_token: str,
        control_token: str,
        metaharness_port: int,
        local_client: LocalMetaHarnessClient,
    ) -> None:
        self.remote_token = remote_token
        self.control_token = control_token
        self.metaharness_port = metaharness_port
        self.local_client = local_client
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
        started = time.monotonic()
        raw = bytes(self._request)
        if self.port == self._server.server_port:
            served = _gateway_bytes(self._server, raw)
        else:
            served = _upstream_bytes(self._upstream, raw)
        if self.timeout is not None and time.monotonic() - started > self.timeout:
            raise TimeoutError("local server exceeded the client timeout")
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


class MutationCase(unittest.TestCase):
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
                local_client=LocalMetaHarnessClient(
                    port=self.upstream_port,
                    control_token=CONTROL_TOKEN,
                    timeout_seconds=TEST_TIMEOUT,
                ),
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
        method: str,
        path: str,
        *,
        token: str | None = REMOTE_TOKEN,
        body: bytes | None = None,
        declared_length: int | None = None,
        content_type: str | None = "application/json",
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> Response:
        connection = http.client.HTTPConnection(
            LOCAL_HOST, self.gateway_port, timeout=TEST_TIMEOUT
        )
        try:
            connection.putrequest(method, path, skip_accept_encoding=True)
            if token is not None:
                connection.putheader("Authorization", f"Bearer {token}")
            if content_type is not None:
                connection.putheader("Content-Type", content_type)
            for name, value in extra_headers:
                connection.putheader(name, value)
            if declared_length is not None:
                connection.putheader("Content-Length", str(declared_length))
            elif body is not None:
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

    def post(
        self,
        path: str,
        payload: object | None = None,
        *,
        raw: bytes | None = None,
        **kwargs,
    ) -> Response:
        if raw is None:
            raw = json.dumps({} if payload is None else payload).encode("utf-8")
        return self.request("POST", path, body=raw, **kwargs)

    def records(self) -> list[RecordedRequest]:
        return self.upstream.records()

    def assert_single_call(self, path: str) -> RecordedRequest:
        """Exactly one local call, on the single mapped target of *path*."""

        records = self.records()
        self.assertEqual(len(records), 1, [record.path for record in records])
        self.assertEqual(records[0].method, "POST")
        self.assertEqual(records[0].path, LOCAL_TARGETS[path])
        return records[0]

    def local_body(self, path: str) -> object:
        return json.loads(self.assert_single_call(path).body.decode("utf-8"))

    def assert_upstream_untouched(self) -> None:
        self.assertEqual(self.records(), [])


# ---------------------------------------------------------------------------
# CREATE RUN
# ---------------------------------------------------------------------------


class CreateRunTests(MutationCase):
    def test_create_run_forwards_the_allowed_payload_and_relays_202(self) -> None:
        payload = {
            "spec": "Implement the widget.",
            "run_id": RUN_ID,
            "planner_profile": "planner",
            "mechanical_profile": "mechanical",
            "reasoning_profile": "reasoning",
            "agentic_profile": "agentic",
            "final_reviewer_profile": "final-reviewer",
            "semantic_reviser_profile": "reviser",
            "check_repair_profile": "repairer",
            "semantic_revision_enabled": True,
            "max_check_repair_attempts": 2,
            "max_correction_cycles": 1,
            "decomposition": "auto",
            "execution_mode_policy": "auto",
            "single_step_max_mutable_paths": 4,
            "staged_step_max_mutable_paths": 8,
            "repair_scope_policy": "strict",
            "repair_scope_max_added_paths": 2,
        }
        response = self.post(CREATE_PATH, payload)
        self.assertEqual(response.status, 202)
        self.assertEqual(response.payload, CREATE_RESULT)
        self.assertEqual(self.local_body(CREATE_PATH), payload)

    def test_unknown_create_field_is_refused_before_any_local_call(self) -> None:
        for extra in ("unknown", "spec_extra", "step_profiles", "approved"):
            with self.subTest(field=extra):
                self.upstream.reset()
                response = self.post(CREATE_PATH, {"spec": "Do it.", extra: 1})
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {"error": "invalid_request", "message": "unknown request field"},
                )
                self.assert_upstream_untouched()

    def test_missing_or_empty_spec_is_refused(self) -> None:
        for payload in ({}, {"spec": ""}, {"spec": "   "}, {"spec": 3}, {"spec": None}):
            with self.subTest(payload=payload):
                response = self.post(CREATE_PATH, payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {
                        "error": "invalid_request",
                        "message": "spec must be a non-empty string",
                    },
                )
        self.assert_upstream_untouched()

    def test_oversized_create_body_is_413_before_any_local_call(self) -> None:
        response = self.post(CREATE_PATH, raw=b"x" * (64 * 1024 + 1))
        self.assertEqual(response.status, 413)
        self.assertEqual(response.payload, TOO_LARGE)
        self.assert_upstream_untouched()

    def test_body_must_be_one_json_object(self) -> None:
        cases = (
            (b"", "body must be valid JSON"),
            (b"not json", "body must be valid JSON"),
            (b'{"spec": "a", "spec": "b"}', "body must be valid JSON"),
            (b"[]", "body must be a JSON object"),
            (b'"spec"', "body must be a JSON object"),
            (b"null", "body must be a JSON object"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw):
                response = self.post(CREATE_PATH, raw=raw)
                self.assertEqual(response.status, 400)
                self.assertEqual(response.payload, {"error": "invalid_request", "message": message})
        self.assert_upstream_untouched()

    def test_local_conflict_is_relayed(self) -> None:
        conflict = {"error": "run already exists", "message": "run already exists"}
        self.upstream.respond(
            "POST", LOCAL_CREATE, 409, conflict
        )
        response = self.post(CREATE_PATH, {"spec": "Do it.", "run_id": RUN_ID})
        self.assertEqual(response.status, 409)
        self.assertEqual(response.payload, conflict)
        self.assert_single_call(CREATE_PATH)


# ---------------------------------------------------------------------------
# APPROVAL
# ---------------------------------------------------------------------------


class ApprovalTests(MutationCase):
    def test_approve_translates_step_profiles_to_local_field_names(self) -> None:
        response = self.post(
            APPROVAL_PATH,
            {
                "decision": "APPROVE",
                "final_reviewer_profile": "reviewer",
                "semantic_reviser_profile": "reviser",
                "check_repair_profile": "repairer",
                "step_profiles": {"S01": "implementer-a", "S42": "implementer-b"},
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload, APPROVAL_RESULT)
        self.assertEqual(
            self.local_body(APPROVAL_PATH),
            {
                "decision": "APPROVE",
                "final_reviewer_profile": "reviewer",
                "semantic_reviser_profile": "reviser",
                "check_repair_profile": "repairer",
                "step_profile__S01": "implementer-a",
                "step_profile__S42": "implementer-b",
            },
        )

    def test_reject_is_forwarded_alone(self) -> None:
        response = self.post(
            APPROVAL_PATH,
            {
                "decision": "REJECT",
                "final_reviewer_profile": "reviewer",
                "step_profiles": {"S01": "implementer"},
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(self.local_body(APPROVAL_PATH), {"decision": "REJECT"})

    def test_null_profile_fields_are_omitted(self) -> None:
        response = self.post(
            APPROVAL_PATH,
            {
                "decision": "APPROVE",
                "final_reviewer_profile": "reviewer",
                "semantic_reviser_profile": None,
                "check_repair_profile": None,
                "step_profiles": {"S01": "implementer"},
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(
            self.local_body(APPROVAL_PATH),
            {
                "decision": "APPROVE",
                "final_reviewer_profile": "reviewer",
                "step_profile__S01": "implementer",
            },
        )

    def test_unsafe_step_ids_are_refused_before_any_local_call(self) -> None:
        unsafe = (
            "S1", "S100", "s01", "S01 ", " S01", "X01", "", "S00", "S01\n",
            "../S01", "step_profile__S01",
        )
        for step_id in unsafe:
            with self.subTest(step_id=step_id):
                response = self.post(
                    APPROVAL_PATH,
                    {
                        "decision": "APPROVE",
                        "final_reviewer_profile": "reviewer",
                        "step_profiles": {step_id: "implementer"},
                    },
                )
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {"error": "invalid_request", "message": "invalid step profile field"},
                )
        self.assert_upstream_untouched()

    def test_invalid_step_profile_containers_and_values_are_refused(self) -> None:
        cases = (
            {"step_profiles": ["S01"]},
            {"step_profiles": "S01"},
            {"step_profiles": {"S01": 42}},
            {"step_profiles": {"S01": ""}},
            {"step_profiles": {"S01": None}},
            {"final_reviewer_profile": 3},
            {"final_reviewer_profile": ""},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                response = self.post(
                    APPROVAL_PATH,
                    {"decision": "APPROVE", "final_reviewer_profile": "reviewer", **extra},
                )
                self.assertEqual(response.status, 400)
        self.assert_upstream_untouched()

    def test_unknown_field_and_invalid_decision_are_refused(self) -> None:
        cases = (
            ({"decision": "APPROVE", "profile_hint": "x"}, "unknown approval field"),
            ({"decision": "MAYBE"}, "decision must be APPROVE or REJECT"),
            ({}, "decision must be APPROVE or REJECT"),
            ({"decision": 3}, "decision must be APPROVE or REJECT"),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                response = self.post(APPROVAL_PATH, payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload, {"error": "invalid_request", "message": message}
                )
        self.assert_upstream_untouched()

    def test_local_refusal_is_relayed_with_its_status(self) -> None:
        refusal = {
            "error": "run is not awaiting plan approval",
            "message": "run is not awaiting plan approval",
        }
        self.upstream.respond("POST", LOCAL_APPROVAL, 409, refusal)
        response = self.post(APPROVAL_PATH, VALID_BODIES[APPROVAL_PATH])
        self.assertEqual(response.status, 409)
        self.assertEqual(response.payload, refusal)
        self.assert_single_call(APPROVAL_PATH)


# ---------------------------------------------------------------------------
# SCOPE APPROVAL
# ---------------------------------------------------------------------------


class ScopeApprovalTests(MutationCase):
    def test_scope_decisions_are_forwarded_exactly(self) -> None:
        for decision in ("APPROVE", "REJECT"):
            with self.subTest(decision=decision):
                self.upstream.reset()
                response = self.post(SCOPE_PATH, {"decision": decision})
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.payload,
                    {"ok": True, "decision": "APPROVE", "scope_delta_sha256": "0" * 64},
                )
                self.assertEqual(self.local_body(SCOPE_PATH), {"decision": decision})

    def test_scope_body_must_be_exactly_one_decision(self) -> None:
        cases = (
            {}, {"decision": "APPROVE", "reason": "ok"},
            {"decision": "approve"}, {"decision": None},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                response = self.post(SCOPE_PATH, payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {
                        "error": "invalid_request",
                        "message": "decision must be APPROVE or REJECT",
                    },
                )
        self.assert_upstream_untouched()


# ---------------------------------------------------------------------------
# RESUME
# ---------------------------------------------------------------------------


class ResumeTests(MutationCase):
    def test_resume_forwards_exactly_an_empty_object(self) -> None:
        response = self.post(RESUME_PATH, {})
        self.assertEqual(response.status, 202)
        self.assertEqual(response.payload, RESUME_RESULT)
        self.assertEqual(self.assert_single_call(RESUME_PATH).body, b"{}")

    def test_resume_refuses_any_other_body(self) -> None:
        for payload in ({"reason": "retry"}, {"force": False}, {"": ""}):
            with self.subTest(payload=payload):
                response = self.post(RESUME_PATH, payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {
                        "error": "invalid_request",
                        "message": "resume body must be an empty JSON object",
                    },
                )
        self.assert_upstream_untouched()


# ---------------------------------------------------------------------------
# RECOVER PLAN
# ---------------------------------------------------------------------------


class RecoverPlanTests(MutationCase):
    def test_recover_plan_forwards_the_plan(self) -> None:
        plan = "# META PLAN v2\n\nS01 do the thing\n"
        response = self.post(RECOVERY_PATH, {"plan": plan})
        self.assertEqual(response.status, 202)
        self.assertEqual(response.payload, RECOVERY_RESULT)
        self.assertEqual(self.local_body(RECOVERY_PATH), {"plan": plan})

    def test_recover_plan_accepts_a_body_above_the_standard_limit(self) -> None:
        plan = "# META PLAN v2\n" + "S01 step line\n" * 6000
        self.assertGreater(len(plan), 64 * 1024)
        response = self.post(RECOVERY_PATH, {"plan": plan})
        self.assertEqual(response.status, 202)
        self.assertEqual(self.local_body(RECOVERY_PATH), {"plan": plan})

    def test_recover_plan_limit_is_compatible_with_the_plan_limit(self) -> None:
        # The local server reserves 4 * MAX_REPLACEMENT_PLAN_BYTES for the
        # JSON encoding of one plan; the gateway must not accept less.
        self.assertGreaterEqual(
            gateway_module._MAX_RECOVERY_BODY_BYTES, 4 * MAX_REPLACEMENT_PLAN_BYTES
        )
        response = self.request(
            "POST",
            RECOVERY_PATH,
            declared_length=gateway_module._MAX_RECOVERY_BODY_BYTES + 1,
        )
        self.assertEqual(response.status, 413)
        self.assertEqual(response.payload, TOO_LARGE)
        self.assert_upstream_untouched()

    def test_recover_plan_refuses_other_bodies(self) -> None:
        for payload in ({}, {"plan": 3}, {"plan": "x", "extra": 1}):
            with self.subTest(payload=payload):
                response = self.post(RECOVERY_PATH, payload)
                self.assertEqual(response.status, 400)
                self.assertEqual(
                    response.payload,
                    {
                        "error": "invalid_request",
                        "message": "body must contain exactly one plan string",
                    },
                )
        self.assert_upstream_untouched()


class RequestContractTests(MutationCase):
    """Media type and framing rules of a mutation body."""

    def test_post_requires_a_json_content_type(self) -> None:
        for content_type in (
            "application/json",
            "Application/JSON",
            "application/json; charset=utf-8",
        ):
            with self.subTest(content_type=content_type):
                self.upstream.reset()
                response = self.post(
                    CREATE_PATH, {"spec": "Do it."}, content_type=content_type
                )
                self.assertEqual(response.status, 202)
                self.assert_single_call(CREATE_PATH)
        for content_type in (
            None,
            "",
            "text/plain",
            "application/x-www-form-urlencoded",
            "application/jsonx",
            "application/ld+json",
        ):
            with self.subTest(content_type=content_type):
                self.upstream.reset()
                response = self.request("POST", CREATE_PATH, content_type=content_type)
                self.assertEqual(response.status, 415)
                self.assertEqual(response.payload, NOT_JSON)
                self.assert_upstream_untouched()
        # A read route is unchanged: no media type is required for GET, which
        # is routed (and answered by the stand-in) instead of refused.
        self.assertNotEqual(
            self.request("GET", "/v1/health", content_type=None).status, 415
        )

    def test_ambiguous_body_framing_is_refused_before_any_local_call(self) -> None:
        cases = (
            ((("Transfer-Encoding", "chunked"),), None, BAD_FRAMING),
            ((("Transfer-Encoding", "chunked"),), 0, BAD_FRAMING),
            ((("Content-Length", "0"),), 0, DUPLICATED_LENGTH),
        )
        for extra, declared, expected in cases:
            with self.subTest(extra=extra, declared=declared):
                self.upstream.reset()
                response = self.request(
                    "POST", CREATE_PATH, declared_length=declared, extra_headers=extra
                )
                self.assertEqual(response.status, 400)
                self.assertEqual(response.payload, expected)
                self.assert_upstream_untouched()

    def test_malformed_request_targets_are_refused_without_echo(self) -> None:
        # http.client refuses to serialise such a target; the gateway must
        # answer bounded JSON instead of letting urlsplit's error escape.
        for target in ("http://[", "http://[::1"):
            with self.subTest(target=target):
                raw = self.raw_exchange(
                    f"POST {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    f"Authorization: Bearer {REMOTE_TOKEN}\r\n"
                    "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".encode()
                )
                self.assertTrue(raw.startswith(b"HTTP/1.0 400"), raw[:40])
                self.assertIn(b"invalid_request", raw)
                self.assertNotIn(target.encode(), raw)
        self.assert_upstream_untouched()

    def test_a_pathologically_nested_body_is_refused(self) -> None:
        nested = b'{"plan": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
        self.assertLess(len(nested), gateway_module._MAX_RECOVERY_BODY_BYTES)
        response = self.post(RECOVERY_PATH, raw=nested)
        self.assertEqual(response.status, 400)
        self.assertEqual(response.payload, BAD_JSON)
        self.assert_upstream_untouched()


class LocalBoundTests(MutationCase):
    """The loopback mutation leg is bounded in time and in size."""

    def test_an_oversized_local_response_is_never_relayed(self) -> None:
        self.upstream.respond(
            "POST", LOCAL_CREATE, 202, {"blob": "a" * MAX_RESPONSE_BYTES}
        )
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 502)
        self.assertEqual(response.payload, UPSTREAM_ERROR)
        self.assert_single_call(CREATE_PATH)

    def test_a_stalled_local_server_is_bounded_and_never_retried(self) -> None:
        self.upstream.delay = 0.4
        self.gateway.local_client = LocalMetaHarnessClient(  # type: ignore[attr-defined]
            port=self.upstream_port, control_token=CONTROL_TOKEN, timeout_seconds=0.05
        )
        started = time.monotonic()
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 502)
        self.assertEqual(response.payload, UPSTREAM_ERROR)
        self.assertLess(time.monotonic() - started, 5)
        self.assert_single_call(CREATE_PATH)

    def test_a_surrogate_escape_from_localhost_is_relayed_safely(self) -> None:
        self.upstream.respond("POST", LOCAL_CREATE, 202, {"run_id": "\ud800"})
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 202)
        self.assertEqual(response.payload, {"run_id": "\ud800"})


# ---------------------------------------------------------------------------
# Authentication, routing and the two tokens
# ---------------------------------------------------------------------------


class MutationAuthorizationTests(MutationCase):
    def test_every_mutation_requires_the_bearer_token(self) -> None:
        for path in MUTATION_PATHS:
            for token in (None, "", "wrong-token"):
                with self.subTest(path=path, token=token):
                    response = self.post(path, VALID_BODIES[path], token=token)
                    self.assertEqual(response.status, 401)
                    self.assertEqual(response.payload, UNAUTHORIZED)
        self.assert_upstream_untouched()

    def test_mutation_routes_are_not_read_routes(self) -> None:
        for path in MUTATION_PATHS[1:]:
            with self.subTest(path=path):
                response = self.request("GET", path)
                self.assertEqual(response.status, 404)
                self.assertEqual(response.payload, UNKNOWN_ROUTE)
        self.assert_upstream_untouched()

    def test_post_is_refused_on_read_only_routes(self) -> None:
        paths = (
            "/v1/health",
            "/v1/config",
            "/v1/model-profiles",
            f"/v1/runs/{RUN_ID}",
            f"/v1/runs/{RUN_ID}/progress",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.post(path, {})
                self.assertEqual(response.status, 405)
                self.assertEqual(response.payload, METHOD_REFUSED)
        self.assert_upstream_untouched()

    def test_unknown_post_routes_are_404(self) -> None:
        paths = (
            "/",
            "/v1",
            "/v1/runs/",
            f"/v1/runs/{RUN_ID}/approval/",
            f"/v1/runs/{RUN_ID}/live",
            f"/v1/runs/{RUN_ID}/artifact",
            "/v2/runs",
        )
        for path in paths:
            with self.subTest(path=path):
                response = self.post(path, {})
                self.assertEqual(response.status, 404)
                self.assertEqual(response.payload, UNKNOWN_ROUTE)
        self.assert_upstream_untouched()

    def test_unsafe_run_ids_are_refused_on_mutations(self) -> None:
        # Only encoded forms: http.client refuses a request-target with a raw
        # space or backslash, and the gateway must answer what it receives.
        for run_id in ("..", ".", "%2e%2e", "a%2Fb", "a%5Cb", "run%20id", "a%00b"):
            with self.subTest(run_id=run_id):
                response = self.post(f"/v1/runs/{run_id}/approval", {"decision": "APPROVE"})
                self.assertEqual(response.status, 400)
                self.assertEqual(response.payload, BAD_RUN_ID)
        self.assert_upstream_untouched()


class ControlTokenTests(MutationCase):
    def test_every_mutation_sends_the_control_token_to_localhost(self) -> None:
        for path in MUTATION_PATHS:
            with self.subTest(path=path):
                self.upstream.reset()
                response = self.post(path, VALID_BODIES[path])
                self.assertIn(response.status, (200, 202))
                record = self.assert_single_call(path)
                self.assertEqual(record.headers.get("x-metaharness-token"), CONTROL_TOKEN)
                self.assertEqual(record.headers.get("host"), f"{LOCAL_HOST}:{self.upstream_port}")

    def test_the_remote_token_never_reaches_the_local_server(self) -> None:
        for path in MUTATION_PATHS:
            with self.subTest(path=path):
                self.upstream.reset()
                self.post(path, VALID_BODIES[path])
                record = self.assert_single_call(path)
                self.assertNotIn("authorization", record.headers)
                for name, value in record.headers.items():
                    self.assertNotIn(REMOTE_TOKEN, value, name)
                self.assertNotIn(REMOTE_TOKEN.encode("utf-8"), record.body)

    def test_every_mutation_costs_exactly_one_local_call(self) -> None:
        for path in MUTATION_PATHS:
            with self.subTest(path=path):
                self.upstream.reset()
                self.post(path, VALID_BODIES[path])
                self.assert_single_call(path)

    def test_a_refused_mutation_is_never_retried(self) -> None:
        refusal = {"error": "request could not be served", "message": "request could not be served"}
        self.upstream.respond("POST", LOCAL_CREATE, 503, refusal)
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 503)
        self.assertEqual(response.payload, refusal)
        self.assertEqual(len(self.records()), 1)

    def test_responses_repeat_neither_token(self) -> None:
        for path in MUTATION_PATHS:
            with self.subTest(path=path):
                response = self.post(path, VALID_BODIES[path])
                self.assertNotIn(REMOTE_TOKEN, response.text)
                self.assertNotIn(CONTROL_TOKEN, response.text)
                for value in response.headers.values():
                    self.assertNotIn(REMOTE_TOKEN, value)
                    self.assertNotIn(CONTROL_TOKEN, value)


class LocalStatusRelayTests(MutationCase):
    def test_local_error_statuses_are_relayed_with_their_json(self) -> None:
        cases = (
            (400, {
                "error": "decision must be APPROVE or REJECT",
                "message": "decision must be APPROVE or REJECT",
            }),
            (403, {"error": "mutation token required", "message": "mutation token required"}),
            (404, {"error": "not found", "message": "not found"}),
            (409, {"error": "run is not resumable", "message": "run is not resumable"}),
            (413, {"error": "request body is too large", "message": "request body is too large"}),
            (500, {"error": "internal error", "message": "internal error"}),
            (503, {
                "error": "request could not be served",
                "message": "request could not be served",
            }),
        )
        for status, payload in cases:
            with self.subTest(status=status):
                self.upstream.reset()
                self.upstream.respond("POST", LOCAL_RESUME, status, payload)
                response = self.post(RESUME_PATH, {})
                self.assertEqual(response.status, status)
                self.assertEqual(response.payload, payload)
                self.assert_single_call(RESUME_PATH)

    def test_an_unexpected_local_status_is_502(self) -> None:
        for status in (201, 204, 302, 418):
            with self.subTest(status=status):
                self.upstream.reset()
                self.upstream.respond("POST", LOCAL_CREATE, status, {"ok": True})
                response = self.post(CREATE_PATH, {"spec": "Do it."})
                self.assertEqual(response.status, 502)
                self.assertEqual(response.payload, UPSTREAM_ERROR)

    def test_a_non_object_local_body_is_502(self) -> None:
        self.upstream.respond("POST", LOCAL_CREATE, 202, ["not", "an", "object"])
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 502)
        self.assertEqual(response.payload, UPSTREAM_ERROR)

    def test_an_unreachable_local_server_is_502(self) -> None:
        if self.upstream_server is not None:
            self.upstream_server.shutdown()
            self.upstream_server.server_close()
        else:
            self.upstream.down = True
        response = self.post(CREATE_PATH, {"spec": "Do it."})
        self.assertEqual(response.status, 502)
        self.assertEqual(response.payload, UPSTREAM_ERROR)


if __name__ == "__main__":
    unittest.main()
