"""Read-only HTTP gateway in front of a local MetaHarness server.

The gateway always binds ``127.0.0.1``: network exposure belongs to Tailscale
Serve, so no host is configurable and no CORS header is ever emitted.  Every
request, ``GET`` included, must carry ``Authorization: Bearer <remote token>``
read from ``remote_token_file``.  The request line is never logged, echoed or
forwarded: each allowlisted ``/v1`` route maps to exactly one MetaHarness
target, and an unknown route or method never reaches the local server.

This phase is read-only.  The control token is loaded at startup so a broken
deployment fails before the first request, and is held for the mutation routes
that are not implemented yet; it is never sent to MetaHarness and never leaves
the process.
"""

from __future__ import annotations

import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .auth import bearer_token_from_header, load_token_file, token_matches
from .client import LOCAL_HOST, MAX_RESPONSE_BYTES

DEFAULT_METAHARNESS_PORT = 8765

# A longer offset cannot be an index into any run and would make int() raise.
_MAX_OFFSET_DIGITS = 18
_UPSTREAM_TIMEOUT_SECONDS = 10.0
_FIXED_TARGETS = {
    "/v1/health": "/api/v1/health",
    "/v1/config": "/api/v1/config",
    "/v1/model-profiles": "/api/v1/model-profiles",
    "/v1/runs": "/api/v1/runs",
}
# Sending any of these could escape the single path component the local
# server expects; run-id policy beyond that stays owned by MetaHarness.
_RUN_ID_REFUSED = ("/", "\\", "..", "%", "\x00")
_UNAUTHORIZED_PAYLOAD = {"error": "unauthorized", "message": "authentication required"}


class _RouteError(Exception):
    """A request the gateway refuses before contacting MetaHarness."""

    def __init__(self, status: int, error: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message


class _UpstreamError(RuntimeError):
    """The local MetaHarness server did not answer with bounded JSON."""


def _upstream_target(path: str, query: str) -> str:
    """Return the single MetaHarness target of an allowlisted ``/v1`` route."""

    fixed = _FIXED_TARGETS.get(path)
    if fixed is not None:
        return fixed
    parts = path.split("/")
    if len(parts) == 4 and parts[:3] == ["", "v1", "runs"]:
        return f"/api/v1/runs/{_run_id(parts[3])}"
    if len(parts) == 5 and parts[:3] == ["", "v1", "runs"] and parts[4] == "progress":
        run_id = _run_id(parts[3])
        return f"/api/v1/runs/{run_id}/progress?offset={_offset(query)}"
    raise _RouteError(404, "not_found", "unknown route")


def _run_id(value: str) -> str:
    """Validate one run-id path component without echoing it anywhere."""

    if not value:
        raise _RouteError(404, "not_found", "unknown route")
    if value == "." or any(
        marker in value for marker in _RUN_ID_REFUSED
    ) or any(not 0x21 <= ord(character) <= 0x7E for character in value):
        raise _RouteError(400, "invalid_run_id", "invalid run id")
    return value


def _offset(query: str) -> int:
    """Return the single ASCII ``offset`` parameter of a progress request."""

    values = parse_qs(query, keep_blank_values=True).get("offset", [])
    if len(values) != 1 or not _ascii_digits(values[0]):
        raise _RouteError(400, "invalid_offset", "offset must be a non-negative integer")
    return int(values[0])


def _ascii_digits(value: str) -> bool:
    # ``str.isdigit`` accepts non-ASCII digits; the offset must be plain ASCII
    # and short enough that int() cannot refuse it.
    return bool(value) and len(value) <= _MAX_OFFSET_DIGITS and all(
        "0" <= character <= "9" for character in value
    )


def _upstream_json(port: int, target: str) -> tuple[int, object]:
    """Return ``(status, decoded JSON)`` of one loopback MetaHarness ``GET``."""

    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(
            LOCAL_HOST, port, timeout=_UPSTREAM_TIMEOUT_SECONDS
        )
        connection.request("GET", target, headers={"Accept": "application/json"})
        response = connection.getresponse()
        status = response.status
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, http.client.HTTPException, ValueError):
        # The message of these exceptions may quote request data; only a fixed
        # description is propagated.
        raise _UpstreamError("local MetaHarness request failed") from None
    finally:
        if connection is not None:
            connection.close()

    if len(raw) > MAX_RESPONSE_BYTES:
        raise _UpstreamError("local MetaHarness response exceeds the size limit")
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _UpstreamError("local MetaHarness response is not valid JSON") from None


class _RemoteGatewayServer(ThreadingHTTPServer):
    """Loopback-only HTTP server holding the gateway configuration."""

    remote_token: str
    control_token: str
    metaharness_port: int


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    """One remote request; every response is JSON.

    ``protocol_version`` stays at the HTTP/1.0 default: request bodies are
    never read, so a kept-alive connection could not be resynchronised.
    ``default_request_version`` is HTTP/1.0 as well, so a legacy or malformed
    request line never receives an HTTP/0.9 body without status line.
    """

    server: _RemoteGatewayServer
    default_request_version = "HTTP/1.0"

    def log_message(self, *_args: object) -> None:
        # BaseHTTPRequestHandler logs the request line, URL included; the
        # gateway stays silent about every request.
        return

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # BaseHTTPRequestHandler answers malformed request lines with an HTML
        # body echoing them; answer JSON that carries no request data instead.
        self._json(
            code,
            {
                "error": "request_error",
                "message": http.client.responses.get(code, "request could not be served"),
            },
        )

    def __getattr__(self, name: str) -> object:
        # Verbs BaseHTTPRequestHandler does not implement would otherwise be
        # answered with 501; this gateway answers 405 to every non-GET verb.
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    def do_GET(self) -> None:
        if not self._authorized():
            self._json(401, _UNAUTHORIZED_PAYLOAD)
            return
        parsed = urlsplit(self.path)
        try:
            target = _upstream_target(parsed.path, parsed.query)
        except _RouteError as error:
            self._json(error.status, {"error": error.error, "message": error.message})
            return
        try:
            status, payload = _upstream_json(self.server.metaharness_port, target)
        except _UpstreamError:
            self._json(
                502,
                {"error": "upstream_error", "message": "local MetaHarness request failed"},
            )
            return
        self._json(status, payload)

    def _method_not_allowed(self) -> None:
        # Authentication is checked before the method so an unauthenticated
        # caller learns nothing about the routes this gateway serves.
        if not self._authorized():
            self._json(401, _UNAUTHORIZED_PAYLOAD)
            return
        self._json(405, {"error": "method_not_allowed", "message": "only GET is supported"})

    def _authorized(self) -> bool:
        values = self.headers.get_all("Authorization") or []
        if len(values) != 1:
            return False
        return token_matches(
            bearer_token_from_header(values[0]), self.server.remote_token
        )

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if getattr(self, "command", None) == "HEAD":
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def create_remote_gateway(
    *,
    port: int,
    remote_token_file: str | Path,
    control_token_file: str | Path,
    metaharness_port: int = DEFAULT_METAHARNESS_PORT,
) -> ThreadingHTTPServer:
    """Bind the loopback gateway and return it without serving it.

    The host is the module constant ``127.0.0.1`` and is not a parameter.
    ``port`` 0 selects an ephemeral loopback port.  Both token files are read
    eagerly so a broken deployment fails before the first request.
    """

    remote_token = load_token_file(remote_token_file)
    control_token = load_token_file(control_token_file)
    server = _RemoteGatewayServer(
        (LOCAL_HOST, _validated_port(port, allow_zero=True)), _GatewayRequestHandler
    )
    server.remote_token = remote_token
    server.control_token = control_token
    server.metaharness_port = _validated_port(metaharness_port)
    return server


def serve_remote_gateway(
    *,
    port: int,
    remote_token_file: str | Path,
    control_token_file: str | Path,
    metaharness_port: int = DEFAULT_METAHARNESS_PORT,
) -> None:
    """Serve the read-only gateway on ``127.0.0.1:port`` until interrupted."""

    server = create_remote_gateway(
        port=port,
        remote_token_file=remote_token_file,
        control_token_file=control_token_file,
        metaharness_port=metaharness_port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _validated_port(port: int, *, allow_zero: bool = False) -> int:
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be an integer")
    if not (0 if allow_zero else 1) <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return port


__all__ = [
    "DEFAULT_METAHARNESS_PORT",
    "create_remote_gateway",
    "serve_remote_gateway",
]
