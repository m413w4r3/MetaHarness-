"""HTTP gateway in front of a local MetaHarness server.

The gateway always binds ``127.0.0.1``: network exposure belongs to Tailscale
Serve, so no host is configurable and no CORS header is ever emitted.  Every
request, ``GET`` included, must carry ``Authorization: Bearer <remote token>``
read from ``remote_token_file``.  The request line is never logged, echoed or
forwarded: each allowlisted ``/v1`` route maps to exactly one MetaHarness
target, and an unknown route or method never reaches the local server.

The observation routes are read-only.  The five mutation routes are validated
and translated here, then sent to MetaHarness by exactly one loopback request
carrying ``X-MetaHarness-Token``: the remote bearer token is never forwarded,
no mutation is ever retried, and the local status and JSON value are relayed
unchanged (non-ASCII text may be escaped, never dropped).
"""

from __future__ import annotations

import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import SplitResult, parse_qs, urlsplit

from ..plan_recovery import MAX_REPLACEMENT_PLAN_BYTES
from ..step_ids import is_step_id
from .auth import bearer_token_from_header, load_token_file, token_matches
from .client import (
    LOCAL_HOST,
    MAX_RESPONSE_BYTES,
    LocalMetaHarnessClient,
    LocalMetaHarnessError,
)

DEFAULT_METAHARNESS_PORT = 8765

# A longer offset cannot be an index into any run and would make int() raise.
_MAX_OFFSET_DIGITS = 18
_UPSTREAM_TIMEOUT_SECONDS = 10.0
# Request bodies of the mutation routes: 64 KiB, and the room the local
# server itself reserves for a replacement plan after JSON encoding.
_MAX_BODY_BYTES = 64 * 1024
_MAX_RECOVERY_BODY_BYTES = 4 * MAX_REPLACEMENT_PLAN_BYTES
_FIXED_TARGETS = {
    "/v1/health": "/api/v1/health",
    "/v1/config": "/api/v1/config",
    "/v1/model-profiles": "/api/v1/model-profiles",
    "/v1/runs": "/api/v1/runs",
}
_MUTATION_SUFFIXES = ("approval", "scope-approval", "resume", "recover-plan")
# The statuses a mutation relays from MetaHarness; any other status, and any
# non-object body, is reported as an upstream failure instead.
_RELAYED_STATUSES = frozenset({200, 202, 400, 403, 404, 409, 413, 500, 503})
_CREATE_FIELDS = frozenset({
    "spec", "run_id", "planner_profile", "mechanical_profile",
    "reasoning_profile", "agentic_profile", "final_reviewer_profile",
    "semantic_reviser_profile", "check_repair_profile",
    "semantic_revision_enabled", "max_check_repair_attempts",
    "max_review_repair_cycles", "decomposition", "execution_mode_policy",
    "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
    "repair_scope_policy", "repair_scope_max_added_paths",
})
_APPROVAL_FIELDS = frozenset({
    "decision", "final_reviewer_profile", "semantic_reviser_profile",
    "check_repair_profile", "step_profiles",
})
_APPROVAL_PROFILE_FIELDS = (
    "final_reviewer_profile", "semantic_reviser_profile", "check_repair_profile",
)
_DECISIONS = ("APPROVE", "REJECT")
# Sending any of these could escape the single path component the local
# server expects; run-id policy beyond that stays owned by MetaHarness.
_RUN_ID_REFUSED = ("/", "\\", "..", "%", "\x00")
_UNAUTHORIZED_PAYLOAD = {"error": "unauthorized", "message": "authentication required"}
_UPSTREAM_ERROR_PAYLOAD = {
    "error": "upstream_error",
    "message": "local MetaHarness request failed",
}
_METHOD_MESSAGE = "method not allowed for this route"


class _RouteError(Exception):
    """A request the gateway refuses before contacting MetaHarness."""

    def __init__(self, status: int, error: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message


class _UpstreamError(RuntimeError):
    """The local MetaHarness server did not answer with bounded JSON."""


def _split_target(value: str) -> SplitResult:
    """Split one request target; a malformed target is refused, never echoed."""

    try:
        return urlsplit(value)
    except ValueError:
        # ``urlsplit`` refuses some bracketed hosts and its message may quote
        # the target it was given; only a fixed description is propagated.
        raise _RouteError(400, "invalid_request", "invalid request target") from None


def _json_media_type(value: str | None) -> bool:
    """True when a ``Content-Type`` header names the JSON media type."""

    if not isinstance(value, str):
        return False
    return value.split(";", 1)[0].strip().lower() == "application/json"


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


def _mutation_target(path: str) -> tuple[str, str, int]:
    """Return ``(action, run_id, max_body_bytes)`` of a ``POST`` route."""

    if path == "/v1/runs":
        return "create", "", _MAX_BODY_BYTES
    parts = path.split("/")
    if len(parts) == 5 and parts[:3] == ["", "v1", "runs"] and parts[4] in _MUTATION_SUFFIXES:
        action = parts[4]
        limit = _MAX_RECOVERY_BODY_BYTES if action == "recover-plan" else _MAX_BODY_BYTES
        return action, _run_id(parts[3]), limit
    if _read_route(path):
        raise _RouteError(405, "method_not_allowed", _METHOD_MESSAGE)
    raise _RouteError(404, "not_found", "unknown route")


def _read_route(path: str) -> bool:
    """True for the path shapes the read-only ``GET`` routes answer."""

    if path in _FIXED_TARGETS:
        return True
    parts = path.split("/")
    if len(parts) == 4 and parts[:3] == ["", "v1", "runs"]:
        return bool(parts[3])
    return (
        len(parts) == 5
        and parts[:3] == ["", "v1", "runs"]
        and bool(parts[3])
        and parts[4] == "progress"
    )


def _create_payload(payload: dict[str, object]) -> dict[str, object]:
    """Refuse an unknown creation field or a missing spec before any call."""

    if set(payload) - _CREATE_FIELDS:
        raise _RouteError(400, "invalid_request", "unknown request field")
    spec = payload.get("spec")
    if not isinstance(spec, str) or not spec.strip():
        raise _RouteError(400, "invalid_request", "spec must be a non-empty string")
    return payload


def _approval_payload(payload: dict[str, object]) -> dict[str, object]:
    """Translate the external approval contract into the local field names."""

    decision = payload.get("decision")
    if decision not in _DECISIONS:
        raise _RouteError(400, "invalid_request", "decision must be APPROVE or REJECT")
    if set(payload) - _APPROVAL_FIELDS:
        raise _RouteError(400, "invalid_request", "unknown approval field")
    local: dict[str, object] = {"decision": decision}
    if decision == "REJECT":
        # A rejection needs no profile: the known profile fields are dropped
        # rather than forwarded to a local route that would refuse them.
        return local
    for name in _APPROVAL_PROFILE_FIELDS:
        value = payload.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise _RouteError(400, "invalid_request", f"{name} is invalid")
        local[name] = value
    step_profiles = payload.get("step_profiles")
    if step_profiles is None:
        return local
    if not isinstance(step_profiles, dict):
        raise _RouteError(400, "invalid_request", "step_profiles must be a JSON object")
    for step_id, profile_id in step_profiles.items():
        # The step IDs of an approval are plan step IDs, never free text:
        # they build local field names, so only the exact form is accepted.
        if not is_step_id(step_id) or not isinstance(profile_id, str) or not profile_id:
            raise _RouteError(400, "invalid_request", "invalid step profile field")
        local[f"step_profile__{step_id}"] = profile_id
    return local


def _scope_decision(payload: dict[str, object]) -> str:
    """Return the single decision of an exact scope-approval body."""

    decision = payload.get("decision")
    if set(payload) != {"decision"} or decision not in _DECISIONS:
        raise _RouteError(400, "invalid_request", "decision must be APPROVE or REJECT")
    return decision


def _require_empty_body(payload: dict[str, object]) -> None:
    if payload:
        raise _RouteError(400, "invalid_request", "resume body must be an empty JSON object")


def _plan_text(payload: dict[str, object]) -> str:
    if set(payload) != {"plan"} or not isinstance(payload.get("plan"), str):
        raise _RouteError(400, "invalid_request", "body must contain exactly one plan string")
    return payload["plan"]  # type: ignore[return-value]


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # The local body reader refuses duplicated JSON fields; the gateway
    # refuses them too so a field can never be shadowed after validation.
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicated JSON field")
        result[key] = value
    return result


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
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise _UpstreamError("local MetaHarness response is not valid JSON") from None


class _RemoteGatewayServer(ThreadingHTTPServer):
    """Loopback-only HTTP server holding the gateway configuration."""

    remote_token: str
    control_token: str
    metaharness_port: int
    local_client: LocalMetaHarnessClient


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    """One remote request; every response is JSON.

    ``protocol_version`` stays at the HTTP/1.0 default: a mutation reads
    exactly the body length it was given, so a kept-alive connection could
    not be resynchronised.
    ``default_request_version`` is HTTP/1.0 as well, so a legacy or malformed
    request line never receives an HTTP/0.9 body without status line.
    """

    server: _RemoteGatewayServer
    default_request_version = "HTTP/1.0"
    # A mutation reads its declared request body, so a stalled client must
    # not hold a handler thread forever.
    timeout = _UPSTREAM_TIMEOUT_SECONDS

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
        # answered with 501; this gateway answers 405 to every unsupported
        # verb.
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    def do_GET(self) -> None:
        if not self._authorized():
            self._json(401, _UNAUTHORIZED_PAYLOAD)
            return
        try:
            parsed = _split_target(self.path)
            target = _upstream_target(parsed.path, parsed.query)
        except _RouteError as error:
            self._json(error.status, {"error": error.error, "message": error.message})
            return
        try:
            status, payload = _upstream_json(self.server.metaharness_port, target)
        except _UpstreamError:
            self._json(502, _UPSTREAM_ERROR_PAYLOAD)
            return
        self._json(status, payload)

    def do_POST(self) -> None:
        # Authentication is checked before routing so an unauthenticated
        # caller learns nothing about the routes this gateway serves.
        if not self._authorized():
            self._json(401, _UNAUTHORIZED_PAYLOAD)
            return
        try:
            action, run_id, max_bytes = _mutation_target(_split_target(self.path).path)
            payload = self._body(max_bytes)
            status, response = self._mutate(action, run_id, payload)
        except _RouteError as error:
            self._json(error.status, {"error": error.error, "message": error.message})
            return
        except LocalMetaHarnessError:
            self._json(502, _UPSTREAM_ERROR_PAYLOAD)
            return
        if status not in _RELAYED_STATUSES or not isinstance(response, dict):
            self._json(502, _UPSTREAM_ERROR_PAYLOAD)
            return
        self._json(status, response)

    def _mutate(
        self, action: str, run_id: str, payload: dict[str, object]
    ) -> tuple[int, object]:
        """Validate, translate and send exactly one local mutation.

        The control client is the only path to MetaHarness and knows neither
        the remote bearer token nor any other caller header.  Exactly one
        request is sent here: nothing is retried, ever.
        """

        client = self.server.local_client
        if action == "create":
            return client.create_run(_create_payload(payload))
        if action == "approval":
            return client.approve_run(run_id, _approval_payload(payload))
        if action == "scope-approval":
            return client.approve_scope(run_id, _scope_decision(payload))
        if action == "resume":
            _require_empty_body(payload)
            return client.resume_run(run_id)
        return client.recover_plan(run_id, _plan_text(payload))

    def _body(self, max_bytes: int) -> dict[str, object]:
        """Read the single bounded JSON object of a mutation request.

        The body must declare exactly one ``Content-Length``, carry no
        ``Transfer-Encoding`` and be sent as ``application/json``: a request
        whose framing or media type is ambiguous is refused before the first
        byte is read.
        """

        if not _json_media_type(self.headers.get("Content-Type")):
            raise _RouteError(415, "unsupported_media_type", "body must be application/json")
        if self.headers.get_all("Transfer-Encoding"):
            raise _RouteError(400, "invalid_request", "body must use Content-Length framing")
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) > 1:
            raise _RouteError(400, "invalid_request", "body must declare one Content-Length")
        try:
            length = int(lengths[0]) if lengths else 0
        except ValueError:
            raise _RouteError(400, "invalid_request", "invalid request body") from None
        if length < 0 or length > max_bytes:
            raise _RouteError(413, "body_too_large", "request body is too large")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("short body")
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_fields)
        except (OSError, UnicodeError, ValueError, RecursionError):
            # A hostile nesting depth ends in RecursionError; the parser
            # message may quote the body, so only a fixed description leaves.
            raise _RouteError(400, "invalid_request", "body must be valid JSON") from None
        if not isinstance(payload, dict):
            raise _RouteError(400, "invalid_request", "body must be a JSON object")
        return payload

    def _method_not_allowed(self) -> None:
        # Authentication is checked before the method so an unauthenticated
        # caller learns nothing about the routes this gateway serves.
        if not self._authorized():
            self._json(401, _UNAUTHORIZED_PAYLOAD)
            return
        self._json(405, {"error": "method_not_allowed", "message": _METHOD_MESSAGE})

    def _authorized(self) -> bool:
        values = self.headers.get_all("Authorization") or []
        if len(values) != 1:
            return False
        return token_matches(
            bearer_token_from_header(values[0]), self.server.remote_token
        )

    def _json(self, status: int, payload: object) -> None:
        # ``ensure_ascii`` escapes every non-ASCII code unit, so a lone
        # surrogate relayed by the local server can never raise here; the
        # decoded JSON value is unchanged.
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
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
        except OSError:
            # A stalled or closed client is never retried and never logged.
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
    eagerly so a broken deployment fails before the first request, and the
    control client is built once from the validated control token.
    """

    remote_token = load_token_file(remote_token_file)
    control_token = load_token_file(control_token_file)
    server = _RemoteGatewayServer(
        (LOCAL_HOST, _validated_port(port, allow_zero=True)), _GatewayRequestHandler
    )
    server.remote_token = remote_token
    server.control_token = control_token
    server.metaharness_port = _validated_port(metaharness_port)
    server.local_client = LocalMetaHarnessClient(
        port=server.metaharness_port,
        control_token=control_token,
        timeout_seconds=_UPSTREAM_TIMEOUT_SECONDS,
    )
    return server


def serve_remote_gateway(
    *,
    port: int,
    remote_token_file: str | Path,
    control_token_file: str | Path,
    metaharness_port: int = DEFAULT_METAHARNESS_PORT,
) -> None:
    """Serve the gateway on ``127.0.0.1:port`` until interrupted."""

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
