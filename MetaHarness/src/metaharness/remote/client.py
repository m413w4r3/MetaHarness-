"""HTTP client for a local MetaHarness control gateway.

The gateway is always reached at ``127.0.0.1``: the host is a module
constant and callers cannot substitute another one.  The client follows no
redirect, caps every response, and never logs bodies or headers.  Failure
messages repeat neither the control token nor response content.

A mutation method performs exactly one local request and returns the local
``(status, payload)`` unchanged: it never retries and never turns a non-2xx
answer into an exception, so a relaying caller can never replay a mutation.
"""

from __future__ import annotations

import http.client
import json
import math

LOCAL_HOST = "127.0.0.1"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

_API_PREFIX = "/api/v1"
_ALLOWED_METHODS = ("GET", "POST")
_RUN_ID_REFUSED = ("/", "\\", "..")


class LocalMetaHarnessError(RuntimeError):
    """A local MetaHarness exchange failed.

    ``status`` is set when an HTTP response was received, and ``payload``
    when that response carried decodable JSON.  The message never contains
    the control token, request bodies or response bodies.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        payload: object | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.payload = payload


class LocalMetaHarnessClient:
    """Minimal stdlib client for the MetaHarness control server.

    Parameters are validated eagerly: a bad ``port``, ``control_token``,
    ``timeout_seconds``, ``run_id`` or ``offset`` raises ``ValueError``
    without echoing the rejected value.
    """

    def __init__(
        self,
        *,
        port: int = 8765,
        control_token: str,
        timeout_seconds: float = 10.0,
    ):
        self._port = _validated_port(port)
        self._control_token = _validated_token(control_token)
        self._timeout_seconds = _validated_timeout(timeout_seconds)

    def health(self) -> dict[str, object]:
        """Return the gateway health document."""

        return self._object(self._request_json("GET", f"{_API_PREFIX}/health"))

    def config(self) -> dict[str, object]:
        """Return the configuration description served by the gateway."""

        return self._object(self._request_json("GET", f"{_API_PREFIX}/config"))

    def model_profiles(self) -> dict[str, object]:
        """Return the model profiles served by the gateway."""

        return self._object(self._request_json("GET", f"{_API_PREFIX}/model-profiles"))

    def list_runs(self) -> dict[str, object]:
        """Return the run list served by the gateway."""

        return self._object(self._request_json("GET", f"{_API_PREFIX}/runs"))

    def get_run(self, run_id: str) -> dict[str, object]:
        """Return one run document."""

        return self._object(
            self._request_json("GET", f"{_API_PREFIX}/runs/{_validated_run_id(run_id)}")
        )

    def progress(self, run_id: str, offset: int) -> dict[str, object]:
        """Return the progress document of *run_id* from *offset*."""

        target = _validated_run_id(run_id)
        position = _validated_offset(offset)
        return self._object(
            self._request_json("GET", f"{_API_PREFIX}/runs/{target}/progress?offset={position}")
        )

    def create_run(self, payload: dict[str, object]) -> tuple[int, object]:
        """Create a run with exactly one local POST; nothing is retried.

        The local status and JSON body are returned unchanged so a caller
        that relays them never repeats the mutation.
        """

        return self._request_json("POST", f"{_API_PREFIX}/runs", payload)

    def approve_run(self, run_id: str, payload: dict[str, object]) -> tuple[int, object]:
        """Approve or reject the plan of *run_id* with one local POST."""

        target = _validated_run_id(run_id)
        return self._request_json(
            "POST", f"{_API_PREFIX}/runs/{target}/approval", payload
        )

    def approve_scope(self, run_id: str, decision: str) -> tuple[int, object]:
        """Answer the scope gate of *run_id* with one local POST."""

        target = _validated_run_id(run_id)
        if decision not in ("APPROVE", "REJECT"):
            raise ValueError("decision must be APPROVE or REJECT")
        return self._request_json(
            "POST", f"{_API_PREFIX}/runs/{target}/scope-approval", {"decision": decision}
        )

    def resume_run(self, run_id: str) -> tuple[int, object]:
        """Resume *run_id* with one local POST carrying exactly ``{}``."""

        target = _validated_run_id(run_id)
        return self._request_json("POST", f"{_API_PREFIX}/runs/{target}/resume", {})

    def recover_plan(self, run_id: str, plan: str) -> tuple[int, object]:
        """Replace the plan of *run_id* with one local POST."""

        target = _validated_run_id(run_id)
        if not isinstance(plan, str):
            raise ValueError("plan must be a string")
        return self._request_json(
            "POST", f"{_API_PREFIX}/runs/{target}/recover-plan", {"plan": plan}
        )

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> tuple[int, object]:
        """Perform one local request and return ``(status, decoded JSON)``.

        Only 127.0.0.1 is reachable; redirects are never followed.  A
        response larger than ``MAX_RESPONSE_BYTES``, a non-UTF-8 body or an
        invalid JSON body raises ``LocalMetaHarnessError``.
        """

        verb = _validated_method(method)
        target = _validated_path(path)
        headers = {"Accept": "application/json"}
        payload: bytes | None = None

        if verb == "POST":
            headers["Content-Type"] = "application/json"
            headers["X-MetaHarness-Token"] = self._control_token
            payload = _encoded_body(body)
        elif body is not None:
            raise ValueError("body is only supported for POST requests")

        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection(
                LOCAL_HOST, self._port, timeout=self._timeout_seconds
            )
            connection.request(verb, target, body=payload, headers=headers)
            response = connection.getresponse()
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except TimeoutError:
            raise LocalMetaHarnessError("local MetaHarness request timed out") from None
        except http.client.IncompleteRead:
            raise LocalMetaHarnessError("local MetaHarness response was truncated") from None
        except (http.client.HTTPException, OSError, ValueError) as exc:
            # http.client messages can quote request data; only the class
            # name is reported.
            raise LocalMetaHarnessError(
                f"local MetaHarness request failed ({type(exc).__name__})"
            ) from None
        finally:
            if connection is not None:
                connection.close()

        if len(raw) > MAX_RESPONSE_BYTES:
            raise LocalMetaHarnessError(
                f"local MetaHarness response exceeds {MAX_RESPONSE_BYTES} bytes",
                status=status,
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise LocalMetaHarnessError(
                f"local MetaHarness response is not valid UTF-8 (HTTP {status})",
                status=status,
            ) from None
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            raise LocalMetaHarnessError(
                f"local MetaHarness response is not valid JSON (HTTP {status})",
                status=status,
            ) from None
        return status, decoded

    def _object(self, result: tuple[int, object]) -> dict[str, object]:
        """Refuse non-2xx statuses and non-object JSON documents."""

        status, payload = result
        # http.client never follows redirects, so a 3xx arrives here as-is.
        if not 200 <= status < 300:
            raise LocalMetaHarnessError(
                f"local MetaHarness request failed with HTTP {status}",
                status=status,
                payload=payload,
            )
        if not isinstance(payload, dict):
            raise LocalMetaHarnessError(
                "local MetaHarness JSON response must be an object",
                status=status,
                payload=payload,
            )
        return payload


def _validated_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return port


def _validated_token(control_token: str) -> str:
    # A space or control character would make http.client raise an error
    # quoting the header value; reject it here without echoing the value.
    if (
        not isinstance(control_token, str)
        or not control_token
        or any(not 0x21 <= ord(character) <= 0x7E for character in control_token)
    ):
        raise ValueError("control_token must be a non-empty printable ASCII string")
    return control_token


def _validated_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    value = float(timeout_seconds)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    return value


def _validated_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if any(marker in run_id for marker in _RUN_ID_REFUSED):
        raise ValueError("run_id must be a single path component without '..'")
    if any(not 0x21 <= ord(character) <= 0x7E for character in run_id):
        raise ValueError("run_id must contain printable ASCII characters only")
    return run_id


def _validated_offset(offset: int) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError("offset must be a non-negative integer")
    if offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return offset


def _validated_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string")
    verb = method.upper()
    if verb not in _ALLOWED_METHODS:
        raise ValueError(f"method must be one of {', '.join(_ALLOWED_METHODS)}")
    return verb


def _validated_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must be an absolute local API path")
    if any(not 0x21 <= ord(character) <= 0x7E for character in path):
        raise ValueError("path must contain printable ASCII characters only")
    return path


def _encoded_body(body: dict[str, object] | None) -> bytes | None:
    if body is None:
        return None
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    try:
        return json.dumps(body, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("body must be JSON-serializable") from None


__all__ = [
    "LOCAL_HOST",
    "MAX_RESPONSE_BYTES",
    "LocalMetaHarnessClient",
    "LocalMetaHarnessError",
]
