"""Threaded localhost HTTP server for the MetaHarness observation UI.

The server binds 127.0.0.1 only.  Because a run page carries the approval
mutation token, every request must also name this exact local server in its
``Host`` header (DNS rebinding / host spoofing), and a mutation carrying an
``Origin`` header must come from this exact local origin.
"""

from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlsplit

from ..config import load_config
from ..models import HarnessConfig
from .api import (
    WebAPIError,
    approve_run,
    create_run,
    get_run,
    list_runs,
    live_status,
    model_profiles,
    progress,
    resume_run_request,
    validate_run_id,
)
from .pages import render_index, render_new_run, render_run, run_page_polls
from .run_manager import RunManager

HOST = "127.0.0.1"
_MAX_BODY_BYTES = 64 * 1024
_LOCAL_HOST_NAMES = ("127.0.0.1", "localhost")
_SECURITY_HEADERS = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cache-Control", "no-store"),
)
_API_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
_STATIC_DIR = Path(__file__).with_name("static")
_RUN_JS_CACHE: list[bytes] = []
RUN_JS_PATH = "/static/run.js"


def _run_js() -> bytes:
    """The single static run-page script, read once and never templated."""

    if not _RUN_JS_CACHE:
        _RUN_JS_CACHE.append((_STATIC_DIR / "run.js").read_bytes())
    return _RUN_JS_CACHE[0]


def html_csp(nonce: str, *, script_self: bool = False) -> str:
    """CSP for the server-rendered UI.

    No page has an inline script.  Only a running run page loads the static
    same-origin ``/static/run.js`` (``script-src 'self'``); every other page
    keeps ``script-src 'none'``.
    """

    script_source = "'self'" if script_self else "'none'"
    return (
        "default-src 'none'; "
        f"script-src {script_source}; "
        f"style-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )


def allowed_hosts(port: int) -> frozenset[str]:
    """Exact ``Host`` header values naming this local server."""

    hosts = {f"{name}:{port}" for name in _LOCAL_HOST_NAMES}
    if port == 80:
        # A browser omits the default port from Host.
        hosts.update(_LOCAL_HOST_NAMES)
    return frozenset(hosts)


def allowed_origins(port: int) -> frozenset[str]:
    """Exact ``Origin`` header values of pages served by this local server."""

    return frozenset(f"http://{host}" for host in allowed_hosts(port))


class MetaHarnessHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying configuration, mutation token and run capacity."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: HarnessConfig):
        self.config = config
        self.token = secrets.token_urlsafe(32)
        self.run_manager = RunManager(
            config,
            max_active_runs=config.ui.max_active_runs,
        )
        super().__init__(address, MetaHarnessRequestHandler)


class MetaHarnessRequestHandler(BaseHTTPRequestHandler):
    server: MetaHarnessHTTPServer

    def log_message(self, *_args: object) -> None:
        # Do not log paths, headers, or bodies: the mutation token never enters
        # server logs, and the UI is intended for a local operator.
        return

    def _send(
        self, status: int, body: bytes, content_type: str, *, csp: str = _API_CSP
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in _SECURITY_HEADERS:
            self.send_header(name, value)
        self.send_header("Content-Security-Policy", csp)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, error: WebAPIError) -> None:
        self._json(error.status, {"error": error.message, "message": error.message})

    def _path_parts(self) -> list[str]:
        return urlsplit(self.path).path.split("/")

    def _run_id(self, value: str) -> str:
        return validate_run_id(value)

    def _single_header(self, name: str) -> str | None:
        values = self.headers.get_all(name) or []
        if len(values) > 1:
            raise WebAPIError(403, f"duplicated {name} header")
        return values[0] if values else None

    def _check_host(self) -> None:
        host = self._single_header("Host")
        # Exact comparison: no substring, suffix or prefix match.
        if host is None or host.strip().lower() not in allowed_hosts(self.server.server_port):
            raise WebAPIError(403, "host not allowed")

    def _check_origin(self, *, allow_opaque: bool = False) -> None:
        origin = self._single_header("Origin")
        if origin is None:
            return
        value = origin.strip()
        if allow_opaque and value == "null":
            return
        if value not in allowed_origins(self.server.server_port):
            raise WebAPIError(403, "origin not allowed")

    def do_GET(self) -> None:
        try:
            self._check_host()
            parsed = urlsplit(self.path)
            parts = parsed.path.split("/")
            root = self.server.config.runs_root
            if parsed.path == "/":
                nonce = secrets.token_urlsafe(18)
                self._html(render_index(list_runs(root), nonce=nonce), nonce)
                return
            if parsed.path == "/new":
                nonce = secrets.token_urlsafe(18)
                self._html(
                    render_new_run(self.server.config, self.server.token, nonce=nonce),
                    nonce,
                )
                return
            if parsed.path == RUN_JS_PATH:
                self._send(200, _run_js(), "application/javascript; charset=utf-8")
                return
            if len(parts) == 3 and parts[1] == "runs":
                nonce = secrets.token_urlsafe(18)
                run = get_run(root, self._run_id(parts[2]), config=self.server.config)
                # render_run embeds the token only on a page able to decide
                # a plan approval or to resume a resumable run.
                self._html(
                    render_run(
                        run, self.server.token, config=self.server.config, nonce=nonce
                    ),
                    nonce,
                    script_self=run_page_polls(run),
                )
                return
            if parsed.path == "/api/runs":
                self._json(200, {"runs": list_runs(root)})
                return
            if parsed.path == "/api/model-profiles":
                self._json(200, model_profiles(self.server.config))
                return
            if len(parts) == 4 and parts[1:3] == ["api", "runs"]:
                self._json(200, get_run(root, self._run_id(parts[3]), config=self.server.config))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "live":
                self._json(200, live_status(root, self._run_id(parts[3]), self.server.config))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "progress":
                query = parse_qs(parsed.query, keep_blank_values=True)
                values = query.get("offset", ["0"])
                if len(values) != 1 or not values[0].isdigit():
                    raise WebAPIError(400, "offset must be a non-negative integer")
                self._json(200, progress(root, self._run_id(parts[3]), int(values[0])))
                return
            raise WebAPIError(404, "not found")
        except WebAPIError as exc:
            self._error(exc)
        except (OSError, ValueError, UnicodeError):
            self._error(WebAPIError(503, "request could not be served"))

    def _authorized(self) -> None:
        supplied = self.headers.get("X-MetaHarness-Token")
        if supplied is None or not secrets.compare_digest(supplied, self.server.token):
            raise WebAPIError(403, "mutation token required")

    def _authorized_form(self, token: str | None) -> None:
        if token is None or not secrets.compare_digest(token, self.server.token):
            raise WebAPIError(403, "mutation token required")

    def _form(self, expected: set[str], *, exact: bool = True) -> dict[str, str]:
        content_type = self.headers.get("Content-Type")
        if (
            content_type is None
            or content_type.split(";", 1)[0].strip().lower()
            != "application/x-www-form-urlencoded"
        ):
            raise WebAPIError(400, "body must be application/x-www-form-urlencoded")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except ValueError as exc:
            raise WebAPIError(400, "invalid request body") from exc
        if length < 0 or length > _MAX_BODY_BYTES:
            raise WebAPIError(413, "request body is too large")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("short body")
            pairs = parse_qsl(
                raw.decode("utf-8"),
                keep_blank_values=True,
                strict_parsing=True,
                encoding="utf-8",
                errors="strict",
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise WebAPIError(400, "body must be valid form data") from exc
        result: dict[str, str] = {}
        for key, value in pairs:
            if key not in expected:
                raise WebAPIError(400, "unknown request field")
            if key in result:
                raise WebAPIError(400, "duplicated request field")
            result[key] = value
        if exact and set(result) != expected:
            raise WebAPIError(400, "missing request field")
        return result

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "0")
        for name, value in _SECURITY_HEADERS:
            self.send_header(name, value)
        self.send_header("Content-Security-Policy", _API_CSP)
        self.end_headers()

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except ValueError as exc:
            raise WebAPIError(400, "invalid request body") from exc
        if length < 0 or length > _MAX_BODY_BYTES:
            raise WebAPIError(413, "request body is too large")
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WebAPIError(400, "body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise WebAPIError(400, "body must be a JSON object")
        return payload

    def do_POST(self) -> None:
        try:
            self._check_host()
            parts = self._path_parts()
            html_form_route = (
                parts == ["", "runs"]
                or (
                    len(parts) == 4
                    and parts[1] == "runs"
                    and parts[3] in {"approval", "resume"}
                )
            )
            self._check_origin(allow_opaque=html_form_route)
            if parts == ["", "api", "runs"]:
                self._authorized()
                payload = self._body()
                unknown = set(payload) - {"spec", "run_id", "planner_profile"}
                if unknown:
                    raise WebAPIError(400, "unknown request field")
                result = create_run(
                    self.server.run_manager,
                    spec=payload.get("spec"),
                    run_id=payload.get("run_id"),
                    planner_profile=payload.get("planner_profile"),
                )
                self._json(202, result)
                return
            if parts == ["", "runs"]:
                payload = self._form({"_token", "spec", "run_id", "planner_profile"})
                self._authorized_form(payload.get("_token"))
                result = create_run(
                    self.server.run_manager,
                    spec=payload["spec"],
                    run_id=payload["run_id"] or None,
                    planner_profile=payload["planner_profile"] or None,
                )
                self._redirect(result["location"])
                return
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "resume":
                # The only resume mutation: same exact-Host, origin and token
                # policy as the approval form; the RunManager resumes the same
                # run id at its durable checkpoint.
                payload = self._form({"_token"})
                self._authorized_form(payload.get("_token"))
                result = resume_run_request(
                    self.server.run_manager,
                    self.server.config.runs_root,
                    self._run_id(parts[2]),
                )
                self._redirect(result["location"])
                return
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "approval":
                payload = self._form(
                    {"_token", "decision", "implementer_profile", "reviewer_profile", "reviser_profile", "repair_profile"}
                    | {f"step_profile__S{index:02d}" for index in range(1, 7)},
                    exact=False,
                )
                self._authorized_form(payload.get("_token"))
                decision = payload.get("decision")
                if decision == "APPROVE":
                    step_fields = {key for key in payload if key.startswith("step_profile__")}
                    if step_fields:
                        expected = {"_token", "decision", "reviewer_profile"} | step_fields
                        if "reviser_profile" in payload or "repair_profile" in payload:
                            expected |= {"reviser_profile", "repair_profile"}
                        if set(payload) != expected:
                            raise WebAPIError(400, "missing approval field")
                    elif set(payload) != {"_token", "decision", "implementer_profile", "reviewer_profile"}:
                        raise WebAPIError(400, "missing approval field")
                elif decision == "REJECT":
                    if set(payload) != {"_token", "decision"}:
                        raise WebAPIError(400, "unknown approval field")
                else:
                    raise WebAPIError(400, "decision must be APPROVE or REJECT")
                approve_run(
                    self.server.config.runs_root,
                    self._run_id(parts[2]),
                    decision,
                    config=self.server.config,
                    implementer_profile=payload.get("implementer_profile"),
                    reviewer_profile=payload.get("reviewer_profile")
                    if decision == "APPROVE"
                    else None,
                    reviser_profile=payload.get("reviser_profile") if decision == "APPROVE" else None,
                    repair_profile=payload.get("repair_profile") if decision == "APPROVE" else None,
                    step_profiles={key.removeprefix("step_profile__"): value for key, value in payload.items() if key.startswith("step_profile__")}
                    if decision == "APPROVE" and any(key.startswith("step_profile__") for key in payload)
                    else None,
                )
                self._redirect(f"/runs/{self._run_id(parts[2])}")
                return
            if len(parts) != 5 or parts[1:3] != ["api", "runs"] or parts[4] != "approval":
                raise WebAPIError(404, "not found")
            self._authorized()
            payload = self._body()
            decision = payload.get("decision")
            if not isinstance(decision, str):
                raise WebAPIError(400, "decision must be APPROVE or REJECT")
            allowed = {"decision"} if decision == "REJECT" else {
                "decision", "implementer_profile", "reviewer_profile", "reviser_profile", "repair_profile"
            } | {key for key in payload if key.startswith("step_profile__")}
            if set(payload) - allowed:
                raise WebAPIError(400, "unknown approval field")
            result = approve_run(
                self.server.config.runs_root,
                self._run_id(parts[3]),
                decision,
                config=self.server.config,
                implementer_profile=payload.get("implementer_profile"),
                reviewer_profile=payload.get("reviewer_profile"),
                reviser_profile=payload.get("reviser_profile"),
                repair_profile=payload.get("repair_profile"),
                step_profiles={key.removeprefix("step_profile__"): value for key, value in payload.items() if key.startswith("step_profile__")}
                if any(key.startswith("step_profile__") for key in payload) else None,
            )
            self._json(200, result)
        except WebAPIError as exc:
            self._error(exc)
        except (OSError, ValueError, UnicodeError):
            self._error(WebAPIError(503, "request could not be served"))

    def _html(self, content: str, nonce: str, *, script_self: bool = False) -> None:
        self._send(
            200,
            content.encode("utf-8"),
            "text/html; charset=utf-8",
            csp=html_csp(nonce, script_self=script_self),
        )


def create_server(config: HarnessConfig | str | Path, port: int = 8765) -> MetaHarnessHTTPServer:
    loaded = load_config(config) if not isinstance(config, HarnessConfig) else config
    return MetaHarnessHTTPServer((HOST, port), loaded)


def serve(config: HarnessConfig | str | Path, port: int = 8765) -> None:
    server = create_server(config, port=port)
    print(f"MetaHarness UI: http://{HOST}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "HOST",
    "MetaHarnessHTTPServer",
    "MetaHarnessRequestHandler",
    "RUN_JS_PATH",
    "allowed_hosts",
    "allowed_origins",
    "create_server",
    "html_csp",
    "serve",
]
