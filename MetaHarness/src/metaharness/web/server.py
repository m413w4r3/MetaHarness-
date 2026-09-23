"""Threaded localhost HTTP server for the MetaHarness observation UI.

The server binds 127.0.0.1 only.  Because a run page carries the approval
mutation token, every request must also name this exact local server in its
``Host`` header (DNS rebinding / host spoofing), and a mutation carrying an
``Origin`` header must come from this exact local origin.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, parse_qsl, urlsplit

from ..config import load_config
from ..models import HarnessConfig
from ..plan_recovery import MAX_REPLACEMENT_PLAN_BYTES
from ..step_ids import is_step_id
from .api import (
    WebAPIError,
    approve_run,
    approve_repair_scope,
    create_run,
    get_run,
    list_runs,
    live_status,
    model_profiles,
    progress,
    recover_plan_request,
    resume_run_request,
    validate_run_id,
)
from .pages import render_index, render_new_run, render_run, run_page_polls
from .run_manager import RunManager

HOST = "127.0.0.1"
_MAX_BODY_BYTES = 64 * 1024
_MAX_CONTROL_TOKEN_BYTES = 4096
# Only the REPLACE PLAN routes accept more: room for a 128 KiB plan after
# form/JSON encoding.  The decoded text is bounded again, exactly.
_MAX_RECOVERY_BODY_BYTES = 4 * MAX_REPLACEMENT_PLAN_BYTES
_MAX_CONFIG_CHECKS = 80
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


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicated JSON field")
        result[key] = value
    return result


def _run_js() -> bytes:
    """The single static run-page script, read once and never templated."""

    if not _RUN_JS_CACHE:
        _RUN_JS_CACHE.append((_STATIC_DIR / "run.js").read_bytes())
    return _RUN_JS_CACHE[0]


def _actual_plan_step_ids(run_dir: Path) -> frozenset[str]:
    """Return only step IDs declared by this run's implementation bundle."""

    try:
        payload = json.loads((run_dir / "implementation_bundle.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return frozenset()
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        return frozenset()
    return frozenset(
        item["id"] for item in steps
        if isinstance(item, dict) and isinstance(item.get("id"), str)
        and is_step_id(item["id"])
    )


def _step_profile_fields(run_dir: Path) -> set[str]:
    return {f"step_profile__{step_id}" for step_id in _actual_plan_step_ids(run_dir)}


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

    def __init__(
        self,
        address: tuple[str, int],
        config: HarnessConfig,
        *,
        control_token_file: str | Path | None = None,
        config_fingerprint: str | None = None,
    ):
        self.config = config
        self.config_fingerprint = config_fingerprint
        self.browser_token = secrets.token_urlsafe(32)
        self.control_token = (
            load_or_create_control_token(control_token_file)
            if control_token_file is not None else None
        )
        self.run_manager = RunManager(
            config,
            max_active_runs=config.ui.max_active_runs,
        )
        super().__init__(address, MetaHarnessRequestHandler)

    @property
    def token(self) -> str:
        """Compatibility alias for the token embedded in browser forms."""

        return self.browser_token


def load_or_create_control_token(token_file: str | Path) -> str:
    """Read one bounded token or atomically create a private token file."""

    path = Path(token_file).expanduser()

    def read_token() -> str:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_CONTROL_TOKEN_BYTES + 1)
        if len(raw) > _MAX_CONTROL_TOKEN_BYTES:
            raise ValueError("control token file is too large")
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("control token file must be UTF-8") from exc
        value = value.removesuffix("\n").removesuffix("\r")
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("control token file must contain one non-empty token")
        return value

    try:
        return read_token()
    except FileNotFoundError:
        pass

    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            try:
                os.chmod(temporary_path, 0o600)
            except (OSError, NotImplementedError):
                pass
            stream.write(token.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
    return read_token()


class MetaHarnessRequestHandler(BaseHTTPRequestHandler):
    server: MetaHarnessHTTPServer

    def log_message(self, *_args: object) -> None:
        # Do not log paths, headers, or bodies: the mutation token never enters
        # server logs, and the UI is intended for a local operator.
        return

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        # BaseHTTPRequestHandler emits HTML for unsupported methods and parse
        # errors. Keep even those responses JSON under the versioned API prefix.
        path = urlsplit(getattr(self, "path", "")).path
        if path == "/api/v1" or path.startswith("/api/v1/"):
            detail = message or "request could not be served"
            self._json(code, {"error": "request_error", "message": detail})
            return
        super().send_error(code, message, explain)

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
            if parsed.path == "/api/v1/health":
                self._json(200, {
                    "service": "metaharness",
                    "api_version": 1,
                    "status": "ok",
                    "control_api": True,
                    "capabilities_endpoint": "/api/v1/config",
                })
                return
            if parsed.path == "/api/v1/config":
                self._json(200, configuration_description(
                    self.server.config,
                    config_fingerprint=self.server.config_fingerprint,
                ))
                return
            if parsed.path == "/api/v1/model-profiles":
                self._json(200, model_profiles(self.server.config))
                return
            if parsed.path == "/api/v1/runs":
                self._json(200, {"runs": list_runs(root)})
                return
            if len(parts) == 5 and parts[1:4] == ["api", "v1", "runs"]:
                run_id = self._run_id(parts[4])
                self._json(200, get_run(root, run_id, config=self.server.config))
                return
            if (
                len(parts) == 6
                and parts[1:4] == ["api", "v1", "runs"]
                and parts[5] == "live"
            ):
                self._json(200, live_status(root, self._run_id(parts[4]), self.server.config))
                return
            if (
                len(parts) == 6
                and parts[1:4] == ["api", "v1", "runs"]
                and parts[5] == "progress"
            ):
                query = parse_qs(parsed.query, keep_blank_values=True)
                values = query.get("offset", ["0"])
                if len(values) != 1 or not values[0].isdigit():
                    raise WebAPIError(400, "offset must be a non-negative integer")
                self._json(200, progress(root, self._run_id(parts[4]), int(values[0])))
                return
            if parsed.path == "/":
                nonce = secrets.token_urlsafe(18)
                self._html(render_index(list_runs(root), nonce=nonce), nonce)
                return
            if parsed.path == "/new":
                nonce = secrets.token_urlsafe(18)
                self._html(
                    render_new_run(self.server.config, self.server.browser_token, nonce=nonce),
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
                        run, self.server.browser_token, config=self.server.config, nonce=nonce
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

    def _authorized_api(self) -> None:
        supplied = self.headers.get("X-MetaHarness-Token")
        supplied_bytes = supplied.encode("utf-8") if supplied is not None else None
        browser_match = supplied_bytes is not None and secrets.compare_digest(
            supplied_bytes, self.server.browser_token.encode("utf-8")
        )
        control_match = (
            supplied_bytes is not None
            and self.server.control_token is not None
            and secrets.compare_digest(
                supplied_bytes, self.server.control_token.encode("utf-8")
            )
        )
        if not (browser_match or control_match):
            raise WebAPIError(403, "mutation token required")

    def _authorized_form(self, token: str | None) -> None:
        if token is None or not secrets.compare_digest(
            token.encode("utf-8"), self.server.browser_token.encode("utf-8")
        ):
            raise WebAPIError(403, "mutation token required")

    def _form(
        self, expected: set[str], *, exact: bool = True, max_bytes: int = _MAX_BODY_BYTES,
    ) -> dict[str, str]:
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
        if length < 0 or length > max_bytes:
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

    def _body(self, *, max_bytes: int = _MAX_BODY_BYTES) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else 0
        except ValueError as exc:
            raise WebAPIError(400, "invalid request body") from exc
        if length < 0 or length > max_bytes:
            raise WebAPIError(413, "request body is too large")
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_fields)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WebAPIError(400, "body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise WebAPIError(400, "body must be a JSON object")
        return payload

    def do_POST(self) -> None:
        try:
            self._check_host()
            parts = self._path_parts()
            root = self.server.config.runs_root
            html_form_route = (
                parts == ["", "runs"]
                or (
                    len(parts) == 4
                    and parts[1] == "runs"
                    and parts[3] in {"approval", "scope-approval", "resume", "recover-plan"}
                )
            )
            self._check_origin(allow_opaque=html_form_route)
            if parts == ["", "api", "v1", "runs"]:
                self._authorized_api()
                payload = self._body()
                allowed = {
                    "spec", "run_id", "planner_profile", "mechanical_profile",
                    "reasoning_profile", "agentic_profile", "final_reviewer_profile",
                    "semantic_reviser_profile", "check_repair_profile",
                    "semantic_revision_enabled", "max_check_repair_attempts",
                    "max_review_repair_cycles", "decomposition", "execution_mode_policy",
                    "single_step_max_mutable_paths", "staged_step_max_mutable_paths",
                    "repair_scope_policy", "repair_scope_max_added_paths",
                }
                if set(payload) - allowed:
                    raise WebAPIError(400, "unknown request field")
                result = create_run(
                    self.server.run_manager,
                    spec=payload.get("spec"), run_id=payload.get("run_id"),
                    planner_profile=payload.get("planner_profile"),
                    mechanical_profile=payload.get("mechanical_profile"),
                    reasoning_profile=payload.get("reasoning_profile"),
                    agentic_profile=payload.get("agentic_profile"),
                    final_reviewer_profile=payload.get("final_reviewer_profile"),
                    semantic_reviser_profile=payload.get("semantic_reviser_profile"),
                    check_repair_profile=payload.get("check_repair_profile"),
                    semantic_revision_enabled=payload.get("semantic_revision_enabled"),
                    max_check_repair_attempts=payload.get("max_check_repair_attempts"),
                    max_review_repair_cycles=payload.get("max_review_repair_cycles"),
                    decomposition=payload.get("decomposition"),
                    execution_mode_policy=payload.get("execution_mode_policy"),
                    single_step_max_mutable_paths=payload.get("single_step_max_mutable_paths"),
                    staged_step_max_mutable_paths=payload.get("staged_step_max_mutable_paths"),
                    repair_scope_policy=payload.get("repair_scope_policy"),
                    repair_scope_max_added_paths=payload.get("repair_scope_max_added_paths"),
                )
                self._json(202, {**result, "accepted": True})
                return
            if len(parts) == 6 and parts[1:4] == ["api", "v1", "runs"]:
                run_id = self._run_id(parts[4])
                action = parts[5]
                self._authorized_api()
                if action == "resume":
                    payload = self._body()
                    if payload:
                        raise WebAPIError(400, "resume body must be an empty JSON object")
                    result = resume_run_request(
                        self.server.run_manager, self.server.config.runs_root, run_id
                    )
                    self._json(202, {**result, "accepted": True})
                    return
                if action == "recover-plan":
                    payload = self._body(max_bytes=_MAX_RECOVERY_BODY_BYTES)
                    if set(payload) != {"plan"} or not isinstance(payload.get("plan"), str):
                        raise WebAPIError(400, "body must contain exactly one plan string")
                    result = recover_plan_request(
                        self.server.run_manager, self.server.config.runs_root,
                        run_id, payload["plan"],
                    )
                    self._json(202, {**result, "accepted": True})
                    return
                if action == "scope-approval":
                    payload = self._body()
                    if set(payload) != {"decision"} or not isinstance(
                        payload.get("decision"), str
                    ):
                        raise WebAPIError(400, "decision must be APPROVE or REJECT")
                    self._json(200, approve_repair_scope(
                        self.server.config.runs_root, run_id, payload["decision"]
                    ))
                    return
                if action == "approval":
                    payload = self._body()
                    decision = payload.get("decision")
                    if not isinstance(decision, str):
                        raise WebAPIError(400, "decision must be APPROVE or REJECT")
                    allowed = {"decision"} if decision == "REJECT" else (
                        {"decision"} | _APPROVAL_PROFILE_FIELDS | _step_profile_fields(root / run_id)
                    )
                    if set(payload) - allowed:
                        raise WebAPIError(400, "unknown approval field")
                    result = approve_run(
                        self.server.config.runs_root, run_id, decision,
                        config=self.server.config, **_approval_profiles(payload),
                    )
                    self._json(200, result)
                    return
                raise WebAPIError(404, "not found")
            if parts == ["", "api", "runs"]:
                self._authorized_api()
                payload = self._body()
                allowed = {
                    "spec", "run_id", "planner_profile", "mechanical_profile",
                    "reasoning_profile", "agentic_profile",
                    "final_reviewer_profile", "semantic_reviser_profile", "check_repair_profile",
                    "semantic_revision_enabled", "max_check_repair_attempts",
                    "max_review_repair_cycles",
                    "decomposition",
                    "execution_mode_policy", "single_step_max_mutable_paths",
                    "staged_step_max_mutable_paths",
                    "repair_scope_policy", "repair_scope_max_added_paths",
                }
                unknown = set(payload) - allowed
                if unknown:
                    raise WebAPIError(400, "unknown request field")
                result = create_run(
                    self.server.run_manager,
                    spec=payload.get("spec"),
                    run_id=payload.get("run_id"),
                    planner_profile=payload.get("planner_profile"),
                    mechanical_profile=payload.get("mechanical_profile"),
                    reasoning_profile=payload.get("reasoning_profile"),
                    agentic_profile=payload.get("agentic_profile"),
                    final_reviewer_profile=payload.get("final_reviewer_profile"),
                    semantic_reviser_profile=payload.get("semantic_reviser_profile"),
                    check_repair_profile=payload.get("check_repair_profile"),
                    semantic_revision_enabled=payload.get("semantic_revision_enabled"),
                    max_check_repair_attempts=payload.get("max_check_repair_attempts"),
                    max_review_repair_cycles=payload.get("max_review_repair_cycles"),
                    decomposition=payload.get("decomposition"),
                    execution_mode_policy=payload.get("execution_mode_policy"),
                    single_step_max_mutable_paths=payload.get("single_step_max_mutable_paths"),
                    staged_step_max_mutable_paths=payload.get("staged_step_max_mutable_paths"),
                    repair_scope_policy=payload.get("repair_scope_policy"),
                    repair_scope_max_added_paths=payload.get("repair_scope_max_added_paths"),
                )
                self._json(202, result)
                return
            if parts == ["", "runs"]:
                payload = self._form({
                    "_token", "spec", "run_id", "planner_profile", "mechanical_profile",
                    "reasoning_profile", "agentic_profile",
                    "final_reviewer_profile", "semantic_reviser_profile", "check_repair_profile",
                    "semantic_revision_enabled", "max_check_repair_attempts",
                    "max_review_repair_cycles",
                    "decomposition",
                    "execution_mode_policy", "single_step_max_mutable_paths",
                    "staged_step_max_mutable_paths",
                    "repair_scope_policy", "repair_scope_max_added_paths",
                }, exact=False)
                if not {"_token", "spec", "run_id", "planner_profile"}.issubset(payload):
                    raise WebAPIError(400, "missing request field")
                self._authorized_form(payload.get("_token"))
                result = create_run(
                    self.server.run_manager,
                    spec=payload["spec"],
                    run_id=payload["run_id"] or None,
                    planner_profile=payload["planner_profile"],
                    mechanical_profile=payload.get("mechanical_profile"),
                    reasoning_profile=payload.get("reasoning_profile"),
                    agentic_profile=payload.get("agentic_profile"),
                    final_reviewer_profile=payload.get("final_reviewer_profile"),
                    semantic_reviser_profile=payload.get("semantic_reviser_profile"),
                    check_repair_profile=payload.get("check_repair_profile"),
                    semantic_revision_enabled=payload.get("semantic_revision_enabled"),
                    max_check_repair_attempts=payload.get("max_check_repair_attempts"),
                    max_review_repair_cycles=payload.get("max_review_repair_cycles"),
                    decomposition=payload.get("decomposition"),
                    execution_mode_policy=payload.get("execution_mode_policy"),
                    single_step_max_mutable_paths=payload.get("single_step_max_mutable_paths"),
                    staged_step_max_mutable_paths=payload.get("staged_step_max_mutable_paths"),
                    repair_scope_policy=payload.get("repair_scope_policy"),
                    repair_scope_max_added_paths=payload.get("repair_scope_max_added_paths"),
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
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "recover-plan":
                # REPLACE PLAN: same Host/origin/token policy as resume.  The
                # replacement META PLAN v2 is the only operator-supplied field.
                payload = self._form({"_token", "plan"}, max_bytes=_MAX_RECOVERY_BODY_BYTES)
                self._authorized_form(payload.get("_token"))
                result = recover_plan_request(
                    self.server.run_manager,
                    self.server.config.runs_root,
                    self._run_id(parts[2]),
                    payload["plan"],
                )
                self._redirect(result["location"])
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "recover-plan":
                self._authorized_api()
                payload = self._body(max_bytes=_MAX_RECOVERY_BODY_BYTES)
                if set(payload) != {"plan"} or not isinstance(payload.get("plan"), str):
                    raise WebAPIError(400, "body must contain exactly one plan string")
                self._json(202, recover_plan_request(
                    self.server.run_manager,
                    self.server.config.runs_root,
                    self._run_id(parts[3]),
                    payload["plan"],
                ))
                return
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "scope-approval":
                payload = self._form({"_token", "decision"})
                self._authorized_form(payload.get("_token"))
                approve_repair_scope(
                    self.server.config.runs_root, self._run_id(parts[2]), payload.get("decision", "")
                )
                self._redirect(f"/runs/{self._run_id(parts[2])}")
                return
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "approval":
                run_id = self._run_id(parts[2])
                payload = self._form(
                    {"_token", "decision"} | _APPROVAL_PROFILE_FIELDS
                    | _step_profile_fields(root / run_id),
                    exact=False,
                )
                self._authorized_form(payload.get("_token"))
                decision = payload.get("decision")
                if decision == "APPROVE":
                    step_fields = {key for key in payload if key.startswith("step_profile__")}
                    required = {"_token", "decision", "final_reviewer_profile"} | step_fields
                    if not step_fields or not required.issubset(payload):
                        raise WebAPIError(400, "missing approval field")
                elif decision == "REJECT":
                    if set(payload) != {"_token", "decision"}:
                        raise WebAPIError(400, "unknown approval field")
                else:
                    raise WebAPIError(400, "decision must be APPROVE or REJECT")
                approve_run(
                    self.server.config.runs_root,
                    run_id,
                    decision,
                    config=self.server.config,
                    **_approval_profiles(payload),
                )
                self._redirect(f"/runs/{run_id}")
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "scope-approval":
                self._authorized_api()
                payload = self._body()
                if set(payload) != {"decision"} or not isinstance(payload.get("decision"), str):
                    raise WebAPIError(400, "decision must be APPROVE or REJECT")
                self._json(200, approve_repair_scope(
                    self.server.config.runs_root, self._run_id(parts[3]), payload["decision"]
                ))
                return
            if len(parts) != 5 or parts[1:3] != ["api", "runs"] or parts[4] != "approval":
                raise WebAPIError(404, "not found")
            self._authorized_api()
            payload = self._body()
            decision = payload.get("decision")
            if not isinstance(decision, str):
                raise WebAPIError(400, "decision must be APPROVE or REJECT")
            run_id = self._run_id(parts[3])
            allowed = {"decision"} if decision == "REJECT" else (
                {"decision"} | _APPROVAL_PROFILE_FIELDS | _step_profile_fields(root / run_id)
            )
            if set(payload) - allowed:
                raise WebAPIError(400, "unknown approval field")
            result = approve_run(
                self.server.config.runs_root,
                run_id,
                decision,
                config=self.server.config,
                **_approval_profiles(payload),
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


_APPROVAL_PROFILE_FIELDS = frozenset({
    "final_reviewer_profile", "semantic_reviser_profile", "check_repair_profile",
})


def _approval_profiles(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The profile fields of an approval, by their current names only."""

    if payload.get("decision") != "APPROVE":
        return {}
    step_profiles = {
        key.removeprefix("step_profile__"): value
        for key, value in payload.items() if key.startswith("step_profile__")
    }
    return {
        "final_reviewer_profile": payload.get("final_reviewer_profile"),
        "semantic_reviser_profile": payload.get("semantic_reviser_profile"),
        "check_repair_profile": payload.get("check_repair_profile"),
        "step_profiles": step_profiles or None,
    }


def create_server(
    config: HarnessConfig | str | Path,
    port: int = 8765,
    *,
    control_token_file: str | Path | None = None,
) -> MetaHarnessHTTPServer:
    fingerprint = None
    if isinstance(config, HarnessConfig):
        loaded = config
    else:
        source = Path(config).expanduser()
        loaded = load_config(source)
        # Hash the original file bytes. In particular, do not serialize the
        # expanded config, which may contain values loaded from .env files.
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    return MetaHarnessHTTPServer(
        (HOST, port), loaded, control_token_file=control_token_file,
        config_fingerprint=fingerprint,
    )


def configuration_description(
    config: HarnessConfig, *, config_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return the bounded, credential-free subset intended for UI forms."""

    planning = config.planning
    revision = config.revision
    checks = config.check_catalog[:_MAX_CONFIG_CHECKS]
    return {
        "repository": {
            "repo": str(config.repo),
            "base_ref": config.base_ref,
            "remote": config.repository.remote,
        },
        "planning": {
            "protocol": planning.protocol,
            "decomposition": planning.decomposition,
            "execution_mode_policy": planning.execution_mode_policy,
            "single_step_max_mutable_paths": planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": planning.staged_step_max_mutable_paths,
            "max_steps_per_plan": planning.max_steps_per_plan,
        },
        "revision": {
            "enabled": revision.enabled,
            "max_check_repair_attempts": revision.max_check_repair_attempts,
            "max_review_repair_cycles": revision.max_review_repair_cycles,
        },
        "approval": {
            "require_plan_approval": config.approval.require_plan_approval,
        },
        "publish": {
            "enabled": config.publish.enabled,
            "mode": config.publish.mode,
            "remote": config.publish.remote,
        },
        "ui": {"max_active_runs": config.ui.max_active_runs},
        "routing": {
            "mechanical_profile": config.routing.mechanical_profile,
            "reasoning_profile": config.routing.reasoning_profile,
            "agentic_profile": config.routing.agentic_profile,
        },
        # Do not include argv, preflight_argv, cwd or environment values.
        "checks": [
            {"id": check.id[:256], "description": check.description[:300]}
            for check in checks
        ],
        "checks_truncated": len(config.check_catalog) > len(checks),
        "capabilities": {
            "plan_approval": bool(config.approval.require_plan_approval),
            "semantic_revision": bool(revision.enabled),
            "resume": callable(resume_run_request),
            "scope_approval": revision.max_review_repair_cycles > 0
                and callable(approve_repair_scope),
            "recover_plan": callable(recover_plan_request),
            "publish": bool(config.publish.enabled),
        },
        "config_fingerprint": config_fingerprint,
    }


def serve(
    config: HarnessConfig | str | Path,
    port: int = 8765,
    *,
    control_token_file: str | Path | None = None,
) -> None:
    server = create_server(config, port=port, control_token_file=control_token_file)
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
    "configuration_description",
    "html_csp",
    "serve",
]
