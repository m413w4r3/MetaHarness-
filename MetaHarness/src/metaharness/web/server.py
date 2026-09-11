"""Threaded localhost HTTP server for the MetaHarness observation UI."""

from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..config import load_config
from ..models import HarnessConfig
from .api import WebAPIError, approve_run, get_run, list_runs, progress, validate_run_id
from .pages import render_index, render_run

HOST = "127.0.0.1"
_MAX_BODY_BYTES = 64 * 1024


class MetaHarnessHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying only in-memory configuration and mutation token."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: HarnessConfig):
        self.config = config
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, MetaHarnessRequestHandler)


class MetaHarnessRequestHandler(BaseHTTPRequestHandler):
    server: MetaHarnessHTTPServer

    def log_message(self, *_args: object) -> None:
        # Do not log paths, headers, or bodies: the mutation token never enters
        # server logs, and the UI is intended for a local operator.
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
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

    def do_GET(self) -> None:
        try:
            parsed = urlsplit(self.path)
            parts = parsed.path.split("/")
            root = self.server.config.runs_root
            if parsed.path == "/":
                self._html(render_index(list_runs(root), self.server.token))
                return
            if len(parts) == 3 and parts[1] == "runs":
                self._html(render_run(get_run(root, self._run_id(parts[2])), self.server.token))
                return
            if parsed.path == "/api/runs":
                self._json(200, {"runs": list_runs(root)})
                return
            if len(parts) == 4 and parts[1:3] == ["api", "runs"]:
                self._json(200, get_run(root, self._run_id(parts[3])))
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
            self._authorized()
            parts = self._path_parts()
            if len(parts) != 5 or parts[1:3] != ["api", "runs"] or parts[4] != "approval":
                raise WebAPIError(404, "not found")
            payload = self._body()
            decision = payload.get("decision")
            if not isinstance(decision, str):
                raise WebAPIError(400, "decision must be APPROVE or REJECT")
            result = approve_run(
                self.server.config.runs_root,
                self._run_id(parts[3]),
                decision,
            )
            self._json(200, result)
        except WebAPIError as exc:
            self._error(exc)
        except (OSError, ValueError, UnicodeError):
            self._error(WebAPIError(503, "request could not be served"))

    def _html(self, content: str) -> None:
        self._send(200, content.encode("utf-8"), "text/html; charset=utf-8")


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


__all__ = ["HOST", "MetaHarnessHTTPServer", "MetaHarnessRequestHandler", "create_server", "serve"]
