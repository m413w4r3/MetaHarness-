from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Path as FastAPIPath
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


Provider = Literal["bridge", "webai"]
Surface = Literal["chat", "responses", "native", "stateless"]

BRIDGE_BASE_URL = os.getenv("BRIDGE_BASE_URL", "http://chatgpt-bridge:8001").rstrip("/")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")
WEBAI_BASE_URL = os.getenv("WEBAI_BASE_URL", "http://web_ai:6969").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LAB_REQUEST_TIMEOUT_SECONDS", "300"))
MAX_REQUEST_BYTES = int(os.getenv("LAB_MAX_REQUEST_BYTES", str(1024 * 1024)))
MAX_RESPONSE_BYTES = int(os.getenv("LAB_MAX_RESPONSE_BYTES", str(4 * 1024 * 1024)))

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Bridge Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SendRequest(BaseModel):
    provider: Provider
    surface: Surface
    payload: dict[str, Any]
    idempotency_key: str | None = Field(default=None, max_length=255)


def _provider_base(provider: Provider) -> str:
    return BRIDGE_BASE_URL if provider == "bridge" else WEBAI_BASE_URL


def _headers(provider: Provider, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider == "bridge" and BRIDGE_API_KEY:
        headers["Authorization"] = f"Bearer {BRIDGE_API_KEY}"
    if idempotency_key:
        headers["X-Idempotency-Key"] = idempotency_key
    return headers


def _surface_path(provider: Provider, surface: Surface) -> str:
    mapping = {
        "bridge": {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "native": "/v1/bridge/runs",
        },
        "webai": {
            "chat": "/v1/chat/completions",
            "stateless": "/v1/stateless/chat/completions",
        },
    }
    try:
        return mapping[provider][surface]
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Surface {surface!r} non supportée pour {provider!r}.",
        ) from exc


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _extract_response_text(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None

    direct = raw.get("output_text")
    if isinstance(direct, str):
        return direct

    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    output = raw.get("output")
    if isinstance(output, list):
        pieces: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        if pieces:
            return "".join(pieces)

    response = raw.get("response")
    if isinstance(response, dict):
        return _extract_response_text(response)

    return None


def _extract_sse_text(raw: str) -> str:
    pieces: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                pieces.append(delta["content"])
                continue

        # Best-effort support for Responses-style SSE should a tested backend
        # expose it later. The current Bridge does not promise Responses streaming.
        delta = event.get("delta")
        if isinstance(delta, str):
            pieces.append(delta)
    return "".join(pieces)


def _sse_diagnostics(raw: str) -> dict[str, Any]:
    """Summarize buffered SSE without pretending it was rendered live."""
    frame_count = 0
    event_types: list[str] = []
    last_event: str | None = None
    pending_event: str | None = None

    for line in raw.splitlines():
        if line.startswith("event:"):
            pending_event = line[6:].strip() or None
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        frame_count += 1
        event_type = pending_event
        pending_event = None
        if event_type is None and data != "[DONE]":
            try:
                decoded = json.loads(data)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                event_type = str(decoded.get("type") or "chat.completion.chunk")
        if data == "[DONE]":
            event_type = event_type or "done"
        if event_type:
            last_event = event_type
            if event_type not in event_types:
                event_types.append(event_type)

    return {
        "frame_count": frame_count,
        "event_types": event_types,
        "last_event": last_event,
        "text_reconstructed": True,
    }


def _provider_metadata(raw: Any) -> dict[str, Any]:
    """Keep useful provider fields visible without duplicating the raw body."""
    if not isinstance(raw, dict):
        return {}
    keys = (
        "id", "object", "status", "model", "created", "created_at", "usage",
        "service_tier", "system_fingerprint", "metadata", "error",
    )
    return {key: raw[key] for key in keys if key in raw}


def _auth_state(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    status = body.get("gemini_webapi")
    if isinstance(status, dict):
        status = status.get("status")
    return str(status) if status is not None else None


async def _get_json(provider: Provider, path: str) -> dict[str, Any]:
    url = _provider_base(provider) + path
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0)) as client:
            response = await client.get(url, headers=_headers(provider))
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "url": path,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:2000]
    return {
        "ok": response.is_success,
        "http_status": response.status_code,
        "url": path,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "body": body,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status/{provider}")
async def provider_status(provider: Provider) -> dict[str, Any]:
    paths = ["/health", "/v1/models"]
    if provider == "bridge":
        paths.extend(["/ready", "/v1/bridge/capabilities"])
    else:
        # WebAI's readiness is BrowserEngine/Playwright readiness. It is not a
        # readiness signal for the Gemini WebAPI transport itself.
        paths.extend(["/v1/auth/status", "/v1/runtime/status", "/ready"])
    results = {path: await _get_json(provider, path) for path in paths}
    if provider == "webai":
        process_ok = bool(results["/health"].get("ok"))
        auth_state = _auth_state(results["/v1/auth/status"].get("body"))
        webapi_ok = process_ok and auth_state == "AUTHENTICATED"
        summary = {
            "label": "opérationnel WebAPI" if webapi_ok else "joignable" if process_ok else "hors ligne",
            "ok": webapi_ok,
            "reachable": process_ok,
            "process": "OK" if process_ok else "ERR",
            "gemini_webapi_auth": auth_state or "UNKNOWN",
            "browser_readiness": "READY" if results["/ready"].get("ok") else "NOT READY / N/A pour webapi",
        }
    else:
        summary = {
            "label": "opérationnel" if results["/health"].get("ok") else "hors ligne",
            "ok": bool(results["/health"].get("ok")),
        }
    return {"provider": provider, "base_url": _provider_base(provider), "checks": results, "summary": summary}


@app.get("/api/models/{provider}")
async def models(provider: Provider) -> dict[str, Any]:
    return await _get_json(provider, "/v1/models")


@app.get("/api/poll/responses/{response_id}")
async def poll_response(
    response_id: str = FastAPIPath(..., min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._:-]+$")
) -> dict[str, Any]:
    """Retrieve one background Responses result; never resubmits the POST."""
    path = f"/v1/responses/{quote(response_id, safe='')}"
    result = await _get_json("bridge", path)
    raw = result.get("body")
    result.update({
        "provider": "bridge",
        "surface": "responses",
        "path": path,
        "extracted_text": _extract_response_text(raw),
        "provider_metadata": _provider_metadata(raw),
    })
    return result


@app.post("/api/send")
async def send(request: SendRequest) -> dict[str, Any]:
    if _json_size(request.payload) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Payload de test trop volumineux.")

    path = _surface_path(request.provider, request.surface)
    url = _provider_base(request.provider) + path
    headers = _headers(request.provider, request.idempotency_key)
    stream = bool(request.payload.get("stream", False))
    started = time.perf_counter()

    try:
        timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response_parse_error: str | None = None
            if stream:
                chunks: list[bytes] = []
                size = 0
                async with client.stream("POST", url, json=request.payload, headers=headers) as response:
                    status_code = response.status_code
                    content_type = response.headers.get("content-type", "")
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise HTTPException(status_code=413, detail="Réponse SSE trop volumineuse.")
                        chunks.append(chunk)
                raw_text = b"".join(chunks).decode("utf-8", errors="replace")
                extracted = _extract_sse_text(raw_text)
                raw: Any = raw_text
            else:
                response = await client.post(url, json=request.payload, headers=headers)
                status_code = response.status_code
                content_type = response.headers.get("content-type", "")
                body = response.content
                if len(body) > MAX_RESPONSE_BYTES:
                    raise HTTPException(status_code=413, detail="Réponse trop volumineuse.")
                try:
                    raw = response.json()
                except ValueError as exc:
                    raw = response.text
                    response_parse_error = f"Réponse non JSON ({type(exc).__name__})."
                extracted = _extract_response_text(raw)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    latency_ms = round((time.perf_counter() - started) * 1000)
    return {
        "provider": request.provider,
        "surface": request.surface,
        "path": path,
        "http_status": status_code,
        "content_type": content_type,
        "latency_ms": latency_ms,
        "stream": stream,
        "raw": raw,
        "extracted_text": extracted,
        "provider_metadata": _provider_metadata(raw),
        "response_parse_error": response_parse_error,
        "sse": _sse_diagnostics(raw) if stream else None,
    }
