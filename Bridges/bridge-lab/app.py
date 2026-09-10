from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator
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
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


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


def _parse_diagnostics(text: str | None, schema: dict[str, Any] | None) -> dict[str, Any]:
    if text is None:
        return {
            "has_text": False,
            "strict_json": False,
            "schema_valid": None,
            "fenced": False,
            "error": "Aucun texte assistant extrait.",
        }

    result: dict[str, Any] = {
        "has_text": True,
        "chars": len(text),
        "fenced": "```" in text,
        "strict_json": False,
        "schema_valid": None,
        "json": None,
        "error": None,
    }
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        result["error"] = f"JSON strict invalide: {exc.msg} (ligne {exc.lineno}, colonne {exc.colno})"
        return result

    result["strict_json"] = True
    result["json"] = value

    if schema is not None:
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            result["schema_valid"] = False
            result["schema_errors"] = [
                {
                    "path": list(error.absolute_path),
                    "message": error.message,
                    "validator": error.validator,
                }
                for error in errors[:25]
            ]
        else:
            result["schema_valid"] = True
    return result


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
    paths = ["/health", "/ready", "/v1/models"]
    if provider == "bridge":
        paths.append("/v1/bridge/capabilities")
    else:
        paths.extend(["/v1/auth/status", "/v1/runtime/status"])
    results = {path: await _get_json(provider, path) for path in paths}
    return {"provider": provider, "base_url": _provider_base(provider), "checks": results}


@app.get("/api/models/{provider}")
async def models(provider: Provider) -> dict[str, Any]:
    return await _get_json(provider, "/v1/models")


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
                except ValueError:
                    raw = response.text
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
        "parse": _parse_diagnostics(extracted, request.schema_),
    }
