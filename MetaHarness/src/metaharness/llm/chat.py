"""Transport HTTP minimal pour les endpoints OpenAI-compatible texte."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..models import LLMEndpointConfig


class LLMError(RuntimeError):
    """Erreur générale du transport ou du protocole LLM."""


class LLMHTTPError(LLMError):
    """Erreur de transport HTTP ou réseau."""


class LLMProtocolError(LLMError):
    """Réponse ou configuration incompatible avec le protocole attendu."""


@dataclass(frozen=True)
class TextLLMResult:
    text: str
    model: str | None
    usage: dict[str, int]
    raw_response: dict


_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_PROTECTED_BODY_KEYS = frozenset(
    {"messages", "stream", "model", "response_format"}
)
_MAX_BACKOFF_SECONDS = 1.0
_INITIAL_BACKOFF_SECONDS = 0.05


class OpenAIChatTextClient:
    """Client d'un endpoint chat OpenAI-compatible ne retournant que du texte."""

    def __init__(self, config: LLMEndpointConfig):
        self.config = config
        self._url = _join_url(config.base_url, config.endpoint_path)
        conflicting_keys = _PROTECTED_BODY_KEYS.intersection(config.extra_body)
        if conflicting_keys:
            keys = ", ".join(sorted(conflicting_keys))
            raise LLMProtocolError(
                f"extra_body cannot override protected request keys: {keys}"
            )

    def complete(self, prompt: str) -> TextLLMResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        }
        payload.update(self.config.extra_body)

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key_env is not None:
            try:
                api_key = os.environ[self.config.api_key_env]
            except KeyError as exc:
                raise LLMError(
                    f"API key environment variable {self.config.api_key_env!r} is not set"
                ) from exc
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        response_data = self._request_json(request)
        return _parse_completion_response(response_data)

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        attempts = self.config.retries + 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    exc.close()
                except OSError:
                    pass
                if status in _RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                    _sleep_before_retry(attempt)
                    continue
                raise LLMHTTPError(
                    f"LLM endpoint returned HTTP {status} after {attempt + 1} attempt(s)"
                ) from None
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                if _is_timeout_error(exc):
                    message = "LLM request timed out"
                else:
                    message = "LLM request failed before receiving an HTTP response"
                raise LLMHTTPError(message) from None

            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise LLMProtocolError(
                    "LLM endpoint returned invalid JSON"
                ) from None
            if not isinstance(decoded, dict):
                raise LLMProtocolError("LLM endpoint JSON response must be an object")
            return decoded

        raise LLMHTTPError("LLM endpoint request failed")  # pragma: no cover


def _join_url(base_url: str, endpoint_path: str) -> str:
    if not isinstance(base_url, str) or not isinstance(endpoint_path, str):
        raise TypeError("base_url and endpoint_path must be strings")
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _sleep_before_retry(attempt: int) -> None:
    delay = min(_INITIAL_BACKOFF_SECONDS * (2**attempt), _MAX_BACKOFF_SECONDS)
    time.sleep(delay)


def _is_timeout_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def _parse_completion_response(response: dict[str, Any]) -> TextLLMResult:
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise LLMProtocolError("LLM response is missing a choices array")
    if not choices:
        raise LLMProtocolError("LLM response choices array is empty")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMProtocolError("LLM response first choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMProtocolError("LLM response choice is missing a message object")
    if "content" not in message:
        raise LLMProtocolError("LLM response message is missing content")
    text = _extract_text_content(message["content"])

    model = response.get("model")
    if model is not None and not isinstance(model, str):
        raise LLMProtocolError("LLM response model must be a string or null")

    return TextLLMResult(
        text=text,
        model=model,
        usage=_normalize_usage(response.get("usage")),
        raw_response=response,
    )


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        if not content.strip():
            raise LLMProtocolError("LLM response content contains no usable text")
        return content
    if not isinstance(content, list):
        raise LLMProtocolError("LLM response content must be text or text parts")
    if not content:
        raise LLMProtocolError("LLM response content parts are empty")

    parts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") != "text":
            raise LLMProtocolError(
                f"LLM response content part {index} is not a standard text part"
            )
        part_text = part.get("text")
        if not isinstance(part_text, str):
            raise LLMProtocolError(
                f"LLM response content text part {index} has no text value"
            )
        parts.append(part_text)
    text = "".join(parts)
    if not text.strip():
        raise LLMProtocolError("LLM response content contains no usable text")
    return text


def _normalize_usage(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LLMProtocolError("LLM response usage must be an object")

    usage: dict[str, int] = {}
    for canonical, aliases in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
        ("total_tokens", ("total_tokens",)),
    ):
        for alias in aliases:
            if alias in value:
                number = value[alias]
                if isinstance(number, bool) or not isinstance(number, int):
                    raise LLMProtocolError(
                        f"LLM response usage field {alias!r} must be an integer"
                    )
                usage[canonical] = number
                break
    return usage
