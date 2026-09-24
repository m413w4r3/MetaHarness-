"""Transport HTTP minimal pour les endpoints OpenAI-compatible texte.

Contrat : ``POST`` sur un endpoint configurable, exactement un message
``user``, ``stream=false``, réponse lue dans ``choices[0].message.content``.
Aucun message système, aucun ``response_format``, aucune sortie JSON exigée
du modèle.  La clé API n'apparaît jamais dans une exception, un état ou un
artefact.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ..models import LLMEndpointConfig


class LLMError(RuntimeError):
    """Erreur générale du transport ou du protocole LLM."""


class LLMHTTPError(LLMError):
    """Erreur de transport HTTP ou réseau."""


class LLMProtocolError(LLMError):
    """Réponse ou configuration incompatible avec le protocole attendu."""


class ConversationUnavailableError(LLMError):
    """The driver explicitly reports that a conversation cannot be continued."""


@dataclass(frozen=True)
class LLMConversationHandle:
    """A stable conversation identifier officially exposed by a driver.

    MetaHarness never fabricates one and never scrapes a UI to guess it: a
    handle exists only when the driver/bridge returns it.  A handle may be
    exposed by a driver for explicitly conversation-aware workflows.
    A planning transaction may continue this conversation while validation
    rejects planner answers. A later review repair starts a new transaction.
    """

    provider_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        for name in ("provider_id", "conversation_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or any(not character.isprintable() for character in value)
            ):
                raise ValueError(f"conversation {name} is invalid")


def conversation_handle(result: object) -> LLMConversationHandle | None:
    """The handle a completion officially carries, else ``None``."""

    handle = getattr(result, "conversation", None)
    return handle if isinstance(handle, LLMConversationHandle) else None


@runtime_checkable
class ConversationContinuationClient(Protocol):
    def continue_conversation(
        self, handle: LLMConversationHandle, prompt: str,
    ) -> "TextLLMResult | str": ...


@dataclass(frozen=True)
class TextFileAttachment:
    """One bounded UTF-8 text file offered to the endpoint as an attachment.

    An attachment is a transport mode for evidence that is already durable in
    the run artifacts; it never carries protocol authority.
    """

    filename: str
    text: str
    media_type: str = "text/markdown"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or len(self.filename) > 128
            or "/" in self.filename
            or "\\" in self.filename
            or "\x00" in self.filename
        ):
            raise ValueError("attachment filename is invalid")

        if not isinstance(self.text, str):
            raise TypeError("attachment text must be a string")

        if (
            not isinstance(self.media_type, str)
            or not re.fullmatch(
                r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+",
                self.media_type,
            )
        ):
            raise ValueError("attachment media type is invalid")


def _attachment_part(
    attachment: TextFileAttachment,
) -> dict[str, Any]:
    encoded = base64.b64encode(
        attachment.text.encode("utf-8")
    ).decode("ascii")

    return {
        "type": "input_file",
        "file": {
            "filename": attachment.filename,
            "file_data": (
                f"data:{attachment.media_type};base64,{encoded}"
            ),
        },
    }


@dataclass(frozen=True)
class TextLLMResult:
    text: str
    model: str | None
    usage: dict[str, int]
    raw_response: dict
    # Set only by a driver that exposes a stable conversation id.  The
    # OpenAI-compatible bridge client never sets it.
    conversation: LLMConversationHandle | None = None


_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
# Keys that would change the wire contract (message list, streaming, model,
# structured or tool output) cannot come from static extra_body.
PROTECTED_BODY_KEYS = frozenset(
    {
        "messages",
        "stream",
        "stream_options",
        "model",
        "response_format",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
    }
)
_PROTECTED_BODY_KEYS = PROTECTED_BODY_KEYS
_MAX_BACKOFF_SECONDS = 1.0
_INITIAL_BACKOFF_SECONDS = 0.05
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_TRUNCATED_FINISH_REASONS = frozenset({"length", "content_filter"})
_TEXT_PART_TYPES = frozenset({"text", "output_text"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: a POST must not be replayed elsewhere, and the
    Authorization header must never be forwarded to another location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


class OpenAIChatTextClient:
    """Client d'un endpoint chat OpenAI-compatible ne retournant que du texte."""

    def __init__(
        self,
        config: LLMEndpointConfig,
        *,
        environment: Mapping[str, str] | None = None,
        on_transport: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        # ``None`` deliberately retains the old library-level behavior for
        # callers/tests; MetaHarness always supplies its runtime mapping.
        self._environment = os.environ if environment is None else environment
        self._url = _join_url(config.base_url, config.endpoint_path)
        conflicting_keys = PROTECTED_BODY_KEYS.intersection(config.extra_body)
        if conflicting_keys:
            keys = ", ".join(sorted(conflicting_keys))
            raise LLMProtocolError(
                f"extra_body cannot override protected request keys: {keys}"
            )
        self._opener = _opener()
        self._on_transport = on_transport

    def _transport_event(self, name: str, **data: Any) -> None:
        callback = self._on_transport
        if callback is not None:
            try:
                callback({"event": name, "operation": "chat_completion", **data})
            except Exception:
                # Observation must never affect a model request.
                pass

    def complete(self, prompt: str) -> TextLLMResult:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        request = self._build_request(prompt)
        response_data = self._request_json(request)
        return _parse_completion_response(response_data)

    def complete_with_file_fallback(
        self,
        prompt: str,
        *,
        fallback_prompt: str,
        attachments: tuple[TextFileAttachment, ...],
        fallback_attempt: int = 3,
    ) -> TextLLMResult:
        """Send *prompt* inline, moving evidence to files only late.

        The attachments are used exactly when the existing retry loop reaches
        *fallback_attempt*; no extra attempt is ever added, and a non-retryable
        status still fails on its own attempt.
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(fallback_prompt, str):
            raise TypeError("fallback_prompt must be a string")
        if not isinstance(attachments, tuple) or not attachments:
            raise TypeError("attachments must be a non-empty tuple")
        if any(not isinstance(item, TextFileAttachment) for item in attachments):
            raise TypeError("attachments must contain TextFileAttachment values")
        if (
            isinstance(fallback_attempt, bool)
            or not isinstance(fallback_attempt, int)
            or fallback_attempt < 2
        ):
            raise ValueError("fallback_attempt must be an integer of at least 2")

        response_data = self._request_json_with_factory(
            lambda attempt: (
                self._build_request(prompt)
                if attempt < fallback_attempt
                else self._build_request(
                    [
                        {
                            "type": "text",
                            "text": fallback_prompt,
                        },
                        *[
                            _attachment_part(item)
                            for item in attachments
                        ],
                    ]
                )
            )
        )
        return _parse_completion_response(response_data)

    def _build_request(
        self,
        content: str | list[dict[str, Any]],
    ) -> urllib.request.Request:
        payload: dict[str, Any] = dict(self.config.extra_body)
        # Protected keys are written last: even a mutated extra_body cannot
        # replace the single user message or the non-streaming contract.
        payload.update(
            {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "stream": False,
            }
        )

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key_env is not None:
            headers["Authorization"] = f"Bearer {_api_key(self.config.api_key_env, self._environment)}"

        return urllib.request.Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _request_json(self, request: urllib.request.Request) -> dict[str, Any]:
        return self._request_json_with_factory(lambda _attempt: request)

    def _request_json_with_factory(
        self,
        request_factory: Callable[[int], urllib.request.Request],
    ) -> dict[str, Any]:
        attempts = self.config.retries + 1
        started = time.monotonic()
        self._transport_event("request_started", attempts=attempts)
        for attempt in range(attempts):
            # 1-based: the factory decides what attempt #N carries.
            request = request_factory(attempt + 1)
            attempt_started = time.monotonic()
            self._transport_event("attempt_started", attempt=attempt + 1, attempts=attempts)
            try:
                deadline = time.monotonic() + self.config.timeout_seconds
                with self._opener.open(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = _read_bounded(response, deadline)
            except urllib.error.HTTPError as exc:
                status = exc.code
                elapsed_ms = round((time.monotonic() - attempt_started) * 1000)
                try:
                    exc.close()
                except OSError:
                    pass
                if status in _RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                    self._transport_event("http_response", attempt=attempt + 1, attempts=attempts, http_status=status, elapsed_ms=elapsed_ms)
                    self._transport_event("retrying", attempt=attempt + 1, attempts=attempts, http_status=status)
                    _sleep_before_retry(attempt)
                    continue
                self._transport_event("http_response", attempt=attempt + 1, attempts=attempts, http_status=status, elapsed_ms=elapsed_ms)
                self._transport_event("waiting_external", attempt=attempt + 1, attempts=attempts, http_status=status, elapsed_ms=round((time.monotonic() - started) * 1000))
                raise LLMHTTPError(
                    f"LLM endpoint returned HTTP {status} after {attempt + 1} attempt(s)"
                ) from None
            except LLMError:
                raise
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                self._transport_event("waiting_external", attempt=attempt + 1, attempts=attempts, elapsed_ms=round((time.monotonic() - started) * 1000))
                if _is_timeout_error(exc):
                    message = "LLM request timed out"
                else:
                    reason = getattr(exc, "reason", exc)
                    message = (
                        "LLM request failed before receiving an HTTP response "
                        f"({type(reason).__name__})"
                    )
                raise LLMHTTPError(message) from None
            except (http.client.HTTPException, ValueError) as exc:
                # http.client errors (and invalid header values) can embed
                # request data; only the class name is reported.
                raise LLMHTTPError(
                    f"LLM request failed ({type(exc).__name__})"
                ) from None

            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise LLMProtocolError(
                    "LLM endpoint returned invalid JSON"
                ) from None
            if not isinstance(decoded, dict):
                raise LLMProtocolError("LLM endpoint JSON response must be an object")
            self._transport_event("request_completed", attempt=attempt + 1, attempts=attempts, elapsed_ms=round((time.monotonic() - started) * 1000))
            return decoded

        raise LLMHTTPError("LLM endpoint request failed")  # pragma: no cover


def _api_key(env_name: str, environment: Mapping[str, str] | None = None) -> str:
    source = os.environ if environment is None else environment
    try:
        api_key = source[env_name]
    except KeyError:
        raise LLMError(
            f"API key environment variable {env_name!r} is not set"
        ) from None
    # A control character or space would make http.client raise an error that
    # quotes the header value; reject it here without echoing the value.
    if not api_key or any(not 0x21 <= ord(character) <= 0x7E for character in api_key):
        raise LLMError(
            f"API key environment variable {env_name!r} contains an invalid value"
        )
    return api_key


def _read_bounded(response: Any, deadline: float) -> bytes:
    """Read a response body under a total deadline and a size bound."""

    # ``read1`` returns after one underlying read, so the deadline is checked
    # even when a server trickles bytes; ``read(n)`` would wait for n bytes.
    read = getattr(response, "read1", response.read)
    chunks: list[bytes] = []
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise LLMHTTPError("LLM request timed out")
        chunk = read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise LLMProtocolError("LLM endpoint response is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def validate_endpoint(base_url: str, endpoint_path: str) -> str:
    """Return the joined request URL or raise :class:`LLMProtocolError`."""

    return _join_url(base_url, endpoint_path)


def _join_url(base_url: str, endpoint_path: str) -> str:
    if not isinstance(base_url, str) or not isinstance(endpoint_path, str):
        raise TypeError("base_url and endpoint_path must be strings")
    for label, value in (("base_url", base_url), ("endpoint_path", endpoint_path)):
        if any(ord(character) < 0x21 or ord(character) == 0x7F for character in value):
            raise LLMProtocolError(f"{label} must not contain whitespace or control characters")
    base = urllib.parse.urlsplit(base_url)
    if base.scheme not in {"http", "https"} or not base.hostname:
        raise LLMProtocolError("base_url must be an absolute http(s) URL")
    if base.username is not None or base.password is not None:
        raise LLMProtocolError("base_url must not embed credentials; use api_key_env")
    if base.query or base.fragment:
        raise LLMProtocolError("base_url must not contain a query or fragment")
    endpoint = urllib.parse.urlsplit(endpoint_path)
    if endpoint.scheme or endpoint.netloc or endpoint_path.startswith("//"):
        raise LLMProtocolError("endpoint_path must be a path, not a URL")
    if endpoint.fragment:
        raise LLMProtocolError("endpoint_path must not contain a fragment")
    if any(segment in {".", ".."} for segment in endpoint.path.split("/")):
        raise LLMProtocolError("endpoint_path must not contain . or .. segments")
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
    finish_reason = first_choice.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason in _TRUNCATED_FINISH_REASONS:
        # A truncated plan or review may still look well formed; never use it.
        raise LLMProtocolError(
            f"LLM response is incomplete (finish_reason={finish_reason})"
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMProtocolError("LLM response choice is missing a message object")
    if "content" not in message:
        raise LLMProtocolError("LLM response message is missing content")
    text = _extract_text_content(message["content"])

    # Model and usage are informational: a provider-specific shape must not
    # make an otherwise valid text answer unusable.
    model = response.get("model")
    if not isinstance(model, str):
        model = None

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
    if content is None:
        raise LLMProtocolError("LLM response content is null (no text answer)")
    if not isinstance(content, list):
        raise LLMProtocolError("LLM response content must be text or text parts")
    if not content:
        raise LLMProtocolError("LLM response content parts are empty")

    parts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in _TEXT_PART_TYPES:
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
    if not isinstance(value, dict):
        return {}

    usage: dict[str, int] = {}
    for canonical, aliases in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens")),
        ("completion_tokens", ("completion_tokens", "output_tokens")),
        ("total_tokens", ("total_tokens",)),
    ):
        for alias in aliases:
            number = value.get(alias)
            if isinstance(number, int) and not isinstance(number, bool):
                usage[canonical] = number
                break
    # Cache and reasoning counters are kept only when the provider supplies
    # them, flat or in OpenAI-style ``*_details`` objects.
    for canonical, aliases, nested in (
        (
            "cached_input_tokens",
            ("cached_input_tokens", "cache_read_input_tokens"),
            (("prompt_tokens_details", "cached_tokens"), ("input_tokens_details", "cached_tokens")),
        ),
        (
            "cache_write_input_tokens",
            ("cache_write_input_tokens", "cache_creation_input_tokens"),
            (("prompt_tokens_details", "cache_write_tokens"), ("input_tokens_details", "cache_write_tokens")),
        ),
        (
            "reasoning_output_tokens",
            ("reasoning_output_tokens",),
            (("completion_tokens_details", "reasoning_tokens"), ("output_tokens_details", "reasoning_tokens")),
        ),
    ):
        candidates = [value.get(alias) for alias in aliases]
        candidates.extend(
            value[container].get(key)
            for container, key in nested
            if isinstance(value.get(container), dict)
        )
        for number in candidates:
            if isinstance(number, int) and not isinstance(number, bool):
                usage[canonical] = number
                break
    return usage
