"""OpenAIChatTextClient against local fake HTTP servers only."""

import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm import chat  # noqa: E402
from metaharness.llm.chat import (  # noqa: E402
    LLMError,
    LLMHTTPError,
    LLMProtocolError,
    OpenAIChatTextClient,
    validate_endpoint,
)
from metaharness.models import LLMEndpointConfig  # noqa: E402

SECRET = "sk-test-0123456789abcdef"


class Server:
    """A local server whose behavior is one function of the request handler."""

    def __init__(self, behavior):
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                owner.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": body}
                )
                behavior(self)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def send_json(handler, status: int, payload, headers: dict | None = None) -> None:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def ok(content="answer", **choice):
    return {"choices": [{"message": {"role": "assistant", "content": content}, **choice}]}


def endpoint(base_url: str, **overrides) -> LLMEndpointConfig:
    values = dict(
        base_url=base_url,
        endpoint_path="/v1/chat/completions",
        model="label",
        timeout_seconds=2,
        retries=0,
    )
    values.update(overrides)
    return LLMEndpointConfig(**values)


class WithSecret:
    def __init__(self, value: str = SECRET):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get("META_TRANSPORT_KEY")
        os.environ["META_TRANSPORT_KEY"] = self.value
        return self

    def __exit__(self, *_args):
        if self.old is None:
            os.environ.pop("META_TRANSPORT_KEY", None)
        else:
            os.environ["META_TRANSPORT_KEY"] = self.old


class RequestShapeTests(unittest.TestCase):
    def test_single_user_message_without_structured_output(self) -> None:
        with Server(lambda h: send_json(h, 200, ok())) as server, WithSecret():
            client = OpenAIChatTextClient(
                endpoint(
                    server.base_url + "/prefix/",
                    endpoint_path="v1/temporary/chat/completions",
                    api_key_env="META_TRANSPORT_KEY",
                    extra_body={"new_chat": True},
                )
            )
            self.assertEqual(client.complete("PROMPT").text, "answer")
        request = server.requests[0]
        payload = json.loads(request["body"])
        self.assertEqual(request["path"], "/prefix/v1/temporary/chat/completions")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "PROMPT"}])
        self.assertIs(payload["stream"], False)
        self.assertTrue(payload["new_chat"])
        self.assertEqual(set(payload), {"model", "messages", "stream", "new_chat"})
        self.assertEqual(request["headers"]["Authorization"], f"Bearer {SECRET}")

    def test_extra_body_cannot_replace_the_contract(self) -> None:
        for key in ("messages", "stream", "model", "response_format", "tools", "tool_choice", "functions"):
            with self.subTest(key=key):
                with self.assertRaises(LLMProtocolError):
                    OpenAIChatTextClient(endpoint("http://127.0.0.1:9", extra_body={key: 1}))

    def test_endpoint_joining_is_validated(self) -> None:
        self.assertEqual(
            validate_endpoint("http://h:1/api/", "/v1/chat/completions"),
            "http://h:1/api/v1/chat/completions",
        )
        for base, path in (
            ("file:///etc", "/passwd"),
            ("ftp://h", "/x"),
            ("http://user:pass@h", "/x"),
            ("http://h?x=1", "/x"),
            ("h:80", "/x"),
            ("http://h", "http://evil/x"),
            ("http://h", "//evil/x"),
            ("http://h", "/v1/../admin"),
            ("http://h", "/v1/chat completions"),
            ("http://h", "/x#frag"),
        ):
            with self.subTest(base=base, path=path):
                with self.assertRaises(LLMProtocolError):
                    validate_endpoint(base, path)


class RetryTests(unittest.TestCase):
    def test_retryable_statuses_are_bounded_by_configuration(self) -> None:
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                with Server(lambda h, s=status: send_json(h, s, {"error": "x"})) as server, \
                        mock.patch.object(chat, "_sleep_before_retry") as backoff:
                    with self.assertRaisesRegex(LLMHTTPError, f"HTTP {status} after 3"):
                        OpenAIChatTextClient(endpoint(server.base_url, retries=2)).complete("p")
                self.assertEqual(len(server.requests), 3)
                # A backoff precedes each retry, never the final failure.
                self.assertEqual([call.args for call in backoff.call_args_list], [(0,), (1,)])

    def test_non_retryable_statuses_are_sent_once(self) -> None:
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                with Server(lambda h, s=status: send_json(h, s, {"error": "x"})) as server:
                    with self.assertRaises(LLMHTTPError):
                        OpenAIChatTextClient(endpoint(server.base_url, retries=5)).complete("p")
                self.assertEqual(len(server.requests), 1)

    def test_network_failure_is_not_looped(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        started = time.monotonic()
        with self.assertRaisesRegex(LLMHTTPError, "before receiving"):
            OpenAIChatTextClient(endpoint(f"http://127.0.0.1:{port}", retries=5)).complete("p")
        self.assertLess(time.monotonic() - started, 2)

    def test_total_deadline_does_not_depend_on_receiving_bytes(self) -> None:
        def trickle(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.end_headers()
            try:
                for _ in range(100):
                    handler.wfile.write(b" ")
                    handler.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        with Server(trickle) as server:
            started = time.monotonic()
            with self.assertRaisesRegex(LLMHTTPError, "timed out"):
                OpenAIChatTextClient(endpoint(server.base_url, timeout_seconds=0.5)).complete("p")
            self.assertLess(time.monotonic() - started, 3)


class SecretTests(unittest.TestCase):
    def assert_clean(self, error: BaseException) -> None:
        self.assertNotIn(SECRET, str(error))
        self.assertNotIn(SECRET, repr(error))
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)

    def test_auth_failures_do_not_loop_or_leak(self) -> None:
        for status in (401, 403):
            with self.subTest(status=status):
                with Server(lambda h, s=status: send_json(h, s, {"error": SECRET})) as server, WithSecret():
                    with self.assertRaises(LLMHTTPError) as raised:
                        OpenAIChatTextClient(
                            endpoint(server.base_url, api_key_env="META_TRANSPORT_KEY", retries=3)
                        ).complete("p")
                self.assert_clean(raised.exception)
                self.assertEqual(len(server.requests), 1)

    def test_redirect_is_not_followed_and_key_is_not_forwarded(self) -> None:
        with Server(lambda h: send_json(h, 200, ok())) as target:
            def redirect(handler):
                send_json(handler, 307, b"", {"Location": target.base_url + "/steal"})

            with Server(redirect) as origin, WithSecret():
                with self.assertRaises(LLMHTTPError) as raised:
                    OpenAIChatTextClient(
                        endpoint(origin.base_url, api_key_env="META_TRANSPORT_KEY", retries=2)
                    ).complete("p")
            self.assertEqual(target.requests, [])
            self.assert_clean(raised.exception)

    def test_invalid_key_values_are_rejected_without_echo(self) -> None:
        for value in (SECRET + "\nX-Injected: 1", SECRET + " tail", SECRET + "é", ""):
            with self.subTest(value=repr(value)):
                with Server(lambda h: send_json(h, 200, ok())) as server, WithSecret(value):
                    with self.assertRaises(LLMError) as raised:
                        OpenAIChatTextClient(
                            endpoint(server.base_url, api_key_env="META_TRANSPORT_KEY")
                        ).complete("p")
                self.assertNotIn(SECRET, str(raised.exception))
                self.assertIn("META_TRANSPORT_KEY", str(raised.exception))
                self.assertEqual(server.requests, [])

    def test_protocol_failures_do_not_leak(self) -> None:
        for behavior in (
            lambda h: send_json(h, 200, b"not json " + SECRET.encode()),
            lambda h: send_json(h, 200, [1, 2]),
            lambda h: send_json(h, 500, {"error": SECRET}),
        ):
            with Server(behavior) as server, WithSecret():
                with self.assertRaises(LLMError) as raised:
                    OpenAIChatTextClient(
                        endpoint(server.base_url, api_key_env="META_TRANSPORT_KEY")
                    ).complete("p")
            self.assertNotIn(SECRET, str(raised.exception))


class ResponseParsingTests(unittest.TestCase):
    def complete(self, payload):
        with Server(lambda h: send_json(h, 200, payload)) as server:
            return OpenAIChatTextClient(endpoint(server.base_url)).complete("p")

    def test_standard_and_part_content(self) -> None:
        self.assertEqual(self.complete(ok("plain", finish_reason="stop")).text, "plain")
        parts = [{"type": "text", "text": "a"}, {"type": "output_text", "text": "b"}]
        self.assertEqual(self.complete(ok(parts)).text, "ab")

    def test_informational_fields_are_tolerated(self) -> None:
        payload = ok("x")
        payload.update(model=["odd"], usage={"prompt_tokens": 1.5, "total_tokens": 3, "details": {}})
        result = self.complete(payload)
        self.assertIsNone(result.model)
        self.assertEqual(result.usage, {"total_tokens": 3})

    def test_absent_or_malformed_content_fails_clearly(self) -> None:
        cases = {
            "missing a choices": {"object": "chat.completion"},
            "choices array is empty": {"choices": []},
            "missing a message": {"choices": [{"text": "plain"}]},
            "missing content": {"choices": [{"message": {"role": "assistant"}}]},
            "null": ok(None),
            "no usable text": ok("   "),
            "not a standard text part": ok([{"type": "image_url", "image_url": {}}]),
            "incomplete": ok("half a plan", finish_reason="length"),
            "must be an object": ["not", "an", "object"],
        }
        for message, payload in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(LLMProtocolError, message):
                    self.complete(payload)
        tool_call = {
            "choices": [
                {"message": {"content": None, "tool_calls": [{"id": "1"}]}, "finish_reason": "tool_calls"}
            ]
        }
        with self.assertRaises(LLMProtocolError):
            self.complete(tool_call)

    def test_oversized_response_is_rejected(self) -> None:
        with mock.patch.object(chat, "_MAX_RESPONSE_BYTES", 100):
            with self.assertRaisesRegex(LLMProtocolError, "too large"):
                self.complete(ok("x" * 1000))


if __name__ == "__main__":
    unittest.main()
