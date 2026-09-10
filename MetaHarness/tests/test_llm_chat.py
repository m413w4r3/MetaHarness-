import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.chat import (  # noqa: E402
    LLMHTTPError,
    LLMProtocolError,
    OpenAIChatTextClient,
)
from metaharness.models import LLMEndpointConfig  # noqa: E402


class ServerHarness:
    def __init__(self, responder):
        self.requests = []
        self._responder = responder

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                owner.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "payload": json.loads(body),
                    }
                )
                status, response = owner._responder(self, owner.requests[-1])
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def config(server, **kwargs):
    values = {
        "base_url": server.base_url,
        "endpoint_path": "/v1/chat/completions",
        "model": "test-model",
        "timeout_seconds": 2,
        "retries": 0,
    }
    values.update(kwargs)
    return LLMEndpointConfig(**values)


class ChatClientTests(unittest.TestCase):
    def test_standard_completion_has_one_user_message_and_text_only_payload(self):
        def responder(_handler, _request):
            return 200, {
                "id": "completion-1",
                "model": "served-model",
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }

        with ServerHarness(responder) as server:
            result = OpenAIChatTextClient(config(server)).complete("Say hello")

        request = server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(
            request["payload"],
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
            },
        )
        self.assertNotIn("response_format", request["payload"])
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.model, "served-model")
        self.assertEqual(result.usage["prompt_tokens"], 2)

    def test_configurable_chatgpt_like_and_temporary_gemini_like_endpoints(self):
        def responder(_handler, _request):
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        with ServerHarness(responder) as server:
            client = OpenAIChatTextClient(
                config(server, endpoint_path="/v1/temporary/chat/completions")
            )
            self.assertEqual(client.complete("prompt").text, "ok")
            self.assertEqual(server.requests[0]["path"], "/v1/temporary/chat/completions")

    def test_extra_body_is_merged_but_cannot_override_wire_contract(self):
        def responder(_handler, _request):
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        with ServerHarness(responder) as server:
            OpenAIChatTextClient(
                config(server, extra_body={"new_chat": True, "temperature": 0.2})
            ).complete("prompt")
            self.assertTrue(server.requests[0]["payload"]["new_chat"])
            self.assertEqual(server.requests[0]["payload"]["temperature"], 0.2)
            for protected in ("messages", "stream", "model", "response_format"):
                with self.assertRaises(LLMProtocolError):
                    OpenAIChatTextClient(config(server, extra_body={protected: "bad"}))

    def test_string_and_text_content_parts_are_supported(self):
        responses = iter(
            [
                {"choices": [{"message": {"content": "plain"}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "one"},
                                    {"type": "text", "text": " two"},
                                ]
                            }
                        }
                    ]
                },
            ]
        )

        def responder(_handler, _request):
            return 200, next(responses)

        with ServerHarness(responder) as server:
            client = OpenAIChatTextClient(config(server))
            self.assertEqual(client.complete("one").text, "plain")
            self.assertEqual(client.complete("two").text, "one two")

    def test_usage_aliases_are_normalized_and_usage_may_be_absent(self):
        def responder(_handler, request):
            if request["payload"]["messages"][0]["content"] == "aliases":
                return 200, {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
                }
            return 200, {"choices": [{"message": {"content": "ok"}}]}

        with ServerHarness(responder) as server:
            client = OpenAIChatTextClient(config(server))
            self.assertEqual(
                client.complete("aliases").usage,
                {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            )
            self.assertEqual(client.complete("none").usage, {})

    def test_retryable_500_is_retried_then_succeeds(self):
        attempts = 0

        def responder(_handler, _request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return 500, {"error": "temporary"}
            return 200, {"choices": [{"message": {"content": "recovered"}}]}

        with ServerHarness(responder) as server:
            result = OpenAIChatTextClient(config(server, retries=1)).complete("prompt")
        self.assertEqual(result.text, "recovered")
        self.assertEqual(attempts, 2)

    def test_401_is_not_retried_and_secret_is_not_in_exception(self):
        def responder(_handler, _request):
            return 401, {"error": "unauthorized"}

        secret = "super-secret-key-value"
        old_value = os.environ.get("META_TEST_SECRET")
        os.environ["META_TEST_SECRET"] = secret
        try:
            with ServerHarness(responder) as server:
                with self.assertRaises(LLMHTTPError) as raised:
                    OpenAIChatTextClient(
                        config(server, api_key_env="META_TEST_SECRET", retries=3)
                    ).complete("prompt")
                self.assertNotIn(secret, str(raised.exception))
                self.assertEqual(len(server.requests), 1)
        finally:
            if old_value is None:
                os.environ.pop("META_TEST_SECRET", None)
            else:
                os.environ["META_TEST_SECRET"] = old_value

    def test_timeout_is_reported_without_exposing_headers(self):
        def responder(_handler, _request):
            time.sleep(0.2)
            return 200, {"choices": [{"message": {"content": "late"}}]}

        with ServerHarness(responder) as server:
            with self.assertRaises(LLMHTTPError) as raised:
                OpenAIChatTextClient(
                    config(server, timeout_seconds=0.02, retries=0)
                ).complete("prompt")
        self.assertIn("timed out", str(raised.exception))
        self.assertNotIn("Authorization", str(raised.exception))

    def test_malformed_response_is_a_protocol_error(self):
        def responder(_handler, _request):
            return 200, {"choices": []}

        with ServerHarness(responder) as server:
            with self.assertRaisesRegex(LLMProtocolError, "empty"):
                OpenAIChatTextClient(config(server)).complete("prompt")


if __name__ == "__main__":
    unittest.main()
