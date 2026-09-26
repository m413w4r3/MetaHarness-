import base64
import email.message
import io
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm import chat  # noqa: E402
from metaharness.llm.chat import (  # noqa: E402
    LLMHTTPError,
    LLMProtocolError,
    LLMTransportExhaustedError,
    OpenAIChatTextClient,
    TextFileAttachment,
)
from metaharness.models import LLMEndpointConfig  # noqa: E402


class FakeClock:
    """A monotonic clock whose sleeps advance it: no test ever waits."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.started = start
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)

    @property
    def elapsed(self) -> float:
        return self.now - self.started


class FakeResponse:
    """The minimal response surface ``_read_bounded`` consumes."""

    def __init__(self, body: bytes) -> None:
        self._buffer = io.BytesIO(body)

    def read1(self, size: int = -1) -> bytes:
        return self._buffer.read1(size)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


class FakeOpener:
    """An in-process transport scripted by (status, headers, body) answers.

    No socket is opened: every answer is either an HTTP status the client
    rejects, a response body, or a local exception the real urllib raises.
    """

    def __init__(self, answers: list, clock: FakeClock | None = None) -> None:
        self.answers = list(answers)
        self.requests: list[dict] = []
        self.clock = clock
        # The elapsed transport time of every attempt, so a test can prove no
        # attempt was ever made after the deadline.
        self.times: list[float] = []

    def open(self, request, timeout=None):  # noqa: ANN001
        body = request.data
        self.requests.append(
            {"url": request.full_url, "payload": json.loads(body.decode("utf-8"))}
        )
        if self.clock is not None:
            self.times.append(self.clock.elapsed)
        answer = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        status, headers, payload = answer
        if status != 200:
            raise urllib.error.HTTPError(
                request.full_url, status, "error", email_message(headers), None,
            )
        return FakeResponse(json.dumps(payload).encode("utf-8"))


def email_message(headers: dict):
    message = email.message.Message()
    for key, value in headers.items():
        message[key] = value
    return message


def completion(text="ok"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class HorizonHarness:
    """One client over a scripted opener and an injected clock."""

    def __init__(self, answers, **overrides) -> None:
        self.clock = FakeClock()
        self.opener = FakeOpener(answers, self.clock)
        values = {
            "base_url": "http://127.0.0.1:9",
            "endpoint_path": "/v1/chat/completions",
            "model": "test-model",
            "timeout_seconds": 2,
            "max_wait_seconds": 60,
        }
        values.update(overrides)
        self.endpoint = LLMEndpointConfig(**values)
        self.observations: list[dict] = []

    def client(self) -> OpenAIChatTextClient:
        return OpenAIChatTextClient(
            self.endpoint, environment={}, on_transport=self.observations.append,
            opener=self.opener,
        )

    def call(self, prompt: str = "prompt"):
        with (
            mock.patch("time.monotonic", self.clock.monotonic),
            mock.patch("time.sleep", self.clock.sleep),
        ):
            return self.client().complete(prompt)


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
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

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
        "max_wait_seconds": 60,
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
                        config(server, api_key_env="META_TEST_SECRET")
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
            # A one-second horizon: the retryable timeout is reported as a
            # temporary exhaustion without waiting for the default budget.
            with self.assertRaises(LLMHTTPError) as raised:
                OpenAIChatTextClient(
                    config(server, timeout_seconds=0.02, max_wait_seconds=1)
                ).complete("prompt")
        self.assertIn("timed out", str(raised.exception))
        self.assertNotIn("Authorization", str(raised.exception))

    def test_malformed_response_is_a_protocol_error(self):
        def responder(_handler, _request):
            return 200, {"choices": []}

        with ServerHarness(responder) as server:
            with self.assertRaisesRegex(LLMProtocolError, "empty"):
                OpenAIChatTextClient(config(server)).complete("prompt")


def _completion(text="ok"):
    return {"choices": [{"message": {"content": text}}]}


def _content(request):
    return request["payload"]["messages"][0]["content"]


def _payload_content(request):
    return request["payload"]["messages"][0]["content"]


class TransportHorizonTests(unittest.TestCase):
    """The transport retries inside a deadline, never a fixed attempt count."""

    def test_a_transient_503_burst_then_success_retries_with_growing_waits(self):
        harness = HorizonHarness([
            (503, {}, {"error": "unavailable"}),
            (503, {}, {"error": "unavailable"}),
            (200, {}, completion("recovered")),
        ])

        result = harness.call()

        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(harness.opener.requests), 3)
        self.assertEqual(len(harness.clock.sleeps), 2)
        self.assertEqual(harness.clock.sleeps, sorted(harness.clock.sleeps))
        self.assertLess(harness.clock.sleeps[0], harness.clock.sleeps[1])
        # The whole retry sequence happens inside the configured horizon.
        self.assertLess(harness.clock.sleeps[-1], harness.endpoint.max_wait_seconds)
        self.assertEqual(
            [item["event"] for item in harness.observations],
            [
                "request_started", "attempt_started", "http_response", "retrying",
                "attempt_started", "http_response", "retrying",
                "attempt_started", "request_completed",
            ],
        )
        self.assertEqual(harness.observations[0]["max_wait_seconds"], 60)
        self.assertNotIn("prompt", json.dumps(harness.observations))

    def test_a_valid_retry_after_wins_over_the_local_backoff(self):
        harness = HorizonHarness([
            (429, {"Retry-After": "7"}, {"error": "slow down"}),
            (200, {}, completion("accepted")),
        ])

        result = harness.call()

        self.assertEqual(result.text, "accepted")
        self.assertEqual(harness.clock.sleeps, [7.0])

    def test_an_http_date_retry_after_is_honoured(self):
        target = datetime.now(timezone.utc) + timedelta(seconds=9)
        harness = HorizonHarness([
            (503, {"Retry-After": target.strftime("%a, %d %b %Y %H:%M:%S GMT")}, {}),
            (200, {}, completion("accepted")),
        ])

        result = harness.call()

        self.assertEqual(result.text, "accepted")
        self.assertEqual(len(harness.clock.sleeps), 1)
        self.assertAlmostEqual(harness.clock.sleeps[0], 9.0, delta=1.0)

    def test_an_invalid_retry_after_falls_back_to_the_local_backoff(self):
        for headers in ({"Retry-After": "soon"}, {"Retry-After": ""}, {"Retry-After": "0"}):
            with self.subTest(headers=headers):
                harness = HorizonHarness([
                    (429, headers, {}),
                    (200, {}, completion("accepted")),
                ])

                self.assertEqual(harness.call().text, "accepted")
                # The local backoff starts at two seconds, with bounded jitter.
                self.assertEqual(len(harness.clock.sleeps), 1)
                self.assertAlmostEqual(harness.clock.sleeps[0], 2.0, delta=0.5)

    def test_reaching_the_deadline_signals_a_temporary_exhaustion(self):
        harness = HorizonHarness(
            [(503, {}, {"error": "unavailable"})], max_wait_seconds=10,
        )

        with self.assertRaises(LLMTransportExhaustedError) as raised:
            harness.call()

        self.assertEqual(raised.exception.code, "LLM_TRANSPORT_EXHAUSTED")
        self.assertIn("horizon of 10s", str(raised.exception))
        self.assertIn("HTTP 503", str(raised.exception))
        # No attempt is ever made after the deadline, and the waits grow
        # instead of spinning: the transport waits out its horizon once.
        self.assertTrue(harness.opener.times)
        self.assertTrue(all(offset < 10 for offset in harness.opener.times))
        self.assertGreaterEqual(harness.clock.elapsed, 10)
        self.assertLessEqual(len(harness.opener.requests), 4)
        self.assertGreaterEqual(len(harness.clock.sleeps), 2)
        self.assertTrue(all(wait > 0 for wait in harness.clock.sleeps))
        # The final wait is clamped to what the horizon still allowed.
        self.assertLessEqual(sum(harness.clock.sleeps), 10.0 + 1e-6)

    def test_a_non_retryable_status_is_sent_once_without_sleeping(self):
        harness = HorizonHarness([(401, {}, {"error": "unauthorized"})])

        with self.assertRaises(LLMHTTPError) as raised:
            harness.call()

        self.assertNotIsInstance(raised.exception, LLMTransportExhaustedError)
        self.assertEqual(str(raised.exception), "LLM endpoint returned HTTP 401")
        self.assertEqual(len(harness.opener.requests), 1)
        self.assertEqual(harness.clock.sleeps, [])

    def test_a_timeout_then_success_recovers_inside_the_horizon(self):
        harness = HorizonHarness([
            TimeoutError("LLM request timed out"),
            (200, {}, completion("recovered")),
        ])

        result = harness.call()

        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(harness.opener.requests), 2)
        self.assertEqual(len(harness.clock.sleeps), 1)

    def test_a_network_failure_uses_the_same_horizon(self):
        harness = HorizonHarness(
            [urllib.error.URLError(OSError("network is unreachable"))],
            max_wait_seconds=1,
        )

        with self.assertRaises(LLMTransportExhaustedError) as raised:
            harness.call()

        self.assertIn("before receiving an HTTP response", str(raised.exception))
        self.assertEqual(len(harness.opener.requests), 1)

    def test_the_waits_are_jittered_but_stay_bounded(self):
        random_values = iter([0.0, 0.9])
        def deterministic_uniform(low, high):
            return low + next(random_values) * (high - low)

        with mock.patch.object(chat.random, "uniform", deterministic_uniform):
            first = chat._jittered_delay(2.0)
            second = chat._jittered_delay(4.0)
        self.assertGreaterEqual(first, 1.6)
        self.assertLessEqual(first, 2.4)
        self.assertGreaterEqual(second, 3.2)
        self.assertLessEqual(second, 4.8)
        self.assertLess(first, second)


class FileFallbackTests(unittest.TestCase):
    EVIDENCE = TextFileAttachment(
        filename="repair-evidence.md",
        text="EVIDENCE_SENTINEL",
    )

    def call(self, harness, **kwargs):
        with (
            mock.patch("time.monotonic", harness.clock.monotonic),
            mock.patch("time.sleep", harness.clock.sleep),
        ):
            return harness.client().complete_with_file_fallback(
                "INLINE_SENTINEL",
                fallback_prompt="FALLBACK_CONTROL",
                attachments=(self.EVIDENCE,),
                **kwargs,
            )

    def test_the_inline_payload_is_tried_first_and_the_file_after_the_delay(self):
        harness = HorizonHarness(
            [
                (502, {}, {"error": "bad gateway"}),
                (502, {}, {"error": "bad gateway"}),
                (200, {}, completion("repaired")),
            ],
            max_wait_seconds=120,
        )

        result = self.call(harness, fallback_after_seconds=5)

        self.assertEqual(result.text, "repaired")
        contents = [
            request["payload"]["messages"][0]["content"]
            for request in harness.opener.requests
        ]
        self.assertEqual(contents[:2], ["INLINE_SENTINEL", "INLINE_SENTINEL"])
        attachment = contents[2]
        self.assertIsInstance(attachment, list)
        self.assertEqual(attachment[0], {"type": "text", "text": "FALLBACK_CONTROL"})
        self.assertEqual(attachment[1]["type"], "input_file")
        self.assertEqual(attachment[1]["file"]["filename"], "repair-evidence.md")
        prefix, encoded = attachment[1]["file"]["file_data"].split(",", 1)
        self.assertEqual(prefix, "data:text/markdown;base64")
        self.assertEqual(base64.b64decode(encoded).decode("utf-8"), "EVIDENCE_SENTINEL")
        # The evidence travels only as a file, never as inline control text.
        self.assertNotIn("EVIDENCE_SENTINEL", attachment[0]["text"])

    def test_a_retry_inside_the_delay_never_attaches_a_file(self):
        harness = HorizonHarness([
            (502, {}, {"error": "bad gateway"}),
            (200, {}, completion()),
        ])

        self.call(harness, fallback_after_seconds=3600)

        self.assertEqual(len(harness.opener.requests), 2)
        for request in harness.opener.requests:
            self.assertEqual(_payload_content(request), "INLINE_SENTINEL")
        self.assertNotIn("input_file", json.dumps(harness.opener.requests))

    def test_a_non_retryable_status_never_attaches_a_file(self):
        harness = HorizonHarness([(401, {}, {"error": "unauthorized"})])

        with self.assertRaises(LLMHTTPError):
            self.call(harness)

        self.assertEqual(len(harness.opener.requests), 1)
        self.assertEqual(_payload_content(harness.opener.requests[0]), "INLINE_SENTINEL")
        self.assertNotIn("input_file", json.dumps(harness.opener.requests))

    def test_a_failed_completion_still_exhausts_the_same_horizon(self):
        harness = HorizonHarness(
            [(502, {}, {"error": "bad gateway"})], max_wait_seconds=5,
        )

        with self.assertRaises(LLMTransportExhaustedError):
            self.call(harness)

        self.assertGreaterEqual(len(harness.opener.requests), 1)
        self.assertNotIn("input_file", json.dumps(harness.opener.requests))

    def test_standard_complete_is_unchanged_by_the_file_fallback_feature(self):
        harness = HorizonHarness(
            [(502, {}, {"error": "bad gateway"})], max_wait_seconds=5,
        )

        with (
            mock.patch("time.monotonic", harness.clock.monotonic),
            mock.patch("time.sleep", harness.clock.sleep),
            self.assertRaises(LLMHTTPError),
        ):
            harness.client().complete("prompt")

        for request in harness.opener.requests:
            self.assertEqual(
                request["payload"],
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "stream": False,
                },
            )

    def test_the_fallback_delay_must_be_a_non_negative_number(self):
        for value in (-1, "later", True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    HorizonHarness([(200, {}, completion())]).client(
                    ).complete_with_file_fallback(
                        "inline",
                        fallback_prompt="control",
                        attachments=(self.EVIDENCE,),
                        fallback_after_seconds=value,
                    )

    def test_attachment_validation_rejects_unsafe_values(self):
        for filename in ("", "a/b.md", "a\\b.md", "a\x00b", "x" * 129):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    TextFileAttachment(filename=filename, text="x")
        with self.assertRaises(ValueError):
            TextFileAttachment(filename="a.md", text="x", media_type="not a type")
        with self.assertRaises(TypeError):
            TextFileAttachment(filename="a.md", text=None)


if __name__ == "__main__":
    unittest.main()
