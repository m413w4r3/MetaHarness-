import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.chat import LLMHTTPError, TextLLMResult  # noqa: E402
from metaharness.planning import PlanParseError  # noqa: E402
from scripts.manual import planner_real_smoke as smoke  # noqa: E402


READY = """STATUS: READY
TITLE: Add bounded HTTP retry
OBJECTIVE: Retry selected transient HTTP errors.
CONSTRAINTS: Preserve the public request signature.
FILES: src/client.py; tests/test_client.py
IMPLEMENTATION: Add bounded exponential retry behavior.
ACCEPTANCE: Retryable and non-retryable statuses behave as specified.
TESTS: Cover success, exhaustion, and non-retryable errors.
RISKS: Backoff must not exceed the attempt limit.
BLOCKERS: NONE
"""

BLOCKED = """STATUS: BLOCKED
TITLE: Waiting for transport details
OBJECTIVE: Define the retry behavior.
IMPLEMENTATION: Inspect the HTTP client contract.
ACCEPTANCE: The missing transport detail is supplied.
TESTS: Re-run the planner after clarification.
BLOCKERS: The transport implementation is not available in the supplied context.
"""


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return TextLLMResult(text=response, model="fake-served-model", usage={"total_tokens": 1}, raw_response={})


class ManualPlannerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "META_SMOKE_GPT_BASE_URL": "http://127.0.0.1:31001",
            "META_SMOKE_GEMINI_BASE_URL": "http://127.0.0.1:31002",
            "META_SMOKE_GEMINI_MODEL": "gemini-real-model",
        }

    def test_provider_contracts_use_required_endpoints_labels_and_env_model(self):
        with patch.dict(os.environ, self.environment, clear=True):
            chatgpt = smoke.provider_run("chatgpt")
            gemini = smoke.provider_run("gemini")

        self.assertEqual(chatgpt.config.endpoint_path, "/v1/chat/completions")
        self.assertEqual(chatgpt.config.model, "chatgpt-web")
        self.assertEqual(chatgpt.config.extra_body, {"new_chat": True})
        self.assertIsNone(chatgpt.config.api_key_env)
        self.assertEqual(gemini.config.endpoint_path, "/v1/stateless/chat/completions")
        self.assertNotEqual(gemini.config.endpoint_path, "/v1/temporary/chat/completions")
        self.assertEqual(gemini.config.model, "gemini-real-model")
        self.assertEqual(gemini.config.extra_body, {})

    def test_api_key_is_passed_as_variable_name_only(self):
        environment = {**self.environment, "META_SMOKE_GPT_API_KEY": "secret-value"}
        with patch.dict(os.environ, environment, clear=True):
            config = smoke.provider_run("chatgpt").config
        self.assertEqual(config.api_key_env, "META_SMOKE_GPT_API_KEY")
        self.assertNotIn("secret-value", repr(config))

    def test_repeat_validation_rejects_values_outside_one_to_ten(self):
        with self.assertRaises(SystemExit) as zero:
            smoke.parse_args(["chatgpt", "--repeat", "0"])
        self.assertEqual(zero.exception.code, 2)
        with self.assertRaises(SystemExit) as eleven:
            smoke.parse_args(["chatgpt", "--repeat", "11"])
        self.assertEqual(eleven.exception.code, 2)
        self.assertEqual(smoke.parse_args(["chatgpt"]).repeat, 3)

    def test_run_records_parse_success_parse_failure_and_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fake = FakeClient([READY, BLOCKED, "not a plan", LLMHTTPError("secret-value")])
            with patch.dict(os.environ, self.environment, clear=True):
                counts = smoke.run_provider(
                    smoke.provider_run("chatgpt"),
                    repeat=4,
                    run_dir=directory,
                    client_factory=lambda _config: fake,
                )

            self.assertEqual(
                counts,
                {
                    "attempts": 4,
                    "parsed": 2,
                    "parse_failed": 1,
                    "transport_failed": 1,
                    "ready": 1,
                    "blocked": 1,
                },
            )
            attempt_one = directory / "chatgpt" / "attempt-01"
            self.assertEqual((attempt_one / "response.raw.md").read_text(), READY)
            self.assertEqual(json.loads((attempt_one / "parsed.json").read_text())["decision"], "READY")
            attempt_two = directory / "chatgpt" / "attempt-02"
            self.assertEqual(json.loads((attempt_two / "parsed.json").read_text())["decision"], "BLOCKED")
            attempt_three = directory / "chatgpt" / "attempt-03"
            self.assertTrue((attempt_three / "parse-error.txt").exists())
            self.assertFalse((attempt_three / "parsed.json").exists())
            transport_result = (directory / "chatgpt" / "attempt-04" / "result.json").read_text()
            self.assertNotIn("secret-value", transport_result)
            self.assertNotIn("secret-value", (attempt_three / "parse-error.txt").read_text())

    def test_matrix_executes_both_configured_providers_and_writes_summary(self):
        calls = []

        def factory(config):
            calls.append(config)
            return FakeClient([READY])

        with tempfile.TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, self.environment, clear=True):
                run_dir, summary = smoke.run_smoke(
                    ("chatgpt", "gemini"),
                    repeat=1,
                    output_root=Path(directory_name),
                    timestamp="20260911T120000000000Z",
                    client_factory=factory,
                )

            self.assertEqual([config.model for config in calls], ["chatgpt-web", "gemini-real-model"])
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["providers"]["chatgpt"]["parsed"], 1)
            self.assertEqual(summary["providers"]["gemini"]["ready"], 1)
            self.assertEqual(json.loads((run_dir / "summary.json").read_text()), summary)

    def test_matrix_validates_all_endpoints_before_contacting_any_provider(self):
        calls = []

        def factory(config):
            calls.append(config)
            return FakeClient([READY])

        environment = {**self.environment, "META_SMOKE_GEMINI_BASE_URL": "not-an-url"}
        with tempfile.TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(smoke.SmokeConfigurationError):
                    smoke.run_smoke(
                        ("chatgpt", "gemini"),
                        repeat=1,
                        output_root=Path(directory_name),
                        client_factory=factory,
                    )
            self.assertEqual(calls, [])
            self.assertEqual(list(Path(directory_name).iterdir()), [])

    def test_invalid_client_result_is_transport_failure_without_error_text(self):
        class InvalidClient:
            def complete(self, _prompt):
                return object()

        with tempfile.TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, self.environment, clear=True):
                counts = smoke.run_provider(
                    smoke.provider_run("chatgpt"),
                    repeat=1,
                    run_dir=Path(directory_name),
                    client_factory=lambda _config: InvalidClient(),
                )
            self.assertEqual(counts["transport_failed"], 1)
            result = (Path(directory_name) / "chatgpt" / "attempt-01" / "result.json").read_text()
            self.assertIn("InvalidClientResult", result)


if __name__ == "__main__":
    unittest.main()
