import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.chat import LLMHTTPError, TextLLMResult  # noqa: E402
from metaharness.planning_v2 import PlanParseError  # noqa: E402
from scripts.manual import planner_real_smoke as smoke  # noqa: E402
from tests.pipeline_support import plan  # noqa: E402


READY = plan(("S01", "src/client.py", "Add bounded HTTP retry")).replace(
    "{implementer}", "smoke-implementer"
).replace("REVIEWER_PROFILE: reviewer", "REVIEWER_PROFILE: smoke-reviewer")

BLOCKED = """META PLAN v2

STATUS: BLOCKED
TITLE: Waiting for transport details

OBJECTIVE
Define the retry behavior.

BLOCKERS
The transport implementation is not available in the supplied context.

END META PLAN
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
            # The transport error message is now persisted; the configured
            # key value must be what redaction removes.
            environment = {**self.environment, "META_SMOKE_GPT_API_KEY": "secret-value"}
            with patch.dict(os.environ, environment, clear=True):
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
            # The smoke sends the production v2 planner request.
            self.assertIn("META PLAN v2", fake.prompts[0])
            self.assertEqual((attempt_one / "request.txt").read_text(), fake.prompts[0])
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

    def transport_result(self, error: BaseException, environment: dict[str, str]) -> dict:
        with tempfile.TemporaryDirectory() as directory_name:
            with patch.dict(os.environ, environment, clear=True):
                smoke.run_provider(
                    smoke.provider_run("chatgpt"),
                    repeat=1,
                    run_dir=Path(directory_name),
                    client_factory=lambda _config: FakeClient([error]),
                )
            path = Path(directory_name) / "chatgpt" / "attempt-01" / "result.json"
            return json.loads(path.read_text())

    def test_transport_failure_records_redacted_bounded_error(self):
        for error, expected in (
            (LLMHTTPError("LLM endpoint returned HTTP 401 after 1 attempt(s)"), "HTTP 401"),
            (LLMHTTPError("LLM request failed before receiving an HTTP response (ConnectionRefusedError)"), "ConnectionRefusedError"),
            (LLMHTTPError("LLM request timed out"), "timed out"),
        ):
            with self.subTest(expected=expected):
                result = self.transport_result(error, self.environment)
                self.assertEqual(result["status"], "TRANSPORT_FAILED")
                self.assertEqual(result["error_type"], "LLMHTTPError")
                self.assertIn(expected, result["error"])

    def test_transport_error_never_persists_credentials_or_traceback(self):
        secret = "sk-live-smoke-secret-0123456789"
        environment = {**self.environment, "META_SMOKE_GPT_API_KEY": secret}
        error = RuntimeError(
            f"bridge refused key {secret}\n"
            "Authorization: Bearer other-token-abcdef\n"
            "Cookie: session=cookie-value-xyz; csrf=token-csrf\n"
            "Set-Cookie: sid=set-cookie-value\n"
            "headers={'authorization': 'Bearer dict-token-123'}\n"
            + "x" * 5000
        )
        result = self.transport_result(error, environment)
        text = json.dumps(result)
        for leaked in (
            secret,
            "other-token-abcdef",
            "cookie-value-xyz",
            "token-csrf",
            "set-cookie-value",
            "dict-token-123",
            "Traceback",
        ):
            self.assertNotIn(leaked, text)
        self.assertIn("[REDACTED]", result["error"])
        self.assertLessEqual(len(result["error"]), 1000)
        self.assertNotIn("\n", result["error"])

    def test_safe_error_text_redacts_before_truncating(self):
        secret = "sk-truncated-secret-0123456789"
        with patch.dict(os.environ, {"META_SMOKE_GEMINI_API_KEY": secret}, clear=True):
            text = smoke._safe_error_text(ValueError("a" * 990 + secret))
        self.assertNotIn(secret[:10], text)
        self.assertLessEqual(len(text), 1000)

    def test_preflight_is_a_cli_choice(self):
        self.assertEqual(smoke.parse_args(["preflight"]).provider, "preflight")

    def test_preflight_ready_checks_catalogue_without_generating(self):
        secret = "sk-preflight-secret-0123456789"
        environment = {**self.environment, "META_SMOKE_GEMINI_API_KEY": secret}
        seen = []

        def fetch(config):
            seen.append(config)
            return ["other-model", "gemini-real-model"]

        with patch.dict(os.environ, environment, clear=True):
            with patch.object(smoke, "OpenAIChatTextClient", side_effect=AssertionError("no generation")):
                code, lines = smoke.run_preflight(fetch_models=fetch)
        self.assertEqual(code, 0, lines)
        self.assertEqual([config.model for config in seen], ["gemini-real-model"])
        report = "\n".join(lines)
        self.assertIn("META_SMOKE_GEMINI_API_KEY set", report)
        self.assertIn("gemini catalogue lists gemini-real-model", report)
        self.assertNotIn(secret, report)

    def test_preflight_reports_missing_model_and_probe_failure(self):
        with patch.dict(os.environ, self.environment, clear=True):
            code, lines = smoke.run_preflight(fetch_models=lambda _config: ["other-model"])
        self.assertEqual(code, 1)
        self.assertIn("does not list gemini-real-model", "\n".join(lines))

        secret = "sk-probe-secret-0123456789"
        environment = {**self.environment, "META_SMOKE_GEMINI_API_KEY": secret}

        def failing(_config):
            raise OSError(f"HTTP Error 401: Unauthorized Authorization: Bearer {secret}")

        with patch.dict(os.environ, environment, clear=True):
            code, lines = smoke.run_preflight(fetch_models=failing)
        self.assertEqual(code, 1)
        report = "\n".join(lines)
        self.assertIn("HTTP Error 401", report)
        self.assertNotIn(secret, report)

    def test_preflight_configuration_errors_do_not_contact_providers(self):
        calls = []

        def fetch(config):
            calls.append(config)
            return []

        environment = {**self.environment, "META_SMOKE_GEMINI_BASE_URL": "not-an-url"}
        del environment["META_SMOKE_GPT_BASE_URL"]
        with tempfile.TemporaryDirectory() as directory_name:
            missing_fixture = Path(directory_name) / "missing.md"
            with patch.dict(os.environ, environment, clear=True):
                code, lines = smoke.run_preflight(
                    fixtures=(missing_fixture, smoke.CONTEXT_PATH), fetch_models=fetch
                )
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        report = "\n".join(lines)
        self.assertIn("META_SMOKE_GPT_BASE_URL", report)
        self.assertIn("FAIL gemini", report)
        self.assertIn("FAIL fixture", report)

    def test_preflight_rejects_invalid_key_value_by_name_only(self):
        environment = {**self.environment, "META_SMOKE_GEMINI_API_KEY": "bad key with spaces"}
        with patch.dict(os.environ, environment, clear=True):
            code, lines = smoke.run_preflight(fetch_models=lambda _config: [])
        self.assertEqual(code, 2)
        report = "\n".join(lines)
        self.assertIn("META_SMOKE_GEMINI_API_KEY contains an invalid value", report)
        self.assertNotIn("bad key with spaces", report)


if __name__ == "__main__":
    unittest.main()
