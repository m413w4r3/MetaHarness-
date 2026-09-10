import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.config import ConfigError, load_config


VALID_CONFIG = """
repo = "../AutoWork"
base_ref = "main"
runs_root = "../MetaHarness-runs"
worktrees_root = "../MetaHarness-worktrees"
require_clean_base = true
max_diff_bytes = 400000

[agent]
provider = "codex"
model = "gpt-5.6-luna"
effort = "high"
sandbox = "workspace-write"
timeout_seconds = 5400

[planner]
base_url = "${META_PLANNER_BASE_URL}"
endpoint_path = "${META_PLANNER_ENDPOINT}"
model = "${META_PLANNER_MODEL}"
api_key_env = "META_PLANNER_API_KEY"
timeout_seconds = 300
retries = 2

[planner.extra_body]
new_chat = true
nested = { label = "${META_NESTED}" }

[reviewer]
base_url = "https://review.example"
endpoint_path = "/v1/chat"
model = "review-model"
timeout_seconds = 420
retries = 2

[context]
always_files = ["AGENTS.md", "README.md"]
locator_argv = ["ctx", "query", "{query}", "-k", "8"]
locator_timeout_seconds = 120
max_hits = 8
max_bytes = 160000
require_locator_head_at_base = true

[[checks]]
name = "test"
argv = ["make", "test"]
"""


class ConfigTests(unittest.TestCase):
    def write_config(self, directory: Path, contents: str = VALID_CONFIG) -> Path:
        path = directory / "config.toml"
        path.write_text(contents, encoding="utf-8")
        return path

    def setUp(self) -> None:
        self.environment = {
            "META_PLANNER_BASE_URL": "https://planner.example",
            "META_PLANNER_ENDPOINT": "/v1/chat",
            "META_PLANNER_MODEL": "planner-model",
            "META_NESTED": "nested-value",
        }
        self.old_environment = {
            key: os.environ.get(key) for key in self.environment
        }
        os.environ.update(self.environment)

    def tearDown(self) -> None:
        for key, old_value in self.old_environment.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def test_valid_config_resolves_paths_and_expands_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = load_config(self.write_config(directory))

        self.assertEqual(config.repo, (directory / "../AutoWork").resolve())
        self.assertEqual(config.runs_root, (directory / "../MetaHarness-runs").resolve())
        self.assertEqual(config.planner.base_url, "https://planner.example")
        self.assertEqual(config.planner.model, "planner-model")
        self.assertEqual(config.planner.extra_body["nested"]["label"], "nested-value")
        self.assertEqual(config.context.locator_argv[0], "ctx")
        self.assertEqual(config.checks[0].argv, ("make", "test"))
        self.assertEqual(config.planner.api_key_env, "META_PLANNER_API_KEY")

    def test_missing_environment_variable_is_explicit_error(self) -> None:
        os.environ.pop("META_PLANNER_MODEL")
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "META_PLANNER_MODEL"):
                load_config(self.write_config(Path(directory_name)))

    def test_check_argv_string_is_rejected(self) -> None:
        contents = VALID_CONFIG.replace('argv = ["make", "test"]', 'argv = "make test"')
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "argv must be an array"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_api_key_env_must_be_a_name_without_echoing_the_value(self) -> None:
        secret = "sk-test-secret-value"
        contents = VALID_CONFIG.replace(
            'api_key_env = "META_PLANNER_API_KEY"',
            f'api_key_env = "{secret}"',
        )
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(
                ConfigError, "environment variable name"
            ) as raised:
                load_config(self.write_config(Path(directory_name), contents))
        self.assertNotIn(secret, str(raised.exception))

    def test_locator_argv_array_is_accepted_and_string_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name)))
            self.assertEqual(config.context.locator_argv[2], "{query}")

            contents = VALID_CONFIG.replace(
                'locator_argv = ["ctx", "query", "{query}", "-k", "8"]',
                'locator_argv = "ctx query"',
            )
            with self.assertRaisesRegex(ConfigError, "locator_argv must be an array"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_empty_limits_and_unknown_sandbox_are_rejected(self) -> None:
        for replacement, message in (
            ('max_hits = 8', 'max_hits must be greater'),
            ('max_bytes = 160000', 'max_bytes must be greater'),
            ('sandbox = "workspace-write"', 'unknown agent.sandbox'),
        ):
            contents = VALID_CONFIG.replace(
                replacement,
                replacement.replace("8", "0") if "max_hits" in replacement else
                replacement.replace("160000", "0") if "max_bytes" in replacement else
                'sandbox = "unknown"',
            )
            with tempfile.TemporaryDirectory() as directory_name:
                with self.assertRaisesRegex(ConfigError, message):
                    load_config(self.write_config(Path(directory_name), contents))


if __name__ == "__main__":
    unittest.main()
