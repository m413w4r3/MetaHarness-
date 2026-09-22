import os
import sys
import tempfile
import tomllib
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
        self.assertEqual(config.agent.env_allowlist, (
            "PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR",
            "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
        ))

    def test_autowork_example_uses_its_two_megabyte_diff_bound(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "autowork.toml"
        # This assertion is about the literal example declaration.  Loading
        # the full config would require the optional Bridges environment file,
        # which is not present in hermetic CI.
        with example.open("rb") as stream:
            raw = tomllib.load(stream)
        self.assertEqual(raw["max_diff_bytes"], 2_000_000)

    def test_pull_request_creation_requires_run_branch_publication(self) -> None:
        cases = (
            (
                '[github]\nenabled = true\npull_request_mode = "create"\n'
                '\n[publish]\nenabled = false\nmode = "run-branch"\n',
                True,
            ),
            (
                '[github]\nenabled = true\npull_request_mode = "create"\n'
                '\n[publish]\nenabled = true\nmode = "fast-forward-base"\n',
                True,
            ),
            (
                '[github]\nenabled = true\npull_request_mode = "create"\n'
                '\n[publish]\nenabled = true\nmode = "run-branch"\n',
                False,
            ),
        )
        for suffix, invalid in cases:
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as directory_name:
                    path = self.write_config(Path(directory_name), VALID_CONFIG + suffix)
                    if invalid:
                        with self.assertRaisesRegex(ConfigError, "publish.enabled|publish.mode"):
                            load_config(path)
                    else:
                        config = load_config(path)
                        self.assertTrue(config.publish.enabled)
                        self.assertEqual(config.publish.mode, "run-branch")

    def test_planning_protocol_defaults_to_v2_and_rejects_other_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name)))
        self.assertEqual(config.planning.protocol, "v2")

        contents = VALID_CONFIG + '\n[planning]\nprotocol = "v2"\n'
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.planning.protocol, "v2")

        for protocol in ("v1", "v3"):
            contents = VALID_CONFIG + f'\n[planning]\nprotocol = "{protocol}"\n'
            with tempfile.TemporaryDirectory() as directory_name:
                with self.assertRaisesRegex(ConfigError, "planning.protocol"):
                    load_config(self.write_config(Path(directory_name), contents))

    def test_staged_step_max_mutable_paths_defaults_and_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name)))
        self.assertEqual(config.planning.staged_step_max_mutable_paths, 6)
        self.assertEqual(config.planning.single_step_max_mutable_paths, 2)

        contents = (
            VALID_CONFIG
            + '\n[planning]\nprotocol = "v2"\ndecomposition = "aggressive"\n'
            + "staged_step_max_mutable_paths = 4\n"
        )
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.planning.staged_step_max_mutable_paths, 4)

        for value in ("0", "-1", "true", '"6"'):
            with self.subTest(value=value):
                contents = (
                    VALID_CONFIG
                    + f"\n[planning]\nstaged_step_max_mutable_paths = {value}\n"
                )
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(
                        ConfigError, "planning.staged_step_max_mutable_paths"
                    ):
                        load_config(self.write_config(Path(directory_name), contents))

    def test_planning_config_rejects_invalid_mutable_path_limits(self) -> None:
        from metaharness.models import PlanningConfig

        self.assertEqual(PlanningConfig().staged_step_max_mutable_paths, 6)
        for name in ("single_step_max_mutable_paths", "staged_step_max_mutable_paths"):
            for value in (0, -1, True, "6"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        PlanningConfig(**{name: value})

    def test_plan_approval_config_defaults_and_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name)))
        self.assertFalse(config.approval.require_plan_approval)
        self.assertEqual(config.approval.poll_interval_seconds, 0.5)

        for value, message in (("0", "greater than zero"), ("10.1", "at most"), ("true", "number"), ("\"0.5\"", "number")):
            contents = VALID_CONFIG + f"\n[approval]\npoll_interval_seconds = {value}\n"
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(self.write_config(Path(directory_name), contents))

        contents = VALID_CONFIG + "\n[approval]\nrequire_plan_approval = true\npoll_interval_seconds = 2\n"
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertTrue(config.approval.require_plan_approval)
        self.assertEqual(config.approval.poll_interval_seconds, 2.0)

    def test_agent_environment_allowlist_is_configurable_and_validated(self) -> None:
        contents = VALID_CONFIG.replace(
            'timeout_seconds = 5400\n\n[planner]',
            'timeout_seconds = 5400\nenv_allowlist = ["PATH", "CUSTOM_VALUE"]\n\n[planner]',
        )
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.agent.env_allowlist, ("PATH", "CUSTOM_VALUE"))

        invalid = contents.replace('"CUSTOM_VALUE"', '"not-valid-name"')
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "environment variable names"):
                load_config(self.write_config(Path(directory_name), invalid))

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

    def test_environment_expansion_is_not_recursive(self) -> None:
        os.environ["META_PLANNER_MODEL"] = "model-${META_NESTED}"
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name)))
        self.assertEqual(config.planner.model, "model-${META_NESTED}")

    def test_endpoint_urls_are_validated_at_load(self) -> None:
        for base_url, endpoint_path, message in (
            ("file:///etc", "/v1/chat", "http"),
            ("https://user:secret@planner.example", "/v1/chat", "credentials"),
            ("https://planner.example", "https://evil.example/v1", "path"),
            ("https://planner.example", "/v1/../admin", r"\.\."),
        ):
            os.environ["META_PLANNER_BASE_URL"] = base_url
            os.environ["META_PLANNER_ENDPOINT"] = endpoint_path
            with self.subTest(base_url=base_url, endpoint_path=endpoint_path):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(ConfigError, message) as raised:
                        load_config(self.write_config(Path(directory_name)))
                self.assertNotIn("secret", str(raised.exception))

    def test_extra_body_cannot_override_the_wire_contract(self) -> None:
        for key in ("messages", "response_format", "stream", "tools"):
            contents = VALID_CONFIG.replace("new_chat = true", f"{key} = true")
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(ConfigError, "protected"):
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

    def test_config_requires_one_required_check_by_default(self) -> None:
        cases = (
            (VALID_CONFIG.replace("\n[[checks]]\nname = \"test\"\nargv = [\"make\", \"test\"]\n", "\n"), "required check"),
            (VALID_CONFIG.replace('name = "test"', 'name = "optional"').replace("\n[[checks]]", "\n[[checks]]\nrequired = false"), "required check"),
        )
        for contents, message in cases:
            with self.subTest(contents=contents):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(self.write_config(Path(directory_name), contents))

        for checks in ("", "\n[[checks]]\nname = \"optional\"\nargv = [\"make\", \"test\"]\nrequired = false\n"):
            contents = (
                VALID_CONFIG.replace(
                    '\n[[checks]]\nname = "test"\nargv = ["make", "test"]\n', checks
                )
                .replace("max_diff_bytes = 400000\n", "max_diff_bytes = 400000\nallow_no_required_checks = true\n")
            )
            with tempfile.TemporaryDirectory() as directory_name:
                config = load_config(self.write_config(Path(directory_name), contents))
            self.assertTrue(config.allow_no_required_checks)


if __name__ == "__main__":
    unittest.main()
