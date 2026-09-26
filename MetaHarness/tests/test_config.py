import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.config import ConfigError, load_config
from metaharness.models import CheckConfig, RoutingConfig
from metaharness.planning.protocol import render_safe_check_catalogue
from metaharness.recovery_policy import ExecutionFallbacks, RecoveryBudgets
from metaharness.run_options import (
    RUN_SCHEMA_UNSUPPORTED,
    SCHEMA_VERSION,
    RunOptions,
    RunOptionsError,
    canonical_run_options_bytes,
)


VALID_CONFIG = """
repo = "../AutoWork"
base_ref = "main"
runs_root = "../MetaHarness-runs"
worktrees_root = "../MetaHarness-worktrees"
require_clean_base = true
max_diff_bytes = 400000

[codex_runtime]
home = "codex-home"

[ui]
default_planner_profile = "planner-chat"
default_implementer_profile = "implementer-codex"
default_reviewer_profile = "reviewer-chat"

[model_profiles.planner-chat]
display_name = "Planner"
roles = ["planner"]
driver = "openai-chat"
provider = "bridge"
model = "${META_PLANNER_MODEL}"
selection_mode = "request"
base_url = "${META_PLANNER_BASE_URL}"
endpoint_path = "${META_PLANNER_ENDPOINT}"
api_key_env = "META_PLANNER_API_KEY"
timeout_seconds = 300
retries = 2

[model_profiles.planner-chat.extra_body]
new_chat = true
nested = { label = "${META_NESTED}" }

[model_profiles.implementer-codex]
display_name = "Implementer"
roles = ["implementer"]
driver = "codex"
provider = "bridge"
model = "gpt-5.6-luna"
effort = "high"
sandbox = "workspace-write"
selection_mode = "cli"
timeout_seconds = 5400

[model_profiles.reviewer-chat]
display_name = "Reviewer"
roles = ["reviewer"]
driver = "openai-chat"
provider = "bridge"
model = "review-model"
selection_mode = "request"
base_url = "https://review.example"
endpoint_path = "/v1/chat"
timeout_seconds = 420
retries = 2

[context]
always_files = ["AGENTS.md", "README.md"]
locator_argv = ["ctx", "query", "{query}", "-k", "8"]
locator_timeout_seconds = 120
max_hits = 8
max_bytes = 160000
require_locator_head_at_base = true

[[check_catalog]]
id = "test"
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
        self.assertEqual(config.model_profiles[config.ui.default_planner_profile].base_url, "https://planner.example")
        self.assertEqual(config.model_profiles[config.ui.default_planner_profile].model, "planner-model")
        self.assertEqual(config.model_profiles[config.ui.default_planner_profile].extra_body["nested"]["label"], "nested-value")
        self.assertEqual(config.context.locator_argv[0], "ctx")
        self.assertEqual(config.check_catalog[0].argv, ("make", "test"))
        self.assertEqual(config.model_profiles[config.ui.default_planner_profile].api_key_env, "META_PLANNER_API_KEY")
        self.assertEqual(config.codex_runtime.env_allowlist, (
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

    def test_autowork_aw010_trusted_check_catalogue(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "autowork.toml"
        with example.open("rb") as stream:
            raw = tomllib.load(stream)

        checks = {check["id"]: check for check in raw["check_catalog"]}
        self.assertEqual(
            raw["routing"],
            {
                "mechanical_profile": "codex-luna-high",
                "reasoning_profile": "codex-luna-xhigh",
                "agentic_profile": "codex-deepseek-flash-max",
            },
        )
        self.assertEqual(
            raw["recovery"]["execution_fallbacks"],
            {
                "mechanical": ["codex-luna-xhigh"],
                "reasoning": ["codex-sol-high"],
                "agentic": ["codex-sol-high"],
                "semantic_reviser": ["codex-astra-medium"],
            },
        )
        self.assertEqual(raw["default_check_ids"], ["lint", "typecheck", "test"])
        self.assertEqual(
            tuple(check["id"] for check in raw["check_catalog"]),
            (
                "lint",
                "typecheck",
                "test",
                "test-integration",
                "frontend-e2e",
                "alembic-heads",
            ),
        )
        self.assertIn("frontend-e2e", checks)
        self.assertEqual(checks["frontend-e2e"]["cwd"], "frontend")
        self.assertEqual(tuple(checks["frontend-e2e"]["argv"]), ("pnpm", "test:e2e"))
        self.assertNotIn("frontend-e2e", raw["default_check_ids"])
        self.assertIn("alembic-heads", checks)
        self.assertEqual(checks["alembic-heads"]["cwd"], "backend")
        self.assertEqual(
            tuple(checks["alembic-heads"]["argv"]),
            (
                "sh",
                "-c",
                "set -eu; out=\"$(uv run alembic heads)\"; printf '%s\\n' \"$out\"; test \"$out\" = '0001_baseline (head)'",
            ),
        )

        trusted = tuple(
            CheckConfig(
                name=check["id"],
                argv=tuple(check["argv"]),
                cwd=check.get("cwd", "."),
                timeout_seconds=check.get("timeout_seconds", 3600),
                required=check.get("required", True),
                preflight_argv=tuple(check.get("preflight_argv", ())),
                description=check.get("description", ""),
            )
            for check in raw["check_catalog"]
        )
        rendered_catalogue = render_safe_check_catalogue(trusted)
        for check_id in checks:
            with self.subTest(check_id=check_id):
                self.assertIn(f"ID: {check_id}", rendered_catalogue)

    def test_recovery_budgets_are_configurable_and_bounded(self) -> None:
        contents = VALID_CONFIG + """

[recovery]
max_transient_attempts = 3
max_executor_fallbacks = 1
max_check_infra_retries = 2
max_review_transport_retries = 0
max_workspace_setup_retries = 2
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.recovery, RecoveryBudgets(
            max_transient_attempts=3,
            max_executor_fallbacks=1,
            max_check_infra_retries=2,
            max_review_transport_retries=0,
            max_workspace_setup_retries=2,
        ))

    def test_execution_fallback_profiles_are_provider_neutral_config(self) -> None:
        contents = VALID_CONFIG + """

[recovery]
[recovery.execution_fallbacks]
mechanical = ["mechanical-rescue"]
reasoning = ["reasoning-rescue"]
agentic = ["agentic-rescue"]
semantic_reviser = ["reviser-rescue"]
check_repair = ["repair-rescue"]
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.recovery.execution_fallbacks, ExecutionFallbacks(
            mechanical=("mechanical-rescue",),
            reasoning=("reasoning-rescue",),
            agentic=("agentic-rescue",),
            semantic_reviser=("reviser-rescue",),
            check_repair=("repair-rescue",),
        ))

    def test_execution_fallback_profiles_must_be_arrays(self) -> None:
        contents = VALID_CONFIG + """

[recovery.execution_fallbacks]
mechanical = "rescue"
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config_path = self.write_config(Path(directory_name), contents)
            with self.assertRaisesRegex(ConfigError, "execution_fallbacks.mechanical"):
                load_config(config_path)

    def test_recovery_budget_rejects_unknown_keys(self) -> None:
        contents = VALID_CONFIG + "\n[recovery]\nretries = 9\n"
        with tempfile.TemporaryDirectory() as directory_name:
            config_path = self.write_config(Path(directory_name), contents)
            with self.assertRaisesRegex(ConfigError, "recovery.retries"):
                load_config(config_path)

    def test_alembic_heads_check_accepts_only_the_baseline_head(self) -> None:
        example = Path(__file__).resolve().parents[1] / "examples" / "autowork.toml"
        with example.open("rb") as stream:
            raw = tomllib.load(stream)
        check = next(item for item in raw["check_catalog"] if item["id"] == "alembic-heads")

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            fake_uv = directory / "uv"
            fake_uv.write_text(
                "#!/bin/sh\nprintf '%s' \"$FAKE_HEADS\"\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}{os.pathsep}{environment.get('PATH', '')}"

            def run_heads(output: str) -> subprocess.CompletedProcess[str]:
                test_environment = environment | {"FAKE_HEADS": output}
                return subprocess.run(
                    check["argv"],
                    cwd=directory,
                    env=test_environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            exact = run_heads("0001_baseline (head)")
            self.assertEqual(exact.returncode, 0)
            self.assertEqual(exact.stdout, "0001_baseline (head)\n")

            multiple = run_heads("0001_baseline (head)\n0002_other (head)")
            self.assertNotEqual(multiple.returncode, 0)
            self.assertEqual(
                multiple.stdout,
                "0001_baseline (head)\n0002_other (head)\n",
            )

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

    def test_run_branch_publication_uses_the_candidate_staging_remote(self) -> None:
        # Run-branch publication is the reviewed candidate already pushed to
        # repository.remote; another publish remote would claim a push that
        # never happened.
        cases = (
            ('[repository]\nremote = "origin"\n\n[publish]\nenabled = true\n'
             'remote = "upstream"\nmode = "run-branch"\n', True),
            ('[repository]\nremote = "origin"\n\n[publish]\nenabled = true\n'
             'remote = "origin"\nmode = "run-branch"\n', False),
            ('[repository]\nremote = "origin"\n\n[publish]\nenabled = true\n'
             'remote = "upstream"\nmode = "fast-forward-base"\n', False),
        )
        for suffix, invalid in cases:
            with self.subTest(suffix=suffix):
                with tempfile.TemporaryDirectory() as directory_name:
                    path = self.write_config(Path(directory_name), VALID_CONFIG + suffix)
                    if invalid:
                        with self.assertRaisesRegex(ConfigError, "publish.remote must equal repository.remote"):
                            load_config(path)
                    else:
                        load_config(path)

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
        self.assertEqual(config.planning.decomposition, "aggressive")
        self.assertEqual(config.planning.execution_mode_policy, "auto")
        self.assertEqual(config.planning.staged_step_max_mutable_paths, 5)
        self.assertEqual(config.planning.single_step_max_mutable_paths, 2)
        self.assertEqual(config.planning.max_steps_per_plan, 8)
        self.assertEqual(config.planning.max_read_paths_per_step, 8)
        self.assertEqual(config.planning.max_step_contract_chars, 5000)

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

        self.assertEqual(PlanningConfig().staged_step_max_mutable_paths, 5)
        self.assertEqual(PlanningConfig().max_steps_per_plan, 8)
        self.assertEqual(PlanningConfig().max_read_paths_per_step, 8)
        self.assertEqual(PlanningConfig().max_step_contract_chars, 5000)
        for name in ("single_step_max_mutable_paths", "staged_step_max_mutable_paths"):
            for value in (0, -1, True, "6"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        PlanningConfig(**{name: value})

        for name in ("max_steps_per_plan", "max_read_paths_per_step", "max_step_contract_chars"):
            for value in (0, -1, True, "8"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        PlanningConfig(**{name: value})
        with self.assertRaisesRegex(ValueError, "max_steps_per_plan"):
            PlanningConfig(max_steps_per_plan=100)

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

    def test_agent_section_is_rejected(self) -> None:
        contents = VALID_CONFIG + '\n[agent]\nmodel = "not-configurable"\n'
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, r"\[agent\]"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_missing_environment_variable_is_explicit_error(self) -> None:
        os.environ.pop("META_PLANNER_MODEL")
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "META_PLANNER_MODEL"):
                load_config(self.write_config(Path(directory_name)))

    def test_declared_missing_environment_file_fails_closed(self) -> None:
        contents = VALID_CONFIG + '\n[environment]\nfiles = ["missing.env"]\n'
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, r"cannot read environment file .*missing\.env"):
                load_config(self.write_config(Path(directory_name), contents))

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
        self.assertEqual(config.model_profiles[config.ui.default_planner_profile].model, "model-${META_NESTED}")

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
            ('sandbox = "workspace-write"', 'unknown model_profiles.implementer-codex.sandbox'),
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
            (VALID_CONFIG.replace("\n[[check_catalog]]\nid = \"test\"\nargv = [\"make\", \"test\"]\n", "\n"), "required check"),
            (VALID_CONFIG.replace('id = "test"', 'id = "optional"').replace("\n[[check_catalog]]", "\n[[check_catalog]]\nrequired = false"), "required check"),
        )
        for contents, message in cases:
            with self.subTest(contents=contents):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(self.write_config(Path(directory_name), contents))

        for checks in ("", "\n[[check_catalog]]\nid = \"optional\"\nargv = [\"make\", \"test\"]\nrequired = false\n"):
            contents = (
                VALID_CONFIG.replace(
                    '\n[[check_catalog]]\nid = "test"\nargv = ["make", "test"]\n', checks
                )
                .replace("max_diff_bytes = 400000\n", "max_diff_bytes = 400000\nallow_no_required_checks = true\n")
            )
            with tempfile.TemporaryDirectory() as directory_name:
                config = load_config(self.write_config(Path(directory_name), contents))
            self.assertTrue(config.allow_no_required_checks)

    def test_model_profiles_and_ui_defaults_are_required(self) -> None:
        missing_profiles = VALID_CONFIG.split('[model_profiles.planner-chat]', 1)[0]
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "model_profiles"):
                load_config(self.write_config(Path(directory_name), missing_profiles))
        for key in (
            "default_planner_profile",
            "default_implementer_profile",
            "default_reviewer_profile",
        ):
            contents = VALID_CONFIG.replace(f'{key} = "', f'# {key} = "', 1)
            with tempfile.TemporaryDirectory() as directory_name:
                with self.assertRaisesRegex(ConfigError, f"ui.{key} is required"):
                    load_config(self.write_config(Path(directory_name), contents))

    def test_old_sections_and_check_table_are_rejected(self) -> None:
        for section in ("planner", "reviewer", "agent"):
            contents = VALID_CONFIG + f"\n[{section}]\nmodel = \"old\"\n"
            with tempfile.TemporaryDirectory() as directory_name:
                with self.assertRaisesRegex(ConfigError, rf"\[{section}\]"):
                    load_config(self.write_config(Path(directory_name), contents))
        contents = VALID_CONFIG + '\n[[checks]]\nname = "old"\nargv = ["make", "test"]\n'
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "check_catalog"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_profile_roles_are_provider_neutral(self) -> None:
        contents = VALID_CONFIG.replace('provider = "bridge"', 'provider = "arbitrary-provider"')
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.model_profiles["planner-chat"].provider, "arbitrary-provider")

    def test_modern_role_matrix_accepts_external_repair_and_claude_reviser(self) -> None:
        contents = VALID_CONFIG.replace(
            'default_reviewer_profile = "reviewer-chat"',
            'default_reviewer_profile = "reviewer-chat"\n'
            'default_reviser_profile = "reviser-claude"\n'
            'default_repair_profile = "repair-external"',
        ).replace(
            'driver = "codex"\nprovider = "bridge"\nmodel = "gpt-5.6-luna"\neffort = "high"\nsandbox = "workspace-write"\nselection_mode = "cli"',
            'driver = "external"\nprovider = "deepseek"\nmodel = "deepseek-worker"\nselection_mode = "cli"\nargv = ["trusted-worker"]',
        ) + '''
[revision]
enabled = true
max_check_repair_attempts = 1
max_review_repair_cycles = 1

[claude_runtime]
home = "claude-home"

[model_profiles.repair-external]
display_name = "External repair"
roles = ["repair"]
driver = "external"
provider = "deepseek"
model = "deepseek-repair"
selection_mode = "cli"
argv = ["trusted-repair"]

[model_profiles.reviser-claude]
display_name = "Claude reviser"
roles = ["reviser"]
driver = "claude-code"
provider = "anthropic"
model = "opus"
effort = "medium"
permission_mode = "acceptEdits"
selection_mode = "cli"
'''
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.model_profiles["implementer-codex"].driver, "external")
        self.assertEqual(config.ui.default_repair_profile, "repair-external")
        self.assertEqual(config.ui.default_reviser_profile, "reviser-claude")


class RunOptionsStrictSchemaTests(unittest.TestCase):
    """`run_options.json` has exactly one shape: schema 3 is read, nothing else."""

    ENVIRONMENT = {
        "META_PLANNER_BASE_URL": "https://planner.example",
        "META_PLANNER_ENDPOINT": "/v1/chat",
        "META_PLANNER_MODEL": "planner-model",
        "META_NESTED": "nested-value",
    }

    def config(self):
        with mock.patch.dict(os.environ, self.ENVIRONMENT):
            with tempfile.TemporaryDirectory() as directory_name:
                path = Path(directory_name) / "config.toml"
                path.write_text(VALID_CONFIG, encoding="utf-8")
                return load_config(path)

    def snapshot(self) -> dict:
        return RunOptions.from_config(self.config()).to_dict()

    def test_current_schema_three_round_trips_identically(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
        options = RunOptions.from_mapping(snapshot)
        self.assertEqual(options.to_dict(), snapshot)
        encoded = canonical_run_options_bytes(options)
        self.assertEqual(
            canonical_run_options_bytes(RunOptions.from_mapping(json.loads(encoded))),
            encoded,
        )

    def test_previous_schema_is_rejected_without_conversion(self) -> None:
        old_snapshot = self.snapshot()
        old_snapshot["schema_version"] = 2
        with self.assertRaises(RunOptionsError) as caught:
            RunOptions.from_mapping(old_snapshot)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))
        with self.assertRaises(RunOptionsError) as caught:
            replace(RunOptions.from_mapping(self.snapshot()), schema_version=2)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))

    def test_missing_recovery_fields_are_rejected(self) -> None:
        for field in (
            "max_transient_attempts",
            "max_contract_repair_planner_restarts",
            "execution_fallbacks",
        ):
            snapshot = self.snapshot()
            del snapshot["recovery"][field]
            with self.assertRaisesRegex(RunOptionsError, f"missing {field}"):
                RunOptions.from_mapping(snapshot)
        snapshot = self.snapshot()
        del snapshot["recovery"]
        with self.assertRaisesRegex(RunOptionsError, "missing recovery"):
            RunOptions.from_mapping(snapshot)

    def test_historical_default_implementer_profile_is_rejected(self) -> None:
        self.assertNotIn("default_implementer_profile", RunOptions.__dataclass_fields__)
        snapshot = self.snapshot()
        snapshot["profiles"]["default_implementer_profile"] = "implementer-codex"
        with self.assertRaisesRegex(RunOptionsError, "unknown key default_implementer_profile"):
            RunOptions.from_mapping(snapshot)
        with self.assertRaisesRegex(RunOptionsError, "unknown run option: default_implementer_profile"):
            RunOptions.from_config(self.config(), default_implementer_profile="implementer-codex")

    def test_unknown_routing_profiles_are_rejected(self) -> None:
        config = self.config()
        unknown_routing = RoutingConfig(
            mechanical_profile="ghost",
            reasoning_profile="ghost",
            agentic_profile="ghost",
        )
        with self.assertRaisesRegex(RunOptionsError, "mechanical_profile is invalid or incompatible"):
            RunOptions.from_config(replace(config, routing=unknown_routing))

    def test_unknown_keys_and_missing_sections_are_rejected(self) -> None:
        for mutate, message in (
            (lambda snapshot: snapshot.update(topology="unused"), "unknown key topology"),
            (lambda snapshot: snapshot["planning"].update(repair_rounds=1), "planning has unknown key repair_rounds"),
            (lambda snapshot: snapshot["profiles"].pop("final_reviewer_profile"), "profiles is missing final_reviewer_profile"),
            (lambda snapshot: snapshot.pop("profiles"), "missing profiles"),
            (lambda snapshot: snapshot["recovery"].update(max_extra_attempts=1), "recovery has unknown key max_extra_attempts"),
        ):
            snapshot = self.snapshot()
            mutate(snapshot)
            with self.assertRaisesRegex(RunOptionsError, message):
                RunOptions.from_mapping(snapshot)

    def test_pipeline_without_max_step_contract_repairs_is_rejected(self) -> None:
        snapshot = self.snapshot()
        del snapshot["pipeline"]["max_step_contract_repairs"]
        with self.assertRaisesRegex(RunOptionsError, "missing max_step_contract_repairs"):
            RunOptions.from_mapping(snapshot)


if __name__ == "__main__":
    unittest.main()
