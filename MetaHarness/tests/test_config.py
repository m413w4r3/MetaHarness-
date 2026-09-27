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
from metaharness.models import CheckConfig, RoutingConfig, TransportConfig
from metaharness.planning.protocol import render_safe_check_catalogue
from metaharness.recovery_policy import AutonomyBudget, ExecutionFallbacks
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
default_audit_profile = "implementer-codex"

[routing]
mechanical_profile = "implementer-codex"
reasoning_profile = "implementer-codex"
agentic_profile = "implementer-codex"

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

[model_profiles.planner-chat.extra_body]
new_chat = true
nested = { label = "${META_NESTED}" }

[model_profiles.implementer-codex]
display_name = "Implementer"
roles = ["implementer", "auditor"]
driver = "codex"
provider = "bridge"
model = "gpt-5.6-luna"
effort = "high"
sandbox = "workspace-write"
selection_mode = "cli"
timeout_seconds = 5400

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

    def test_the_removed_ui_defaults_are_refused_by_name(self) -> None:
        """No alias, no migration: a removed [ui] default is a configuration error."""

        for removed in (
            "default_implementer_profile", "default_reviewer_profile",
            "default_reviser_profile", "default_repair_profile",
        ):
            with self.subTest(removed=removed):
                contents = VALID_CONFIG.replace(
                    'default_planner_profile = "planner-chat"',
                    f'default_planner_profile = "planner-chat"\n{removed} = "implementer-codex"',
                )
                with tempfile.TemporaryDirectory() as directory_name:
                    path = self.write_config(Path(directory_name), contents)
                    with self.assertRaisesRegex(ConfigError, removed):
                        load_config(path)

    def test_routing_is_required_and_never_defaulted(self) -> None:
        without_routing = VALID_CONFIG.replace(
            "[routing]\n"
            'mechanical_profile = "implementer-codex"\n'
            'reasoning_profile = "implementer-codex"\n'
            'agentic_profile = "implementer-codex"\n',
            "",
        )
        self.assertNotIn("[routing]", without_routing)
        with tempfile.TemporaryDirectory() as directory_name:
            path = self.write_config(Path(directory_name), without_routing)
            with self.assertRaisesRegex(ConfigError, "routing is required"):
                load_config(path)

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

    def test_the_one_budget_section_is_configurable_and_bounded(self) -> None:
        contents = VALID_CONFIG + """

[budget]
step_attempts = 4
audit_repairs = 1
max_iterations = 5
max_wall_clock_hours = 3.5
max_cost = 0
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.budget, AutonomyBudget(
            step_attempts=4, audit_repairs=1, max_iterations=5,
            max_wall_clock_hours=3.5, max_cost=0,
        ))

    def test_the_budget_defaults_are_the_spec_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), VALID_CONFIG))
        self.assertEqual(config.budget, AutonomyBudget(
            step_attempts=3, audit_repairs=2, max_iterations=8,
            max_wall_clock_hours=12, max_cost=0,
        ))
        self.assertEqual(
            [field for field in AutonomyBudget.__dataclass_fields__],
            ["step_attempts", "audit_repairs", "max_iterations",
             "max_wall_clock_hours", "max_cost"],
        )

    def test_removed_numeric_retry_budgets_are_refused(self) -> None:
        for section in (
            "max_transient_attempts = 3",
            "max_executor_fallbacks = 1",
            "max_check_infra_retries = 2",
            "max_workspace_setup_retries = 2",
        ):
            contents = VALID_CONFIG + "\n[recovery]\n" + section + "\n"
            with self.subTest(section=section):
                with tempfile.TemporaryDirectory() as directory_name:
                    with self.assertRaises(ConfigError) as caught:
                        load_config(self.write_config(Path(directory_name), contents))
                self.assertIn("is not allowed", str(caught.exception))
                self.assertIn("[budget]", str(caught.exception))

    def test_a_preapproval_correction_budget_is_refused(self) -> None:
        contents = VALID_CONFIG + "\n[planning]\nmax_preapproval_corrections = 2\n"
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "planning.max_preapproval_corrections is not allowed"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_a_positive_cost_cap_is_refused_because_no_price_is_known(self) -> None:
        contents = VALID_CONFIG + "\n[budget]\nmax_cost = 0.5\n"
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "max_cost > 0 is not supported"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_an_unknown_budget_key_is_refused(self) -> None:
        contents = VALID_CONFIG + "\n[budget]\nmax_retries = 3\n"
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "budget.max_retries is not allowed"):
                load_config(self.write_config(Path(directory_name), contents))

    def test_execution_fallback_profiles_are_provider_neutral_config(self) -> None:
        contents = VALID_CONFIG + """

[recovery]
[recovery.execution_fallbacks]
mechanical = ["mechanical-rescue"]
reasoning = ["reasoning-rescue"]
agentic = ["agentic-rescue"]
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.execution_fallbacks, ExecutionFallbacks(
            mechanical=("mechanical-rescue",),
            reasoning=("reasoning-rescue",),
            agentic=("agentic-rescue",),
        ))

    def test_removed_execution_fallbacks_are_rejected(self) -> None:
        # The old check-repair and semantic-revision rungs are gone: a config
        # that still names them is refused instead of silently ignored.
        for removed in ("semantic_reviser", "check_repair", "final_reviewer"):
            with self.subTest(removed=removed):
                contents = VALID_CONFIG + f"""

[recovery.execution_fallbacks]
{removed} = ["rescue"]
"""
                with tempfile.TemporaryDirectory() as directory_name:
                    path = self.write_config(Path(directory_name), contents)
                    with self.assertRaisesRegex(
                        ConfigError, f"execution_fallbacks.{removed} is not allowed"
                    ):
                        load_config(path)

    def test_removed_semantic_revision_section_is_rejected(self) -> None:
        for body in (
            "[revision]\nenabled = true\n",
            "[revision]\nmax_step_contract_repairs = 2\n",
            "[revision]\nmax_check_repair_attempts = 1\n",
            "[revision]\nmax_correction_cycles = 1\n",
            '[ui]\ndefault_reviewer_profile = "implementer-codex"\n',
            '[ui]\ndefault_reviser_profile = "implementer-codex"\n',
            '[ui]\ndefault_repair_profile = "implementer-codex"\n',
        ):
            with self.subTest(body=body):
                with tempfile.TemporaryDirectory() as directory_name:
                    path = self.write_config(Path(directory_name), VALID_CONFIG + body)
                    with self.assertRaises(ConfigError):
                        load_config(path)

    def test_execution_fallback_profiles_must_be_arrays(self) -> None:
        contents = VALID_CONFIG + """

[recovery.execution_fallbacks]
mechanical = "rescue"
"""
        with tempfile.TemporaryDirectory() as directory_name:
            config_path = self.write_config(Path(directory_name), contents)
            with self.assertRaisesRegex(ConfigError, "execution_fallbacks.mechanical"):
                load_config(config_path)

    def test_the_transport_horizon_defaults_to_thirty_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), VALID_CONFIG))
        self.assertEqual(config.transport, TransportConfig())
        self.assertEqual(config.transport.max_wait_seconds, 1800)

    def test_the_transport_horizon_is_one_positive_integer_seconds_budget(self) -> None:
        contents = VALID_CONFIG + "\n[transport]\nmax_wait_seconds = 90\n"
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertEqual(config.transport.max_wait_seconds, 90)

    def test_the_transport_horizon_rejects_invalid_values(self) -> None:
        for body, message in (
            ("max_wait_seconds = 0", "greater than zero"),
            ("max_wait_seconds = -1", "greater than zero"),
            ("max_wait_seconds = 1.5", "must be an integer"),
            ("max_wait_seconds = true", "must be an integer"),
            ("max_wait = 10", "transport.max_wait is not allowed"),
        ):
            with self.subTest(body=body):
                contents = VALID_CONFIG + f"\n[transport]\n{body}\n"
                with tempfile.TemporaryDirectory() as directory_name:
                    config_path = self.write_config(Path(directory_name), contents)
                    with self.assertRaisesRegex(ConfigError, message):
                        load_config(config_path)

    def test_a_profile_has_no_per_endpoint_attempt_budget(self) -> None:
        contents = VALID_CONFIG.replace(
            "timeout_seconds = 300\n", "timeout_seconds = 300\nretries = 3\n",
        )
        with tempfile.TemporaryDirectory() as directory_name:
            config_path = self.write_config(Path(directory_name), contents)
            with self.assertRaisesRegex(ConfigError, "retries is not allowed"):
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
        self.assertEqual(config.planning.staged_step_max_mutable_paths, 3)
        self.assertEqual(config.planning.single_step_max_mutable_paths, 3)
        self.assertEqual(config.planning.max_steps_per_plan, 12)
        self.assertEqual(config.planning.max_read_paths_per_step, 8)
        self.assertEqual(config.planning.max_step_contract_chars, 9000)

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

        self.assertEqual(PlanningConfig().staged_step_max_mutable_paths, 3)
        self.assertEqual(PlanningConfig().max_steps_per_plan, 12)
        self.assertEqual(PlanningConfig().max_read_paths_per_step, 8)
        self.assertEqual(PlanningConfig().max_step_contract_chars, 9000)
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
        contents = VALID_CONFIG.replace(
            'default_planner_profile = "', '# default_planner_profile = "', 1,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "ui.default_planner_profile is required"):
                load_config(self.write_config(Path(directory_name), contents))
        contents = VALID_CONFIG.replace(
            'mechanical_profile = "implementer-codex"', 'mechanical_profile = "planner-chat"', 1,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "must have the implementer role"):
                load_config(self.write_config(Path(directory_name), contents))
        # The audit default is optional as long as the audit role is routable.
        contents = VALID_CONFIG.replace(
            'default_audit_profile = "implementer-codex"\n', "", 1,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            config = load_config(self.write_config(Path(directory_name), contents))
        self.assertIsNone(config.ui.default_audit_profile)

    def test_ui_active_run_capacity_is_bounded_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            path = self.write_config(Path(directory_name), VALID_CONFIG)
            self.assertEqual(load_config(path).ui.max_active_runs, 1)
        contents = VALID_CONFIG.replace("[ui]\n", "[ui]\nmax_active_runs = 5\n", 1)
        with tempfile.TemporaryDirectory() as directory_name:
            with self.assertRaisesRegex(ConfigError, "at most 4"):
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

    def test_modern_role_matrix_accepts_external_implementer_and_auditor(self) -> None:
        contents = VALID_CONFIG.replace(
            'driver = "codex"\nprovider = "bridge"\nmodel = "gpt-5.6-luna"\neffort = "high"\nsandbox = "workspace-write"\nselection_mode = "cli"',
            'driver = "external"\nprovider = "deepseek"\nmodel = "deepseek-worker"\nselection_mode = "cli"\nargv = ["trusted-worker"]',
        ).replace(
            'default_audit_profile = "implementer-codex"',
            'default_audit_profile = "auditor-claude"',
        ) + '''
[claude_runtime]
home = "claude-home"

[model_profiles.auditor-claude]
display_name = "Claude auditor"
roles = ["auditor"]
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
        self.assertEqual(config.ui.default_audit_profile, "auditor-claude")


class RunOptionsStrictSchemaTests(unittest.TestCase):
    """`run_options.json` has exactly one current shape."""

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

    def test_current_schema_eight_round_trips_identically(self) -> None:
        snapshot = self.snapshot()
        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 8)
        self.assertEqual(snapshot["profiles"]["audit_profile"], "implementer-codex")
        for removed in (
            "final_reviewer_profile", "semantic_reviser_profile",
            "check_repair_profile", "default_implementer_profile",
        ):
            self.assertNotIn(removed, snapshot["profiles"])
        self.assertNotIn("pipeline", snapshot)
        self.assertEqual(
            set(snapshot["execution_fallbacks"]), {"mechanical", "reasoning", "agentic"},
        )
        self.assertEqual(
            set(snapshot["budget"]),
            {"step_attempts", "audit_repairs", "max_iterations",
             "max_wall_clock_hours", "max_cost"},
        )
        options = RunOptions.from_mapping(snapshot)
        self.assertEqual(options.to_dict(), snapshot)
        encoded = canonical_run_options_bytes(options)
        self.assertEqual(
            canonical_run_options_bytes(RunOptions.from_mapping(json.loads(encoded))),
            encoded,
        )

    def test_previous_schema_is_rejected_without_conversion(self) -> None:
        old_snapshot = self.snapshot()
        old_snapshot["schema_version"] = 6
        with self.assertRaises(RunOptionsError) as caught:
            RunOptions.from_mapping(old_snapshot)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))
        with self.assertRaises(RunOptionsError) as caught:
            replace(RunOptions.from_mapping(self.snapshot()), schema_version=3)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))

    def test_a_schema_five_snapshot_is_rejected_without_conversion(self) -> None:
        """C7 drops the check-repair and semantic-revision surfaces: clean break."""

        legacy = self.snapshot()
        legacy["schema_version"] = 5
        legacy["pipeline"] = {
            "semantic_revision_enabled": True,
            "max_check_repair_attempts": 2,
            "max_correction_cycles": 1,
        }
        legacy["profiles"].update({
            "check_repair_profile": "implementer-codex",
            "semantic_reviser_profile": "implementer-codex",
            "final_reviewer_profile": "implementer-codex",
        })
        legacy["execution_fallbacks"].update({
            "semantic_reviser": ["implementer-codex"],
            "check_repair": ["implementer-codex"],
        })
        with self.assertRaises(RunOptionsError) as caught:
            RunOptions.from_mapping(legacy)
        self.assertIn(RUN_SCHEMA_UNSUPPORTED, str(caught.exception))

    def test_every_removed_option_name_is_rejected(self) -> None:
        cases = (
            (lambda s: s.update(pipeline={"obsolete": True}),
             "unknown key pipeline"),
            (lambda s: s["profiles"].update(check_repair_profile="implementer-codex"),
             "profiles has unknown key check_repair_profile"),
            (lambda s: s["profiles"].update(semantic_reviser_profile="implementer-codex"),
             "profiles has unknown key semantic_reviser_profile"),
            (lambda s: s["profiles"].update(final_reviewer_profile="implementer-codex"),
             "profiles has unknown key final_reviewer_profile"),
            (lambda s: s["execution_fallbacks"].update(
                semantic_reviser=["implementer-codex"]),
             "unknown key semantic_reviser"),
            (lambda s: s["execution_fallbacks"].update(
                check_repair=["implementer-codex"]),
             "unknown key check_repair"),
        )
        for mutate, message in cases:
            with self.subTest(message=message):
                snapshot = self.snapshot()
                mutate(snapshot)
                with self.assertRaisesRegex(RunOptionsError, message):
                    RunOptions.from_mapping(snapshot)

    def test_missing_budget_and_fallback_fields_are_rejected(self) -> None:
        for section, field in (
            ("budget", "step_attempts"),
            ("budget", "max_wall_clock_hours"),
            ("execution_fallbacks", "mechanical"),
        ):
            snapshot = self.snapshot()
            del snapshot[section][field]
            with self.assertRaisesRegex(RunOptionsError, f"missing {field}"):
                RunOptions.from_mapping(snapshot)
        for section in ("budget", "execution_fallbacks"):
            snapshot = self.snapshot()
            del snapshot[section]
            with self.assertRaisesRegex(RunOptionsError, f"missing {section}"):
                RunOptions.from_mapping(snapshot)

    def test_a_positive_cost_cap_in_a_snapshot_is_refused(self) -> None:
        snapshot = self.snapshot()
        snapshot["budget"]["max_cost"] = 3.0
        with self.assertRaisesRegex(RunOptionsError, "max_cost > 0 is not supported"):
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
            (lambda snapshot: snapshot["profiles"].pop("audit_profile"), "profiles is missing audit_profile"),
            (lambda snapshot: snapshot.pop("profiles"), "missing profiles"),
            (lambda snapshot: snapshot["budget"].update(max_extra_attempts=1), "budget has unknown key max_extra_attempts"),
            (lambda snapshot: snapshot.pop("budget"), "missing budget"),
        ):
            snapshot = self.snapshot()
            mutate(snapshot)
            with self.assertRaisesRegex(RunOptionsError, message):
                RunOptions.from_mapping(snapshot)

if __name__ == "__main__":
    unittest.main()
