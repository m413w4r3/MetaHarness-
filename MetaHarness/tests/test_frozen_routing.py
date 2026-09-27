import json
import tempfile
import unittest
from pathlib import Path

from metaharness.agent.codex import build_agent_environment
from metaharness.agent.runtime import prepare_codex_home
from metaharness.execution_selection import resolve_execution_selection
from metaharness.models import (
    AgentConfig,
    CodexProviderConfig,
    ContextConfig,
    CodexRuntimeConfig,
    ExecutionClass,
    ExecutionRole,
    HarnessConfig,
    ImplementationStep,
    ModelProfile,
    ProfileDriver,
    RoutingConfig,
    SelectionMode,
    UIConfig,
)
from metaharness.run_options import (
    RUN_SCHEMA_UNSUPPORTED,
    SCHEMA_VERSION,
    RunOptions,
    RunOptionsError,
)
from metaharness.recovery_policy import ExecutionFallbacks


def frozen_routing_config() -> HarnessConfig:
    """Build the smallest secret-free config needed by the routing tests."""

    profiles = {
        "planner-chat": ModelProfile(
            "planner-chat",
            "Planner",
            (ExecutionRole.PLANNER,),
            ProfileDriver.OPENAI_CHAT,
            "planner-model",
            SelectionMode.REQUEST,
            base_url="https://planner.example",
            endpoint_path="/v1/chat",
            api_key_env="PLANNER_API_KEY",
        ),
        "auditor-chat": ModelProfile(
            "auditor-chat",
            "Auditor",
            (ExecutionRole.AUDITOR,),
            ProfileDriver.OPENAI_CHAT,
            "auditor-model",
            SelectionMode.REQUEST,
            base_url="https://auditor.example",
            endpoint_path="/v1/chat",
            api_key_env="AUDITOR_API_KEY",
        ),
        "codex-luna-high": ModelProfile(
            "codex-luna-high",
            "Luna High",
            (ExecutionRole.IMPLEMENTER,),
            ProfileDriver.CODEX,
            "gpt-6-luna",
            SelectionMode.CLI,
            effort="high",
            sandbox="workspace-write",
        ),
        "codex-luna-xhigh": ModelProfile(
            "codex-luna-xhigh",
            "Luna XHigh",
            (ExecutionRole.IMPLEMENTER,),
            ProfileDriver.CODEX,
            "gpt-6-luna",
            SelectionMode.CLI,
            effort="xhigh",
            sandbox="workspace-write",
        ),
        "codex-deepseek-flash-max": ModelProfile(
            "codex-deepseek-flash-max",
            "DeepSeek Flash Max",
            (ExecutionRole.IMPLEMENTER,),
            ProfileDriver.CODEX,
            "deepseek-flash",
            SelectionMode.CLI,
            provider="deepseek",
            effort="max",
            sandbox="workspace-write",
        ),
    }
    repository = Path(__file__).resolve().parents[1]
    return HarnessConfig(
        repo=repository,
        base_ref="main",
        runs_root=repository / ".test-runs",
        worktrees_root=repository / ".test-worktrees",
        require_clean_base=True,
        context=ContextConfig(always_files=()),
        check_catalog=(),
        allow_no_required_checks=True,
        ui=UIConfig(
            default_planner_profile="planner-chat",
            default_audit_profile="auditor-chat",
        ),
        model_profiles=profiles,
        routing=RoutingConfig(
            mechanical_profile="codex-luna-high",
            reasoning_profile="codex-luna-xhigh",
            agentic_profile="codex-deepseek-flash-max",
        ),
        codex_providers={
            "deepseek": CodexProviderConfig(
                "deepseek",
                "https://api.deepseek.com/",
                "responses",
                "DEEPSEEK_API_KEY",
            )
        },
        execution_fallbacks=ExecutionFallbacks(
            mechanical=("codex-luna-xhigh",),
            reasoning=("codex-luna-high",),
            agentic=("codex-luna-high",),
        ),
    )


class FrozenRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = frozen_routing_config()

    def test_execution_classes_resolve_to_frozen_profiles(self) -> None:
        options = RunOptions.from_config(self.config)
        steps = tuple(
            ImplementationStep(
                id=step_id,
                title=step_id,
                execution_class=execution_class,
                depends_on=None,
                context="context",
                read_set=(),
                write_set=(),
                create_set=(),
                delete_set=(),
                instructions="1. do",
                interfaces="NONE",
                examples="NONE",
                tests="- test",
                pitfalls="- none",
                done_when="- done",
                verify="- check",
            )
            for step_id, execution_class in (
                ("S01", ExecutionClass.MECHANICAL),
                ("S02", ExecutionClass.REASONING),
                ("S03", ExecutionClass.AGENTIC),
            )
        )
        selection = resolve_execution_selection(
            self.config,
            planner_profile_id=options.planner_profile,
            plan_steps=steps,
            audit_profile_id=options.audit_profile,
        )
        self.assertEqual(
            [item.implementer.profile_id for item in selection.steps],
            ["codex-luna-high", "codex-luna-xhigh", "codex-deepseek-flash-max"],
        )

    def test_snapshot_is_secret_free_and_managed_catalog_is_minimal(self) -> None:
        options = RunOptions.from_config(self.config)
        self.assertNotIn("default_implementer_profile", json.dumps(options.to_dict()))
        snapshot = options.to_dict()
        self.assertEqual(snapshot["budget"]["step_attempts"], 3)
        self.assertEqual(
            snapshot["execution_fallbacks"]["mechanical"], ("codex-luna-xhigh",),
        )
        self.assertEqual(
            RunOptions.from_mapping(snapshot).budget, options.budget,
        )
        self.assertEqual(
            RunOptions.from_mapping(snapshot).execution_fallbacks,
            options.execution_fallbacks,
        )
        incomplete = json.loads(json.dumps(snapshot))
        del incomplete["budget"]["audit_repairs"]
        with self.assertRaisesRegex(RunOptionsError, "missing audit_repairs"):
            RunOptions.from_mapping(incomplete)
        without_budget = json.loads(json.dumps(snapshot))
        del without_budget["budget"]
        with self.assertRaisesRegex(RunOptionsError, "missing budget"):
            RunOptions.from_mapping(without_budget)
        older = json.loads(json.dumps(snapshot))
        older["schema_version"] = SCHEMA_VERSION - 1
        with self.assertRaisesRegex(RunOptionsError, RUN_SCHEMA_UNSUPPORTED):
            RunOptions.from_mapping(older)
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "codex"
            runtime_config = HarnessConfig(
                repo=self.config.repo,
                base_ref=self.config.base_ref,
                runs_root=Path(directory) / "runs",
                worktrees_root=Path(directory) / "worktrees",
                require_clean_base=True,
                context=ContextConfig(always_files=()),
                check_catalog=(),
                allow_no_required_checks=True,
                codex_runtime=CodexRuntimeConfig(home),
                model_profiles=self.config.model_profiles,
                routing=self.config.routing,
                codex_providers=self.config.codex_providers,
            )
            prepare_codex_home(runtime_config)
            catalog = json.loads((home / "models.json").read_text())
            self.assertEqual(catalog["models"][0]["slug"], "deepseek-flash")
            self.assertEqual(
                {item["effort"] for item in catalog["models"][0]["supported_reasoning_levels"]},
                {"low", "high", "max"},
            )
            self.assertNotIn("api-key-value", (home / "config.toml").read_text())

    def test_provider_key_is_profile_scoped(self) -> None:
        source = {"PATH": "/bin", "DEEPSEEK_API_KEY": "api-key-value"}
        openai = build_agent_environment(
            AgentConfig(env_allowlist=("PATH",), provider="openai"),
            source_environment=source,
        )
        deepseek = build_agent_environment(
            AgentConfig(
                env_allowlist=("PATH",),
                provider="deepseek",
                provider_api_key_env="DEEPSEEK_API_KEY",
            ),
            source_environment=source,
        )
        self.assertNotIn("DEEPSEEK_API_KEY", openai)
        self.assertEqual(deepseek["DEEPSEEK_API_KEY"], "api-key-value")


if __name__ == "__main__":
    unittest.main()
