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
from metaharness.run_options import RunOptions
from metaharness.recovery_policy import ExecutionFallbacks, RecoveryBudgets


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
        "reviewer-chat": ModelProfile(
            "reviewer-chat",
            "Reviewer",
            (ExecutionRole.REVIEWER,),
            ProfileDriver.OPENAI_CHAT,
            "reviewer-model",
            SelectionMode.REQUEST,
            base_url="https://reviewer.example",
            endpoint_path="/v1/chat",
            api_key_env="REVIEWER_API_KEY",
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
            default_reviewer_profile="reviewer-chat",
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
        recovery=RecoveryBudgets(
            execution_fallbacks=ExecutionFallbacks(
                mechanical=("codex-luna-xhigh",),
                reasoning=("codex-luna-high",),
                agentic=("codex-luna-high",),
            )
        ),
    )


class FrozenRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = frozen_routing_config()

    def test_execution_classes_resolve_to_frozen_profiles(self) -> None:
        options = RunOptions.from_config(self.config)
        steps = tuple(
            ImplementationStep(
                step_id,
                step_id,
                execution_class,
                None,
                "",
                (),
                (),
                "",
                "",
                "",
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
            check_repair_profile_id=options.check_repair_profile,
            semantic_reviser_profile_id=options.semantic_reviser_profile,
            final_reviewer_profile_id=options.final_reviewer_profile,
        )
        self.assertEqual(
            [item.implementer.profile_id for item in selection.steps],
            ["codex-luna-high", "codex-luna-xhigh", "codex-deepseek-flash-max"],
        )

    def test_snapshot_is_secret_free_and_managed_catalog_is_minimal(self) -> None:
        options = RunOptions.from_config(self.config)
        self.assertNotIn("default_implementer_profile", json.dumps(options.to_dict()))
        snapshot = options.to_dict()
        self.assertEqual(snapshot["recovery"]["max_transient_attempts"], 2)
        self.assertEqual(
            RunOptions.from_mapping(snapshot).recovery,
            options.recovery,
        )
        legacy_snapshot = dict(snapshot)
        legacy_snapshot.pop("recovery")
        self.assertEqual(
            RunOptions.from_mapping(legacy_snapshot).recovery,
            RecoveryBudgets(),
        )
        # Snapshots frozen before output corrections keep their recovery table.
        pre_correction = json.loads(json.dumps(snapshot))
        del pre_correction["recovery"]["max_contract_repair_output_corrections"]
        self.assertEqual(
            RunOptions.from_mapping(pre_correction).recovery.max_contract_repair_output_corrections, 2,
        )
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
