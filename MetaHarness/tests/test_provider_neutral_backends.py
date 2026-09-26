from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from metaharness.agent import (
    AgentRunRequest,
    AgentRunResult,
    ClaudeCodeExecutor,
    CodexExecutor,
    ExecutorRuntimeConfig,
    ExternalAgentExecutor,
    executor_for_profile,
    register_executor_driver,
)
from metaharness.config import load_config
from metaharness.models import (
    AgentExecutorCapabilities,
    ExecutionRole,
    ModelProfile,
    ProfileDriver,
    SelectionMode,
)
from metaharness.profiles import profile_execution_fingerprint


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def profile(
    driver: ProfileDriver | str,
    *,
    provider: str = "provider-a",
    roles: tuple[ExecutionRole, ...] = (ExecutionRole.IMPLEMENTER,),
    argv: tuple[str, ...] = (),
    driver_version: str | None = None,
) -> ModelProfile:
    return ModelProfile(
        id=f"{str(driver)}-{provider}",
        display_name="Test profile",
        roles=roles,
        driver=driver,
        model="exact-test-model",
        selection_mode=SelectionMode.CLI,
        argv=argv,
        effort="test-effort",
        sandbox="workspace-write" if driver is ProfileDriver.CODEX else None,
        permission_mode="acceptEdits" if driver is ProfileDriver.CLAUDE_CODE else None,
        driver_version=driver_version,
        provider=provider,
    )


class _FakeExecutor:
    capabilities = AgentExecutorCapabilities()
    driver_version = None

    def __init__(self, driver: str) -> None:
        self.driver = driver
        self.roles: list[ExecutionRole] = []

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.roles.append(request.role)
        return AgentRunResult(
            status="completed",
            exit_reason=None,
            tree_before="before",
            tree_after="after",
            usage=None,
            external_session_id=None,
            report_path=None,
            driver=self.driver,
        )


class _CodexDouble:
    config = None


class _ClaudeDouble:
    pass


class ProviderNeutralBackendTests(unittest.TestCase):
    def test_capabilities_do_not_claim_unavailable_external_features(self) -> None:
        external = ExternalAgentExecutor(
            profile(ProfileDriver.EXTERNAL, argv=(sys.executable, "-c", "pass")),
            ExecutorRuntimeConfig(environment={}),
        )
        self.assertEqual(external.capabilities, AgentExecutorCapabilities())
        codex = CodexExecutor(
            profile(ProfileDriver.CODEX),
            ExecutorRuntimeConfig(environment={}),
            agent=_CodexDouble(),
        )
        claude = ClaudeCodeExecutor(
            profile(ProfileDriver.CLAUDE_CODE, roles=(ExecutionRole.REVISER,)),
            ExecutorRuntimeConfig(environment={}),
            agent=_ClaudeDouble(),
        )
        self.assertTrue(codex.capabilities.edits_workspace)
        self.assertTrue(codex.capabilities.exposes_usage)
        self.assertFalse(codex.capabilities.exposes_session_id)
        self.assertTrue(claude.capabilities.edits_workspace)
        self.assertFalse(claude.capabilities.exposes_reasoning_usage)
        self.assertIsNone(external.capabilities.isolation_mode)

    def test_registry_uses_one_backend_neutral_role_path_for_implementers(self) -> None:
        executors: dict[str, _FakeExecutor] = {}
        for driver in ("test-driver-a", "test-driver-b", "test-driver-c"):
            def factory(profile: ModelProfile, _runtime: object, *, driver=driver, **_: object):
                value = _FakeExecutor(driver)
                executors[profile.id] = value
                return value

            register_executor_driver(driver, factory, replace=True)

        request = AgentRunRequest(
            role=ExecutionRole.IMPLEMENTER,
            profile_id="unused",
            prompt="same prompt",
            worktree=Path("."),
            artifact_dir=Path("."),
            mutable_paths=(),
        )
        results = []
        for driver in ("test-driver-a", "test-driver-b", "test-driver-c"):
            selected = profile(driver)
            executor = executor_for_profile(selected, ExecutorRuntimeConfig(environment={}))
            results.append(executor.run(request))
        self.assertEqual([result.status for result in results], ["completed"] * 3)
        self.assertEqual([result.tree_after for result in results], ["after"] * 3)
        self.assertEqual(
            [executors[f"{driver}-provider-a"].roles for driver in (
                "test-driver-a", "test-driver-b", "test-driver-c"
            )],
            [[ExecutionRole.IMPLEMENTER]] * 3,
        )

    def test_registry_uses_one_backend_neutral_role_path_for_semantic_revisers(self) -> None:
        for driver in ("test-reviser-a", "test-reviser-b"):
            register_executor_driver(
                driver,
                lambda _profile, _runtime, **_: _FakeExecutor(driver),
                replace=True,
            )
        request = AgentRunRequest(
            role=ExecutionRole.REVISER,
            profile_id="reviser",
            prompt="review and repair",
            worktree=Path("."),
            artifact_dir=Path("."),
            mutable_paths=(),
        )
        for driver in ("test-reviser-a", "test-reviser-b"):
            result = executor_for_profile(
                profile(driver, roles=(ExecutionRole.REVISER,)),
                ExecutorRuntimeConfig(environment={}),
            ).run(request)
            self.assertEqual(result.status, "completed")

    def test_fingerprint_distinguishes_driver_and_provider(self) -> None:
        base = profile("registered-harness", provider="provider-a")
        other_driver = profile("other-harness", provider="provider-a")
        other_provider = profile("registered-harness", provider="provider-b")
        self.assertNotEqual(profile_execution_fingerprint(base), profile_execution_fingerprint(other_driver))
        self.assertNotEqual(profile_execution_fingerprint(base), profile_execution_fingerprint(other_provider))

    def test_unsupported_concrete_driver_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported concrete driver"):
            executor_for_profile(profile("unregistered-concrete-driver"))

    def test_external_config_schema_and_process_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            git(repo, "init", "-q")
            git(repo, "config", "user.name", "backend test")
            git(repo, "config", "user.email", "backend@example.invalid")
            (repo / "README.md").write_text("base\n", encoding="utf-8")
            git(repo, "add", "README.md")
            git(repo, "commit", "-qm", "base")
            script = (
                "import pathlib, sys; "
                "pathlib.Path('prompt.txt').write_text(sys.stdin.read()); "
                "print('external done')"
            )
            config_path = root / "config.toml"
            config_path.write_text(
                f"""repo = {str(repo)!r}
base_ref = "HEAD"
runs_root = {str(root / 'runs')!r}
worktrees_root = {str(root / 'worktrees')!r}
allow_no_required_checks = true

[ui]
default_planner_profile = "chat"
default_implementer_profile = "worker"
default_reviewer_profile = "chat"

[model_profiles.chat]
display_name = "Chat"
roles = ["planner", "reviewer"]
driver = "openai-chat"
model = "planner-model"
selection_mode = "request"
base_url = "https://chat.invalid"
endpoint_path = "/chat"

[model_profiles.worker]
display_name = "Trusted worker"
roles = ["implementer", "repair", "reviser"]
driver = "external"
provider = "provider-x"
model = "exact-worker-model"
selection_mode = "cli"
argv = [{json.dumps(sys.executable)}, "-c", {json.dumps(script)}]
effort = "configured-effort"
driver_version = "local-test"
timeout_seconds = 10
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            worker = config.model_profiles["worker"]
            self.assertIs(worker.driver, ProfileDriver.EXTERNAL)
            self.assertEqual(worker.provider, "provider-x")
            self.assertEqual(worker.model, "exact-worker-model")
            self.assertEqual(worker.effort, "configured-effort")
            self.assertEqual(worker.driver_version, "local-test")
            self.assertEqual(worker.argv[0], sys.executable)

            executor = executor_for_profile(
                worker,
                ExecutorRuntimeConfig(
                    config=config,
                    environment={},
                ),
            )
            result = executor.run(
                AgentRunRequest(
                    role=ExecutionRole.IMPLEMENTER,
                    profile_id=worker.id,
                    prompt="safe prompt\n",
                    worktree=repo,
                    artifact_dir=root / "artifacts",
                    mutable_paths=("prompt.txt",),
                )
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.driver_version, "local-test")
            self.assertIsNone(result.usage)
            self.assertNotEqual(result.tree_before, result.tree_after)
            self.assertEqual((repo / "prompt.txt").read_text(), "safe prompt\n")
            self.assertLessEqual(len(result.stderr_tail.encode()), 16 * 1024)


if __name__ == "__main__":
    unittest.main()
