import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metaharness.agent import (
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AgentError,
    AgentExecutor,
    AgentProtocolError,
    AgentRunRequest,
    AgentRunResult,
    AgentScopeError,
    ClaudeCodeExecutor,
    CodexExecutor,
    ExecutorRuntimeConfig,
    executor_for_profile,
)
from metaharness.agent.base import AgentResult
from metaharness.models import ExecutionRole, ModelProfile, ProfileDriver, SelectionMode
from metaharness.orchestrator import Orchestrator, ResumeError
from metaharness.resume import pipeline_version_from_state
from metaharness.state import RunStateStore


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def profile(driver: ProfileDriver, role: ExecutionRole) -> ModelProfile:
    return ModelProfile(
        id=f"{driver.value}-{role.value}",
        display_name="test profile",
        roles=(role,),
        driver=driver,
        model="test-model",
        selection_mode=SelectionMode.CLI if driver is ProfileDriver.CODEX else SelectionMode.EXTERNAL_UI,
        effort="high",
        sandbox="workspace-write" if driver is ProfileDriver.CODEX else None,
        permission_mode="acceptEdits" if driver is ProfileDriver.CLAUDE_CODE else None,
        retries=0 if driver is ProfileDriver.CLAUDE_CODE else 2,
    )


class _CodexDouble:
    config = None

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def run_prompt(self, prompt: str, worktree: Path, artifacts_dir: Path, **_: object) -> AgentResult:
        if self.error is not None:
            raise self.error
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "agent.final.md").write_text("done\n", encoding="utf-8")
        return AgentResult(0, False, "done\n", {"output_tokens": 2}, "")


class _ClaudeDouble:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    def run_revision(self, prompt: str, worktree: Path, **kwargs: object) -> AgentResult:
        if self.error is not None:
            raise self.error
        artifact_dir = Path(kwargs["artifacts_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "agent.final.md").write_text("done\n", encoding="utf-8")
        return AgentResult(0, False, "done\n", {"output_tokens": 3}, "")


class AgentExecutionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "contract test")
        git(self.root, "config", "user.email", "contract@example.invalid")
        (self.root / "README.md").write_text("base\n", encoding="utf-8")
        git(self.root, "add", "README.md")
        git(self.root, "commit", "-qm", "base")
        self.request = AgentRunRequest(
            role=ExecutionRole.IMPLEMENTER,
            profile_id="codex-implementer",
            prompt="implement",
            worktree=self.root,
            artifact_dir=self.root / "artifacts",
            mutable_paths=("README.md",),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_both_adapters_satisfy_one_backend_neutral_contract(self) -> None:
        runtime = ExecutorRuntimeConfig(
            environment={},
            codex_home=self.root / "codex-home",
            claude_home=self.root / "claude-home",
        )
        codex = CodexExecutor(profile(ProfileDriver.CODEX, ExecutionRole.IMPLEMENTER), runtime, agent=_CodexDouble())
        claude_profile = profile(ProfileDriver.CLAUDE_CODE, ExecutionRole.REVISER)
        claude_request = replace(self.request, role=ExecutionRole.REVISER, profile_id=claude_profile.id)
        claude = ClaudeCodeExecutor(claude_profile, runtime, agent=_ClaudeDouble())
        self.assertIsInstance(codex, AgentExecutor)
        self.assertIsInstance(claude, AgentExecutor)
        self.assertEqual(codex.run(self.request).status, "completed")
        self.assertEqual(claude.run(claude_request).status, "completed")

    def test_resolver_selects_driver_adapter_without_orchestrator_branching(self) -> None:
        runtime = ExecutorRuntimeConfig(environment={}, claude_home=self.root / "claude-home")
        codex = executor_for_profile(profile(ProfileDriver.CODEX, ExecutionRole.IMPLEMENTER), runtime, agent=_CodexDouble())
        claude = executor_for_profile(profile(ProfileDriver.CLAUDE_CODE, ExecutionRole.REVISER), runtime, reviser=_ClaudeDouble())
        self.assertIsInstance(codex, CodexExecutor)
        self.assertIsInstance(claude, ClaudeCodeExecutor)

    def test_adapter_maps_start_protocol_scope_and_runtime_failures_generically(self) -> None:
        runtime = ExecutorRuntimeConfig(environment={}, codex_home=self.root / "codex-home")
        selected = profile(ProfileDriver.CODEX, ExecutionRole.IMPLEMENTER)
        cases = (
            (AgentError("start"), AGENT_START_FAILED),
            (AgentProtocolError("protocol"), AGENT_PROTOCOL_FAILED),
            (AgentScopeError("scope"), AGENT_SCOPE_VIOLATION),
            (OSError("runtime"), AGENT_RUNTIME_FAILED),
        )
        for error, expected in cases:
            with self.subTest(expected=expected):
                result = CodexExecutor(selected, runtime, agent=_CodexDouble(error)).run(self.request)
                self.assertEqual(result.exit_reason, expected)
                self.assertEqual(result.status, "failed")


class PipelineVersionTests(unittest.TestCase):
    def test_new_state_is_v2_and_absent_field_is_historical_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = RunStateStore(path).initialize("new")
            self.assertEqual(state["pipeline_version"], 2)
            historic = dict(state)
            historic.pop("pipeline_version")
            self.assertEqual(pipeline_version_from_state(historic), 1)

    def test_resume_dispatch_never_crosses_pipeline_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "runs" / "v1" / "state.json"
            store = RunStateStore(state_path)
            state_path.parent.mkdir(parents=True)
            state = store.initialize("v1", pipeline_version=1)
            state.pop("pipeline_version")
            state_path.write_text(json.dumps(state), encoding="utf-8")
            harness = object.__new__(Orchestrator)
            harness.config = type("Config", (), {"runs_root": root / "runs"})()
            with mock.patch.object(harness, "_resume_impl", return_value="v1") as impl:
                self.assertEqual(harness.resume("v1"), "v1")
                impl.assert_called_once()
            with self.assertRaises(ResumeError):
                harness.resume_pipeline_v2("v1")


if __name__ == "__main__":
    unittest.main()
