"""Backend adapters and centralized profile-to-executor resolution."""

from __future__ import annotations

import dataclasses
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..gitops import GitError, candidate_tree_sha, current_head
from ..models import (
    AgentConfig,
    ExecutionRole,
    HarnessConfig,
    ModelProfile,
    ProfileDriver,
    profile_driver_name,
)
from ..profiles import build_agent_config, build_claude_profile
from ..usage import normalize_usage
from .base import (
    AGENT_PROTOCOL_FAILED,
    AGENT_RUNTIME_FAILED,
    AGENT_SCOPE_VIOLATION,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentExecutor,
    AgentExecutorCapabilities,
    AgentProtocolError,
    AgentRunRequest,
    AgentRunResult,
    AgentScopeError,
)
from .codex import AgentCommittedError, CodexAgent, classify_codex_failure
from ..claude.agent import (
    ClaudeCodeAgent,
    ClaudeCommittedError,
    classify_claude_failure,
)
from ..claude.runtime import ClaudeRuntimeError, prepare_claude_home
from .runtime import CodexRuntimeError, prepare_codex_home
from ..agent.codex import build_agent_environment
from ..claude.agent import build_claude_environment
from .external import ExternalAgentExecutor


@dataclass(frozen=True)
class ExecutorRuntimeConfig:
    """Runtime-only values shared by infrastructure adapters.

    ``api_key_env`` and environment values are never persisted by this class;
    the environment is used only while starting the selected subprocess.
    """

    config: HarnessConfig | None = None
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    codex_home: Path | None = None
    claude_home: Path | None = None
    forbidden_env_names: tuple[str | None, ...] = ()
    # Compatibility hook for the historical in-process Codex double.  It is
    # consumed by the Codex adapter only; the orchestration layer does not
    # select a backend from this callback.
    legacy_agent_factory: Callable[[str], Any] | None = None


def _runtime(value: Any) -> ExecutorRuntimeConfig:
    if isinstance(value, ExecutorRuntimeConfig):
        return value
    if isinstance(value, HarnessConfig):
        environment = value.runtime_environment or os.environ
        return ExecutorRuntimeConfig(
            config=value,
            environment=environment,
            codex_home=value.codex_runtime.home,
            claude_home=value.claude_runtime.home,
            legacy_agent_factory=None,
        )
    if isinstance(value, Mapping):
        config = value.get("config")
        return ExecutorRuntimeConfig(
            config=config if isinstance(config, HarnessConfig) else None,
            environment=value.get("environment", os.environ),
            codex_home=value.get("codex_home"),
            claude_home=value.get("claude_home"),
            forbidden_env_names=tuple(value.get("forbidden_env_names", ())),
            legacy_agent_factory=value.get("legacy_agent_factory"),
        )
    config = getattr(value, "config", None)
    return ExecutorRuntimeConfig(
        config=config if isinstance(config, HarnessConfig) else None,
        environment=getattr(value, "environment", os.environ),
        codex_home=getattr(value, "codex_home", None),
        claude_home=getattr(value, "claude_home", None),
        forbidden_env_names=tuple(getattr(value, "forbidden_env_names", ())),
        legacy_agent_factory=getattr(value, "legacy_agent_factory", None),
    )


def _tree(path: Path) -> str:
    try:
        return candidate_tree_sha(path)
    except (GitError, OSError, ValueError):
        return ""


def _result(
    *,
    raw: Any,
    driver: str,
    before: str,
    after: str,
    artifact_dir: Path,
    exit_reason: str | None = None,
    backend_reason: str | None = None,
    driver_version: str | None = None,
) -> AgentRunResult:
    timed_out = bool(getattr(raw, "timed_out", False))
    exit_code = getattr(raw, "exit_code", None)
    terminal_is_error = getattr(raw, "terminal_is_error", None) is True
    if timed_out:
        status = "timed_out"
        exit_reason = exit_reason or AGENT_TIMEOUT
    elif terminal_is_error:
        status = "failed"
        exit_reason = exit_reason or AGENT_PROTOCOL_FAILED
    elif exit_code not in (None, 0):
        status = "failed"
        exit_reason = exit_reason or AGENT_RUNTIME_FAILED
    else:
        status = "completed"
    final_message = str(getattr(raw, "final_message", ""))
    stderr_tail = str(getattr(raw, "stderr_tail", ""))
    report = artifact_dir / "agent.final.md"
    return AgentRunResult(
        status=status,
        exit_reason=exit_reason,
        tree_before=before,
        tree_after=after,
        usage=normalize_usage(getattr(raw, "usage", None)),
        external_session_id=getattr(raw, "external_session_id", None),
        report_path=str(report) if report.exists() else None,
        exit_code=exit_code,
        timed_out=timed_out,
        final_message=final_message,
        stderr_tail=stderr_tail,
        driver=driver,
        driver_version=driver_version,
        backend_reason=backend_reason,
        terminal_type=getattr(raw, "terminal_type", None),
        terminal_subtype=getattr(raw, "terminal_subtype", None),
        terminal_is_error=getattr(raw, "terminal_is_error", None),
        terminal_num_turns=getattr(raw, "terminal_num_turns", None),
        terminal_stop_reason=getattr(raw, "terminal_stop_reason", None),
        terminal_errors=tuple(getattr(raw, "terminal_errors", ()) or ()),
        raw_result=raw,
    )


def _failed_result(
    *,
    driver: str,
    request: AgentRunRequest,
    before: str,
    exc: BaseException,
) -> AgentRunResult:
    code = getattr(exc, "code", None)
    backend_reason = code if isinstance(code, str) else type(exc).__name__
    if isinstance(exc, (AgentCommittedError, ClaudeCommittedError, AgentScopeError)):
        reason = AGENT_SCOPE_VIOLATION
    elif isinstance(exc, AgentProtocolError):
        reason = AGENT_PROTOCOL_FAILED
    elif isinstance(exc, (CodexRuntimeError, ClaudeRuntimeError)):
        reason = AGENT_START_FAILED
    elif isinstance(exc, AgentError):
        reason = AGENT_START_FAILED
    else:
        reason = AGENT_RUNTIME_FAILED
    return AgentRunResult(
        status="failed",
        exit_reason=reason,
        tree_before=before,
        tree_after=_tree(request.worktree),
        usage=None,
        external_session_id=None,
        report_path=None,
        exit_code=None,
        driver=driver,
        backend_reason=backend_reason,
        stderr_tail="",
    )


class CodexExecutor:
    """Adapter exposing the generic contract for the Codex harness."""

    driver = ProfileDriver.CODEX.value

    def __init__(
        self,
        profile: ModelProfile,
        runtime_config: Any = None,
        *,
        agent: Any | None = None,
    ) -> None:
        self.profile = profile
        self.runtime = _runtime(runtime_config)
        if agent is not None:
            self.agent = agent
        else:
            compatibility_factory = self.runtime.legacy_agent_factory
            compatibility_agent = (
                compatibility_factory(profile.id)
                if callable(compatibility_factory)
                else None
            )
            if compatibility_agent is not None:
                self.agent = compatibility_agent
            else:
                config = self.runtime.config
                if config is not None:
                    agent_config = dataclasses.replace(
                        build_agent_config(profile),
                        env_allowlist=config.agent.env_allowlist,
                    )
                else:
                    agent_config = build_agent_config(profile)
                self.agent = CodexAgent(agent_config)
        self.capabilities = AgentExecutorCapabilities(
            edits_workspace=True,
            exposes_session_id=False,
            exposes_usage=True,
            exposes_reasoning_usage=True,
            exposes_tool_count=False,
            isolation_mode=profile.sandbox,
        )

    @property
    def driver_version(self) -> str | None:
        value = getattr(self.agent, "driver_version", None)
        return value if isinstance(value, str) and value.strip() else self.profile.driver_version

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        before = _tree(request.worktree)
        try:
            config = self.runtime.config
            codex_home = self.runtime.codex_home
            if config is not None:
                codex_home = prepare_codex_home(config)
            agent_config = getattr(self.agent, "config", None)
            if not isinstance(agent_config, AgentConfig):
                agent_config = build_agent_config(self.profile)
            environment = build_agent_environment(
                agent_config,
                source_environment=self.runtime.environment,
                codex_home=codex_home,
                forbidden_names=self.runtime.forbidden_env_names,
            )
            base_sha = current_head(request.worktree)
            if request.prompt_mode == "plan" and hasattr(self.agent, "run"):
                raw = self.agent.run(
                    request.prompt, request.worktree, request.artifact_dir,
                    base_sha=base_sha, env=environment,
                )
            elif hasattr(self.agent, "run_prompt"):
                raw = self.agent.run_prompt(
                    request.prompt, request.worktree, request.artifact_dir,
                    base_sha=base_sha, env=environment,
                )
            elif hasattr(self.agent, "run_step"):
                # The fallback adapter also receives the exact already-rendered
                # payload. It must not construct a competing prompt.
                kwargs = {
                    "base_sha": base_sha,
                    "env": environment,
                }
                raw = self.agent.run_step(
                    request.prompt,
                    request.worktree,
                    request.artifact_dir,
                    **kwargs,
                )
            else:
                raw = self.agent.run(
                    request.prompt, request.worktree, request.artifact_dir,
                    base_sha=base_sha, env=environment,
                )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return _failed_result(
                driver=self.driver, request=request, before=before, exc=exc
            )
        after = _tree(request.worktree)
        backend_reason = classify_codex_failure(
            str(getattr(raw, "stderr_tail", "")),
            _read_artifact_tail(request.artifact_dir / "agent.events.jsonl"),
        )
        return _result(
            raw=raw, driver=self.driver, before=before, after=after,
            artifact_dir=request.artifact_dir,
            backend_reason=backend_reason,
            driver_version=self.driver_version,
        )


class ClaudeCodeExecutor:
    """Adapter exposing the generic contract for Claude Code."""

    driver = ProfileDriver.CLAUDE_CODE.value

    def __init__(
        self,
        profile: ModelProfile,
        runtime_config: Any = None,
        *,
        agent: Any | None = None,
    ) -> None:
        self.profile = build_claude_profile(profile)
        self.runtime = _runtime(runtime_config)
        self.agent = agent or ClaudeCodeAgent()
        self.capabilities = AgentExecutorCapabilities(
            edits_workspace=True,
            exposes_session_id=False,
            exposes_usage=True,
            exposes_reasoning_usage=False,
            exposes_tool_count=False,
            isolation_mode="restricted",
        )

    @property
    def driver_version(self) -> str | None:
        value = getattr(self.agent, "driver_version", None)
        return value if isinstance(value, str) and value.strip() else self.profile.driver_version

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        before = _tree(request.worktree)
        try:
            config = self.runtime.config
            home = self.runtime.claude_home
            if config is not None:
                home = prepare_claude_home(config)
            if home is None:
                raise AgentError("managed Claude runtime home is not configured")
            environment = build_claude_environment(
                self.runtime.environment, claude_home=Path(home)
            )
            raw = self.agent.run_revision(
                request.prompt, request.worktree,
                artifacts_dir=request.artifact_dir,
                profile=self.profile,
                environment=environment,
                revision_dir=request.artifact_dir,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return _failed_result(
                driver=self.driver, request=request, before=before, exc=exc
            )
        after = _tree(request.worktree)
        backend_reason = classify_claude_failure(
            str(getattr(raw, "stderr_tail", "")),
            _read_artifact_tail(request.artifact_dir / "agent.events.jsonl"),
        )
        return _result(
            raw=raw, driver=self.driver, before=before, after=after,
            artifact_dir=request.artifact_dir,
            backend_reason=backend_reason,
            driver_version=self.driver_version,
        )


class LegacyAgentExecutor:
    """Compatibility adapter for callers injecting the pre-v2 agent API."""

    def __init__(self, executor: AgentExecutor, *, legacy_failure_names: bool = True):
        self.executor = executor
        self.legacy_failure_names = legacy_failure_names

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        return self.executor.run(request)

    @property
    def capabilities(self) -> AgentExecutorCapabilities:
        return self.executor.capabilities

    @property
    def driver_version(self) -> str | None:
        return self.executor.driver_version


def legacy_codex_agent_factory(config: AgentConfig) -> Any:
    """Compatibility seam for callers that patched the old constructor.

    The orchestrator only sees this factory as an injection hook; the concrete
    constructor remains owned by this infrastructure module.
    """

    return CodexAgent(config)


def _read_artifact_tail(path: Path, limit: int = 256 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


ExecutorFactory = Callable[..., AgentExecutor]


class ExecutorDriverRegistry:
    """Allowlisted mapping from trusted driver IDs to infrastructure adapters."""

    def __init__(self) -> None:
        self._factories: dict[str, ExecutorFactory] = {}

    def register(self, driver: ProfileDriver | str, factory: ExecutorFactory, *, replace: bool = False) -> None:
        name = profile_driver_name(driver)
        if not callable(factory):
            raise TypeError("executor factory must be callable")
        if name in self._factories and not replace:
            raise ValueError(f"executor driver {name!r} is already registered")
        self._factories[name] = factory

    def resolve(
        self,
        profile: ModelProfile,
        runtime_config: Any = None,
        *,
        agent: Any | None = None,
        reviser: Any | None = None,
    ) -> AgentExecutor:
        name = profile_driver_name(profile.driver)
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(
                f"unsupported concrete driver {name!r} for profile {profile.id!r}; "
                "register an executor driver adapter"
            )
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        if accepts_kwargs or "agent" in parameters:
            kwargs["agent"] = agent
        if accepts_kwargs or "reviser" in parameters:
            kwargs["reviser"] = reviser
        return factory(profile, runtime_config, **kwargs)


EXECUTOR_REGISTRY = ExecutorDriverRegistry()


def register_executor_driver(
    driver: ProfileDriver | str,
    factory: ExecutorFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one trusted concrete driver adapter for profile resolution."""

    EXECUTOR_REGISTRY.register(driver, factory, replace=replace)


def _codex_factory(
    profile: ModelProfile, runtime_config: Any = None, *, agent: Any | None = None, reviser: Any | None = None
) -> AgentExecutor:
    del reviser
    return CodexExecutor(profile, runtime_config, agent=agent)


def _claude_factory(
    profile: ModelProfile, runtime_config: Any = None, *, agent: Any | None = None, reviser: Any | None = None
) -> AgentExecutor:
    return ClaudeCodeExecutor(profile, runtime_config, agent=reviser or agent)


def _external_factory(
    profile: ModelProfile, runtime_config: Any = None, *, agent: Any | None = None, reviser: Any | None = None
) -> AgentExecutor:
    del agent, reviser
    return ExternalAgentExecutor(profile, runtime_config)


register_executor_driver(ProfileDriver.CODEX, _codex_factory)
register_executor_driver(ProfileDriver.CLAUDE_CODE, _claude_factory)
register_executor_driver(ProfileDriver.EXTERNAL, _external_factory)


def executor_for_profile(
    profile: ModelProfile,
    runtime_config: Any = None,
    *,
    agent: Any | None = None,
    reviser: Any | None = None,
) -> AgentExecutor:
    """Resolve exactly one infrastructure adapter for a selected profile."""

    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    return EXECUTOR_REGISTRY.resolve(
        profile, runtime_config, agent=agent, reviser=reviser
    )


__all__ = [
    "ClaudeCodeExecutor",
    "CodexExecutor",
    "EXECUTOR_REGISTRY",
    "ExecutorDriverRegistry",
    "ExecutorRuntimeConfig",
    "ExternalAgentExecutor",
    "LegacyAgentExecutor",
    "executor_for_profile",
    "legacy_codex_agent_factory",
    "register_executor_driver",
]
