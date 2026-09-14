"""Chargement, expansion et validation de la configuration TOML."""

from __future__ import annotations

import math
import os
import re
import tomllib
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

from .environment import EnvironmentFileError, build_runtime_environment
from .llm.chat import PROTECTED_BODY_KEYS, LLMProtocolError, validate_endpoint
from .models import (
    AgentConfig,
    ApprovalConfig,
    CheckConfig,
    CodexRuntimeConfig,
    ContextConfig,
    EnvironmentConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    PlanningConfig,
    ProfileDriver,
    RepositoryConfig,
    SelectionMode,
    UIConfig,
    WorkspaceSetupCommand,
)


class ConfigError(ValueError):
    """Configuration absente, mal formée ou invalide."""


_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_KNOWN_SANDBOXES = frozenset({
    "read-only",
    "workspace-write",
    "danger-full-access",
})
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PROFILE_COMMON_KEYS = frozenset({
    "display_name",
    "roles",
    "driver",
    "model",
    "selection_mode",
    "timeout_seconds",
    "description",
    "strengths",
    "cost_tier",
    "latency_tier",
})
_PROFILE_DRIVER_KEYS = {
    ProfileDriver.OPENAI_CHAT: frozenset({
        "base_url", "endpoint_path", "api_key_env", "retries", "extra_body",
    }),
    # Codex has no retry policy: ``retries`` would be a silently unused option.
    ProfileDriver.CODEX: frozenset({"effort", "sandbox"}),
}
_PROFILE_ROLE_COMPATIBILITY = {
    ProfileDriver.OPENAI_CHAT: frozenset({
        ExecutionRole.PLANNER, ExecutionRole.REVIEWER, ExecutionRole.AUDITOR,
    }),
    ProfileDriver.CODEX: frozenset({
        ExecutionRole.IMPLEMENTER, ExecutionRole.REPAIR,
    }),
}


def _expand_string(value: str, environment: Mapping[str, str]) -> str:
    """Expand ``${NAME}`` references in one non-recursive pass.

    A value taken from the environment is inserted literally: a ``${...}``
    sequence inside it is never expanded again.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environment:
            raise ConfigError(f"environment variable {name!r} is not set")
        return environment[name]

    return _ENV_VAR.sub(replace, value)


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    """Recursively expand strings in TOML tables and arrays."""

    if isinstance(value, str):
        return _expand_string(value, environment)
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item, environment) for item in value)
    if isinstance(value, dict):
        return {key: _expand(item, environment) for key, item in value.items()}
    return value


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"section [{name}] must be a table")
    return value


def _required_string(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"{where}.{key} must be a string")
    if not value.strip():
        raise ConfigError(f"{where}.{key} must not be empty")
    return value


def _optional_string(
    data: Mapping[str, Any], key: str, default: str | None, where: str
) -> str | None:
    value = data.get(key, default)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"{where}.{key} must be a string or null")
    if isinstance(value, str) and not value.strip():
        raise ConfigError(f"{where}.{key} must not be empty")
    return value


def _repository_web_url(data: Mapping[str, Any]) -> str | None:
    value = _optional_string(data, "web_url", None, "repository")
    if value is None:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ConfigError("repository.web_url must be a valid HTTPS URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
    ):
        raise ConfigError("repository.web_url must be an HTTPS URL without credentials, query, or fragment")
    return value


def _optional_env_name(
    data: Mapping[str, Any], key: str, default: str | None, where: str
) -> str | None:
    value = _optional_string(data, key, default, where)
    if value is not None and _ENV_NAME.fullmatch(value) is None:
        # Do not include the value: this field is intended to contain a name,
        # and an invalid value could itself be an API key.
        raise ConfigError(f"{where}.{key} must be an environment variable name")
    return value


def _positive_int(data: Mapping[str, Any], key: str, default: int, where: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer")
    if value <= 0:
        raise ConfigError(f"{where}.{key} must be greater than zero")
    return value


def _nonnegative_int(
    data: Mapping[str, Any], key: str, default: int, where: str
) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer")
    if value < 0:
        raise ConfigError(f"{where}.{key} must not be negative")
    return value


def _bool(data: Mapping[str, Any], key: str, default: bool, where: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be a boolean")
    return value


def _bounded_float(
    data: Mapping[str, Any],
    key: str,
    default: float,
    where: str,
    *,
    maximum: float,
) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{where}.{key} must be greater than zero")
    if result > maximum:
        raise ConfigError(f"{where}.{key} must be at most {maximum:g}")
    return result


def _bounded_int(
    data: Mapping[str, Any],
    key: str,
    default: int,
    where: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{where}.{key} must be at least {minimum}")
    if value > maximum:
        raise ConfigError(f"{where}.{key} must be at most {maximum}")
    return value


def _string_array(data: Mapping[str, Any], key: str, default: tuple[str, ...], where: str,
                  *, allow_empty: bool = True) -> tuple[str, ...]:
    value = data.get(key, list(default))
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}.{key} must be an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{where}.{key} must be an array of strings")
    result = tuple(value)
    if not allow_empty and not result:
        raise ConfigError(f"{where}.{key} must not be empty")
    return result


def _env_name_array(
    data: Mapping[str, Any], key: str, default: tuple[str, ...], where: str
) -> tuple[str, ...]:
    result = _string_array(data, key, default, where)
    if any(_ENV_NAME.fullmatch(name) is None for name in result):
        raise ConfigError(f"{where}.{key} must contain environment variable names")
    return result


def _path(value: Any, key: str, config_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _endpoint(data: Mapping[str, Any], name: str) -> LLMEndpointConfig:
    base_url = _required_string(data, "base_url", name)
    endpoint_path = _required_string(data, "endpoint_path", name)
    model = _required_string(data, "model", name)
    api_key_env = _optional_env_name(data, "api_key_env", None, name)
    timeout_seconds = _positive_int(data, "timeout_seconds", 300, name)
    retries = _nonnegative_int(data, "retries", 2, name)
    try:
        validate_endpoint(base_url, endpoint_path)
    except LLMProtocolError as exc:
        raise ConfigError(f"{name}: {exc}") from None
    extra_body = data.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ConfigError(f"{name}.extra_body must be a table")
    protected = PROTECTED_BODY_KEYS.intersection(extra_body)
    if protected:
        raise ConfigError(
            f"{name}.extra_body cannot override protected request keys: "
            + ", ".join(sorted(protected))
        )
    return LLMEndpointConfig(
        base_url=base_url,
        endpoint_path=endpoint_path,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        retries=retries,
        extra_body=dict(extra_body),
    )


def _profile_endpoint(profile: ModelProfile) -> LLMEndpointConfig:
    if profile.base_url is None or profile.endpoint_path is None:
        raise ConfigError(f"profile {profile.id!r} has no OpenAI endpoint")
    return LLMEndpointConfig(
        base_url=profile.base_url,
        endpoint_path=profile.endpoint_path,
        model=profile.model,
        api_key_env=profile.api_key_env,
        timeout_seconds=profile.timeout_seconds,
        retries=profile.retries,
        extra_body=dict(profile.extra_body),
    )


def _profile_agent(profile: ModelProfile) -> AgentConfig:
    if profile.driver is not ProfileDriver.CODEX:
        raise ConfigError(f"profile {profile.id!r} is not a Codex profile")
    return AgentConfig(
        model=profile.model,
        effort=profile.effort or AgentConfig.effort,
        sandbox=profile.sandbox or AgentConfig.sandbox,
        timeout_seconds=profile.timeout_seconds,
    )


def _legacy_selection_mode(data: Mapping[str, Any], name: str) -> SelectionMode:
    value = data.get("selection_mode", SelectionMode.REQUEST.value)
    try:
        mode = SelectionMode(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}.selection_mode is invalid") from exc
    if mode is SelectionMode.CLI:
        raise ConfigError(f"{name}.selection_mode must be request or external-ui")
    return mode


def _model_profiles(
    data: Mapping[str, Any],
) -> tuple[dict[str, ModelProfile], bool]:
    raw = data.get("model_profiles", {})
    if not isinstance(raw, dict):
        raise ConfigError("model_profiles must be a table")
    if not raw:
        return {}, False
    result: dict[str, ModelProfile] = {}
    for profile_id, profile_data in raw.items():
        if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
            raise ConfigError("model profile id is invalid")
        if not isinstance(profile_data, dict):
            raise ConfigError(f"model_profiles.{profile_id} must be a table")
        where = f"model_profiles.{profile_id}"
        try:
            driver = ProfileDriver(profile_data.get("driver"))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}.driver is invalid") from exc
        # Fail closed on typos and on options the driver does not use.
        unknown = sorted(
            set(profile_data) - _PROFILE_COMMON_KEYS - _PROFILE_DRIVER_KEYS[driver]
        )
        if unknown:
            raise ConfigError(f"{where}.{unknown[0]} is not allowed for {driver.value}")
        display_name = _required_string(profile_data, "display_name", where)
        roles_raw = profile_data.get("roles")
        if isinstance(roles_raw, (str, bytes)) or not isinstance(roles_raw, (list, tuple)) or not roles_raw:
            raise ConfigError(f"{where}.roles must be a non-empty array")
        roles: list[ExecutionRole] = []
        for value in roles_raw:
            try:
                role = ExecutionRole(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{where}.roles contains an invalid role") from exc
            if role in roles:
                raise ConfigError(f"{where}.roles must not contain duplicates")
            roles.append(role)
        model = _required_string(profile_data, "model", where)
        description = profile_data.get("description", "")
        if not isinstance(description, str):
            raise ConfigError(f"{where}.description must be a string")
        if len(description) > 300:
            raise ConfigError(f"{where}.description must be at most 300 characters")
        strengths = _string_array(profile_data, "strengths", (), where)
        if len(strengths) > 8 or any(len(item) > 80 for item in strengths):
            raise ConfigError(
                f"{where}.strengths must contain at most 8 entries of at most 80 characters"
            )
        cost_tier = profile_data.get("cost_tier", "standard")
        if not isinstance(cost_tier, str) or cost_tier not in {"low", "standard", "high"}:
            raise ConfigError(f"{where}.cost_tier is invalid")
        latency_tier = profile_data.get("latency_tier", "standard")
        if not isinstance(latency_tier, str) or latency_tier not in {"fast", "standard", "slow"}:
            raise ConfigError(f"{where}.latency_tier is invalid")
        try:
            selection_mode = SelectionMode(profile_data.get("selection_mode"))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where}.selection_mode is invalid") from exc
        incompatible = set(roles) - _PROFILE_ROLE_COMPATIBILITY[driver]
        if incompatible:
            raise ConfigError(f"{where}: driver/role mismatch")
        if driver is ProfileDriver.OPENAI_CHAT:
            if selection_mode is SelectionMode.CLI:
                raise ConfigError(f"{where}.selection_mode must not be cli")
            endpoint = _endpoint(profile_data, where)
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                base_url=endpoint.base_url,
                endpoint_path=endpoint.endpoint_path,
                api_key_env=endpoint.api_key_env,
                timeout_seconds=endpoint.timeout_seconds,
                retries=endpoint.retries,
                extra_body=endpoint.extra_body,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
        else:
            effort = _required_string(profile_data, "effort", where)
            sandbox = _required_string(profile_data, "sandbox", where)
            if sandbox not in _KNOWN_SANDBOXES:
                raise ConfigError(f"unknown {where}.sandbox")
            if selection_mode is not SelectionMode.CLI:
                raise ConfigError(f"{where}.selection_mode must be cli")
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                timeout_seconds=_positive_int(profile_data, "timeout_seconds", 300, where),
                effort=effort,
                sandbox=sandbox,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
    return result, True


def _check_default(
    profiles: Mapping[str, ModelProfile], value: str | None, role: ExecutionRole
) -> str:
    if value is None:
        raise ConfigError(f"ui.default_{role.value}_profile is required")
    profile = profiles.get(value)
    if profile is None:
        raise ConfigError(f"default profile {value!r} does not exist")
    if role not in profile.roles:
        raise ConfigError(f"default profile {value!r} is incompatible with {role.value}")
    return value


def _checks(value: Any) -> tuple[CheckConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("checks must be an array of tables")
    result: list[CheckConfig] = []
    for index, item in enumerate(value):
        where = f"checks[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be a table")
        name = _required_string(item, "name", where)
        argv = _string_array(item, "argv", (), where, allow_empty=False)
        cwd = item.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ConfigError(f"{where}.cwd must be a non-empty string")
        timeout = _positive_int(item, "timeout_seconds", 3600, where)
        required = _bool(item, "required", True, where)
        result.append(CheckConfig(name, argv, cwd, timeout, required))
    return tuple(result)


def _environment_config(raw: Mapping[str, Any], config_dir: Path) -> EnvironmentConfig:
    data = _table(raw, "environment")
    files = data.get("files", [])
    if isinstance(files, str) or not isinstance(files, (list, tuple)):
        raise ConfigError("environment.files must be an array of paths")
    paths: list[Path] = []
    for index, value in enumerate(files):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"environment.files[{index}] must be a non-empty path string")
        paths.append(_path(value, f"environment.files[{index}]", config_dir))
    return EnvironmentConfig(tuple(paths))


def _codex_runtime(
    raw: Mapping[str, Any], config_dir: Path, *, required: bool
) -> CodexRuntimeConfig:
    data = _table(raw, "codex_runtime")
    if "home" not in data:
        if required:
            raise ConfigError("codex_runtime.home is required for explicit Codex profiles")
        return CodexRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "codex"
        )
    return CodexRuntimeConfig(_path(data["home"], "codex_runtime.home", config_dir))


def _workspace_setup(value: Any) -> tuple[WorkspaceSetupCommand, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("workspace_setup must be an array of tables")
    result: list[WorkspaceSetupCommand] = []
    for index, item in enumerate(value):
        where = f"workspace_setup[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be a table")
        name = _required_string(item, "name", where)
        argv = _string_array(item, "argv", (), where, allow_empty=False)
        cwd = item.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip() or "\x00" in cwd:
            raise ConfigError(f"{where}.cwd must be a valid relative path")
        if Path(cwd).is_absolute():
            raise ConfigError(f"{where}.cwd must be relative to the worktree")
        env_allowlist = _env_name_array(
            item, "env_allowlist", WorkspaceSetupCommand.env_allowlist, where
        )
        result.append(
            WorkspaceSetupCommand(
                name=name,
                argv=argv,
                cwd=cwd,
                timeout_seconds=_positive_int(item, "timeout_seconds", 1200, where),
                env_allowlist=env_allowlist,
            )
        )
    return tuple(result)


def load_config(config_path: str | Path) -> HarnessConfig:
    """Load and validate a TOML configuration file."""

    path = Path(config_path).expanduser().resolve()
    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc

    config_dir = path.parent
    try:
        environment = _environment_config(raw, config_dir)
        runtime_environment = build_runtime_environment(
            environment.files, os.environ
        )
    except EnvironmentFileError as exc:
        raise ConfigError(str(exc)) from None

    expanded = _expand(raw, runtime_environment)
    if not isinstance(expanded, dict):  # pragma: no cover - tomllib guarantee
        raise ConfigError("configuration root must be a table")
    repo = _path(expanded.get("repo"), "repo", config_dir)
    base_ref = _required_string(expanded, "base_ref", "root")
    runs_root = _path(expanded.get("runs_root"), "runs_root", config_dir)
    worktrees_root = _path(expanded.get("worktrees_root"), "worktrees_root", config_dir)
    require_clean_base = _bool(expanded, "require_clean_base", True, "root")
    max_diff_bytes = _positive_int(expanded, "max_diff_bytes", 400_000, "root")

    planning_data = _table(expanded, "planning")
    protocol = planning_data.get("protocol", "v1")
    if not isinstance(protocol, str) or protocol not in {"v1", "v2"}:
        raise ConfigError("planning.protocol must be 'v1' or 'v2'")
    decomposition = planning_data.get("decomposition", "balanced")
    if not isinstance(decomposition, str) or decomposition not in {"balanced", "aggressive"}:
        raise ConfigError("planning.decomposition must be 'balanced' or 'aggressive'")
    single_step_max_mutable_paths = _positive_int(
        planning_data, "single_step_max_mutable_paths", 2, "planning"
    )
    planning = PlanningConfig(
        protocol=protocol,
        decomposition=decomposition,
        single_step_max_mutable_paths=single_step_max_mutable_paths,
    )

    repository_data = _table(expanded, "repository")
    repository_remote = _required_string(repository_data, "remote", "repository") if "remote" in repository_data else "origin"
    if "\x00" in repository_remote or any(char.isspace() for char in repository_remote):
        raise ConfigError("repository.remote must be a non-whitespace remote name")
    repository = RepositoryConfig(
        remote=repository_remote,
        planner_remote_exploration=_bool(
            repository_data, "planner_remote_exploration", True, "repository"
        ),
        web_url=_repository_web_url(repository_data),
    )

    agent_data = _table(expanded, "agent")
    provider = agent_data.get("provider", "codex")
    if provider != "codex":
        raise ConfigError("agent.provider must be 'codex' in V0")
    agent_model = agent_data.get("model", AgentConfig.model)
    effort = agent_data.get("effort", AgentConfig.effort)
    sandbox = agent_data.get("sandbox", AgentConfig.sandbox)
    for key, value in (("model", agent_model), ("effort", effort), ("sandbox", sandbox)):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"agent.{key} must be a non-empty string")
    if sandbox not in _KNOWN_SANDBOXES:
        raise ConfigError("unknown agent.sandbox")
    agent = AgentConfig(
        provider=provider,
        model=agent_model,
        effort=effort,
        sandbox=sandbox,
        timeout_seconds=_positive_int(agent_data, "timeout_seconds", 5400, "agent"),
        env_allowlist=_env_name_array(
            agent_data, "env_allowlist", AgentConfig.env_allowlist, "agent"
        ),
    )

    model_profiles, explicit_profiles = _model_profiles(expanded)
    has_explicit_codex = any(
        profile.driver is ProfileDriver.CODEX for profile in model_profiles.values()
    )
    planner_data = _table(expanded, "planner")
    reviewer_data = _table(expanded, "reviewer")
    agent_data_present = "agent" in expanded
    if explicit_profiles:
        ui_data = _table(expanded, "ui")
        planner_default = _check_default(
            model_profiles,
            ui_data.get("default_planner_profile"),
            ExecutionRole.PLANNER,
        )
        implementer_default = _check_default(
            model_profiles,
            ui_data.get("default_implementer_profile"),
            ExecutionRole.IMPLEMENTER,
        )
        reviewer_default = _check_default(
            model_profiles,
            ui_data.get("default_reviewer_profile"),
            ExecutionRole.REVIEWER,
        )
        planner_profile = model_profiles[planner_default]
        reviewer_profile = model_profiles[reviewer_default]
        implementer_profile = model_profiles[implementer_default]
        if planner_data:
            planner = _endpoint(planner_data, "planner")
        else:
            planner = _profile_endpoint(planner_profile)
        if reviewer_data:
            reviewer = _endpoint(reviewer_data, "reviewer")
        else:
            reviewer = _profile_endpoint(reviewer_profile)
        if agent_data_present:
            # Keep accepting the P15 section while the profile is the source
            # of truth for production execution.
            old_agent_data = _table(expanded, "agent")
            agent = AgentConfig(
                model=_required_string(old_agent_data, "model", "agent"),
                effort=_required_string(old_agent_data, "effort", "agent"),
                sandbox=_required_string(old_agent_data, "sandbox", "agent"),
                timeout_seconds=_positive_int(old_agent_data, "timeout_seconds", 5400, "agent"),
                env_allowlist=_env_name_array(
                    old_agent_data, "env_allowlist", AgentConfig.env_allowlist, "agent"
                ),
            )
        else:
            agent = _profile_agent(implementer_profile)
    else:
        planner = _endpoint(planner_data, "planner")
        reviewer = _endpoint(reviewer_data, "reviewer")
        # Synthetic profiles deliberately mirror the legacy sections without
        # being written back to TOML.
        model_profiles = {
            "legacy-planner": ModelProfile(
                id="legacy-planner",
                display_name="Legacy Planner",
                roles=(ExecutionRole.PLANNER,),
                driver=ProfileDriver.OPENAI_CHAT,
                model=planner.model,
                selection_mode=_legacy_selection_mode(planner_data, "planner"),
                base_url=planner.base_url,
                endpoint_path=planner.endpoint_path,
                api_key_env=planner.api_key_env,
                timeout_seconds=planner.timeout_seconds,
                retries=planner.retries,
                extra_body=planner.extra_body,
            ),
            "legacy-implementer": ModelProfile(
                id="legacy-implementer",
                display_name="Legacy Implementer",
                roles=(ExecutionRole.IMPLEMENTER,),
                driver=ProfileDriver.CODEX,
                model=agent.model,
                selection_mode=SelectionMode.CLI,
                effort=agent.effort,
                sandbox=agent.sandbox,
                timeout_seconds=agent.timeout_seconds,
            ),
            "legacy-reviewer": ModelProfile(
                id="legacy-reviewer",
                display_name="Legacy Reviewer",
                roles=(ExecutionRole.REVIEWER,),
                driver=ProfileDriver.OPENAI_CHAT,
                model=reviewer.model,
                selection_mode=_legacy_selection_mode(reviewer_data, "reviewer"),
                base_url=reviewer.base_url,
                endpoint_path=reviewer.endpoint_path,
                api_key_env=reviewer.api_key_env,
                timeout_seconds=reviewer.timeout_seconds,
                retries=reviewer.retries,
                extra_body=reviewer.extra_body,
            ),
        }

    context_data = _table(expanded, "context")
    context = ContextConfig(
        always_files=_string_array(
            context_data, "always_files", ContextConfig.always_files, "context"
        ),
        locator_argv=_string_array(context_data, "locator_argv", (), "context"),
        locator_timeout_seconds=_positive_int(
            context_data, "locator_timeout_seconds", 120, "context"
        ),
        max_hits=_positive_int(context_data, "max_hits", 8, "context"),
        max_bytes=_positive_int(context_data, "max_bytes", 160_000, "context"),
        require_locator_head_at_base=_bool(
            context_data, "require_locator_head_at_base", True, "context"
        ),
    )

    approval_data = _table(expanded, "approval")
    approval = ApprovalConfig(
        require_plan_approval=_bool(
            approval_data, "require_plan_approval", False, "approval"
        ),
        poll_interval_seconds=_bounded_float(
            approval_data,
            "poll_interval_seconds",
            0.5,
            "approval",
            maximum=10.0,
        ),
    )

    ui_data = _table(expanded, "ui")
    ui = UIConfig(
        max_active_runs=_bounded_int(
            ui_data,
            "max_active_runs",
            1,
            "ui",
            minimum=1,
            maximum=4,
        ),
        default_planner_profile=(
            planner_default if explicit_profiles else "legacy-planner"
        ),
        default_implementer_profile=(
            implementer_default if explicit_profiles else "legacy-implementer"
        ),
        default_reviewer_profile=(
            reviewer_default if explicit_profiles else "legacy-reviewer"
        ),
        enable_profile_recommendation=_bool(
            ui_data, "enable_profile_recommendation", True, "ui"
        ),
    )

    checks = _checks(expanded.get("checks", []))
    codex_runtime = _codex_runtime(
        expanded, config_dir, required=has_explicit_codex
    )
    for label, root in (
        ("repo", repo),
        ("runs_root", runs_root),
        ("worktrees_root", worktrees_root),
    ):
        try:
            codex_runtime.home.relative_to(root)
        except ValueError:
            continue
        raise ConfigError(f"codex_runtime.home must not be inside {label}")
    workspace_setup = _workspace_setup(expanded.get("workspace_setup", []))
    allow_no_required_checks = _bool(
        expanded, "allow_no_required_checks", False, "root"
    )
    if not allow_no_required_checks and not any(check.required for check in checks):
        raise ConfigError(
            "at least one required check is configured; set allow_no_required_checks = true for docs-only projects"
        )

    return HarnessConfig(
        repo=repo,
        base_ref=base_ref,
        runs_root=runs_root,
        worktrees_root=worktrees_root,
        require_clean_base=require_clean_base,
        planner=planner,
        reviewer=reviewer,
        context=context,
        agent=agent,
        checks=checks,
        max_diff_bytes=max_diff_bytes,
        allow_no_required_checks=allow_no_required_checks,
        approval=approval,
        ui=ui,
        model_profiles=model_profiles,
        environment=environment,
        runtime_environment=runtime_environment,
        codex_runtime=codex_runtime,
        workspace_setup=workspace_setup,
        planning=planning,
        repository=repository,
        repository_section_explicit="repository" in expanded,
    )
