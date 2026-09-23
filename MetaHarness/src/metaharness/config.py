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
    ClaudeRuntimeConfig,
    CodexRuntimeConfig,
    CodexProviderConfig,
    ContextConfig,
    EnvironmentConfig,
    ExecutionModePolicy,
    ExecutionRole,
    GitHubConfig,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    PlanningConfig,
    PromptBudgetConfig,
    PublishConfig,
    PublishMode,
    ProfileDriver,
    RoutingConfig,
    RepositoryConfig,
    RevisionConfig,
    SelectionMode,
    UIConfig,
    WorkspaceSetupCommand,
    profile_driver_name,
    validate_revision_budget,
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
    "provider",
    "model",
    "selection_mode",
    "timeout_seconds",
    "driver_version",
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
    ProfileDriver.CLAUDE_CODE: frozenset({"effort", "permission_mode"}),
    # Generic trusted process adapter.  No provider command line protocol is
    # implied: argv is passed exactly as configured and prompt bytes go to
    # stdin.
    ProfileDriver.EXTERNAL: frozenset({"argv", "effort"}),
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


def _revision_budget(
    data: Mapping[str, Any], key: str, default: int, where: str
) -> int:
    """Read one correction budget through the shared model validator."""

    value = data.get(key, default)
    try:
        return validate_revision_budget(value, f"{where}.{key}")
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


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


def _model_profiles(
    data: Mapping[str, Any],
) -> dict[str, ModelProfile]:
    raw = data.get("model_profiles")
    if not isinstance(raw, dict):
        raise ConfigError("model_profiles is required and must be a table")
    if not raw:
        raise ConfigError("model_profiles must contain at least one profile")
    result: dict[str, ModelProfile] = {}
    for profile_id, profile_data in raw.items():
        if not isinstance(profile_id, str) or _PROFILE_ID.fullmatch(profile_id) is None:
            raise ConfigError("model profile id is invalid")
        if not isinstance(profile_data, dict):
            raise ConfigError(f"model_profiles.{profile_id} must be a table")
        where = f"model_profiles.{profile_id}"
        raw_driver = profile_data.get("driver")
        try:
            driver: ProfileDriver | str = ProfileDriver(raw_driver)
        except (TypeError, ValueError) as exc:
            # Extension IDs are accepted as metadata/configuration only.  A
            # concrete adapter must still be registered before execution.
            if (
                not isinstance(raw_driver, str)
                or not raw_driver.strip()
                or "-" not in raw_driver
                or not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", raw_driver)
            ):
                raise ConfigError(f"{where}.driver is invalid") from exc
            driver = raw_driver
        # Fail closed on typos and on options the driver does not use.
        driver_keys = _PROFILE_DRIVER_KEYS.get(driver, frozenset({"effort"}))
        unknown = sorted(set(profile_data) - _PROFILE_COMMON_KEYS - driver_keys)
        if unknown:
            raise ConfigError(
                f"{where}.{unknown[0]} is not allowed for {profile_driver_name(driver)}"
            )
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
        provider = profile_data.get("provider", "openai")
        if not isinstance(provider, str) or not provider.strip():
            raise ConfigError(f"{where}.provider must be a non-empty string")
        driver_version = _optional_string(profile_data, "driver_version", None, where)
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
                provider=provider,
                base_url=endpoint.base_url,
                endpoint_path=endpoint.endpoint_path,
                api_key_env=endpoint.api_key_env,
                timeout_seconds=endpoint.timeout_seconds,
                retries=endpoint.retries,
                extra_body=endpoint.extra_body,
                driver_version=driver_version,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
        elif driver is ProfileDriver.CODEX:
            if selection_mode is not SelectionMode.CLI:
                raise ConfigError(f"{where}.selection_mode must be cli")
            effort = _required_string(profile_data, "effort", where)
            sandbox = _required_string(profile_data, "sandbox", where)
            if sandbox not in _KNOWN_SANDBOXES:
                raise ConfigError(f"unknown {where}.sandbox")
            permission_mode = None
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                provider=provider,
                timeout_seconds=_positive_int(profile_data, "timeout_seconds", 300, where),
                retries=2,
                effort=effort,
                sandbox=sandbox,
                permission_mode=permission_mode,
                driver_version=driver_version,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
        elif driver is ProfileDriver.CLAUDE_CODE:
            if selection_mode is not SelectionMode.CLI:
                raise ConfigError(f"{where}.selection_mode must be cli")
            effort = _required_string(profile_data, "effort", where)
            permission_mode = _required_string(profile_data, "permission_mode", where)
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                provider=provider,
                timeout_seconds=_positive_int(profile_data, "timeout_seconds", 300, where),
                retries=0,
                effort=effort,
                permission_mode=permission_mode,
                driver_version=driver_version,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
        elif driver is ProfileDriver.EXTERNAL:
            if selection_mode is not SelectionMode.CLI:
                raise ConfigError(f"{where}.selection_mode must be cli")
            argv = _string_array(profile_data, "argv", (), where, allow_empty=False)
            if any(not item or "\x00" in item for item in argv):
                raise ConfigError(f"{where}.argv must contain non-empty strings without NUL")
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                provider=provider,
                timeout_seconds=_positive_int(profile_data, "timeout_seconds", 300, where),
                retries=0,
                argv=argv,
                effort=_optional_string(profile_data, "effort", None, where),
                driver_version=driver_version,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
        else:
            # Unknown extension drivers intentionally have no invented command
            # syntax.  Their trusted adapter owns any additional runtime
            # contract once registered.
            result[profile_id] = ModelProfile(
                id=profile_id,
                display_name=display_name,
                roles=tuple(roles),
                driver=driver,
                model=model,
                selection_mode=selection_mode,
                provider=provider,
                timeout_seconds=_positive_int(profile_data, "timeout_seconds", 300, where),
                retries=0,
                effort=_optional_string(profile_data, "effort", None, where),
                driver_version=driver_version,
                description=description,
                strengths=strengths,
                cost_tier=cost_tier,
                latency_tier=latency_tier,
            )
    return result


def _routing(data: Mapping[str, Any], profiles: Mapping[str, ModelProfile]) -> RoutingConfig:
    raw = data.get("routing")
    if raw is None:
        # Configs written before class routing was introduced can still be
        # loaded while their run snapshot is upgraded to the modern shape.
        legacy = _table(data, "ui").get("default_implementer_profile")
        if isinstance(legacy, str) and legacy.strip():
            raw = {
                "mechanical_profile": legacy,
                "reasoning_profile": legacy,
                "agentic_profile": legacy,
            }
        else:
            raise ConfigError("ui.default_implementer_profile is required")
    if not isinstance(raw, dict):
        raise ConfigError("routing must be a table")
    values = {
        name: _required_string(raw, name, "routing")
        for name in ("mechanical_profile", "reasoning_profile", "agentic_profile")
    }
    routing = RoutingConfig(**values)
    for name in ("mechanical_profile", "reasoning_profile", "agentic_profile"):
        profile_id = getattr(routing, name)
        profile = profiles.get(profile_id)
        if profile is None:
            raise ConfigError(f"routing.{name} references unknown profile {profile_id!r}")
        if ExecutionRole.IMPLEMENTER not in profile.roles:
            raise ConfigError(f"routing.{name} profile must have the implementer role")
    return routing


def _codex_providers(data: Mapping[str, Any]) -> dict[str, CodexProviderConfig]:
    raw = data.get("codex_providers", {})
    if not isinstance(raw, dict):
        raise ConfigError("codex_providers must be a table")
    result: dict[str, CodexProviderConfig] = {}
    for name, provider_data in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(provider_data, dict):
            raise ConfigError("codex provider configuration is invalid")
        where = f"codex_providers.{name}"
        unknown = sorted(set(provider_data) - {"base_url", "wire_api", "api_key_env"})
        if unknown:
            raise ConfigError(f"{where}.{unknown[0]} is not allowed")
        base_url = _required_string(provider_data, "base_url", where)
        wire_api = _required_string(provider_data, "wire_api", where)
        api_key_env = _optional_env_name(provider_data, "api_key_env", None, where)
        if api_key_env is None:
            raise ConfigError(f"{where}.api_key_env is required")
        result[name] = CodexProviderConfig(name, base_url, wire_api, api_key_env)
    return result


def _check_default(
    profiles: Mapping[str, ModelProfile], value: str | None, role: ExecutionRole
) -> str:
    if value is None:
        raise ConfigError(f"ui.default_{role.value}_profile is required")
    profile = profiles.get(value)
    if profile is None:
        raise ConfigError(f"default profile {value!r} does not exist")
    if role not in profile.roles:
        raise ConfigError(
            f"default profile {value!r} is incompatible with {role.value}"
        )
    return value


def _validate_revision(
    revision: RevisionConfig,
    planning: PlanningConfig,
    ui: UIConfig,
    profiles: Mapping[str, ModelProfile],
) -> None:
    """Validate role/profile presence without coupling roles to drivers."""
    if (revision.enabled or revision.max_review_repair_cycles > 0) and planning.protocol != "v2":
        raise ConfigError("revision corrections require planning.protocol = 'v2'")

    if revision.max_check_repair_attempts > 0:
        value = ui.default_repair_profile
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("revision check-repair budget requires ui.default_repair_profile")
        repair = profiles.get(value)
        if repair is None or ExecutionRole.REPAIR not in repair.roles:
            raise ConfigError("revision check-repair profile must resolve for repair")

    if revision.enabled or revision.max_review_repair_cycles > 0:
        value = ui.default_reviser_profile
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("revision review budget requires ui.default_reviser_profile")
        reviser = profiles.get(value)
        if reviser is None or ExecutionRole.REVISER not in reviser.roles:
            raise ConfigError("revision semantic reviser profile must resolve for reviser")


def _checks(value: Any, *, catalogue: bool = False) -> tuple[CheckConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError("check_catalog must be an array of tables")
    result: list[CheckConfig] = []
    for index, item in enumerate(value):
        where = f"check_catalog[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be a table")
        name = _required_string(item, "id", where)
        if name in {entry.name for entry in result}:
            raise ConfigError(f"{where}.id must be unique")
        argv = _string_array(item, "argv", (), where, allow_empty=False)
        cwd = item.get("cwd", ".")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ConfigError(f"{where}.cwd must be a non-empty string")
        timeout = _positive_int(item, "timeout_seconds", 3600, where)
        required = _bool(item, "required", True, where)
        preflight_argv = _string_array(item, "preflight_argv", (), where)
        description = item.get("description", "")
        if not isinstance(description, str) or len(description) > 300:
            raise ConfigError(f"{where}.description must be a string of at most 300 characters")
        result.append(CheckConfig(name, argv, cwd, timeout, required, preflight_argv, description))
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
    return CodexRuntimeConfig(
        _path(data["home"], "codex_runtime.home", config_dir),
        _env_name_array(data, "env_allowlist", AgentConfig.env_allowlist, "codex_runtime"),
    )


def _claude_runtime(
    raw: Mapping[str, Any], config_dir: Path, *, required: bool
) -> ClaudeRuntimeConfig:
    data = _table(raw, "claude_runtime")
    if "home" not in data:
        if required:
            raise ConfigError("claude_runtime.home is required for explicit Claude Code profiles")
        return ClaudeRuntimeConfig(
            Path.home() / ".local" / "share" / "metaharness" / "claude"
        )
    return ClaudeRuntimeConfig(_path(data["home"], "claude_runtime.home", config_dir))


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
    protocol = planning_data.get("protocol", "v2")
    if protocol != "v2":
        raise ConfigError("planning.protocol must be 'v2'")
    decomposition = planning_data.get("decomposition", "aggressive")
    if not isinstance(decomposition, str) or decomposition not in {"balanced", "aggressive"}:
        raise ConfigError("planning.decomposition must be 'balanced' or 'aggressive'")
    single_step_max_mutable_paths = _positive_int(
        planning_data, "single_step_max_mutable_paths", 2, "planning"
    )
    staged_step_max_mutable_paths = _positive_int(
        planning_data, "staged_step_max_mutable_paths", 5, "planning"
    )
    max_steps_per_plan = _bounded_int(
        planning_data, "max_steps_per_plan", 8, "planning", minimum=1, maximum=99
    )
    max_read_paths_per_step = _positive_int(
        planning_data, "max_read_paths_per_step", 8, "planning"
    )
    max_step_contract_chars = _positive_int(
        planning_data, "max_step_contract_chars", 5000, "planning"
    )
    execution_mode_policy = planning_data.get(
        "execution_mode_policy", ExecutionModePolicy.AUTO.value
    )
    if not isinstance(execution_mode_policy, str) or execution_mode_policy not in {
        item.value for item in ExecutionModePolicy
    }:
        raise ConfigError(
            "planning.execution_mode_policy must be 'auto' or 'require-staged'"
        )
    planning = PlanningConfig(
        protocol=protocol,
        decomposition=decomposition,
        single_step_max_mutable_paths=single_step_max_mutable_paths,
        staged_step_max_mutable_paths=staged_step_max_mutable_paths,
        execution_mode_policy=execution_mode_policy,
        max_steps_per_plan=max_steps_per_plan,
        max_read_paths_per_step=max_read_paths_per_step,
        max_step_contract_chars=max_step_contract_chars,
        max_preapproval_corrections=_bounded_int(
            planning_data, "max_preapproval_corrections", 2, "planning", minimum=0, maximum=10
        ),
    )

    revision_data = _table(expanded, "revision")
    allowed_revision = {
        "enabled", "max_review_repair_cycles", "max_check_repair_attempts",
        "max_step_contract_repairs",
    }
    unknown_revision = sorted(set(revision_data) - allowed_revision)
    if unknown_revision:
        raise ConfigError(f"revision.{unknown_revision[0]} is not allowed")
    revision_enabled = _bool(revision_data, "enabled", False, "revision")
    review_budget = _revision_budget(
        revision_data, "max_review_repair_cycles", 1, "revision"
    )
    check_budget = _revision_budget(
        revision_data, "max_check_repair_attempts", 2, "revision"
    )
    contract_budget = _revision_budget(
        revision_data, "max_step_contract_repairs", 2, "revision"
    )
    if not revision_data:
        # A config with no [revision] section has no correction pipeline.
        review_budget = check_budget = contract_budget = 0
    revision = RevisionConfig(
        enabled=revision_enabled,
        max_review_repair_cycles=review_budget,
        max_check_repair_attempts=check_budget,
        max_step_contract_repairs=contract_budget,
    )

    prompt_budget_data = _table(expanded, "prompt_budget")
    prompt_budget = PromptBudgetConfig(
        planner_max_bytes=_positive_int(
            prompt_budget_data, "planner_max_bytes", 160_000, "prompt_budget"
        ),
        implementer_max_bytes=_positive_int(
            prompt_budget_data, "implementer_max_bytes", 120_000, "prompt_budget"
        ),
        check_repair_max_bytes=_positive_int(
            prompt_budget_data, "check_repair_max_bytes", 40_000, "prompt_budget"
        ),
        semantic_revision_max_bytes=_positive_int(
            prompt_budget_data, "semantic_revision_max_bytes", 120_000, "prompt_budget"
        ),
        final_review_max_bytes=_positive_int(
            prompt_budget_data, "final_review_max_bytes", 120_000, "prompt_budget"
        ),
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

    github_data = _table(expanded, "github")
    github_enabled = _bool(github_data, "enabled", False, "github")
    issue_mode = github_data.get("issue_mode", "off")
    if not isinstance(issue_mode, str) or issue_mode not in {"off", "link-existing", "create"}:
        raise ConfigError("github.issue_mode must be 'off', 'link-existing', or 'create'")
    pull_request_mode = github_data.get("pull_request_mode", "off")
    if not isinstance(pull_request_mode, str) or pull_request_mode not in {"off", "create"}:
        raise ConfigError("github.pull_request_mode must be 'off' or 'create'")
    issue_number = github_data.get("issue_number")
    if issue_number is not None and (
        isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0
    ):
        raise ConfigError("github.issue_number must be a positive integer or null")
    if github_enabled and issue_mode == "link-existing" and issue_number is None:
        raise ConfigError("github.issue_number is required for github.issue_mode = 'link-existing'")
    github = GitHubConfig(
        enabled=github_enabled,
        issue_mode=issue_mode,
        pull_request_mode=pull_request_mode,
        issue_number=issue_number,
        api_key_env=_optional_env_name(github_data, "api_key_env", None, "github"),
    )

    publish_data = _table(expanded, "publish")
    publish_remote = (
        _required_string(publish_data, "remote", "publish")
        if "remote" in publish_data else "origin"
    )
    if (
        "\x00" in publish_remote
        or any(char.isspace() for char in publish_remote)
        or publish_remote.startswith("-")
    ):
        raise ConfigError("publish.remote must be a valid remote name")
    publish_mode = publish_data.get("mode", PublishMode.RUN_BRANCH.value)
    if not isinstance(publish_mode, str) or publish_mode not in {
        item.value for item in PublishMode
    }:
        raise ConfigError("publish.mode must be 'run-branch' or 'fast-forward-base'")
    publish = PublishConfig(
        enabled=_bool(publish_data, "enabled", False, "publish"),
        remote=publish_remote,
        mode=publish_mode,
    )
    if (
        publish.enabled
        and publish.mode == PublishMode.RUN_BRANCH.value
        and publish.remote != repository.remote
    ):
        # Run-branch publication is the reviewed candidate already pushed to
        # repository.remote; it never pushes anything to another remote.
        raise ConfigError(
            "publish.remote must equal repository.remote when publish.mode = 'run-branch'"
        )
    if (
        github.enabled
        and github.pull_request_mode == "create"
        and (
            not publish.enabled
            or publish.mode != PublishMode.RUN_BRANCH.value
        )
    ):
        raise ConfigError(
            "github.pull_request_mode = 'create' requires publish.enabled = true "
            "and publish.mode = 'run-branch'"
        )

    unsupported_sections = sorted({"planner", "reviewer", "agent"}.intersection(expanded))
    if unsupported_sections:
        raise ConfigError(
            f"unsupported configuration section [{unsupported_sections[0]}]; "
            "configure model_profiles and ui defaults"
        )

    model_profiles = _model_profiles(expanded)
    routing = _routing(expanded, model_profiles)
    codex_providers = _codex_providers(expanded)
    for profile in model_profiles.values():
        if profile.driver is ProfileDriver.CODEX and profile.provider == "deepseek":
            provider = codex_providers.get(profile.provider)
            if provider is None:
                raise ConfigError(
                    f"Codex profile {profile.id!r} references unknown provider {profile.provider!r}"
                )
    has_explicit_codex = any(
        profile.driver is ProfileDriver.CODEX for profile in model_profiles.values()
    )
    has_explicit_claude = any(
        profile.driver is ProfileDriver.CLAUDE_CODE for profile in model_profiles.values()
    )
    ui_data = _table(expanded, "ui")
    planner_default = _check_default(
        model_profiles, ui_data.get("default_planner_profile"), ExecutionRole.PLANNER
    )
    # Implementer selection is class-based and comes from [routing].  The
    # former ui.default_implementer_profile is intentionally not consulted.
    implementer_default = None
    reviewer_default = _check_default(
        model_profiles, ui_data.get("default_reviewer_profile"), ExecutionRole.REVIEWER
    )
    reviser_default = ui_data.get("default_reviser_profile")
    if reviser_default is not None:
        reviser_default = _check_default(model_profiles, reviser_default, ExecutionRole.REVISER)
    repair_default = ui_data.get("default_repair_profile")
    if repair_default is not None:
        repair_default = _check_default(model_profiles, repair_default, ExecutionRole.REPAIR)
    # The default planner and reviewer are text endpoints; every selected
    # profile is validated again against its role when a run is prepared.
    _profile_endpoint(model_profiles[planner_default])
    _profile_endpoint(model_profiles[reviewer_default])

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
        default_planner_profile=planner_default,
        default_implementer_profile=implementer_default,
        default_reviewer_profile=reviewer_default,
        default_reviser_profile=reviser_default,
        default_repair_profile=repair_default,
        enable_profile_recommendation=_bool(
            ui_data, "enable_profile_recommendation", True, "ui"
        ),
    )

    _validate_revision(revision, planning, ui, model_profiles)

    if "checks" in expanded:
        raise ConfigError("unsupported configuration key 'checks'; use [[check_catalog]]")
    configured_catalog = _checks(expanded.get("check_catalog", []), catalogue=True)
    default_check_ids = _string_array(
        expanded, "default_check_ids",
        tuple(
            check.id for check in configured_catalog if check.required
        ),
        "root",
    )
    catalogue_ids = {check.id for check in configured_catalog}
    unknown_defaults = [item for item in default_check_ids if item not in catalogue_ids]
    if unknown_defaults:
        raise ConfigError(
            "default_check_ids contains unknown trusted check ID(s): "
            + ", ".join(unknown_defaults)
        )
    if len(set(default_check_ids)) != len(default_check_ids):
        raise ConfigError("default_check_ids must not contain duplicates")
    codex_runtime = _codex_runtime(
        expanded, config_dir, required=has_explicit_codex
    )
    claude_runtime = _claude_runtime(
        expanded, config_dir, required=has_explicit_claude
    )
    for label, root in (
        ("repo", repo),
        ("runs_root", runs_root),
        ("worktrees_root", worktrees_root),
    ):
        try:
            codex_runtime.home.relative_to(root)
        except ValueError:
            pass
        else:
            raise ConfigError(f"codex_runtime.home must not be inside {label}")
        try:
            claude_runtime.home.relative_to(root)
        except ValueError:
            pass
        else:
            raise ConfigError(f"claude_runtime.home must not be inside {label}")
    try:
        claude_runtime.home.relative_to(codex_runtime.home)
    except ValueError:
        try:
            codex_runtime.home.relative_to(claude_runtime.home)
        except ValueError:
            pass
        else:
            raise ConfigError("claude_runtime.home must not overlap managed CODEX_HOME")
    else:
        raise ConfigError("claude_runtime.home must not overlap managed CODEX_HOME")
    workspace_setup = _workspace_setup(expanded.get("workspace_setup", []))
    allow_no_required_checks = _bool(
        expanded, "allow_no_required_checks", False, "root"
    )
    if not allow_no_required_checks and not default_check_ids:
        raise ConfigError(
            "at least one required check is configured; set allow_no_required_checks = true for docs-only projects"
        )

    return HarnessConfig(
        repo=repo,
        base_ref=base_ref,
        runs_root=runs_root,
        worktrees_root=worktrees_root,
        require_clean_base=require_clean_base,
        context=context,
        check_catalog=configured_catalog,
        default_check_ids=tuple(default_check_ids),
        max_diff_bytes=max_diff_bytes,
        allow_no_required_checks=allow_no_required_checks,
        approval=approval,
        ui=ui,
        model_profiles=model_profiles,
        routing=routing,
        codex_providers=codex_providers,
        environment=environment,
        runtime_environment=runtime_environment,
        codex_runtime=codex_runtime,
        claude_runtime=claude_runtime,
        workspace_setup=workspace_setup,
        planning=planning,
        revision=revision,
        prompt_budget=prompt_budget,
        repository=repository,
        github=github,
        publish=publish,
        repository_section_explicit="repository" in expanded,
    )
