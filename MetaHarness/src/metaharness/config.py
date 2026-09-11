"""Chargement, expansion et validation de la configuration TOML."""

from __future__ import annotations

import math
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .llm.chat import PROTECTED_BODY_KEYS, LLMProtocolError, validate_endpoint
from .models import (
    AgentConfig,
    ApprovalConfig,
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
    UIConfig,
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


def _expand_string(value: str) -> str:
    """Expand ``${NAME}`` references in one non-recursive pass.

    A value taken from the environment is inserted literally: a ``${...}``
    sequence inside it is never expanded again.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"environment variable {name!r} is not set")
        return os.environ[name]

    return _ENV_VAR.sub(replace, value)


def _expand(value: Any) -> Any:
    """Recursively expand strings in TOML tables and arrays."""

    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
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

    expanded = _expand(raw)
    if not isinstance(expanded, dict):  # pragma: no cover - tomllib guarantee
        raise ConfigError("configuration root must be a table")
    config_dir = path.parent

    repo = _path(expanded.get("repo"), "repo", config_dir)
    base_ref = _required_string(expanded, "base_ref", "root")
    runs_root = _path(expanded.get("runs_root"), "runs_root", config_dir)
    worktrees_root = _path(expanded.get("worktrees_root"), "worktrees_root", config_dir)
    require_clean_base = _bool(expanded, "require_clean_base", True, "root")
    max_diff_bytes = _positive_int(expanded, "max_diff_bytes", 400_000, "root")

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
        raise ConfigError(f"unknown agent.sandbox: {sandbox!r}")
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

    planner = _endpoint(_table(expanded, "planner"), "planner")
    reviewer = _endpoint(_table(expanded, "reviewer"), "reviewer")

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
        )
    )

    checks = _checks(expanded.get("checks", []))
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
    )
