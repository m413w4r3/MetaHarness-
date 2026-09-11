"""Allowlisted model profiles and driver-specific configuration adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import (
    AgentConfig,
    ExecutionRole,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ProfileDriver,
    SelectionMode,
)


class ProfileError(ValueError):
    pass


def _legacy_profiles(config: HarnessConfig) -> dict[str, ModelProfile]:
    return {
        "legacy-planner": ModelProfile(
            id="legacy-planner",
            display_name="Legacy Planner",
            roles=(ExecutionRole.PLANNER,),
            driver=ProfileDriver.OPENAI_CHAT,
            model=config.planner.model,
            selection_mode=SelectionMode.REQUEST,
            base_url=config.planner.base_url,
            endpoint_path=config.planner.endpoint_path,
            api_key_env=config.planner.api_key_env,
            timeout_seconds=config.planner.timeout_seconds,
            retries=config.planner.retries,
            extra_body=config.planner.extra_body,
        ),
        "legacy-implementer": ModelProfile(
            id="legacy-implementer",
            display_name="Legacy Implementer",
            roles=(ExecutionRole.IMPLEMENTER,),
            driver=ProfileDriver.CODEX,
            model=config.agent.model,
            selection_mode=SelectionMode.CLI,
            effort=config.agent.effort,
            sandbox=config.agent.sandbox,
            timeout_seconds=config.agent.timeout_seconds,
        ),
        "legacy-reviewer": ModelProfile(
            id="legacy-reviewer",
            display_name="Legacy Reviewer",
            roles=(ExecutionRole.REVIEWER,),
            driver=ProfileDriver.OPENAI_CHAT,
            model=config.reviewer.model,
            selection_mode=SelectionMode.REQUEST,
            base_url=config.reviewer.base_url,
            endpoint_path=config.reviewer.endpoint_path,
            api_key_env=config.reviewer.api_key_env,
            timeout_seconds=config.reviewer.timeout_seconds,
            retries=config.reviewer.retries,
            extra_body=config.reviewer.extra_body,
        ),
    }


def _profiles(config: HarnessConfig) -> dict[str, ModelProfile]:
    if config.model_profiles:
        return dict(config.model_profiles)
    return _legacy_profiles(config)


def profiles_for_config(config: HarnessConfig) -> dict[str, ModelProfile]:
    """Return the effective profiles for the UI and API."""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    return _profiles(config)


def profile_for_role(
    config: HarnessConfig,
    profile_id: str,
    role: ExecutionRole,
) -> ModelProfile:
    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    try:
        requested_role = ExecutionRole(role)
    except (TypeError, ValueError) as exc:
        raise ProfileError("execution role is invalid") from exc
    if not isinstance(profile_id, str) or not profile_id:
        raise ProfileError("profile id is invalid")
    profile = _profiles(config).get(profile_id)
    if profile is None:
        raise ProfileError(f"unknown profile: {profile_id}")
    if requested_role not in profile.roles:
        raise ProfileError(f"profile {profile_id!r} is incompatible with {requested_role.value}")
    return profile


def safe_profile_metadata(profile: ModelProfile) -> dict[str, Any]:
    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "roles": [role.value for role in profile.roles],
        "driver": profile.driver.value,
        "model_label": profile.model,
        "selection_mode": profile.selection_mode.value,
        "effort": profile.effort,
        "sandbox": profile.sandbox,
        "description": profile.description,
        "strengths": list(profile.strengths),
        "cost_tier": profile.cost_tier,
        "latency_tier": profile.latency_tier,
    }


def profile_execution_fingerprint(
    profile: ModelProfile,
    *,
    agent_env_allowlist: tuple[str, ...] = (),
    codex_home: Path | None = None,
) -> str:
    """SHA-256 of the profile fields that change execution.

    Advisory UI fields (display name, description, strengths, tiers) are
    excluded.  ``api_key_env`` is only the variable name, never its value.
    """

    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    if isinstance(agent_env_allowlist, (str, bytes)) or not all(
        isinstance(name, str) for name in agent_env_allowlist
    ):
        raise ProfileError("agent_env_allowlist must contain variable names")
    payload: dict[str, Any] = {
        "id": profile.id,
        "roles": [role.value for role in profile.roles],
        "driver": profile.driver.value,
        "model": profile.model,
        "selection_mode": profile.selection_mode.value,
        "timeout_seconds": profile.timeout_seconds,
    }
    if profile.driver is ProfileDriver.OPENAI_CHAT:
        payload.update(
            base_url=profile.base_url,
            endpoint_path=profile.endpoint_path,
            api_key_env=profile.api_key_env,
            retries=profile.retries,
            extra_body=dict(profile.extra_body),
        )
    elif profile.driver is ProfileDriver.CODEX:
        payload.update(
            effort=profile.effort,
            sandbox=profile.sandbox,
            agent_env_allowlist=list(agent_env_allowlist),
            codex_home=(
                str(Path(codex_home).expanduser().resolve())
                if codex_home is not None
                else None
            ),
        )
    else:  # pragma: no cover - ProfileDriver is closed
        raise ProfileError("profile driver is unknown")
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"profile {profile.id!r} is not canonically serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_llm_endpoint(profile: ModelProfile) -> LLMEndpointConfig:
    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    if profile.driver is not ProfileDriver.OPENAI_CHAT:
        raise ProfileError("profile driver is not openai-chat")
    if profile.base_url is None or profile.endpoint_path is None:
        raise ProfileError("openai-chat profile has no endpoint")
    return LLMEndpointConfig(
        base_url=profile.base_url,
        endpoint_path=profile.endpoint_path,
        model=profile.model,
        api_key_env=profile.api_key_env,
        timeout_seconds=profile.timeout_seconds,
        retries=profile.retries,
        extra_body=dict(profile.extra_body),
    )


def build_agent_config(profile: ModelProfile) -> AgentConfig:
    if not isinstance(profile, ModelProfile):
        raise TypeError("profile must be a ModelProfile")
    if profile.driver is not ProfileDriver.CODEX:
        raise ProfileError("profile driver is not codex")
    if profile.effort is None or profile.sandbox is None:
        raise ProfileError("codex profile has no effort or sandbox")
    return AgentConfig(
        model=profile.model,
        effort=profile.effort,
        sandbox=profile.sandbox,
        timeout_seconds=profile.timeout_seconds,
    )


__all__ = [
    "ProfileError",
    "profile_for_role",
    "profile_execution_fingerprint",
    "safe_profile_metadata",
    "build_llm_endpoint",
    "build_agent_config",
]
