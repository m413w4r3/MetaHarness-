"""Managed runtime directory for the Codex CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path

from ..models import HarnessConfig


class CodexRuntimeError(RuntimeError):
    """The managed Codex runtime cannot be safely prepared."""

    code = "CODEX_RUNTIME_FAILURE"


_MANAGED_CONFIG = (
    "# Managed by MetaHarness.\n"
    "# Intentionally contains no MCP server configuration.\n"
)


def prepare_codex_home(config: HarnessConfig) -> Path:
    """Create and validate the isolated Codex home without copying credentials."""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    home = Path(config.codex_runtime.home).expanduser().resolve()
    for root in (config.repo, config.runs_root, config.worktrees_root):
        try:
            home.relative_to(Path(root).expanduser().resolve())
        except ValueError:
            continue
        raise CodexRuntimeError(
            "managed CODEX_HOME must be outside repo, runs_root and worktrees_root"
        )
    try:
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        config_path = home / "config.toml"
        if not config_path.exists():
            config_path.write_text(_MANAGED_CONFIG, encoding="utf-8")
        else:
            with config_path.open("rb") as stream:
                parsed = tomllib.load(stream)
            if parsed.get("mcp_servers"):
                raise CodexRuntimeError(
                    "MetaHarness CODEX_HOME must not configure MCP servers"
                )
    except CodexRuntimeError:
        raise
    except (OSError, tomllib.TOMLDecodeError):
        raise CodexRuntimeError(
            "could not prepare managed CODEX_HOME"
        ) from None
    return home


__all__ = ["CodexRuntimeError", "prepare_codex_home"]
