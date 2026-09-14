"""Managed, empty Claude Code runtime configuration."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import HarnessConfig


class ClaudeRuntimeError(RuntimeError):
    """The managed Claude Code runtime cannot be safely prepared."""

    code = "CLAUDE_RUNTIME_FAILURE"


_EMPTY_MCP = '{\n  "mcpServers": {}\n}\n'
MANAGED_SUBDIRECTORIES = ("home", "cache", "tmp")


def _outside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    return False


def prepare_claude_home(config: HarnessConfig) -> Path:
    """Create Claude's private home without importing any personal files."""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    home = Path(config.claude_runtime.home).expanduser().resolve()
    roots = (
        ("repo", config.repo),
        ("runs_root", config.runs_root),
        ("worktrees_root", config.worktrees_root),
        ("managed CODEX_HOME", config.codex_runtime.home),
    )
    for label, root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not _outside(home, root) or not _outside(root, home):
            raise ClaudeRuntimeError(
                f"managed Claude config home must be outside {label}"
            )
    try:
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        home.chmod(0o700)
        # Private HOME, cache and temporary directories: Claude never sees
        # the personal ones, and nothing is copied into them.
        for name in MANAGED_SUBDIRECTORIES:
            directory = home / name
            if directory.is_symlink():
                raise ClaudeRuntimeError(f"managed Claude {name} directory must not be a symlink")
            directory.mkdir(exist_ok=True, mode=0o700)
            if not directory.is_dir():
                raise ClaudeRuntimeError(f"managed Claude {name} path is not a directory")
            directory.chmod(0o700)
        mcp_path = home / "empty-mcp.json"
        if mcp_path.is_symlink():
            raise ClaudeRuntimeError("managed Claude MCP configuration must not be a symlink")
        if not mcp_path.exists():
            mcp_path.write_text(_EMPTY_MCP, encoding="utf-8")
        else:
            # Parse and compare the exact managed shape.  A personal or
            # inherited MCP file is never accepted or amended in place.
            raw = mcp_path.read_text(encoding="utf-8")
            if raw != _EMPTY_MCP or json.loads(raw) != {"mcpServers": {}}:
                raise ClaudeRuntimeError(
                    "managed Claude config home contains a non-empty MCP configuration"
                )
        if mcp_path.stat().st_mode & 0o170000 != 0o100000:
            raise ClaudeRuntimeError("managed Claude MCP configuration is not a regular file")
    except ClaudeRuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        raise ClaudeRuntimeError("could not prepare managed Claude config home") from None
    return home


__all__ = ["MANAGED_SUBDIRECTORIES", "ClaudeRuntimeError", "prepare_claude_home"]
