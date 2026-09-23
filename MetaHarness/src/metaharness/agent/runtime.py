"""Managed runtime directory for the Codex CLI."""

from __future__ import annotations

import json
import tomllib
import os
import stat
import tempfile
from pathlib import Path

from ..models import HarnessConfig


class CodexRuntimeError(RuntimeError):
    """The managed Codex runtime cannot be safely prepared."""

    code = "CODEX_RUNTIME_FAILURE"


_MANAGED_CONFIG = (
    "# Managed by MetaHarness.\n"
    "# Dedicated to bounded mechanical implementation workers.\n"
    "\n"
    "approval_policy = \"never\"\n"
    "sandbox_mode = \"workspace-write\"\n"
    "web_search = \"disabled\"\n"
    "\n"
    "developer_instructions = \"\"\"\n"
    "You are the MetaHarness mechanical implementation worker.\n"
    "\n"
    "The approved step contract supplied on stdin is authoritative.\n"
    "MetaHarness owns architecture, repository discovery, decomposition, review,\n"
    "commits, publishing, and repair routing.\n"
    "\n"
    "Do not redesign the task, perform repository-wide discovery, use network\n"
    "research, broaden scope, commit, push, switch branches, create worktrees, or\n"
    "move HEAD.\n"
    "\n"
    "Read only the declared READ_SET, applicable project instructions loaded by\n"
    "Codex, and strictly necessary direct local imports needed to resolve named\n"
    "symbols from the contract.\n"
    "\n"
    "Modify only WRITE_SET, CREATE_SET and DELETE_SET paths.\n"
    "\n"
    "Before editing, verify that the named paths, anchors and structural\n"
    "preconditions needed by the contract exist. If a material structural\n"
    "contradiction makes the prescribed change non-mechanical, do not invent an\n"
    "alternative architecture.\n"
    "\n"
    "When that happens, make no edits and start the final response with exactly:\n"
    "\n"
    "META CONTRACT MISMATCH v1\n"
    "\n"
    "Then give a short concrete explanation.\n"
    "\n"
    "For normal successful work, give a concise implementation report. No strict\n"
    "machine protocol is required for successful completion.\n"
    "\"\"\"\n"
    "\n"
    "[sandbox_workspace_write]\n"
    "network_access = false\n"
    "\n"
    "[agents]\n"
    "enabled = false\n"
    "\n"
    "[features]\n"
    "multi_agent = false\n"
    "apps = false\n"
    "plugins = false\n"
    "remote_plugin = false\n"
    "plugin_sharing = false\n"
    "recommended_plugins = false\n"
    "tool_suggest = false\n"
    "skill_search = false\n"
    "skill_mcp_dependency_install = false\n"
    "enable_mcp_apps = false\n"
    "hooks = false\n"
    "worktrees = false\n"
    "memories = false\n"
    "memory_tool = false\n"
    "\n"
    "[features.multi_agent_v2]\n"
    "enabled = false\n"
    "\n"
    "[model_providers.deepseek]\n"
    "name = \"deepseek\"\n"
    "base_url = \"https://api.deepseek.com/\"\n"
    "wire_api = \"responses\"\n"
    "env_key = \"DEEPSEEK_API_KEY\"\n"
)

_MANAGED_CONFIG_SHAPE = tomllib.loads(_MANAGED_CONFIG)
_MANAGED_MODELS = {
    "models": [{
        "slug": "deepseek-flash",
        "display_name": "DeepSeek V4.1 Flash",
        "default_reasoning_level": "high",
        "supported_reasoning_levels": [
            {"effort": "low"}, {"effort": "high"}, {"effort": "max"}
        ],
    }]
}
_MANAGED_MODELS_TEXT = json.dumps(
    _MANAGED_MODELS, ensure_ascii=False, indent=2, sort_keys=True
) + "\n"


def _atomic_write_managed_config(path: Path, content: str) -> None:
    """Replace one managed config without ever following the destination."""

    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise CodexRuntimeError("could not write managed Codex config") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


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
        home.chmod(0o700)
        config_path = home / "config.toml"
        try:
            config_stat = config_path.lstat()
        except FileNotFoundError:
            config_stat = None
        if config_stat is not None:
            if stat.S_ISLNK(config_stat.st_mode):
                raise CodexRuntimeError("managed Codex config.toml must not be a symlink")
            if not stat.S_ISREG(config_stat.st_mode):
                raise CodexRuntimeError("managed Codex config.toml must be a regular file")
        if config_stat is None or config_path.read_bytes() != _MANAGED_CONFIG.encode("utf-8"):
            _atomic_write_managed_config(config_path, _MANAGED_CONFIG)
        with config_path.open("rb") as stream:
            parsed = tomllib.load(stream)
        if parsed != _MANAGED_CONFIG_SHAPE:
            raise CodexRuntimeError("managed Codex config.toml has an unexpected shape")
        models_path = home / "models.json"
        try:
            models_stat = models_path.lstat()
        except FileNotFoundError:
            models_stat = None
        if models_stat is not None and (
            stat.S_ISLNK(models_stat.st_mode) or not stat.S_ISREG(models_stat.st_mode)
        ):
            raise CodexRuntimeError("managed Codex models.json must be a regular file")
        if models_stat is None or models_path.read_text(encoding="utf-8") != _MANAGED_MODELS_TEXT:
            _atomic_write_managed_config(models_path, _MANAGED_MODELS_TEXT)
        if json.loads(models_path.read_text(encoding="utf-8")) != _MANAGED_MODELS:
            raise CodexRuntimeError("managed Codex models.json has an unexpected shape")
    except CodexRuntimeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError):
        raise CodexRuntimeError(
            "could not prepare managed CODEX_HOME"
        ) from None
    return home


__all__ = ["CodexRuntimeError", "prepare_codex_home"]
