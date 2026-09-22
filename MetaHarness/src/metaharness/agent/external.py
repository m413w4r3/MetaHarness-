"""Safe execution of a trusted external worker process.

This module intentionally knows nothing about a provider protocol.  It only
owns the process boundary that MetaHarness can verify for every driver:
trusted argv, a MetaHarness-owned worktree and deadline, file-backed stdin and
diagnostics, exit status, and before/after candidate trees.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..gitops import GitError, candidate_tree_sha
from ..procutil import read_capped, run_bounded
from ..redaction import config_secret_values, redact_file
from .base import (
    AGENT_RUNTIME_FAILED,
    AGENT_START_FAILED,
    AGENT_TIMEOUT,
    AgentError,
    AgentExecutorCapabilities,
    AgentRunRequest,
    AgentRunResult,
)


@dataclass(frozen=True)
class ExternalAgentConfig:
    """Trusted process settings supplied by configuration.

    ``argv`` is never assembled from a prompt or model metadata.  The
    optional diagnostic limits affect only what is returned in the normalized
    result; the raw files remain local artifacts for operators.
    """

    argv: tuple[str, ...]
    timeout_seconds: int
    stdout_max_bytes: int = 16 * 1024
    stderr_max_bytes: int = 16 * 1024
    driver_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple) or not self.argv or any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in self.argv
        ):
            raise ValueError("external argv must be a non-empty tuple of strings")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("external timeout_seconds must be greater than zero")
        for name in ("stdout_max_bytes", "stderr_max_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"external {name} must be greater than zero")
        if self.driver_version is not None and (
            not isinstance(self.driver_version, str) or not self.driver_version.strip()
        ):
            raise ValueError("external driver_version must be a non-empty string or null")


def _tree(path: Path) -> str:
    try:
        return candidate_tree_sha(path)
    except (GitError, OSError, ValueError):
        return ""


def _safe_text(path: Path, max_bytes: int) -> str:
    text = read_capped(path, max_bytes)[0]
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode("utf-8", errors="ignore")


class ExternalAgentExecutor:
    """Run one trusted external process through the backend-neutral contract."""

    capabilities = AgentExecutorCapabilities()

    def __init__(
        self,
        profile: Any,
        runtime_config: Any = None,
        *,
        config: ExternalAgentConfig | None = None,
    ) -> None:
        self.profile = profile
        self.runtime = runtime_config
        self.config = config or ExternalAgentConfig(
            argv=tuple(getattr(profile, "argv", ())),
            timeout_seconds=int(getattr(profile, "timeout_seconds", 0)),
            driver_version=getattr(profile, "driver_version", None),
        )
        driver = getattr(profile, "driver", "external")
        self.driver = getattr(driver, "value", driver)

    @property
    def driver_version(self) -> str | None:
        return self.config.driver_version

    def _environment(self) -> dict[str, str]:
        environment = getattr(self.runtime, "environment", None)
        if isinstance(environment, Mapping):
            return {
                str(name): value
                for name, value in environment.items()
                if isinstance(name, str) and isinstance(value, str)
            }
        return dict(os.environ)

    def _secrets(self) -> tuple[str, ...]:
        runtime_config = getattr(self.runtime, "config", None)
        if runtime_config is None:
            return ()
        try:
            return config_secret_values(runtime_config, self._environment())
        except (AttributeError, TypeError, ValueError):
            return ()

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        before = _tree(request.worktree)
        artifact_dir = Path(request.artifact_dir).expanduser().resolve()
        worktree = Path(request.worktree).expanduser().resolve()
        prompt_path = artifact_dir / "agent.prompt.txt"
        stdout_path = artifact_dir / "agent.stdout.log"
        stderr_path = artifact_dir / "agent.stderr.log"
        final_path = artifact_dir / "agent.final.md"

        try:
            if not isinstance(request.prompt, str):
                raise TypeError("prompt must be a string")
            if not worktree.is_dir():
                raise AgentError("external worker worktree is not a directory")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(request.prompt, encoding="utf-8")
            with (
                prompt_path.open("rb") as stdin,
                stdout_path.open("wb") as stdout,
                stderr_path.open("wb") as stderr,
            ):
                exit_code, timed_out = run_bounded(
                    self.config.argv,
                    cwd=worktree,
                    timeout_seconds=self.config.timeout_seconds,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    interrupt_signal=signal.SIGTERM,
                    env=self._environment(),
                )
        except (OSError, ValueError, TypeError, AgentError) as exc:
            return AgentRunResult(
                status="failed",
                exit_reason=AGENT_START_FAILED,
                tree_before=before,
                tree_after=_tree(worktree),
                usage=None,
                external_session_id=None,
                report_path=None,
                exit_code=None,
                driver=self.driver,
                driver_version=self.driver_version,
                backend_reason=type(exc).__name__,
                stderr_tail="",
            )

        secrets = self._secrets()
        for path in (stdout_path, stderr_path):
            redact_file(path, secrets)
        stdout_tail = _safe_text(stdout_path, self.config.stdout_max_bytes)
        stderr_tail = _safe_text(stderr_path, self.config.stderr_max_bytes)
        if not final_path.exists():
            final_path.write_text(stdout_tail, encoding="utf-8")
        after = _tree(worktree)
        if timed_out:
            status = "timed_out"
            reason = AGENT_TIMEOUT
        elif exit_code == 0:
            status = "completed"
            reason = None
        else:
            status = "failed"
            reason = AGENT_RUNTIME_FAILED
        return AgentRunResult(
            status=status,
            exit_reason=reason,
            tree_before=before,
            tree_after=after,
            usage=None,
            external_session_id=None,
            report_path=str(final_path),
            exit_code=124 if timed_out else exit_code,
            timed_out=timed_out,
            final_message=stdout_tail,
            stderr_tail=stderr_tail,
            driver=self.driver,
            driver_version=self.driver_version,
            raw_result=None,
        )


__all__ = ["ExternalAgentConfig", "ExternalAgentExecutor"]
