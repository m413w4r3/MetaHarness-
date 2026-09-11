"""Exécution bornée de l'agent Codex dans un worktree."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..gitops import GitError, current_head
from ..models import AgentConfig
from ..procutil import run_bounded
from .base import AgentError, AgentResult
from .events import extract_final, extract_usage, parse_event


class AgentCommittedError(AgentError):
    """The agent changed HEAD, which is forbidden for an implementation run."""

    code = "AGENT_COMMITTED"

    def __init__(self, expected: str, actual: str):
        super().__init__(f"{self.code}: expected HEAD {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


_DEFAULT_TAIL_BYTES = 16_384
# Longer JSONL lines are kept in the artifact but not parsed in memory.
_MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def build_agent_environment(
    config: AgentConfig,
    *,
    forbidden_names: Iterable[str] = (),
) -> dict[str, str]:
    """Build the explicit, minimal environment passed to Codex."""

    if not isinstance(config, AgentConfig):
        raise TypeError("config must be an AgentConfig")
    names = tuple(config.env_allowlist)
    if any(_ENV_NAME.fullmatch(name) is None for name in names):
        raise ValueError("agent environment allowlist contains an invalid name")
    forbidden = frozenset(name for name in forbidden_names if name)
    return {
        name: os.environ[name]
        for name in names
        if name not in forbidden and name in os.environ
    }


def _template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts" / "implementer.txt"


def build_implementer_prompt(plan: str, *, template: str | None = None) -> str:
    """Put the authoritative plan into the fixed implementation prompt."""

    if not isinstance(plan, str):
        raise TypeError("plan must be a string")
    if template is None:
        template = _template_path().read_text(encoding="utf-8")
    if not isinstance(template, str):
        raise TypeError("template must be a string")
    return template.replace("{{PLAN}}", plan)


def _tail(path: Path, limit: int) -> str:
    if limit <= 0 or not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def _scan_events(path: Path) -> tuple[dict[str, int], str | None]:
    """Extract usage and final message from a JSONL file with bounded memory."""

    usage: dict[str, int] = {}
    final: str | None = None
    if not path.exists():
        return usage, final
    with path.open("rb") as stream:
        skipping = False
        while True:
            chunk = stream.readline(_MAX_EVENT_LINE_BYTES)
            if not chunk:
                break
            complete = chunk.endswith(b"\n")
            if skipping or (not complete and len(chunk) >= _MAX_EVENT_LINE_BYTES):
                # Remainder (or head) of an oversized line: not parseable.
                skipping = not complete
                continue
            event = parse_event(chunk.decode("utf-8", errors="replace"))
            if event is None:
                continue
            found_usage = extract_usage(event)
            if found_usage is not None:
                usage = found_usage
            found_final = extract_final(event)
            if found_final is not None:
                final = found_final
    return usage, final


class CodexAgent:
    """Run one implementation plan with the Codex CLI."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        executable: str = "codex",
        interrupt_grace_seconds: float = 5.0,
        stderr_tail_bytes: int = _DEFAULT_TAIL_BYTES,
        optional_flags: bool | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.executable = executable
        self.interrupt_grace_seconds = interrupt_grace_seconds
        self.stderr_tail_bytes = stderr_tail_bytes
        self._optional_flags = optional_flags

    def build_argv(self, worktree: str | Path, final_path: str | Path) -> list[str]:
        """Build the non-interactive command, including the stdin marker."""

        worktree_path = Path(worktree).expanduser().resolve()
        final_file = Path(final_path).expanduser().resolve()
        argv = [
            self.executable,
            "exec",
            "--json",
            "--sandbox",
            self.config.sandbox,
            "--output-last-message",
            str(final_file),
            "-m",
            self.config.model,
            "-c",
            f'model_reasoning_effort="{self.config.effort}"',
            "-C",
            str(worktree_path),
            "-",
        ]
        # These flags are version-dependent in the Codex CLI.  A failed help
        # probe is treated as support, matching the safe behavior for wrappers
        # and older CLIs that do not expose useful help text.
        optional = self._supports_optional_flags()
        insertion = 3
        if optional:
            argv[insertion:insertion] = ["--color", "never", "--skip-git-repo-check"]
        return argv

    build_command = build_argv

    def _supports_optional_flags(self) -> bool:
        if self._optional_flags is not None:
            return self._optional_flags
        # A custom executable is normally a wrapper or a test double; probing
        # it can execute the real workload twice.  The actual ``codex`` binary
        # is the only command for which the compatibility probe is meaningful.
        if self.executable != "codex":
            self._optional_flags = True
            return True
        try:
            help_result = subprocess.run(
                [self.executable, "exec", "--help"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=20,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            self._optional_flags = True
            return True
        help_text = (help_result.stdout or "") + (help_result.stderr or "")
        self._optional_flags = help_result.returncode != 0 or not help_text.strip()
        if help_result.returncode == 0 and help_text.strip():
            self._optional_flags = "--color" in help_text and "--skip-git-repo-check" in help_text
        return self._optional_flags

    def run(
        self,
        plan: str,
        worktree: str | Path,
        artifacts_dir: str | Path,
        *,
        base_sha: str | None = None,
        env: dict[str, str] | None = None,
    ) -> AgentResult:
        """Execute *plan* and persist the five run artifacts."""

        if not isinstance(plan, str):
            raise TypeError("plan must be a string")
        worktree_path = Path(worktree).expanduser().resolve()
        directory = Path(artifacts_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        prompt = build_implementer_prompt(plan)
        prompt_path = directory / "agent.prompt.txt"
        events_path = directory / "agent.events.jsonl"
        stderr_path = directory / "agent.stderr.log"
        final_path = directory / "agent.final.md"
        result_path = directory / "agent.result.json"
        prompt_path.write_text(prompt, encoding="utf-8")

        expected_head = base_sha or self._head(worktree_path)
        if self._head(worktree_path) != expected_head:
            raise AgentError("worktree HEAD does not match the agent base SHA")

        argv = self.build_argv(worktree_path, final_path)
        agent_environment = (
            build_agent_environment(self.config) if env is None else dict(env)
        )
        exit_code, timed_out = self._execute(
            argv, prompt_path, worktree_path, events_path, stderr_path, agent_environment
        )
        usage, event_final = _scan_events(events_path)

        actual_head = self._head(worktree_path)
        if actual_head != expected_head:
            raise AgentCommittedError(expected_head, actual_head)

        if final_path.exists():
            final_message = final_path.read_text(encoding="utf-8", errors="replace")
        else:
            final_message = event_final or ""
            final_path.write_text(final_message, encoding="utf-8")

        result = AgentResult(
            exit_code=124 if timed_out else exit_code,
            timed_out=timed_out,
            final_message=final_message,
            usage=usage,
            stderr_tail=_tail(stderr_path, self.stderr_tail_bytes),
        )
        result_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    execute = run

    @staticmethod
    def _head(worktree: Path) -> str:
        try:
            return current_head(worktree)
        except GitError as exc:
            raise AgentError(f"cannot establish worktree HEAD: {exc}") from exc

    def _execute(
        self,
        argv: list[str],
        prompt_path: Path,
        worktree: Path,
        events_path: Path,
        stderr_path: Path,
        env: dict[str, str],
    ) -> tuple[int, bool]:
        """Run Codex with file-backed stdin/stdout/stderr and a hard deadline.

        Nothing here blocks on a pipe: the prompt is read from a file, events
        are written to a file, and the deadline is enforced by polling the
        process.  On timeout the process group receives SIGINT, then SIGKILL
        after the grace period; leftover background processes are always
        terminated once Codex exits.
        """

        events_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with prompt_path.open("rb") as stdin, events_path.open(
                "wb"
            ) as stdout, stderr_path.open("wb") as stderr:
                return run_bounded(
                    argv,
                    cwd=worktree,
                    timeout_seconds=self.config.timeout_seconds,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    interrupt_signal=signal.SIGINT,
                    grace_seconds=self.interrupt_grace_seconds,
                    env=env,
                )
        except (OSError, ValueError) as exc:
            raise AgentError(f"could not start Codex: {exc}") from exc


def run_codex(
    plan: str,
    worktree: str | Path,
    artifacts_dir: str | Path,
    *,
    config: AgentConfig | None = None,
    base_sha: str | None = None,
    env: dict[str, str] | None = None,
) -> AgentResult:
    """Functional convenience wrapper for one Codex execution."""

    return CodexAgent(config).run(plan, worktree, artifacts_dir, base_sha=base_sha, env=env)


__all__ = [
    "AgentCommittedError",
    "CodexAgent",
    "build_agent_environment",
    "build_implementer_prompt",
    "run_codex",
]
