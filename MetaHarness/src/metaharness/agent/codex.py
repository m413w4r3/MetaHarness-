"""Exécution bornée de l'agent Codex dans un worktree."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..gitops import GitError, current_head
from ..models import AgentConfig
from .base import AgentError, AgentResult
from .events import extract_final, extract_usage, parse_event


class AgentCommittedError(AgentError):
    """The agent changed HEAD, which is forbidden for an implementation run."""

    code = "AGENT_COMMITTED"

    def __init__(self, expected: str, actual: str):
        super().__init__(f"{self.code}: expected HEAD {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


_END = object()
_POLL_SECONDS = 0.2
_DEFAULT_TAIL_BYTES = 16_384


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


def _kill_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        # The process may have exited between poll() and killpg().
        if process.poll() is None:
            raise


def _read_stdout(stream: Any, output: queue.Queue[object]) -> None:
    try:
        for line in stream:
            output.put(line)
    except (OSError, ValueError):
        pass
    finally:
        output.put(_END)


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
        exit_code, timed_out, usage, event_final = self._execute(
            argv,
            prompt,
            worktree_path,
            events_path,
            stderr_path,
        )

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
        prompt: str,
        worktree: Path,
        events_path: Path,
        stderr_path: Path,
    ) -> tuple[int, bool, dict[str, int], str | None]:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        output: queue.Queue[object] = queue.Queue()
        usage: dict[str, int] = {}
        event_final: str | None = None
        timed_out = False
        process: subprocess.Popen[str] | None = None

        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    cwd=str(worktree),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except (OSError, ValueError) as exc:
                raise AgentError(f"could not start Codex: {exc}") from exc

            assert process.stdin is not None
            assert process.stdout is not None
            reader = threading.Thread(
                target=_read_stdout, args=(process.stdout, output), daemon=True
            )
            reader.start()
            try:
                try:
                    process.stdin.write(prompt)
                    process.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    # The exit code and stderr remain the authoritative failure.
                    try:
                        process.stdin.close()
                    except (OSError, ValueError):
                        pass

                with events_path.open("w", encoding="utf-8") as events_file:
                    eof = False
                    deadline = time.monotonic() + self.config.timeout_seconds
                    grace_deadline: float | None = None
                    while True:
                        now = time.monotonic()
                        if not timed_out and now >= deadline:
                            timed_out = True
                            grace_deadline = now + self.interrupt_grace_seconds
                            _kill_process_group(process, signal.SIGINT)

                        if (
                            timed_out
                            and grace_deadline is not None
                            and now >= grace_deadline
                        ):
                            _kill_process_group(process, signal.SIGKILL)
                            grace_deadline = None

                        wait = _POLL_SECONDS
                        if not timed_out:
                            wait = min(wait, max(0.0, deadline - now))
                        elif grace_deadline is not None:
                            wait = min(wait, max(0.0, grace_deadline - now))
                        try:
                            item = output.get(timeout=wait)
                        except queue.Empty:
                            item = None

                        if item is not None:
                            if item is _END:
                                eof = True
                            else:
                                line = str(item)
                                events_file.write(line)
                                events_file.flush()
                                event = parse_event(line)
                                if event is not None:
                                    found_usage = extract_usage(event)
                                    if found_usage is not None:
                                        usage = found_usage
                                    found_final = extract_final(event)
                                    if found_final is not None:
                                        event_final = found_final

                        if process.poll() is not None and output.empty() and (
                            eof or not timed_out
                        ):
                            if not eof:
                                # A child that inherited stdout can otherwise
                                # keep the reader blocked after the CLI exits.
                                try:
                                    process.stdout.close()
                                except (OSError, ValueError):
                                    pass
                            break

                    exit_code = process.wait()
            finally:
                if process.poll() is None:
                    _kill_process_group(process, signal.SIGKILL)
                reader.join(timeout=1.0)
                try:
                    process.stdout.close()
                except (OSError, ValueError):
                    pass

        return exit_code, timed_out, usage, event_final


def run_codex(
    plan: str,
    worktree: str | Path,
    artifacts_dir: str | Path,
    *,
    config: AgentConfig | None = None,
    base_sha: str | None = None,
) -> AgentResult:
    """Functional convenience wrapper for one Codex execution."""

    return CodexAgent(config).run(plan, worktree, artifacts_dir, base_sha=base_sha)


__all__ = [
    "AgentCommittedError",
    "CodexAgent",
    "build_implementer_prompt",
    "run_codex",
]
