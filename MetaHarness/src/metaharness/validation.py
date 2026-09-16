"""Execution of the deterministic checks configured for a run.

The important boundary in this module is that the command line comes only
from :attr:`HarnessConfig.checks`.  No text produced by another agent is ever
interpreted as a command.
"""

from __future__ import annotations

import re
import tempfile
import time
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .approval import ApprovalError, read_check_authority
from .gitops import GitError, candidate_tree_sha, current_head, symbolic_head
from .models import CheckConfig, HarnessConfig
from .procutil import read_capped, run_bounded
from .redaction import redact_file


DEFAULT_TAIL_BYTES = 4_096
# Full logs always stay on disk; only this much is kept in memory.
_IN_MEMORY_LOG_BYTES = 8 * 1024 * 1024
_CHECK_GRACE_SECONDS = 5.0


class ValidationError(RuntimeError):
    """A configured check cannot be run safely."""


def config_with_check_authority(
    config: HarnessConfig, run_dir: str | Path, *,
    requested_check_ids: tuple[str, ...] | list[str] | None = None,
    expected_sha256: str | None = None,
) -> tuple[HarnessConfig, tuple[str, ...] | None]:
    """Return the run's frozen check config, or the legacy config.

    The current TOML is used only to confirm that the requested IDs remain in
    the trusted catalogue.  The command-bearing values (argv, cwd, timeout,
    preflight argv) always come from the durable authority artifact: once a
    run owns a ``check_authority.json`` there is no fallback to today's
    configuration, even for an ID the initial C01 selection did not include.

    *expected_sha256* is the already durable hash the approval bound.  It is
    verified on every live use, not only on resume.
    """

    directory = Path(run_dir).expanduser().resolve()
    while True:
        if (directory / "check_authority.json").is_file():
            break
        parent = directory.parent
        if parent == directory:
            return config, requested_check_ids
        directory = parent
    try:
        authority = read_check_authority(
            directory,
            expected_sha256=expected_sha256,
            trusted_check_ids=tuple(check.id for check in config.trusted_checks()),
        )
    except ApprovalError as exc:
        raise ValidationError(str(exc)) from exc
    if authority is None:
        return config, requested_check_ids
    authority_ids, catalogue = authority
    selected_ids = authority_ids
    checks = catalogue
    if requested_check_ids is not None:
        selected_ids = tuple(requested_check_ids)
        if len(set(selected_ids)) != len(selected_ids):
            raise ValidationError("required check IDs must be unique")
        frozen_by_id = {check.id: check for check in catalogue}
        # The installation may still veto an ID, but it can never supply one.
        trusted_ids = {check.id for check in config.trusted_checks()}
        resolved: list[CheckConfig] = []
        for check_id in selected_ids:
            frozen_check = frozen_by_id.get(check_id)
            if frozen_check is None:
                raise ValidationError(
                    "requested check is absent from the run's approved check authority: "
                    + check_id
                )
            if check_id not in trusted_ids:
                raise ValidationError("unknown trusted check ID(s): " + check_id)
            resolved.append(frozen_check)
        checks = tuple(resolved)
    frozen = dataclasses.replace(
        config,
        checks=checks,
        check_catalog=catalogue,
        default_check_ids=selected_ids,
    )
    return frozen, selected_ids


@dataclass(frozen=True)
class CheckResult:
    name: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    stdout_log: str
    stderr_log: str
    stdout_tail: str
    stderr_tail: str
    workspace_mutated: bool


@dataclass(frozen=True)
class _GitSnapshot:
    """The candidate that a subsequent ``git add -A`` would submit."""

    head: str
    head_ref: str | None
    candidate_tree: str


def bounded_tail(value: str, max_bytes: int = DEFAULT_TAIL_BYTES) -> str:
    """Return a UTF-8 tail whose encoded size is at most *max_bytes*."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return encoded[-max_bytes:].decode("utf-8", errors="ignore") if max_bytes else ""


def _git_snapshot(worktree: Path) -> _GitSnapshot:
    try:
        return _GitSnapshot(
            head=current_head(worktree),
            head_ref=symbolic_head(worktree),
            candidate_tree=candidate_tree_sha(worktree),
        )
    except GitError as exc:
        raise ValidationError(f"could not snapshot the candidate worktree: {exc}") from exc


def resolve_check_cwd(worktree: str | Path, check: CheckConfig) -> Path:
    """Resolve and validate a check's cwd inside the worktree.

    ``Path.resolve`` is deliberately used before the containment test so a
    symlink cannot make a seemingly relative cwd escape the worktree.
    """

    root = Path(worktree).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"worktree is not a directory: {root}")
    if "\x00" in check.cwd:
        raise ValidationError(f"check cwd contains NUL: {check.name}")
    if Path(check.cwd).is_absolute():
        raise ValidationError(
            f"check cwd must be relative to the worktree: {check.name!r} -> {check.cwd!r}"
        )
    try:
        candidate = (root / check.cwd).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValidationError(f"invalid cwd for check {check.name!r}: {exc}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValidationError(
            f"check cwd escapes worktree: {check.name!r} -> {check.cwd!r}"
        ) from exc
    if not candidate.is_dir():
        raise ValidationError(f"check cwd is not a directory: {check.cwd!r}")
    return candidate


def _safe_log_stem(name: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip(".") or "check"
    candidate = stem
    index = 2
    while candidate in used:
        candidate = f"{stem}-{index}"
        index += 1
    used.add(candidate)
    return candidate


def run_checks(
    worktree: str | Path,
    config: HarnessConfig,
    *,
    required_check_ids: tuple[str, ...] | list[str] | None = None,
    logs_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    secrets: tuple[str, ...] = (),
) -> tuple[CheckResult, ...]:
    """Run every configured check, continuing after failures and timeouts.

    Each check runs in its own process group with stdin closed and outputs
    written to log files; its whole group is terminated at the deadline and
    after it exits.  The candidate tree is snapshotted before and after each
    check to detect any mutation of the submitted code.
    """

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    root = Path(worktree).expanduser().resolve()
    output_dir = Path(logs_dir).expanduser().resolve() if logs_dir is not None else None
    if output_dir is not None:
        try:
            output_dir.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValidationError("logs_dir must be outside the worktree")
    try:
        selected = config.select_checks(required_check_ids)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    # Every cwd is validated before any command runs.
    cwds = [resolve_check_cwd(root, check) for check in selected]
    used_log_stems: set[str] = set()
    results: list[CheckResult] = []

    with tempfile.TemporaryDirectory(prefix="metaharness-checks-") as scratch:
        log_root = output_dir if output_dir is not None else Path(scratch)
        log_root.mkdir(parents=True, exist_ok=True)
        for check, cwd in zip(selected, cwds):
            stem = _safe_log_stem(check.name, used_log_stems)
            stdout_path = log_root / f"{stem}.stdout.log"
            stderr_path = log_root / f"{stem}.stderr.log"
            before = _git_snapshot(root)
            started = time.monotonic()
            exit_code = -1
            timed_out = False
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    exit_code, timed_out = run_bounded(
                        check.argv,
                        cwd=cwd,
                        timeout_seconds=check.timeout_seconds,
                        stdout=stdout,
                        stderr=stderr,
                        grace_seconds=_CHECK_GRACE_SECONDS,
                    )
                except (OSError, ValueError) as exc:
                    stderr.write(f"could not start check: {exc}\n".encode("utf-8", "replace"))
                if timed_out:
                    exit_code = 124
                    stderr.write(
                        f"\ncheck timed out after {check.timeout_seconds}s\n".encode("utf-8")
                    )
            duration = time.monotonic() - started
            after = _git_snapshot(root)
            redact_file(stdout_path, secrets)
            redact_file(stderr_path, secrets)
            stdout_text, _ = read_capped(stdout_path, _IN_MEMORY_LOG_BYTES)
            stderr_text, _ = read_capped(stderr_path, _IN_MEMORY_LOG_BYTES)
            results.append(
                CheckResult(
                    name=check.name,
                    argv=tuple(check.argv),
                    cwd=str(cwd),
                    exit_code=exit_code,
                    timed_out=timed_out,
                    duration_seconds=duration,
                    stdout_log=stdout_text,
                    stderr_log=stderr_text,
                    stdout_tail=bounded_tail(stdout_text, tail_bytes),
                    stderr_tail=bounded_tail(stderr_text, tail_bytes),
                    workspace_mutated=before != after,
                )
            )

    return tuple(results)


def run_check_preflights(
    worktree: str | Path,
    config: HarnessConfig,
    required_check_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Run trusted preflight argv for the selected checks only.

    The returned values are stable gate reasons.  No planner/model text is
    interpreted as a command; both argv and cwd come from the trusted config.
    """

    root = Path(worktree).expanduser().resolve()
    try:
        selected = config.select_checks(required_check_ids)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    failures: list[str] = []
    for check in selected:
        if not check.preflight_argv:
            continue
        cwd = resolve_check_cwd(root, check)
        try:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                exit_code, timed_out = run_bounded(
                    check.preflight_argv,
                    cwd=cwd,
                    timeout_seconds=check.timeout_seconds,
                    stdout=stdout,
                    stderr=stderr,
                    grace_seconds=_CHECK_GRACE_SECONDS,
                )
        except (OSError, ValueError):
            exit_code, timed_out = -1, False
        if timed_out or exit_code != 0:
            failures.append(f"CHECK_PREFLIGHT_FAILED:{check.id}")
    return tuple(failures)


def check_result_json(result: CheckResult, *, log_stem: str | None = None) -> dict[str, Any]:
    """Serialize reviewer-safe check metadata; full logs remain in log files."""

    data = asdict(result)
    data.pop("stdout_log")
    data.pop("stderr_log")
    if log_stem is not None:
        data["stdout_log_path"] = f"checks/{log_stem}.stdout.log"
        data["stderr_log_path"] = f"checks/{log_stem}.stderr.log"
    return data


# Descriptive aliases make the boundary easy to find for callers.
execute_checks = run_checks
validate_checks = run_checks
validate = run_checks
