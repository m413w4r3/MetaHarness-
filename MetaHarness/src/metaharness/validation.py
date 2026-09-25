"""Execution of the deterministic checks configured for a run.

The important boundary in this module is that the command line comes only
from :attr:`HarnessConfig.checks`.  No text produced by another agent is ever
interpreted as a command.
"""

from __future__ import annotations

import re
import shutil
import json
import tempfile
import time
import dataclasses
import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .approval import ApprovalError, read_check_authority
from .attempt_transaction import (
    AttemptViolation,
    SideEffect,
    contain_trusted_process,
    observe_side_effects,
    restore_exact,
)
from .gitops import (
    CandidateState,
    GitError,
    snapshot_candidate_state,
)
from .models import CheckConfig, HarnessConfig
from .procutil import read_capped, run_bounded
from .redaction import redact_file
from .redaction import redact
from .result import ResultArtifactError, atomic_write_text


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
    """Return the run's frozen check configuration.

    The current TOML is used only to confirm that the requested IDs remain in
    the trusted catalogue.  The command-bearing values (argv, cwd, timeout,
    preflight argv) always come from the durable authority artifact: once a
    run owns a ``check_authority.json`` there is no fallback to today's
    configuration, even for an ID the initial selection did not include.

    *expected_sha256* is the already durable hash the approval bound.  It is
    verified on every live use, not only on resume.
    """

    directory = Path(run_dir).expanduser().resolve()
    while True:
        if (directory / "check_authority.json").is_file():
            break
        parent = directory.parent
        if parent == directory:
            raise ValidationError("the run has no check authority")
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
        raise ValidationError("the run has no check authority")
    authority_ids, catalogue = authority
    selected_ids = authority_ids
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
    frozen = dataclasses.replace(
        config,
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
    failure_kind: str = "unknown"
    mutation_recovered: bool = False
    mutated_tree_sha: str | None = None
    mutated_paths: tuple[str, ...] = ()
    infrastructure_retries: int = 0


def bounded_tail(value: str, max_bytes: int = DEFAULT_TAIL_BYTES) -> str:
    """Return a UTF-8 tail whose encoded size is at most *max_bytes*."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return encoded[-max_bytes:].decode("utf-8", errors="ignore") if max_bytes else ""


def _snapshot(worktree: Path) -> CandidateState:
    try:
        return snapshot_candidate_state(worktree)
    except GitError as exc:
        raise ValidationError(f"TREE_MISMATCH: could not snapshot candidate state: {exc}") from exc


def _observe(worktree: Path, before: CandidateState) -> SideEffect | None:
    try:
        return observe_side_effects(worktree, before)
    except GitError as exc:
        raise ValidationError(f"TREE_MISMATCH: could not snapshot candidate state: {exc}") from exc


def _archive_mutation(
    directory: Path | None,
    *,
    check_id: str,
    attempt: int,
    before: CandidateState,
    after: CandidateState,
    changed_paths: tuple[str, ...],
    secrets: tuple[str, ...],
    rollback: str,
) -> None:
    """Durably retain mutation identity and paths without storing file contents."""

    if directory is None:
        return
    path = directory / "mutations.jsonl"
    try:
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        entry = {
            "check_id": redact(check_id, secrets),
            "attempt": attempt,
            "candidate_tree_before": before.candidate_tree,
            "candidate_tree_after": after.candidate_tree,
            "index_tree_before": before.index_tree,
            "index_tree_after": after.index_tree,
            "mutated_paths": [redact(item, secrets) for item in changed_paths[:100]],
            "rollback": rollback,
        }
        atomic_write_text(path, previous + json.dumps(entry, sort_keys=True) + "\n")
    except (OSError, ResultArtifactError) as exc:
        raise ValidationError(f"DURABLE_ARTIFACT_CORRUPTED: could not archive check mutation: {type(exc).__name__}") from None


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


def _gate_infrastructure_failure(stdout: str, stderr: str) -> bool:
    """Recognize explicit Docker/PostgreSQL launch outages from gate output.

    A pytest report naming failed tests is authoritative product evidence even
    if one log line also mentions a service. Only a clear environment failure
    with no reported pytest failures is routed to infrastructure recovery.
    """

    output = f"{stdout}\n{stderr}".casefold()
    has_pytest_failures = bool(
        re.search(r"(?m)^\s*FAILED\s+\S+::", output)
        or re.search(r"\b\d+\s+failed\b", output)
    )
    if has_pytest_failures:
        return False
    markers = (
        "docker socket access denied",
        "permission denied while trying to connect to the docker daemon socket",
        "cannot connect to the docker daemon",
        "docker daemon is not running",
        "error during connect: this error may indicate that the docker daemon is not running",
        "could not translate host name",
        "could not connect to server: connection refused",
        "postgresql server is unavailable",
        "postgres server is unavailable",
    )
    return any(marker in output for marker in markers)


def run_checks(
    worktree: str | Path,
    config: HarnessConfig,
    *,
    required_check_ids: tuple[str, ...] | list[str] | None = None,
    logs_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    secrets: tuple[str, ...] = (),
    retry_infrastructure: Callable[[str, str], bool] | None = None,
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
            canonical_stdout = log_root / f"{stem}.stdout.log"
            canonical_stderr = log_root / f"{stem}.stderr.log"
            started = time.monotonic()
            attempt_number = 0
            infra_retries = 0
            mutation_retry_used = False
            prior_mutation: tuple[str, str, tuple[str, ...], tuple[str, ...]] | None = None
            any_mutation = False
            mutation_paths: set[str] = set()
            mutated_tree_sha: str | None = None
            while True:
                attempt_number += 1
                stdout_path = (
                    canonical_stdout if attempt_number == 1
                    else log_root / f"{stem}.attempt-{attempt_number}.stdout.log"
                )
                stderr_path = (
                    canonical_stderr if attempt_number == 1
                    else log_root / f"{stem}.attempt-{attempt_number}.stderr.log"
                )
                before = _snapshot(root)
                exit_code = -1
                timed_out = False
                process_error: OSError | ValueError | None = None
                attempt_started = time.monotonic()
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
                        process_error = exc
                        stderr.write(
                            f"could not start check: {type(exc).__name__}\n".encode("utf-8")
                        )
                    if timed_out:
                        exit_code = 124
                        stderr.write(
                            f"\ncheck timed out after {check.timeout_seconds}s\n".encode("utf-8")
                        )
                duration = time.monotonic() - attempt_started
                effect = _observe(root, before)
                redact_file(stdout_path, secrets)
                redact_file(stderr_path, secrets)

                if effect is not None:
                    after, changed_paths = effect.after, list(effect.changed_paths)

                    archive = functools.partial(
                        _archive_mutation, output_dir,
                        check_id=check.id, attempt=attempt_number,
                        before=before, after=after, changed_paths=changed_paths,
                        secrets=secrets,
                    )

                    if not effect.ownership_preserved:
                        archive(rollback="ownership_changed")
                        raise ValidationError(
                            "AGENT_GIT_VIOLATION: check changed Git ownership; "
                            f"mutated_tree={after.candidate_tree}; "
                            f"mutated_paths={','.join(changed_paths[:20])}"
                        )
                    mutation_signature = effect.signature
                    any_mutation = True
                    mutation_paths.update(changed_paths)
                    mutated_tree_sha = after.candidate_tree
                    try:
                        restore_exact(root, before, label="check mutation")
                    except AttemptViolation as violation:
                        if violation.code == "ROLLBACK_FAILED":
                            archive(rollback="failed")
                            raise ValidationError(
                                "ROLLBACK_FAILED: check mutation could not be restored; "
                                f"mutated_tree={after.candidate_tree}; "
                                f"mutated_paths={','.join(changed_paths[:20])}; "
                                f"error={violation.detail.rsplit(' ', 1)[-1]}"
                            ) from None
                        archive(rollback="mismatch")
                        raise ValidationError(
                            "ROLLBACK_TREE_MISMATCH: check rollback did not restore exact state; "
                            f"mutated_tree={after.candidate_tree}"
                        ) from None
                    archive(rollback="verified")
                    if not mutation_retry_used:
                        mutation_retry_used = True
                        prior_mutation = mutation_signature
                        continue
                    failure_kind = (
                        "side_effect_repeated"
                        if mutation_signature == prior_mutation else "side_effect_unstable"
                    )
                    final_timed_out = timed_out
                    final_exit_code = 124 if timed_out else exit_code
                    break

                if timed_out:
                    failure_kind = "timeout"
                elif process_error is not None:
                    failure_kind = (
                        "missing_executable"
                        if isinstance(process_error, FileNotFoundError)
                        else "process_start_failed"
                    )
                elif exit_code < 0:
                    failure_kind = "signal_terminated"
                elif exit_code == 0:
                    failure_kind = "passed"
                elif _gate_infrastructure_failure(
                    read_capped(stdout_path, _IN_MEMORY_LOG_BYTES)[0],
                    read_capped(stderr_path, _IN_MEMORY_LOG_BYTES)[0],
                ):
                    failure_kind = "infrastructure_unavailable"
                else:
                    failure_kind = "nonzero_exit"
                final_exit_code = 124 if timed_out else exit_code
                final_timed_out = timed_out
                if failure_kind in {
                    "timeout", "missing_executable", "process_start_failed", "signal_terminated",
                    "infrastructure_unavailable",
                } and retry_infrastructure is not None and retry_infrastructure(check.id, failure_kind):
                    infra_retries += 1
                    continue
                break

            # The canonical logs always refer to the final trusted invocation;
            # earlier invocations stay archived beside them for diagnosis.
            if attempt_number > 1 and output_dir is not None:
                if canonical_stdout.exists():
                    shutil.move(str(canonical_stdout), str(log_root / f"{stem}.attempt-1.stdout.log"))
                if canonical_stderr.exists():
                    shutil.move(str(canonical_stderr), str(log_root / f"{stem}.attempt-1.stderr.log"))
                shutil.copyfile(stdout_path, canonical_stdout)
                shutil.copyfile(stderr_path, canonical_stderr)
            stdout_text, _ = read_capped(stdout_path, _IN_MEMORY_LOG_BYTES)
            stderr_text, _ = read_capped(stderr_path, _IN_MEMORY_LOG_BYTES)
            results.append(CheckResult(
                name=check.name,
                argv=tuple(check.argv),
                cwd=str(cwd),
                exit_code=final_exit_code,
                timed_out=final_timed_out,
                duration_seconds=time.monotonic() - started,
                stdout_log=stdout_text,
                stderr_log=stderr_text,
                stdout_tail=bounded_tail(stdout_text, tail_bytes),
                stderr_tail=bounded_tail(stderr_text, tail_bytes),
                workspace_mutated=any_mutation,
                failure_kind=failure_kind,
                mutation_recovered=any_mutation,
                mutated_tree_sha=mutated_tree_sha,
                mutated_paths=tuple(sorted(redact(path, secrets) for path in mutation_paths)),
                infrastructure_retries=infra_retries,
            ))

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
        mutation_retry_used = False
        first_mutation: tuple[str, str, tuple[str, ...], tuple[str, ...]] | None = None
        while True:
            before = _snapshot(root)
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
            try:
                effect = contain_trusted_process(root, before, label="preflight mutation")
            except AttemptViolation as violation:
                raise ValidationError(f"{violation.code}: {violation.detail}") from None
            except GitError as exc:
                raise ValidationError(f"TREE_MISMATCH: could not snapshot candidate state: {exc}") from exc
            if effect is not None:
                signature = effect.signature
                if not mutation_retry_used:
                    mutation_retry_used = True
                    first_mutation = signature
                    continue
                code = (
                    "CHECK_SIDE_EFFECT_REPEATED"
                    if signature == first_mutation else "CHECK_SIDE_EFFECT_UNSTABLE"
                )
                failures.append(f"{code}:{check.id}")
                break
            if timed_out or exit_code != 0:
                failures.append(f"CHECK_PREFLIGHT_FAILED:{check.id}")
            break
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
