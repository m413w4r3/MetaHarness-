"""Immutable evidence collection and deterministic gate evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .gitops import (
    StagedChange,
    current_head,
    index_tree_sha,
    read_staged_blob,
    stage_all,
    staged_binary_files,
    staged_changed_blobs,
    staged_changed_files,
    staged_changed_files_from,
    staged_changes,
    staged_diff,
    staged_diff_from,
)
from .models import CheckConfig, HarnessConfig
from .redaction import contains_secret, redact
from .validation import (
    DEFAULT_TAIL_BYTES,
    CheckResult,
    check_result_json,
    run_checks,
)


DIFF_SIZE_FAILURE = "DETERMINISTIC_GATE_FAILED:diff_too_large"
DIFF_SECRET_FAILURE = "COMMIT_SECURITY_FAILURE:secret_in_diff"
BLOB_SECRET_FAILURE = "COMMIT_SECURITY_FAILURE:secret_in_staged_blob"
BLOB_CONTENT_FAILURE = "COMMIT_SECURITY_FAILURE:unscannable_staged_blob"
COMMIT_SECURITY_FAILURE = "COMMIT_SECURITY_FAILURE"
MAX_SECRET_SCAN_BLOB_BYTES = 16 * 1024 * 1024

_FAILURE_EVIDENCE_MAX_BYTES = 48 * 1024
_FAILURE_SUMMARY_MAX_BYTES = 8 * 1024


def _bounded_utf8(value: str, budget: int) -> str:
    data = value.encode("utf-8", errors="replace")
    if len(data) <= budget:
        return value
    marker = "\n[... excerpt shortened ...]\n"
    marker_size = len(marker.encode("utf-8"))
    if budget <= marker_size:
        return data[:budget].decode("utf-8", errors="ignore")
    head_size = (budget - marker_size) // 2
    tail_size = budget - marker_size - head_size
    return (
        data[:head_size].decode("utf-8", errors="ignore")
        + marker
        + data[-tail_size:].decode("utf-8", errors="ignore")
    )


def extract_failure_evidence(
    *,
    stdout_log: str = "",
    stderr_log: str = "",
    stdout_tail: str = "",
    stderr_tail: str = "",
    stdout_log_path: str | None = None,
    stderr_log_path: str | None = None,
    max_bytes: int = _FAILURE_EVIDENCE_MAX_BYTES,
) -> str:
    """Build a deterministic, bounded excerpt around a failed check's cause.

    The complete log is scanned only when supplied by the check artifact. The
    result favors pytest/unittest failure sections, tracebacks and compiler
    diagnostics, with a head/tail fallback for unknown command formats.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    streams = [("stdout", stdout_log), ("stderr", stderr_log)]
    nonempty_logs = [(name, value) for name, value in streams if value]
    all_lines: list[tuple[str, str]] = [
        (name, line) for name, value in nonempty_logs for line in value.splitlines()
    ]
    summary = "\n".join(
        f"[{name} tail]\n{_bounded_utf8(value, _FAILURE_SUMMARY_MAX_BYTES)}"
        for name, value in (("stdout", stdout_tail), ("stderr", stderr_tail))
        if value
    )

    selected: list[str] = []
    full_text = "\n".join(line for _name, line in all_lines)
    pytest_failed = re.search(r"(?m)^FAILED\s+([^\s]+)", full_text)
    if pytest_failed is None:
        pytest_failed = re.search(r"(?m)^FAILED\s+([^\s]+)", summary)

    lines = [line for _name, line in all_lines]
    failure_boundary = re.search(r"(?m)^={3,}\s*FAILURES\s*={3,}\s*$", full_text)
    pytest_headers = [
        index for index, line in enumerate(lines)
        if re.match(r"^_{3,}.*_{3,}\s*$", line)
    ]
    if pytest_headers:
        target = pytest_failed.group(1) if pytest_failed else ""
        target_name = target.rsplit("::", 1)[-1].split("[", 1)[0]
        start = next(
            (
                index for index in pytest_headers
                if target and (target in lines[index] or (target_name and target_name in lines[index]))
            ),
            None,
        )
        if start is None:
            after_failures = 0
            if failure_boundary is not None:
                after_failures = full_text[:failure_boundary.start()].count("\n")
            start = next((index for index in pytest_headers if index >= after_failures), pytest_headers[0])
        end = next((index for index in pytest_headers if index > start), len(lines))
        end = min(
            end,
            next((index for index in range(start + 1, len(lines))
                  if "short test summary info" in lines[index]), len(lines)),
        )
        selected = lines[start:end]

    if not selected:
        unit_header = re.compile(r"^(?:FAIL|ERROR):\s+.*$", re.IGNORECASE)
        start = next((index for index, line in enumerate(lines) if unit_header.match(line)), None)
        if start is not None:
            end = next(
                (index for index in range(start + 1, len(lines))
                 if unit_header.match(lines[index]) or lines[index].startswith("=" * 5)),
                min(len(lines), start + 120),
            )
            selected = lines[start:end]

    if not selected:
        error_pattern = re.compile(
            r"Traceback \(most recent call last\)|^\s*E\s{2,}|"
            r"\b(?:fatal\s+)?error:|\b(?:type error|assertionerror|assertion failed)\b|"
            r"\b(?:FAILED|FAILURE|ERROR)\b",
            re.IGNORECASE,
        )
        anchors = [index for index, line in enumerate(lines) if error_pattern.search(line)]
        if anchors:
            anchor = anchors[0]
            traceback_start = next(
                (index for index in range(anchor, -1, -1)
                 if "Traceback (most recent call last)" in lines[index]),
                None,
            )
            start = max(0, (traceback_start if traceback_start is not None else anchor) - 4)
            end = min(len(lines), max(anchor + 7, (traceback_start or anchor) + 24))
            selected = lines[start:end]

    if not selected and lines:
        # Unknown command output: retain a small deterministic head and tail.
        selected = lines[:10]
        if len(lines) > 20:
            selected += ["[... middle omitted ...]"]
        selected += lines[-10:] if len(lines) > 10 else []

    cause = "\n".join(selected)
    log_refs = [
        f"stdout: {stdout_log_path}" if stdout_log_path else "",
        f"stderr: {stderr_log_path}" if stderr_log_path else "",
    ]
    parts = [
        "SUMMARY TAIL\n" + (summary or "No output tail was recorded."),
        "FIRST FAILURE / TRACEBACK / DIAGNOSTIC\n" + (cause or "No recognizable diagnostic was found."),
        "COMPLETE LOGS\n" + ("\n".join(item for item in log_refs if item) or "No complete log path was recorded."),
    ]
    return _bounded_utf8("\n\n".join(parts), max_bytes)

_TEXT_DIFF_SUFFIXES = frozenset(
    {
        ".py",
        ".pyx",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".sh",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
    }
)


def _utf8_prefix(value: str, limit: int) -> str:
    """Return a valid UTF-8 prefix no larger than *limit* bytes."""

    if limit <= 0:
        return ""
    return value.encode("utf-8", errors="replace")[:limit].decode(
        "utf-8", errors="ignore"
    )






@dataclass(frozen=True)
class EvidenceBundle:
    base_sha: str
    staged_tree_sha: str | None
    changed_files: tuple[str, ...]
    diff: str
    checks: tuple[CheckResult, ...]
    deterministic_passed: bool
    failures: tuple[str, ...]
    required_check_ids: tuple[str, ...] = ()
    # Durable, non-blocking facts about this episode: a check skipped for
    # unavailable infrastructure, a failure already red on the base commit.
    warnings: tuple[str, ...] = ()
    # Required checks whose candidate failure the baseline comparison cleared
    # (the same failure, or a subset of it, already present on the base
    # commit).  They count as answered evidence, never as a hidden PASS.
    baseline_cleared: tuple[str, ...] = ()


def required_checks_passed(bundle: EvidenceBundle) -> bool:
    """Verify every required ID has a durable normal PASS result.

    A check whose trusted preflight proved its infrastructure unavailable is
    explicitly *skipped*: it counts as answered, and its warning is durable.
    """

    by_name: dict[str, Any] = {}
    for raw in bundle.checks:
        if isinstance(raw, CheckResult):
            by_name[raw.name] = raw
        elif isinstance(raw, dict) and isinstance(raw.get("name"), str):
            by_name[raw["name"]] = raw
    for check_id in bundle.required_check_ids:
        result = by_name.get(check_id)
        if result is None:
            return False
        get = result.get if isinstance(result, dict) else lambda key, default=None: getattr(result, key, default)
        if get("failure_kind", "passed") == "skipped_infra":
            continue
        if check_id in set(bundle.baseline_cleared):
            continue
        if (
            get("exit_code") != 0
            or bool(get("timed_out", False))
            or get("failure_kind", "passed") != "passed"
            or (bool(get("workspace_mutated", False)) and not bool(get("mutation_recovered", False)))
        ):
            return False
    return True


class EvidenceError(RuntimeError):
    """Evidence cannot be collected safely."""


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _log_stems(checks: tuple[CheckResult, ...]) -> tuple[str, ...]:
    # Keep this in sync with validation._safe_log_stem without exposing a
    # filesystem naming helper as part of the public API.
    import re

    used: set[str] = set()
    stems: list[str] = []
    for check in checks:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", check.name).strip(".") or "check"
        candidate = stem
        index = 2
        while candidate in used:
            candidate = f"{stem}-{index}"
            index += 1
        used.add(candidate)
        stems.append(candidate)
    return tuple(stems)


def persist_evidence(
    bundle: EvidenceBundle,
    evidence_dir: str | Path,
    *,
    write_logs: bool = True,
    secrets: tuple[str, ...] = (),
) -> None:
    """Persist full diff/log artifacts and audit-safe JSON metadata.

    ``write_logs=False`` keeps the complete log files already written by
    :func:`run_checks` instead of replacing them with in-memory copies.
    """

    directory = Path(evidence_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stems = _log_stems(bundle.checks)
    checks_dir = directory / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    if write_logs:
        for check, stem in zip(bundle.checks, stems):
            _write_atomic(checks_dir / f"{stem}.stdout.log", redact(check.stdout_log, secrets))
            _write_atomic(checks_dir / f"{stem}.stderr.log", redact(check.stderr_log, secrets))

    checks_payload = [
        check_result_json(check, log_stem=stem)
        for check, stem in zip(bundle.checks, stems)
    ]
    _write_atomic(directory / "checks.json", _json_text(checks_payload))
    diff = redact(bundle.diff, secrets)
    _write_atomic(directory / "diff.patch", diff)
    changed_files = "".join(f"{path}\n" for path in bundle.changed_files)
    _write_atomic(directory / "changed-files.txt", changed_files)
    evidence_payload = asdict(bundle)
    evidence_payload["diff"] = diff
    evidence_payload["changed_files"] = list(bundle.changed_files)
    evidence_payload["checks"] = checks_payload
    evidence_payload["failures"] = list(bundle.failures)
    _write_atomic(directory / "evidence.json", _json_text(evidence_payload))


def _gate_failures(
    *,
    head_matches: bool,
    diff: str,
    changed_files: tuple[str, ...],
    max_diff_bytes: int,
) -> list[str]:
    failures: list[str] = []
    if not head_matches:
        failures.append("COMMIT_GATE_FAILED")
    if not diff or not changed_files:
        failures.append("DETERMINISTIC_GATE_FAILED:empty_diff")
    if len(diff.encode("utf-8", errors="replace")) > max_diff_bytes:
        failures.append(DIFF_SIZE_FAILURE)
    return failures


def _staged_blob_failures(
    root: Path,
    changed_files: tuple[str, ...],
    changes: tuple[StagedChange, ...],
    secrets: tuple[str, ...],
) -> list[str]:
    """Scan the changed index blobs without loading oversized blobs.

    Only changed paths are described (one diff and one batched cat-file
    call), so the cost follows the change, not the repository size.
    """

    failures: list[str] = []
    described = {change.path for change in changes}
    # Both lists come from the same index; a path the raw diff cannot
    # describe is never silently treated as a deletion.
    for path in changed_files:
        if path not in described:
            failures.append(f"{BLOB_CONTENT_FAILURE}:{path}")
    encoded_secrets = tuple(secret.encode("utf-8") for secret in secrets if secret)
    for blob in staged_changed_blobs(root, changes):
        if blob.size > MAX_SECRET_SCAN_BLOB_BYTES:
            failures.append(f"{BLOB_CONTENT_FAILURE}:{blob.path}")
            continue
        if not encoded_secrets:
            continue
        content = read_staged_blob(root, blob.object_id)
        if any(secret in content for secret in encoded_secrets):
            failures.append(f"{BLOB_SECRET_FAILURE}:{blob.path}")
    return failures


def _unreviewable_text_diff_failures(
    root: Path,
    changed_files: tuple[str, ...],
    changes: tuple[StagedChange, ...],
) -> list[str]:
    """Reject source-like paths whose staged diff is binary-only."""

    binary_paths = staged_binary_files(root)
    submodule_paths = frozenset(change.path for change in changes if change.is_gitlink)
    return [
        f"{COMMIT_SECURITY_FAILURE}:{path}"
        for path in changed_files
        if path not in submodule_paths
        and Path(path).suffix.casefold() in _TEXT_DIFF_SUFFIXES
        and path in binary_paths
    ]


def scan_staged_security(
    worktree: str | Path,
    *,
    secrets: tuple[str, ...] = (),
    max_diff_bytes: int | None = None,
) -> tuple[str, ...]:
    """Reuse the canonical blob, secret, binary and size policies.

    This is intentionally a scanner over the already frozen index.  It is
    used by :func:`collect_evidence` and by the reusable commit gate, so a
    changed blob cannot pass merely because an earlier evidence snapshot was
    later modified.
    """

    root = Path(worktree).expanduser().resolve()
    changed_files = staged_changed_files(root)
    changes = staged_changes(root)
    diff = staged_diff(root)
    failures = _staged_blob_failures(root, changed_files, changes, secrets)
    failures.extend(_unreviewable_text_diff_failures(root, changed_files, changes))
    if contains_secret(diff, secrets):
        failures.append(DIFF_SECRET_FAILURE)
    if max_diff_bytes is not None and len(diff.encode("utf-8", errors="replace")) > max_diff_bytes:
        failures.append(DIFF_SIZE_FAILURE)
    return tuple(failures)


def collect_evidence(
    worktree: str | Path,
    base_sha: str,
    config: HarnessConfig,
    *,
    evidence_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    secrets: tuple[str, ...] = (),
    check_failures_hard: bool = True,
    expected_head_sha: str | None = None,
    required_check_ids: tuple[str, ...] | list[str] | None = None,
    enforce_diff_size: bool = True,
    allow_empty_diff: bool = False,
    retry_check_infrastructure: Callable[[str, str], bool] | None = None,
    skip_checks: Mapping[str, str] | None = None,
    judge_check: Callable[[CheckConfig, CheckResult], tuple[str | None, str | None]] | None = None,
) -> EvidenceBundle:
    """Run all configured checks, then stage and freeze the submitted tree.

    *judge_check* is the gate's comparison of one check against its baseline:
    it returns the failure code to record, if any, and a durable warning.  A
    check it clears is not a failure, whatever the exit status was.
    """

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be a HarnessConfig")
    if not isinstance(base_sha, str) or not base_sha.strip():
        raise ValueError("base_sha must be a non-empty string")

    root = Path(worktree).expanduser().resolve()
    logs_dir = None
    if evidence_dir is not None:
        evidence_path = Path(evidence_dir).expanduser().resolve()
        try:
            evidence_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise EvidenceError("evidence_dir must be outside the worktree")
        logs_dir = evidence_path / "checks"
    checks = run_checks(
        root, config, required_check_ids=required_check_ids,
        logs_dir=logs_dir, tail_bytes=tail_bytes, secrets=secrets,
        retry_infrastructure=retry_check_infrastructure,
        skip=skip_checks,
    )

    head_matches = current_head(root) == (expected_head_sha or base_sha)
    # This is intentionally after every check, including failed checks, so the
    # tree written below is the exact tree offered to the audit.
    stage_all(root)
    # Pipeline v2 freezes accepted implementation steps as commits. The
    # audit therefore needs the cumulative delta from the immutable run
    # base, while security checks still inspect staged post-commit changes.
    changed_files = staged_changed_files_from(root, base_sha)
    diff = staged_diff_from(root, base_sha)
    tree_sha = index_tree_sha(root)

    failures = _gate_failures(
        head_matches=head_matches,
        diff=diff,
        changed_files=changed_files,
        max_diff_bytes=config.max_diff_bytes if enforce_diff_size else 2**63 - 1,
    )
    if allow_empty_diff:
        failures = [failure for failure in failures if failure != "DETERMINISTIC_GATE_FAILED:empty_diff"]
    failures.extend(scan_staged_security(
        root,
        secrets=secrets,
        # _gate_failures owns the evidence-level diff-size decision; avoid
        # duplicating its stable failure code in the existing payload.
        max_diff_bytes=None,
    ))
    try:
        selected_configs = config.select_checks(required_check_ids)
    except ValueError as exc:
        raise EvidenceError(str(exc)) from exc
    warnings: list[str] = []
    cleared: list[str] = []
    for check, check_config in zip(checks, selected_configs):
        # A safely rolled back mutation is archived in check evidence, then
        # the exact check is rerun. Only an unrecovered mutation can close the
        # gate; rollback/ownership failures already raise a hard error.
        if check.failure_kind in {"side_effect_repeated", "side_effect_unstable"}:
            failures.append(f"CHECK_SIDE_EFFECT_REPEATED:{check.name}")
        elif check.workspace_mutated and not check.mutation_recovered:
            failures.append(f"CHECK_FAILED:{check.name}")
        # A check whose trusted preflight answered "no" never ran: the run
        # keeps that fact as a durable warning instead of a red gate.
        if check.failure_kind == "skipped_infra":
            warnings.append(
                f"skipped:{check.name}:{check.skipped_reason or 'infrastructure unavailable'}"
            )
            continue
        # P42 selection itself is the mandatory contract.  ``required`` is
        # metadata for the trusted catalogue and does not alter selection.
        if required_check_ids is None and not check_config.required:
            continue
        if check.timed_out:
            failures.append(f"CHECK_INFRASTRUCTURE_UNAVAILABLE:{check.name}")
        elif check.failure_kind in {
            "missing_executable", "process_start_failed", "signal_terminated",
            "infrastructure_unavailable",
        }:
            failures.append(f"CHECK_INFRASTRUCTURE_UNAVAILABLE:{check.name}")
        elif check.exit_code != 0:
            if judge_check is None:
                failures.append(f"CHECK_FAILED:{check.name}")
                continue
            failure, warning = judge_check(check_config, check)
            if warning:
                warnings.append(warning)
            if failure:
                failures.append(failure)
            else:
                cleared.append(check.name)

    bundle = EvidenceBundle(
        base_sha=base_sha,
        staged_tree_sha=tree_sha,
        changed_files=changed_files,
        diff=diff,
        checks=checks,
        deterministic_passed=not failures,
        failures=tuple(failures),
        required_check_ids=tuple(check.id for check in selected_configs),
        warnings=tuple(warnings),
        baseline_cleared=tuple(cleared),
    )
    if evidence_dir is not None:
        persist_evidence(bundle, evidence_dir, write_logs=False, secrets=secrets)
    return bundle
