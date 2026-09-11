"""Immutable evidence collection and deterministic gate evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gitops import (
    current_head,
    index_tree_sha,
    read_staged_blob,
    stage_all,
    staged_binary_files,
    staged_blobs,
    staged_changed_files,
    staged_diff,
    staged_submodule_paths,
)
from .models import HarnessConfig
from .redaction import contains_secret, redact
from .validation import (
    DEFAULT_TAIL_BYTES,
    CheckResult,
    check_result_json,
    run_checks,
)


DIFF_TOO_LARGE = "DIFF_TOO_LARGE"
SECRET_IN_DIFF = "SECRET_IN_DIFF"
SECRET_IN_STAGED_BLOB = "SECRET_IN_STAGED_BLOB"
UNSCANNABLE_STAGED_BLOB = "UNSCANNABLE_STAGED_BLOB"
UNREVIEWABLE_TEXT_DIFF = "UNREVIEWABLE_TEXT_DIFF"
MAX_SECRET_SCAN_BLOB_BYTES = 16 * 1024 * 1024

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


@dataclass(frozen=True)
class EvidenceBundle:
    base_sha: str
    staged_tree_sha: str | None
    changed_files: tuple[str, ...]
    diff: str
    checks: tuple[CheckResult, ...]
    deterministic_passed: bool
    failures: tuple[str, ...]


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
    """Persist full diff/log artifacts and reviewer-safe JSON metadata.

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
        failures.append("HEAD_MISMATCH")
    if not diff or not changed_files:
        failures.append("EMPTY_DIFF")
    if len(diff.encode("utf-8", errors="replace")) > max_diff_bytes:
        failures.append(DIFF_TOO_LARGE)
    return failures


def _staged_blob_failures(
    root: Path,
    changed_files: tuple[str, ...],
    secrets: tuple[str, ...],
) -> list[str]:
    """Scan changed index blobs without loading oversized blobs."""

    blobs = {blob.path: blob for blob in staged_blobs(root)}
    encoded_secrets = tuple(secret.encode("utf-8") for secret in secrets if secret)
    failures: list[str] = []
    for path in changed_files:
        blob = blobs.get(path)
        if blob is None:
            # Deletions and gitlinks do not have a staged blob to scan.
            continue
        if blob.size > MAX_SECRET_SCAN_BLOB_BYTES:
            failures.append(f"{UNSCANNABLE_STAGED_BLOB}:{path}")
            continue
        if not encoded_secrets:
            continue
        content = read_staged_blob(root, blob.object_id)
        if any(secret in content for secret in encoded_secrets):
            failures.append(f"{SECRET_IN_STAGED_BLOB}:{path}")
    return failures


def _unreviewable_text_diff_failures(
    root: Path,
    changed_files: tuple[str, ...],
) -> list[str]:
    """Reject source-like paths whose staged diff is binary-only."""

    binary_paths = staged_binary_files(root)
    submodule_paths = staged_submodule_paths(root)
    return [
        f"{UNREVIEWABLE_TEXT_DIFF}:{path}"
        for path in changed_files
        if path not in submodule_paths
        and Path(path).suffix.casefold() in _TEXT_DIFF_SUFFIXES
        and path in binary_paths
    ]


def collect_evidence(
    worktree: str | Path,
    base_sha: str,
    config: HarnessConfig,
    *,
    evidence_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    secrets: tuple[str, ...] = (),
) -> EvidenceBundle:
    """Run all configured checks, then stage and freeze the submitted tree."""

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
        root, config, logs_dir=logs_dir, tail_bytes=tail_bytes, secrets=secrets
    )

    head_matches = current_head(root) == base_sha
    # This is intentionally after every check, including failed checks, so the
    # tree written below is the exact tree offered to the reviewer.
    stage_all(root)
    changed_files = staged_changed_files(root)
    diff = staged_diff(root)
    tree_sha = index_tree_sha(root)

    failures = _gate_failures(
        head_matches=head_matches,
        diff=diff,
        changed_files=changed_files,
        max_diff_bytes=config.max_diff_bytes,
    )
    failures.extend(_staged_blob_failures(root, changed_files, secrets))
    failures.extend(_unreviewable_text_diff_failures(root, changed_files))
    if contains_secret(diff, secrets):
        # The diff would be sent to the reviewer endpoint and persisted.
        failures.append(SECRET_IN_DIFF)
    for check, check_config in zip(checks, config.checks):
        # A mutation changes the code that is reviewed, whatever the check's
        # importance: it always closes the gate.
        if check.workspace_mutated:
            failures.append(f"CHECK_MUTATED:{check.name}")
        if not check_config.required:
            continue
        if check.timed_out:
            failures.append(f"CHECK_TIMEOUT:{check.name}")
        elif check.exit_code != 0:
            failures.append(f"CHECK_FAILED:{check.name}")

    bundle = EvidenceBundle(
        base_sha=base_sha,
        staged_tree_sha=tree_sha,
        changed_files=changed_files,
        diff=diff,
        checks=checks,
        deterministic_passed=not failures,
        failures=tuple(failures),
    )
    if evidence_dir is not None:
        persist_evidence(bundle, evidence_dir, write_logs=False, secrets=secrets)
    return bundle


# Names that read naturally at call sites and preserve a small, stable API.
build_evidence = collect_evidence
build_evidence_bundle = collect_evidence
freeze_evidence = collect_evidence
