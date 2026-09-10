"""Immutable evidence collection and deterministic gate evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gitops import current_head, index_tree_sha, stage_all, staged_changed_files, staged_diff
from .models import HarnessConfig
from .validation import (
    DEFAULT_TAIL_BYTES,
    CheckResult,
    check_result_json,
    run_checks,
)


DIFF_TOO_LARGE = "DIFF_TOO_LARGE"


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


def persist_evidence(bundle: EvidenceBundle, evidence_dir: str | Path) -> None:
    """Persist full diff/log artifacts and reviewer-safe JSON metadata."""

    directory = Path(evidence_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stems = _log_stems(bundle.checks)
    checks_dir = directory / "checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    for check, stem in zip(bundle.checks, stems):
        _write_atomic(checks_dir / f"{stem}.stdout.log", check.stdout_log)
        _write_atomic(checks_dir / f"{stem}.stderr.log", check.stderr_log)

    checks_payload = [
        check_result_json(check, log_stem=stem)
        for check, stem in zip(bundle.checks, stems)
    ]
    _write_atomic(directory / "checks.json", _json_text(checks_payload))
    _write_atomic(directory / "diff.patch", bundle.diff)
    changed_files = "".join(f"{path}\n" for path in bundle.changed_files)
    _write_atomic(directory / "changed-files.txt", changed_files)
    evidence_payload = asdict(bundle)
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


def collect_evidence(
    worktree: str | Path,
    base_sha: str,
    config: HarnessConfig,
    *,
    evidence_dir: str | Path | None = None,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
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
    checks = run_checks(root, config, logs_dir=logs_dir, tail_bytes=tail_bytes)

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
    for check, check_config in zip(checks, config.checks):
        if not check_config.required:
            continue
        if check.timed_out:
            failures.append(f"CHECK_TIMEOUT:{check.name}")
        elif check.exit_code != 0:
            failures.append(f"CHECK_FAILED:{check.name}")
        if check.workspace_mutated:
            failures.append(f"CHECK_MUTATED:{check.name}")

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
        persist_evidence(bundle, evidence_dir)
    return bundle


# Names that read naturally at call sites and preserve a small, stable API.
build_evidence = collect_evidence
build_evidence_bundle = collect_evidence
freeze_evidence = collect_evidence
