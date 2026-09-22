"""Immutable evidence collection and deterministic gate evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gitops import (
    StagedChange,
    current_head,
    index_tree_sha,
    read_staged_blob,
    stage_all,
    staged_binary_files,
    staged_changed_blobs,
    staged_changed_files,
    staged_changes,
    staged_diff,
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


def _utf8_prefix(value: str, limit: int) -> str:
    """Return a valid UTF-8 prefix no larger than *limit* bytes."""

    if limit <= 0:
        return ""
    return value.encode("utf-8", errors="replace")[:limit].decode(
        "utf-8", errors="ignore"
    )


def _diff_sections(diff: str) -> list[str]:
    """Split a git diff into deterministic per-file sections."""

    sections: list[str] = []
    current: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections or ([diff] if diff else [])


def _bounded_section(section: str, budget: int) -> str:
    """Keep a section header and both ends of its body within *budget*."""

    encoded = section.encode("utf-8", errors="replace")
    if len(encoded) <= budget:
        return section
    if budget <= 0:
        return ""
    lines = section.splitlines(keepends=True)
    header = lines[0] if lines else section
    marker = "\n[... file diff body abbreviated ...]\n"
    marker_bytes = len(marker.encode("utf-8"))
    if budget <= marker_bytes:
        return _utf8_prefix(section, budget)
    header_text = _utf8_prefix(header, max(1, budget // 4))
    remaining = budget - len(header_text.encode("utf-8")) - marker_bytes
    if remaining <= 0:
        return _utf8_prefix(header_text + marker, budget)
    body = "".join(lines[1:])
    head_budget = (remaining + 1) // 2
    tail_budget = remaining - head_budget
    head = _utf8_prefix(body, head_budget)
    tail = _utf8_prefix(body[-max(1, tail_budget):], tail_budget)
    result = header_text + head + marker + tail
    return _utf8_prefix(result, budget)


def bounded_semantic_diff(diff: str, max_bytes: int) -> tuple[str, bool, int]:
    """Build a deterministic, representative diff payload for semantic models.

    The exact diff remains the evidence artifact.  This helper only bounds the
    copy embedded in an LLM request and distributes the budget across all git
    file sections so early files cannot crowd out later ones.
    """

    if not isinstance(diff, str):
        raise TypeError("diff must be a string")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    full_bytes = len(diff.encode("utf-8", errors="replace"))
    if full_bytes <= max_bytes:
        return diff, False, full_bytes

    sections = _diff_sections(diff)
    metadata = (
        "SEMANTIC DIFF EXCERPT\n"
        f"FULL_DIFF_BYTES: {full_bytes}\n"
        "TRUNCATED: true\n"
        f"FILES_CHANGED: {len(sections)}\n\n"
        "The exact candidate commit/tree remains authoritative.\n"
        "Some diff bodies are abbreviated because this is an LLM context budget,\n"
        "not an execution or correctness gate.\n\n"
    )
    metadata_bytes = len(metadata.encode("utf-8"))
    if metadata_bytes >= max_bytes:
        return _utf8_prefix(metadata, max_bytes), True, full_bytes

    available = max_bytes - metadata_bytes
    # Every section gets a deterministic share.  This guarantees that a
    # large plan still exposes the last file rather than stopping at a prefix.
    base, extra = divmod(available, len(sections) or 1)
    excerpts: list[str] = []
    for index, section in enumerate(sections):
        budget = base + (1 if index < extra else 0)
        excerpt = _bounded_section(section, budget)
        if excerpt:
            excerpts.append(excerpt)
    payload = metadata + "\n".join(excerpts)
    return _utf8_prefix(payload, max_bytes), True, full_bytes


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
            failures.append(f"{UNSCANNABLE_STAGED_BLOB}:{path}")
    encoded_secrets = tuple(secret.encode("utf-8") for secret in secrets if secret)
    for blob in staged_changed_blobs(root, changes):
        if blob.size > MAX_SECRET_SCAN_BLOB_BYTES:
            failures.append(f"{UNSCANNABLE_STAGED_BLOB}:{blob.path}")
            continue
        if not encoded_secrets:
            continue
        content = read_staged_blob(root, blob.object_id)
        if any(secret in content for secret in encoded_secrets):
            failures.append(f"{SECRET_IN_STAGED_BLOB}:{blob.path}")
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
        f"{UNREVIEWABLE_TEXT_DIFF}:{path}"
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
        failures.append(SECRET_IN_DIFF)
    if max_diff_bytes is not None and len(diff.encode("utf-8", errors="replace")) > max_diff_bytes:
        failures.append(DIFF_TOO_LARGE)
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
        root, config, required_check_ids=required_check_ids,
        logs_dir=logs_dir, tail_bytes=tail_bytes, secrets=secrets
    )

    head_matches = current_head(root) == (expected_head_sha or base_sha)
    # This is intentionally after every check, including failed checks, so the
    # tree written below is the exact tree offered to the reviewer.
    stage_all(root)
    changed_files = staged_changed_files(root)
    changes = staged_changes(root)
    diff = staged_diff(root)
    tree_sha = index_tree_sha(root)

    failures = _gate_failures(
        head_matches=head_matches,
        diff=diff,
        changed_files=changed_files,
        max_diff_bytes=config.max_diff_bytes if enforce_diff_size else 2**63 - 1,
    )
    if allow_empty_diff:
        failures = [failure for failure in failures if failure != "EMPTY_DIFF"]
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
    for check, check_config in zip(checks, selected_configs):
        # A mutation changes the code that is reviewed, whatever the check's
        # importance: it always closes the gate.
        if check.workspace_mutated:
            failures.append(f"CHECK_MUTATED:{check.name}")
        # P42 selection itself is the mandatory contract.  ``required`` is
        # metadata for the trusted catalogue and does not alter selection.
        if required_check_ids is None and not check_config.required:
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
        required_check_ids=tuple(check.id for check in selected_configs),
    )
    if evidence_dir is not None:
        persist_evidence(bundle, evidence_dir, write_logs=False, secrets=secrets)
    return bundle


# Names that read naturally at call sites and preserve a small, stable API.
build_evidence = collect_evidence
build_evidence_bundle = collect_evidence
freeze_evidence = collect_evidence
