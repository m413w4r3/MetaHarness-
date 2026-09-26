"""The check baseline: what the base commit already answered, and how it compares.

A gate answers one question -- *did this candidate introduce a new regression?*
-- and that question only exists relative to the immutable base commit of the
run.  This module owns the whole baseline boundary:

* the deterministic identity of a check subset (:func:`check_config_sha`);
* the durable, atomic cache under ``<runs_root>/.baseline/``;
* the execution of the checks against the exact content of ``base_sha`` in a
  throwaway detached worktree, through the one existing check runner;
* the failure-id parsers (pytest, vitest, jest, JUnit XML) and the comparison
  that turns one baseline and one candidate result into a verdict.

Nothing here decides a pipeline outcome: it reports verdicts, and the gate
projects them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .gitops import (
    GitError,
    create_detached_worktree,
    remove_detached_worktree,
)
from .models import CheckConfig, HarnessConfig
from .result import ResultArtifactError, atomic_write_text
from .validation import CheckResult, ValidationError, run_checks
from .workspace import WorkspaceSetupError, prepare_workspace

BASELINE_DIRNAME = ".baseline"
BASELINE_SCHEMA_VERSION = 1

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIPPED_INFRA = "SKIPPED_INFRA"
# No usable signal: the check could not run against the base content at all.
STATUS_UNAVAILABLE = "UNAVAILABLE"

CHECK_STATUSES = frozenset({STATUS_PASS, STATUS_FAIL, STATUS_SKIPPED_INFRA, STATUS_UNAVAILABLE})

_SHA_PREFIX_BYTES = 12


class CheckVerdict(StrEnum):
    """What one candidate check result is, relative to its baseline."""

    PASS = "PASS"
    REGRESSION = "REGRESSION"
    IMPROVED = "IMPROVED"
    BASELINE_WARNING = "PASS_WITH_BASELINE_WARNING"
    BASELINE_RED = "BASELINE_RED"
    SKIPPED_INFRA = "SKIPPED_INFRA"


# ---------------------------------------------------------------------------
# failure ids
# ---------------------------------------------------------------------------

# ``FAILED tests/test_x.py::test_y`` and its collection-error sibling.  This is
# the exact line pytest prints in its short test summary; the optional
# ``- AssertionError: ...`` tail pytest appends is never part of the id.
_PYTEST_FAILURE = re.compile(r"(?m)^\s*(?:FAILED|ERROR)\s+(\S+?::\S+?)(?:\s+-\s.*)?\s*$")
# A vitest/jest file line, and the case markers of both reporters.
_JS_FILE = re.compile(r"(?m)^\s*(?:❯|FAIL)\s+(\S+?\.(?:[cm]?[jt]sx?|vue|svelte))\b")
_JS_CASE = re.compile(r"(?m)^\s*[×✗✕]\s+(.+?)\s*$")
_JEST_HEADER = re.compile(r"(?m)^\s*●\s+(.+?)\s*$")
_JEST_SIGNATURE = re.compile(r"(?m)^\s*(?:●|×|✗|✕|PASS |FAIL )")
_DURATION_SUFFIX = re.compile(r"\s+\d+(?:\.\d+)?\s*m?s$")


def _pytest_ids(text: str) -> list[str]:
    return [match.group(1) for match in _PYTEST_FAILURE.finditer(text)]


def _javascript_ids(text: str) -> list[str]:
    """Best-effort vitest/jest ids, in output order.

    Both reporters name a case (``×``, ``✗``) and, for vitest, the file the
    case belongs to (``❯ path``).  An unnamed case is reported verbatim: the
    comparison only ever needs the *same* spelling on both sides.
    """

    ids: list[str] = []
    current_file: str | None = None
    files = list(_JS_FILE.finditer(text))
    next_file = 0
    for match in _JS_CASE.finditer(text):
        while next_file < len(files) and files[next_file].start() < match.start():
            current_file = files[next_file].group(1)
            next_file += 1
        name = _DURATION_SUFFIX.sub("", match.group(1).strip()).strip().rstrip(",")
        if not name:
            continue
        ids.append(f"{current_file}::{name}" if current_file else name)
    for match in _JEST_HEADER.finditer(text):
        name = match.group(1).strip()
        if name:
            ids.append(name)
    return ids


def parse_failure_ids(text: str) -> tuple[bool, tuple[str, ...]]:
    """Extract failing test ids from one check's output.

    ``(parsed, ids)``: ``parsed`` is false whenever the output carries no
    recognisable runner marker, so an opaque failure is never mistaken for a
    fully understood one.
    """

    if not isinstance(text, str):
        return False, ()
    pytest_ids = _pytest_ids(text)
    if pytest_ids:
        return True, tuple(dict.fromkeys(pytest_ids))
    if not _JEST_SIGNATURE.search(text):
        return False, ()
    javascript_ids = _javascript_ids(text)
    if not javascript_ids:
        return False, ()
    return True, tuple(dict.fromkeys(javascript_ids))


def parse_junit_ids(path: Path) -> tuple[bool, tuple[str, ...]]:
    """Read failing test ids from a JUnit XML report the runner wrote itself."""

    try:
        root = ElementTree.parse(Path(path)).getroot()
    except (OSError, ElementTree.ParseError, ValueError):
        return False, ()
    if root is None:
        return False, ()
    ids: list[str] = []
    for case in root.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        classname = (case.get("classname") or "").strip()
        name = (case.get("name") or "").strip()
        if not name:
            continue
        ids.append(f"{classname}::{name}" if classname else name)
    if not ids:
        return False, ()
    return True, tuple(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# durable records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckBaseline:
    """One check's durable answer on the run's base commit."""

    check_id: str
    status: str
    exit_code: int
    failure_ids: tuple[str, ...] = ()
    failure_ids_parsed: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError("baseline status is unknown")

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "failure_ids": list(self.failure_ids),
            "failure_ids_parsed": self.failure_ids_parsed,
            "detail": self.detail,
        }

    @classmethod
    def from_payload(cls, value: object) -> "CheckBaseline | None":
        if not isinstance(value, dict):
            return None
        check_id = value.get("id")
        status = value.get("status")
        exit_code = value.get("exit_code")
        ids = value.get("failure_ids", [])
        parsed = value.get("failure_ids_parsed", False)
        detail = value.get("detail", "")
        if (
            not isinstance(check_id, str) or not check_id
            or status not in CHECK_STATUSES
            or isinstance(exit_code, bool) or not isinstance(exit_code, int)
            or isinstance(ids, (str, bytes)) or not isinstance(ids, list)
            or any(not isinstance(item, str) for item in ids)
            or not isinstance(parsed, bool)
            or not isinstance(detail, str)
        ):
            return None
        return cls(check_id, status, exit_code, tuple(ids), parsed, detail)


@dataclass(frozen=True)
class BaselineRecord:
    """The baseline of one check subset on one base commit."""

    base_sha: str
    check_config_sha: str
    checks: tuple[CheckBaseline, ...]
    check_ids: tuple[str, ...] = ()
    unavailable_reason: str = ""
    # The effective definition of every captured check, so a later subset can
    # be sliced out of this record only when the definitions really agree.
    fingerprints: tuple[tuple[str, str], ...] = ()
    schema_version: int = BASELINE_SCHEMA_VERSION

    def by_id(self) -> dict[str, CheckBaseline]:
        return {entry.check_id: entry for entry in self.checks}

    def entry(self, check_id: str) -> CheckBaseline | None:
        return self.by_id().get(check_id)

    def fingerprint_of(self, check_id: str) -> str | None:
        return dict(self.fingerprints).get(check_id)

    def subset(self, checks: Sequence[CheckConfig]) -> "BaselineRecord | None":
        """Slice this record for a check subset whose definitions it proves."""

        own = self.by_id()
        entries: list[CheckBaseline] = []
        for check in checks:
            entry = own.get(check.id)
            if entry is None or self.fingerprint_of(check.id) != check_fingerprint(check):
                return None
            entries.append(entry)
        return BaselineRecord(
            base_sha=self.base_sha,
            check_config_sha=check_config_sha(checks),
            checks=tuple(entries),
            check_ids=tuple(check.id for check in checks),
            unavailable_reason=self.unavailable_reason,
            fingerprints=tuple((check.id, check_fingerprint(check)) for check in checks),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_sha": self.base_sha,
            "check_config_sha": self.check_config_sha,
            "check_ids": list(self.check_ids),
            "unavailable_reason": self.unavailable_reason,
            "check_fingerprints": {key: value for key, value in self.fingerprints},
            "checks": [entry.payload() for entry in self.checks],
        }

    @classmethod
    def from_payload(cls, value: object) -> "BaselineRecord | None":
        if not isinstance(value, dict) or value.get("schema_version") != BASELINE_SCHEMA_VERSION:
            return None
        base_sha = value.get("base_sha")
        config_sha = value.get("check_config_sha")
        raw_checks = value.get("checks")
        check_ids = value.get("check_ids", [])
        reason = value.get("unavailable_reason", "")
        if (
            not isinstance(base_sha, str) or not base_sha
            or not isinstance(config_sha, str) or not config_sha
            or isinstance(raw_checks, (str, bytes)) or not isinstance(raw_checks, list)
            or isinstance(check_ids, (str, bytes)) or not isinstance(check_ids, list)
            or any(not isinstance(item, str) for item in check_ids)
            or not isinstance(reason, str)
        ):
            return None
        raw_fingerprints = value.get("check_fingerprints", {})
        if (
            not isinstance(raw_fingerprints, dict)
            or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in raw_fingerprints.items()
            )
        ):
            return None
        entries: list[CheckBaseline] = []
        for raw in raw_checks:
            entry = CheckBaseline.from_payload(raw)
            if entry is None:
                return None
            entries.append(entry)
        if len({entry.check_id for entry in entries}) != len(entries):
            return None
        return cls(
            base_sha, config_sha, tuple(entries), tuple(check_ids), reason,
            tuple(sorted(raw_fingerprints.items())),
        )


def check_fingerprint(check: CheckConfig) -> str:
    """The deterministic identity of one effective check definition.

    Only the values that can change a check's result take part: the identifier,
    the command, its working directory, its timeout, its preflight, whether the
    catalogue declares it required, whether it is explicitly blocking, and the
    optional JUnit report it writes.
    """

    payload = json.dumps({
        "id": check.id,
        "argv": list(check.argv),
        "cwd": check.cwd,
        "timeout_seconds": check.timeout_seconds,
        "preflight_argv": list(check.preflight_argv),
        "required": check.required,
        "blocking": check.blocking,
        "junit_xml": check.junit_xml,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_config_sha(checks: Sequence[CheckConfig]) -> str:
    """The deterministic identity of one effective check subset."""

    fingerprints = tuple((check.id, check_fingerprint(check)) for check in checks)
    canonical = json.dumps(fingerprints, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def junit_report_path(check: CheckConfig, result: CheckResult) -> Path | None:
    """The JUnit report one check declared, resolved against its own cwd."""

    if not check.junit_xml:
        return None
    return Path(result.cwd) / check.junit_xml


def baseline_of_result(
    check_id: str, result: CheckResult, *, junit_path: Path | None = None, detail: str = "",
) -> CheckBaseline:
    """Project one runner result into the durable baseline shape."""

    if result.failure_kind == "skipped_infra":
        return CheckBaseline(check_id, STATUS_SKIPPED_INFRA, result.exit_code, (), False, detail)
    if result.failure_kind == "passed" and result.exit_code == 0 and not result.timed_out:
        return CheckBaseline(check_id, STATUS_PASS, 0, (), True, detail)
    if result.failure_kind in {"missing_executable", "process_start_failed"}:
        # The base content was never exercised: no baseline signal exists.
        return CheckBaseline(check_id, STATUS_UNAVAILABLE, result.exit_code, (), False, detail)
    parsed, ids = parse_output_failure_ids(result, junit_path=junit_path)
    return CheckBaseline(check_id, STATUS_FAIL, result.exit_code, ids, parsed, detail)


def parse_output_failure_ids(
    result: CheckResult, *, junit_path: Path | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Failure ids of one check result: JUnit XML first, then runner output."""

    if junit_path is not None:
        parsed, ids = parse_junit_ids(junit_path)
        if parsed:
            return True, ids
    return parse_failure_ids(f"{result.stdout_log}\n{result.stderr_log}")


PREFLIGHT_FILE = "preflights.json"
PREFLIGHT_SCHEMA_VERSION = 1


def preflight_fingerprint(check: CheckConfig) -> str:
    """The identity of one preflight definition: a changed one is re-evaluated."""

    payload = json.dumps({
        "id": check.id,
        "argv": list(check.argv),
        "cwd": check.cwd,
        "preflight_argv": list(check.preflight_argv),
        "timeout_seconds": check.timeout_seconds,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_preflight_verdicts(run_dir: str | Path) -> dict[str, Any]:
    """Read one run's durable preflight verdicts; a corrupt file is ignored."""

    try:
        payload = json.loads(
            (Path(run_dir) / PREFLIGHT_FILE).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or not isinstance(payload.get("checks"), dict)
    ):
        return {}
    return payload["checks"]


def preflight_skips(run_dir: str | Path, checks: Sequence[CheckConfig]) -> dict[str, str]:
    """The checks this run must skip because their trusted preflight said no."""

    verdicts = load_preflight_verdicts(run_dir)
    skipped: dict[str, str] = {}
    for check in checks:
        if not check.preflight_argv:
            continue
        entry = verdicts.get(check.id)
        if (
            isinstance(entry, dict)
            and entry.get("status") == "FAIL"
            and entry.get("fingerprint") == preflight_fingerprint(check)
        ):
            skipped[check.id] = "PREFLIGHT_FAILED"
    return skipped


def unavailable_record(
    base_sha: str, checks: Sequence[CheckConfig], reason: str,
) -> BaselineRecord:
    """The honest baseline of a check subset whose capture could not happen."""

    entries = tuple(
        CheckBaseline(check.id, STATUS_UNAVAILABLE, -1, (), False, reason) for check in checks
    )
    return BaselineRecord(
        base_sha=base_sha,
        check_config_sha=check_config_sha(checks),
        checks=entries,
        check_ids=tuple(check.id for check in checks),
        unavailable_reason=reason,
        fingerprints=tuple((check.id, check_fingerprint(check)) for check in checks),
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckJudgement:
    """One candidate check, judged against its baseline."""

    check_id: str
    verdict: CheckVerdict
    failure: str | None = None
    warning: str | None = None
    new_failure_ids: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.verdict is CheckVerdict.REGRESSION


def _warning(check_id: str, detail: str) -> str:
    return f"baseline:{check_id}:{detail}"


def compare_check(
    check_id: str,
    candidate: CheckBaseline,
    baseline: CheckBaseline | None,
) -> CheckJudgement:
    """The whole gate rule for one check, in one place."""

    if candidate.status == STATUS_SKIPPED_INFRA:
        return CheckJudgement(
            check_id, CheckVerdict.SKIPPED_INFRA,
            warning=_warning(check_id, "skipped: infrastructure unavailable"),
        )
    if candidate.status in {STATUS_PASS, STATUS_UNAVAILABLE} and candidate.exit_code == 0:
        if baseline is not None and baseline.status == STATUS_FAIL:
            return CheckJudgement(check_id, CheckVerdict.IMPROVED)
        return CheckJudgement(check_id, CheckVerdict.PASS)
    if baseline is None:
        # No baseline was ever captured for this check: fail closed.
        return CheckJudgement(
            check_id, CheckVerdict.REGRESSION, failure=f"CHECK_FAILED:{check_id}",
        )
    if baseline.status == STATUS_PASS:
        return CheckJudgement(
            check_id, CheckVerdict.REGRESSION, failure=f"CHECK_FAILED:{check_id}",
            new_failure_ids=candidate.failure_ids if candidate.failure_ids_parsed else (),
        )
    if baseline.status in {STATUS_UNAVAILABLE, STATUS_SKIPPED_INFRA}:
        return CheckJudgement(
            check_id, CheckVerdict.BASELINE_RED,
            warning=_warning(check_id, "no usable baseline signal; the failure is not blocking"),
        )
    if not baseline.failure_ids_parsed:
        # A red base nobody could explain never blocks new work.
        return CheckJudgement(
            check_id, CheckVerdict.BASELINE_RED,
            warning=_warning(check_id, "already red on the base commit; not blocking"),
        )
    if not candidate.failure_ids_parsed:
        return CheckJudgement(
            check_id, CheckVerdict.REGRESSION, failure=f"CHECK_FAILED:{check_id}",
            new_failure_ids=(f"unparsed:{check_id}",),
        )
    known = set(baseline.failure_ids)
    new_ids = tuple(item for item in candidate.failure_ids if item not in known)
    if new_ids:
        return CheckJudgement(
            check_id, CheckVerdict.REGRESSION, failure=f"CHECK_FAILED:{check_id}",
            new_failure_ids=new_ids,
        )
    return CheckJudgement(
        check_id, CheckVerdict.BASELINE_WARNING,
        warning=_warning(
            check_id,
            f"{len(candidate.failure_ids)} failure(s) already red on the base commit",
        ),
    )


def judge_results(
    baseline: BaselineRecord | None,
    results: Iterable[CheckResult],
    checks: Sequence[CheckConfig],
    *,
    skipped: Mapping[str, str] | None = None,
) -> tuple[CheckJudgement, ...]:
    """Judge every candidate check of one selection against its baseline."""

    by_id = baseline.by_id() if baseline is not None else {}
    skipped_ids = dict(skipped or {})
    judgements: list[CheckJudgement] = []
    for check, result in zip(checks, results):
        if check.id in skipped_ids:
            judgements.append(CheckJudgement(
                check.id, CheckVerdict.SKIPPED_INFRA,
                warning=_warning(check.id, f"skipped: {skipped_ids[check.id]}"),
            ))
            continue
        judgement = compare_check(
            check.id,
            baseline_of_result(
                check.id, result, junit_path=junit_report_path(check, result),
            ),
            by_id.get(check.id),
        )
        judgements.append(judgement)
    return tuple(judgements)


# ---------------------------------------------------------------------------
# the durable cache
# ---------------------------------------------------------------------------


class BaselineCache:
    """The atomic on-disk cache of baselines, keyed by (base_sha, config sha)."""

    def __init__(self, runs_root: str | Path) -> None:
        self.root = Path(runs_root).expanduser().resolve() / BASELINE_DIRNAME

    def key(self, base_sha: str, config_sha: str) -> str:
        return f"{base_sha}-{config_sha[:_SHA_PREFIX_BYTES]}"

    def record_path(self, base_sha: str, config_sha: str) -> Path:
        return self.root / f"{self.key(base_sha, config_sha)}.json"

    def logs_dir(self, base_sha: str, config_sha: str) -> Path:
        return self.root / "logs" / self.key(base_sha, config_sha)

    def worktrees_dir(self) -> Path:
        return self.root / "worktrees"

    def setup_dir(self, base_sha: str, config_sha: str) -> Path:
        return self.root / "setup" / self.key(base_sha, config_sha)

    def load(self, base_sha: str, config_sha: str) -> BaselineRecord | None:
        """Read one cached baseline; a corrupt entry is ignored, never fatal."""

        path = self.record_path(base_sha, config_sha)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        record = BaselineRecord.from_payload(payload)
        if record is None:
            return None
        if record.base_sha != base_sha or record.check_config_sha != config_sha:
            return None
        return record

    def cached_superset(
        self, base_sha: str, checks: Sequence[CheckConfig],
    ) -> BaselineRecord | None:
        """Slice an already captured superset instead of running the checks again."""

        if not self.root.is_dir():
            return None
        for path in sorted(self.root.glob(f"{base_sha}-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            record = BaselineRecord.from_payload(payload)
            if record is None or record.base_sha != base_sha:
                continue
            subset = record.subset(checks)
            if subset is not None:
                return subset
        return None

    def store(self, record: BaselineRecord) -> None:
        """Publish one baseline atomically; identical publication is idempotent."""

        path = self.record_path(record.base_sha, record.check_config_sha)
        content = json.dumps(
            record.payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"
        try:
            atomic_write_text(path, content)
        except (OSError, ResultArtifactError) as exc:
            raise BaselineError(f"could not publish the check baseline: {exc}") from None
        self._stamp(record)


    def ensure(
        self,
        *,
        repo: Path,
        base_sha: str,
        config: HarnessConfig,
        check_ids: Sequence[str],
        environment: Mapping[str, str],
        setup_commands: Sequence[Any] = (),
        secrets: tuple[str, ...] = (),
        skipped: Mapping[str, str] | None = None,
    ) -> BaselineRecord:
        """Return the baseline of *check_ids*, capturing it at most once.

        The whole subset shares one cache entry.  A capture that cannot happen
        (Git failure, unusable workspace) is *recorded* as unavailable instead
        of blocking the run: an unexplained base condition never forbids new
        work.
        """

        try:
            checks = config.select_checks(tuple(check_ids))
        except ValueError as exc:
            raise BaselineError(f"baseline check selection is invalid: {exc}") from exc
        config_sha = check_config_sha(checks)
        cached = self.load(base_sha, config_sha)
        if cached is not None:
            return cached
        covering = self.cached_superset(base_sha, checks)
        if covering is not None:
            return covering
        try:
            record = self._capture(
                repo=repo, base_sha=base_sha, config=config, checks=checks,
                config_sha=config_sha, environment=environment,
                setup_commands=setup_commands, secrets=secrets, skipped=dict(skipped or {}),
            )
        except (GitError, OSError) as exc:
            record = unavailable_record(
                base_sha, checks, f"{type(exc).__name__}: could not capture the baseline",
            )
            self.store(record)
            return record
        self.store(record)
        return record

    def _capture(
        self,
        *,
        repo: Path,
        base_sha: str,
        config: HarnessConfig,
        checks: Sequence[CheckConfig],
        config_sha: str,
        environment: Mapping[str, str],
        setup_commands: Sequence[Any],
        secrets: tuple[str, ...],
        skipped: Mapping[str, str],
    ) -> BaselineRecord:
        key = self.key(base_sha, config_sha)
        self.worktrees_dir().mkdir(parents=True, exist_ok=True)
        scratch = tempfile.mkdtemp(prefix=f"{key}-", dir=self.worktrees_dir())
        worktree = Path(scratch) / "repo"
        logs_dir = self.logs_dir(base_sha, config_sha)
        setup_artifacts = self.setup_dir(base_sha, config_sha)
        try:
            create_detached_worktree(repo, commit_sha=base_sha, worktree_path=worktree)
            if setup_commands:
                prepare_workspace(
                    worktree, tuple(setup_commands), environment=environment,
                    artifacts_dir=setup_artifacts, secrets=secrets,
                )
            results = run_checks(
                worktree, config, required_check_ids=[check.id for check in checks],
                logs_dir=logs_dir, secrets=secrets,
                skip={check_id: reason for check_id, reason in skipped.items()},
            )
            entries = tuple(
                baseline_of_result(
                    check.id, result, junit_path=junit_report_path(check, result),
                )
                for check, result in zip(checks, results)
            )
            return BaselineRecord(
                base_sha=base_sha, check_config_sha=config_sha, checks=entries,
                check_ids=tuple(check.id for check in checks),
                fingerprints=tuple((check.id, check_fingerprint(check)) for check in checks),
            )
        except WorkspaceSetupError as exc:
            return unavailable_record(
                base_sha, checks, f"workspace setup did not complete ({exc.code})",
            )
        except ValidationError as exc:
            return unavailable_record(base_sha, checks, f"checks could not run: {exc}")
        finally:
            remove_detached_worktree(repo, worktree)
            shutil.rmtree(scratch, ignore_errors=True)

    def _stamp(self, record: BaselineRecord) -> None:
        """Keep the newest capture of one base commit discoverable."""

        pointer = self.root / f"{record.base_sha}.json"
        if pointer.exists():
            return
        try:
            atomic_write_text(
                pointer,
                json.dumps({
                    "schema_version": record.schema_version,
                    "base_sha": record.base_sha,
                    "check_config_sha": record.check_config_sha,
                    "check_fingerprints": dict(record.fingerprints),
                    "checks": [entry.payload() for entry in record.checks],
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            )
        except (OSError, ResultArtifactError):
            return


def baseline_payload(record: BaselineRecord) -> dict[str, Any]:
    """The bounded durable summary of one baseline, safe for ``state.json``."""

    return {
        "schema_version": record.schema_version,
        "base_sha": record.base_sha,
        "check_config_sha": record.check_config_sha,
        "unavailable_reason": record.unavailable_reason,
        "checks": [entry.payload() for entry in record.checks],
    }


class BaselineError(RuntimeError):
    """The baseline of one run cannot be established."""
