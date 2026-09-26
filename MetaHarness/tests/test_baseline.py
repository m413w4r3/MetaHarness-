"""The check baseline: what it records, how it compares, and when it is reused.

The comparison is the whole point of the gate: a candidate only has to answer
for the failures the base commit did *not* already have.  These tests exercise
the real runner, the real detached worktree capture and the real on-disk cache,
with tiny Python checks standing in for a project's test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.baseline import (  # noqa: E402
    BaselineCache,
    CheckBaseline,
    CheckVerdict,
    STATUS_FAIL,
    STATUS_PASS,
    baseline_of_result,
    check_config_sha,
    compare_check,
    judge_results,
    parse_failure_ids,
    parse_junit_ids,
)
from metaharness.models import CheckConfig, ContextConfig, HarnessConfig  # noqa: E402
from metaharness.validation import run_checks  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def failing_report(*ids: str, marker: str = "FAILED") -> str:
    """A check whose output is a real pytest short summary."""

    joined = "\n".join(f"{marker} {item}" for item in ids)
    return (
        "import sys\n"
        f"sys.stdout.write({joined!r} + '\\n')\n"
        "raise SystemExit(1)\n"
    )


class BaselineTestCase(unittest.TestCase):
    """A real repository, real checks, and a throwaway runs root."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "MetaHarness baseline tests")
        git(self.repo, "config", "user.email", "baseline@example.invalid")
        (self.repo / "feature.txt").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.marker = self.root / "captures.txt"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def check(self, check_id: str = "test", code: str = "raise SystemExit(0)") -> CheckConfig:
        return CheckConfig(check_id, (sys.executable, "-c", code), timeout_seconds=60)

    def config(self, *checks: CheckConfig) -> HarnessConfig:
        return HarnessConfig(
            repo=self.repo,
            base_ref="main",
            runs_root=self.root / "runs",
            worktrees_root=self.root / "worktrees",
            require_clean_base=True,
            context=ContextConfig(),
            check_catalog=tuple(checks),
            max_diff_bytes=400_000,
        )

    def result(self, check_id: str = "test", code: str = "raise SystemExit(0)") -> CheckBaseline:
        check = self.check(check_id, code)
        run = run_checks(
            self.repo, self.config(check), required_check_ids=(check_id,),
        )[0]
        return baseline_of_result(check_id, run)

    def counter_code(self, exit_code: int) -> str:
        """A check that records its own invocations before failing or passing."""

        return (
            "import pathlib\n"
            f"marker = pathlib.Path({str(self.marker)!r})\n"
            "count = int(marker.read_text()) if marker.exists() else 0\n"
            "marker.write_text(str(count + 1))\n"
            f"raise SystemExit({exit_code})\n"
        )


class ComparisonTests(BaselineTestCase):
    def test_baseline_green_candidate_green(self) -> None:
        baseline = self.result()
        candidate = self.result()

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.PASS)
        self.assertFalse(judgement.blocking)
        self.assertEqual(judgement.new_failure_ids, ())

    def test_baseline_green_candidate_red_is_regression(self) -> None:
        baseline = self.result()
        candidate = self.result(code=failing_report("tests/test_x.py::test_y"))

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.REGRESSION)
        self.assertTrue(judgement.blocking)
        self.assertEqual(judgement.new_failure_ids, ("tests/test_x.py::test_y",))
        self.assertEqual(judgement.failure, "CHECK_FAILED:test")

    def test_same_baseline_failure_is_not_regression(self) -> None:
        baseline = self.result(code=failing_report("tests/test_x.py::test_y"))
        candidate = self.result(code=failing_report("tests/test_x.py::test_y"))

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.BASELINE_WARNING)
        self.assertFalse(judgement.blocking)
        self.assertEqual(judgement.new_failure_ids, ())
        self.assertIn("baseline", judgement.warning or "")

    def test_subset_of_baseline_failures_is_not_regression(self) -> None:
        baseline = self.result(code=failing_report(
            "tests/test_x.py::test_y", "tests/test_x.py::test_z",
        ))
        candidate = self.result(code=failing_report("tests/test_x.py::test_z"))

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.BASELINE_WARNING)
        self.assertFalse(judgement.blocking)

    def test_new_failure_id_is_regression(self) -> None:
        baseline = self.result(code=failing_report("tests/test_x.py::test_y"))
        candidate = self.result(code=failing_report(
            "tests/test_x.py::test_y", "tests/test_x.py::test_new",
        ))

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.REGRESSION)
        self.assertTrue(judgement.blocking)
        self.assertEqual(judgement.new_failure_ids, ("tests/test_x.py::test_new",))

    def test_unparseable_baseline_red_is_warning(self) -> None:
        opaque = "import sys\nsys.stdout.write('the compiler exploded\\n')\nraise SystemExit(1)\n"
        baseline = self.result(code=opaque)
        candidate = self.result(code=opaque)

        self.assertEqual(baseline.status, STATUS_FAIL)
        self.assertFalse(baseline.failure_ids_parsed)
        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.BASELINE_RED)
        self.assertFalse(judgement.blocking)
        self.assertIn("not blocking", judgement.warning or "")

    def test_an_unparsed_candidate_fails_closed(self) -> None:
        baseline = self.result(code=failing_report("tests/test_x.py::test_y"))
        opaque = "import sys\nsys.stdout.write('opaque\\n')\nraise SystemExit(1)\n"
        candidate = self.result(code=opaque)

        judgement = compare_check("test", candidate, baseline)

        self.assertEqual(judgement.verdict, CheckVerdict.REGRESSION)
        self.assertTrue(judgement.blocking)

    def test_judge_results_keeps_skipped_checks_out_of_the_comparison(self) -> None:
        check = self.check("integration")
        candidate = CheckBaseline("integration", STATUS_PASS, 0)

        judgements = judge_results(
            None, (candidate,), (check,), skipped={"integration": "preflight failed"},
        )

        self.assertEqual(judgements[0].verdict, CheckVerdict.SKIPPED_INFRA)
        self.assertFalse(judgements[0].blocking)

    def test_missing_baseline_fails_closed(self) -> None:
        candidate = CheckBaseline("test", STATUS_FAIL, 1, ("tests/test_x.py::test_y",), True)

        judgement = compare_check("test", candidate, None)

        self.assertEqual(judgement.verdict, CheckVerdict.REGRESSION)
        self.assertTrue(judgement.blocking)


class ParserTests(unittest.TestCase):
    def test_pytest_ids_ignore_the_assertion_tail(self) -> None:
        parsed, ids = parse_failure_ids(
            "FAILED tests/test_x.py::test_y - AssertionError: nope\n"
            "ERROR tests/test_z.py::test_w\n"
            "1 failed, 1 error in 0.12s\n"
        )

        self.assertTrue(parsed)
        self.assertEqual(ids, ("tests/test_x.py::test_y", "tests/test_z.py::test_w"))

    def test_vitest_ids_are_extracted_best_effort(self) -> None:
        parsed, ids = parse_failure_ids(
            " ❯ tests/math.test.ts (2 tests | 1 failed)\n"
            "   × adds two numbers 12ms\n"
            " Test Files  1 failed (1)\n"
        )

        self.assertTrue(parsed)
        self.assertIn("tests/math.test.ts::adds two numbers", ids)

    def test_jest_ids_are_extracted_best_effort(self) -> None:
        parsed, ids = parse_failure_ids(
            " FAIL  src/sum.test.js\n"
            "  ● sum > adds numbers\n"
            "Tests:       1 failed, 1 total\n"
        )

        self.assertTrue(parsed)
        self.assertIn("sum > adds numbers", ids)

    def test_opaque_output_is_reported_as_unparsed(self) -> None:
        parsed, ids = parse_failure_ids("make: *** [Makefile:3: test] Error 2\n")

        self.assertFalse(parsed)
        self.assertEqual(ids, ())

    def test_junit_report_is_preferred_when_it_exists(self) -> None:
        report = Path(tempfile.mkdtemp()) / "report.xml"
        report.write_text(
            "<testsuites><testsuite><testcase classname='tests.test_x' name='test_y'>"
            "<failure message='nope'/></testcase><testcase classname='tests.test_x' "
            "name='test_ok'/></testsuite></testsuites>",
            encoding="utf-8",
        )

        parsed, ids = parse_junit_ids(report)

        self.assertTrue(parsed)
        self.assertEqual(ids, ("tests.test_x::test_y",))


class CacheTests(BaselineTestCase):
    def cache(self) -> BaselineCache:
        cache = BaselineCache(self.root / "runs")
        cache.root.mkdir(parents=True, exist_ok=True)
        return cache

    def test_baseline_cache_reused_for_same_base_and_check_config(self) -> None:
        check = self.check(code=self.counter_code(0))
        config = self.config(check)
        cache = self.cache()

        first = cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=config,
            check_ids=("test",), environment={},
        )
        second = cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=config,
            check_ids=("test",), environment={},
        )

        self.assertEqual(first.check_config_sha, second.check_config_sha)
        self.assertEqual(self.marker.read_text(), "1", "the base was captured twice")
        path = cache.record_path(self.base_sha, check_config_sha((check,)))
        self.assertTrue(path.is_file())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["base_sha"], self.base_sha)
        self.assertEqual(payload["checks"][0]["status"], STATUS_PASS)
        # The newest capture of one base commit stays discoverable, and a
        # second identical publication is idempotent.
        self.assertTrue((cache.root / f"{self.base_sha}.json").is_file())
        before = path.read_bytes()
        cache.store(first)
        self.assertEqual(path.read_bytes(), before)

    def test_changed_check_config_invalidates_baseline_cache(self) -> None:
        first_check = self.check(code=self.counter_code(0))
        cache = self.cache()
        cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=self.config(first_check),
            check_ids=("test",), environment={},
        )
        changed_check = self.check(code=self.counter_code(0) + "import os\n")
        changed_config = self.config(changed_check)

        record = cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=changed_config,
            check_ids=("test",), environment={},
        )

        changed_sha = check_config_sha((changed_check,))
        self.assertNotEqual(changed_sha, check_config_sha((first_check,)))
        self.assertEqual(record.check_config_sha, changed_sha)
        self.assertEqual(self.marker.read_text(), "2", "the changed check was not recaptured")
        self.assertTrue(cache.record_path(self.base_sha, changed_sha).is_file())

    def test_a_captured_superset_serves_a_narrower_selection(self) -> None:
        wide = self.config(self.check("test", self.counter_code(0)), self.check("lint"))
        cache = self.cache()
        cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=wide,
            check_ids=("test", "lint"), environment={},
        )

        narrow = cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=wide,
            check_ids=("lint",), environment={},
        )

        self.assertEqual(narrow.check_ids, ("lint",))
        self.assertEqual(self.marker.read_text(), "1")

    def test_corrupt_baseline_cache_is_rebuilt(self) -> None:
        check = self.check(code=self.counter_code(0))
        config = self.config(check)
        cache = self.cache()
        cache.record_path(self.base_sha, check_config_sha((check,))).write_text(
            "{not json", encoding="utf-8",
        )

        record = cache.ensure(
            repo=self.repo, base_sha=self.base_sha, config=config,
            check_ids=("test",), environment={},
        )

        self.assertEqual(record.entry("test").status, STATUS_PASS)
        self.assertEqual(self.marker.read_text(), "1")

    def test_an_uncapturable_base_is_recorded_as_unavailable(self) -> None:
        check = self.check(code=self.counter_code(0))
        cache = self.cache()

        record = cache.ensure(
            repo=self.repo, base_sha="0" * 40, config=self.config(check),
            check_ids=("test",), environment={},
        )

        self.assertEqual(record.entry("test").status, "UNAVAILABLE")
        self.assertTrue(record.unavailable_reason)
        self.assertFalse((self.root / "captures.txt").exists())


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
