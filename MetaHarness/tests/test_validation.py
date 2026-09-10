import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import (  # noqa: E402
    AgentConfig,
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.validation import (  # noqa: E402
    ValidationError,
    resolve_check_cwd,
    run_checks,
)


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        shell=False,
    )


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.repo = root / "repo with spaces"
        self.repo.mkdir()
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "MetaHarness Tests")
        run_git(self.repo, "config", "user.email", "tests@example.invalid")
        (self.repo / "tracked.txt").write_text("base\n", encoding="utf-8")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", "base")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def config(self, checks: tuple[CheckConfig, ...]) -> HarnessConfig:
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
        return HarnessConfig(
            repo=self.repo,
            base_ref="HEAD",
            runs_root=self.repo.parent / "runs",
            worktrees_root=self.repo.parent / "worktrees",
            require_clean_base=True,
            planner=endpoint,
            reviewer=endpoint,
            context=ContextConfig(),
            agent=AgentConfig(),
            checks=checks,
            max_diff_bytes=400_000,
        )

    @staticmethod
    def command(code: str) -> tuple[str, ...]:
        return (sys.executable, "-c", code)

    def test_success_failure_and_all_checks_continue(self) -> None:
        marker = Path(self.tempdir.name) / "second-ran"
        checks = (
            CheckConfig("lint", self.command("print('lint output'); raise SystemExit(3)")),
            CheckConfig(
                "typecheck",
                self.command(
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
                ),
            ),
        )
        results = run_checks(self.repo, self.config(checks))
        self.assertEqual([result.exit_code for result in results], [3, 0])
        self.assertTrue(marker.exists())
        self.assertEqual(results[0].stdout_log, "lint output\n")
        self.assertFalse(results[0].workspace_mutated)

    def test_timeout_is_reported(self) -> None:
        check = CheckConfig(
            "test",
            self.command("import time; time.sleep(2)"),
            timeout_seconds=1,
        )
        result = run_checks(self.repo, self.config((check,)))[0]
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)

    def test_mutation_and_new_file_are_detected(self) -> None:
        check = CheckConfig(
            "mutator",
            self.command(
                "from pathlib import Path; Path('created.txt').write_text('created')"
            ),
        )
        result = run_checks(self.repo, self.config((check,)))[0]
        self.assertTrue(result.workspace_mutated)

    def test_cwd_rejects_parent_and_symlink_escape(self) -> None:
        outside = Path(self.tempdir.name) / "outside"
        outside.mkdir()
        link = self.repo / "outside-link"
        link.symlink_to(outside, target_is_directory=True)
        for cwd in ("../outside", "outside-link"):
            with self.subTest(cwd=cwd):
                with self.assertRaises(ValidationError):
                    resolve_check_cwd(self.repo, CheckConfig("unsafe", ("true",), cwd))

    def test_cwd_inside_symlink_and_parent_normalization_are_allowed(self) -> None:
        nested = self.repo / "nested"
        nested.mkdir()
        (nested / "alias").symlink_to(nested, target_is_directory=True)
        result = resolve_check_cwd(self.repo, CheckConfig("safe", ("true",), "nested/alias"))
        self.assertEqual(result, nested.resolve())

    def test_full_logs_are_written_and_tails_are_bounded(self) -> None:
        output = Path(self.tempdir.name) / "evidence"
        check = CheckConfig(
            "logs",
            self.command("print('x' * 10000); print('error' * 100, file=__import__('sys').stderr)"),
        )
        results = run_checks(self.repo, self.config((check,)), logs_dir=output / "checks", tail_bytes=32)
        result = results[0]
        self.assertGreater(len(result.stdout_log), len(result.stdout_tail))
        self.assertLessEqual(len(result.stdout_tail.encode()), 32)
        self.assertEqual((output / "checks" / "logs.stdout.log").read_text(), result.stdout_log)
        self.assertEqual((output / "checks" / "logs.stderr.log").read_text(), result.stderr_log)


if __name__ == "__main__":
    unittest.main()
