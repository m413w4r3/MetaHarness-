import dataclasses
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import (  # noqa: E402
    AgentConfig,
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.approval import write_check_authority  # noqa: E402
from metaharness.validation import (  # noqa: E402
    ValidationError,
    config_with_check_authority,
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


class ValidationTestBase(unittest.TestCase):
    """Shared repository fixture and helpers; deliberately holds no tests."""

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


class ValidationTests(ValidationTestBase):
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
        # Only the mapping of a bounded-process timeout onto the check result
        # is tested here.  The real OS deadline is covered by test_procutil and
        # by test_subprocess_adversarial (run_checks with a hanging check).
        check = CheckConfig(
            "test",
            self.command("import time; time.sleep(2)"),
            timeout_seconds=1,
        )
        with mock.patch(
            "metaharness.validation.run_bounded", return_value=(-15, True),
        ) as bounded:
            result = run_checks(self.repo, self.config((check,)))[0]
        self.assertEqual(bounded.call_args.kwargs["timeout_seconds"], 1)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)
        self.assertIn("check timed out after 1s", result.stderr_log)

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


class CheckAuthorityTests(ValidationTestBase):
    """A run owning a check authority never reads argv from today's TOML."""

    CATALOGUE = (
        CheckConfig("lint", ("frozen-lint",), timeout_seconds=11),
        CheckConfig("test", ("frozen-test",), timeout_seconds=12),
        CheckConfig("integration", ("frozen-integration",), timeout_seconds=13),
    )

    def authority_run(self, name: str = "run") -> Path:
        directory = Path(self.tempdir.name) / name
        directory.mkdir(parents=True, exist_ok=True)
        write_check_authority(
            directory, self.CATALOGUE, required_check_ids=("lint", "test"),
        )
        return directory

    def current_config(self) -> HarnessConfig:
        """Today's configuration, with every command deliberately changed."""

        return dataclasses.replace(
            self.config(()),
            check_catalog=tuple(
                CheckConfig(check.id, ("CHANGED", check.id), "src", timeout_seconds=99,
                            preflight_argv=("CHANGED-preflight",))
                for check in self.CATALOGUE
            ),
            default_check_ids=("lint", "test"),
        )

    def test_selection_and_catalogue_come_from_the_authority(self) -> None:
        run_dir = self.authority_run()
        frozen, ids = config_with_check_authority(self.current_config(), run_dir)
        self.assertEqual(ids, ("lint", "test"))
        self.assertEqual([check.argv for check in frozen.select_checks(ids)],
                         [("frozen-lint",), ("frozen-test",)])
        self.assertEqual([check.id for check in frozen.trusted_checks()],
                         ["lint", "test", "integration"])

    def test_a_requested_check_outside_c01_uses_the_frozen_command(self) -> None:
        run_dir = self.authority_run()
        frozen, ids = config_with_check_authority(
            self.current_config(), run_dir, requested_check_ids=("test", "integration"),
        )
        self.assertEqual(ids, ("test", "integration"))
        selected = frozen.select_checks(ids)
        self.assertEqual([check.argv for check in selected],
                         [("frozen-test",), ("frozen-integration",)])
        # Nothing of the current TOML's command surface survives.
        self.assertEqual([check.cwd for check in selected], [".", "."])
        self.assertEqual([check.timeout_seconds for check in selected], [12, 13])
        self.assertEqual([check.preflight_argv for check in selected], [(), ()])

    def test_a_requested_check_absent_from_the_authority_fails_closed(self) -> None:
        run_dir = self.authority_run()
        config = dataclasses.replace(
            self.current_config(),
            check_catalog=self.current_config().check_catalog + (
                CheckConfig("late", ("CHANGED", "late")),
            ),
            default_check_ids=("lint", "test"),
        )
        with self.assertRaises(ValidationError) as caught:
            config_with_check_authority(config, run_dir, requested_check_ids=("late",))
        self.assertIn("late", str(caught.exception))

    def test_a_requested_check_untrusted_today_fails_closed(self) -> None:
        run_dir = self.authority_run()
        config = dataclasses.replace(
            self.current_config(),
            check_catalog=tuple(
                check for check in self.current_config().check_catalog
                if check.id != "integration"
            ),
        )
        with self.assertRaises(ValidationError):
            config_with_check_authority(config, run_dir, requested_check_ids=("integration",))

    def test_a_rewritten_authority_is_rejected_against_the_approved_hash(self) -> None:
        run_dir = self.authority_run()
        approved = hashlib.sha256(
            (run_dir / "check_authority.json").read_bytes()
        ).hexdigest()
        config = self.current_config()
        # The unchanged bytes still verify.
        config_with_check_authority(config, run_dir, expected_sha256=approved)
        (run_dir / "check_authority.json").unlink()
        write_check_authority(
            run_dir,
            tuple(dataclasses.replace(check, argv=("attacker", check.id))
                  for check in self.CATALOGUE),
            required_check_ids=("lint", "test"),
        )
        with self.assertRaises(ValidationError):
            config_with_check_authority(config, run_dir, expected_sha256=approved)

    def test_a_run_without_an_authority_keeps_the_legacy_behavior(self) -> None:
        directory = Path(self.tempdir.name) / "legacy"
        directory.mkdir()
        config = self.config((CheckConfig("lint", ("make", "lint")),))
        frozen, ids = config_with_check_authority(
            config, directory, requested_check_ids=("lint",),
        )
        self.assertIs(frozen, config)
        self.assertEqual(ids, ("lint",))


if __name__ == "__main__":
    unittest.main()
