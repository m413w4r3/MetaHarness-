import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.evidence import bounded_semantic_diff, collect_evidence  # noqa: E402
from metaharness.gitops import current_head, index_tree_sha, stage_all  # noqa: E402
from metaharness.models import (  # noqa: E402
    AgentConfig,
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.validation import CheckResult  # noqa: E402


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        shell=False,
    )


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "MetaHarness Tests")
        run_git(self.repo, "config", "user.email", "tests@example.invalid")
        (self.repo / "keep.txt").write_text("keep\n", encoding="utf-8")
        (self.repo / "delete.txt").write_text("delete\n", encoding="utf-8")
        (self.repo / "rename.txt").write_text("rename\n", encoding="utf-8")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", "base")
        self.base_sha = current_head(self.repo)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def command(code: str) -> tuple[str, ...]:
        return (sys.executable, "-c", code)

    def config(self, checks: tuple[CheckConfig, ...] = (), max_diff_bytes: int = 400_000) -> HarnessConfig:
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
        return HarnessConfig(
            repo=self.repo,
            base_ref=self.base_sha,
            runs_root=self.repo.parent / "runs",
            worktrees_root=self.repo.parent / "worktrees",
            require_clean_base=True,
            planner=endpoint,
            reviewer=endpoint,
            context=ContextConfig(),
            agent=AgentConfig(),
            check_catalog=checks,
            max_diff_bytes=max_diff_bytes,
        )

    def test_pass_persists_exact_tree_and_rename_delete_new_files(self) -> None:
        (self.repo / "keep.txt").write_text("changed\n", encoding="utf-8")
        (self.repo / "delete.txt").unlink()
        (self.repo / "rename.txt").rename(self.repo / "renamed.txt")
        (self.repo / "new.txt").write_text("new\n", encoding="utf-8")
        evidence_dir = Path(self.tempdir.name) / "evidence"

        bundle = collect_evidence(
            self.repo,
            self.base_sha,
            self.config((CheckConfig("test", self.command("print('ok')")),)),
            evidence_dir=evidence_dir,
        )

        self.assertTrue(bundle.deterministic_passed)
        self.assertEqual(bundle.staged_tree_sha, index_tree_sha(self.repo))
        self.assertTrue({"delete.txt", "renamed.txt", "new.txt"}.issubset(bundle.changed_files))
        self.assertIn("new.txt", bundle.diff)
        self.assertEqual((evidence_dir / "diff.patch").read_text(), bundle.diff)
        self.assertTrue((evidence_dir / "checks" / "test.stdout.log").exists())
        self.assertIn("new.txt\n", (evidence_dir / "changed-files.txt").read_text().splitlines(keepends=True))

        payload = json.loads((evidence_dir / "evidence.json").read_text())
        self.assertNotIn("stdout_log", payload["checks"][0])
        self.assertEqual(payload["checks"][0]["stdout_tail"], "ok\n")

    def test_fail_timeout_and_required_mutation_close_gate(self) -> None:
        # The gate logic is tested against explicit check results: the real
        # process timeout and mutation detection belong to run_checks and are
        # covered by test_validation, test_procutil and test_subprocess_adversarial.
        marker = self.repo / "mutated.txt"
        checks = (
            CheckConfig("fail", self.command("raise SystemExit(2)")),
            CheckConfig("timeout", self.command("import time; time.sleep(2)"), timeout_seconds=1),
            CheckConfig(
                "mutate",
                self.command(f"from pathlib import Path; Path({str(marker)!r}).write_text('x')"),
            ),
        )

        def result(check: CheckConfig, exit_code: int, *, timed_out: bool = False,
                   mutated: bool = False) -> CheckResult:
            return CheckResult(
                name=check.name, argv=check.argv, cwd=str(self.repo), exit_code=exit_code,
                timed_out=timed_out, duration_seconds=0.0, stdout_log="", stderr_log="",
                stdout_tail="", stderr_tail="", workspace_mutated=mutated,
            )

        def fake_run_checks(*_args, **_kwargs) -> tuple[CheckResult, ...]:
            # The mutating check really changes the worktree before staging.
            marker.write_text("x", encoding="utf-8")
            return (
                result(checks[0], 2),
                result(checks[1], 124, timed_out=True),
                result(checks[2], 0, mutated=True),
            )

        (self.repo / "keep.txt").write_text("change\n", encoding="utf-8")
        with mock.patch("metaharness.evidence.run_checks", side_effect=fake_run_checks):
            bundle = collect_evidence(self.repo, self.base_sha, self.config(checks))
        self.assertFalse(bundle.deterministic_passed)
        self.assertIn("CHECK_FAILED:fail", bundle.failures)
        self.assertIn("CHECK_TIMEOUT:timeout", bundle.failures)
        self.assertIn("CHECK_MUTATED:mutate", bundle.failures)
        self.assertIn("mutated.txt", bundle.changed_files)

    def test_checks_continue_after_failure(self) -> None:
        marker = Path(self.tempdir.name) / "continued"
        checks = (
            CheckConfig("lint", self.command("raise SystemExit(1)")),
            CheckConfig("test", self.command(f"from pathlib import Path; Path({str(marker)!r}).touch()")),
        )
        (self.repo / "keep.txt").write_text("change\n", encoding="utf-8")
        bundle = collect_evidence(self.repo, self.base_sha, self.config(checks))
        self.assertTrue(marker.exists())
        self.assertEqual([check.name for check in bundle.checks], ["lint", "test"])

    def test_empty_and_large_diff_are_rejected_without_truncating_diff(self) -> None:
        empty = collect_evidence(self.repo, self.base_sha, self.config())
        self.assertFalse(empty.deterministic_passed)
        self.assertIn("EMPTY_DIFF", empty.failures)

        content = "A" * 200
        (self.repo / "keep.txt").write_text(content, encoding="utf-8")
        large = collect_evidence(self.repo, self.base_sha, self.config(max_diff_bytes=1))
        self.assertFalse(large.deterministic_passed)
        self.assertIn("DIFF_TOO_LARGE", large.failures)
        self.assertIn(content, large.diff)

    def test_bounded_semantic_diff_is_exactly_bounded_and_covers_all_files(self) -> None:
        diff = "".join(
            f"diff --git a/file-{index}.py b/file-{index}.py\n"
            f"@@ -1 +1 @@\n-old-{index}\n+new-{index}\n"
            for index in range(20)
        )
        excerpt, truncated, full_bytes = bounded_semantic_diff(diff, 1000)

        self.assertTrue(truncated)
        self.assertEqual(full_bytes, len(diff.encode("utf-8")))
        self.assertLessEqual(len(excerpt.encode("utf-8")), 1000)
        self.assertTrue(excerpt.startswith("SEMANTIC DIFF EXCERPT\n"))
        self.assertIn("TRUNCATED: true", excerpt)
        self.assertEqual(
            sum(line.startswith("diff --git ") for line in excerpt.splitlines()), 20
        )

    def test_v2_evidence_keeps_large_diff_as_evidence_without_size_failure(self) -> None:
        content = "A" * 200
        (self.repo / "keep.txt").write_text(content, encoding="utf-8")
        bundle = collect_evidence(
            self.repo, self.base_sha, self.config(max_diff_bytes=1),
            enforce_diff_size=False,
        )

        self.assertTrue(bundle.deterministic_passed)
        self.assertNotIn("DIFF_TOO_LARGE", bundle.failures)
        self.assertIn(content, bundle.diff)

    def test_head_mismatch_and_tree_sha_change_with_index_content(self) -> None:
        stage_all(self.repo)
        base_tree = index_tree_sha(self.repo)
        (self.repo / "keep.txt").write_text("one\n", encoding="utf-8")
        stage_all(self.repo)
        first_tree = index_tree_sha(self.repo)
        self.assertNotEqual(base_tree, first_tree)

        # A second index content must produce a different tree object.
        (self.repo / "keep.txt").write_text("two\n", encoding="utf-8")
        stage_all(self.repo)
        second_tree = index_tree_sha(self.repo)
        self.assertNotEqual(first_tree, second_tree)

        run_git(self.repo, "commit", "-m", "unexpected")
        bundle = collect_evidence(self.repo, self.base_sha, self.config())
        self.assertFalse(bundle.deterministic_passed)
        self.assertIn("HEAD_MISMATCH", bundle.failures)

    def test_secret_in_diff_suppressed_by_gitattributes_is_found_in_staged_blob(self) -> None:
        secret = "sk-staged-secret-012345"
        (self.repo / ".gitattributes").write_text("*.py -diff\n", encoding="utf-8")
        (self.repo / "secret.py").write_text(f"TOKEN = {secret!r}\n", encoding="utf-8")

        bundle = collect_evidence(
            self.repo,
            self.base_sha,
            self.config(),
            secrets=(secret,),
        )

        self.assertNotIn(secret, bundle.diff)
        self.assertFalse(bundle.deterministic_passed)
        self.assertIn("SECRET_IN_STAGED_BLOB:secret.py", bundle.failures)

    def test_normal_text_blob_secret_and_clean_blob(self) -> None:
        secret = "sk-normal-secret-012345"
        (self.repo / "secret.txt").write_text(secret, encoding="utf-8")
        secret_bundle = collect_evidence(
            self.repo, self.base_sha, self.config(), secrets=(secret,)
        )
        self.assertIn("SECRET_IN_STAGED_BLOB:secret.txt", secret_bundle.failures)

        clean = self.repo / "secret.txt"
        clean.write_text("safe\n", encoding="utf-8")
        clean_bundle = collect_evidence(
            self.repo, self.base_sha, self.config(), secrets=(secret,)
        )
        self.assertTrue(clean_bundle.deterministic_passed)

    def test_large_changed_blob_fails_closed_without_reading_secret(self) -> None:
        from unittest.mock import patch

        (self.repo / "large.txt").write_text("0123456789", encoding="utf-8")
        with patch("metaharness.evidence.MAX_SECRET_SCAN_BLOB_BYTES", 4):
            bundle = collect_evidence(self.repo, self.base_sha, self.config())

        self.assertFalse(bundle.deterministic_passed)
        self.assertIn("UNSCANNABLE_STAGED_BLOB:large.txt", bundle.failures)

    def test_binary_only_source_diff_is_not_reviewable_but_binary_asset_is_allowed(self) -> None:
        (self.repo / ".gitattributes").write_text("*.py -diff\n", encoding="utf-8")
        (self.repo / "source.py").write_text("print('changed')\n", encoding="utf-8")
        (self.repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01")

        bundle = collect_evidence(self.repo, self.base_sha, self.config())

        self.assertFalse(bundle.deterministic_passed)
        self.assertIn("UNREVIEWABLE_TEXT_DIFF:source.py", bundle.failures)
        self.assertNotIn("UNREVIEWABLE_TEXT_DIFF:image.png", bundle.failures)

    def test_deleted_file_does_not_trigger_blob_read(self) -> None:
        (self.repo / "delete.txt").unlink()
        bundle = collect_evidence(self.repo, self.base_sha, self.config(), secrets=("unused-secret",))
        self.assertTrue(bundle.deterministic_passed)


if __name__ == "__main__":
    unittest.main()
