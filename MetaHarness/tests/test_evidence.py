import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.evidence import collect_evidence  # noqa: E402
from metaharness.gitops import current_head, index_tree_sha, stage_all  # noqa: E402
from metaharness.models import (  # noqa: E402
    AgentConfig,
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)


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
            checks=checks,
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
        marker = self.repo / "mutated.txt"
        checks = (
            CheckConfig("fail", self.command("raise SystemExit(2)")),
            CheckConfig("timeout", self.command("import time; time.sleep(2)"), timeout_seconds=1),
            CheckConfig(
                "mutate",
                self.command(f"from pathlib import Path; Path({str(marker)!r}).write_text('x')"),
            ),
        )
        (self.repo / "keep.txt").write_text("change\n", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
