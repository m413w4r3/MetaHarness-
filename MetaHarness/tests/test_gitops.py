import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.gitops import (  # noqa: E402
    GitError,
    assert_agent_did_not_commit,
    assert_clean,
    branch_exists,
    commit_staged,
    create_run_worktree,
    current_head,
    git_root,
    index_tree_sha,
    read_file_at_commit,
    resolve_commit,
    stage_all,
    staged_changed_files,
    staged_diff,
    status_porcelain,
)


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        shell=False,
    )


class GitOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.repo = self.root / "source repo with spaces"
        self.repo.mkdir()
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "MetaHarness Tests")
        run_git(self.repo, "config", "user.email", "tests@example.invalid")

        (self.repo / "README.md").write_text("base contents\n", encoding="utf-8")
        (self.repo / "delete me.txt").write_text("remove me\n", encoding="utf-8")
        (self.repo / "rename me.txt").write_text("rename me\n", encoding="utf-8")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", "initial commit")
        self.initial_sha = current_head(self.repo)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_root_base_ref_clean_status_and_branch_lookup(self) -> None:
        self.assertEqual(git_root(self.repo), self.repo.resolve())
        self.assertEqual(resolve_commit(self.repo, "HEAD"), self.initial_sha)
        self.assertEqual(current_head(self.repo), self.initial_sha)
        self.assertEqual(status_porcelain(self.repo), ())
        assert_clean(self.repo)

        branch = run_git(self.repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
        self.assertTrue(branch_exists(self.repo, branch))
        self.assertFalse(branch_exists(self.repo, "does-not-exist"))

        (self.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        self.assertTrue(status_porcelain(self.repo))
        with self.assertRaises(GitError):
            assert_clean(self.repo)

    def test_create_worktree_checks_clean_base_collisions_and_head(self) -> None:
        dirty_worktree = self.root / "dirty worktree"
        (self.repo / "uncommitted.txt").write_text("not committed\n", encoding="utf-8")
        with self.assertRaises(GitError):
            create_run_worktree(
                self.repo,
                base_ref="HEAD",
                branch="run-dirty",
                worktree_path=dirty_worktree,
                require_clean_base=True,
            )
        self.assertFalse(dirty_worktree.exists())
        (self.repo / "uncommitted.txt").unlink()

        worktree = self.root / "run worktree with spaces"
        info = create_run_worktree(
            self.repo,
            base_ref="HEAD",
            branch="metaharness/run-1",
            worktree_path=worktree,
            require_clean_base=True,
        )
        self.assertEqual(info.source_repo, self.repo.resolve())
        self.assertEqual(info.worktree, worktree.resolve())
        self.assertEqual(info.base_ref, "HEAD")
        self.assertEqual(info.base_sha, self.initial_sha)
        self.assertEqual(current_head(worktree), self.initial_sha)
        assert_agent_did_not_commit(info)

        collision_path = self.root / "existing path"
        collision_path.mkdir()
        with self.assertRaises(GitError):
            create_run_worktree(
                self.repo,
                base_ref=self.initial_sha,
                branch="metaharness/run-2",
                worktree_path=collision_path,
                require_clean_base=False,
            )

        with self.assertRaises(GitError):
            create_run_worktree(
                self.repo,
                base_ref=self.initial_sha,
                branch="metaharness/run-1",
                worktree_path=self.root / "another worktree",
                require_clean_base=False,
            )

    def test_stage_diff_changed_files_tree_and_harness_commit(self) -> None:
        worktree = self.root / "run worktree"
        info = create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/run-stage",
            worktree_path=worktree,
            require_clean_base=True,
        )

        (worktree / "README.md").write_text("changed\n", encoding="utf-8")
        (worktree / "new file.txt").write_text("new\n", encoding="utf-8")
        (worktree / "delete me.txt").unlink()
        (worktree / "rename me.txt").rename(worktree / "renamed file.txt")

        stage_all(worktree)
        changed = staged_changed_files(worktree)
        self.assertIn("README.md", changed)
        self.assertIn("new file.txt", changed)
        self.assertIn("delete me.txt", changed)
        self.assertTrue({"rename me.txt", "renamed file.txt"}.intersection(changed))

        diff = staged_diff(worktree)
        self.assertIn("changed", diff)
        self.assertIn("new file.txt", diff)
        self.assertIn("delete me.txt", diff)
        self.assertIn("renamed file.txt", diff)

        tree_sha = index_tree_sha(worktree)
        self.assertEqual(tree_sha, index_tree_sha(worktree))
        commit_sha = commit_staged(
            worktree,
            subject="  harness commit  ",
            body="The harness owns this commit.",
        )
        self.assertEqual(commit_sha, current_head(worktree))
        self.assertNotEqual(commit_sha, info.base_sha)
        self.assertEqual(
            run_git(worktree, "show", "-s", "--format=%s", commit_sha).stdout.strip(),
            "harness commit",
        )
        self.assertEqual(status_porcelain(worktree), ())

    def test_commit_staged_rejects_empty_and_long_subjects(self) -> None:
        worktree = self.root / "run empty commit"
        create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/run-empty",
            worktree_path=worktree,
            require_clean_base=True,
        )
        with self.assertRaises(GitError):
            commit_staged(worktree, subject="subject", body="body")

        (worktree / "README.md").write_text("change\n", encoding="utf-8")
        stage_all(worktree)
        with self.assertRaises(GitError):
            commit_staged(worktree, subject="x" * 73, body="")
        with self.assertRaises(GitError):
            commit_staged(worktree, subject="   ", body="")

    def test_agent_commit_is_detected(self) -> None:
        worktree = self.root / "run agent commit"
        info = create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/run-agent",
            worktree_path=worktree,
            require_clean_base=True,
        )
        (worktree / "README.md").write_text("agent change\n", encoding="utf-8")
        run_git(worktree, "add", "--all")
        run_git(worktree, "commit", "-m", "agent commit")
        with self.assertRaises(GitError):
            assert_agent_did_not_commit(info)

    def test_read_file_at_exact_commit_and_rejects_traversal(self) -> None:
        self.assertEqual(
            read_file_at_commit(
                self.repo,
                commit_sha=self.initial_sha,
                relative_path="README.md",
            ),
            "base contents\n",
        )
        with self.assertRaises(GitError):
            read_file_at_commit(
                self.repo,
                commit_sha=self.initial_sha,
                relative_path="../secret",
            )
        with self.assertRaises(GitError):
            read_file_at_commit(
                self.repo,
                commit_sha=self.initial_sha,
                relative_path="",
            )
        with self.assertRaises(GitError):
            read_file_at_commit(
                self.repo,
                commit_sha=self.initial_sha,
                relative_path="/etc/passwd",
            )
        with self.assertRaises(GitError):
            read_file_at_commit(
                self.repo,
                commit_sha=self.initial_sha,
                relative_path="safe\x00path",
            )


if __name__ == "__main__":
    unittest.main()
