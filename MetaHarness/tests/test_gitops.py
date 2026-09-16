import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.gitops import (  # noqa: E402
    GitError,
    RepositoryReference,
    assert_agent_did_not_commit,
    assert_clean,
    branch_exists,
    candidate_tree_sha,
    commit_reviewed_tree,
    create_run_worktree,
    current_head,
    immutable_commit_web_url,
    compare_commits_web_url,
    delete_run_branch,
    git_root,
    index_tree_sha,
    local_branches,
    read_file_at_commit,
    resolve_commit,
    remote_run_branch_tip,
    stage_all,
    staged_changed_blobs,
    staged_changed_files,
    staged_changes,
    staged_diff,
    status_porcelain,
    symbolic_head,
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

    def test_immutable_commit_and_compare_urls_use_exact_shas(self) -> None:
        reference = RepositoryReference(
            remote_name="origin",
            web_url="https://github.com/acme/project",
            base_sha="a" * 40,
            immutable_url="https://github.com/acme/project/tree/" + "a" * 40,
        )

        self.assertEqual(
            immutable_commit_web_url(reference, "b" * 40),
            "https://github.com/acme/project/tree/" + "b" * 40,
        )
        self.assertEqual(
            compare_commits_web_url(reference, "a" * 40, "b" * 40),
            "https://github.com/acme/project/compare/"
            + "a" * 40
            + "..."
            + "b" * 40,
        )

    def test_immutable_urls_are_unavailable_without_github_exploration(self) -> None:
        reference = RepositoryReference("origin", None, "a" * 40, None)
        self.assertIsNone(immutable_commit_web_url(reference, "b" * 40))
        self.assertIsNone(compare_commits_web_url(reference, "a" * 40, "b" * 40))

    def test_immutable_urls_reject_invalid_shas(self) -> None:
        reference = RepositoryReference("origin", "https://github.com/acme/project", "a" * 40, None)
        for invalid in ("HEAD", "", "abc", "z" * 40):
            with self.subTest(invalid=invalid):
                with self.assertRaises(GitError):
                    immutable_commit_web_url(reference, invalid)
                with self.assertRaises(GitError):
                    compare_commits_web_url(reference, invalid, "b" * 40)
                with self.assertRaises(GitError):
                    compare_commits_web_url(reference, "a" * 40, invalid)

    def test_delete_run_branch_is_idempotent_and_exact(self) -> None:
        bare = self.root / "origin.git"
        run_git(self.repo, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "origin", str(bare))
        branch = "harness/test/run"

        absent = delete_run_branch(
            self.repo, remote="origin", branch=branch, expected_commit_sha=self.initial_sha,
        )
        self.assertEqual(absent.status, "already_absent")

        run_git(self.repo, "branch", branch, self.initial_sha)
        run_git(self.repo, "push", "-q", "origin", f"{branch}:{branch}")
        deleted = delete_run_branch(
            self.repo, remote="origin", branch=branch, expected_commit_sha=self.initial_sha,
        )
        self.assertEqual(deleted.status, "success")
        self.assertIsNone(remote_run_branch_tip(self.repo, remote="origin", branch=branch))

    def test_delete_run_branch_keeps_a_moved_remote_branch(self) -> None:
        bare = self.root / "origin-moved.git"
        run_git(self.repo, "init", "--bare", str(bare))
        run_git(self.repo, "remote", "add", "moved", str(bare))
        branch = "harness/test/moved"
        run_git(self.repo, "branch", branch, self.initial_sha)
        run_git(self.repo, "push", "-q", "moved", f"{branch}:{branch}")
        (self.repo / "README.md").write_text("moved contents\n", encoding="utf-8")
        run_git(self.repo, "add", "README.md")
        run_git(self.repo, "commit", "-qm", "moved")
        moved_sha = current_head(self.repo)
        run_git(self.repo, "push", "-q", "moved", f"{moved_sha}:refs/heads/{branch}")
        with self.assertRaisesRegex(GitError, "no longer points"):
            delete_run_branch(
                self.repo, remote="moved", branch=branch, expected_commit_sha=self.initial_sha,
            )

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
        commit_sha = commit_reviewed_tree(
            worktree,
            tree_sha=tree_sha,
            parent_sha=info.base_sha,
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

    def test_commit_reviewed_tree_rejects_empty_tree_and_bad_subjects(self) -> None:
        worktree = self.root / "run empty commit"
        info = create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/run-empty",
            worktree_path=worktree,
            require_clean_base=True,
        )
        base_tree = index_tree_sha(worktree)
        with self.assertRaisesRegex(GitError, "no changes"):
            commit_reviewed_tree(
                worktree, tree_sha=base_tree, parent_sha=info.base_sha, subject="s", body=""
            )

        (worktree / "README.md").write_text("change\n", encoding="utf-8")
        stage_all(worktree)
        tree = index_tree_sha(worktree)
        for subject in ("x" * 73, "   ", "two\nlines"):
            with self.subTest(subject=subject):
                with self.assertRaises(GitError):
                    commit_reviewed_tree(
                        worktree, tree_sha=tree, parent_sha=info.base_sha, subject=subject, body=""
                    )
        with self.assertRaises(GitError):
            commit_reviewed_tree(
                worktree, tree_sha="HEAD", parent_sha=info.base_sha, subject="s", body=""
            )
        self.assertEqual(current_head(worktree), info.base_sha)

    def _staged_worktree(self, name: str):
        worktree = self.root / name
        info = create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch=f"metaharness/{name.replace(' ', '-')}",
            worktree_path=worktree,
            require_clean_base=True,
        )
        (worktree / "README.md").write_text("reviewed\n", encoding="utf-8")
        stage_all(worktree)
        return worktree, info, index_tree_sha(worktree)

    def test_commit_reviewed_tree_commits_the_reviewed_tree_not_the_index(self) -> None:
        worktree, info, reviewed_tree = self._staged_worktree("exact tree")
        (worktree / "README.md").write_text("tampered after review\n", encoding="utf-8")
        stage_all(worktree)
        self.assertNotEqual(index_tree_sha(worktree), reviewed_tree)

        commit_sha = commit_reviewed_tree(
            worktree, tree_sha=reviewed_tree, parent_sha=info.base_sha, subject="s", body=""
        )

        self.assertEqual(
            run_git(worktree, "rev-parse", f"{commit_sha}^{{tree}}").stdout.strip(), reviewed_tree
        )
        self.assertEqual(
            run_git(worktree, "show", f"{commit_sha}:README.md").stdout, "reviewed\n"
        )

    def test_commit_reviewed_tree_refuses_when_head_moved(self) -> None:
        worktree, info, reviewed_tree = self._staged_worktree("moved head")
        run_git(worktree, "commit", "-qm", "someone else")
        moved = current_head(worktree)

        with self.assertRaises(GitError):
            commit_reviewed_tree(
                worktree, tree_sha=reviewed_tree, parent_sha=info.base_sha, subject="s", body=""
            )
        self.assertEqual(current_head(worktree), moved)

    def test_commit_reviewed_tree_runs_no_commit_hooks(self) -> None:
        worktree, info, reviewed_tree = self._staged_worktree("hooks")
        marker = self.root / "hook-ran"
        hooks = self.repo / ".git" / "hooks"
        for name in ("pre-commit", "commit-msg", "prepare-commit-msg"):
            hook = hooks / name
            hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n", encoding="utf-8")
            hook.chmod(0o755)

        commit_sha = commit_reviewed_tree(
            worktree, tree_sha=reviewed_tree, parent_sha=info.base_sha, subject="s", body="b"
        )

        self.assertEqual(current_head(worktree), commit_sha)
        self.assertFalse(marker.exists())

    def test_candidate_tree_matches_add_all_without_touching_the_index(self) -> None:
        worktree = self.root / "candidate"
        create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/candidate",
            worktree_path=worktree,
            require_clean_base=True,
        )
        base_tree = index_tree_sha(worktree)
        (worktree / "README.md").write_text("modified\n", encoding="utf-8")
        (worktree / "untracked file.txt").write_text("new\n", encoding="utf-8")
        (worktree / "delete me.txt").unlink()

        candidate = candidate_tree_sha(worktree)

        self.assertEqual(index_tree_sha(worktree), base_tree)
        stage_all(worktree)
        self.assertEqual(index_tree_sha(worktree), candidate)

    def test_staged_blobs_preserve_paths_and_skip_gitlinks(self) -> None:
        (self.repo / "unicode file é.txt").write_text("blob\n", encoding="utf-8")
        stage_all(self.repo)
        blobs = staged_changed_blobs(self.repo)
        blob = next(item for item in blobs if item.path == "unicode file é.txt")
        self.assertEqual(blob.size, len("blob\n".encode()))

        nested = self.root / "nested"
        nested.mkdir()
        run_git(nested, "init", "-q")
        run_git(nested, "config", "user.name", "MetaHarness Tests")
        run_git(nested, "config", "user.email", "tests@example.invalid")
        (nested / "README.md").write_text("nested\n", encoding="utf-8")
        run_git(nested, "add", "README.md")
        run_git(nested, "commit", "-m", "nested")
        nested_sha = run_git(nested, "rev-parse", "HEAD").stdout.strip()
        run_git(
            self.repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{nested_sha},submodule.py",
        )
        changes = {change.path: change for change in staged_changes(self.repo)}
        self.assertTrue(changes["submodule.py"].is_gitlink)
        self.assertNotIn("submodule.py", {item.path for item in staged_changed_blobs(self.repo)})

    def test_changed_blob_selection_ignores_unmodified_files(self) -> None:
        for index in range(300):
            (self.repo / f"untouched {index:03d}.txt").write_text(f"{index}\n", encoding="utf-8")
        (self.repo / "link.txt").symlink_to("README.md")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-qm", "many files")

        (self.repo / "README.md").write_text("modified\n", encoding="utf-8")
        (self.repo / "delete me.txt").unlink()
        (self.repo / "nouveau é.txt").write_text("unicode\n", encoding="utf-8")
        (self.repo / "data file.bin").write_bytes(b"\x00\x01\xff binary")
        (self.repo / "link.txt").unlink()
        (self.repo / "link.txt").symlink_to("rename me.txt")
        (self.repo / "new link").symlink_to("README.md")
        stage_all(self.repo)

        changes = staged_changes(self.repo)
        self.assertEqual(
            {change.path for change in changes},
            set(staged_changed_files(self.repo)),
        )
        self.assertTrue(next(c for c in changes if c.path == "delete me.txt").deleted)
        blobs = {blob.path: blob for blob in staged_changed_blobs(self.repo)}
        self.assertEqual(
            set(blobs),
            {"README.md", "nouveau é.txt", "data file.bin", "link.txt", "new link"},
        )
        self.assertEqual(blobs["data file.bin"].size, len(b"\x00\x01\xff binary"))
        self.assertEqual(blobs["link.txt"].size, len("rename me.txt"))
        self.assertFalse(any(path.startswith("untouched") for path in blobs))

    def test_ref_snapshots_and_option_like_refs(self) -> None:
        worktree = self.root / "refs"
        create_run_worktree(
            self.repo,
            base_ref=self.initial_sha,
            branch="metaharness/refs",
            worktree_path=worktree,
            require_clean_base=True,
        )
        self.assertEqual(symbolic_head(worktree), "refs/heads/metaharness/refs")
        self.assertIn("refs/heads/metaharness/refs", local_branches(self.repo))
        run_git(worktree, "switch", "-q", "--detach")
        self.assertIsNone(symbolic_head(worktree))
        with self.assertRaises(GitError):
            resolve_commit(self.repo, "--all")

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
        (self.repo / "dir").mkdir()
        (self.repo / "dir" / "file.txt").write_text("x\n", encoding="utf-8")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-qm", "dir")
        with self.assertRaises(GitError):
            read_file_at_commit(
                self.repo, commit_sha=current_head(self.repo), relative_path="dir"
            )


if __name__ == "__main__":
    unittest.main()
