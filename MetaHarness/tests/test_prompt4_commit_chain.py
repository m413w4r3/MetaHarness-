"""Prompt 4: accepted trees, explicit deferred verification and linear chains."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from metaharness.commit_gate import (
    CommitSafetyError,
    accepted_step_record,
    assert_deferred_verifications_resolved,
    commit_safety_gate,
    parse_deferred_verification,
)
from metaharness.gitops import (
    commit_step_tree,
    current_head,
    GitError,
    index_tree_sha,
    stage_all,
    validate_linear_commit_chain,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class Prompt4CommitChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "MetaHarness tests")
        git(self.repo, "config", "user.email", "tests@example.invalid")
        (self.repo / "state.txt").write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "base")
        self.base = current_head(self.repo)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def accept_step(self, step_id: str, value: str) -> tuple[str, dict[str, object]]:
        path = self.repo / f"{step_id}.txt"
        path.write_text(value, encoding="utf-8")
        stage_all(self.repo)
        tree = index_tree_sha(self.repo)
        parent = current_head(self.repo)
        gate = commit_safety_gate(
            self.repo,
            tree_sha=tree,
            parent_sha=parent,
            mutable_scope=(f"{step_id}.txt",),
        )
        commit = commit_step_tree(
            self.repo,
            tree_sha=tree,
            parent_sha=parent,
            step_id=step_id,
            step_title=f"step {step_id}",
        )
        record = accepted_step_record(
            step_id=step_id,
            verification_status="passed",
            parent_sha=parent,
            commit_sha=commit,
            tree_before=git(self.repo, "rev-parse", f"{parent}^{{tree}}"),
            tree_after=tree,
            changed_paths=gate.changed_paths,
        )
        return commit, record

    def test_five_accepted_steps_are_one_linear_chain(self) -> None:
        records = []
        for number in range(1, 6):
            _commit, record = self.accept_step(f"S{number:02d}", f"{number}\n")
            records.append(record)
        tip = current_head(self.repo)
        self.assertEqual(
            validate_linear_commit_chain(
                self.repo,
                base_sha=self.base,
                tip_sha=tip,
                accepted_commits=records,
                approved_tree_sha=records[-1]["tree_after"],
            )[-1],
            tip,
        )
        for previous, current in zip(records, records[1:]):
            self.assertEqual(current["parent_sha"], previous["commit_sha"])

    def test_secret_is_rejected_before_commit(self) -> None:
        (self.repo / "secret.txt").write_text("token=do-not-commit\n", encoding="utf-8")
        stage_all(self.repo)
        with self.assertRaisesRegex(CommitSafetyError, "SECRET_IN"):
            commit_safety_gate(
                self.repo,
                tree_sha=index_tree_sha(self.repo),
                parent_sha=self.base,
                mutable_scope=("secret.txt",),
                secrets=("do-not-commit",),
            )
        self.assertEqual(current_head(self.repo), self.base)

    def test_deferred_verification_requires_a_future_dependency(self) -> None:
        report = (
            "DEFERRED VERIFY DEPENDENCY\n"
            "- command: pytest tests/test_future.py\n"
            "- dependent step: S02\n"
        )
        deferred = parse_deferred_verification(
            report, current_step_id="S01", future_step_ids=("S02",)
        )
        self.assertIsNotNone(deferred)
        record = {
            "step_id": "S01",
            "verification_status": "deferred",
            "dependent_step_ids": ["S02"],
        }
        with self.assertRaises(CommitSafetyError):
            assert_deferred_verifications_resolved([record])
        assert_deferred_verifications_resolved(
            [record, {"step_id": "S02", "verification_status": "passed"}]
        )

    def test_failed_verification_does_not_create_a_commit(self) -> None:
        (self.repo / "red.txt").write_text("red\n", encoding="utf-8")
        stage_all(self.repo)
        with self.assertRaises(CommitSafetyError):
            commit_safety_gate(
                self.repo,
                tree_sha=index_tree_sha(self.repo),
                parent_sha=self.base,
                mutable_scope=("red.txt",),
                verification_status="failed",
            )
        self.assertEqual(current_head(self.repo), self.base)

    def test_merge_commit_is_rejected_from_the_accepted_chain(self) -> None:
        accepted, record = self.accept_step("S01", "green\n")
        tree = git(self.repo, "rev-parse", f"{accepted}^{{tree}}")
        merge = git(
            self.repo,
            "commit-tree", tree,
            "-p", self.base,
            "-p", accepted,
            "-m", "merge",
        )
        git(self.repo, "update-ref", "HEAD", merge, accepted)
        with self.assertRaisesRegex(GitError, "merge"):
            validate_linear_commit_chain(
                self.repo,
                base_sha=self.base,
                tip_sha=merge,
                accepted_commits=[record],
            )


if __name__ == "__main__":
    unittest.main()
