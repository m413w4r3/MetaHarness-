"""The guarded commit paths and every precondition of the candidate gate."""

import ast
import dataclasses
import functools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.evidence import EvidenceBundle  # noqa: E402
from metaharness.gitops import (  # noqa: E402
    create_run_worktree,
    current_head,
    index_tree_sha,
    stage_all,
)
from metaharness.orchestrator import CommitBoundaryError  # noqa: E402
from metaharness.orchestration.publication import PublicationService  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "metaharness"


@functools.cache
def parsed_sources() -> tuple[tuple[Path, ast.Module], ...]:
    """Parse the production sources once for every static structure test."""

    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(SRC.rglob("*.py"))
    )

def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class CommitPathStructureTests(unittest.TestCase):
    """Static proof that there is exactly one reachable commit primitive."""

    def calls(self, name: str) -> list[tuple[Path, ast.Call]]:
        found = []
        for path, tree in parsed_sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    if called == name:
                        found.append((path, node))
        return found

    def test_only_gitops_names_history_changing_git_commands(self) -> None:
        forbidden = {"commit", "commit-tree", "update-ref", "merge", "push", "reset", "rebase", "stash"}
        for path, tree in parsed_sources():
            literals = {
                node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            used = forbidden & literals
            if path.name == "gitops.py":
                self.assertEqual(used, {"commit-tree", "update-ref", "push"})
            else:
                self.assertEqual(used, set(), path)

    def test_every_commit_primitive_runs_after_the_commit_safety_gate(self) -> None:
        primitives = {
            "commit_step_tree", "commit_candidate_tree",
            "commit_repair_tree", "commit_revision_tree",
        }
        found: dict[str, list[str]] = {}
        for path, tree in parsed_sources():
            if path.name == "gitops.py":
                continue
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                calls = [
                    node for node in ast.walk(function)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                ]
                names = [node.func.id for node in calls]
                for primitive in primitives & set(names):
                    found.setdefault(primitive, []).append(f"{path.name}:{function.name}")
                    gate = min(
                        (node.lineno for node in calls if node.func.id == "commit_safety_gate"),
                        default=None,
                    )
                    first = min(node.lineno for node in calls if node.func.id == primitive)
                    self.assertIsNotNone(gate, f"{function.name} commits without the safety gate")
                    self.assertLess(gate, first, function.name)
        self.assertEqual(found, {
            "commit_step_tree": ["implementation.py:_accept_v2_step_tree"],
            "commit_repair_tree": ["check_repair.py:accept"],
            "commit_revision_tree": ["check_repair.py:accept"],
        })


class CandidateGateTests(unittest.TestCase):
    """Every precondition of the immutable candidate commit, re-derived."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        run_git(self.repo, "init", "-q")
        run_git(self.repo, "config", "user.name", "t")
        run_git(self.repo, "config", "user.email", "t@example.invalid")
        (self.repo / "a.txt").write_text("base\n", encoding="utf-8")
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-qm", "base")
        self.base = current_head(self.repo)
        info = create_run_worktree(
            self.repo,
            base_ref=self.base,
            branch="harness/t/run",
            worktree_path=root / "wt",
            require_clean_base=True,
        )
        self.worktree = info.worktree
        (self.worktree / "a.txt").write_text("change\n", encoding="utf-8")
        stage_all(self.worktree)
        self.tree = index_tree_sha(self.worktree)
        self.evidence = EvidenceBundle(self.base, self.tree, ("a.txt",), "diff", (), True, ())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def authorize(self, evidence: EvidenceBundle | None = None) -> str:
        owner = object.__new__(PublicationService)
        return owner.authorize_candidate_tree(
            evidence or self.evidence, self.worktree, self.base, "refs/heads/harness/t/run",
        )

    def assert_refused(self, message: str, evidence: EvidenceBundle | None = None) -> None:
        with self.assertRaisesRegex(CommitBoundaryError, message):
            self.authorize(evidence)

    def test_all_preconditions_hold(self) -> None:
        self.assertEqual(self.authorize(), self.tree)

    def test_deterministic_gate_failed(self) -> None:
        failed = dataclasses.replace(
            self.evidence, deterministic_passed=False, failures=("CHECK_FAILED:test",)
        )
        self.assert_refused("deterministic", failed)

    def test_tree_identity_mismatch(self) -> None:
        other = dataclasses.replace(
            self.evidence, staged_tree_sha=run_git(self.worktree, "rev-parse", f"{self.base}^{{tree}}")
        )
        self.assert_refused("tree changed", other)

    def test_staged_index_changed(self) -> None:
        (self.worktree / "a.txt").write_text("other\n", encoding="utf-8")
        stage_all(self.worktree)
        self.assert_refused("tree changed")

    def test_tracked_working_tree_change(self) -> None:
        (self.worktree / "a.txt").write_text("unstaged\n", encoding="utf-8")
        self.assert_refused("tree changed")

    def test_new_untracked_file(self) -> None:
        (self.worktree / "late.txt").write_text("late\n", encoding="utf-8")
        self.assert_refused("tree changed|changes")

    def test_head_changed(self) -> None:
        run_git(self.worktree, "commit", "-qm", "moved")
        self.assert_refused("HEAD changed")

    def test_branch_switched(self) -> None:
        run_git(self.worktree, "switch", "-q", "-c", "elsewhere")
        self.assert_refused("HEAD changed")


if __name__ == "__main__":
    unittest.main()
