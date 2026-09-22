"""The single commit path and every precondition of ``authorize_commit``."""

import ast
import dataclasses
import functools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.evidence import EvidenceBundle  # noqa: E402
from metaharness.gitops import (  # noqa: E402
    create_run_worktree,
    current_head,
    index_tree_sha,
    stage_all,
)
from metaharness.models import ReviewRoute, ReviewVerdict  # noqa: E402
from metaharness.orchestrator import CommitBoundaryError, authorize_commit  # noqa: E402
from metaharness.planning import parse_task_plan  # noqa: E402
from metaharness.review import parse_review  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "metaharness"


@functools.cache
def parsed_sources() -> tuple[tuple[Path, ast.Module], ...]:
    """Parse the production sources once for every static structure test."""

    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(SRC.rglob("*.py"))
    )

PLAN = """STATUS: READY
TITLE: t
OBJECTIVE: o
IMPLEMENTATION: i
ACCEPTANCE: a
TESTS: t
"""
PASS = """VERDICT: PASS
ROUTE: NONE
SUMMARY: fine
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
"""


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

    def test_single_commit_call_is_guarded_by_authorize_commit(self) -> None:
        commit_calls = self.calls("commit_reviewed_tree")
        self.assertEqual(len(commit_calls), 1)
        path, _ = commit_calls[0]
        self.assertEqual(path.name, "orchestrator.py")

        tree = dict(parsed_sources())[path]
        execute = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_execute"
        )
        statements = execute.body

        def is_commit_call(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "commit_reviewed_tree"
            )

        index = next(
            i for i, statement in enumerate(statements)
            if any(is_commit_call(node) for node in ast.walk(statement))
        )
        call = next(node for node in ast.walk(statements[index]) if is_commit_call(node))
        guard = statements[index - 1]
        self.assertIsInstance(guard, ast.Assign)
        self.assertEqual(guard.value.func.id, "authorize_commit")
        approved_name = guard.targets[0].id
        tree_argument = next(k.value for k in call.keywords if k.arg == "tree_sha")
        # The guard's result is the tree committed, and guard and commit are
        # adjacent top-level statements of _execute (no branch between them).
        self.assertEqual(tree_argument.id, approved_name)


class AuthorizeCommitTests(unittest.TestCase):
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
        self.values = dict(
            plan=parse_task_plan(PLAN),
            agent_result=AgentResult(0, False, "done", {}, ""),
            evidence=EvidenceBundle(self.base, self.tree, ("a.txt",), "diff", (), True, ()),
            review=parse_review(PASS),
            worktree=self.worktree,
            base_sha=self.base,
            branch_ref="refs/heads/harness/t/run",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def assert_refused(self, message: str, **overrides) -> None:
        values = {**self.values, **overrides}
        with self.assertRaisesRegex(CommitBoundaryError, message):
            authorize_commit(**values)

    def test_all_preconditions_hold(self) -> None:
        self.assertEqual(authorize_commit(**self.values), self.tree)

    def test_planner_not_ready(self) -> None:
        blocked = parse_task_plan("STATUS: BLOCKED\nBLOCKERS: missing\n")
        self.assert_refused("READY", plan=blocked)

    def test_agent_failure_or_timeout(self) -> None:
        self.assert_refused("agent", agent_result=AgentResult(1, False, "", {}, ""))
        self.assert_refused("agent", agent_result=AgentResult(124, True, "", {}, ""))

    def test_deterministic_gate_failed(self) -> None:
        failed = dataclasses.replace(
            self.values["evidence"], deterministic_passed=False, failures=("CHECK_FAILED:test",)
        )
        self.assert_refused("deterministic", evidence=failed)

    def test_review_not_pass_or_route_not_none(self) -> None:
        review = self.values["review"]
        revise = parse_review(
            "VERDICT: REVISE\nROUTE: IMPLEMENTATION\nREQUIRED FIXES: fix it\n"
        )
        self.assert_refused("not authorize|PASS", review=revise)
        forged_route = dataclasses.replace(review, route=ReviewRoute.REPLAN)
        self.assert_refused("route", review=forged_route)
        forged_verdict = dataclasses.replace(review, verdict=ReviewVerdict.FAIL)
        self.assert_refused("PASS", review=forged_verdict)

    def test_review_raw_is_reparsed(self) -> None:
        review = self.values["review"]
        for raw in (
            PASS.replace("FINDINGS: NONE", "FINDINGS: MAJOR | data loss"),
            PASS.replace("REQUIRED FIXES: NONE", "REQUIRED FIXES: add a lock"),
            PASS + "VERDICT: REVISE\n",
            "not a review",
        ):
            with self.subTest(raw=raw):
                self.assert_refused("reviewer", review=dataclasses.replace(review, raw=raw))

    def test_tree_identity_mismatch(self) -> None:
        other = dataclasses.replace(self.values["evidence"], staged_tree_sha=self.base_tree())
        self.assert_refused("index changed", evidence=other)

    def base_tree(self) -> str:
        return run_git(self.worktree, "rev-parse", f"{self.base}^{{tree}}")

    def test_staged_index_changed(self) -> None:
        (self.worktree / "a.txt").write_text("other\n", encoding="utf-8")
        stage_all(self.worktree)
        self.assert_refused("index changed")

    def test_tracked_working_tree_change(self) -> None:
        (self.worktree / "a.txt").write_text("unstaged\n", encoding="utf-8")
        self.assert_refused("unstaged change")

    def test_new_untracked_file(self) -> None:
        (self.worktree / "late.txt").write_text("late\n", encoding="utf-8")
        self.assert_refused("untracked")

    def test_head_changed(self) -> None:
        run_git(self.worktree, "commit", "-qm", "moved")
        self.assert_refused("HEAD changed")

    def test_branch_switched(self) -> None:
        run_git(self.worktree, "switch", "-q", "-c", "elsewhere")
        self.assert_refused("run branch")


if __name__ == "__main__":
    unittest.main()
