"""A6 -- scope is a signal: soft acceptance, strict discard, hard-deny bounds.

One authority decides every scope question (``metaharness.scope``).  An
ordinary path outside the declared mutable scope is never a boundary: the
worker attempt records it (``soft``) or restores it before continuing
(``strict``).  Only a physical escape, a hard-denied path, a push, a foreign
ref or an unrecoverable Git state stay fatal.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from pathlib import Path

from metaharness.attempt_transaction import (
    BRANCH_MODIFIED_OUTSIDE_AUTHORITY,
    REMOTE_AUTHORITY_MISMATCH,
    audit_git_mutation,
    git_ownership,
    recover_worker_git_state,
)
from metaharness.gitops import (
    current_head,
    local_branches,
    symbolic_head,
)
from metaharness.models import ExecutionRole, RunStatus
from metaharness.scope import (
    DEFAULT_HARD_DENY_PATTERNS,
    HARD_DENY_PATH_MUTATION,
    UNSAFE_PATH_MUTATION,
    ScopePolicy,
    ScopeViolation,
    assert_worktree_containment,
    normalize_repo_path,
)

from tests.autonomy.support import SPEC, AutonomyHarness, Step, meta_plan
from tests.pipeline_support import PipelineHarness, git

RUN_BRANCH = "refs/heads/metaharness/run"
NEIGHBOUR = "def test_a():\n    assert True\n"
NEIGHBOUR_TOUCHED = "def test_a():\n    assert True\n# touched by the worker\n"


class ScopePolicyTests(unittest.TestCase):
    """The single matching authority, its defaults and its path safety."""

    def test_the_default_hard_deny_list_is_exact(self) -> None:
        self.assertEqual(DEFAULT_HARD_DENY_PATTERNS, (
            ".git/**", ".github/workflows/**", ".env", ".env.*", ".env*", "**/*secret*",
        ))
        policy = ScopePolicy()
        self.assertEqual(policy.mode, "soft")
        for path in (".git/config", ".github/workflows/ci.yml", ".env", "app/.env.local",
                     "secrets/api_key.txt", "src/secret_notes.py"):
            with self.subTest(path=path):
                self.assertTrue(policy.is_hard_denied(path))

    def test_makefile_is_not_hard_denied_by_default(self) -> None:
        self.assertFalse(ScopePolicy().is_hard_denied("Makefile"))
        self.assertFalse(ScopePolicy().is_hard_denied("tools/Makefile"))

    def test_pyproject_is_not_hard_denied_by_default(self) -> None:
        policy = ScopePolicy()
        for path in ("pyproject.toml", "package-lock.json", "uv.lock", "requirements.txt"):
            with self.subTest(path=path):
                self.assertFalse(policy.is_hard_denied(path))

    def test_a_hard_denied_path_is_fatal(self) -> None:
        with self.assertRaises(ScopeViolation) as caught:
            ScopePolicy().check(["src/a.py", ".env"])
        self.assertEqual(caught.exception.code, HARD_DENY_PATH_MUTATION)
        self.assertEqual(caught.exception.paths, (".env",))

    def test_the_active_configuration_file_inside_the_repository_is_denied(self) -> None:
        policy = ScopePolicy(config_path="metaharness.toml")
        self.assertTrue(policy.is_hard_denied("metaharness.toml"))
        self.assertFalse(ScopePolicy().is_hard_denied("metaharness.toml"))

    def test_unsafe_paths_are_fatal(self) -> None:
        for path in ("/etc/passwd", "../escape", "a/../../escape", "a\\b", "", " a.py"):
            with self.subTest(path=path):
                with self.assertRaises(ScopeViolation) as caught:
                    normalize_repo_path(path)
                self.assertEqual(caught.exception.code, UNSAFE_PATH_MUTATION)

    def test_a_symlink_that_leaves_the_worktree_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            worktree = root / "worktree"
            outside = root / "outside"
            worktree.mkdir()
            outside.mkdir()
            (outside / "elsewhere.txt").write_text("outside\n", encoding="utf-8")
            os.symlink(outside, worktree / "escape")
            assert_worktree_containment(worktree, ["inside.txt"])
            with self.assertRaises(ScopeViolation) as caught:
                assert_worktree_containment(worktree, ["escape/elsewhere.txt"])
            self.assertEqual(caught.exception.code, UNSAFE_PATH_MUTATION)

    def test_unknown_scope_modes_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            ScopePolicy(mode="bounded")


class GitRecoveryTests(PipelineHarness):
    """Local Git mistakes of one worker attempt are taken back, content kept."""

    def setUp(self) -> None:
        super().setUp()
        self.worktree = self.root / "worktree"
        git(self.repo, "worktree", "add", "-q", "-b", "metaharness/run", str(self.worktree))
        self.base = current_head(self.worktree)
        self.snapshot()

    def snapshot(self) -> None:
        """The ownership of the attempt: taken after every fixture is in place."""

        self.before = git_ownership(self.repo, self.worktree)

    def audit(self):
        after = git_ownership(self.repo, self.worktree)
        return audit_git_mutation(
            self.before, after, branch_ref=RUN_BRANCH, base_sha=self.base, repo=self.repo,
        )

    def recover(self, audit) -> None:
        recover_worker_git_state(
            self.repo, self.worktree, audit, branch_ref=RUN_BRANCH, base_sha=self.base,
        )

    def test_worker_local_commit_is_normalized(self) -> None:
        (self.worktree / "src").mkdir()
        (self.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
        git(self.worktree, "add", "--all")
        git(self.worktree, "commit", "-qm", "worker parasite commit")

        audit = self.audit()
        self.assertIsNone(audit.fatal_code)
        self.assertTrue(audit.worker_commits)
        self.recover(audit)

        self.assertEqual(current_head(self.worktree), self.base)
        self.assertEqual(symbolic_head(self.worktree), RUN_BRANCH)
        # The content stays exploitable: it is staged on top of the base.
        self.assertEqual((self.worktree / "src/a.py").read_text(encoding="utf-8"), "a = 2\n")
        self.assertIn("src/a.py", git(self.worktree, "diff", "--cached", "--name-only"))
        self.assertNotIn(
            "worker parasite commit",
            git(self.worktree, "log", "--format=%s").splitlines(),
        )

    def test_worker_new_local_branch_is_removed(self) -> None:
        git(self.worktree, "checkout", "-q", "-b", "worker/scratch")
        (self.worktree / "scratch.txt").write_text("scratch\n", encoding="utf-8")
        git(self.worktree, "add", "--all")
        git(self.worktree, "commit", "-qm", "scratch work")

        audit = self.audit()
        self.assertIsNone(audit.fatal_code)
        self.assertEqual(audit.created_branches, ("refs/heads/worker/scratch",))
        self.assertEqual(audit.adopted_branch, "refs/heads/worker/scratch")
        self.recover(audit)

        self.assertEqual(symbolic_head(self.worktree), RUN_BRANCH)
        self.assertEqual(current_head(self.worktree), self.base)
        self.assertNotIn("refs/heads/worker/scratch", local_branches(self.repo))
        self.assertEqual(
            (self.worktree / "scratch.txt").read_text(encoding="utf-8"), "scratch\n",
        )

    def test_preexisting_branch_is_never_removed(self) -> None:
        fixture = (self.repo / "feature.txt").read_text(encoding="utf-8")
        git(self.repo, "branch", "keep-me")
        self.snapshot()
        (self.worktree / "src").mkdir()
        (self.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
        git(self.worktree, "add", "--all")
        git(self.worktree, "commit", "-qm", "worker parasite commit")

        audit = self.audit()
        self.recover(audit)

        self.assertIn("refs/heads/keep-me", local_branches(self.repo))
        self.assertEqual(self.base, current_head(self.worktree))
        self.assertEqual(fixture, (self.repo / "feature.txt").read_text(encoding="utf-8"))

    def test_foreign_ref_mutation_is_fatal(self) -> None:
        git(self.repo, "branch", "foreign")
        self.snapshot()
        moved = git(
            self.repo, "commit-tree", f"{self.base}^{{tree}}", "-p", self.base, "-m", "elsewhere",
        ).strip()
        git(self.repo, "update-ref", "refs/heads/foreign", moved)

        audit = self.audit()
        self.assertEqual(audit.fatal_code, BRANCH_MODIFIED_OUTSIDE_AUTHORITY)
        self.assertIn("refs/heads/foreign", audit.fatal_detail)
        self.assertFalse(audit.recoverable)

    def test_push_attempt_is_fatal(self) -> None:
        (self.worktree / "src").mkdir()
        (self.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
        git(self.worktree, "add", "--all")
        git(self.worktree, "commit", "-qm", "worker parasite commit")
        git(self.worktree, "push", "-q", "origin", "metaharness/run")

        audit = self.audit()
        self.assertEqual(audit.fatal_code, REMOTE_AUTHORITY_MISMATCH)
        self.assertFalse(audit.recoverable)


class SoftScopeTests(AutonomyHarness):
    """A safe extra path is admitted for the attempt and recorded for the audit."""

    def plan(self, *steps: Step) -> str:
        return meta_plan(*steps)

    def one_step(self, write: str = "src/a.py") -> str:
        return self.plan(Step(id="S01", title="Change a", read=(write,), write=(write,)))

    def run_plan(self, plan: str, *, scope_mode: str = "soft"):
        return self.orchestrator(
            self.config(scope_mode=scope_mode), planner=[plan],
        ).run_text(SPEC, run_id="run")

    def test_soft_scope_accepts_safe_neighbour(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n", "tests/test_a.py": NEIGHBOUR})
        self.green_check()

        def action(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            (request.worktree / "tests/test_a.py").write_text(
                NEIGHBOUR_TOUCHED, encoding="utf-8",
            )
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, action)
        result = self.run_plan(self.one_step())

        self.assert_run_completed(result)
        step = self.step_record()
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["out_of_scope_paths"], ["tests/test_a.py"])
        self.assertEqual(
            (self.worktree() / "tests/test_a.py").read_text(encoding="utf-8"),
            NEIGHBOUR_TOUCHED,
        )

    def test_soft_scope_records_sorted_extra_paths(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n"})
        self.green_check()

        def action(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            for path in ("zzz.md", "aaa.md", "mmm.md"):
                (request.worktree / path).write_text("extra\n", encoding="utf-8")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, action)
        result = self.run_plan(self.one_step())

        self.assert_run_completed(result)
        step = self.step_record()
        self.assertEqual(step["out_of_scope_paths"], ["aaa.md", "mmm.md", "zzz.md"])
        self.assertEqual(step["changed_paths"], ["aaa.md", "mmm.md", "src/a.py", "zzz.md"])

    def test_soft_scope_does_not_expand_future_step_authority(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n", "src/b.py": "b = 1\n"})
        self.green_check()

        def first(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            (request.worktree / "notes.md").write_text("first\n", encoding="utf-8")
            return "done\n"

        def second(request):
            (request.worktree / "src/b.py").write_text("b = 2\n", encoding="utf-8")
            (request.worktree / "notes.md").write_text("second\n", encoding="utf-8")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, first, second)
        plan = self.plan(
            Step(id="S01", title="Change a", read=("src/a.py",), write=("src/a.py",)),
            Step(id="S02", title="Change b", read=("src/b.py",), write=("src/b.py",),
                 depends_on="S01"),
        )
        result = self.run_plan(plan)

        self.assert_run_completed(result)
        first_record = self.step_record("S01")
        second_record = self.step_record("S02")
        self.assertEqual(first_record["out_of_scope_paths"], ["notes.md"])
        self.assertEqual(second_record["out_of_scope_paths"], ["notes.md"])
        # The recorded extra path of S01 never widened S02's declared scope.
        implementer_calls = [
            call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER
        ]
        self.assertEqual(implementer_calls[-1].mutable_paths, ("src/b.py",))
        self.assertEqual(second_record["changed_paths"], ["notes.md", "src/b.py"])


class StrictScopeTests(AutonomyHarness):
    """Strict mode restores the extra paths and continues on the in-scope diff."""

    def one_step(self) -> str:
        return meta_plan(Step(id="S01", title="Change a", read=("src/a.py",), write=("src/a.py",)))

    def run_plan(self, plan: str):
        return self.orchestrator(
            self.config(scope_mode="strict"), planner=[plan],
        ).run_text(SPEC, run_id="run")

    def test_strict_scope_discards_only_extra_paths(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n", "notes.md": "base\n"})
        self.green_check()

        def action(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            (request.worktree / "notes.md").write_text("out of scope\n", encoding="utf-8")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, action)
        result = self.run_plan(self.one_step())

        self.assert_run_completed(result)
        step = self.step_record()
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["out_of_scope_paths"], [])
        self.assertEqual(step["changed_paths"], ["src/a.py"])
        # Only the extra path was restored; the in-scope work was kept.
        self.assertEqual(
            (self.worktree() / "notes.md").read_text(encoding="utf-8"), "base\n",
        )
        self.assertEqual(
            (self.worktree() / "src/a.py").read_text(encoding="utf-8"), "a = 2\n",
        )

    def test_strict_scope_retries_if_only_extra_paths_changed(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n", "notes.md": "base\n"})
        self.green_check()

        def discarded(request):
            (request.worktree / "notes.md").write_text("out of scope\n", encoding="utf-8")
            return "done\n"

        def corrected(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, discarded, corrected)
        result = self.run_plan(self.one_step())

        self.assert_run_completed(result)
        self.assertEqual(self.workers.roles(), ["implementer", "implementer", "auditor"])
        retry_prompt = [
            call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER
        ][-1].prompt
        self.assertIn("changes outside mutable scope were discarded", retry_prompt)
        step = self.step_record()
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["changed_paths"], ["src/a.py"])
        self.assertEqual(
            (self.worktree() / "notes.md").read_text(encoding="utf-8"), "base\n",
        )


class FatalScopeTests(AutonomyHarness):
    """The real boundaries stay strict in both modes."""

    def test_hard_deny_path_is_fatal(self) -> None:
        self.commit_files({"src/a.py": "a = 1\n"})
        self.green_check()

        def action(request):
            (request.worktree / "src/a.py").write_text("a = 2\n", encoding="utf-8")
            (request.worktree / ".env").write_text("TOKEN=x\n", encoding="utf-8")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, action)
        result = self.orchestrator(
            self.config(), planner=[
                meta_plan(Step(id="S01", title="Change a", read=("src/a.py",), write=("src/a.py",))),
            ],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED, self.failure_reason())
        self.assertEqual(self.failure_reason(), HARD_DENY_PATH_MUTATION)

if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
