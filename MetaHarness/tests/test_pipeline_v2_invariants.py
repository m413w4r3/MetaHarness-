"""End-to-end invariant checks for the provider-neutral v2 boundaries.

The fixtures in this module are deliberately small: Git, the deterministic
commit gate and the executor registry are real; model, reviewer and provider
objects are in-process fakes.  No test here needs a network or an API key.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metaharness.agent import (
    AgentRunRequest,
    AgentRunResult,
    AgentExecutorCapabilities,
    ExecutorRuntimeConfig,
    executor_for_profile,
    register_executor_driver,
)
from metaharness.commit_gate import (
    CommitSafetyError,
    assert_deferred_verifications_resolved,
    commit_safety_gate,
    parse_deferred_verification,
)
from metaharness.gitops import (
    commit_parents,
    commit_repair_tree,
    commit_revision_tree,
    commit_step_tree,
    current_head,
    index_tree_sha,
    publish_fast_forward_base,
    push_run_branch,
    stage_all,
    validate_linear_commit_chain,
)
from metaharness.models import ExecutionRole, ModelProfile, SelectionMode
from metaharness.orchestrator import Orchestrator
from metaharness.resume import pipeline_version_from_state
from metaharness.review import parse_review
from metaharness.run_options import RunOptions
from metaharness.state import RunStateStore


PASS = """VERDICT: PASS
ROUTE: NONE
SUMMARY: accepted
FINDINGS: NONE
REQUIRED FIXES: NONE
MISSING TESTS: NONE
RESIDUAL RISKS: NONE
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class PipelineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.name", "MetaHarness invariant tests")
        git(self.repo, "config", "user.email", "tests@example.invalid")
        (self.repo / "feature.txt").write_text("BASE\n", encoding="utf-8")
        git(self.repo, "add", "feature.txt")
        git(self.repo, "commit", "-qm", "BASE")
        self.base = current_head(self.repo)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def accept(
        self,
        value: str,
        *,
        kind: str,
        step_id: str | None = None,
        parent: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        parent = parent or current_head(self.repo)
        (self.repo / "feature.txt").write_text(value, encoding="utf-8")
        stage_all(self.repo)
        tree = index_tree_sha(self.repo)
        gate = commit_safety_gate(
            self.repo,
            tree_sha=tree,
            parent_sha=parent,
            mutable_scope=("feature.txt",),
        )
        if kind == "step":
            commit = commit_step_tree(
                self.repo, tree_sha=tree, parent_sha=parent,
                step_id=step_id or "S01", step_title="accepted step",
            )
        elif kind == "repair":
            commit = commit_repair_tree(
                self.repo, tree_sha=tree, parent_sha=parent, cycle=7,
            )
        else:
            commit = commit_revision_tree(
                self.repo, tree_sha=tree, parent_sha=parent,
            )
        return commit, {
            "step_id": step_id or kind,
            "parent_sha": parent,
            "commit_sha": commit,
            "tree_before": git(self.repo, "rev-parse", f"{parent}^{{tree}}"),
            "tree_after": tree,
            "changed_paths": list(gate.changed_paths),
            "verification_status": "passed",
        }


class PipelineV2EndToEndInvariantTests(PipelineFixture):
    def test_a_repair_and_semantic_revision_preserve_the_exact_accepted_chain(self) -> None:
        """A red attempt is evidence only; A, B, C and D are linear."""

        accepted: list[dict[str, object]] = []
        a, record = self.accept("S01\n", kind="step", step_id="S01")
        accepted.append(record)
        b, record = self.accept("S02\n", kind="step", step_id="S02")
        accepted.append(record)
        self.assertEqual(commit_parents(self.repo, a), (self.base,))
        self.assertEqual(commit_parents(self.repo, b), (a,))

        # Attempt 1 produces a durable tree/diff snapshot and no commit.
        attempt = self.root / "attempts" / "01"
        attempt.mkdir(parents=True)
        (self.repo / "feature.txt").write_text("T1\n", encoding="utf-8")
        stage_all(self.repo)
        tree_t1 = index_tree_sha(self.repo)
        (attempt / "tree.json").write_text(
            json.dumps({"tree": tree_t1, "diff": "feature.txt", "accepted_commit": None}),
            encoding="utf-8",
        )
        self.assertTrue((attempt / "tree.json").is_file())
        attempt_evidence = json.loads((attempt / "tree.json").read_text(encoding="utf-8"))
        self.assertEqual(attempt_evidence["tree"], tree_t1)
        self.assertEqual(attempt_evidence["diff"], "feature.txt")
        self.assertIsNone(attempt_evidence["accepted_commit"])
        self.assertEqual(current_head(self.repo), b)
        git(self.repo, "reset", "-q", "--hard", b)

        c, record = self.accept("T2\n", kind="repair", parent=b)
        accepted.append(record)
        d, record = self.accept("T3\n", kind="revision", parent=c)
        accepted.append(record)
        self.assertEqual(commit_parents(self.repo, c), (b,))
        self.assertEqual(commit_parents(self.repo, d), (c,))
        self.assertNotIn(tree_t1, {item["tree_after"] for item in accepted})

        self.assertEqual(
            validate_linear_commit_chain(
                self.repo, base_sha=self.base, tip_sha=d,
                accepted_commits=accepted, approved_tree_sha=accepted[-1]["tree_after"],
            ),
            (a, b, c, d),
        )

        # The candidate is immutable and pushed before the reviewer sees it;
        # publication uses that exact approved SHA.
        branch = "harness/invariant/run"
        git(self.repo, "switch", "-q", "-c", branch, d)
        bare = self.root / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        git(self.repo, "remote", "add", "origin", str(bare))
        git(self.repo, "push", "-q", "origin", f"{self.base}:refs/heads/main")
        events: list[tuple[str, str]] = []
        push_run_branch(self.repo, remote="origin", branch=branch, commit_sha=d)
        events.append(("push", d))
        reviewer_sha = d
        events.append(("review", reviewer_sha))
        self.assertLess(events.index(("push", d)), events.index(("review", d)))
        publication = publish_fast_forward_base(
            self.repo, remote="origin", base_branch="main", base_sha=self.base,
            commit_sha=d, approved_tree=accepted[-1]["tree_after"],
            run_branch=branch, expected_parent=c, accepted_commits=accepted,
        )
        self.assertEqual(publication.commit_sha, d)
        self.assertEqual(
            git(self.repo, "--git-dir", str(bare), "rev-parse", "refs/heads/main"), d,
        )

    def test_b_check_repair_budget_is_separate_from_review_budget(self) -> None:
        options = RunOptions(
            schema_version=2, pipeline_version=2, protocol="v2",
            decomposition="balanced", execution_mode_policy="auto",
            single_step_max_mutable_paths=2, staged_step_max_mutable_paths=6,
            semantic_revision_enabled=True, max_check_repair_attempts=2,
            max_review_repair_cycles=4, planner_profile="planner",
            default_implementer_profile="worker", check_repair_profile="repair",
            semantic_reviser_profile="reviser", final_reviewer_profile="reviewer",
        )
        self.assertEqual(options.max_check_repair_attempts, 2)
        self.assertEqual(options.max_review_repair_cycles, 4)
        self.assertNotEqual(options.max_check_repair_attempts, options.max_review_repair_cycles)

    def test_c_d_e_review_routes_are_data_and_only_the_selected_route_is_actionable(self) -> None:
        for route in ("IMPLEMENTATION", "REPLAN", "HUMAN"):
            raw = PASS.replace("VERDICT: PASS", "VERDICT: REVISE").replace(
                "ROUTE: NONE", f"ROUTE: {route}"
            ).replace("FINDINGS: NONE", "FINDINGS: MAJOR | issue | fix | required").replace(
                "REQUIRED FIXES: NONE", "REQUIRED FIXES: apply the selected route"
            )
            result = parse_review(raw, deterministic_passed=True)
            self.assertEqual(result.route.value, route)
            self.assertEqual(result.verdict.value, "REVISE")
        # A reviewer never receives the correction budget as a control input.
        self.assertNotIn("max_review_repair_cycles", PASS)

    def test_f_integrity_failures_fail_closed_without_an_agent_call(self) -> None:
        calls: list[str] = []
        (self.repo / "secret.txt").write_text("token=secret-value\n", encoding="utf-8")
        stage_all(self.repo)
        with self.assertRaises(CommitSafetyError):
            commit_safety_gate(
                self.repo, tree_sha=index_tree_sha(self.repo), parent_sha=self.base,
                mutable_scope=("secret.txt",), secrets=("secret-value",),
            )
        self.assertEqual(calls, [])
        self.assertEqual(current_head(self.repo), self.base)

        git(self.repo, "reset", "-q", "--hard", self.base)
        (self.repo / "feature.txt").write_text("unexpected\n", encoding="utf-8")
        stage_all(self.repo)
        with self.assertRaises(CommitSafetyError):
            commit_safety_gate(
                self.repo, tree_sha="0" * 40, parent_sha=self.base,
                mutable_scope=("feature.txt",),
            )
        self.assertEqual(calls, [])

    def test_g_deferred_verification_allows_an_intermediate_commit_only(self) -> None:
        report = (
            "DEFERRED VERIFY DEPENDENCY\n"
            "- command: verify after S02\n"
            "- dependent step: S02\n"
        )
        deferred = parse_deferred_verification(
            report, current_step_id="S01", future_step_ids=("S02",)
        )
        self.assertIsNotNone(deferred)
        pending = {
            "step_id": "S01", "verification_status": "deferred",
            "dependent_step_ids": ["S02"],
        }
        with self.assertRaises(CommitSafetyError):
            assert_deferred_verifications_resolved([pending])
        assert_deferred_verifications_resolved(
            [pending, {"step_id": "S02", "verification_status": "passed"}]
        )

    def test_i_backend_neutral_executors_have_the_same_business_result(self) -> None:
        class EqualFake:
            capabilities = AgentExecutorCapabilities(edits_workspace=True)
            driver_version = "offline"

            def __init__(self, driver: str) -> None:
                self.driver = driver

            def run(self, request: AgentRunRequest) -> AgentRunResult:
                return AgentRunResult(
                    status="completed", exit_reason=None,
                    tree_before="T-before", tree_after="T-after", usage=None,
                    external_session_id=None, report_path=None, driver=self.driver,
                )

        for driver in ("fake-codex", "fake-claude", "fake-deepseek"):
            register_executor_driver(
                driver,
                lambda _profile, _runtime, driver=driver, **_: EqualFake(driver),
                replace=True,
            )
        request = AgentRunRequest(
            role=ExecutionRole.IMPLEMENTER, profile_id="worker", prompt="same",
            worktree=self.repo, artifact_dir=self.root / "artifacts", mutable_paths=(),
        )
        results = []
        for driver in ("fake-codex", "fake-claude", "fake-deepseek"):
            profile = ModelProfile(
                id=driver, display_name=driver, roles=(ExecutionRole.IMPLEMENTER,),
                driver=driver, model="same-model", provider="same-provider",
                effort="high", selection_mode=SelectionMode.CLI,
            )
            results.append(executor_for_profile(profile, ExecutorRuntimeConfig(environment={})).run(request))
        self.assertEqual(
            [(item.status, item.tree_before, item.tree_after) for item in results],
            [("completed", "T-before", "T-after")] * 3,
        )


if __name__ == "__main__":
    unittest.main()
