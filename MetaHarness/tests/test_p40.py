"""P40: immutable candidate commit/push precedes semantic review."""

from __future__ import annotations

import json
import hashlib
import subprocess
import threading
import time
import unittest
from pathlib import Path

from metaharness.claude.agent import ClaudeResult
from metaharness.gitops import commit_parents
from metaharness.models import RunStatus
from metaharness.approval import write_scope_approval
from metaharness.run_options import RunOptions
from metaharness.resume import ResumePhase, read_checkpoint
from tests.test_p29 import (
    P29Harness, QueueClient, FakeClaude, FakeLuna, SINGLE_PLAN, REPAIR_PLAN, PASS,
    REVISE_IMPLEMENTATION, SPEC, writer, git, plan_text, step_block, write,
    TransportFailingClient,
)

REVISE_REPLAN = """VERDICT: REVISE
ROUTE: REPLAN
SUMMARY: The initial decomposition omitted two required files.
FINDINGS
- MAJOR | scope | the implementation must update the missing contract and backend configuration
REQUIRED FIXES
- Add the omitted contract and backend configuration changes required by the SPEC.
MISSING TESTS
NONE
RESIDUAL RISKS
NONE
END META REVIEW
"""


class FailingC02Claude(FakeClaude):
    """Claude double that revises C01 and fails C02 after its pre-checks."""

    def run_revision(self, prompt, worktree, *, artifacts_dir, profile, environment,
                     revision_dir=None) -> ClaudeResult:
        target = Path(revision_dir) if revision_dir is not None else Path(artifacts_dir) / "revision"
        if target.name != "C02":
            return super().run_revision(prompt, worktree, artifacts_dir=artifacts_dir, profile=profile,
                                        environment=environment, revision_dir=revision_dir)
        self.calls.append({"cycle": 2, "prompt": prompt})
        target.mkdir(parents=True, exist_ok=True)
        (target / "agent.stderr.log").write_text("claude C02 crashed\n", encoding="utf-8")
        (target / "agent.events.jsonl").write_text("", encoding="utf-8")
        return ClaudeResult(1, False, "", {}, "claude C02 crashed")


class P40CandidatePipelineTests(P29Harness):
    def _scope_repair_files(self) -> str:
        write(self.repo / "docs/agent/CONTRACT.md", "contract v1\n")
        write(self.repo / "backend/pyproject.toml", "[project]\nname = 'aw001'\n")
        git(self.repo, "add", "docs/agent/CONTRACT.md", "backend/pyproject.toml")
        git(self.repo, "commit", "-qm", "AW-001 approval files")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        return plan_text(
            step_block(1, read=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                       write_set=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                       operation="Repair omitted files"), title="AW-001 approval repair")

    def test_claude_c02_checkpoint_binds_c01_candidate_and_resumes_claude_only(self) -> None:
        config = self.make_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 3\n"),
        })
        orchestrator, planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_PLAN], reviews=[REVISE_IMPLEMENTATION],
            luna=luna, claude=FailingC02Claude(log=self.events),
        )
        with self.count_pushes() as first_pushes:
            failed = self.run_approved(config, orchestrator, "claude-c02")
        self.assertEqual(failed.state["failure"]["reason"], "CLAUDE_FAILED")
        self.assertEqual(first_pushes.call_count, 1)  # the C01 candidate only
        run_dir = failed.run_dir
        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.CLAUDE_C02)
        self.assertEqual(checkpoint.expected_head_sha, c01["commit_sha"])
        repair_step = json.loads((run_dir / "repair/C02/steps/S01/step.json").read_text())
        self.assertEqual(checkpoint.expected_tree_sha, repair_step["tree_after"])
        self.assertEqual(len(planner.prompts), 2)
        self.assertFalse((run_dir / "candidate/C02/commit.json").exists())

        second, second_planner, _r, second_luna, second_claude = self.orchestrator(
            config, reviews=[PASS]
        )
        with self.count_pushes() as pushes:
            resumed = second.resume("claude-c02")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(second_planner.prompts, [])
        self.assertEqual(second_luna.calls, [])
        self.assertEqual([call["cycle"] for call in second_claude.calls], [2])
        self.assertEqual(pushes.call_count, 1)  # the C02 candidate, never C01 again
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())
        self.assertEqual(c02["parent_sha"], c01["commit_sha"])
        self.assertEqual(
            json.loads((run_dir / "candidate/C01/commit.json").read_text())["commit_sha"],
            c01["commit_sha"],
        )
        self.assertEqual(resumed.state["commit_sha"], c02["commit_sha"])
        self.assertEqual(
            git(self.repo, "rev-list", "--count", f"{self.base_sha}..{c02['commit_sha']}"), "2"
        )

    def test_scope_approval_persists_repair_step_before_first_repair_worker(self) -> None:
        repair_plan = self._scope_repair_files()
        config = self.make_config()
        options = RunOptions.from_config(config, repair_scope_policy="require-approval")
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan], reviews=[REVISE_REPLAN],
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
        )
        waiting = self.run_approved(config, orchestrator, "scope-step", run_options=options)
        self.assertEqual(waiting.status, RunStatus.WAITING_SCOPE_APPROVAL)
        run_dir = waiting.run_dir
        paused = read_checkpoint(run_dir)
        self.assertEqual(paused.phase, ResumePhase.SCOPE_APPROVAL)
        delta_sha = hashlib.sha256((run_dir / "repair/C02/scope_delta.json").read_bytes()).hexdigest()
        self.assertEqual(paused.scope_delta_sha256, delta_sha)
        write_scope_approval(run_dir / "repair/C02", decision="APPROVE",
                             scope_delta_sha256=delta_sha, source="test")
        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        observed = []

        def repair(root: Path) -> None:
            # Observed from inside the first repair worker invocation.
            observed.append(read_checkpoint(run_dir))
            write(root / "docs/agent/CONTRACT.md", "contract v2\n")
            write(root / "backend/pyproject.toml", "[project]\nname = 'aw001-fixed'\n")

        second, second_planner, *_rest = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({(2, "S01"): repair}),
        )
        resumed = second.resume("scope-step")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(second_planner.prompts, [])
        self.assertEqual(len(observed), 1)
        durable = observed[0]
        self.assertEqual((durable.phase, durable.step_id), (ResumePhase.REPAIR_STEP, "S01"))
        self.assertEqual(durable.expected_head_sha, c01["commit_sha"])
        self.assertEqual(durable.expected_tree_sha, paused.expected_tree_sha)
        self.assertEqual(durable.repair_bundle_sha256, paused.repair_bundle_sha256)
        self.assertEqual(durable.scope_delta_sha256, paused.scope_delta_sha256)

    def test_require_approval_pauses_and_resumes_against_exact_delta(self) -> None:
        write(self.repo / "docs/agent/CONTRACT.md", "contract v1\n")
        write(self.repo / "backend/pyproject.toml", "[project]\nname = 'aw001'\n")
        git(self.repo, "add", "docs/agent/CONTRACT.md", "backend/pyproject.toml")
        git(self.repo, "commit", "-qm", "AW-001 approval files")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        repair_plan = plan_text(
            step_block(1, read=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                       write_set=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                       operation="Repair omitted files"), title="AW-001 approval repair")
        config = self.make_config()
        options = RunOptions.from_config(config, repair_scope_policy="require-approval")
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): lambda root: (
                write(root / "docs/agent/CONTRACT.md", "contract v2\n"),
                write(root / "backend/pyproject.toml", "[project]\nname = 'aw001-fixed'\n"),
            ),
        })
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan], reviews=[REVISE_REPLAN, PASS], luna=luna,
        )
        holder: dict[str, object] = {}
        thread = threading.Thread(target=lambda: holder.setdefault(
            "result", orchestrator.run_text(SPEC, run_id="aw-approval", run_options=options)))
        thread.start()
        state_path = config.runs_root / "aw-approval" / "state.json"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if state_path.exists() and json.loads(state_path.read_text())["status"] == "awaiting_plan_approval":
                break
            time.sleep(0.01)
        from metaharness.web.api import approve_run
        approve_run(config.runs_root, "aw-approval", "APPROVE", config=config,
                    reviewer_profile="reviewer", step_profiles={"S01": "luna"},
                    reviser_profile="claude", repair_profile="luna")
        thread.join(60)
        self.assertFalse(thread.is_alive())
        first = holder["result"]
        self.assertEqual(first.status, RunStatus.WAITING_SCOPE_APPROVAL)
        run_dir = config.runs_root / "aw-approval"
        delta_path = run_dir / "repair/C02/scope_delta.json"
        delta_sha = hashlib.sha256(delta_path.read_bytes()).hexdigest()
        write_scope_approval(run_dir / "repair/C02", decision="APPROVE",
                             scope_delta_sha256=delta_sha, source="test")
        resumed = orchestrator.resume("aw-approval")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))

    def test_aw001_replan_is_bounded_repair_with_exact_scope_delta(self) -> None:
        write(self.repo / "docs/agent/CONTRACT.md", "contract v1\n")
        write(self.repo / "backend/pyproject.toml", "[project]\nname = 'aw001'\n")
        git(self.repo, "add", "docs/agent/CONTRACT.md", "backend/pyproject.toml")
        git(self.repo, "commit", "-qm", "AW-001 base files")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        repair_plan = plan_text(
            step_block(
                1,
                read=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                write_set=("docs/agent/CONTRACT.md", "backend/pyproject.toml"),
                operation="Repair omitted files",
            ),
            title="AW-001 bounded repair",
        )
        config = self.make_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): lambda root: (
                write(root / "docs/agent/CONTRACT.md", "contract v2\n"),
                write(root / "backend/pyproject.toml", "[project]\nname = 'aw001-fixed'\n"),
            ),
        })
        orchestrator, _planner, reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan],
            reviews=[REVISE_REPLAN, PASS], luna=luna,
        )
        result = self.run_approved(
            config, orchestrator, "aw-001",
            run_options=RunOptions.from_config(config),
        )
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        delta = json.loads((result.run_dir / "repair/C02/scope_delta.json").read_text())
        self.assertEqual(delta["added_paths"], ["backend/pyproject.toml", "docs/agent/CONTRACT.md"])
        self.assertEqual(delta["unchanged_paths"], [])
        self.assertIn("backend/pyproject.toml", delta["added_path_reasons"])
        self.assertIn("docs/agent/CONTRACT.md", delta["added_path_reasons"])
        c01 = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        c02 = json.loads((result.run_dir / "candidate/C02/commit.json").read_text())
        self.assertEqual(c02["parent_sha"], c01["commit_sha"])
        self.assertTrue(c02["pushed_at"])
        self.assertEqual(len(reviewer.prompts), 2)

    def test_candidate_commit_and_push_precede_reviewer_and_main_is_unchanged(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "fetch", "-q", "origin")
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        harness = self
        class ObservingReviewer(QueueClient):
            def complete(self, prompt: str):
                self.main_before_pass = git(harness.repo, "rev-parse", "refs/heads/main")
                return super().complete(prompt)
        reviewer = ObservingReviewer("reviewer", [PASS], self.events)
        orchestrator, _planner, _reviewer, _luna, _claude = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviewer=reviewer, luna=luna
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "candidate-order")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        candidate = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        self.assertEqual(candidate["commit_sha"], result.state["commit_sha"])
        self.assertEqual(candidate["parent_sha"], self.base_sha)
        self.assertTrue(candidate["pushed_at"])
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(self.bare), "rev-parse", f"refs/heads/{result.state['branch']}"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            candidate["commit_sha"],
        )
        self.assertEqual(reviewer.main_before_pass, self.base_sha)
        self.assertLess(self.events.index("push"), next(
            index for index, value in enumerate(self.events) if value.startswith("reviewer:")
        ))

    def test_c02_candidate_is_parented_to_c01(self) -> None:
        config = self.make_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 3\n"),
        })
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_PLAN],
            reviews=[REVISE_IMPLEMENTATION, PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "candidate-ancestry")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        c01 = json.loads((result.run_dir / "candidate/C01/commit.json").read_text())
        c02 = json.loads((result.run_dir / "candidate/C02/commit.json").read_text())
        self.assertEqual(c02["parent_sha"], c01["commit_sha"])
        self.assertEqual(commit_parents(self.repo, c02["commit_sha"]), (c01["commit_sha"],))

    def test_reviewer_transport_resumes_without_replaying_candidate_push(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        first_reviewer = TransportFailingClient()
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviewer=first_reviewer, luna=luna
        )
        failed = self.run_approved(config, orchestrator, "candidate-transport")
        self.assertEqual(failed.state["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        self.assertEqual(read_checkpoint(failed.run_dir).phase, ResumePhase.REVIEWER_C01)
        candidate = json.loads((failed.run_dir / "candidate/C01/commit.json").read_text())
        self.assertTrue(candidate["pushed_at"])
        second, *_rest = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushes:
            resumed = second.resume("candidate-transport")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(pushes.call_count, 0)


if __name__ == "__main__":
    unittest.main()
