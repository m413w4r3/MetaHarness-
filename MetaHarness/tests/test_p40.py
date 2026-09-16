"""P40: immutable candidate commit/push precedes semantic review."""

from __future__ import annotations

import json
import hashlib
import subprocess
import threading
import time
import unittest
from pathlib import Path

import contextlib
import dataclasses
from unittest import mock

from metaharness import orchestrator as orchestrator_module
from metaharness.agent.base import AgentResult
from metaharness.claude.agent import ClaudeResult
from metaharness.gitops import commit_parents
from metaharness.llm.chat import LLMHTTPError
from metaharness.models import RunStatus
from metaharness.approval import write_scope_approval
from metaharness.orchestrator import (
    Orchestrator,
    _compact_step_history,
    _failure_reason,
    _step_reports_text,
)
from metaharness.run_options import RunOptions
from metaharness.resume import (
    ResumeIntegrityError, ResumeNotAllowedError, ResumePhase, ResumeRequiresOperatorError,
    read_checkpoint, resume_info, write_checkpoint,
)
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
    def test_step_reports_keep_all_steps_and_bound_each_body(self) -> None:
        results = [
            {
                "id": f"S{index:02d}", "profile_id": "luna",
                "tree_before": f"before-{index}", "tree_after": f"after-{index}",
                "usage": {"input_tokens": index, "output_tokens": index},
                "final": "x" * 10_000,
            }
            for index in range(1, 21)
        ]
        rendered = _step_reports_text(results)

        for index in range(1, 21):
            self.assertIn(f"S{index:02d}\n", rendered)
        bodies = rendered.split("final report:\n")[1:]
        self.assertEqual(len(bodies), 20)
        for body in bodies:
            report = body.split("\n\nS", 1)[0].rstrip("\n")
            self.assertLessEqual(len(report.encode()), 2_048)

    def test_compact_cycle_history_does_not_duplicate_final_reports(self) -> None:
        records = [{
            "id": "S01", "profile_id": "luna", "tree_after": "tree",
            "changed_paths": ["src/a.py"], "usage": {"output_tokens": 4},
            "final": "full report must remain in luna_reports",
        }]

        compact = _compact_step_history(records)

        self.assertEqual(compact, [{
            "id": "S01", "profile_id": "luna", "tree_after": "tree",
            "changed_paths": ["src/a.py"], "usage": {"output_tokens": 4},
        }])

    def test_thirty_two_step_report_prompt_keeps_the_last_step(self) -> None:
        results = [
            {
                "id": f"S{index:02d}", "profile_id": "luna",
                "tree_before": "before", "tree_after": "after",
                "usage": {}, "final": f"report-{index}",
            }
            for index in range(1, 33)
        ]

        rendered = _step_reports_text(results)

        self.assertIn("S01\n", rendered)
        self.assertIn("S32\n", rendered)
        self.assertEqual(rendered.count("final report:\n"), 32)

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
                self.run_branch_before_pass = bool(harness.remote_refs())
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
        self.assertTrue(reviewer.run_branch_before_pass)
        self.assertNotIn(f"refs/heads/{result.state['branch']}", self.remote_refs())
        self.assertEqual(result.state["publish"]["run_branch_cleanup"]["status"], "success")
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


AW001_FILES = ("backend/pyproject.toml", "docs/agent/CONTRACT.md")


class EditingFailingC02Claude(FakeClaude):
    """Claude C01 succeeds; Claude C02 edits an added path, then crashes."""

    def run_revision(self, prompt, worktree, *, artifacts_dir, profile, environment,
                     revision_dir=None) -> ClaudeResult:
        target = Path(revision_dir) if revision_dir is not None else Path(artifacts_dir) / "revision"
        if target.name != "C02":
            return super().run_revision(prompt, worktree, artifacts_dir=artifacts_dir, profile=profile,
                                        environment=environment, revision_dir=revision_dir)
        self.calls.append({"cycle": 2, "prompt": prompt})
        if self.edit:
            write(Path(worktree) / "docs/agent/CONTRACT.md", "contract CLAUDE PARTIAL\n")
        target.mkdir(parents=True, exist_ok=True)
        (target / "agent.stderr.log").write_text("claude C02 crashed\n", encoding="utf-8")
        (target / "agent.events.jsonl").write_text("", encoding="utf-8")
        return ClaudeResult(1, False, "", {}, "claude C02 crashed")

    edit = True


class PartialC02Luna(FakeLuna):
    """Luna C02 edits both added paths, then exits non-zero."""

    def run_step(self, contract, worktree, artifacts_dir, *, base_sha=None, env=None) -> AgentResult:
        directory = Path(artifacts_dir)
        if directory.parent.parent.name != "C02":
            return super().run_step(contract, worktree, artifacts_dir, base_sha=base_sha, env=env)
        self.calls.append({"cycle": 2, "step": directory.name, "dir": directory})
        write(Path(worktree) / "docs/agent/CONTRACT.md", "contract PARTIAL\n")
        write(Path(worktree) / "backend/pyproject.toml", "[project]\nname = 'partial'\n")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.events.jsonl").write_text("", encoding="utf-8")
        return AgentResult(exit_code=1, timed_out=False, final_message="", usage={}, stderr_tail="boom")


class RevisesThenTransportFails(QueueClient):
    """Reviewer #1 answers REVISE; reviewer #2 hits a bridge transport failure."""

    def complete(self, prompt: str):
        if not self.responses:
            self.prompts.append(prompt)
            raise LLMHTTPError("HTTP 503 from reviewer bridge")
        return super().complete(prompt)


class PlansThenTransportFails(RevisesThenTransportFails):
    """Planner: the initial plan answers; the repair planner hits a transport failure."""


def real_then_interrupt(real):
    """Let the real Git operation land, then crash before it is recorded."""

    def call(*args, **kwargs):
        real(*args, **kwargs)
        raise KeyboardInterrupt

    return call


class P40ResumeAuthorityTests(P29Harness):
    """Tree/scope/ancestry authority across Claude edits, C02 and publication."""

    def no_model_calls(self, planner, reviewer, luna, claude) -> None:
        self.assertEqual((planner.prompts, reviewer.prompts, luna.calls, claude.calls), ([], [], [], []))

    def aw001_repair_plan(self) -> str:
        write(self.repo / "docs/agent/CONTRACT.md", "contract v1\n")
        write(self.repo / "backend/pyproject.toml", "[project]\nname = 'aw001'\n")
        git(self.repo, "add", *AW001_FILES)
        git(self.repo, "commit", "-qm", "AW-001 base files")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        return plan_text(
            step_block(1, read=AW001_FILES, write_set=AW001_FILES, operation="Repair omitted files"),
            title="AW-001 bounded repair",
        )

    @staticmethod
    def aw001_repair(root: Path) -> None:
        write(root / "docs/agent/CONTRACT.md", "contract v2\n")
        write(root / "backend/pyproject.toml", "[project]\nname = 'aw001-fixed'\n")

    def origin_ref(self, ref: str) -> str:
        return subprocess.run(["git", "--git-dir", str(self.bare), "rev-parse", ref],
                              capture_output=True, text=True).stdout.strip()

    def publish_main_to_origin(self) -> None:
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "fetch", "-q", "origin")

    # -- C01: Claude really modifies the candidate ---------------------------
    def c01_claude_edit_run(self, run_id: str, *patches):
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        claude = FakeClaude({1: writer("src/a.py", "A = 20\n")}, log=self.events)
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN], reviews=[PASS], luna=luna, claude=claude,
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            interrupted = self.run_approved(config, orchestrator, run_id)
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED, interrupted.state.get("failure"))
        run_dir = interrupted.run_dir
        step = json.loads((run_dir / "steps/S01/step.json").read_text())
        revision = json.loads((run_dir / "revision/report.json").read_text())
        self.assertEqual(revision["status"], "COMPLETED")
        self.assertNotEqual(revision["tree_after"], step["tree_after"])
        self.assertEqual(read_checkpoint(run_dir).expected_tree_sha, revision["tree_after"])
        return config, run_dir, revision

    def resume_c01_published(self, config, run_id: str, *, expected_pushes: int):
        second, planner, reviewer, luna, claude = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushes:
            resumed = second.resume(run_id)
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((planner.prompts, luna.calls, claude.calls), ([], [], []))
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual(pushes.call_count, expected_pushes)
        self.assertEqual(self.commits_on_run_branch(run_id), "1")
        self.assertEqual(git(self.worktree(run_id), "show", "HEAD:src/a.py"), "A = 20")
        return resumed

    def test_c01_claude_edit_final_checks_crash_resumes_final_checks_only(self) -> None:
        patch = mock.patch.object(Orchestrator, "_final_evidence", side_effect=KeyboardInterrupt)
        config, run_dir, revision = self.c01_claude_edit_run("c01-final", patch)
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.FINAL_CHECKS_C01)
        checks_before = self.checks_ran()
        self.resume_c01_published(config, "c01-final", expected_pushes=1)
        self.assertEqual(self.checks_ran(), checks_before + 1)   # final checks only

    def test_c01_claude_edit_candidate_commit_crash_resumes_without_replay(self) -> None:
        patch = mock.patch.object(orchestrator_module, "commit_candidate_tree",
                                  side_effect=real_then_interrupt(orchestrator_module.commit_candidate_tree))
        config, run_dir, revision = self.c01_claude_edit_run("c01-commit", patch)
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.CANDIDATE_COMMIT_C01)
        self.assertFalse((run_dir / "candidate/C01/commit.json").exists())
        orphan = git(self.worktree("c01-commit"), "rev-parse", "HEAD")
        resumed = self.resume_c01_published(config, "c01-commit", expected_pushes=1)
        self.assertEqual(resumed.state["commit_sha"], orphan)   # reconciled, not recreated

    def test_c01_claude_edit_candidate_push_crash_resumes_without_duplicate_push(self) -> None:
        patch = mock.patch.object(orchestrator_module, "push_run_branch",
                                  side_effect=real_then_interrupt(orchestrator_module.push_run_branch))
        config, run_dir, revision = self.c01_claude_edit_run("c01-push", patch)
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.CANDIDATE_PUSH_C01)
        candidate = json.loads((run_dir / "candidate/C01/commit.json").read_text())
        resumed = self.resume_c01_published(config, "c01-push", expected_pushes=0)
        self.assertEqual(resumed.state["commit_sha"], candidate["commit_sha"])

    # -- C02: Claude C02 really modifies the candidate -----------------------
    def test_c02_claude_edit_reviewer_transport_resumes_reviewer_two_only(self) -> None:
        config = self.make_config()
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): writer("src/a.py", "A = 3\n")})
        claude = FakeClaude({2: writer("src/a.py", "A = 30\n")}, log=self.events)
        reviewer = RevisesThenTransportFails("reviewer", [REVISE_IMPLEMENTATION], self.events)
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_PLAN], reviewer=reviewer, luna=luna, claude=claude,
        )
        with self.count_pushes() as first_pushes:
            failed = self.run_approved(config, orchestrator, "c02-transport")
        self.assertEqual(failed.state["failure"]["reason"], "REVIEWER_TRANSPORT_FAILURE")
        self.assertEqual(first_pushes.call_count, 2)
        run_dir = failed.run_dir
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.REVIEWER_C02)
        step = json.loads((run_dir / "repair/C02/steps/S01/step.json").read_text())
        revision = json.loads((run_dir / "revision/C02/report.json").read_text())
        self.assertNotEqual(revision["tree_after"], step["tree_after"])
        self.assertEqual(checkpoint.expected_tree_sha, revision["tree_after"])
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())
        self.assertTrue(c02["pushed_at"])

        second, planner, reviewer2, luna2, claude2 = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushes:
            resumed = second.resume("c02-transport")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual((planner.prompts, luna2.calls, claude2.calls), ([], [], []))
        self.assertEqual(len(reviewer2.prompts), 1)
        self.assertEqual(pushes.call_count, 0)
        self.assertEqual(resumed.state["commit_sha"], c02["commit_sha"])
        self.assertEqual(self.commits_on_run_branch("c02-transport"), "2")
        self.assertEqual(git(self.worktree("c02-transport"), "show", "HEAD:src/a.py"), "A = 30")

    # -- AW-001: C02 scope expansion stays authorized on resume --------------
    def aw001_interrupted(self, run_id: str, *, luna: FakeLuna, claude: FakeClaude | None = None):
        repair_plan = self.aw001_repair_plan()
        config = self.make_config()
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan], reviews=[REVISE_REPLAN], luna=luna,
            claude=claude,
        )
        failed = self.run_approved(config, orchestrator, run_id,
                                   run_options=RunOptions.from_config(config))
        self.assertEqual(failed.status, RunStatus.FAILED)
        delta = json.loads((failed.run_dir / "repair/C02/scope_delta.json").read_text())
        self.assertEqual(delta["added_paths"], sorted(AW001_FILES))
        self.assertEqual(delta["original_mutable_paths"], ["src/a.py"])
        return config, failed

    def test_aw001_added_paths_survive_claude_c02_failure_and_resume(self) -> None:
        config, failed = self.aw001_interrupted(
            "aw001-claude", luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"),
                                           (2, "S01"): self.aw001_repair}),
            claude=EditingFailingC02Claude(log=self.events),
        )
        self.assertEqual(failed.state["failure"]["reason"], "CLAUDE_FAILED")
        self.assertEqual(read_checkpoint(failed.run_dir).phase, ResumePhase.CLAUDE_C02)
        second, planner, reviewer, luna, claude = self.orchestrator(config, reviews=[PASS])
        with self.count_pushes() as pushes:
            resumed = second.resume("aw001-claude")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # The failed Claude C02 edit on an added path is restored exactly.
        self.assertEqual(resumed.state["resume"]["restored_paths"], ["docs/agent/CONTRACT.md"])
        self.assertEqual((planner.prompts, luna.calls), ([], []))
        self.assertEqual([call["cycle"] for call in claude.calls], [2])
        self.assertEqual(pushes.call_count, 1)   # the C02 candidate only
        head = self.worktree("aw001-claude")
        self.assertEqual(git(head, "show", "HEAD:docs/agent/CONTRACT.md"), "contract v2")
        self.assertEqual(git(head, "show", "HEAD:backend/pyproject.toml"), "[project]\nname = 'aw001-fixed'")

    def test_aw001_failed_repair_worker_on_added_paths_is_restored_and_retried(self) -> None:
        config, failed = self.aw001_interrupted(
            "aw001-luna", luna=PartialC02Luna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
        )
        self.assertEqual(failed.state["failure"]["reason"], "AGENT_FAILED")
        checkpoint = read_checkpoint(failed.run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.step_id), (ResumePhase.REPAIR_STEP, "S01"))
        self.assertTrue(resume_info(failed.run_dir, failed.state).resumable)
        second, planner, reviewer, luna, claude = self.orchestrator(
            config, reviews=[PASS], luna=FakeLuna({(2, "S01"): self.aw001_repair}),
        )
        resumed = second.resume("aw001-luna")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual(resumed.state["resume"]["restored_paths"], sorted(AW001_FILES))
        self.assertEqual([(call["cycle"], call["step"]) for call in luna.calls], [(2, "S01")])
        self.assertEqual(planner.prompts, [])

    def test_path_outside_c01_and_repair_scope_is_resume_integrity_failure(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): self.aw001_repair})
        claude = EditingFailingC02Claude(log=self.events)
        claude.edit = False
        config, failed = self.aw001_interrupted("aw001-outside", luna=luna, claude=claude)
        with self.subTest("unknown writer outside both scopes"):
            write(self.worktree("aw001-outside") / "src/b.py", "B = OUTSIDE\n")
            second, *models = self.orchestrator(config)
            refused = second.resume("aw001-outside")
            self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
            self.assertFalse(resume_info(refused.run_dir, refused.state).resumable)
            self.no_model_calls(*models)
        with self.subTest("forged delta granting an extra path"):
            # Fresh run: the forged delta is re-bound to the checkpoint hash,
            # but it no longer matches the parsed repair plan.
            forged_config, forged = self.aw001_second_run("aw001-forged", config)
            run_dir = forged.run_dir
            delta_path = run_dir / "repair/C02/scope_delta.json"
            delta = json.loads(delta_path.read_text())
            delta["added_paths"] = sorted([*delta["added_paths"], "src/b.py"])
            delta["requested_write_paths"] = sorted([*delta["requested_write_paths"], "src/b.py"])
            delta_path.write_text(json.dumps(delta, indent=2) + "\n")
            write_checkpoint(run_dir, dataclasses.replace(
                read_checkpoint(run_dir),
                scope_delta_sha256=hashlib.sha256(delta_path.read_bytes()).hexdigest(),
            ))
            second, *models = self.orchestrator(forged_config)
            refused = second.resume("aw001-forged")
            self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
            self.assertIn("does not match the repair plan", refused.state["failure"]["detail"])
            self.no_model_calls(*models)

    def aw001_second_run(self, run_id: str, config):
        """Another AW-001 run on the already prepared repository base."""

        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): self.aw001_repair})
        claude = EditingFailingC02Claude(log=self.events)
        claude.edit = False
        repair_plan = plan_text(
            step_block(1, read=AW001_FILES, write_set=AW001_FILES, operation="Repair omitted files"),
            title="AW-001 bounded repair",
        )
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan], reviews=[REVISE_REPLAN], luna=luna, claude=claude,
        )
        failed = self.run_approved(config, orchestrator, run_id, run_options=RunOptions.from_config(config))
        self.assertEqual(failed.state["failure"]["reason"], "CLAUDE_FAILED")
        return config, failed

    # -- ResumeIntegrityError classification ---------------------------------
    def test_integrity_errors_keep_their_stable_reason(self) -> None:
        self.assertEqual(_failure_reason(ResumeIntegrityError("x")), "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(_failure_reason(ResumeRequiresOperatorError("x")), "RESUME_REQUIRES_OPERATOR")

    def test_scope_delta_hash_mismatch_during_resume_is_permanent(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): self.aw001_repair})
        claude = EditingFailingC02Claude(log=self.events)
        claude.edit = False
        config, failed = self.aw001_interrupted("aw001-delta", luna=luna, claude=claude)
        delta_path = failed.run_dir / "repair/C02/scope_delta.json"
        persisted = delta_path.read_bytes()
        real = orchestrator_module._build_scope_delta

        def recomputed_differently(*args, **kwargs):
            payload, content = real(*args, **kwargs)
            return payload, content + "\n"

        second, *models = self.orchestrator(config, reviews=[PASS])
        # The mismatch surfaces inside the C02 execution path, through the
        # generic resume handler rather than the pre-claim validation.
        with mock.patch.object(orchestrator_module, "_build_scope_delta", side_effect=recomputed_differently):
            refused = second.resume("aw001-delta")
        self.assertEqual(refused.status, RunStatus.FAILED)
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(delta_path.read_bytes(), persisted)   # never rewritten
        self.assertFalse(resume_info(refused.run_dir, refused.state).resumable)
        self.no_model_calls(*models)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(config)[0].resume("aw001-delta")

    def test_scope_delta_mutation_after_resume_validation_fails_closed_without_overwrite(self) -> None:
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): self.aw001_repair})
        claude = EditingFailingC02Claude(log=self.events)
        claude.edit = False
        config, failed = self.aw001_interrupted("aw001-toctou", luna=luna, claude=claude)
        run_dir = failed.run_dir
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.CLAUDE_C02)
        delta_path = run_dir / "repair/C02/scope_delta.json"
        self.assertEqual(hashlib.sha256(delta_path.read_bytes()).hexdigest(), checkpoint.scope_delta_sha256)
        delta = json.loads(delta_path.read_text())
        delta["added_paths"] = sorted([*delta["added_paths"], "src/b.py"])
        delta["requested_write_paths"] = sorted([*delta["requested_write_paths"], "src/b.py"])
        tampered = (json.dumps(delta, indent=2) + "\n").encode("utf-8")
        claimed: list[tuple[object, str]] = []

        def tamper_after_validation(claimed_dir: Path) -> None:
            # Runs after _validate_resume() succeeded and the run was claimed,
            # before _execute_v2(): the validate -> claim -> execute window.
            state = json.loads((claimed_dir / "state.json").read_text())
            claimed.append((read_checkpoint(claimed_dir), state["resume"]["status"]))
            delta_path.write_bytes(tampered)

        second, *models = self.orchestrator(config, reviews=[PASS])
        refused = second.resume("aw001-toctou", on_claimed=tamper_after_validation)
        self.assertEqual(claimed, [(checkpoint, "running")])   # initial validation passed
        self.assertEqual(refused.status, RunStatus.FAILED)
        self.assertEqual(refused.state["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(delta_path.read_bytes(), tampered)   # not repaired to canonical bytes
        self.assertFalse(resume_info(refused.run_dir, refused.state).resumable)
        self.no_model_calls(*models)
        with self.assertRaises(ResumeNotAllowedError):
            self.orchestrator(config)[0].resume("aw001-toctou")

    # -- REPAIR_PLANNER: transport failure after reviewer #1 REVISE ----------
    def repair_planner_transport_then_resume(
        self, run_id: str, *, review: str, repair_plan: str, repair,
    ) -> Path:
        config = self.make_config()
        planner = PlansThenTransportFails("planner", [SINGLE_PLAN], self.events)
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")})
        claude = FakeClaude({1: writer("src/a.py", "A = 20\n")}, log=self.events)
        orchestrator, _planner, reviewer, _luna, _claude = self.orchestrator(
            config, planner=planner, reviews=[review], luna=luna, claude=claude,
        )
        with self.count_pushes() as first_pushes:
            failed = self.run_approved(config, orchestrator, run_id,
                                       run_options=RunOptions.from_config(config))
        self.assertEqual(failed.state["failure"]["reason"], "LLM_FAILURE")
        self.assertEqual(len(planner.prompts), 2)   # initial plan + failed repair planner
        self.assertEqual(len(reviewer.prompts), 1)
        self.assertEqual([call["cycle"] for call in luna.calls], [1])
        self.assertEqual([call["cycle"] for call in claude.calls], [1])
        self.assertEqual(first_pushes.call_count, 1)   # the C01 candidate
        run_dir = failed.run_dir
        c01_bytes = (run_dir / "candidate/C01/commit.json").read_bytes()
        c01 = json.loads(c01_bytes)
        c01_tree = git(self.repo, "rev-parse", f"{c01['commit_sha']}^{{tree}}")
        revision = json.loads((run_dir / "revision/report.json").read_text())
        self.assertEqual(revision["tree_after"], c01_tree)   # Claude C01 really changed C01
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.cycle), (ResumePhase.REPAIR_PLANNER, 2))
        self.assertEqual(checkpoint.expected_head_sha, c01["commit_sha"])
        self.assertNotEqual(checkpoint.expected_head_sha, self.base_sha)
        self.assertEqual(checkpoint.expected_tree_sha, c01_tree)
        self.assertEqual(git(self.worktree(run_id), "rev-parse", "HEAD"), c01["commit_sha"])
        self.assertTrue(resume_info(run_dir, failed.state).resumable)
        self.assertFalse((run_dir / "repair/C02/scope_delta.json").exists())

        second, planner2, reviewer2, luna2, claude2 = self.orchestrator(
            config, plans=[repair_plan], reviews=[PASS], luna=FakeLuna({(2, "S01"): repair}),
        )
        with self.count_pushes() as pushes:
            resumed = second.resume(run_id)
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # Only the repair planner is replayed, from the exact C01 candidate.
        self.assertEqual(len(planner2.prompts), 1)
        self.assertIn(c01["commit_sha"], planner2.prompts[0])
        self.assertEqual([(call["cycle"], call["step"]) for call in luna2.calls], [(2, "S01")])
        self.assertEqual([call["cycle"] for call in claude2.calls], [2])
        self.assertEqual(len(reviewer2.prompts), 1)   # reviewer #2 only
        self.assertEqual(pushes.call_count, 1)   # the C02 candidate only
        self.assertEqual((run_dir / "candidate/C01/commit.json").read_bytes(), c01_bytes)
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())
        self.assertTrue(c02["pushed_at"])
        self.assertEqual(commit_parents(self.repo, c02["commit_sha"]), (c01["commit_sha"],))
        self.assertEqual(commit_parents(self.repo, c01["commit_sha"]), (self.base_sha,))
        self.assertEqual(resumed.state["commit_sha"], c02["commit_sha"])
        self.assertEqual(self.commits_on_run_branch(run_id), "2")
        return run_dir

    def test_repair_planner_transport_failure_resumes_repair_planner_only_replan(self) -> None:
        run_dir = self.repair_planner_transport_then_resume(
            "planner-replan", review=REVISE_REPLAN, repair_plan=self.aw001_repair_plan(),
            repair=self.aw001_repair,
        )
        delta = json.loads((run_dir / "repair/C02/scope_delta.json").read_text())
        self.assertEqual(delta["added_paths"], sorted(AW001_FILES))
        head = self.worktree("planner-replan")
        self.assertEqual(git(head, "show", "HEAD:docs/agent/CONTRACT.md"), "contract v2")
        self.assertEqual(git(head, "show", "HEAD:src/a.py"), "A = 20")

    def test_repair_planner_transport_failure_resumes_repair_planner_only_implementation(self) -> None:
        self.repair_planner_transport_then_resume(
            "planner-impl", review=REVISE_IMPLEMENTATION, repair_plan=REPAIR_PLAN,
            repair=writer("src/a.py", "A = 3\n"),
        )
        self.assertEqual(git(self.worktree("planner-impl"), "show", "HEAD:src/a.py"), "A = 3")

    # -- C02 fast-forward publication ------------------------------------------
    def test_fast_forward_publishes_exact_c02_chain(self) -> None:
        config = self.make_config(mode="fast-forward-base")
        self.publish_main_to_origin()
        harness = self
        observed: list[dict[str, str]] = []

        class ObservingReviewer(QueueClient):
            def complete(self, prompt: str):
                runs = subprocess.run(
                    ["git", "--git-dir", str(harness.bare), "for-each-ref",
                     "--format=%(objectname)", "refs/heads/harness/"],
                    capture_output=True, text=True, check=True,
                ).stdout.split()
                observed.append({
                    "local_main": git(harness.repo, "rev-parse", "refs/heads/main"),
                    "tracking_main": git(harness.repo, "rev-parse", "refs/remotes/origin/main"),
                    "origin_main": harness.origin_ref("refs/heads/main"),
                    "origin_run": runs[0] if len(runs) == 1 else ",".join(runs),
                })
                return super().complete(prompt)

        reviewer = ObservingReviewer("reviewer", [REVISE_IMPLEMENTATION, PASS], self.events)
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): writer("src/a.py", "A = 3\n")})
        git_argv: list[list[str]] = []
        real_run = subprocess.run

        def recording_run(argv, *args, **kwargs):
            if isinstance(argv, (list, tuple)) and argv and argv[0] == "git":
                git_argv.append([str(item) for item in argv])
            return real_run(argv, *args, **kwargs)

        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_PLAN], reviewer=reviewer, luna=luna,
        )
        with mock.patch("subprocess.run", side_effect=recording_run):
            result = self.run_approved(config, orchestrator, "ff-c02")
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        run_dir = result.run_dir
        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())["commit_sha"]
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())["commit_sha"]
        # BASE -> C01 -> C02, exactly.
        self.assertEqual(commit_parents(self.repo, c01), (self.base_sha,))
        self.assertEqual(commit_parents(self.repo, c02), (c01,))
        self.assertEqual(result.state["commit_sha"], c02)
        # main stays BASE during both reviews; the run branch holds each exact candidate.
        self.assertEqual(len(observed), 2)
        for index, candidate in enumerate((c01, c02)):
            self.assertEqual(observed[index], {
                "local_main": self.base_sha, "tracking_main": self.base_sha,
                "origin_main": self.base_sha, "origin_run": candidate,
            })
        # After PASS #2: local and remote main are exactly C02; no new commit.
        self.assertEqual(git(self.repo, "rev-parse", "refs/heads/main"), c02)
        self.assertEqual(self.origin_ref("refs/heads/main"), c02)
        self.assertEqual(git(self.repo, "rev-list", f"{self.base_sha}..main").split(), [c02, c01])
        publish = json.loads((run_dir / "publish.json").read_text())
        self.assertEqual((publish["mode"], publish["commit_sha"], publish["base_sha"]),
                         ("fast-forward-base", c02, self.base_sha))
        self.assertEqual(publish["run_branch_cleanup"]["status"], "success")
        self.assertNotIn("refs/heads/" + result.state["branch"], self.remote_refs())
        forbidden = {"merge", "rebase", "--force", "-f", "--force-with-lease", "--squash",
                     "cherry-pick", "reset"}
        for argv in git_argv:
            self.assertFalse(forbidden & set(argv), argv)
            if "push" in argv:
                self.assertFalse(any(item.startswith("+") for item in argv), argv)

    def test_c02_publication_failure_retries_publication_only(self) -> None:
        repair_plan = self.aw001_repair_plan()
        config = self.make_config(mode="fast-forward-base")
        self.publish_main_to_origin()
        lock = self.root / "main-locked"
        lock.write_text("locked\n")
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text(
            "#!/bin/sh\n"
            "while read old new ref; do\n"
            f"  if [ \"$ref\" = refs/heads/main ] && [ -e '{lock}' ]; then exit 1; fi\n"
            "done\n"
        )
        hook.chmod(0o755)
        luna = FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n"), (2, "S01"): self.aw001_repair})
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, repair_plan], reviews=[REVISE_REPLAN, PASS], luna=luna,
        )
        failed = self.run_approved(config, orchestrator, "ff-c02-retry",
                                   run_options=RunOptions.from_config(config))
        self.assertEqual(failed.state["failure"]["reason"], "PUSH_FAILED")
        run_dir = failed.run_dir
        c01 = json.loads((run_dir / "candidate/C01/commit.json").read_text())["commit_sha"]
        c02 = json.loads((run_dir / "candidate/C02/commit.json").read_text())["commit_sha"]
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.cycle, checkpoint.expected_head_sha),
                         (ResumePhase.PUBLISH, 2, c02))
        self.assertTrue(failed.state["publish"]["local_base_updated"])
        self.assertEqual(git(self.repo, "rev-parse", "refs/heads/main"), c02)
        self.assertEqual(self.origin_ref("refs/heads/main"), self.base_sha)
        lock.unlink()
        second, *models = self.orchestrator(config)
        with self.count_pushes() as pushes:
            resumed = second.resume("ff-c02-retry")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.no_model_calls(*models)
        self.assertEqual(pushes.call_count, 0)   # no candidate re-push
        self.assertEqual(self.origin_ref("refs/heads/main"), c02)
        self.assertEqual(commit_parents(self.repo, c02), (c01,))
        self.assertEqual(git(self.repo, "rev-list", f"{self.base_sha}..main").split(), [c02, c01])


if __name__ == "__main__":
    unittest.main()
