"""Operator plan recovery (REPLACE PLAN) and the AW-002 regression.

AW-002 failed with PLANNER_OUTPUT_INVALID at its PLANNER checkpoint.  These
tests paste a corrected READY META PLAN v2 into such a run and prove that it
continues through the normal approval and Luna S01..S07/S08 without ever
calling the planner again.  Real temporary Git repositories, real
orchestrator and Git primitives; planner, Luna, Claude and reviewers are
in-process fakes: no network, no LLM.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Callable
from unittest import mock
from urllib.parse import urlencode

from metaharness.approval import (
    ApprovalDecision,
    ApprovalError,
    compute_plan_identity_from_run,
    write_plan_approval,
)
from metaharness.models import RunStatus
from metaharness.orchestrator import Orchestrator
import metaharness.orchestrator as orchestrator_module
from metaharness.plan_recovery import (
    MAX_REPLACEMENT_PLAN_BYTES,
    PLAN_RECOVERY_ARTIFACT,
    PlanRecoveryError,
    plan_recovery_info,
)
from metaharness.resume import (
    ResumeCheckpoint,
    ResumePhase,
    read_checkpoint,
    resume_info,
    write_checkpoint,
)
from metaharness.state import RunStateStore
from metaharness.web.api import approve_run, get_run
from metaharness.web.pages import render_run
from metaharness.web.server import create_server
from tests.test_p28_full_pipeline import git, plan_text, step_block, write, writer
from tests.test_p29 import BLOCKED_PLAN, FakeClaude, FakeLuna, P29Harness, PASS, QueueClient, SPEC

# The observed AW-002 planner answer shape: BLOCKED carrying execution
# metadata, which the strict parser rejects as PLANNER_OUTPUT_INVALID.
AW002_INVALID_PLAN = """META PLAN v2

STATUS: BLOCKED
TITLE: Cannot safely plan

OBJECTIVE
The requested change cannot be planned safely.

CONSTRAINTS
The repository context is insufficient.

EXECUTION_MODE: STAGED
STEP_COUNT: 7
REVIEWER_PROFILE: reviewer

BLOCKERS
The required architectural information is missing.

END META PLAN
"""


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _ids(count: int) -> list[str]:
    return [f"S{number:02d}" for number in range(1, count + 1)]


def recovery_plan(count: int) -> str:
    """S01 edits src/a.py; every later step creates its own new file."""

    blocks = [step_block(1)] + [
        step_block(number, read=("src/a.py",), write_set=(), create=(f"src/s{number:02d}.py",))
        for number in range(2, count + 1)
    ]
    return plan_text(*blocks, title=f"AW-002 recovered {count}")


def luna_behaviors(count: int) -> dict[tuple[int, str], Any]:
    behaviors: dict[tuple[int, str], Any] = {(1, "S01"): writer("src/a.py", "A = 2\n")}
    for number in range(2, count + 1):
        behaviors[(1, f"S{number:02d}")] = writer(f"src/s{number:02d}.py", f"S = {number}\n")
    return behaviors


def _interrupt(_root: Path) -> None:
    raise KeyboardInterrupt


class PlanRecoveryHarness(P29Harness):
    def failed_planner_run(self, run_id: str):
        config = self.make_config()
        planner = QueueClient("planner", [AW002_INVALID_PLAN], self.events)
        failed = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=FakeLuna({}), reviser=FakeClaude(log=self.events),
        ).run_text(SPEC, run_id=run_id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(failed.state["failure"]["reason"], "PLANNER_OUTPUT_INVALID")
        self.assertEqual(read_checkpoint(failed.run_dir).phase, ResumePhase.PLANNER)
        self.assertEqual(len(planner.prompts), 1)
        return config, failed

    def blocked_planner_run(self, run_id: str):
        config = self.make_config()
        planner = QueueClient("planner", [BLOCKED_PLAN], self.events)
        blocked = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=FakeLuna({}), reviser=FakeClaude(log=self.events),
        ).run_text(SPEC, run_id=run_id)
        self.assertEqual(blocked.status, RunStatus.BLOCKED)
        self.assertEqual(blocked.state["status"], RunStatus.BLOCKED.value)
        self.assertEqual(blocked.state["failure"]["reason"], "PLANNER_BLOCKED")
        self.assertEqual(read_checkpoint(blocked.run_dir).phase, ResumePhase.PLANNER)
        self.assertEqual(len(planner.prompts), 1)
        return config, blocked

    def state(self, run_id: str) -> dict[str, Any]:
        return RunStateStore(self.runs / run_id / "state.json").load()

    def wait_for_approval_gate(
        self, run_id: str, thread: threading.Thread, *, expected_step_count: int | None = None,
    ) -> None:
        """Wait for the settled approval gate, not the transient resume claim.

        A PLAN_APPROVAL resume claims the run as ``awaiting_plan_approval``,
        re-validates the plan under ``planning`` and only then opens the gate.
        The cycle record is written just before the gate, so both conditions
        together mean the worker is really waiting for a decision.
        """

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = self.state(run_id)
            cycles = state.get("cycles") if isinstance(state.get("cycles"), list) else []
            cycle = cycles[0] if cycles and isinstance(cycles[0], dict) else {}
            steps = cycle.get("steps_summary") if isinstance(cycle.get("steps_summary"), list) else []
            if (
                state["status"] == "awaiting_plan_approval"
                and state.get("cycles")
                and (expected_step_count is None or len(steps) == expected_step_count)
            ):
                return
            if not thread.is_alive():
                self.fail(f"run {run_id} stopped before the approval gate: {state.get('failure')}")
            time.sleep(0.01)
        self.fail(f"run {run_id} never reached the approval gate")

    def recover_in_thread(
        self, config: Any, run_id: str, raw: str, *, planner: QueueClient,
        luna: FakeLuna, reviews: list[str],
    ) -> tuple[threading.Thread, dict[str, Any]]:
        orchestrator = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", reviews, self.events),
            agent=luna, reviser=FakeClaude(log=self.events),
        )
        holder: dict[str, Any] = {}

        def target() -> None:
            try:
                holder["result"] = orchestrator.recover_plan(run_id, raw)
            except BaseException as exc:  # surfaced by the assertions
                holder["error"] = exc

        thread = threading.Thread(target=target)
        thread.start()
        # A failing assertion must never leave a worker polling the approval
        # gate while the temporary run directory is removed.
        self.addCleanup(self.release_worker, run_id, thread)
        return thread, holder

    def release_worker(self, run_id: str, thread: threading.Thread) -> None:
        if not thread.is_alive():
            return
        try:
            directory = self.runs / run_id
            write_plan_approval(
                directory, decision=ApprovalDecision.REJECT,
                identity=compute_plan_identity_from_run(directory), source="test",
            )
        except (ApprovalError, OSError):
            pass
        thread.join(timeout=60)

    def approve(self, config: Any, run_id: str, count: int) -> None:
        approve_run(
            config.runs_root, run_id, "APPROVE", config=config, reviewer_profile="reviewer",
            step_profiles={step_id: "luna" for step_id in _ids(count)},
        )

    def wait_and_approve(self, config: Any, run_id: str, count: int, thread: threading.Thread) -> None:
        self.wait_for_approval_gate(run_id, thread)
        self.approve(config, run_id, count)

    def assert_recovered_awaiting_approval(
        self, config: Any, run_id: str, *, count: int, replacement: str,
        invalid_raw: bytes, planner: QueueClient, luna: FakeLuna,
    ) -> None:
        run_dir = self.runs / run_id
        state = self.state(run_id)
        self.assertEqual(state["status"], "awaiting_plan_approval", state.get("failure"))
        # No planner, no worker, no worktree, no branch before approval.
        self.assertEqual(planner.prompts, [])
        self.assertEqual(luna.calls, [])
        self.assertFalse((self.root / "worktrees" / run_id).exists())
        self.assertEqual(git(self.repo, "branch", "--list", "harness/*"), "")
        # Old planner answer archived with the attempt mechanism; inputs kept.
        self.assertEqual((run_dir / "attempts" / "01" / "planner.raw.md").read_bytes(), invalid_raw)
        self.assertEqual((run_dir / "planner.raw.md").read_text(encoding="utf-8"), replacement)
        self.assertEqual((run_dir / "spec.md").read_text(encoding="utf-8"), SPEC)
        self.assertTrue((run_dir / "context.txt").is_file())
        self.assertFalse((run_dir / "planner.usage.json").exists())
        record = json.loads((run_dir / PLAN_RECOVERY_ARTIFACT).read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["source"], "operator")
        self.assertEqual(record["previous_raw_sha256"], _sha(invalid_raw))
        self.assertEqual(record["replacement_raw_sha256"], _sha(replacement))
        self.assertIs(record["planner_called"], False)
        self.assertEqual(record["archived_attempt"], "attempts/01")
        self.assertTrue(record["recovered_at"])
        bundle = json.loads((run_dir / "implementation_bundle.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["id"] for entry in bundle["steps"]], _ids(count))
        for step_id in _ids(count):
            self.assertTrue((run_dir / "steps" / step_id / "contract.md").is_file())
        checkpoint = read_checkpoint(run_dir)
        self.assertEqual(checkpoint.phase, ResumePhase.PLAN_APPROVAL)
        self.assertEqual(checkpoint.plan_identity.raw_sha256, _sha(replacement))
        self.assertEqual(checkpoint.expected_head_sha, self.base_sha)
        self.assertEqual(state["plan_identity"]["raw_sha256"], _sha(replacement))
        self.assertEqual(state["planner"]["source"], "operator_recovery")
        page = render_run(get_run(self.runs, run_id, config=config), "tok", config=config)
        self.assertIn("Recovered plan — awaiting approval", page)
        # No retry action is offered any more (the archived diagnostics of the
        # failed attempt may still quote its own historical label).
        self.assertNotIn('<button class="resume" type="submit">Retry planner</button>', page)
        self.assertNotIn('action="/runs/{run_id}/resume"'.format(run_id=run_id), page)
        self.assertIn(f'name="step_profile__S{count:02d}"', page)

    def assert_approval_binds_replacement(self, run_id: str, replacement: str, count: int) -> None:
        run_dir = self.runs / run_id
        approval = json.loads((run_dir / "plan_approval.json").read_text(encoding="utf-8"))
        self.assertEqual(approval["decision"], "APPROVE")
        self.assertEqual(approval["raw_sha256"], _sha(replacement))
        self.assertEqual(approval["contract_sha256"], _sha((run_dir / "implementation_contract.md").read_bytes()))
        self.assertEqual(approval["bundle_sha256"], _sha((run_dir / "implementation_bundle.json").read_bytes()))
        self.assertEqual(approval["execution_sha256"], _sha((run_dir / "execution_selection.json").read_bytes()))
        selection = json.loads((run_dir / "execution_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["schema_version"], 5)
        self.assertEqual([item["step_id"] for item in selection["steps"]], _ids(count))


class PlanRecoveryTests(PlanRecoveryHarness):
    def assert_blocked_recovery_persistence_failure(self, run_id: str) -> tuple[Any, ...]:
        config, blocked = self.blocked_planner_run(run_id)
        replacement = recovery_plan(7)
        planner = QueueClient("planner", [], self.events)
        luna = FakeLuna(luna_behaviors(7))
        orchestrator = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=luna, reviser=FakeClaude(log=self.events),
        )

        return config, replacement, blocked, planner, luna, orchestrator

    def assert_recovery_failure_keeps_blocked_shape(
        self, run_id: str, planner: QueueClient, luna: FakeLuna,
    ) -> None:
        run_dir = self.runs / run_id
        state = self.state(run_id)
        self.assertEqual(state["status"], RunStatus.BLOCKED.value)
        self.assertEqual(state["failure"]["reason"], "PLANNER_BLOCKED")
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.PLANNER)
        self.assertFalse((self.root / "worktrees" / run_id).exists())
        self.assertEqual(git(self.repo, "branch", "--list", "harness/*"), "")
        self.assertEqual(planner.prompts, [])
        self.assertEqual(luna.calls, [])
        self.assertTrue(plan_recovery_info(run_dir, state).eligible)

    def finish_blocked_recovery_after_persistence_failure(
        self, config: Any, run_id: str, replacement: str,
        planner: QueueClient, luna: FakeLuna,
    ) -> None:
        thread, holder = self.recover_in_thread(
            config, run_id, replacement, planner=planner, luna=luna, reviews=[PASS],
        )
        self.wait_for_approval_gate(run_id, thread, expected_step_count=7)
        self.assertNotIn("error", holder)
        self.assertEqual(planner.prompts, [])
        self.assertEqual(luna.calls, [])
        self.assertEqual(read_checkpoint(self.runs / run_id).phase, ResumePhase.PLAN_APPROVAL)
        self.approve(config, run_id, 7)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", holder)
        self.assertEqual(holder["result"].status, RunStatus.PUBLISHED, holder["result"].state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls], _ids(7))
        self.assertEqual(planner.prompts, [])

    def test_blocked_recovery_checkpoint_failure_preserves_recoverability(self) -> None:
        config, replacement, _blocked, planner, luna, orchestrator = (
            self.assert_blocked_recovery_persistence_failure("aw-002-blocked-checkpoint-crash")
        )
        with mock.patch.object(
            orchestrator_module.Orchestrator,
            "_write_phase_checkpoint",
            side_effect=RuntimeError("checkpoint persistence crash"),
        ):
            with self.assertRaises(RuntimeError):
                orchestrator.recover_plan("aw-002-blocked-checkpoint-crash", replacement)

        self.assert_recovery_failure_keeps_blocked_shape(
            "aw-002-blocked-checkpoint-crash", planner, luna,
        )
        self.finish_blocked_recovery_after_persistence_failure(
            config, "aw-002-blocked-checkpoint-crash", replacement, planner, luna,
        )

    def test_blocked_recovery_artifact_failure_preserves_recoverability(self) -> None:
        config, replacement, _blocked, planner, luna, orchestrator = (
            self.assert_blocked_recovery_persistence_failure("aw-002-blocked-artifact-crash")
        )
        with mock.patch.object(
            orchestrator_module,
            "persist_recovered_plan_artifacts",
            side_effect=RuntimeError("artifact persistence crash"),
        ):
            with self.assertRaises(RuntimeError):
                orchestrator.recover_plan("aw-002-blocked-artifact-crash", replacement)

        self.assert_recovery_failure_keeps_blocked_shape(
            "aw-002-blocked-artifact-crash", planner, luna,
        )
        self.finish_blocked_recovery_after_persistence_failure(
            config, "aw-002-blocked-artifact-crash", replacement, planner, luna,
        )

    def test_aw002_blocked_planner_recovers_with_seven_step_plan_without_planner(self) -> None:
        config, blocked = self.blocked_planner_run("aw-002-blocked-seven")
        run_dir = blocked.run_dir
        blocked_raw = (run_dir / "planner.raw.md").read_bytes()
        self.assertTrue(plan_recovery_info(run_dir, blocked.state).eligible)
        page = render_run(get_run(self.runs, "aw-002-blocked-seven", config=config), "tok", config=config)
        self.assertIn("REPLACE PLAN", page)
        self.assertNotIn("Retry planner", page)

        replacement = recovery_plan(7)
        planner = QueueClient("planner", [], self.events)
        luna = FakeLuna(luna_behaviors(7))
        thread, holder = self.recover_in_thread(
            config, "aw-002-blocked-seven", replacement, planner=planner, luna=luna, reviews=[PASS],
        )
        self.wait_for_approval_gate("aw-002-blocked-seven", thread, expected_step_count=7)
        self.assertNotIn("error", holder)
        self.assert_recovered_awaiting_approval(
            config, "aw-002-blocked-seven", count=7, replacement=replacement,
            invalid_raw=blocked_raw, planner=planner, luna=luna,
        )
        self.assertEqual(self.state("aw-002-blocked-seven")["status"], RunStatus.AWAITING_PLAN_APPROVAL.value)
        self.approve(config, "aw-002-blocked-seven", 7)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        self.assertEqual(holder["result"].status, RunStatus.PUBLISHED, holder["result"].state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls], _ids(7))
        self.assertEqual(planner.prompts, [])

    def test_aw002_blocked_planner_recovers_with_twelve_step_plan(self) -> None:
        config, blocked = self.blocked_planner_run("aw-002-blocked-twelve")
        blocked_raw = (blocked.run_dir / "planner.raw.md").read_bytes()
        replacement = recovery_plan(12)
        planner = QueueClient("planner", [], self.events)
        luna = FakeLuna(luna_behaviors(12))
        thread, holder = self.recover_in_thread(
            config, "aw-002-blocked-twelve", replacement, planner=planner, luna=luna, reviews=[PASS],
        )
        self.wait_for_approval_gate("aw-002-blocked-twelve", thread, expected_step_count=12)
        self.assertNotIn("error", holder)
        self.assert_recovered_awaiting_approval(
            config, "aw-002-blocked-twelve", count=12, replacement=replacement,
            invalid_raw=blocked_raw, planner=planner, luna=luna,
        )
        self.approve(config, "aw-002-blocked-twelve", 12)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        self.assertEqual(holder["result"].status, RunStatus.PUBLISHED, holder["result"].state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls], _ids(12))
        self.assert_approval_binds_replacement("aw-002-blocked-twelve", replacement, 12)
        self.assertEqual(planner.prompts, [])

    def test_blocked_without_planner_checkpoint_cannot_recover(self) -> None:
        config, blocked = self.blocked_planner_run("aw-002-blocked-no-checkpoint")
        run_dir = blocked.run_dir
        before = (run_dir / "state.json").read_bytes()
        (run_dir / "resume_checkpoint.json").unlink()
        self.assertFalse(plan_recovery_info(run_dir, self.state("aw-002-blocked-no-checkpoint")).eligible)
        with self.assertRaises(PlanRecoveryError):
            Orchestrator(config, planner_client=QueueClient("planner", [], self.events)).recover_plan(
                "aw-002-blocked-no-checkpoint", recovery_plan(7)
            )
        self.assertEqual((run_dir / "state.json").read_bytes(), before)

    def test_blocked_with_approval_worktree_or_execution_artifact_cannot_recover(self) -> None:
        cases = {
            "approval": "plan_approval.json",
            "execution": "execution_selection.json",
            "reviewer": "review.json",
        }
        for suffix, artifact in cases.items():
            with self.subTest(case=suffix):
                config, blocked = self.blocked_planner_run(f"aw-002-blocked-{suffix}")
                run_dir = blocked.run_dir
                write(run_dir / artifact, "{}\n")
                try:
                    self.assertFalse(plan_recovery_info(run_dir, self.state(f"aw-002-blocked-{suffix}")).eligible)
                    with self.assertRaises(PlanRecoveryError):
                        Orchestrator(config, planner_client=QueueClient("planner", [], self.events)).recover_plan(
                            f"aw-002-blocked-{suffix}", recovery_plan(7)
                        )
                    self.assertEqual(self.state(f"aw-002-blocked-{suffix}")["status"], RunStatus.BLOCKED.value)
                    self.assertTrue((run_dir / artifact).is_file())
                finally:
                    (run_dir / artifact).unlink()

        config, blocked = self.blocked_planner_run("aw-002-blocked-worktree")
        worktree = self.root / "worktrees" / "aw-002-blocked-worktree"
        worktree.mkdir(parents=True)
        try:
            with self.assertRaises(PlanRecoveryError):
                Orchestrator(config, planner_client=QueueClient("planner", [], self.events)).recover_plan(
                    "aw-002-blocked-worktree", recovery_plan(7)
                )
            self.assertEqual(self.state("aw-002-blocked-worktree")["status"], RunStatus.BLOCKED.value)
        finally:
            worktree.rmdir()

    def test_invalid_blocked_replacement_leaves_state_and_artifacts_intact(self) -> None:
        config, blocked = self.blocked_planner_run("aw-002-blocked-invalid")
        run_dir = blocked.run_dir

        def snapshot() -> dict[str, bytes]:
            return {
                path.relative_to(run_dir).as_posix(): path.read_bytes()
                for path in sorted(run_dir.rglob("*")) if path.is_file()
            }

        before = snapshot()
        with self.assertRaises(PlanRecoveryError):
            Orchestrator(config, planner_client=QueueClient("planner", [], self.events)).recover_plan(
                "aw-002-blocked-invalid", BLOCKED_PLAN
            )
        self.assertEqual(snapshot(), before)
        self.assertEqual(self.state("aw-002-blocked-invalid")["status"], RunStatus.BLOCKED.value)
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.PLANNER)

    def test_aw002_failed_planner_recovers_with_seven_step_plan_without_planner(self) -> None:
        config, failed = self.failed_planner_run("aw-002")
        run_dir = failed.run_dir
        invalid_raw = (run_dir / "planner.raw.md").read_bytes()
        self.assertTrue(plan_recovery_info(run_dir, failed.state).eligible)
        page = render_run(get_run(self.runs, "aw-002", config=config), "tok", config=config)
        self.assertIn('action="/runs/aw-002/recover-plan"', page)
        self.assertIn("REPLACE PLAN", page)

        replacement = recovery_plan(7)
        planner = QueueClient("planner", [], self.events)
        luna = FakeLuna(luna_behaviors(7))
        thread, holder = self.recover_in_thread(
            config, "aw-002", replacement, planner=planner, luna=luna, reviews=[PASS],
        )
        self.wait_for_approval_gate("aw-002", thread)
        self.assertNotIn("error", holder)
        self.assert_recovered_awaiting_approval(
            config, "aw-002", count=7, replacement=replacement,
            invalid_raw=invalid_raw, planner=planner, luna=luna,
        )

        self.approve(config, "aw-002", 7)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", holder)
        result = holder["result"]
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assert_approval_binds_replacement("aw-002", replacement, 7)
        self.assertEqual([call["step"] for call in luna.calls if call["cycle"] == 1], _ids(7))
        self.assertEqual(planner.prompts, [])
        diagnostics = (run_dir / "diagnostics.md").read_text(encoding="utf-8")
        self.assertIn("plan source: operator recovery", diagnostics)
        self.assertNotIn("plan source: planner model completion", diagnostics)

    def test_eight_step_recovery_is_bound_to_stored_base_not_moved_main(self) -> None:
        config, failed = self.failed_planner_run("aw-002-eight")
        invalid_raw = (failed.run_dir / "planner.raw.md").read_bytes()
        # main moves after the failure: the run stays bound to its stored BASE.
        write(self.repo / "README.md", "moved main\n")
        git(self.repo, "commit", "-qam", "main moved")
        self.assertNotEqual(git(self.repo, "rev-parse", "main"), self.base_sha)

        replacement = recovery_plan(8)
        planner = QueueClient("planner", [], self.events)
        luna = FakeLuna(luna_behaviors(8))
        thread, holder = self.recover_in_thread(
            config, "aw-002-eight", replacement, planner=planner, luna=luna, reviews=[PASS],
        )
        self.wait_for_approval_gate("aw-002-eight", thread)
        self.assertNotIn("error", holder)
        self.assert_recovered_awaiting_approval(
            config, "aw-002-eight", count=8, replacement=replacement,
            invalid_raw=invalid_raw, planner=planner, luna=luna,
        )
        self.approve(config, "aw-002-eight", 8)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        result = holder["result"]
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assert_approval_binds_replacement("aw-002-eight", replacement, 8)
        self.assertEqual([call["step"] for call in luna.calls if call["cycle"] == 1], _ids(8))
        self.assertEqual(result.state["base_sha"], self.base_sha)
        self.assertEqual(planner.prompts, [])

    def test_recovered_run_interrupted_at_s07_resumes_at_s07_without_planner(self) -> None:
        config, _failed = self.failed_planner_run("aw-002-s07")
        replacement = recovery_plan(7)
        planner = QueueClient("planner", [], self.events)
        behaviors = luna_behaviors(7)
        behaviors[(1, "S07")] = _interrupt
        luna = FakeLuna(behaviors)
        thread, holder = self.recover_in_thread(
            config, "aw-002-s07", replacement, planner=planner, luna=luna, reviews=[],
        )
        self.wait_and_approve(config, "aw-002-s07", 7, thread)
        thread.join(timeout=120)
        self.assertFalse(thread.is_alive())
        interrupted = holder["result"]
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED, interrupted.state.get("failure"))
        checkpoint = read_checkpoint(interrupted.run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.step_id), (ResumePhase.INITIAL_STEP, "S07"))
        info = resume_info(interrupted.run_dir, interrupted.state)
        self.assertEqual((info.resumable, info.step_id), (True, "S07"), info.reason)
        self.assertEqual([call["step"] for call in luna.calls], _ids(7))

        resumed_luna = FakeLuna({(1, "S07"): writer("src/s07.py", "S = 7\n")})
        resumed = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [PASS], self.events),
            agent=resumed_luna, reviser=FakeClaude(log=self.events),
        ).resume("aw-002-s07")
        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in resumed_luna.calls], ["S07"])
        self.assertEqual(planner.prompts, [])

    def test_invalid_replacements_leave_the_run_authority_unchanged(self) -> None:
        config, failed = self.failed_planner_run("aw-002-invalid")
        run_dir = failed.run_dir

        def snapshot() -> dict[str, bytes]:
            return {
                path.relative_to(run_dir).as_posix(): path.read_bytes()
                for path in sorted(run_dir.rglob("*")) if path.is_file()
            }

        before = snapshot()
        nine = plan_text(
            step_block(1),
            *[step_block(number, read=("src/a.py",), write_set=(), create=(f"src/s{number:02d}.py",))
              for number in range(2, 10)],
        )
        cases = {
            "100 steps": plan_text(*[step_block(number) for number in range(1, 101)]),
            "S100 id": recovery_plan(8).replace("BEGIN STEP S08", "BEGIN STEP S100").replace("END STEP S08", "END STEP S100"),
            "blocked": BLOCKED_PLAN,
            "AW-002 blocked with execution metadata": AW002_INVALID_PLAN,
            "unknown implementer": recovery_plan(7).replace("IMPLEMENTER_PROFILE: luna", "IMPLEMENTER_PROFILE: ghost", 1),
            "reviewer outside catalogue": recovery_plan(7).replace("REVIEWER_PROFILE: reviewer", "REVIEWER_PROFILE: claude"),
            "prose outside envelope": "Here is the plan:\n" + recovery_plan(7),
            "empty": "   \n",
            "too large": recovery_plan(7) + "#" * MAX_REPLACEMENT_PLAN_BYTES,
            "not text": b"META PLAN v2",
        }
        planner = QueueClient("planner", [], self.events)
        orchestrator = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=FakeLuna({}), reviser=FakeClaude(log=self.events),
        )
        for name, raw in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(PlanRecoveryError):
                    orchestrator.recover_plan("aw-002-invalid", raw)  # type: ignore[arg-type]
                self.assertEqual(snapshot(), before)
        self.assertEqual(planner.prompts, [])
        self.assertEqual(read_checkpoint(run_dir).phase, ResumePhase.PLANNER)
        self.assertEqual(self.state("aw-002-invalid"), failed.state)

    def test_preconditions_fail_closed(self) -> None:
        config, failed = self.failed_planner_run("aw-002-pre")
        run_dir = failed.run_dir
        store = RunStateStore(run_dir / "state.json")
        checkpoint_bytes = (run_dir / "resume_checkpoint.json").read_bytes()
        planner = QueueClient("planner", [], self.events)
        orchestrator = Orchestrator(
            config, planner_client=planner,
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=FakeLuna({}), reviser=FakeClaude(log=self.events),
        )
        replacement = recovery_plan(7)
        branch = "harness/other-title/aw-002-pre"
        worktree = self.root / "worktrees" / "aw-002-pre"

        def restore_checkpoint() -> None:
            (run_dir / "resume_checkpoint.json").write_bytes(checkpoint_bytes)

        def touch(relative: str) -> Callable[[], None]:
            return lambda: write(run_dir / relative, "{}\n")

        def remove(relative: str) -> Callable[[], None]:
            return lambda: (run_dir / relative).unlink()

        cases: dict[str, tuple[Callable[[], None], Callable[[], None]]] = {
            "plan approval exists": (touch("plan_approval.json"), remove("plan_approval.json")),
            "execution selection exists": (touch("execution_selection.json"), remove("execution_selection.json")),
            "Luna artifact exists": (touch("steps/S01/step.json"), lambda: __import__("shutil").rmtree(run_dir / "steps")),
            "reviewer artifact exists": (touch("review.json"), remove("review.json")),
            "run worktree exists": (lambda: worktree.mkdir(parents=True), lambda: worktree.rmdir()),
            "run branch exists": (
                lambda: git(self.repo, "branch", branch, self.base_sha),
                lambda: git(self.repo, "branch", "-D", branch),
            ),
            "checkpoint is not PLANNER": (
                lambda: write_checkpoint(run_dir, ResumeCheckpoint(ResumePhase.CONTEXT, 1, None, None, None, None, None)),
                restore_checkpoint,
            ),
            "stored BASE tree changed": (
                lambda: write_checkpoint(run_dir, ResumeCheckpoint(
                    ResumePhase.PLANNER, 1, None, self.base_sha, "0" * 40, None, None,
                )),
                restore_checkpoint,
            ),
            "stored BASE SHA unknown": (
                lambda: store.update(status="failed", base_sha="0" * 40),
                lambda: store.update(status="failed", base_sha=self.base_sha),
            ),
            "failure is not a planner failure": (
                lambda: store.update(status="failed", failure={"reason": "AGENT_FAILED"}),
                lambda: store.update(status="failed", failure=failed.state["failure"]),
            ),
            "run is not failed": (
                lambda: store.update(status="blocked"),
                lambda: store.update(status="failed"),
            ),
        }
        for name, (mutate, undo) in cases.items():
            with self.subTest(case=name):
                mutate()
                try:
                    with self.assertRaises(PlanRecoveryError):
                        orchestrator.recover_plan("aw-002-pre", replacement)
                    self.assertEqual((run_dir / "planner.raw.md").read_text(encoding="utf-8"), AW002_INVALID_PLAN)
                    self.assertFalse((run_dir / PLAN_RECOVERY_ARTIFACT).exists())
                    self.assertFalse((run_dir / "attempts").exists())
                finally:
                    undo()
        self.assertEqual(planner.prompts, [])
        self.assertTrue(plan_recovery_info(run_dir, self.state("aw-002-pre")).eligible)


class PlanRecoveryWebTests(PlanRecoveryHarness):
    def setUp(self) -> None:
        super().setUp()
        self.config, self.failed = self.failed_planner_run("aw-002-web")
        self.server = create_server(self.config, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        super().tearDown()

    def post(self, path: str, body: bytes, content_type: str, token: str | None = None) -> tuple[int, Any]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        if token is not None:
            headers["X-MetaHarness-Token"] = token
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read()
        connection.close()
        try:
            payload = json.loads(content) if content else None
        except ValueError:
            payload = content
        return response.status, payload

    def post_json(self, path: str, payload: Any, token: str | None = None) -> tuple[int, Any]:
        return self.post(path, json.dumps(payload).encode(), "application/json", token)

    def test_recover_plan_routes_are_protected_bounded_and_never_call_the_planner(self) -> None:
        path = "/api/runs/aw-002-web/recover-plan"
        token = self.server.token
        run_dir = self.failed.run_dir
        before = (run_dir / "state.json").read_bytes()
        self.assertEqual(self.post_json(path, {"plan": recovery_plan(7)})[0], 403)
        self.assertEqual(self.post_json(path, {"plan": recovery_plan(7)}, "wrong")[0], 403)
        self.assertEqual(self.post_json(path, {"plan": recovery_plan(7), "spec": "x"}, token)[0], 400)
        self.assertEqual(self.post_json(path, {"plan": "#" * (MAX_REPLACEMENT_PLAN_BYTES + 1)}, token)[0], 400)
        self.assertEqual(self.post_json(path, {"plan": BLOCKED_PLAN}, token)[0], 400)
        self.assertEqual(self.post_json("/api/runs/..%2F/recover-plan", {"plan": "x"}, token)[0], 400)
        form = urlencode({"_token": "wrong", "plan": recovery_plan(7)}).encode()
        self.assertEqual(
            self.post("/runs/aw-002-web/recover-plan", form, "application/x-www-form-urlencoded")[0], 403
        )
        self.assertEqual((run_dir / "state.json").read_bytes(), before)

        status, payload = self.post_json(path, {"plan": recovery_plan(7)}, token)
        self.assertEqual(status, 202, payload)
        deadline = time.monotonic() + 30
        while self.state("aw-002-web")["status"] != "awaiting_plan_approval" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.state("aw-002-web")["status"], "awaiting_plan_approval")
        # A second recovery is refused: the run left its PLANNER checkpoint.
        self.assertEqual(self.post_json(path, {"plan": recovery_plan(7)}, token)[0], 409)
        # End the waiting worker through the normal gate.
        status, _ = self.post_json("/api/runs/aw-002-web/approval", {"decision": "REJECT"}, token)
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 30
        while self.state("aw-002-web")["status"] != "plan_rejected" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.state("aw-002-web")["status"], "plan_rejected")
        # The real configured planner endpoint was never contacted.
        self.assertFalse((run_dir / "planner.usage.json").exists())
        self.assertFalse((run_dir / "planner.request.txt").exists())


if __name__ == "__main__":
    unittest.main()
