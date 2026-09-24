"""Durable step contract repair transactions across planner outages and resumes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.gitops import GitError
from metaharness.llm.chat import LLMError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.orchestration.pipeline_v2 import PipelineFailure
from metaharness.resume import resume_info
from metaharness.run_options import RunOptions
from tests.pipeline_support import PipelineHarness, initial_plan, review, write
from tests.test_pipeline_v2_machine import SPEC, STEP, repaired_step_contract

OUTAGE = "LLM endpoint returned HTTP 503 after 3 attempt(s)"
REPAIR_ID = "contract-repair:cycle-001:S01:01"


def mismatch(_request) -> str:
    return CONTRACT_MISMATCH_HEADER + "\nThe step invariants contradict the SPEC."


class ContractRepairTransactionTests(PipelineHarness):
    def step_dir(self, run_id: str = "run") -> Path:
        return self.run_dir(run_id) / "cycles/001/implementation/steps/S01"

    def repair_slots(self, run_id: str = "run") -> list[str]:
        root = self.step_dir(run_id) / "contract_repairs"
        return sorted(path.name for path in root.iterdir()) if root.is_dir() else []

    def transaction(self, number: int = 1, run_id: str = "run") -> dict:
        path = self.step_dir(run_id) / f"contract_repairs/{number:02d}/transaction.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def semantic_records(self, run_id: str = "run") -> list[dict]:
        return [
            item for item in self.state(run_id).get("recovery_attempts", [])
            if item.get("budget_key") == "contract_repairs"
        ]

    def resume(self, planner: list, reviewer: list | None = None):
        return self.orchestrator(
            self.config(), planner=planner, reviewer=reviewer or ["unused"],
        ).resume("run")

    def wait_on_outage(self) -> None:
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP), LLMError(OUTAGE)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL, self.state().get("failure"))

    def test_planner_outage_resumes_the_pending_repair_without_replaying_the_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        self.wait_on_outage()

        self.assertEqual(len(self.workers.calls), 1)
        first_repair_requests = self.planner.requests[1:]
        self.assertEqual(len(first_repair_requests), 1)
        state = self.state()
        self.assertEqual(state["failure"]["reason"], "LLM_FAILURE")
        checkpoint = self.checkpoint()
        self.assertEqual((checkpoint["phase"], checkpoint["step_id"]), ("implement_step", "S01"))
        self.assertEqual(state["contract_repair"]["pending_operation"], "contract_repair")
        self.assertEqual(state["contract_repair"]["contract_repair_number"], 1)
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual(transaction["status"], "waiting_external")
        self.assertEqual(transaction["repair_id"], REPAIR_ID)
        self.assertEqual(transaction["tree_sha"], self.git_tree())
        self.assertEqual(self.semantic_records(), [])
        self.assertTrue(resume_info(self.run_dir(), state).resumable)

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The original worker plus the post-repair worker only.
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.planner.requests, first_repair_requests)
        self.assertEqual(self.repair_slots(), ["01"])
        self.assertEqual(self.transaction()["status"], "completed")
        (record,) = self.semantic_records()
        self.assertEqual((record["attempt"], record["budget_consumed"]), (1, 1))
        self.assertEqual(record["operation_id"], REPAIR_ID)
        self.assertIn("recovery.resumed", self.trace_names())

    def test_a_durable_raw_answer_is_reparsed_without_a_provider_call(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.planning_v2.parse_step_contract_repair",
            side_effect=KeyboardInterrupt(),
        ):
            # The process is interrupted after the paid answer became durable.
            interrupted = self.orchestrator(
                self.config(), planner=[initial_plan(STEP), repaired_step_contract()],
                reviewer=[review()],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
        self.assertEqual(len(self.planner.requests), 2)
        self.assertEqual(self.transaction()["status"], "planner_response_durable")

        resumed = self.resume(["provider must not be called"], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.transaction()["status"], "completed")
        self.assertEqual(len(self.semantic_records()), 1)

    def test_repeated_outages_keep_one_semantic_repair(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        self.wait_on_outage()
        for transport_attempt in (2, 3):
            waiting = self.resume([LLMError(OUTAGE)])
            self.assertEqual(waiting.status, RunStatus.WAITING_EXTERNAL)
            self.assertEqual(len(self.workers.calls), 1)
            self.assertEqual(self.repair_slots(), ["01"])
            transaction = self.transaction()
            self.assertEqual(transaction["repair_number"], 1)
            self.assertEqual(transaction["planner_transport_attempt"], transport_attempt)
            self.assertEqual(self.semantic_records(), [])

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.transaction()["planner_transport_attempt"], 4)
        self.assertEqual([item["operation_id"] for item in self.semantic_records()], [REPAIR_ID])

    def test_two_genuine_repairs_fit_the_budget_and_transport_is_not_counted(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER, mismatch, mismatch, write("feature.txt", "good\n"),
        )
        self.wait_on_outage()
        waiting = self.resume([repaired_step_contract(), LLMError(OUTAGE)])
        self.assertEqual(waiting.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.repair_slots(), ["01", "02"])

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 3)
        self.assertEqual(self.repair_slots(), ["01", "02"])
        self.assertEqual(
            [item["operation_id"] for item in self.semantic_records()],
            [REPAIR_ID, "contract-repair:cycle-001:S01:02"],
        )

    def test_a_third_genuine_mismatch_exhausts_the_repair_budget(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, mismatch, mismatch)
        result = self.orchestrator(
            self.config(max_step_contract_repairs=2),
            planner=[initial_plan(STEP), repaired_step_contract()], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        self.assertEqual(len(self.workers.calls), 3)
        self.assertEqual(len(self.planner.requests), 3)
        self.assertEqual(self.repair_slots(), ["01", "02"])
        self.assertEqual(len(self.semantic_records()), 2)

    def test_a_legacy_pending_slot_is_adopted_without_replaying_the_worker(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        self.wait_on_outage()
        slot = self.step_dir() / "contract_repairs/01"
        # The shape a slot had before transaction markers existed.
        (slot / "transaction.json").unlink()
        (slot / "mismatch.json").unlink()

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 2)
        self.assertTrue(self.transaction()["adopted_legacy_slot"])
        self.assertEqual(self.transaction()["status"], "completed")

    def test_an_unidentifiable_legacy_slot_requires_an_operator(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        self.wait_on_outage()
        slot = self.step_dir() / "contract_repairs/01"
        for name in ("transaction.json", "mismatch.json", "planner.request.txt"):
            (slot / name).unlink()

        result = self.resume([repaired_step_contract()])

        self.assertNotIn(result.status, {RunStatus.COMMITTED, RunStatus.WAITING_EXTERNAL})
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(len(self.workers.calls), 1)
        self.assertEqual(self.planner.requests, [])
        self.assertTrue((slot / "request.meta.json").is_file())

    def test_a_changed_tree_is_a_resume_integrity_failure(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        self.wait_on_outage()
        slot = self.step_dir() / "contract_repairs/01"
        transaction = self.transaction()
        transaction["tree_sha"] = "0" * 40
        (slot / "transaction.json").write_text(json.dumps(transaction), encoding="utf-8")
        record = json.loads((slot / "mismatch.json").read_text(encoding="utf-8"))
        record["tree_before"] = "0" * 40
        (slot / "mismatch.json").write_text(json.dumps(record), encoding="utf-8")

        result = self.resume([repaired_step_contract()])

        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertNotEqual(result.status, RunStatus.COMMITTED)
        self.assertEqual(len(self.workers.calls), 1)
        self.assertEqual(self.planner.requests, [])

    def git_tree(self) -> str:
        from tests.pipeline_support import git

        return git(self.worktree(), "rev-parse", "HEAD^{tree}")


class WaitingDiagnosticsTests(PipelineHarness):
    def diagnostics(self, run_id: str = "run") -> str:
        return (self.run_dir(run_id) / "diagnostics.md").read_text(encoding="utf-8")

    def assert_fresh(self, status: RunStatus, reason: str | None, run_id: str = "run") -> None:
        report = self.diagnostics(run_id)
        self.assertIn(f'"status": "{status.value}"', report)
        if reason is not None:
            self.assertIn(f'"reason": "{reason}"', report)
        self.assertIn(f'"phase": "{self.checkpoint(run_id)["phase"]}"', report)

    def test_waiting_external_writes_fresh_diagnostics(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch)
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP), LLMError(OUTAGE)], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL)
        self.assert_fresh(RunStatus.WAITING_EXTERNAL, "LLM_FAILURE")
        report = self.diagnostics()
        self.assertIn('"pending_operation": "contract_repair"', report)
        self.assertIn('"step_id": "S01"', report)

    def test_waiting_check_infrastructure_writes_fresh_diagnostics(self) -> None:
        with mock.patch(
            "metaharness.orchestrator.PipelineV2Coordinator.run",
            side_effect=PipelineFailure("CHECK_INFRASTRUCTURE_UNAVAILABLE", "diagnostic"),
        ):
            result = self.orchestrator(
                self.config(), planner=[initial_plan(STEP)], reviewer=["unused"],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_INFRASTRUCTURE)
        self.assert_fresh(RunStatus.WAITING_CHECK_INFRASTRUCTURE, "CHECK_INFRASTRUCTURE_UNAVAILABLE")

    def test_waiting_remote_writes_fresh_diagnostics(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.orchestrator.remote_run_branch_tip",
            side_effect=GitError("simulated network outage"),
        ):
            result = self.orchestrator(
                self.config(publish=True), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_REMOTE)
        self.assert_fresh(RunStatus.WAITING_REMOTE, None)

    def test_waiting_scope_approval_of_a_contract_repair_writes_fresh_diagnostics(self) -> None:
        expanded = repaired_step_contract().replace(
            "WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n- other.txt\n",
        ).replace(
            "- feature.txt :: current content\n",
            "- feature.txt :: current content\n- other.txt :: current content\n",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch)
        config = self.config()
        options = RunOptions.from_config(
            config, repair_scope_policy="require-approval", repair_scope_max_added_paths=1,
        )
        result = self.orchestrator(
            config, planner=[initial_plan(STEP), expanded], reviewer=["unused"],
        ).run_text(SPEC, run_id="run", run_options=options)
        self.assertEqual(
            result.status, RunStatus.WAITING_SCOPE_APPROVAL, self.state().get("failure"),
        )
        self.assert_fresh(RunStatus.WAITING_SCOPE_APPROVAL, None)
        transaction = json.loads((
            self.run_dir() / "cycles/001/implementation/steps/S01/contract_repairs/01/transaction.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(transaction["status"], "scope_waiting")

    def test_web_run_detail_rebuilds_diagnostics_read_only(self) -> None:
        from metaharness.web.api import get_run

        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch)
        config = self.config()
        self.orchestrator(
            config, planner=[initial_plan(STEP), LLMError(OUTAGE)], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        stale = "# stale report\n"
        (self.run_dir() / "diagnostics.md").write_text(stale, encoding="utf-8")
        detail = get_run(self.root / "runs", "run", config=config)
        self.assertTrue(detail["diagnostics"]["live"])
        self.assertIn('"status": "waiting_external"', detail["diagnostics"]["content"])
        self.assertEqual((self.run_dir() / "diagnostics.md").read_text(encoding="utf-8"), stale)


class SemanticRecoveryIdentityTests(PipelineHarness):
    def test_a_stable_identity_collapses_legacy_duplicates_and_resumes(self) -> None:
        from metaharness.orchestration.recovery import RecoveryAttempt, RecoveryCoordinator
        from metaharness.state import RunStateStore

        store = RunStateStore(self.root / "state.json")
        store.initialize("run")
        legacy = {
            "phase": "implementation", "reason": "AGENT_CONTRACT_MISMATCH", "attempt": 1,
            "budget_key": "contract_repairs", "budget": 2, "budget_consumed": 1,
            "disposition": "contract_repair", "cycle": 1, "step_id": "S04",
            "profile_id": "worker", "tree_before": "a" * 40, "tree_after": "a" * 40,
        }
        other = {**legacy, "step_id": "S03"}
        store.update(status=RunStatus.WAITING_EXTERNAL, recovery_attempts=[legacy, legacy, other, legacy])
        coordinator = RecoveryCoordinator(store, emit=lambda *_args, **_kwargs: None)
        attempt = RecoveryAttempt(**legacy, operation_id="contract-repair:cycle-001:S04:01")
        coordinator.record(attempt)
        coordinator.record(attempt)

        records = store.load()["recovery_attempts"]
        self.assertEqual(records, [other, {**legacy, "operation_id": "contract-repair:cycle-001:S04:01"}])
