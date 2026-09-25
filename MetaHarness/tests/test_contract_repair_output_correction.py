"""Bounded output corrections of an invalid StepContractRepairPlanner answer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

from metaharness.llm.chat import LLMError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.planning_v2 import (
    StepContractRepairPlanner,
    StepRepairIdentity,
    build_step_contract_repair_prompt,
)
from metaharness.resume import ResumeNotAllowedError, resume_info
from metaharness.run_options import RunOptions
from metaharness.state import RunStateStore
from tests.pipeline_support import PipelineHarness, git, initial_plan, review, write
from tests.test_contract_repair_transaction import OUTAGE, REPAIR_ID, ContractRepairFixtures, mismatch
from tests.test_pipeline_v2_machine import SPEC, STEP, repaired_step_contract

FIXTURES = (
    "frontend/src/components/ProductionStateTransfer.test.tsx",
    "frontend/src/components/EditionDashboard.test.tsx",
)
PLACEHOLDER_ID = "<current step ID>"
PARSE_ERROR = "step contract repair STEP_ID is invalid"


def malformed(step_id: str = PLACEHOLDER_ID) -> str:
    return repaired_step_contract().replace("STEP_ID: S01", f"STEP_ID: {step_id}", 1)


def with_paths(contract: str, *paths: str) -> str:
    """Existing files are read and written: READ_SET + WRITE_SET, never CREATE_SET."""

    reads = "".join(f"- {path} :: fixture content\n" for path in paths)
    writes = "".join(f"- {path}\n" for path in paths)
    return contract.replace(
        "- feature.txt :: current content\n", "- feature.txt :: current content\n" + reads,
    ).replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n" + writes)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IdentityPinningPromptTests(PipelineHarness):
    CONTRACT = (
        "META IMPLEMENTATION STEP v1\n\nRUN TITLE\nAW-010\n\nSTEP\nS05 / 06\n\n"
        "TITLE\nAfficher le corpus REFERENCES\n\nEND META IMPLEMENTATION STEP\n"
    )
    IDENTITY = StepRepairIdentity("S05", "Afficher le corpus REFERENCES", "REASONING", "S04")

    def test_the_repair_prompt_pins_the_bare_step_identity(self) -> None:
        prompt = build_step_contract_repair_prompt(
            original_spec="SPEC", current_tree_sha="a" * 40, original_plan_identity="plan",
            current_contract=self.CONTRACT, mismatch_explanation="needs fixtures",
            read_set="- a :: b", write_set="- a", create_set="NONE", delete_set="NONE",
            identity=self.IDENTITY,
        )

        # The display form stays in the approved contract, verbatim...
        self.assertIn(f"<CURRENT STEP CONTRACT>\n{self.CONTRACT}\n</CURRENT STEP CONTRACT>", prompt)
        self.assertIn("STEP\nS05 / 06\n", prompt)
        # ...but the identity is handed over explicitly, never inferred from it.
        self.assertIn("IMMUTABLE STEP ID: S05\n", prompt)
        self.assertIn("For this repair it is exactly `S05`.", prompt)
        self.assertIn("Do NOT append the plan step count", prompt)
        template = prompt.split("META STEP CONTRACT REPAIR v1\n")[-1]
        self.assertTrue(template.startswith(
            "STEP_ID: S05\nTITLE: Afficher le corpus REFERENCES\n"
            "EXECUTION_CLASS: REASONING\nDEPENDS_ON: S04\n"
        ))
        self.assertNotIn(PLACEHOLDER_ID, prompt)
        self.assertNotIn("<current title>", prompt)

    def test_the_prompt_sent_by_the_pipeline_carries_the_exact_identity(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(), planner=[initial_plan(STEP), repaired_step_contract()], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        # A repair valid on the first answer needs no extra planner call.
        (request,) = self.planner.requests[1:]
        self.assertIn(
            "STEP_ID: S01\nTITLE: Write the feature\nEXECUTION_CLASS: MECHANICAL\nDEPENDS_ON: NONE\n",
            request,
        )
        self.assertNotIn(PLACEHOLDER_ID, request)
        self.assertIn("STEP\nS01 / 01", request)


class OutputCorrectionTests(ContractRepairFixtures):
    def slot(self) -> Path:
        return self.step_dir() / "contract_repairs/01"

    def output_attempt(self, number: int) -> Path:
        return self.slot() / f"output_attempts/{number:03d}"

    def parse_error(self, number: int) -> dict:
        return json.loads((self.output_attempt(number) / "parse_error.json").read_text(encoding="utf-8"))

    def run_repair(self, *answers, options: RunOptions | None = None, config=None):
        return self.orchestrator(
            config or self.config(), planner=[initial_plan(STEP), *answers], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

    def assert_no_false_mismatch(self) -> None:
        state = self.state()
        self.assertNotEqual((state.get("failure") or {}).get("reason"), "AGENT_CONTRACT_MISMATCH")
        record = json.loads((self.step_dir() / "step.json").read_text(encoding="utf-8")) \
            if (self.step_dir() / "step.json").exists() else {}
        self.assertNotEqual(record.get("reason"), "AGENT_CONTRACT_MISMATCH")

    def test_placeholder_step_id_is_corrected_in_the_same_slot(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(malformed(), repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The worker ran once before the repair and once after validation.
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual(transaction["repair_id"], REPAIR_ID)
        self.assertEqual(transaction["status"], "completed")
        self.assertEqual(transaction["output_attempt"], 2)
        self.assertEqual(transaction["output_correction_attempt"], 1)
        (record,) = self.semantic_records()
        self.assertEqual((record["attempt"], record["budget_consumed"]), (1, 1))
        error = self.parse_error(1)
        self.assertEqual(error["code"], "STEP_CONTRACT_REPAIR_OUTPUT_INVALID")
        self.assertEqual(error["detail"], PARSE_ERROR)
        # The first paid answer is immutable evidence; the correction is new.
        self.assertEqual(error["raw_sha256"], sha256(self.slot() / "planner.raw.md"))
        self.assertIn(PLACEHOLDER_ID, (self.slot() / "planner.raw.md").read_text(encoding="utf-8"))
        correction = self.planner.requests[2]
        self.assertEqual(correction, (self.output_attempt(2) / "planner.request.txt").read_text(encoding="utf-8"))
        self.assertIn(f"<PARSE ERROR>\n{PARSE_ERROR}\n</PARSE ERROR>", correction)
        self.assertIn("IMMUTABLE STEP ID:\nS01\n", correction)
        self.assertIn(f"STEP_ID: {PLACEHOLDER_ID}", correction.split("<REJECTED RESPONSE>")[1])
        validation = json.loads((self.slot() / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual((validation["status"], validation["output_attempt"]), ("validated", 2))
        self.assertEqual(validation["raw_sha256"], sha256(self.output_attempt(2) / "planner.raw.md"))
        self.assertIn("contract_repair.output_invalid", self.trace_names())

    def test_a_wrong_real_step_id_is_never_rewritten(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        wrong = malformed("S04")

        result = self.run_repair(wrong, repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertIn("STEP_ID changed", self.parse_error(1)["detail"])
        self.assertEqual((self.slot() / "planner.raw.md").read_text(encoding="utf-8"), wrong)
        self.assertEqual(len(self.planner.requests), 3)
        self.assertEqual(self.repair_slots(), ["01"])

    def test_two_invalid_answers_then_a_valid_one_stay_in_one_slot(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(malformed(), malformed("S01 / 01"), repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.repair_slots(), ["01"])
        self.assertEqual(self.transaction()["output_correction_attempt"], 2)
        self.assertEqual(len(self.semantic_records()), 1)
        self.assertEqual(self.parse_error(2)["detail"], PARSE_ERROR)

    def test_an_exhausted_output_budget_waits_for_a_planner_retry(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(malformed())

        self.assertEqual(result.status, RunStatus.WAITING_CONTRACT_REPAIR)
        self.assertEqual(self.state()["failure"]["reason"], "STEP_CONTRACT_REPAIR_OUTPUT_INVALID")
        self.assert_no_false_mismatch()
        self.assertEqual(len(self.workers.calls), 1)
        self.assertEqual(len(self.planner.requests), 4)
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual(transaction["status"], "output_correction_exhausted")
        self.assertEqual(transaction["output_correction_attempt"], 2)
        for number in (1, 2, 3):
            self.assertEqual(self.parse_error(number)["detail"], PARSE_ERROR)
        self.assertFalse((self.slot() / "validation.json").exists())
        self.assertEqual(self.semantic_records(), [])
        self.assertEqual(self.state()["contract_repair"]["status"], "output_correction_exhausted")
        info = resume_info(self.run_dir(), self.state())
        self.assertTrue(info.resumable)
        self.assertEqual(info.label, "Retry contract repair planner (S01)")
        from metaharness.web.api import live_status

        live = live_status(self.root / "runs", "run", config=self.config())
        self.assertEqual(live["current_label"], "Output correction exhausted · S01 · attempt 2 / 2")

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(len(self.workers.calls), 2)
        transaction = self.transaction()
        self.assertEqual((transaction["output_attempt"], transaction["operator_output_retries"]), (4, 1))
        self.assertEqual(self.repair_slots(), ["01"])

    def test_a_raw_answer_durable_before_its_parse_is_parsed_without_a_new_call(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        with mock.patch(
            "metaharness.planning_v2.parse_step_contract_repair", side_effect=KeyboardInterrupt(),
        ):
            interrupted = self.run_repair(malformed())
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
        self.assertFalse((self.output_attempt(1) / "parse_error.json").exists())

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The durable answer was classified first; the only call is its correction.
        (correction,) = self.planner.requests
        self.assertIn("<REJECTED RESPONSE>", correction)
        self.assertEqual(self.parse_error(1)["detail"], PARSE_ERROR)
        self.assertEqual(len(self.workers.calls), 2)

    def test_a_durable_rejection_resumes_at_the_next_correction(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        with mock.patch.object(
            StepContractRepairPlanner, "_correction_request", side_effect=KeyboardInterrupt(),
        ):
            interrupted = self.run_repair(malformed())
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
        self.assertEqual(self.transaction()["status"], "planner_output_invalid")
        from metaharness import planning_v2

        with mock.patch(
            "metaharness.planning_v2.parse_step_contract_repair",
            wraps=planning_v2.parse_step_contract_repair,
        ) as parse:
            resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # Only the corrected answer is parsed; the rejection is never re-parsed.
        self.assertEqual(parse.call_count, 1)
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.repair_slots(), ["01"])

    def test_a_transport_failure_during_correction_waits_external_in_the_same_slot(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        waiting = self.run_repair(malformed(), LLMError(OUTAGE))

        self.assertEqual(waiting.status, RunStatus.WAITING_EXTERNAL, self.state().get("failure"))
        transaction = self.transaction()
        self.assertEqual(transaction["status"], "waiting_external")
        self.assertEqual((transaction["output_attempt"], transaction["repair_id"]), (2, REPAIR_ID))
        pending = (self.output_attempt(2) / "planner.request.txt").read_text(encoding="utf-8")

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # The pending correction is re-sent byte-for-byte; no worker replay.
        self.assertEqual(self.planner.requests, [pending])
        self.assertEqual(len(self.workers.calls), 2)
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual((transaction["output_correction_attempt"], transaction["planner_transport_attempt"]), (1, 2))

    def add_tracked(self, *paths: str) -> None:
        for path in paths:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "fixtures")
        git(self.repo, "push", "-q", "origin", "main")

    def test_auto_bounded_scope_admits_the_two_frontend_fixtures(self) -> None:
        self.add_tracked(*FIXTURES)
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        config = self.config()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
        )

        result = self.run_repair(
            malformed(), with_paths(repaired_step_contract(), *FIXTURES),
            options=options, config=config,
        )

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        validation = json.loads((self.slot() / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(validation["added_mutable_paths"], sorted(FIXTURES))
        self.assertEqual(validation["create_set"], [])
        self.assertTrue(set(FIXTURES) <= set(self.workers.calls[-1].mutable_paths))

    def test_scope_over_the_auto_bound_is_not_authorized(self) -> None:
        extra = tuple(f"fixtures/f{index}.txt" for index in range(5))
        self.add_tracked(*extra)
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        config = self.config()
        options = RunOptions.from_config(
            config, repair_scope_policy="auto-bounded", repair_scope_max_added_paths=4,
        )

        result = self.run_repair(
            with_paths(repaired_step_contract(), *extra), options=options, config=config,
        )

        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)
        self.assertEqual(self.state()["failure"]["reason"], "CONTRACT_REPAIR_SCOPE_DENIED")
        self.assert_no_false_mismatch()
        self.assertEqual(len(self.workers.calls), 1)
        self.assertEqual(len(self.planner.requests), 2)
        self.assertFalse(resume_info(self.run_dir(), self.state()).resumable)


class StrandedRunRecoveryTests(ContractRepairFixtures):
    """The historical shape: an invalid answer projected as a worker mismatch."""

    STEPS = (("S01", "other.txt", "Prepare the other file"), ("S02", "feature.txt", "Write the feature"))

    def step_dir(self, run_id: str = "run") -> Path:
        return self.run_dir(run_id) / "cycles/001/implementation/steps/S02"

    def strand(self) -> str:
        """Reproduce the legacy stranded state without a legacy code path."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER, write("other.txt", "one\n"), mismatch,
            write("feature.txt", "good\n"),
        )
        with mock.patch(
            "metaharness.planning_v2.parse_step_contract_repair", side_effect=KeyboardInterrupt(),
        ):
            interrupted = self.orchestrator(
                self.config(), planner=[initial_plan(*self.STEPS), malformed("S02 / 02")],
                reviewer=["unused"],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)
        detail = f"step=S02 contract repair failed: {PARSE_ERROR}"
        RunStateStore(self.run_dir() / "state.json").update(
            status=RunStatus.WAITING_HUMAN, recovery_resumable=False, current_step=None,
            failure={"reason": "AGENT_CONTRACT_MISMATCH", "detail": detail},
        )
        (self.step_dir() / "step.json").write_text(json.dumps({
            "id": "S02", "status": "FAILED", "reason": "AGENT_CONTRACT_MISMATCH",
            "tree_before": self.git_tree(), "tree_after": self.git_tree(), "changed_paths": [],
        }), encoding="utf-8")
        self.assertEqual(self.checkpoint()["step_id"], "S02")
        self.assertEqual(self.transaction()["status"], "planner_response_durable")
        return git(self.worktree(), "rev-parse", "HEAD")

    def test_a_proven_stranded_run_is_recovered_without_any_replay(self) -> None:
        s01_commit = self.strand()
        raw_sha = sha256(self.step_dir() / "contract_repairs/01/planner.raw.md")
        info = resume_info(self.run_dir(), self.state())
        self.assertTrue(info.resumable, info.reason)
        self.assertEqual(info.label, "Retry contract repair planner (S02)")
        self.assertEqual(info.operation, "contract_repair")
        valid = repaired_step_contract().replace("STEP_ID: S01", "STEP_ID: S02")

        resumed = self.resume([valid], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        # S01 and the S02 mismatch worker are never replayed.
        self.assertEqual(len(self.workers.calls), 3)
        (correction,) = self.planner.requests
        self.assertIn("IMMUTABLE STEP ID:\nS02\n", correction)
        self.assertEqual(self.repair_slots(), ["01"])
        slot = self.step_dir() / "contract_repairs/01"
        self.assertEqual(sha256(slot / "planner.raw.md"), raw_sha)
        error = json.loads((slot / "output_attempts/001/parse_error.json").read_text(encoding="utf-8"))
        self.assertEqual((error["detail"], error["raw_sha256"]), (PARSE_ERROR, raw_sha))
        transaction = self.transaction()
        self.assertEqual(transaction["repair_id"], "contract-repair:cycle-001:S02:01")
        self.assertEqual(transaction["output_correction_attempt"], 1)
        self.assertIn(s01_commit, git(self.worktree(), "rev-list", "HEAD"))

    def test_the_run_page_and_diagnostics_show_the_pending_repair(self) -> None:
        from metaharness.diagnostics import build_run_diagnostics
        from metaharness.web.api import get_run, live_status
        from metaharness.web.pages import render_run

        self.strand()
        config = self.config()

        page = render_run(get_run(self.root / "runs", "run", config=config), "token", config=config)
        live = live_status(self.root / "runs", "run", config=config)
        report = build_run_diagnostics(config, self.run_dir())

        self.assertNotIn("RESUME REFUSED", page)
        self.assertNotIn("Waiting for operator decision", page)
        self.assertIn("Retry contract repair planner (S02)", page)
        self.assertIn("contract-repair:cycle-001:S02:01", page)
        self.assertEqual(live["contract_repair"]["phase"], "Correcting planner output")
        self.assertEqual(live["current_label"], "Correcting planner output · S02")
        section = report.split("## CONTRACT REPAIR", 1)[1].split("\n## ", 1)[0]
        self.assertIn('"repair_id": "contract-repair:cycle-001:S02:01"', section)
        self.assertIn('"parse_status": "raw"', section)
        self.assertIn(sha256(self.step_dir() / "contract_repairs/01/planner.raw.md"), section)
        self.assertNotIn("META STEP CONTRACT REPAIR v1", section)

    def test_a_corrupted_historical_answer_fails_closed_without_a_model_call(self) -> None:
        self.strand()
        (self.step_dir() / "contract_repairs/01/planner.raw.md").write_text("tampered\n", encoding="utf-8")

        result = self.resume(["planner must not be called"])

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "RESUME_INTEGRITY_FAILURE")
        self.assertEqual(self.planner.requests, [])
        self.assertEqual(len(self.workers.calls), 2)

    def test_a_genuine_waiting_human_stays_non_resumable(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, mismatch, mismatch)
        result = self.orchestrator(
            self.config(max_step_contract_repairs=2),
            planner=[initial_plan(STEP), repaired_step_contract()], reviewer=["unused"],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_HUMAN)

        info = resume_info(self.run_dir(), self.state())

        self.assertFalse(info.resumable)
        self.assertIsNone(info.operation)
        with self.assertRaises(ResumeNotAllowedError):
            self.resume(["planner must not be called"])
        self.assertEqual(self.planner.requests, [])

