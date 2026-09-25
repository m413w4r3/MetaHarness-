"""Bounded output corrections of an invalid StepContractRepairPlanner answer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

from metaharness.llm.chat import LLMError
from metaharness.models import ExecutionRole, RunStatus
from metaharness.planning_v2 import (
    StepContractRepairArtifactError,
    StepContractRepairPlanner,
    StepRepairIdentity,
    V2PlanParseError,
    build_step_contract_repair_prompt,
    normalize_repair_step_id,
    parse_step_contract_repair,
)
from metaharness.orchestration import contract_repair
from metaharness.resume import CONTRACT_REPAIR_OPERATION, ResumeNotAllowedError, resume_info
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


REPAIR_TITLE_S05 = "Afficher le corpus REFERENCES et retirer le lien de conversation côté frontend"


def s05_repair_response(step_id: str = "S05 / 06") -> str:
    return (
        repaired_step_contract()
        .replace("STEP_ID: S01", f"STEP_ID: {step_id}", 1)
        .replace("TITLE: Write the feature", f"TITLE: {REPAIR_TITLE_S05}", 1)
        .replace("EXECUTION_CLASS: MECHANICAL", "EXECUTION_CLASS: REASONING", 1)
        .replace("DEPENDS_ON: NONE", "DEPENDS_ON: S04", 1)
    )


class RepairStepIdNormalizationTests(PipelineHarness):
    EXPECTED = {
        "expected_step_id": "S05",
        "expected_title": REPAIR_TITLE_S05,
        "expected_execution_class": "REASONING",
        "expected_depends_on": "S04",
        "expected_plan_step_count": 6,
    }

    def test_bare_and_matching_plan_count_ids_normalize_to_the_expected_step(self) -> None:
        for value in ("S05", "S05 / 06", "S05/06", " S05 / 06 ", "S05 / 6"):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize_repair_step_id(
                        value, expected_step_id="S05", expected_plan_step_count=6,
                    ),
                    "S05",
                )

    def test_plan_count_suffix_without_known_count_still_requires_exact_primary_id(self) -> None:
        self.assertEqual(
            normalize_repair_step_id(
                "S05/06", expected_step_id="S05", expected_plan_step_count=None,
            ),
            "S05",
        )
        for value in ("S04", "S04/06", "S05 or S06", "foo S05", "S5", "S005", "<current step ID>"):
            with self.subTest(value=value), self.assertRaises(V2PlanParseError):
                normalize_repair_step_id(
                    value, expected_step_id="S05", expected_plan_step_count=None,
                )

    def test_wrong_step_or_count_forms_remain_invalid(self) -> None:
        for value in (
            "S04", "S04 / 06", "S05 / 07", "S05 or S06", "<current step ID>",
            "foo S05", "S5", "S005",
        ):
            with self.subTest(value=value), self.assertRaises(V2PlanParseError):
                parse_step_contract_repair(
                    s05_repair_response(value), max_read_paths_per_step=8,
                    **self.EXPECTED,
                )

    def test_exact_run_shape_parses_and_records_only_the_id_normalization(self) -> None:
        normalizations: list[dict[str, str]] = []
        parsed = parse_step_contract_repair(
            s05_repair_response(), max_read_paths_per_step=8,
            _normalizations=normalizations, **self.EXPECTED,
        )

        self.assertEqual(parsed.id, "S05")
        self.assertEqual(normalizations, [{
            "field": "STEP_ID", "rule": "step_id_with_plan_count_suffix",
            "raw": "S05 / 06", "canonical": "S05",
        }])

    def test_safe_wire_variations_parse_without_relaxing_identity(self) -> None:
        wire = "\ufeff" + s05_repair_response().replace("\n", "\r\n")
        wire = wire.replace("\r\n\r\n", "\r\n\r\n\r\n")
        wire = "\r\n".join(line + "   " for line in wire.split("\r\n"))
        parsed = parse_step_contract_repair(
            wire, max_read_paths_per_step=8, **self.EXPECTED,
        )
        self.assertEqual((parsed.id, parsed.title, parsed.execution_class.value, parsed.depends_on), (
            "S05", REPAIR_TITLE_S05, "REASONING", "S04",
        ))

    def test_title_execution_class_and_dependency_are_still_exact(self) -> None:
        cases = (
            s05_repair_response().replace(REPAIR_TITLE_S05, REPAIR_TITLE_S05 + "!", 1),
            s05_repair_response().replace("EXECUTION_CLASS: REASONING", "EXECUTION_CLASS: MECHANICAL", 1),
            s05_repair_response().replace("DEPENDS_ON: S04", "DEPENDS_ON: S03", 1),
        )
        for response in cases:
            with self.subTest(response=response.splitlines()[2]), self.assertRaises(V2PlanParseError):
                parse_step_contract_repair(
                    response, max_read_paths_per_step=8, **self.EXPECTED,
                )


class IdentityPinningPromptTests(PipelineHarness):
    CONTRACT = (
        "META IMPLEMENTATION STEP v1\n\nRUN TITLE\nAW-010\n\nSTEP\nS05 / 06\n\n"
        "TITLE\nAfficher le corpus REFERENCES\n\nEND META IMPLEMENTATION STEP\n"
    )
    IDENTITY = StepRepairIdentity("S05", "Afficher le corpus REFERENCES", "REASONING", "S04", 6)

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

    @staticmethod
    def with_read_paths(contract: str, *paths: str) -> str:
        additions = "".join(f"- {path} :: fixture content\n" for path in paths)
        return contract.replace(
            "- feature.txt :: current content\n",
            "- feature.txt :: current content\n" + additions,
            1,
        )

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

    def test_missing_historical_paths_receive_nearest_tracked_repository_facts(self) -> None:
        invalid_paths = (
            "frontend/src/features/edition-workflow/EditionDashboard.test.tsx",
            "frontend/src/features/edition/EditionDashboard.test.tsx",
        )
        tracked_path = "frontend/src/features/edition-dashboard/EditionDashboard.test.tsx"
        self.add_tracked(tracked_path)
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(
            self.with_read_paths(repaired_step_contract(), *invalid_paths),
            repaired_step_contract(),
        )

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        correction = self.planner.requests[2]
        facts = correction.split("<REPOSITORY PATH FACTS>", 1)[1].split(
            "</REPOSITORY PATH FACTS>", 1,
        )[0]
        for expected in (*invalid_paths, tracked_path):
            self.assertIn(expected, facts)
        self.assertIn("READ/WRITE paths must come from tracked repository paths.", facts)
        self.assertIn("Only paths explicitly authorized by CREATE_SET may be new.", facts)
        self.assertIn("facts below are authoritative", correction)

    def test_a_wrong_real_step_id_is_never_rewritten(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        wrong = malformed("S04")

        result = self.run_repair(wrong, repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertIn("STEP_ID changed", self.parse_error(1)["detail"])
        self.assertEqual((self.slot() / "planner.raw.md").read_text(encoding="utf-8"), wrong)
        self.assertEqual(len(self.planner.requests), 3)
        self.assertEqual(self.repair_slots(), ["01"])

    def test_a_wrong_plan_count_suffix_uses_output_format_recovery(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(malformed("S01 / 02"), repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.parse_error(1)["code"], "STEP_CONTRACT_REPAIR_OUTPUT_INVALID")
        self.assertEqual(len(self.planner.requests), 3)  # plan, rejected answer, correction
        self.assertEqual(len(self.semantic_records()), 1)
        self.assertEqual(self.transaction()["output_correction_attempt"], 1)
        self.assert_no_false_mismatch()

    def test_removed_mutable_path_is_rejected_without_local_repair(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        removed = repaired_step_contract().replace(
            "WRITE_SET\n- feature.txt\n", "WRITE_SET\n- other.txt\n", 1,
        ).replace(
            "- feature.txt :: current content\n",
            "- feature.txt :: current content\n- other.txt :: current content\n",
            1,
        )

        result = self.run_repair(removed, repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertIn("removed approved mutable paths", self.parse_error(1)["detail"])
        self.assertEqual(len(self.planner.requests), 3)
        self.assertEqual(len(self.semantic_records()), 1)

    def test_existing_create_path_is_not_moved_to_write_set(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        create_existing = repaired_step_contract().replace(
            "WRITE_SET\n- feature.txt\n", "WRITE_SET\nNONE\n", 1,
        ).replace("READ_SET\n- feature.txt :: current content", "READ_SET\nNONE", 1).replace(
            "CREATE_SET\nNONE", "CREATE_SET\n- feature.txt", 1,
        )

        result = self.run_repair(create_existing, repaired_step_contract())

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.parse_error(1)["code"], "STEP_CONTRACT_REPAIR_OUTPUT_INVALID")
        self.assertEqual(len(self.planner.requests), 3)
        (record,) = self.semantic_records()
        self.assertEqual((record["attempt"], record["budget_consumed"]), (1, 1))

    def test_corrupt_repaired_contract_hash_fails_closed_before_another_call(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))
        result = self.run_repair(repaired_step_contract())
        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        slot = self.slot()
        contract_path = slot / "contract.md"
        contract_path.write_text(contract_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

        class NoCallClient:
            def complete(self, _request):
                raise AssertionError("hash-corrupt repair must fail before a planner call")

        transaction = self.transaction()
        planner = StepContractRepairPlanner(NoCallClient(), max_read_paths_per_step=8)
        with self.assertRaisesRegex(StepContractRepairArtifactError, "hash changed"):
            planner.resume(
                artifacts_dir=slot,
                original_plan_identity="original-plan",
                current_contract="approved-contract",
                mismatch_explanation="worker mismatch",
                current_tree_sha=transaction["tree_sha"],
                identity=StepRepairIdentity("S01", "Write the feature", "MECHANICAL", "NONE", 1),
                read_set="- feature.txt :: current content",
                write_set="- feature.txt", create_set="NONE", delete_set="NONE",
            )

    def test_two_invalid_answers_then_a_valid_one_stay_in_one_slot(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, mismatch, write("feature.txt", "good\n"))

        result = self.run_repair(malformed(), malformed("S01 / 02"), repaired_step_contract())

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
        # Plan, the answer, two corrections, then one bounded planner restart
        # of two more corrections -- all inside the same semantic slot.
        self.assertEqual(len(self.planner.requests), 6)
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual(transaction["status"], "output_correction_exhausted")
        self.assertEqual(transaction["output_correction_attempt"], 4)
        self.assertEqual(transaction["planner_restarts"], 1)
        self.assertIn("contract_repair.planner_restart", self.trace_names())
        for number in (1, 2, 3, 4, 5):
            self.assertEqual(self.parse_error(number)["detail"], PARSE_ERROR)
        self.assertFalse((self.slot() / "validation.json").exists())
        self.assertEqual(self.semantic_records(), [])
        self.assertEqual(self.state()["contract_repair"]["status"], "output_correction_exhausted")
        info = resume_info(self.run_dir(), self.state())
        self.assertTrue(info.resumable)
        self.assertEqual(info.label, "Retry contract repair planner (S01)")
        from metaharness.web.api import live_status

        live = live_status(self.root / "runs", "run", config=self.config())
        self.assertEqual(live["current_label"], "Output correction exhausted · S01 · attempt 4 / 4")

        resumed = self.resume([repaired_step_contract()], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(len(self.workers.calls), 2)
        transaction = self.transaction()
        self.assertEqual((transaction["output_attempt"], transaction["operator_output_retries"]), (6, 1))
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


class PlanCountRepairTests(ContractRepairFixtures):
    STEP_PATHS = tuple(f"steps/s{number:02d}.txt" for number in (1, 2, 3, 4, 6))

    def step_dir(self, run_id: str = "run") -> Path:
        return self.run_dir(run_id) / "cycles/001/implementation/steps/S05"

    def plan_with_s05_identity(self) -> str:
        steps = (
            ("S01", self.STEP_PATHS[0], "Prepare source one"),
            ("S02", self.STEP_PATHS[1], "Prepare source two"),
            ("S03", self.STEP_PATHS[2], "Prepare source three"),
            ("S04", self.STEP_PATHS[3], "Prepare source four"),
            ("S05", "feature.txt", REPAIR_TITLE_S05),
            ("S06", self.STEP_PATHS[4], "Finish the implementation"),
        )
        raw = initial_plan(*steps)
        before = (
            f"BEGIN STEP S05\nTITLE: {REPAIR_TITLE_S05}\n"
            "EXECUTION_CLASS: MECHANICAL\nDEPENDS_ON: NONE"
        )
        after = (
            f"BEGIN STEP S05\nTITLE: {REPAIR_TITLE_S05}\n"
            "EXECUTION_CLASS: REASONING\nDEPENDS_ON: S04"
        )
        if before not in raw:
            raise AssertionError("S05 fixture block was not rendered")
        return raw.replace(before, after, 1)

    def strand_s05(self, repair_raw: str | None = None) -> str:
        for path in self.STEP_PATHS:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "six-step fixtures")
        git(self.repo, "push", "-q", "origin", "main")

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            *(write(path, f"{path}\n") for path in self.STEP_PATHS[:4]),
            mismatch,
            write("feature.txt", "good\n"),
            write(self.STEP_PATHS[4], "finished\n"),
        )
        with mock.patch(
            "metaharness.planning_v2.parse_step_contract_repair",
            side_effect=KeyboardInterrupt(),
        ):
            interrupted = self.orchestrator(
                self.config(),
                planner=[self.plan_with_s05_identity(), repair_raw or s05_repair_response()],
                reviewer=[review()],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(interrupted.status, RunStatus.INTERRUPTED)

        detail = f"step=S05 contract repair failed: {PARSE_ERROR}"
        RunStateStore(self.run_dir() / "state.json").update(
            status=RunStatus.WAITING_HUMAN, recovery_resumable=False, current_step=None,
            failure={"reason": "AGENT_CONTRACT_MISMATCH", "detail": detail},
        )
        (self.step_dir() / "step.json").write_text(json.dumps({
            "id": "S05", "status": "FAILED", "reason": "AGENT_CONTRACT_MISMATCH",
            "tree_before": self.git_tree(), "tree_after": self.git_tree(), "changed_paths": [],
        }), encoding="utf-8")
        self.assertEqual(
            (self.checkpoint()["phase"], self.checkpoint()["step_id"]),
            ("implement_step", "S05"),
        )
        self.assertEqual(self.transaction()["status"], "planner_response_durable")
        self.assertEqual(
            (self.step_dir() / "contract_repairs/01/planner.raw.md").read_text(encoding="utf-8"),
            repair_raw or s05_repair_response(),
        )
        return detail

    def test_benign_s05_suffix_retries_worker_without_output_correction(self) -> None:
        for path in self.STEP_PATHS:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "six-step fixtures")
        git(self.repo, "push", "-q", "origin", "main")

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            *(write(path, f"{path}\n") for path in self.STEP_PATHS[:4]),
            mismatch,
            write("feature.txt", "good\n"),
            write(self.STEP_PATHS[4], "finished\n"),
        )
        result = self.orchestrator(
            self.config(),
            planner=[self.plan_with_s05_identity(), s05_repair_response()],
            reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(len(self.planner.requests), 2)  # initial plan + one repair answer
        self.assertEqual(len(self.workers.calls), 7)  # S05 was run once after its repair
        self.assertIn("STEP_ID: S05\n", self.workers.calls[-2].contract or "")
        self.assertIn(REPAIR_TITLE_S05, self.workers.calls[-2].contract or "")
        self.assertEqual(self.repair_slots(), ["01"])
        transaction = self.transaction()
        self.assertEqual(transaction["status"], "completed")
        self.assertEqual((transaction["output_attempt"], transaction["output_correction_attempt"]), (1, 0))
        (record,) = self.semantic_records()
        self.assertEqual((record["attempt"], record["budget_consumed"]), (1, 1))
        validation = json.loads((self.step_dir() / "contract_repairs/01/validation.json").read_text(encoding="utf-8"))
        self.assertEqual(validation["added_mutable_paths"], [])
        self.assertEqual(validation["parser_normalization"], {
            "schema_version": 1,
            "normalizations": [{
                "field": "STEP_ID", "rule": "step_id_with_plan_count_suffix",
                "raw": "S05 / 06", "canonical": "S05",
            }],
        })
        self.assertEqual(self.workers.calls[-2].mutable_paths, ("feature.txt",))

    def test_legacy_s05_failure_uses_strict_detection_then_reuses_paid_raw(self) -> None:
        detail = self.strand_s05()
        config = self.config()
        slot = self.step_dir() / "contract_repairs/01"
        raw_path = slot / "planner.raw.md"
        raw_sha = sha256(raw_path)

        self.assertEqual(
            contract_repair.stranded_output_failure(
                self.step_dir(), step_id="S05", tree_sha=self.git_tree(),
                failure_detail=detail,
                max_read_paths_per_step=config.planning.max_read_paths_per_step,
            ),
            contract_repair.STRANDED_PROVEN,
        )
        info = resume_info(self.run_dir(), self.state())
        self.assertTrue(info.resumable, info.reason)
        self.assertEqual(info.operation, CONTRACT_REPAIR_OPERATION)
        self.assertIn("Retry contract repair planner (S05)", info.label)

        paid_calls = len(self.planner.requests)
        self.assertEqual(paid_calls, 2)  # initial plan and the already-paid repair answer
        self.assertEqual(self.repair_slots(), ["01"])
        self.assertEqual(self.semantic_records(), [])
        resumed = self.resume(["planner must not be called"], [review()])

        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.planner.requests, [])  # no planner call on resume
        self.assertEqual(len(self.workers.calls), 7)
        self.assertEqual(
            [request.artifact_dir.name for request in self.workers.calls],
            ["S01", "S02", "S03", "S04", "S05", "S05", "S06"],
        )  # S01-S04 were not replayed; S05 used its repaired contract
        self.assertEqual(self.repair_slots(), ["01"])
        self.assertEqual(len(self.semantic_records()), 1)
        self.assertEqual(self.transaction()["status"], "completed")
        self.assertEqual(
            (self.transaction()["output_attempt"], self.transaction()["output_correction_attempt"]),
            (1, 0),
        )
        self.assertEqual(sha256(raw_path), raw_sha)
        canonical = (slot / "contract.md").read_text(encoding="utf-8")
        self.assertIn("STEP_ID: S05\n", canonical)
        self.assertNotIn("S05 / 06", canonical)
        validation = json.loads((slot / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(validation["parser_normalization"]["normalizations"], [{
            "field": "STEP_ID", "rule": "step_id_with_plan_count_suffix",
            "raw": "S05 / 06", "canonical": "S05",
        }])
        self.assertIn("STEP_ID: S05\n", self.workers.calls[-2].contract or "")
        self.assertNotIn("S05 / 06", self.workers.calls[-2].contract or "")

    def test_stranded_failure_does_not_accept_a_wrong_real_step_id(self) -> None:
        detail = self.strand_s05(s05_repair_response("S04"))
        info = resume_info(self.run_dir(), self.state())

        self.assertFalse(info.resumable)
        self.assertIsNone(info.operation)
        self.assertEqual(
            contract_repair.stranded_output_failure(
                self.step_dir(), step_id="S05", tree_sha=self.git_tree(),
                failure_detail=detail, max_read_paths_per_step=8,
            ),
            contract_repair.STRANDED_ABSENT,
        )


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
                self.config(), planner=[initial_plan(*self.STEPS), malformed()],
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
