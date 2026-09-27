"""META CONTINUE v1: the authority that closes one audited milestone.

The parser is pure, the service runs on the planner profile through a text
transport, and the artifacts of one decision round-trip.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.chat import LLMHTTPError  # noqa: E402
from metaharness.models import CheckConfig, PlanningConfig  # noqa: E402
from metaharness.planning.continue_request import (  # noqa: E402
    PlannerContinueFacts,
    build_planner_continue_payload,
    planner_continue_dir,
    read_planner_continue_artifacts,
)
from metaharness.planning.planner_continue import (  # noqa: E402
    ContinueDecision,
    PlannerContinue,
    parse_planner_continue,
    stagnation_fingerprint,
)
from metaharness.planning.protocol import (  # noqa: E402
    V2PlanParseError,
    parse_task_plan_v2,
)
from tests.pipeline_support import ScriptedChat, plan  # noqa: E402

CATALOG = (CheckConfig("test", ("python", "-c", "pass")),)
PLANNING = PlanningConfig()
STEP = ("S01", "feature.txt", "Write the feature")
PLAN_RAW = plan(STEP)
SECOND_MILESTONE = PLAN_RAW.replace("MILESTONE_ID: M01", "MILESTONE_ID: M02")
BLOCKED_PLAN = """META PLAN v2

STATUS: BLOCKED
TITLE: Waiting for a product decision
BLOCKER_KIND: SPEC_DECISION

OBJECTIVE
Define the missing behavior.

BLOCKERS
The SPEC does not choose a retry budget.

END META PLAN
"""


def answer(
    decision: str, *, summary: str = "The milestone is closed.",
    remaining: str = "- none", milestone: str = "NONE", question: str = "NONE",
    plan_text: str | None = None,
) -> str:
    """One META CONTINUE v1 answer in the exact wire shape of the prompt."""

    body = ["META CONTINUE v1", "", "DECISION", decision, "", "SUMMARY", summary,
            "", "REMAINING", remaining, "", "NEXT_MILESTONE", milestone,
            "", "SPEC_QUESTION", question]
    if plan_text is not None:
        body += ["", "BEGIN NEXT PLAN", plan_text.rstrip("\n"), "END NEXT PLAN"]
    return "\n".join(body + ["", "END META CONTINUE", ""])


def facts(**overrides: object) -> PlannerContinueFacts:
    values: dict[str, object] = {
        "spec": "SPEC: add the feature.",
        "plan": parse_task_plan_v2(PLAN_RAW, planning=PLANNING, check_catalog=CATALOG),
        "iteration": 1, "milestone_id": "M01",
        "milestone_title": "First milestone", "milestone_goal": "The feature exists.",
        "audit_status": "DONE", "milestones": (("M01", "DONE"),),
        "normalizations": ("step S01 READ_SET gained feature.txt",),
        "failed_steps": (), "audit_remaining": ("Polish the reporting path.",),
        "audit_risks": ("The flag stays undocumented.",), "audit_fixed": ("a defect",),
        "audit_refactored": (), "gate_failures": ("test:failure",), "gate_warnings": (),
        "gate_baseline_warnings": (), "diffstat": " feature.txt | 2 +-",
        "modified_paths": ("feature.txt",),
    }
    values.update(overrides)
    return PlannerContinueFacts(**values)  # type: ignore[arg-type]


def parse(raw: str, planning: PlanningConfig | None = None):
    return parse_planner_continue(raw, planning=planning, check_catalog=CATALOG)


class ContinueProtocolTests(unittest.TestCase):
    def test_complete_carries_no_milestone_plan_or_question(self) -> None:
        result = parse(answer("COMPLETE"))
        self.assertIs(result.decision, ContinueDecision.COMPLETE)
        self.assertEqual(result.summary, "The milestone is closed.")
        self.assertEqual(result.remaining, ())
        self.assertIsNone(result.next_milestone)
        self.assertIsNone(result.spec_question)
        self.assertIsNone(result.next_plan)

    def test_remaining_is_what_the_audit_listed(self) -> None:
        raw = answer("COMPLETE", remaining="- Finish the adapter.\n- NONE\n- Document the flag.")
        self.assertEqual(parse(raw).remaining, ("Finish the adapter.", "Document the flag."))

    def test_next_carries_a_plan_parsed_by_the_c6_parser(self) -> None:
        result = parse(answer("NEXT", milestone="M02", plan_text=SECOND_MILESTONE))
        self.assertIs(result.decision, ContinueDecision.NEXT)
        self.assertEqual(result.next_milestone, "M02")
        self.assertIsNotNone(result.next_plan)
        self.assertEqual(result.next_plan.milestone_id, "M02")
        self.assertEqual(result.next_plan.steps[0].id, "S01")
        self.assertEqual(result.next_plan.steps[0].title, "Write the feature")
        self.assertIn("META PLAN v2", result.next_plan.raw)
        self.assertIsNone(result.spec_question)

    def test_spec_decision_requires_one_concrete_question(self) -> None:
        question = "Which retry budget should the client use?"
        result = parse(answer("SPEC_DECISION", question=question))
        self.assertIs(result.decision, ContinueDecision.SPEC_DECISION)
        self.assertEqual(result.spec_question, question)
        self.assertIsNone(result.next_plan)
        for invalid in (answer("SPEC_DECISION"), answer("SPEC_DECISION", question="N/A")):
            with self.subTest(question=invalid.splitlines()[-3]), self.assertRaisesRegex(
                    V2PlanParseError, "SPEC_QUESTION"):
                parse(invalid)
        with self.assertRaises(V2PlanParseError):
            parse(answer("SPEC_DECISION", question=""))

    def test_a_spec_question_is_content_never_a_question_mark(self) -> None:
        for question in (
            "Choose the canonical persistence format",
            "Should deletion be soft or permanent",
        ):
            with self.subTest(question=question):
                result = parse(answer("SPEC_DECISION", question=question))
                self.assertIs(result.decision, ContinueDecision.SPEC_DECISION)
                self.assertEqual(result.spec_question, question)
        # A question is one bounded sentence, never a paragraph.
        with self.assertRaisesRegex(V2PlanParseError, "SPEC_QUESTION"):
            parse(answer("SPEC_DECISION", question="x" * 501))
        payload = build_planner_continue_payload(
            facts(), planning=PLANNING, check_catalog=CATALOG, default_check_ids=("test",))
        self.assertIn('ending with "?" is recommended, never required', payload.rendered)

    def test_next_milestone_must_match_the_embedded_plan(self) -> None:
        with self.assertRaisesRegex(V2PlanParseError, "differ"):
            parse(answer("NEXT", milestone="M03", plan_text=SECOND_MILESTONE))
        with self.assertRaisesRegex(V2PlanParseError, "milestone ID"):
            parse(answer("NEXT", milestone="m02", plan_text=SECOND_MILESTONE))

    def test_next_requires_its_milestone_and_its_closed_plan_block(self) -> None:
        with self.assertRaisesRegex(V2PlanParseError, "one complete NEXT PLAN block"):
            parse(answer("NEXT", milestone="M02"))
        with self.assertRaisesRegex(V2PlanParseError, "NEXT requires NEXT_MILESTONE"):
            parse(answer("NEXT", plan_text=SECOND_MILESTONE))

    def test_complete_and_spec_decision_refuse_an_embedded_plan(self) -> None:
        for raw in (answer("COMPLETE", plan_text=SECOND_MILESTONE),
                    answer("SPEC_DECISION", question="Which budget?", plan_text=SECOND_MILESTONE)):
            with self.subTest(decision=raw.splitlines()[3]), self.assertRaisesRegex(
                    V2PlanParseError, "must not contain a NEXT PLAN"):
                parse(raw)

    def test_a_blocked_plan_never_becomes_the_next_plan(self) -> None:
        with self.assertRaisesRegex(V2PlanParseError, "READY, never BLOCKED"):
            parse(answer("NEXT", milestone="M02", plan_text=BLOCKED_PLAN))

    def test_next_plan_obeys_the_run_step_budget_and_the_staged_policy(self) -> None:
        two_steps = plan(STEP, ("S02", "other.txt", "Cover the feature"))
        with self.assertRaisesRegex(V2PlanParseError, "max_steps_per_plan"):
            parse(answer("NEXT", milestone="M01", plan_text=two_steps),
                  planning=PlanningConfig(max_steps_per_plan=1))
        with self.assertRaisesRegex(V2PlanParseError, "requires STAGED"):
            parse(answer("NEXT", milestone="M01", plan_text=PLAN_RAW),
                  planning=PlanningConfig(execution_mode_policy="require-staged"))

    def test_a_legacy_plan_inside_the_block_is_refused(self) -> None:
        legacy = "STATUS: READY\nTITLE: Legacy plan\n\nOBJECTIVE\nDo it.\n\nEND META PLAN\n"
        with self.assertRaises(V2PlanParseError):
            parse(answer("NEXT", milestone="M01", plan_text=legacy))

    def test_the_envelope_is_unique_and_nothing_follows_the_footer(self) -> None:
        good = answer("COMPLETE")
        with self.assertRaisesRegex(V2PlanParseError, "exactly one META CONTINUE v1 envelope"):
            parse(good.replace("DECISION\n", "META CONTINUE v1\nDECISION\n", 1))
        with self.assertRaisesRegex(V2PlanParseError, "content after END META CONTINUE"):
            parse(good + "trailing prose\n")
        with self.assertRaisesRegex(V2PlanParseError, "exactly one END META CONTINUE footer"):
            parse(good.replace("END META CONTINUE\n", "", 1))

    def test_a_missing_duplicated_or_unknown_field_is_refused_verbatim(self) -> None:
        good = answer("COMPLETE")
        with self.assertRaisesRegex(V2PlanParseError, "every META CONTINUE field"):
            parse(good.replace("SUMMARY\nThe milestone is closed.\n", "", 1))
        with self.assertRaisesRegex(V2PlanParseError, "duplicate planner continue section SUMMARY"):
            parse(good.replace("SUMMARY\n", "SUMMARY\nExtra.\n\nSUMMARY\n", 1))
        with self.assertRaisesRegex(V2PlanParseError, "DECISION must be exactly"):
            parse(good.replace("COMPLETE", "DONE"))


class StagnationFingerprintTests(unittest.TestCase):
    def test_ordering_case_whitespace_and_duplicates_do_not_move_it(self) -> None:
        self.assertEqual(
            stagnation_fingerprint(["B", " a  b ", "a b"], "tree", ["Z"]),
            stagnation_fingerprint(["a b", "b"], "tree", ["z"]))

    def test_remaining_tree_and_failures_each_move_it(self) -> None:
        base = stagnation_fingerprint(["a"], "tree-1", ["f1"])
        moved = {
            base,
            stagnation_fingerprint(["a", "b"], "tree-1", ["f1"]),
            stagnation_fingerprint(["a"], "tree-2", ["f1"]),
            stagnation_fingerprint(["a"], "tree-1", ["f2"]),
        }
        self.assertEqual(len(moved), 4)

    def test_a_bare_string_is_never_a_sequence(self) -> None:
        with self.assertRaises(TypeError):
            stagnation_fingerprint("a", "tree", [])
        with self.assertRaises(TypeError):
            stagnation_fingerprint([], "tree", "f1")


class PlannerContinueServiceTests(unittest.TestCase):
    def service(self, *answers: str) -> tuple[PlannerContinue, ScriptedChat]:
        client = ScriptedChat(list(answers))
        return PlannerContinue(
            client=client, planning=PLANNING, check_catalog=CATALOG,
            default_check_ids=("test",)), client

    def test_a_transport_failure_propagates_unchanged(self) -> None:
        failure = LLMHTTPError("endpoint refused the connection")
        service, _ = self.service()
        service.client = ScriptedChat([failure])
        with self.assertRaises(LLMHTTPError) as raised:
            service.decide(facts())
        self.assertIs(raised.exception, failure)
        self.assertIsNone(service.last_usage)

    def test_decide_asks_once_on_the_planner_profile(self) -> None:
        service, client = self.service(answer("NEXT", milestone="M02", plan_text=SECOND_MILESTONE))
        result = service.decide(facts())
        self.assertIs(result.decision, ContinueDecision.NEXT)
        self.assertEqual(len(client.requests), 1)
        prompt = client.requests[0]
        self.assertIn("MILESTONE\nM01 :: First milestone", prompt)
        self.assertIn("GOAL\nThe feature exists.", prompt)
        self.assertIn("EXECUTION_MODE_POLICY\nauto", prompt)
        self.assertIn("CLOSED MILESTONES\n- M01 :: DONE", prompt)
        self.assertIn("REMAINING\n- Polish the reporting path.", prompt)
        self.assertIn("FAILURES\n- test:failure", prompt)
        self.assertIn("DIFFSTAT SINCE BASE\nfeature.txt | 2 +-", prompt)
        self.assertIn("REQUIRED_CHECKS\n- test", prompt)
        self.assertIn("SPEC: add the feature.", prompt)

    def test_the_request_holds_the_declared_sections_only(self) -> None:
        payload = build_planner_continue_payload(
            facts(), planning=PLANNING, check_catalog=CATALOG, default_check_ids=("test",))
        self.assertEqual(
            tuple(section.name for section in payload.sections),
            ("spec", "state", "plan", "audit", "evidence", "repository", "rules"))
        self.assertNotIn("{{", payload.rendered)
        self.assertFalse(payload.budget_overrun)
        fields = set(PlannerContinueFacts.__dataclass_fields__)
        for absent in ("logs", "worker_prompt", "worker_prompts", "transcript", "artifacts"):
            self.assertNotIn(absent, fields)

    def test_the_prompt_carries_the_run_policy_and_the_protocol_numbers(self) -> None:
        planning = PlanningConfig(max_steps_per_plan=4, execution_mode_policy="require-staged")
        payload = build_planner_continue_payload(
            facts(), planning=planning, check_catalog=CATALOG, default_check_ids=("test",))
        self.assertIn("compiled for 4 steps at most (S04)", payload.rendered)
        self.assertIn("EXECUTION_MODE_POLICY\nrequire-staged", payload.rendered)
        self.assertIn("REQUIRED_CHECKS\n- test", payload.rendered)

    def test_artifacts_roundtrip_and_never_absorb_the_run_tree(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            iterations = Path(name) / "iterations"
            worker = iterations / "01" / "implementer" / "agent.prompt.txt"
            worker.parent.mkdir(parents=True)
            worker.write_text("WORKER PROMPT SENTINEL\n" + "x" * 40000, encoding="utf-8")
            raw_answer = answer("NEXT", milestone="M02", plan_text=SECOND_MILESTONE)
            service, client = self.service(raw_answer)
            result = service.decide(facts(), iterations_dir=iterations)

            directory = planner_continue_dir(iterations, 1)
            self.assertEqual(directory, iterations / "01" / "planner-continue")
            request, raw, record = read_planner_continue_artifacts(directory)
            self.assertEqual(raw, raw_answer)
            self.assertNotIn("WORKER PROMPT SENTINEL", client.requests[0])
            self.assertEqual(sorted(request["facts"]), ["audit", "evidence", "plan", "repository",
                                                        "rules", "spec", "state"])
            self.assertEqual(request["facts"]["spec"], "SPEC: add the feature.")
            self.assertEqual(request["iteration"], 1)
            self.assertEqual(request["current_plan"]["steps"], ["S01"])
            self.assertEqual(request["prompt"]["sha256"], hashlib.sha256(
                client.requests[0].encode("utf-8")).hexdigest())
            self.assertEqual(record["decision"], "NEXT")
            self.assertEqual(record["next_milestone"], "M02")
            self.assertEqual(record["next_plan"], result.next_plan.raw)
            self.assertTrue(worker.read_text(encoding="utf-8").startswith("WORKER PROMPT"))

    def test_a_rejected_answer_stays_durable_before_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            iterations = Path(name) / "iterations"
            rejected = answer("COMPLETE").replace("COMPLETE", "DONE")
            service, _ = self.service(rejected)
            with self.assertRaises(V2PlanParseError):
                service.decide(facts(), iterations_dir=iterations)
            directory = planner_continue_dir(iterations, 1)
            self.assertEqual((directory / "raw.txt").read_text(encoding="utf-8"), rejected)
            self.assertTrue((directory / "request.json").is_file())
            self.assertFalse((directory / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
