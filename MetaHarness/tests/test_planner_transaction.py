"""Durable preapproval planner corrections with and without continuation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest import mock

from metaharness.gitops import resolve_tree
from metaharness.llm.chat import (
    ConversationUnavailableError, LLMConversationHandle, TextLLMResult,
)
from metaharness.plan_repository_validation import PlanRepositoryPreconditionError, RepositoryPreconditions
from metaharness.planning.planner import PlannerV2
from tests.pipeline_support import PipelineHarness, git
from tests.test_plan_repository_validation import impossible_plan_message, meta_plan


class _Chat:
    def __init__(self, answers: list[str], *, continue_mode: str = "available") -> None:
        self.answers = list(answers)
        self.continue_mode = continue_mode
        self.complete_calls: list[str] = []
        self.continue_calls: list[tuple[LLMConversationHandle, str]] = []

    def _answer(self) -> TextLLMResult:
        return TextLLMResult(
            self.answers.pop(0), "fake", {"total_tokens": 3}, {},
            conversation=LLMConversationHandle("fake", "private-conversation-id"),
        )

    def complete(self, prompt: str) -> TextLLMResult:
        self.complete_calls.append(prompt)
        return self._answer()

    def continue_conversation(self, handle: LLMConversationHandle, prompt: str) -> TextLLMResult:
        self.continue_calls.append((handle, prompt))
        if self.continue_mode == "unavailable":
            raise ConversationUnavailableError("conversation unavailable")
        if self.continue_mode == "timeout":
            raise TimeoutError("ambiguous timeout")
        return self._answer()


class _Stateless:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.answers.pop(0)


class PlannerTransactionTests(PipelineHarness):
    def setUp(self) -> None:
        super().setUp()
        self.target = self.root / "planner-run"
        self.target.mkdir()
        self.config_value = self.config()
        (self.repo / "other.txt").write_text("existing\n", encoding="utf-8")
        (self.repo / "src").mkdir()
        (self.repo / "src/module.py").write_text(
            "def evidence_symbol():\n    return 'immutable source evidence'\n",
            encoding="utf-8",
        )
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "other file")
        self.create_existing = meta_plan(
            {"read": ("feature.txt",), "write": ("feature.txt",), "create": ("other.txt",)},
        )
        # The only mutation is impossible: no Git fact can rescue this plan.
        self.invalid = impossible_plan_message()
        self.valid = meta_plan({"read": ("feature.txt",), "write": ("feature.txt",)})

    @staticmethod
    def blocked_repository_evidence(path: str = "src/module.py", symbol: str = "evidence_symbol") -> str:
        return (
            "META PLAN v2\n\nSTATUS: BLOCKED\nTITLE: Need source evidence\n"
            "BLOCKER_KIND: REPOSITORY_EVIDENCE\n\nOBJECTIVE\nMake feature.txt good.\n\n"
            f"BLOCKERS\n- {path} :: {symbol}\n\nEND META PLAN\n"
        )

    def planner(self, client, *, budget: int = 2, events=None) -> PlannerV2:
        return PlannerV2(
            client, planning=replace(self.config_value.planning, max_preapproval_corrections=budget),
            check_catalog=self.config_value.check_catalog,
            default_check_ids=self.config_value.default_check_ids,
            repository_preconditions=RepositoryPreconditions(self.repo, resolve_tree(self.repo, "HEAD")),
            on_event=events,
        )

    def run_plan(self, client, *, budget: int = 2, events=None):
        return self.planner(client, budget=budget, events=events).plan(
            "Make feature.txt good.", "feature.txt exists", artifacts_dir=self.target,
        )

    def test_valid_initial_answer_has_no_correction(self) -> None:
        chat = _Chat([self.valid])
        self.run_plan(chat)
        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(chat.continue_calls, [])
        self.assertTrue((self.target / "implementation_bundle.json").is_file())

    def test_repository_evidence_blocker_recovers_from_immutable_named_path(self) -> None:
        chat = _Chat([self.blocked_repository_evidence(), self.valid])
        (self.repo / "src/module.py").write_text("def changed_in_worktree(): pass\n", encoding="utf-8")
        plan = self.run_plan(chat)
        self.assertEqual(plan.raw, self.valid)
        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(len(chat.continue_calls), 1)
        prompt = chat.continue_calls[0][1]
        self.assertIn("REPOSITORY EVIDENCE", prompt)
        self.assertIn("immutable source evidence", prompt)
        self.assertNotIn("changed_in_worktree", prompt)
        self.assertIn("IMMUTABLE TREE", prompt)
        self.assertIn("COMPLETE META PLAN v2", prompt)
        validation = json.loads(
            (self.target / "planner-attempts/01/planner.validation.json").read_text()
        )
        self.assertEqual(validation["blocker_kind"], "REPOSITORY_EVIDENCE")

    def test_repository_evidence_paths_are_validated_before_reading(self) -> None:
        chat = _Chat([self.blocked_repository_evidence("../README.md", "heading"), self.valid])
        self.run_plan(chat)
        prompt = chat.continue_calls[0][1]
        self.assertIn("rejected (not a valid repo-relative path)", prompt)
        self.assertNotIn("pipeline fixture", prompt)

    def test_same_conversation_receives_small_correction_and_keeps_handle_private(self) -> None:
        chat = _Chat([self.invalid, self.valid])
        events = []
        self.run_plan(chat, events=lambda name, data: events.append((name, data)))
        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(len(chat.continue_calls), 1)
        self.assertEqual(chat.continue_calls[0][0].conversation_id, "private-conversation-id")
        prompt = chat.continue_calls[0][1]
        self.assertIn("S01: no_mutation", prompt)
        self.assertIn("Re-emit one COMPLETE META PLAN v2", prompt)
        # The correction stays a bounded error list, never a plan copy.
        self.assertNotIn("gone.txt", prompt)
        self.assertNotIn("ORIGINAL SPEC", prompt)
        self.assertNotIn("Make feature.txt good.", prompt)
        self.assertEqual((self.target / "planner.raw.md").read_text(), self.valid)
        self.assertEqual(
            json.loads((self.target / "planner-attempts/01/planner.validation.json").read_text())["errors"][0]["code"],
            "no_mutation",
        )
        self.assertNotIn("private-conversation-id", str(events))
        self.assertEqual((self.target / "planner.session.json").stat().st_mode & 0o777, 0o600)

    def test_stateless_and_explicit_unavailable_use_fresh_request(self) -> None:
        for client in (_Stateless([self.invalid, self.valid]), _Chat([self.invalid, self.valid], continue_mode="unavailable")):
            target = self.target if not (self.target / "planner.raw.md").exists() else self.root / "another"
            target.mkdir(exist_ok=True)
            self.planner(client).plan("Make feature.txt good.", "feature.txt exists", artifacts_dir=target)
            requests = client.calls if isinstance(client, _Stateless) else client.complete_calls
            self.assertEqual(len(requests), 2)
            self.assertIn("ORIGINAL SPEC", requests[1])
            self.assertIn(self.invalid, requests[1])

    def test_ambiguous_timeout_never_starts_fresh(self) -> None:
        chat = _Chat([self.invalid], continue_mode="timeout")
        with self.assertRaises(TimeoutError):
            self.run_plan(chat)
        self.assertEqual(len(chat.complete_calls), 1)

    def test_syntax_error_keeps_handle_and_continues(self) -> None:
        chat = _Chat(["invalid answer", self.valid])
        self.run_plan(chat)
        self.assertEqual(len(chat.continue_calls), 1)

    def test_one_correction_contains_errors_from_multiple_steps(self) -> None:
        body = self.invalid.split("BEGIN STEP S01\n", 1)[1].split("END STEP S01", 1)[0]
        second = body.replace("DEPENDS_ON: NONE", "DEPENDS_ON: S01")
        two_steps = (self.invalid
            .replace("EXECUTION_MODE: SINGLE", "EXECUTION_MODE: STAGED")
            .replace("STEP_COUNT: 1", "STEP_COUNT: 2")
            .replace("END STEP S01", "END STEP S01\n\nBEGIN STEP S02\n" + second + "END STEP S02"))
        chat = _Chat([two_steps, self.valid])
        self.run_plan(chat)
        prompt = chat.continue_calls[0][1]
        self.assertIn("S01: no_mutation", prompt)
        self.assertIn("S02: no_mutation", prompt)

    def test_aw010_existing_reference_corpus_is_normalized_without_a_correction(self) -> None:
        """Run 20260923T131351Z-e9a34fd827: the file exists, the planner wrote CREATE."""

        path = "backend/src/cti_app/domain/reference_corpus.py"
        corpus = self.repo / path
        corpus.parent.mkdir(parents=True)
        corpus.write_text("class ReferenceMember:\n    pass\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "reference corpus")
        answer = meta_plan({"read": ("feature.txt",), "write": ("feature.txt",), "create": (path,)})
        chat = _Chat([answer])

        plan = self.run_plan(chat)

        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(chat.continue_calls, [])
        (step,) = plan.steps
        self.assertEqual(step.create_set, ())
        self.assertEqual(step.write_set, ("feature.txt", path))
        self.assertEqual(
            json.loads((self.target / "plan.normalizations.json").read_text())["steps"]["S01"],
            [
                {"code": "CREATE_EXISTING_TO_WRITE", "path": path},
                {"code": "ADD_MUTATION_TO_READ", "path": path},
            ],
        )
        contract = (self.target / "steps/S01/contract.md").read_text(encoding="utf-8")
        self.assertIn(path, contract.split("WRITE SET", 1)[1].split("CREATE SET", 1)[0])

    def test_budget_zero_stops_before_worker_or_bundle(self) -> None:
        chat = _Chat([self.invalid])
        with self.assertRaises(PlanRepositoryPreconditionError):
            self.run_plan(chat, budget=0)
        self.assertEqual(chat.continue_calls, [])
        self.assertFalse((self.target / "implementation_bundle.json").exists())

    def test_two_corrections_and_exhaustion(self) -> None:
        chat = _Chat([self.invalid, self.invalid, self.valid])
        self.run_plan(chat)
        self.assertEqual(len(chat.continue_calls), 2)
        target = self.root / "exhausted"
        target.mkdir()
        chat2 = _Chat([self.invalid] * 3)
        with self.assertRaises(PlanRepositoryPreconditionError):
            self.planner(chat2).plan("Make feature.txt good.", "feature.txt exists", artifacts_dir=target)
        self.assertEqual(len(chat2.continue_calls), 2)
        self.assertFalse((target / "implementation_bundle.json").exists())

    def test_resume_reuses_paid_raw_before_validation(self) -> None:
        chat = _Chat([self.valid])
        with mock.patch("metaharness.planning.planner.parse_task_plan_v2", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError):
                self.run_plan(chat)
        self.run_plan(chat)
        self.assertEqual(len(chat.complete_calls), 1)

    def test_resume_reuses_raw_if_usage_persistence_crashes(self) -> None:
        chat = _Chat([self.valid])
        with mock.patch("metaharness.planning.planner.write_usage_artifact", side_effect=OSError("crash")):
            with self.assertRaises(OSError):
                self.run_plan(chat)
        self.assertEqual((self.target / "planner.raw.md").read_text(), self.valid)
        self.assertFalse((self.target / "planner.session.json").exists())
        self.run_plan(chat)
        self.assertEqual(len(chat.complete_calls), 1)

    def test_resume_after_failed_validation_continues_once(self) -> None:
        chat = _Chat([self.invalid, self.valid], continue_mode="timeout")
        with self.assertRaises(TimeoutError):
            self.run_plan(chat)
        chat.continue_mode = "available"
        self.run_plan(chat)
        self.assertEqual(len(chat.complete_calls), 1)
        self.assertEqual(len(chat.continue_calls), 2)

    def test_resume_reuses_paid_correction_raw(self) -> None:
        chat = _Chat([self.invalid, self.valid])
        original = __import__(
            "metaharness.planning.planner", fromlist=["parse_task_plan_v2"]
        ).parse_task_plan_v2
        calls = 0
        def parse_then_crash(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("crash")
            return original(*args, **kwargs)
        with mock.patch("metaharness.planning.planner.parse_task_plan_v2", side_effect=parse_then_crash):
            with self.assertRaises(RuntimeError):
                self.run_plan(chat)
        self.run_plan(chat)
        self.assertEqual(len(chat.continue_calls), 1)
