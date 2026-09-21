"""P49: one bounded retry and safe recovery of META CONTRACT MISMATCH.

A step whose first worker changed nothing at all and returned a structural
mismatch is retried exactly once, with the same approved contract, the same
profile, the same mutable scope and the same candidate tree, in a fresh Codex
process.  The retry only adds a prompt addendum; it never widens
WRITE/CREATE/DELETE and there is never a third attempt.

Every pipeline test uses a real temporary Git repository, a real local bare
remote, the real orchestrator state machine and the real Git primitives.
Planner, Luna, Claude and reviewers are in-process fakes: no network, no LLM.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import orchestrator as orchestrator_module  # noqa: E402
from metaharness.agent.base import AgentResult  # noqa: E402
from metaharness.agent.codex import (  # noqa: E402
    build_implementer_step_prompt,
    build_mismatch_retry_addendum,
    deferred_verify_dependency,
)
from metaharness.models import RunStatus  # noqa: E402
from metaharness.orchestrator import (  # noqa: E402
    DeferredStepExecutionOutcome,
    Orchestrator,
    StepExecutionFailure,
)
from metaharness.resume import ResumePhase, read_checkpoint, resume_info  # noqa: E402
from metaharness.usage import add_usage, phase_usage_summary  # noqa: E402

from tests.test_p28_full_pipeline import (  # noqa: E402
    PASS,
    REVISE_IMPLEMENTATION,
    SINGLE_PLAN,
    CleanMismatchLuna,
    FakeClaude,
    FakeLuna,
    QueueClient,
    git,
    plan_text,
    step_block,
    write,
    writer,
)
from tests.test_p29 import P29Harness  # noqa: E402

# The realistic shape of the reported failure: a step's own transformation is
# possible, but a file a *later* step owns still references the removed symbol.
MISMATCH_TEXT = "future.py outside WRITE_SET still references removed symbol"
DEFERRED_REPORT = (
    "S03 report\n"
    "\n"
    "Implemented the approved in-scope change in src/c.py.\n"
    "\n"
    "DEFERRED VERIFY DEPENDENCY\n"
    "- failing command: pytest tests/test_future.py\n"
    "- out-of-scope path: src/future.py\n"
    "- later step that owns it: S04\n"
)


def modify_create_delete(root: Path) -> None:
    write(root / "src/b.py", "B = 3\n")
    write(root / "src/c.py", "C = 3\n")
    (root / "src/future.py").unlink()

# S01 modifies A, S02 modifies B, S03 creates its own file, S04 owns future.py.
FOUR_STEP_PLAN = plan_text(
    step_block(1),
    step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
    step_block(3, read=("src/a.py",), write_set=(), create=("src/c.py",)),
    step_block(4, read=("src/future.py",), write_set=("src/future.py",)),
    title="P49 staged migration",
)
FOUR_STEPS = ("S01", "S02", "S03", "S04")
DIRTY_MUTATION_PLAN = plan_text(
    step_block(1),
    step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
    step_block(
        3,
        read=("src/a.py", "src/b.py", "src/future.py"),
        write_set=("src/b.py",),
        create=("src/c.py",),
        delete=("src/future.py",),
        operation="Modify, create and delete",
    ),
    title="P49 dirty mismatch recovery",
)
# Two C02 repair steps, both inside the C01 mutable scope.
REPAIR_TWO_STEP_PLAN = plan_text(
    step_block(1, operation="Repair"),
    step_block(2, operation="Repair again"),
    title="P49 two-step repair",
)


class ReportingLuna(CleanMismatchLuna):
    """Clean-mismatch double that can also emit a free-form final report."""

    def __init__(self, behaviors: dict[Any, Any], mismatch_steps: set[str],
                 *, reports: dict[str, str] | None = None,
                 mismatch_text: str = MISMATCH_TEXT):
        super().__init__(behaviors, mismatch_steps, mismatch_text=mismatch_text)
        self.reports = reports or {}

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, **kwargs: Any) -> AgentResult:
        result = super().run_step(contract, worktree, artifacts_dir, **kwargs)
        call = self.calls[-1]
        report = self.reports.get(f"{call['step']}#{call['attempt']}")
        if report is None:
            return result
        Path(artifacts_dir, "agent.final.md").write_text(report, encoding="utf-8")
        return dataclasses.replace(result, final_message=report)


class C02DirtyLuna(ReportingLuna):
    """Report a dirty mismatch on the first C02 repair-step attempt."""

    def run_step(self, contract: str, worktree: Any, artifacts_dir: Any, **kwargs: Any) -> AgentResult:
        result = super().run_step(contract, worktree, artifacts_dir, **kwargs)
        call = self.calls[-1]
        if call["cycle"] == 2 and call["step"] == "S01" and call["attempt"] == 1:
            message = "META CONTRACT MISMATCH v1\nC02 tentative mismatch\n"
            Path(artifacts_dir, "agent.final.md").write_text(message, encoding="utf-8")
            return dataclasses.replace(result, final_message=message)
        return result


class LegacyMismatchOrchestrator(Orchestrator):
    """Reproduces the pre-P49 behavior: any clean mismatch is terminal.

    Used only to persist the exact artifacts of an already failed run, so the
    recovery path is exercised against a real historical state and not a
    hand-written fixture.
    """

    def _execute_codex_step(self, **kwargs: Any):  # type: ignore[override]
        kwargs.pop("pending_mismatch_retry", None)
        outcome = self._run_codex_step_attempt(
            **kwargs, initial_mismatch=None, mismatch_retry_count=0,
        )
        if not isinstance(outcome, DeferredStepExecutionOutcome):
            return outcome
        # The old code never wrote a deferred record.
        (Path(kwargs["artifact_dir"]) / "step.json").unlink(missing_ok=True)
        raise StepExecutionFailure(
            "AGENT_CONTRACT_MISMATCH", outcome.step_id, outcome.mismatch,
            profile_id=outcome.profile_id, tree_before=outcome.tree_before,
            tree_after=outcome.tree_after, usage=outcome.usage,
            mismatch=outcome.mismatch,
        )


class DriftingOrchestrator(Orchestrator):
    """Simulates a concurrent writer between mismatch #1 and its retry.

    The whole detection and the fail-closed verdict are the real ones; only
    the foreign write is injected, exactly where a real concurrent writer
    would land: after the first attempt's clean mismatch, before the retry.
    """

    def _run_codex_step_attempt(self, **kwargs: Any):  # type: ignore[override]
        outcome = super()._run_codex_step_attempt(**kwargs)
        if isinstance(outcome, DeferredStepExecutionOutcome):
            worktree = Path(kwargs["worktree"])
            write(worktree / "src/b.py", "B = drifted\n")
            git(worktree, "add", "--all")
        return outcome


class CrashBeforeBoundedRetryOrchestrator(Orchestrator):
    """Crash after a synthetic mismatch is durable, before its retry starts."""

    def _run_codex_step_attempt(self, **kwargs: Any):  # type: ignore[override]
        outcome = super()._run_codex_step_attempt(**kwargs)
        if (
            isinstance(outcome, DeferredStepExecutionOutcome)
            and kwargs.get("mismatch_retry_count") == 0
        ):
            raise RuntimeError("simulated crash before bounded retry")
        return outcome


# The AW-002 recovery shape: eight completed steps, a clean mismatch on S09
# and one step left after it.
TEN_STEPS = tuple(f"S{number:02d}" for number in range(1, 11))
TEN_STEP_PLAN = plan_text(
    *[
        step_block(number, read=("src/a.py",), write_set=(),
                   create=(f"src/m{number:02d}.py",))
        for number in range(1, 11)
    ],
    title="P49 nine-step migration",
)


class PromptAddendumTests(unittest.TestCase):
    def test_contract_is_unchanged_and_the_addendum_follows_it(self) -> None:
        contract = "STEP\nS03 / 4\nWRITE SET\n- src/c.py\n"
        plain = build_implementer_step_prompt(contract)
        self.assertNotIn("MISMATCH RETRY ADDENDUM", plain)
        addendum = build_mismatch_retry_addendum(
            initial_mismatch=MISMATCH_TEXT,
            future_ownership={"S04": ("src/future.py",), "S05": ("src/last.py",)},
        )
        retry = build_implementer_step_prompt(contract, retry_addendum=addendum)
        # The contract itself is byte-identical and still authoritative.
        self.assertIn(contract, retry)
        self.assertEqual(retry[:retry.index("<MISMATCH RETRY ADDENDUM>")],
                         plain[:plain.index("</STEP CONTRACT>")] + "</STEP CONTRACT>\n\n")
        self.assertLess(retry.index("</STEP CONTRACT>"), retry.index("<MISMATCH RETRY ADDENDUM>"))
        for expected in (
            "This addendum does NOT expand your mutable scope.",
            "Do not modify any path outside WRITE_SET / CREATE_SET / DELETE_SET.",
            "Do not move work into a path assigned to a later step.",
            "Do not add compatibility shims merely to make an intermediate verification pass.",
            "DEFERRED VERIFY DEPENDENCY",
            "META CONTRACT MISMATCH v1",
            "<FUTURE APPROVED OWNERSHIP>",
            "S04:\n  src/future.py",
            "S05:\n  src/last.py",
            "This section is informative only.",
            "Future-step paths are NOT writable in this retry.",
            MISMATCH_TEXT,
        ):
            self.assertIn(expected, retry, expected)

    def test_the_retry_must_defer_instead_of_returning_no_change(self) -> None:
        addendum = build_mismatch_retry_addendum(initial_mismatch=MISMATCH_TEXT)
        for expected in (
            "If no repository change is necessary or safely possible inside this step's",
            "scope, do not exit as a normal successful no-change result. Return",
            "META CONTRACT MISMATCH v1 with the bounded explanation so MetaHarness can",
            "defer the step safely.",
        ):
            self.assertIn(expected, addendum, expected)

    def test_no_future_steps_renders_no_ownership_section(self) -> None:
        addendum = build_mismatch_retry_addendum(
            initial_mismatch=MISMATCH_TEXT, future_ownership={},
        )
        self.assertIn("<MISMATCH RETRY ADDENDUM>", addendum)
        self.assertNotIn("FUTURE APPROVED OWNERSHIP", addendum)

    def test_future_ownership_lists_only_remaining_mutation_paths(self) -> None:
        steps = [
            _FakeStep("S01", ("src/a.py",)),
            _FakeStep("S02", ("src/b.py",)),
            _FakeStep("S03", ()),
            _FakeStep("S04", ("src/future.py",)),
        ]
        self.assertEqual(
            orchestrator_module._future_step_ownership(steps, 1),
            {"S04": ("src/future.py",)},
        )
        self.assertEqual(orchestrator_module._future_step_ownership(steps, 3), {})

    def test_deferred_verify_dependency_is_extracted_from_the_report(self) -> None:
        self.assertIn("src/future.py", deferred_verify_dependency(DEFERRED_REPORT))
        self.assertIsNone(deferred_verify_dependency("S03 report\nall good\n"))
        self.assertIsNone(deferred_verify_dependency("DEFERRED VERIFY DEPENDENCY\n"))


@dataclasses.dataclass(frozen=True)
class _FakeStep:
    id: str
    write_set: tuple[str, ...]
    create_set: tuple[str, ...] = ()
    delete_set: tuple[str, ...] = ()


class MismatchRetryHarness(P29Harness):
    def setUp(self) -> None:
        super().setUp()
        # S04's approved path must exist in the base tree.
        write(self.repo / "src/future.py", "from src.a import removed\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "future module")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")

    def migration_luna(self, **kwargs: Any) -> ReportingLuna:
        """The realistic four-step migration; S03 clean-mismatches once."""

        behaviors: dict[Any, Any] = {
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (1, "S02"): writer("src/b.py", "B = 2\n"),
            # S03 attempt 1 changes nothing at all; attempt 2 does its own work.
            (1, "S03", 1): "nochange",
            (1, "S03", 2): writer("src/c.py", "C = 3\n"),
            (1, "S04"): writer("src/future.py", "from src.c import present\n"),
        }
        behaviors.update(kwargs.pop("behaviors", {}))
        return ReportingLuna(behaviors, kwargs.pop("mismatch_steps", {"S03#1"}), **kwargs)


class BoundedRetryTests(MismatchRetryHarness):
    def test_successful_no_change_retries_and_then_completes(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            mismatch_steps=set(),
            behaviors={(1, "S03", 1): "nochange",
                        (1, "S03", 2): writer("src/c.py", "C = 3\n")},
        )
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "nochange-retry-ok", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03", "S04"])
        self.assertIsNone(luna.calls[2]["retry_addendum"])
        self.assertIn("Worker completed successfully without producing an in-scope candidate",
                      luna.calls[3]["retry_addendum"])
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((step["status"], step["mismatch_retry_count"], step["changed_paths"]),
                         ("COMPLETED", 1, ["src/c.py"]))

    def test_successful_no_change_twice_is_deferred_and_chain_continues(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            mismatch_steps=set(), behaviors={(1, "S03", 2): "nochange"},
        )
        orchestrator, _planner, _reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "nochange-deferred", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03", "S04"])
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((step["status"], step["tree_before"], step["tree_after"],
                          step["changed_paths"], step["mismatch_retry_count"]),
                         ("DEFERRED_CONTRACT_MISMATCH", step["tree_before"],
                          step["tree_before"], [], 1))
        self.assertIn("No in-scope change remained necessary after bounded retry", step["mismatch"])
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])

    def test_successful_no_change_twice_without_claude_is_unresolved(self) -> None:
        config = self.make_config(revision=False)
        luna = self.migration_luna(
            mismatch_steps=set(), behaviors={(1, "S03", 2): "nochange"},
        )
        orchestrator, _planner, reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "nochange-unresolved", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "UNRESOLVED_CONTRACT_MISMATCH")
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03", "S04"])
        self.assertEqual((claude.calls, reviewer.prompts), ([], []))

    def test_no_change_with_head_drift_fails_closed(self) -> None:
        def move_head(root: Path) -> None:
            git(root, "commit", "--allow-empty", "-qm", "foreign head move")

        config = self.make_config()
        luna = self.migration_luna(mismatch_steps=set(), behaviors={(1, "S03", 1): move_head})
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "nochange-head-drift", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "AGENT_COMMITTED")
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])

    def test_no_change_with_untracked_out_of_scope_file_fails_closed(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            mismatch_steps=set(),
            behaviors={(1, "S03", 1): writer("src/leftover.py", "leftover = True\n")},
        )
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "nochange-untracked", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])

    def test_exit1_without_change_is_not_retried_as_no_change(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(mismatch_steps=set(), behaviors={(1, "S03", 1): "exit1"})
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "nochange-exit1", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "AGENT_FAILED")
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])

    def test_timeout_without_change_is_not_retried_as_no_change(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(mismatch_steps=set(), behaviors={(1, "S03", 1): "timeout"})
        orchestrator, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        result = self.run_approved(config, orchestrator, "nochange-timeout", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "AGENT_TIMEOUT")
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])

    def test_resume_between_no_change_attempts_never_creates_a_third_attempt(self) -> None:
        config = self.make_config()
        first_luna = self.migration_luna(mismatch_steps=set())
        first = CrashBeforeBoundedRetryOrchestrator(
            config,
            planner_client=QueueClient("planner", [FOUR_STEP_PLAN], self.events),
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=first_luna,
            reviser=FakeClaude(log=self.events),
        )
        failed = self.run_approved(config, first, "nochange-crash", FOUR_STEPS)
        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual([call["step"] for call in first_luna.calls], ["S01", "S02", "S03"])

        retry_luna = self.migration_luna(
            mismatch_steps=set(),
            behaviors={(1, "S03", 1): writer("src/c.py", "C = 3\n"),
                        (1, "S04"): writer("src/future.py", "from src.c import present\n")},
        )
        resumed_orchestrator, planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry_luna,
        )
        with self.count_pushes():
            resumed = resumed_orchestrator.resume("nochange-crash")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in retry_luna.calls], ["S03", "S04"])
        self.assertIn("<MISMATCH RETRY ADDENDUM>", retry_luna.calls[0]["retry_addendum"])
        self.assertEqual(planner.prompts, [])

    def test_clean_mismatch_retry_completes_and_defers_only_the_verification(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(reports={"S03#2": DEFERRED_REPORT})
        orchestrator, planner, _reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "retry-ok", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        # S01 and S02 run exactly once: a retry never replays an earlier step.
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03", "S04"])
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 1, 1, 2, 1])
        # Only the second S03 call carries the addendum.
        self.assertIsNone(luna.calls[2]["retry_addendum"])
        addendum = luna.calls[3]["retry_addendum"]
        self.assertIn("<MISMATCH RETRY ADDENDUM>", addendum)
        self.assertIn("This addendum does NOT expand your mutable scope.", addendum)
        self.assertIn(MISMATCH_TEXT, addendum)
        # The retry is told who owns future.py, and that it stays read-only.
        self.assertIn("S04:\n  src/future.py", addendum)
        self.assertIn("Future-step paths are NOT writable in this retry.", addendum)
        # Same approved contract, byte for byte, on both attempts.
        self.assertEqual(luna.calls[2]["contract"], luna.calls[3]["contract"])
        self.assertIn("src/c.py", luna.calls[3]["contract"])

        # S03 ends COMPLETED with its own in-scope change only.
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertEqual(step["changed_paths"], ["src/c.py"])
        self.assertIn(MISMATCH_TEXT, step["initial_mismatch"])
        self.assertIn("src/future.py", step["deferred_verify"])
        # Attempt 1 keeps its own diagnostics.
        archived = json.loads((result.run_dir / "steps/S03/attempts/01/step.json").read_text())
        self.assertEqual(archived["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertIn(MISMATCH_TEXT, archived["mismatch"])
        self.assertNotIn("MISMATCH RETRY ADDENDUM",
                         (result.run_dir / "steps/S03/attempts/01/agent.final.md").read_text())

        # S03 never touched future.py; S04 is the step that changed it.
        self.assertNotIn("src/future.py", step["changed_paths"])
        s04 = json.loads((result.run_dir / "steps/S04/step.json").read_text())
        self.assertEqual(s04["changed_paths"], ["src/future.py"])

        # The reviewer receives the deferred verify dependency; Claude gets the
        # compact SPEC-only revision request.
        self.assertEqual(len(claude.calls), 1)
        prompt = claude.calls[0]["prompt"]
        self.assertIn("<SPEC>", prompt)
        self.assertNotIn("DEFERRED_VERIFY_DEPENDENCY", prompt)
        self.assertIn("src/future.py", prompt)
        self.assertNotIn("pytest tests/test_future.py", prompt)
        self.assertEqual(len(planner.prompts), 1)
        self.assertEqual(pushed.call_count, 1)

    def test_a_second_clean_mismatch_is_deferred_and_the_chain_continues(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            behaviors={(1, "S03", 2): "nochange"}, mismatch_steps={"S03"},
        )
        orchestrator, _planner, reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
            claude=FakeClaude({1: writer("src/c.py", "C = 3\n")}),
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "retry-deferred", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        # Exactly two attempts of S03, then S04 runs.  Never a third attempt.
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03", "S04"])
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(step["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertEqual(step["changed_paths"], [])
        self.assertEqual(step["tree_before"], step["tree_after"])
        self.assertIn(MISMATCH_TEXT, step["mismatch"])
        self.assertIn(MISMATCH_TEXT, step["initial_mismatch"])
        self.assertEqual([item["status"] for item in result.state["steps"]],
                         ["completed", "completed", "deferred", "completed"])
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])
        self.assertIn("DEFERRED CONTRACT MISMATCHES", reviewer.prompts[0])
        self.assertIn("S03", reviewer.prompts[0])

    def test_a_dirty_mismatch_is_recoverable_after_the_bounded_retry(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            behaviors={(1, "S03", 2): writer("src/c.py", "C = 3\n")},
            mismatch_steps={"S03"},
        )
        orchestrator, _planner, reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "retry-dirty", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        self.assertIn("worker left candidate modifications", result.state["failure"]["detail"])
        self.assertEqual([call["step"] for call in luna.calls],
                         ["S01", "S02", "S03", "S03"])
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((step["status"], step["reason"]), ("FAILED", "AGENT_CONTRACT_MISMATCH"))
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertIs(step["mismatch_clean"], False)
        self.assertEqual((claude.calls, reviewer.prompts, pushed.call_count), ([], [], 0))
        info = resume_info(result.run_dir, result.state)
        self.assertTrue(info.resumable, info.reason)
        self.assertEqual(info.label, "RECOVER S03 AFTER CONTRACT MISMATCH")

        retry = ReportingLuna(
            {(1, "S04"): writer("src/future.py", "from src.c import present\n")}, set(),
        )
        second, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as retry_push:
            resumed = second.resume("retry-dirty")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in retry.calls], ["S04"])
        recovery = json.loads(
            (resumed.run_dir / "steps/S03/mismatch_recovery.json").read_text()
        )
        self.assertEqual(recovery["mode"], "rollback_in_scope_dirty_mismatch")
        self.assertEqual(recovery["restored_paths"], ["src/c.py"])
        self.assertEqual(recovery["mismatch_retry_count_before"], 1)
        self.assertEqual(retry_push.call_count, 1)

    def test_dirty_mismatch_rolls_back_only_the_failed_step_and_retries_once(self) -> None:
        config = self.make_config()
        first = ReportingLuna(
            {
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (1, "S02"): writer("src/b.py", "B = 2\n"),
                (1, "S03"): writer("src/c.py", "C = tentative\n"),
            },
            {"S03"},
        )
        initial, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=first,
        )
        failed = self.run_approved(config, initial, "dirty-recover", FOUR_STEPS)
        self.assertEqual(failed.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        before_s03 = json.loads(
            (failed.run_dir / "steps/S02/step.json").read_text()
        )["tree_after"]

        retry = ReportingLuna(
            {
                (1, "S03"): writer("src/c.py", "C = final\n"),
                (1, "S04"): writer("src/future.py", "from src.c import present\n"),
            },
            set(),
        )
        second, planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("dirty-recover")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in retry.calls], ["S03", "S04"])
        self.assertIn("<MISMATCH RETRY ADDENDUM>", retry.calls[0]["retry_addendum"])
        self.assertEqual(planner.prompts, [])
        self.assertEqual(
            json.loads((resumed.run_dir / "steps/S01/step.json").read_text())["status"],
            "COMPLETED",
        )
        self.assertEqual(
            json.loads((resumed.run_dir / "steps/S02/step.json").read_text())["tree_after"],
            before_s03,
        )
        recovery = json.loads(
            (resumed.run_dir / "steps/S03/mismatch_recovery.json").read_text()
        )
        self.assertEqual(recovery["restored_paths"], ["src/c.py"])
        self.assertEqual(recovery["mismatch_retry_count_before"], 0)
        self.assertEqual(pushed.call_count, 1)

    def test_dirty_mismatch_restores_write_create_and_delete_sets(self) -> None:
        config = self.make_config()
        first = ReportingLuna(
            {
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (1, "S02"): writer("src/b.py", "B = 2\n"),
                (1, "S03"): modify_create_delete,
            },
            {"S03"},
        )
        initial, *_rest = self.orchestrator(
            config, plans=[DIRTY_MUTATION_PLAN], reviews=[PASS], luna=first,
        )
        failed = self.run_approved(config, initial, "dirty-mutations", ("S01", "S02", "S03"))

        retry = ReportingLuna({(1, "S03"): modify_create_delete}, set())
        second, *_rest = self.orchestrator(config, reviews=[PASS], luna=retry)
        with self.count_pushes() as pushed:
            resumed = second.resume("dirty-mutations")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in retry.calls], ["S03"])
        recovery = json.loads(
            (resumed.run_dir / "steps/S03/mismatch_recovery.json").read_text()
        )
        self.assertEqual(
            recovery["restored_paths"], ["src/b.py", "src/c.py", "src/future.py"]
        )
        worktree = Path(resumed.state["worktree"])
        self.assertEqual((worktree / "src/b.py").read_text(), "B = 3\n")
        self.assertEqual((worktree / "src/c.py").read_text(), "C = 3\n")
        self.assertFalse((worktree / "src/future.py").exists())
        self.assertEqual(pushed.call_count, 1)

    def test_dirty_mismatch_recovery_uses_the_hash_bound_c02_step_scope(self) -> None:
        config = self.make_config()
        first = C02DirtyLuna(
            {
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (2, "S01"): writer("src/a.py", "A = tentative\n"),
                (2, "S02"): writer("src/a.py", "A = 4\n"),
            },
            set(),
        )
        initial, *_rest = self.orchestrator(
            config,
            plans=[SINGLE_PLAN, REPAIR_TWO_STEP_PLAN],
            reviews=[REVISE_IMPLEMENTATION, PASS],
            luna=first,
        )
        failed = self.run_approved(config, initial, "dirty-c02", ("S01",))
        self.assertEqual(failed.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        self.assertEqual(resume_info(failed.run_dir, failed.state).label,
                         "RECOVER S01 AFTER CONTRACT MISMATCH")

        retry = ReportingLuna(
            {
                (2, "S01"): writer("src/a.py", "A = 3\n"),
                (2, "S02"): writer("src/a.py", "A = 4\n"),
            },
            set(),
        )
        second, *_rest = self.orchestrator(config, reviews=[PASS], luna=retry)
        with self.count_pushes() as pushed:
            resumed = second.resume("dirty-c02")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([(call["cycle"], call["step"]) for call in retry.calls],
                         [(2, "S01"), (2, "S02")])
        recovery = json.loads(
            (resumed.run_dir / "repair/C02/steps/S01/mismatch_recovery.json").read_text()
        )
        self.assertEqual(recovery["restored_paths"], ["src/a.py"])
        self.assertEqual(pushed.call_count, 1)

    def test_dirty_mismatch_outside_current_step_scope_fails_closed_without_restore(self) -> None:
        config = self.make_config()

        def write_two_paths(root: Path) -> None:
            write(root / "src/c.py", "C = tentative\n")
            write(root / "src/future.py", "future = tentative\n")

        first = ReportingLuna(
            {
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (1, "S02"): writer("src/b.py", "B = 2\n"),
                (1, "S03"): write_two_paths,
            },
            {"S03"},
        )
        initial, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=first,
        )
        failed = self.run_approved(config, initial, "dirty-outside", FOUR_STEPS)
        worktree = Path(failed.state["worktree"])
        c_before = (worktree / "src/c.py").read_text()
        future_before = (worktree / "src/future.py").read_text()
        self.assertTrue(resume_info(failed.run_dir, failed.state).resumable)

        retry = ReportingLuna({}, set())
        second, *_rest = self.orchestrator(config, reviews=[PASS], luna=retry)
        resumed = second.resume("dirty-outside")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(resumed.state["failure"]["reason"], "RESUME_REQUIRES_OPERATOR")
        self.assertEqual(retry.calls, [])
        self.assertEqual((worktree / "src/c.py").read_text(), c_before)
        self.assertEqual((worktree / "src/future.py").read_text(), future_before)
        self.assertFalse((failed.run_dir / "steps/S03/mismatch_recovery.json").exists())

    def test_the_retry_cannot_write_a_future_step_path(self) -> None:
        config = self.make_config()
        # A retry that ignores the addendum and edits S04's path is stopped by
        # Git, not by the prompt.
        luna = self.migration_luna(
            behaviors={(1, "S03", 2): writer("src/future.py", "shim = True\n")},
            mismatch_steps={"S03#1"},
        )
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "retry-violation", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "STEP_WRITE_SET_VIOLATION")
        self.assertIn("src/future.py", result.state["failure"]["detail"])
        self.assertEqual(git(self.worktree("retry-violation"), "rev-parse", "HEAD"), self.base_sha)

    def test_an_ignored_build_artifact_never_makes_a_mismatch_dirty(self) -> None:
        write(self.repo / ".gitignore", "*.pyc\n__pycache__/\n")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "ignore build artifacts")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        config = self.make_config()

        def leave_ignored_artifact(root: Path) -> None:
            write(root / "src/__pycache__/a.cpython-311.pyc", "compiled\n")

        luna = self.migration_luna(
            behaviors={(1, "S03", 1): leave_ignored_artifact},
            reports={"S03#2": DEFERRED_REPORT},
        )
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "retry-ignored", FOUR_STEPS)

        # An ignored file is not a worker modification: the mismatch stays
        # clean and the bounded retry still runs.
        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 1, 1, 2, 1])
        step = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((step["status"], step["mismatch_retry_count"]), ("COMPLETED", 1))

    def test_an_untracked_file_makes_the_mismatch_dirty_and_terminal(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            behaviors={(1, "S03", 1): writer("src/leftover.py", "leftover = 1\n")},
        )
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "retry-untracked", FOUR_STEPS)

        self.assertEqual(result.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        # No retry at all: a dirty mismatch is terminal on the first attempt.
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 1, 1])
        self.assertEqual(pushed.call_count, 0)

    def test_accumulated_earlier_step_changes_never_make_a_mismatch_dirty(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(reports={"S03#2": DEFERRED_REPORT})
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "retry-accumulated", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        # S03's clean mismatch happened on a tree already carrying S01 and S02.
        s02 = json.loads((result.run_dir / "steps/S02/step.json").read_text())
        archived = json.loads((result.run_dir / "steps/S03/attempts/01/step.json").read_text())
        self.assertEqual(archived["tree_before"], s02["tree_after"])
        self.assertEqual(archived["tree_after"], s02["tree_after"])


class PersistedRecoveryTests(MismatchRetryHarness):
    def historical_run(self, run_id: str) -> Any:
        """Persist a run that failed on S03 with a clean mismatch, old-style."""

        config = self.make_config()
        luna = ReportingLuna(
            {
                (1, "S01"): writer("src/a.py", "A = 2\n"),
                (1, "S02"): writer("src/b.py", "B = 2\n"),
            },
            {"S03"},
        )
        orchestrator = LegacyMismatchOrchestrator(
            config,
            planner_client=QueueClient("planner", [FOUR_STEP_PLAN], self.events),
            reviewer_client=QueueClient("reviewer", [], self.events),
            agent=luna,
            reviser=FakeClaude(log=self.events),
        )
        result = self.run_approved(config, orchestrator, run_id, FOUR_STEPS)
        self.assertEqual(result.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        path = result.run_dir / "steps/S03/step.json"
        record = json.loads(path.read_text())
        self.assertEqual((record["status"], record["tree_before"]),
                         ("FAILED", record["tree_after"]))
        # The real run predates the mismatch_clean field entirely.
        record.pop("mismatch_clean", None)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return config, result, luna

    def test_resume_retries_only_the_failed_step_and_then_continues(self) -> None:
        config, failed, first = self.historical_run("legacy-s03")
        info = resume_info(failed.run_dir, failed.state)
        self.assertTrue(info.resumable, info.reason)
        self.assertEqual((info.phase, info.step_id), ("initial_step", "S03"))
        self.assertEqual(info.label, "RETRY S03 AFTER CLEAN MISMATCH")

        retry = ReportingLuna(
            {
                (1, "S03"): writer("src/c.py", "C = 3\n"),
                (1, "S04"): writer("src/future.py", "from src.c import present\n"),
            },
            set(),
            reports={"S03#1": DEFERRED_REPORT},
        )
        second, planner, _reviewer, _l, claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("legacy-s03")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # S01..S02 are never replayed and S03 runs exactly once more.
        self.assertEqual([call["step"] for call in first.calls], ["S01", "S02", "S03"])
        self.assertEqual([call["step"] for call in retry.calls], ["S03", "S04"])
        self.assertIn("<MISMATCH RETRY ADDENDUM>", retry.calls[0]["retry_addendum"])
        self.assertIn(MISMATCH_TEXT, retry.calls[0]["retry_addendum"])
        self.assertIn("S04:\n  src/future.py", retry.calls[0]["retry_addendum"])
        self.assertIsNone(retry.calls[1]["retry_addendum"])
        # No replanning, and the earlier step records are reused as they are.
        self.assertEqual(planner.prompts, [])
        for step_id, path in (("S01", "src/a.py"), ("S02", "src/b.py")):
            record = json.loads((resumed.run_dir / f"steps/{step_id}/step.json").read_text())
            self.assertEqual((record["status"], record["changed_paths"]),
                             ("COMPLETED", [path]))
        step = json.loads((resumed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(step["status"], "COMPLETED")
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertEqual(step["changed_paths"], ["src/c.py"])
        self.assertIn("src/future.py", step["deferred_verify"])
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_VERIFY_DEPENDENCY", claude.calls[0]["prompt"])
        self.assertEqual(pushed.call_count, 1)

    def test_resume_defers_when_the_retry_mismatches_again(self) -> None:
        config, failed, _first = self.historical_run("legacy-deferred")
        retry = ReportingLuna(
            {(1, "S04"): writer("src/future.py", "from src.c import present\n")},
            {"S03"},
        )
        second, _planner, reviewer, _l, claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
            claude=FakeClaude({1: writer("src/c.py", "C = 3\n")}),
        )
        with self.count_pushes():
            resumed = second.resume("legacy-deferred")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        self.assertEqual([call["step"] for call in retry.calls], ["S03", "S04"])
        step = json.loads((resumed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(step["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertEqual(step["mismatch_retry_count"], 1)
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])
        self.assertIn("S03", reviewer.prompts[0])

    def test_a_dirty_resume_retry_stops_without_a_third_attempt(self) -> None:
        config, failed, _first = self.historical_run("legacy-dirty")
        retry = ReportingLuna(
            {(1, "S03"): writer("src/c.py", "C = 3\n")}, {"S03"},
        )
        second, _planner, reviewer, _l, claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("legacy-dirty")

        self.assertEqual(resumed.status, RunStatus.FAILED)
        self.assertEqual(resumed.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        self.assertEqual([call["step"] for call in retry.calls], ["S03"])
        self.assertEqual((claude.calls, reviewer.prompts, pushed.call_count), ([], [], 0))
        self.assertTrue(resume_info(resumed.run_dir, resumed.state).resumable)

    def test_a_spent_retry_budget_is_deferred_instead_of_retried_again(self) -> None:
        config, failed, _first = self.historical_run("legacy-spent")
        # The same clean mismatch, but this step already spent its retry.
        path = failed.run_dir / "steps/S03/step.json"
        record = json.loads(path.read_text())
        record["mismatch_retry_count"] = 1
        record["initial_mismatch"] = "the first attempt mismatch"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        self.assertEqual(
            resume_info(failed.run_dir, failed.state).label,
            "CONTINUE AFTER CLEAN MISMATCH",
        )

        retry = ReportingLuna(
            {(1, "S04"): writer("src/future.py", "from src.c import present\n")}, set(),
        )
        second, _planner, _reviewer, _l, claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
            claude=FakeClaude({1: writer("src/c.py", "C = 3\n")}),
        )
        with self.count_pushes():
            resumed = second.resume("legacy-spent")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # No third attempt of S03: it is deferred and the chain continues.
        self.assertEqual([call["step"] for call in retry.calls], ["S04"])
        step = json.loads((resumed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(step["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertEqual(step["initial_mismatch"], "the first attempt mismatch")
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])


class TransientRetryPersistenceTests(MismatchRetryHarness):
    """A transport failure of the bounded retry keeps the retry mode.

    A timeout, an ``exit != 0`` or an auth failure of the *retry* is a
    transport failure of that exact semantic operation.  Resuming it reruns
    the same contract with the same addendum: never a normal first attempt,
    and never a second semantic mismatch retry.
    """

    def failed_retry(self, run_id: str, behavior: str, reason: str) -> Any:
        config = self.make_config()
        luna = self.migration_luna(behaviors={(1, "S03", 2): behavior})
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes() as pushed:
            failed = self.run_approved(config, orchestrator, run_id, FOUR_STEPS)

        self.assertEqual(failed.state["failure"]["reason"], reason)
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 1, 1, 2])
        self.assertEqual(pushed.call_count, 0)
        # The durable record of the failed attempt still proves the retry mode.
        record = json.loads((failed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((record["status"], record["reason"]), ("FAILED", reason))
        self.assertEqual(record["mismatch_retry_count"], 1)
        self.assertIn(MISMATCH_TEXT, record["initial_mismatch"])
        # Attempt 1's own deferred artifacts are kept apart.
        archived = json.loads(
            (failed.run_dir / "steps/S03/attempts/01/step.json").read_text()
        )
        self.assertEqual(archived["status"], "DEFERRED_CONTRACT_MISMATCH")
        self.assertTrue(resume_info(failed.run_dir, failed.state).resumable)
        return config, failed, luna

    def resume_the_same_retry(self, config: Any, run_id: str) -> Any:
        retry = ReportingLuna(
            {
                (1, "S03"): writer("src/c.py", "C = 3\n"),
                (1, "S04"): writer("src/future.py", "from src.c import present\n"),
            },
            set(),
            reports={"S03#1": DEFERRED_REPORT},
        )
        second, planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume(run_id)

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # Exactly one further S03 invocation, and it IS the same retry.
        self.assertEqual([call["step"] for call in retry.calls], ["S03", "S04"])
        addendum = retry.calls[0]["retry_addendum"]
        self.assertIn("<MISMATCH RETRY ADDENDUM>", addendum)
        self.assertIn(MISMATCH_TEXT, addendum)
        self.assertIn("S04:\n  src/future.py", addendum)
        self.assertIsNone(retry.calls[1]["retry_addendum"])
        self.assertEqual(planner.prompts, [])
        step = json.loads((resumed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((step["status"], step["mismatch_retry_count"]), ("COMPLETED", 1))
        self.assertEqual(step["changed_paths"], ["src/c.py"])
        self.assertEqual(pushed.call_count, 1)
        return resumed, retry

    def test_a_timed_out_retry_is_resumed_as_the_same_retry(self) -> None:
        config, _failed, _first = self.failed_retry(
            "retry-timeout", "timeout", "AGENT_TIMEOUT"
        )
        resumed, _retry = self.resume_the_same_retry(config, "retry-timeout")
        # Three durable Luna invocations of S03: mismatch, timeout, retry.
        self.assertEqual(
            [
                json.loads((path / "step.json").read_text())["status"]
                for path in sorted((resumed.run_dir / "steps/S03/attempts").iterdir())
            ],
            ["DEFERRED_CONTRACT_MISMATCH", "FAILED"],
        )

    def test_a_failed_retry_is_resumed_as_the_same_retry(self) -> None:
        config, _failed, _first = self.failed_retry(
            "retry-exit1", "exit1", "AGENT_FAILED"
        )
        self.resume_the_same_retry(config, "retry-exit1")

    def test_an_auth_failure_of_the_retry_keeps_the_retry_mode(self) -> None:
        config, _failed, _first = self.failed_retry(
            "retry-auth", "auth", "CODEX_AUTH_FAILURE"
        )
        self.resume_the_same_retry(config, "retry-auth")


class BoundaryDriftTests(MismatchRetryHarness):
    def test_boundary_drift_before_the_retry_fails_closed(self) -> None:
        config = self.make_config()
        luna = self.migration_luna()
        reviewer = QueueClient("reviewer", [PASS], self.events)
        claude = FakeClaude(log=self.events)
        orchestrator = DriftingOrchestrator(
            config,
            planner_client=QueueClient("planner", [FOUR_STEP_PLAN], self.events),
            reviewer_client=reviewer, agent=luna, reviser=claude,
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "retry-drift", FOUR_STEPS)

        # Drift is lost authority, not a deferrable mismatch.
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "STEP_CONTRACT_DRIFT")
        self.assertIn("pre-step tree", result.state["failure"]["detail"])
        # No second worker for S03 and no later step at all.
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02", "S03"])
        self.assertEqual((claude.calls, reviewer.prompts, pushed.call_count), ([], [], 0))
        # S04 only ever has its approved contract: it never ran.
        self.assertFalse((result.run_dir / "steps/S04/step.json").exists())
        self.assertFalse((result.run_dir / "steps/S04/agent.final.md").exists())
        # The step's current durable record is the failure, never a deferral.
        record = json.loads((result.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual((record["status"], record["reason"]), ("FAILED", "STEP_CONTRACT_DRIFT"))
        self.assertEqual(
            json.loads((result.run_dir / "steps/S03/attempts/01/step.json").read_text())["status"],
            "DEFERRED_CONTRACT_MISMATCH",
        )
        # A drifted boundary is never resumed automatically.
        self.assertFalse(resume_info(result.run_dir, result.state).resumable)


class AttemptUsageTests(MismatchRetryHarness):
    def test_every_luna_attempt_is_counted_exactly_once(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(reports={"S03#2": DEFERRED_REPORT})
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.count_pushes():
            result = self.run_approved(config, orchestrator, "retry-usage", FOUR_STEPS)

        self.assertEqual(result.status, RunStatus.PUBLISHED, result.state.get("failure"))
        attempt1 = json.loads(
            (result.run_dir / "steps/S03/attempts/01/step.json").read_text()
        )["usage"]
        attempt2 = json.loads((result.run_dir / "steps/S03/step.json").read_text())["usage"]
        summary = phase_usage_summary(result.run_dir)
        row = next(item for item in summary["implementer"]["steps"] if item["id"] == "S03")
        # The logical S03 row is the aggregate of both of its invocations.
        self.assertEqual(row["attempts"], 2)
        self.assertEqual(row["usage"], add_usage((attempt1, attempt2)))
        self.assertEqual(
            row["usage"]["input_tokens"],
            attempt1["input_tokens"] + attempt2["input_tokens"],
        )
        # The grand total counts the retry too, and counts it only once.
        every_record = [
            json.loads(path.read_text())["usage"]
            for path in sorted((result.run_dir / "steps").rglob("step.json"))
        ]
        self.assertEqual(len(every_record), 5)
        self.assertEqual(
            summary["implementer"]["total"], add_usage(every_record)
        )
        self.assertEqual(
            summary["grand_total"]["total_tokens"]
            - add_usage(every_record)["total_tokens"],
            add_usage((
                summary["planner"], summary["reviser"], summary["reviewer"],
            ))["total_tokens"],
        )


class DurableStepCrashTests(MismatchRetryHarness):
    """A durable step record is reconciled, never executed a second time."""

    def crash_before_the_next_checkpoint(self, step_id: str) -> Any:
        real = Orchestrator._checkpoint
        crashed: list[str] = []

        def checkpoint(self_: Any, run_dir: Any, phase: Any, **kwargs: Any) -> Any:
            if not crashed and kwargs.get("step_id") == step_id:
                crashed.append(step_id)
                raise RuntimeError("simulated crash before the checkpoint advanced")
            return real(self_, run_dir, phase, **kwargs)

        return mock.patch.object(Orchestrator, "_checkpoint", checkpoint)

    def test_a_deferred_step_is_not_retried_again_after_a_crash(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(
            behaviors={(1, "S03", 2): "nochange"}, mismatch_steps={"S03"},
        )
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.crash_before_the_next_checkpoint("S04"), self.count_pushes():
            failed = self.run_approved(config, orchestrator, "crash-deferred", FOUR_STEPS)

        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        # S03 spent its retry and its final deferral is durable...
        self.assertEqual([call["attempt"] for call in luna.calls], [1, 1, 1, 2])
        record = json.loads((failed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(
            (record["status"], record["mismatch_retry_count"]),
            ("DEFERRED_CONTRACT_MISMATCH", 1),
        )
        # ...but the checkpoint still names S03.
        self.assertEqual(read_checkpoint(failed.run_dir).step_id, "S03")

        retry = ReportingLuna(
            {(1, "S04"): writer("src/future.py", "from src.c import present\n")}, set(),
        )
        second, planner, _reviewer, _l, claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
            claude=FakeClaude({1: writer("src/c.py", "C = 3\n")}),
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("crash-deferred")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # The resume starts at S04: never a third attempt of S03.
        self.assertEqual([call["step"] for call in retry.calls], ["S04"])
        self.assertEqual(planner.prompts, [])
        step = json.loads((resumed.run_dir / "steps/S03/step.json").read_text())
        self.assertEqual(
            (step["status"], step["mismatch_retry_count"]),
            ("DEFERRED_CONTRACT_MISMATCH", 1),
        )
        self.assertIn("<SPEC>", claude.calls[0]["prompt"])
        self.assertNotIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])
        self.assertEqual(pushed.call_count, 1)

    def test_a_completed_c02_step_is_not_replayed_after_a_crash(self) -> None:
        config = self.make_config()
        luna = FakeLuna({
            (1, "S01"): writer("src/a.py", "A = 2\n"),
            (2, "S01"): writer("src/a.py", "A = 3\n"),
            (2, "S02"): writer("src/a.py", "A = 4\n"),
        })
        orchestrator, *_rest = self.orchestrator(
            config, plans=[SINGLE_PLAN, REPAIR_TWO_STEP_PLAN],
            reviews=[REVISE_IMPLEMENTATION, PASS], luna=luna,
        )
        with self.crash_before_the_next_checkpoint("S02"), self.count_pushes():
            failed = self.run_approved(config, orchestrator, "crash-c02", ("S01",))

        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual([(call["cycle"], call["step"]) for call in luna.calls],
                         [(1, "S01"), (2, "S01")])
        checkpoint = read_checkpoint(failed.run_dir)
        self.assertEqual((checkpoint.phase, checkpoint.step_id),
                         (ResumePhase.REPAIR_STEP, "S01"))
        self.assertEqual(
            json.loads(
                (failed.run_dir / "repair/C02/steps/S01/step.json").read_text()
            )["status"],
            "COMPLETED",
        )

        second_luna = FakeLuna({(2, "S02"): writer("src/a.py", "A = 4\n")})
        second, planner, *_rest = self.orchestrator(
            config, reviews=[PASS], luna=second_luna,
        )
        with self.count_pushes():
            resumed = second.resume("crash-c02")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # The C02 S01 worker is never called again; the resume starts at S02.
        self.assertEqual([(call["cycle"], call["step"]) for call in luna.calls],
                         [(1, "S01"), (2, "S01")])
        self.assertEqual([(call["cycle"], call["step"]) for call in second_luna.calls],
                         [(2, "S02")])
        self.assertEqual(planner.prompts, [])

    def test_a_completed_step_is_not_replayed_after_a_crash(self) -> None:
        config = self.make_config()
        luna = self.migration_luna(reports={"S03#2": DEFERRED_REPORT})
        orchestrator, *_rest = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
        )
        with self.crash_before_the_next_checkpoint("S03"), self.count_pushes():
            failed = self.run_approved(config, orchestrator, "crash-completed", FOUR_STEPS)

        self.assertEqual(failed.state["failure"]["reason"], "RUNTIMEERROR")
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02"])
        self.assertEqual(read_checkpoint(failed.run_dir).step_id, "S02")
        self.assertEqual(
            json.loads((failed.run_dir / "steps/S02/step.json").read_text())["status"],
            "COMPLETED",
        )

        retry = self.migration_luna(reports={"S03#2": DEFERRED_REPORT})
        second, _planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("crash-completed")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # S02's worker is never called again; the run restarts at S03.
        self.assertEqual([call["step"] for call in luna.calls], ["S01", "S02"])
        self.assertEqual([call["step"] for call in retry.calls], ["S03", "S03", "S04"])
        self.assertEqual(pushed.call_count, 1)


class HistoricalNineStepRecoveryTests(MismatchRetryHarness):
    def test_only_s09_is_retried_and_s10_follows(self) -> None:
        config = self.make_config()
        writers = {
            (1, step_id): writer(f"src/m{number:02d}.py", f"M = {number}\n")
            for number, step_id in enumerate(TEN_STEPS, start=1)
        }
        first = ReportingLuna(
            {key: action for key, action in writers.items() if key[1] != "S09"},
            {"S09"},
        )
        failed = self.run_approved(
            config,
            LegacyMismatchOrchestrator(
                config,
                planner_client=QueueClient("planner", [TEN_STEP_PLAN], self.events),
                reviewer_client=QueueClient("reviewer", [], self.events),
                agent=first,
                reviser=FakeClaude(log=self.events),
            ),
            "legacy-s09",
            TEN_STEPS,
        )
        self.assertEqual(failed.state["failure"]["reason"], "AGENT_CONTRACT_MISMATCH")
        self.assertEqual([call["step"] for call in first.calls], list(TEN_STEPS[:9]))
        # The real historical run predates the mismatch_clean field entirely.
        path = failed.run_dir / "steps/S09/step.json"
        record = json.loads(path.read_text())
        record.pop("mismatch_clean", None)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        trees = {
            step_id: json.loads(
                (failed.run_dir / f"steps/{step_id}/step.json").read_text()
            )["tree_after"]
            for step_id in TEN_STEPS[:8]
        }
        info = resume_info(failed.run_dir, failed.state)
        self.assertEqual((info.phase, info.step_id), ("initial_step", "S09"))
        self.assertEqual(info.label, "RETRY S09 AFTER CLEAN MISMATCH")

        retry = ReportingLuna(
            {key: action for key, action in writers.items() if key[1] in {"S09", "S10"}},
            set(),
        )
        second, planner, _reviewer, _l, _claude = self.orchestrator(
            config, reviews=[PASS], luna=retry,
        )
        with self.count_pushes() as pushed:
            resumed = second.resume("legacy-s09")

        self.assertEqual(resumed.status, RunStatus.PUBLISHED, resumed.state.get("failure"))
        # S01..S08 are never replayed; S09 runs exactly once more, then S10.
        self.assertEqual([call["step"] for call in retry.calls], ["S09", "S10"])
        self.assertIn("<MISMATCH RETRY ADDENDUM>", retry.calls[0]["retry_addendum"])
        self.assertIn("S10:\n  src/m10.py", retry.calls[0]["retry_addendum"])
        self.assertIsNone(retry.calls[1]["retry_addendum"])
        self.assertEqual(planner.prompts, [])
        for step_id, tree in trees.items():
            record = json.loads(
                (resumed.run_dir / f"steps/{step_id}/step.json").read_text()
            )
            self.assertEqual((record["status"], record["tree_after"]), ("COMPLETED", tree))
        s09 = json.loads((resumed.run_dir / "steps/S09/step.json").read_text())
        self.assertEqual((s09["status"], s09["mismatch_retry_count"]), ("COMPLETED", 1))
        self.assertEqual(s09["changed_paths"], ["src/m09.py"])
        self.assertEqual(pushed.call_count, 1)


class SafetyTests(MismatchRetryHarness):
    def test_a_deferred_verification_is_never_silently_accepted(self) -> None:
        """A red configured check is terminal even with a deferred dependency."""

        config = self.make_config()
        luna = self.migration_luna(
            behaviors={(1, "S03", 2): writer("src/c.py", "C = 3\n")},
            reports={"S03#2": DEFERRED_REPORT},
        )
        # Claude leaves the gate red: BUG in src/a.py fails the configured check.
        orchestrator, _planner, reviewer, _l, claude = self.orchestrator(
            config, plans=[FOUR_STEP_PLAN], reviews=[PASS], luna=luna,
            claude=FakeClaude({1: writer("src/a.py", "A = 2  # BUG\n")}),
        )
        with self.count_pushes() as pushed:
            result = self.run_approved(config, orchestrator, "deferred-red", FOUR_STEPS)

        # The initial revision leaves the gate red, so P3 spends its bounded
        # automatic check-repair budget on it -- the first pass, then the one
        # second bounded pass inside the same mutable scope.  The check stays
        # red and the run is terminal.  No reviewer, no candidate, no push.
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(
            [(call["cycle"], call["stage"]) for call in claude.calls],
            [(1, "initial-revision"), (1, "check-repair"), (1, "check-repair")],
        )
        self.assertEqual((reviewer.prompts, pushed.call_count), ([], 0))
        self.assertFalse((result.run_dir / "candidate/C01/commit.json").exists())
        self.assertFalse((result.run_dir / "candidate/C02").exists())
        self.assertFalse((result.run_dir / "review").exists())
        self.assertIsNone(result.state.get("commit_sha"))


if __name__ == "__main__":
    unittest.main()
