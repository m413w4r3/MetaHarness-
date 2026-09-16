"""P49: one bounded retry of a clean META CONTRACT MISMATCH.

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
from metaharness.resume import resume_info  # noqa: E402

from tests.test_p28_full_pipeline import (  # noqa: E402
    PASS,
    CleanMismatchLuna,
    FakeClaude,
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

# S01 modifies A, S02 modifies B, S03 creates its own file, S04 owns future.py.
FOUR_STEP_PLAN = plan_text(
    step_block(1),
    step_block(2, read=("src/a.py", "src/b.py"), write_set=("src/b.py",)),
    step_block(3, read=("src/a.py",), write_set=(), create=("src/c.py",)),
    step_block(4, read=("src/future.py",), write_set=("src/future.py",)),
    title="P49 staged migration",
)
FOUR_STEPS = ("S01", "S02", "S03", "S04")


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

        # Claude and the reviewer both receive the deferred verify dependency.
        self.assertEqual(len(claude.calls), 1)
        prompt = claude.calls[0]["prompt"]
        self.assertIn("DEFERRED_VERIFY_DEPENDENCY", prompt)
        self.assertIn("src/future.py", prompt)
        self.assertIn("pytest tests/test_future.py", prompt)
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
        self.assertIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])
        self.assertIn("DEFERRED CONTRACT MISMATCHES", reviewer.prompts[0])
        self.assertIn("S03", reviewer.prompts[0])

    def test_a_dirty_retry_is_terminal(self) -> None:
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
        # A dirty retry is not resumable: the operator owns it.
        self.assertFalse(resume_info(result.run_dir, result.state).resumable)

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
        self.assertIn("DEFERRED_VERIFY_DEPENDENCY", claude.calls[0]["prompt"])
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
        self.assertIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])
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
        self.assertFalse(resume_info(resumed.run_dir, resumed.state).resumable)

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
        self.assertIn("DEFERRED_CONTRACT_MISMATCH", claude.calls[0]["prompt"])


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

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.state["failure"]["reason"], "DETERMINISTIC_GATE_FAILED")
        self.assertEqual(len(claude.calls), 1)
        self.assertEqual((reviewer.prompts, pushed.call_count), ([], 0))
        self.assertFalse((result.run_dir / "candidate/C01/commit.json").exists())


if __name__ == "__main__":
    unittest.main()
