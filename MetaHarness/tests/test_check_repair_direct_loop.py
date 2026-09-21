import unittest
import json
from pathlib import Path

from metaharness.evidence import EvidenceBundle
from metaharness.orchestration.check_repair import (
    CheckRepairAttempt,
    CheckRepairResult,
    _check_repair_prompt,
    _hard_failure_items,
    _soft_check_failures,
)
from metaharness.resume import (
    ResumeCheckpoint,
    ResumePhase,
    checkpoint_payload,
)
from metaharness.approval import PlanIdentity

from tests.test_p28_full_pipeline import (
    PASS,
    FakeClaude,
    FakeLuna,
    P28Harness,
    writer,
    write,
    git,
)
from metaharness.run_options import RunOptions


class DirectCheckRepairTests(unittest.TestCase):
    def evidence(self, failures: tuple[str, ...]) -> EvidenceBundle:
        return EvidenceBundle(
            base_sha="a" * 40,
            staged_tree_sha="b" * 40,
            changed_files=("src/service.py",),
            diff="diff --git a/src/service.py b/src/service.py\n",
            checks=(),
            deterministic_passed=False,
            failures=failures,
            required_check_ids=("unit",),
        )

    def test_structured_result_keeps_generic_attempt_numbers(self) -> None:
        result = CheckRepairResult(
            "passed",
            (
                CheckRepairAttempt(
                    number=1,
                    failed_check_ids_before=("unit",),
                    tree_before="a" * 40,
                    tree_after="b" * 40,
                    mutable_scope=("src/service.py",),
                ),
            ),
        )
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.attempts[0].number, 1)

    def test_hard_failures_are_not_soft_repair_evidence(self) -> None:
        hard = self.evidence(("CHECK_FAILED:unit", "INTEGRITY_MISMATCH:tree"))
        self.assertEqual(_hard_failure_items(hard.failures), ["INTEGRITY_MISMATCH:tree"])
        self.assertEqual(_soft_check_failures(hard), [])

        soft = self.evidence(("CHECK_FAILED:unit",))
        self.assertEqual(_soft_check_failures(soft), ["CHECK_FAILED:unit"])

    def test_new_prompt_renders_spec_and_only_failed_check_context(self) -> None:
        prompt = _check_repair_prompt(
            spec="SPEC: preserve the public API",
            plan=object(),
            approved_contract_index="INVARIANT: keep the requested behavior",
            changed_files="src/service.py",
            evidence=self.evidence(("CHECK_FAILED:unit",)),
            mutable_scope=["src/service.py"],
            previous_report="old worker transcript that must not be sent",
        )
        self.assertIn("SPEC: preserve the public API", prompt)
        self.assertIn("CHECK_FAILED:unit", prompt)
        self.assertIn("INVARIANT: keep the requested behavior", prompt)
        self.assertNotIn("old worker transcript", prompt)
        self.assertNotIn("<PLAN SUMMARY>", prompt)

    def test_attempt_ordinal_is_durable_in_the_checkpoint(self) -> None:
        identity = PlanIdentity(
            raw_sha256="c" * 64,
            contract_sha256="d" * 64,
            bundle_sha256="e" * 64,
            execution_sha256="f" * 64,
            checks_sha256="0" * 64,
        )
        checkpoint = ResumeCheckpoint(
            phase=ResumePhase.FINAL_CHECKS_RETRY_C01,
            cycle=1,
            step_id=None,
            expected_head_sha="1" * 40,
            expected_tree_sha="2" * 40,
            execution_selection_sha256="f" * 64,
            plan_identity=identity,
            check_repair_attempt=2,
        )
        self.assertEqual(checkpoint_payload(checkpoint)["check_repair_attempt"], 2)


class DirectCheckRepairPipelineTests(P28Harness):
    """The durable v2 path uses the generic attempt ledger."""

    def options(self, budget: int = 2, **overrides: object) -> RunOptions:
        return RunOptions.from_config(
            self.load(), check_repair_profile="claude",
            max_check_repair_attempts=budget, **overrides,
        )

    def repair_calls(self, claude: FakeClaude) -> list[dict[str, object]]:
        return [call for call in claude.calls if call["stage"] == "check-repair"]

    def test_initial_pass_makes_no_repair_call(self) -> None:
        claude = FakeClaude()
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.status.value, "published")
        self.assertEqual(self.repair_calls(claude), [])

    def test_first_attempt_passes_with_one_repair_call(self) -> None:
        claude = FakeClaude(stage_actions={
            (1, "check-repair"): writer("src/a.py", "A = 3\n"),
        })
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.status.value, "published")
        self.assertEqual(len(self.repair_calls(claude)), 1)
        self.assertTrue(
            (result.run_dir / "revision/check-repair/C01/attempts/01/attempt.json").is_file()
        )

    def test_second_attempt_is_used_after_a_red_first_attempt(self) -> None:
        calls = 0

        def repair(root: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                write(root / "src/a.py", "A = 3\n")

        claude = FakeClaude(stage_actions={(1, "check-repair"): repair})
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.status.value, "published")
        self.assertEqual(calls, 2)
        self.assertEqual(len(self.repair_calls(claude)), 2)

    def test_budget_zero_makes_no_repair_call(self) -> None:
        claude = FakeClaude()
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(0),
        )
        self.assertEqual(result.state["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(self.repair_calls(claude), [])

    def test_hard_integrity_failure_makes_no_repair_call(self) -> None:
        self.check.write_text(
            "import pathlib\n"
            "pathlib.Path('src/a.py').write_text('MUTATED\\n')\n",
            encoding="utf-8",
        )
        claude = FakeClaude()
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = 2\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.state["failure"]["reason"], "CHECK_MUTATED")
        self.assertEqual(self.repair_calls(claude), [])

    def test_agent_runtime_failure_is_not_a_check_failure(self) -> None:
        claude = FakeClaude(failures={(1, "check-repair"): "timeout"})
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.state["failure"]["reason"], "AGENT_RUNTIME_FAILED")
        self.assertNotEqual(result.state["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertEqual(len(self.repair_calls(claude)), 1)

    def test_scope_is_extended_between_attempts_without_spending_a_retry(self) -> None:
        write(self.repo / "tests/test_service.py", "def test_fake():  # stale\n    pass\n")
        git(self.repo, "add", "tests/test_service.py")
        git(self.repo, "commit", "-qm", "add fixture")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        self.check.write_text(
            "import pathlib, sys\n"
            "source = pathlib.Path('src/a.py').read_text()\n"
            "fixture = pathlib.Path('tests/test_service.py').read_text()\n"
            "if 'BUG' in source:\n"
            "    print('FAILED src/a.py')\n"
            "    sys.exit(1)\n"
            "if 'stale' in fixture:\n"
            "    print('FAILED tests/test_service.py::test_fake')\n"
            "    sys.exit(1)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        calls = 0

        def repair(root: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                write(root / "src/a.py", "A = STILL\n")
            else:
                write(root / "tests/test_service.py", "def test_fake():\n    pass\n")

        claude = FakeClaude(stage_actions={(1, "check-repair"): repair})
        result, _planner, _reviewer, _claude, _pushed = self.run_pipeline(
            luna=FakeLuna({(1, "S01"): writer("src/a.py", "A = BUG\n")}),
            reviews=[PASS], claude=claude, run_options=self.options(),
        )
        self.assertEqual(result.status.value, "published")
        self.assertEqual(calls, 2)
        attempts = [
            json.loads(path.read_text())
            for path in sorted(
                (result.run_dir / "revision/check-repair/C01/attempts").glob("*/attempt.json")
            )
        ]
        self.assertEqual(attempts[0]["mutable_scope"], ["src/a.py"])
        self.assertEqual(
            attempts[1]["mutable_scope"], ["src/a.py", "tests/test_service.py"]
        )


if __name__ == "__main__":
    unittest.main()
