"""Check-repair: ladder, budget, scope authority and rollback."""

from __future__ import annotations

import json
import re
import unittest

from metaharness.gitops import candidate_tree_sha
from metaharness.models import ExecutionRole, RunStatus
from metaharness.recovery_policy import RecoveryBudgets
from metaharness.run_options import RunOptions

from tests.pipeline.support import (
    SPEC,
    STEP,
    PipelineHarness,
    check_repair_result,
    git,
    initial_plan,
    ladder_ledger,
    ladder_strategies,
    review,
    write,
)


class CheckRepairTests(PipelineHarness):
    def _add_tracked_paths(self, *paths: str) -> None:
        for path in paths:
            target = self.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("base test\n", encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "add test fixtures")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")

    @staticmethod
    def _plan_with_approved_paths(*paths: str) -> str:
        plan_text = initial_plan(STEP)
        reads = "".join(f"- {path} :: current content\n" for path in paths)
        writes = "".join(f"- {path}\n" for path in paths)
        return plan_text.replace(
            "- feature.txt :: current content\n",
            "- feature.txt :: current content\n" + reads,
            1,
        ).replace("WRITE_SET\n- feature.txt\n", "WRITE_SET\n- feature.txt\n" + writes, 1)

    def test_failed_test_file_is_readable_but_not_auto_writable(self) -> None:
        self._add_tracked_paths("tests/test_feature.py")
        self.check.write_text(
            "import pathlib, sys\n"
            "if pathlib.Path('feature.txt').read_text().strip() != 'good':\n"
            "    print('feature.txt: required value; tests/test_feature.py: expected fixture update', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda request: (
                (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8"),
                (request.worktree / "tests/test_feature.py").write_text(
                    "repaired test\n", encoding="utf-8",
                ),
                check_repair_result(),
            )[-1],
        )
        config = self.config(check_repair=2)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        repair_request = self.workers.calls[-1]
        self.assertEqual(repair_request.mutable_paths, ("feature.txt",))
        self.assertIn("tests/test_feature.py", repair_request.prompt)
        self.assertIn("remain read-only", repair_request.prompt)
        self.assertFalse(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )

    def test_repair_scope_rejects_paths_over_the_auto_bound(self) -> None:
        self._add_tracked_paths("tests/test_feature.py", "tests/test_other.py")
        self.check.write_text(
            "import pathlib, sys\n"
            "if not (pathlib.Path('feature.txt').read_text().strip() == 'good'\n"
            "        and pathlib.Path('tests/test_feature.py').read_text().strip() == 'repaired'\n"
            "        and pathlib.Path('tests/test_other.py').read_text().strip() == 'repaired'):\n"
            "    print('tests/test_feature.py tests/test_other.py', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda request: (
                (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8"),
                (request.worktree / "tests/test_feature.py").write_text(
                    "repaired\n", encoding="utf-8",
                ),
                (request.worktree / "tests/test_other.py").write_text(
                    "repaired\n", encoding="utf-8",
                ),
                check_repair_result(),
            )[-1],
        )
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(self.state()["failure"]["reason"], "AGENT_SCOPE_VIOLATION")
        self.assertFalse(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").exists()
        )

    def test_noop_repair_attempt_is_existing_head_not_an_empty_repair_commit(self) -> None:
        counter = self.root / "flaky-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    print('tests/test_feature.py: flaky failure', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, lambda _request: check_repair_result())
        result = self.orchestrator(
            self.config(check_repair=1), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        accepted = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/accepted.json").read_text()
        )
        self.assertEqual(accepted["acceptance_kind"], "existing-head")
        self.assertFalse(accepted["commit_created"])
        self.assertEqual(git(self.worktree(), "rev-list", "--count", "HEAD"), "2")

    def test_ladder_expands_the_approved_scope_when_evidence_proves_it(self) -> None:
        related = "fixtures/related.txt"
        self._add_tracked_paths(related)
        self.check.write_text(
            "import pathlib, sys\n"
            "feature = pathlib.Path('feature.txt').read_text().strip()\n"
            f"related = pathlib.Path({related!r}).read_text().strip()\n"
            "if feature != 'good':\n"
            "    print('feature.txt: the primary file is not repaired', file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "if related != 'good':\n"
            f"    print({related + ': the related approved file is not repaired'!r}, file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )

        def expanded_repair(request):
            self.assertEqual(request.mutable_paths, ("feature.txt", related))
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            (request.worktree / related).write_text("good\n", encoding="utf-8")
            return check_repair_result()

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"), expanded_repair)
        config = self.config(check_repair=2)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[self._plan_with_approved_paths(related)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        # The first pass repaired the evidenced file only: the new evidence
        # then named the second approved file, so the ladder expanded the
        # scope exactly once instead of repeating the targeted repair.
        self.assertEqual(ladder_strategies(self), ["repair_targeted", "expand_scope"])
        self.assertEqual(self.state()["check_repair"]["attempt_count"], 2)
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/002"
        # The pending pass reads its authorization, then archives it with the
        # retired artifacts of the attempt it widened.
        expansions = sorted(attempt.rglob("scope-expansion.json"))
        self.assertEqual(len(expansions), 1)
        expansion = json.loads(expansions[0].read_text(encoding="utf-8"))
        self.assertEqual(expansion["added_paths"], [related])
        self.assertEqual(expansion["attempt"], 2)
        self.assertEqual(expansion["bound"], 1)
        scope = json.loads((attempt / "scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["added_paths"], [related])
        self.assertEqual(scope["effective_repair_scope"], ["feature.txt", related])
        self.assertEqual(scope["approved_mutable_scope"], ["feature.txt", related])

    def test_ladder_never_repeats_a_strategy_on_the_same_tree_and_failure(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.REPAIR, lambda _request: check_repair_result())
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "implementer"])
        entries = ladder_ledger(self)["entries"]
        self.assertEqual(
            [entry["strategy"] for entry in entries], ["repair_targeted", "replan_step"],
        )
        # The repair pass left the exact candidate tree and failure unchanged,
        # so the second budgeted pass was refused and the ladder moved on.
        self.assertEqual(self.state()["check_repair"]["attempt_count"], 1)
        identities = [
            (entry["strategy"], entry["tree"], tuple(entry["failed_check_ids"]))
            for entry in entries
        ]
        self.assertEqual(len(set(identities)), len(identities))
        self.assertEqual(entries[0]["tree"], entries[1]["tree"])

    def test_replanned_step_opens_new_facts_for_the_ladder(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "nearly good\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            lambda _request: check_repair_result(), write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(
            self.workers.roles(), ["implementer", "repair", "implementer", "repair"],
        )
        entries = ladder_ledger(self)["entries"]
        self.assertEqual(
            [entry["strategy"] for entry in entries],
            ["repair_targeted", "replan_step", "repair_targeted"],
        )
        # The replan produced a new candidate tree, which is new facts: the
        # targeted repair is available again for it, the replan rung is not.
        replanned, replayed = entries[1], entries[2]
        self.assertNotEqual(replanned["tree"], replanned["tree_after"])
        self.assertEqual(replayed["tree"], replanned["tree_after"])
        self.assertNotEqual(replayed["tree"], entries[0]["tree"])
        self.assertEqual(replayed["repair_attempt"], 2)
        self.assertEqual(self.state()["check_repair"]["attempt_count"], 2)

    def test_every_attempt_of_the_budget_is_used_then_the_gate_is_exhausted(self) -> None:
        def blocked_after_mutation(request):
            (request.worktree / "feature.txt").write_text("half repair\n", encoding="utf-8")
            return check_repair_result(
                "BLOCKED", "NOT_RUN", "INFRASTRUCTURE", "Docker socket access denied",
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(
            ExecutionRole.REPAIR,
            blocked_after_mutation,
            write("feature.txt", "good\n"),
        )
        config = self.config(check_repair=2)
        result = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_EXTERNAL, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_UNAVAILABLE")
        self.assertEqual(self.state()["check_repair"]["attempt_count"], 0)
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertFalse((attempt / "attempt.json").exists())
        self.assertEqual((self.worktree() / "feature.txt").read_text(encoding="utf-8"), "bad\n")
        evidence = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/evidence.json").read_text()
        )
        self.assertEqual(candidate_tree_sha(self.worktree()), evidence["staged_tree_sha"])
        report = json.loads((attempt / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["check_repair_result"]["blocked_kind"], "INFRASTRUCTURE")
        self.assertNotEqual(report["tree_before"], report["tree_after"])
        self.assertEqual(
            (self.checkpoint()["phase"], self.checkpoint()["check_repair_attempt"]),
            ("check_repair", 1),
        )

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertTrue((attempt / "attempts/01/report.json").is_file())

    def test_an_unreviewable_candidate_closes_the_gate_without_repair_or_review(self) -> None:
        def binary_candidate(request):
            (request.worktree / "feature.txt").write_bytes(b"\x00\x01\xff not reviewable\n")
            return "done\n"

        self.workers.on(ExecutionRole.IMPLEMENTER, binary_candidate)
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.FAILED, self.state().get("failure"))
        failure = self.state()["failure"]
        self.assertEqual(failure["reason"], "COMMIT_GATE_FAILED", failure)
        self.assertIn("UNREVIEWABLE_TEXT_DIFF:feature.txt", failure["detail"])
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertEqual(self.reviewer.requests, [])
        self.assertFalse((self.run_dir() / "cycles/001/check-repair").exists())

    def test_two_verified_check_repairs_can_exhaust_the_budget(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "worse still\n"),
        )
        self.workers.on(
            ExecutionRole.REPAIR,
            write("feature.txt", "still bad\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "first targeted repair ran and failed",
            )),
            write("feature.txt", "worse\n", report=check_repair_result(
                "DONE", "FAIL", "NONE", "second targeted repair ran and failed",
            )),
        )
        result = self.orchestrator(
            self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        self.assertIsInstance(self.state()["failure"]["detail"], dict)
        self.assertEqual(self.state()["failure"]["detail"]["failed_check_ids"], ["test"])
        self.assertEqual(self.state()["failure"]["detail"]["attempt_count"], 2)
        self.assertEqual(
            self.state()["failure"]["detail"]["strategy"],
            "repair_targeted+repair_targeted+replan_step",
        )
        self.assertEqual(
            self.state()["failure"]["detail"]["strategies"],
            ["repair_targeted", "repair_targeted", "replan_step"],
        )
        self.assertEqual(self.state()["check_repair"]["failure_classification"], "product_check")
        self.assertEqual(self.state()["check_repair"]["next_action"], "Retry deterministic gate")
        self.assertRegex(self.state()["check_repair"]["latest_evidence_sha256"], r"^[0-9a-f]{64}$")
        reports = self.state()["check_repair"]["repair_reports"]
        self.assertEqual([item["attempt"] for item in reports], [1, 2])
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in reports))
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["phase"], "deterministic_gate")
        self.assertEqual(checkpoint["stage"], "POST_IMPLEMENTATION")
        # The ladder rewrote the tree after the last recorded pass, so the
        # checkpoint attributes it to no pass at all.
        self.assertIsNone(checkpoint["check_repair_attempt"])
        # Two budgeted passes, then the one distinct replan of the responsible
        # approved step, before the exhausted ladder waits for the operator.
        self.assertEqual(
            self.workers.roles(), ["implementer", "repair", "repair", "implementer"],
        )
        self.assertEqual(
            ladder_strategies(self),
            ["repair_targeted", "repair_targeted", "replan_step"],
        )
        diagnostics_path = self.run_dir() / "diagnostics.md"
        self.assertTrue(
            diagnostics_path.is_file(),
            (self.run_dir() / "diagnostics.error.txt").read_text(encoding="utf-8")
            if (self.run_dir() / "diagnostics.error.txt").is_file() else "diagnostics missing",
        )
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        recovery_summary = diagnostics.split("## DETERMINISTIC GATE RECOVERY", 1)[1].split("\n## ", 1)[0]
        for expected in (
            # Two repaired red gates, the replan rung and their reruns.
            "deterministic gate attempt: 4",
            "check-repair attempts used / budget: 2 / 2",
            "latest failed check IDs: test",
            "failure classification: product_check",
            "next recovery action: Retry deterministic gate",
        ):
            self.assertIn(expected, recovery_summary, recovery_summary)
        attempts = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts"
        self.assertEqual(sorted(path.name for path in attempts.iterdir()), ["001", "002"])
        self.assertEqual(self.reviewer.requests, [])

    def test_blocked_scope_request_uses_authority_before_retrying_same_attempt(self) -> None:
        from metaharness.run_options import RunOptions

        self._add_tracked_paths("other.txt")
        self.check.write_text(
            "import pathlib, sys\n"
            "if (pathlib.Path('feature.txt').read_text().strip() != 'good'\n"
            "        or pathlib.Path('other.txt').read_text().strip() != 'good'):\n"
            "    print('feature.txt: required files are not repaired', file=sys.stderr)\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        scope_request = (
            "META SCOPE REQUEST v1\n\n"
            "REASON\nThe failing check requires the related fixture.\n\n"
            "PATHS\n- other.txt\n\n"
            "EVIDENCE\n- The configured test reads other.txt.\n\n"
            "END META SCOPE REQUEST"
        )

        def blocked(request):
            (request.worktree / "feature.txt").write_text("partial\n", encoding="utf-8")
            return scope_request + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "other.txt is outside the current authority",
            )

        def repair(request):
            self.assertIn("other.txt", request.mutable_paths)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            (request.worktree / "other.txt").write_text("good\n", encoding="utf-8")
            return check_repair_result()

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, blocked, repair)
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[self._plan_with_approved_paths("other.txt")], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertEqual(json.loads((attempt / "attempt.json").read_text())["number"], 1)
        authority = json.loads((attempt / "scope_requests/001/authority.json").read_text())
        self.assertEqual(authority["added_paths"], ["other.txt"])
        self.assertEqual(json.loads((attempt / "scope.json").read_text())["added_paths"], ["other.txt"])

    def test_failure_trace_narrows_repair_and_scope_requests_obey_the_addition_limit(self) -> None:
        from metaharness.run_options import RunOptions

        modules = [f"src/repair_{index:02}.py" for index in range(38)]
        approved = ["feature.txt", *modules, "tests/test_failure.py"]
        self.assertEqual(len(approved), 40)
        self._add_tracked_paths(*approved)
        groups = [approved[index:index + 5] for index in range(0, len(approved), 5)]
        plan_text = initial_plan(*(
            (f"S{index:02}", group[0], f"Update {group[0]}")
            for index, group in enumerate(groups, start=1)
        ))
        for index, group in enumerate(groups, start=1):
            primary = group[0]
            read_entries = "".join(f"- {path} :: current content\n" for path in group)
            write_entries = "".join(f"- {path}\n" for path in group)
            plan_text = plan_text.replace(
                f"READ_SET\n- {primary} :: current content\n",
                f"READ_SET\n{read_entries}", 1,
            ).replace(
                f"WRITE_SET\n- {primary}\n",
                f"WRITE_SET\n{write_entries}", 1,
            )
        counter = self.root / "synthetic-check-count"
        self.check.write_text(
            "import pathlib, sys\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    print('=== FAILURES ===')\n"
            "    print('________________ test_gate_failure ________________')\n"
            "    print('Traceback (most recent call last):')\n"
            "    print(f'  File {str(pathlib.Path(\"src/repair_00.py\").resolve())!r}, line 12, in run')\n"
            "    print(f'  File {str(pathlib.Path(\"src/repair_01.py\").resolve())!r}, line 27, in check')\n"
            "    print('AssertionError: synthetic regression')\n"
            "    print('E   AssertionError: synthetic regression')\n"
            "    sys.stdout.write('\\n' * 5000)\n"
            "    print('FAILED tests/test_failure.py::test_gate_failure - AssertionError')\n"
            "    raise SystemExit(1)\n",
            encoding="utf-8",
        )
        def scope_request_for(path: str) -> str:
            return (
                "META SCOPE REQUEST v1\n\n"
                "REASON\nThe failing assertion depends on this approved source file.\n\n"
                f"PATHS\n- {path}\n\n"
                "EVIDENCE\n- The traceback and check output identify the dependency.\n\n"
                "END META SCOPE REQUEST"
            )

        def request_third_path(request):
            self.assertEqual(request.mutable_paths, tuple(modules[:2]))
            self.assertIn("Traceback (most recent call last)", request.prompt)
            self.assertIn("src/repair_00.py", request.prompt)
            self.assertIn("src/repair_01.py", request.prompt)
            self.assertIn("tests/test_failure.py", request.prompt)
            self.assertIn("checks/test.stdout.log", request.prompt)
            return scope_request_for(modules[2]) + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "The related source path needs authority.",
            )

        def request_over_limit(request):
            self.assertEqual(request.mutable_paths, tuple(modules[:3]))
            return scope_request_for(modules[3]) + "\n\n" + check_repair_result(
                "BLOCKED", "NOT_RUN", "SCOPE", "A second path needs authority.",
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, *(
            write(group[0], "good\n") for group in groups
        ))
        self.workers.on(ExecutionRole.REPAIR, request_third_path, request_over_limit)
        config = self.config(check_repair=1)
        options = RunOptions.from_config(config, repair_scope_max_added_paths=1)
        result = self.orchestrator(
            config, planner=[plan_text], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)

        self.assertEqual(
            result.status, RunStatus.WAITING_SCOPE_APPROVAL,
            self.state().get("failure"),
        )
        self.assertEqual(self.workers.roles(), ["implementer"] * 8 + ["repair", "repair"])
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        checks = json.loads(
            (self.run_dir() / "cycles/001/checks/post-implementation/checks.json").read_text()
        )
        self.assertEqual(
            checks[0]["stdout_tail"].strip(),
            "FAILED tests/test_failure.py::test_gate_failure - AssertionError",
        )
        scope = json.loads((attempt / "scope.json").read_text(encoding="utf-8"))
        self.assertEqual(scope["approved_mutable_scope"], sorted(approved))
        self.assertEqual(scope["initial_repair_scope"], sorted(modules[:2]))
        self.assertEqual(scope["added_paths"], [modules[2]])
        self.assertEqual(scope["effective_repair_scope"], sorted(modules[:3]))
        attempts = sorted((attempt / "scope_requests").glob("*/authority.json"))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(json.loads(attempts[1].read_text())["bound"], 1)

    def test_check_repair_infrastructure_exhaustion_resumes_at_the_red_gate(self) -> None:
        from metaharness.agent import AgentRunResult
        from metaharness.gitops import candidate_tree_sha

        def timeout(request):
            tree = candidate_tree_sha(request.worktree)
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before=tree,
                tree_after=tree, usage=None, external_session_id=None,
                report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, timeout, write("feature.txt", "good\n"))
        config = self.config(check_repair=1)
        options = RunOptions.from_config(
            config,
            recovery=RecoveryBudgets(max_transient_attempts=0, max_executor_fallbacks=0),
        )
        failed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run", run_options=options)
        self.assertEqual(failed.status, RunStatus.WAITING_EXTERNAL)
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_UNAVAILABLE")
        self.assertEqual(
            (self.checkpoint()["phase"], self.checkpoint()["check_repair_attempt"]),
            ("check_repair", 1),
        )
        self.assertFalse((self.run_dir() / "cycles/001/candidate/commit.json").exists())

        resumed = self.orchestrator(
            config, planner=["unused"], reviewer=[review()],
        ).resume("run")
        self.assertEqual(resumed.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])

    def test_a_dirty_check_repair_timeout_rolls_back_and_retries_exact_contract(self) -> None:
        def timeout(request):  # the repair worker edits in scope, then times out
            (request.worktree / "feature.txt").write_text("half\n", encoding="utf-8")
            from metaharness.agent import AgentRunResult
            from metaharness.gitops import candidate_tree_sha
            return AgentRunResult(
                status="timed_out", exit_reason="AGENT_TIMEOUT", tree_before="",
                tree_after=candidate_tree_sha(request.worktree), usage=None,
                external_session_id=None, report_path=None, timed_out=True,
            )

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, timeout, write("feature.txt", "good\n"))
        config = self.config(check_repair=2)
        completed = self.orchestrator(
            config, planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(completed.status, RunStatus.COMMITTED, self.state().get("failure"))
        attempt = self.run_dir() / "cycles/001/check-repair/post-implementation/attempts/001"
        self.assertTrue((attempt / "attempt.json").is_file())
        self.assertTrue((attempt / "attempts/01/failure.json").is_file())
        self.assertFalse((attempt / "failure.json").exists())
        self.assertEqual(self.workers.roles(), ["implementer", "repair", "repair"])

    def test_zero_check_repair_budget_still_reaches_non_worker_recovery(self) -> None:
        """A zero budget refuses the worker rungs, never the autonomous ones."""

        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            write("feature.txt", "bad\n"), write("feature.txt", "good\n"),
        )
        result = self.orchestrator(
            self.config(check_repair=0), planner=[initial_plan(STEP)], reviewer=[review()],
        ).run_text(SPEC, run_id="run")

        self.assertEqual(result.status, RunStatus.COMMITTED, self.state().get("failure"))
        self.assertEqual(self.workers.roles(), ["implementer", "implementer"])
        entries = ladder_ledger(self)["entries"]
        self.assertEqual([entry["strategy"] for entry in entries], ["replan_step"])
        self.assertEqual([entry["repair_attempt"] for entry in entries], [None])
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/attempts").exists(),
            "an inapplicable repair rung created an attempt or a report",
        )

    def test_repair_budget_only_limits_worker_repair_rungs(self) -> None:
        """The frozen budget bounds the worker rungs and nothing else."""

        from types import SimpleNamespace

        from metaharness.evidence import EvidenceBundle
        from metaharness.models import GateStage
        from metaharness.orchestration.check_repair import CheckRepairLadder
        from metaharness.recovery_policy import RecoveryStrategy

        red = EvidenceBundle(
            base_sha=self.base_sha, staged_tree_sha=candidate_tree_sha(self.repo),
            changed_files=("feature.txt",), diff="", checks=(),
            deterministic_passed=False, failures=("CHECK_FAILED:test",),
            required_check_ids=("test",),
        )
        step = SimpleNamespace(
            id="S01", write_set=("feature.txt",), create_set=(), delete_set=(),
        )
        cycle_plan = SimpleNamespace(
            cycle=SimpleNamespace(number=1),
            plan=SimpleNamespace(steps=(step,)),
            mutable_scope=("feature.txt",),
        )
        ctx = SimpleNamespace(
            run_dir=self.run_dir(), repo=self.repo,
            info=SimpleNamespace(worktree=self.repo),
            selection=SimpleNamespace(check_repair_fallbacks=()),
            options=SimpleNamespace(
                repair_scope_policy="deny-expansion", repair_scope_max_added_paths=4,
            ),
        )
        ladder = CheckRepairLadder(replay_steps=lambda *args, **kwargs: "b" * 40)

        def gate(attempt: int, budget: int):
            return ladder.gate_step(
                ctx=ctx, cycle_plan=cycle_plan, stage=GateStage.POST_IMPLEMENTATION,
                evidence=red, repair_attempt=attempt, repair_budget=budget,
            )

        refused = gate(1, 0)
        self.assertIs(refused.strategy, RecoveryStrategy.REPLAN_STEP)
        self.assertIsNone(refused.repair_attempt)
        self.assertFalse(refused.exhausted)
        # The refused worker rung consumed nothing and reported nothing.
        ladder_dir = self.run_dir() / "cycles/001/check-repair/post-implementation"
        self.assertFalse((ladder_dir / "attempts").exists())
        self.assertFalse((ladder_dir / "ladder.json").exists())

        worker = gate(1, 1)
        self.assertIs(worker.strategy, RecoveryStrategy.REPAIR_TARGETED)
        self.assertEqual(worker.repair_attempt, 1)

    def test_red_gate_has_no_legacy_direct_repair_path(self) -> None:
        """A red gate only ever obeys the ladder, never a direct repair loop."""

        from unittest import mock

        from metaharness.orchestration.check_repair import CheckRepairLadder
        from metaharness.orchestration.recovery import GateRecoveryStep
        from metaharness.recovery_policy import RecoveryStrategy

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.REPAIR, write("feature.txt", "good\n"))
        terminal = GateRecoveryStep(RecoveryStrategy.WAIT_HUMAN, exhausted=True)
        consulted: list[int] = []

        def gate_step(_ladder, **_kwargs):
            consulted.append(1)
            return terminal

        with mock.patch.object(CheckRepairLadder, "gate_step", gate_step):
            result = self.orchestrator(
                self.config(check_repair=2), planner=[initial_plan(STEP)], reviewer=[review()],
            ).run_text(SPEC, run_id="run")

        self.assertEqual(consulted, [1], "the red gate did not ask the recovery ladder")
        self.assertEqual(result.status, RunStatus.WAITING_CHECK_REPAIR, self.state().get("failure"))
        self.assertEqual(self.state()["failure"]["reason"], "CHECK_REPAIR_EXHAUSTED")
        # Two repair passes were budgeted and a repair worker was scripted: only
        # the ladder's terminal is obeyed, the legacy loop would have run it.
        self.assertEqual(self.workers.roles(), ["implementer"])
        self.assertFalse(
            (self.run_dir() / "cycles/001/check-repair/post-implementation/attempts").exists()
        )


if __name__ == "__main__":
    unittest.main()
