"""The deterministic gate supplies evidence to one writable audit authority."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from unittest.mock import patch

from metaharness.agent.base import AGENT_RUNTIME_FAILED, AgentRunResult
from metaharness.agent.protocol import CONTRACT_MISMATCH_HEADER
from metaharness.models import AgentExecutorCapabilities, ExecutionRole
from metaharness.recovery_policy import ExecutionFallbacks
from tests.autonomy.support import SPEC, Step, meta_plan
from tests.pipeline_support import PipelineHarness, continuation_answer, write


def audit_message(status: str = "DONE", remaining: str = "none") -> str:
    return (
        "META AUDIT v1\n\nSTATUS\n" + status +
        "\n\nFIXED\n- regression\n\nREFACTORED\n- none\n\n"
        "REMAINING\n- " + remaining + "\n\nRISKS\n- none\nEND META AUDIT\n"
    )


class AuditPipelineTests(PipelineHarness):
    def test_repository_only_executor_receives_complete_inline_authority(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        with patch("tests.pipeline_support._Executor.capabilities", AgentExecutorCapabilities(edits_workspace=True)):
            result = self.orchestrator(
                self.config(), planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
            ).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        for request in self.workers.calls:
            self.assertIn(SPEC, request.prompt)
            self.assertEqual(request.read_only_paths, ())
            if request.role is ExecutionRole.AUDITOR:
                self.assertIn('"approved_step_contracts"', request.prompt)
                self.assertIn("repository access only", request.prompt)
                self.assertIn("diff --git", request.prompt)
                self.assertIn("+good", request.prompt)

    def test_soft_prompt_budgets_allow_workers_and_audit_to_finish(self) -> None:
        config = self.config()
        config = replace(config, prompt_budget=replace(config.prompt_budget, implementer_max_bytes=1, audit_max_bytes=1))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(config, planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        self.assertIn("auditor", self.workers.roles())

    def _audit_fallback_config(self):
        config = self.config()
        profiles = dict(config.model_profiles)
        profiles["audit-fallback"] = replace(profiles["auditor"], id="audit-fallback")
        return replace(config, model_profiles=profiles,
                       execution_fallbacks=ExecutionFallbacks(audit=("audit-fallback",)))

    def _limited_audit(self, request):
        (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
        return AgentRunResult(
            status="failed", exit_reason=AGENT_RUNTIME_FAILED, backend_reason="rate_limited",
            tree_before="", tree_after="", usage=None, external_session_id=None,
            report_path=None, exit_code=1, terminal_is_error=True,
            final_message="You've hit your session limit · resets 4:20am (Europe/Paris)",
        )

    def test_rate_limit_switches_auditor_and_preserves_partial_edits(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))

        def fallback(request):
            self.assertEqual(request.profile_id, "audit-fallback")
            self.assertEqual((request.worktree / "feature.txt").read_text(), "good\n")
            self.assertIn("uncommitted edits", request.prompt)
            self.assertIsNone(request.mutable_paths)
            diff = (request.artifact_dir / "diff.patch").read_text()
            self.assertIn("+good", diff)
            self.assertNotIn("+bad", diff)
            return audit_message()

        self.workers.on(ExecutionRole.AUDITOR, self._limited_audit, fallback)
        result = self.orchestrator(self._audit_fallback_config(), planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        report = json.loads((self.run_dir() / "cycles/001/audit/001/report.json").read_text())
        self.assertEqual(report["profile_id"], "audit-fallback")
        self.assertEqual([item["profile_id"] for item in report["executions"]], ["auditor", "audit-fallback"])
        self.assertIn("feature.txt", report["changed_paths"])
        fallback_prompt = next(call.prompt for call in self.workers.calls if call.profile_id == "audit-fallback")
        diagnostics = json.loads((self.run_dir() / "cycles/001/audit/001/executors/002/prompt.diagnostics.json").read_text())
        self.assertEqual(diagnostics["prompt_bytes"], len(fallback_prompt.encode()))
        selection = json.loads((self.run_dir() / "iterations/01/execution_selection.json").read_text())
        self.assertEqual(selection["audit_fallbacks"][0]["profile_id"], "audit-fallback")
        options = json.loads((self.run_dir() / "run_options.json").read_text())
        self.assertEqual(options["execution_fallbacks"]["audit"], ["audit-fallback"])

    def test_audit_fallback_respects_semantic_attempt_budget(self) -> None:
        config = self._audit_fallback_config()
        config = replace(config, budget=replace(config.budget, step_attempts=1))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, self._limited_audit)
        result = self.orchestrator(config, planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], AGENT_RUNTIME_FAILED)
        self.assertEqual(self.workers.roles().count("auditor"), 1)

    def test_rate_limit_without_fallback_retains_real_failure(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, self._limited_audit)
        result = self._run()
        self.assertEqual(result.state["disposition"], "WAIT_EXTERNAL")
        self.assertEqual(result.state["failure"]["reason"], AGENT_RUNTIME_FAILED)
        self.assertFalse((self.run_dir() / "cycles/001/audit/001/report.json").exists())

    def test_all_auditors_rate_limited_wait_without_repeating(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, self._limited_audit, self._limited_audit)
        result = self.orchestrator(self._audit_fallback_config(), planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], AGENT_RUNTIME_FAILED)
        self.assertEqual(self.workers.roles().count("auditor"), 2)

    def test_invalid_audit_report_does_not_trigger_rate_limit_fallback(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: "invalid report")
        result = self.orchestrator(self._audit_fallback_config(), planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], "AGENT_RUNTIME_FAILED")
        self.assertEqual(self.workers.roles().count("auditor"), 1)

    def test_hard_deny_before_rate_limit_prevents_fallback(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        def limited(request):
            (request.worktree / ".env").write_text("opaque\n", encoding="utf-8")
            return self._limited_audit(request)

        self.workers.on(ExecutionRole.AUDITOR, limited)
        result = self.orchestrator(self._audit_fallback_config(), planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], "HARD_DENY_PATH_MUTATION")
        self.assertEqual(self.workers.roles().count("auditor"), 1)

    def test_failed_execution_cannot_supply_an_accepted_report(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))

        def failed(request):
            return replace(self._limited_audit(request), exit_reason="AGENT_RUNTIME_FAILED",
                           backend_reason=None, final_message=audit_message())

        self.workers.on(ExecutionRole.AUDITOR, failed)
        result = self.orchestrator(self._audit_fallback_config(), planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["failure"]["reason"], "AGENT_RUNTIME_FAILED")
        self.assertEqual(self.workers.roles().count("auditor"), 1)
        self.assertFalse((self.run_dir() / "cycles/001/audit/001/report.json").exists())

    def test_audit_fallback_snapshot_rejects_profile_drift(self) -> None:
        from metaharness.execution_selection import (
            ExecutionSelectionError,
            read_execution_selection,
            validate_execution_selection,
        )
        config = self._audit_fallback_config()
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: audit_message())
        self.orchestrator(config, planner=[
            meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",))),
        ]).run_text(SPEC, run_id="run")
        selection = read_execution_selection(self.run_dir())
        profiles = dict(config.model_profiles)
        profiles["audit-fallback"] = replace(profiles["audit-fallback"], model="changed-model")
        with self.assertRaises(ExecutionSelectionError):
            validate_execution_selection(replace(config, model_profiles=profiles), selection)

    def _run(self, *, extra_checks: str = "", continuation: list[str] | None = None):
        return self.orchestrator(
            self.config(extra_checks=extra_checks),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)),
                               required_checks=("test", "integration", "frontend-e2e") if extra_checks else ("test",))],
            continuation=continuation,
        ).run_text(SPEC, run_id="run")

    def test_green_gate_still_calls_auditor(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: audit_message())
        result = self.orchestrator(
            self.config(),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
        ).run_text(SPEC, run_id="run")
        self.assertIn("auditor", self.workers.roles())
        self.assertNotIn("reviewer", self.workers.roles())
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        prompt = next(call.prompt for call in self.workers.calls if call.role is ExecutionRole.AUDITOR)
        self.assertNotIn("diff --git", prompt)
        self.assertNotIn('"raw":', prompt)
        diagnostics = json.loads((self.run_dir() / "cycles/001/audit/001/prompt.diagnostics.json").read_text())
        self.assertEqual(diagnostics["role"], "auditor")
        self.assertLess(diagnostics["prompt_bytes"], 64_000)
        request = next(call for call in self.workers.calls if call.role is ExecutionRole.AUDITOR)
        self.assertEqual(request.read_only_paths, (self.run_dir(),))
        self.assertIn("+good", (request.artifact_dir / "diff.patch").read_text())
        batch = json.loads(prompt.split("CURRENT MILESTONE AND BATCH\n", 1)[1].split("PRECEDING AUDITOR HANDOFF", 1)[0])
        from pathlib import Path
        self.assertTrue(Path(batch["file_refs"]["evidence"]).is_file())
        self.assertEqual(Path(batch["file_refs"]["spec"]).read_text(), SPEC)
        self.assertNotIn(SPEC, prompt)
        worker = next(call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER)
        self.assertNotIn(SPEC, worker.prompt)
        self.assertEqual((worker.read_only_paths[0] / "spec.md").read_text(), SPEC)

    def test_audit_has_no_contractual_scope_and_the_worker_has_a_bounded_one(self) -> None:
        """``mutable_paths`` is explicit: a tuple bounds an agent, ``None`` is no scope."""

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        result = self.orchestrator(
            self.config(),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
        ).run_text(SPEC, run_id="run")
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        (implementer,) = [
            call for call in self.workers.calls if call.role is ExecutionRole.IMPLEMENTER
        ]
        (auditor,) = [call for call in self.workers.calls if call.role is ExecutionRole.AUDITOR]
        # The worker honours the declared step authority; the audit is bounded
        # by the harness ScopePolicy alone, never by a contract of its own.
        self.assertEqual(implementer.mutable_paths, ("feature.txt",))
        self.assertIsNone(auditor.mutable_paths)

    def test_red_gate_calls_auditor_with_failures_and_reruns(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))

        def audit(request):
            self.assertIn('"CHECK_FAILED:test"', request.prompt)
            self.assertIn('"baseline"', request.prompt)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return audit_message()

        self.workers.on(ExecutionRole.AUDITOR, audit)
        result = self.orchestrator(
            self.config(),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
        ).run_text(SPEC, run_id="run")
        self.assertIn("auditor", self.workers.roles())
        report = json.loads((self.run_dir() / "cycles/001/audit/001/report.json").read_text())
        self.assertEqual(report["status"], "DONE")
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))

    def test_two_real_regressions_reach_audit_despite_model_sandbox(self) -> None:
        self.check.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
        integration = self.root / "integration.py"
        e2e = self.root / "e2e.py"
        for path, name in ((integration, "integration"), (e2e, "frontend-e2e")):
            path.write_text(
                "from pathlib import Path\nimport sys\n"
                "value = Path('feature.txt').read_text().strip()\n"
                f"print('FAILED tests/test_{name}.py::test_feature' if value == 'bad' else 'PASS')\n"
                "sys.exit(1 if value == 'bad' else 0)\n",
                encoding="utf-8",
            )
        checks = "\n".join(
            f'[[check_catalog]]\nid = "{name}"\nargv = [{sys.executable!r}, {str(path)!r}]\n'
            for name, path in (("integration", integration), ("frontend-e2e", e2e))
        )
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))

        def audit(request):
            self.assertIn("CHECK_FAILED:integration", request.prompt)
            self.assertIn("CHECK_FAILED:frontend-e2e", request.prompt)
            self.assertIn("tests/test_integration.py::test_feature", request.prompt)
            self.assertIn("tests/test_frontend-e2e.py::test_feature", request.prompt)
            self.assertIn('"baseline"', request.prompt)
            (request.worktree / "feature.txt").write_text("good\n", encoding="utf-8")
            return audit_message().replace("- regression", "- corrected integration and E2E; Docker and Playwright unavailable in agent sandbox")

        self.workers.on(ExecutionRole.AUDITOR, audit)
        result = self._run(extra_checks=checks)
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        self.assertEqual(self.workers.roles().count("auditor"), 1)
        self.assertNotIn("check_repair", self.workers.roles())
        self.assertNotIn("CHECK_REPAIR_UNAVAILABLE", json.dumps(result.state))
        self.assertNotIn("wait_external", json.dumps(result.state).lower())
        evidence = json.loads((self.run_dir() / "cycles/001/checks/post-implementation/evidence.json").read_text())
        self.assertTrue(evidence["deterministic_passed"])

    def test_audit_may_edit_a_failing_test(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_feature.py").write_text("assert False\n", encoding="utf-8")
        from tests.pipeline_support import git
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "baseline test")
        self.base_sha = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "push", "-q", "origin", "main")
        self.check.write_text(
            "from pathlib import Path\nimport sys\n"
            "failing = 'assert False' in Path('tests/test_feature.py').read_text()\n"
            "print('FAILED tests/test_feature.py::test_feature' if failing else 'PASS')\n"
            "sys.exit(1 if failing else 0)\n",
            encoding="utf-8",
        )
        self.workers.on(ExecutionRole.AUDITOR, write("tests/test_feature.py", "assert True\n", audit_message()))
        result = self._run()
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        report = json.loads((self.run_dir() / "cycles/001/audit/001/report.json").read_text())
        self.assertIn("tests/test_feature.py", report["changed_paths"])

    def test_failed_step_with_empty_diff_still_calls_auditor(self) -> None:
        self.workers.on(
            ExecutionRole.IMPLEMENTER,
            *(
                lambda _request: CONTRACT_MISMATCH_HEADER
                + "\nThe approved step instructions were not met."
                for _ in range(3)
            ),
        )
        self.workers.on(ExecutionRole.AUDITOR, write("feature.txt", "good\n", audit_message()))
        result = self._run()
        self.assertIn("auditor", self.workers.roles())
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))

    def test_hard_deny_audit_path_is_refused(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, write(".env", "key=opaque\n", audit_message()))
        result = self._run()
        self.assertEqual(result.state["failure"]["reason"], "HARD_DENY_PATH_MUTATION")

    def test_spec_decision_waits_for_human(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: audit_message("SPEC_DECISION", "Choose format"))
        result = self._run()
        self.assertEqual(result.state["disposition"], "WAIT_HUMAN")

    def test_needs_work_is_durable_without_human_wait(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "bad\n"))
        self.workers.on(ExecutionRole.AUDITOR, *(
            (lambda _request: audit_message("NEEDS_WORK", "Fix test regression")) for _ in range(2)
        ))
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        m02 = meta_plan(Step(id="S01", title="Correct the regression", write=("feature.txt",)))
        m02 = m02.replace("MILESTONE_ID: M01", "MILESTONE_ID: M02")
        result = self._run(continuation=[
            continuation_answer("NEXT", milestone="M02", plan_text=m02),
            continuation_answer("COMPLETE"),
        ])
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))
        self.assertEqual(len(self.continuation.requests), 2)
        self.assertEqual(self.checkpoint()["iteration"], 2)
        report = json.loads((self.run_dir() / "cycles/001/audit/002/report.json").read_text())
        self.assertEqual(report["remaining"], ["Fix test regression"])

    def test_later_milestone_does_not_reaudit_preceding_batch_diff(self) -> None:
        from tests.pipeline_support import git

        first_tree = []

        def first_audit(request):
            first_tree.append(git(request.worktree, "rev-parse", "HEAD^{tree}"))
            return audit_message()

        def next_audit(request):
            text = request.prompt.split("CURRENT MILESTONE AND BATCH\n", 1)[1]
            batch = json.loads(text.split("PRECEDING AUDITOR HANDOFF", 1)[0].strip())
            self.assertEqual(batch["diff_base_tree"], first_tree[0])
            self.assertEqual(batch["changed_paths"], ["other.txt"])
            return audit_message()

        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"),
                        write("other.txt", "other milestone\n"))
        self.workers.on(ExecutionRole.AUDITOR, first_audit, next_audit)
        next_plan = meta_plan(Step(id="S01", title="Other milestone", write=("other.txt",)))
        next_plan = next_plan.replace("MILESTONE_ID: M01", "MILESTONE_ID: M02")
        result = self._run(continuation=[
            continuation_answer("NEXT", milestone="M02", plan_text=next_plan),
            continuation_answer("COMPLETE"),
        ])
        self.assertEqual(result.state["status"], "committed", result.state.get("failure"))

    def test_resume_retries_an_interrupted_audit_checkpoint(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(
            ExecutionRole.AUDITOR,
            lambda _request: "invalid audit reply",
            lambda _request: audit_message(),
        )
        orchestrator = self.orchestrator(
            self.config(),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
        )
        first = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(first.state["disposition"], "WAIT_EXTERNAL")
        resumed = orchestrator.resume("run")
        self.assertEqual(resumed.state["status"], "committed", resumed.state.get("failure"))
        self.assertEqual(self.workers.roles().count("auditor"), 2)

    def test_blocking_harness_preflight_waits_external_before_audit(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        unavailable = f'''[[check_catalog]]
id = "integration"
argv = [{sys.executable!r}, "-c", "import sys; sys.exit(0)"]
preflight_argv = [{sys.executable!r}, "-c", "import sys; sys.exit(3)"]
blocking = true

[[check_catalog]]
id = "frontend-e2e"
argv = [{sys.executable!r}, "-c", "import sys; sys.exit(0)"]
'''
        result = self._run(extra_checks=unavailable)
        self.assertEqual(result.state["disposition"], "WAIT_EXTERNAL")
        self.assertNotIn("auditor", self.workers.roles())
