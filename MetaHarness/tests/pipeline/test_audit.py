"""The deterministic gate supplies evidence to one writable audit authority."""

from __future__ import annotations

import json
import sys

from metaharness.models import ExecutionRole
from tests.pipeline_support import PipelineHarness, write
from tests.autonomy.support import SPEC, Step, meta_plan


def audit_message(status: str = "DONE", remaining: str = "none") -> str:
    return (
        "META AUDIT v1\n\nSTATUS\n" + status +
        "\n\nFIXED\n- regression\n\nREFACTORED\n- none\n\n"
        "REMAINING\n- " + remaining + "\n\nRISKS\n- none\nEND META AUDIT\n"
    )


class AuditPipelineTests(PipelineHarness):
    def _run(self, *, extra_checks: str = ""):
        return self.orchestrator(
            self.config(extra_checks=extra_checks),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)),
                               required_checks=("test", "integration", "frontend-e2e") if extra_checks else ("test",))],
            reviewer=[],
        ).run_text(SPEC, run_id="run")

    def test_green_gate_still_calls_auditor(self) -> None:
        self.workers.on(ExecutionRole.IMPLEMENTER, write("feature.txt", "good\n"))
        self.workers.on(ExecutionRole.AUDITOR, lambda _request: audit_message())
        result = self.orchestrator(
            self.config(),
            planner=[meta_plan(Step(id="S01", title="Write feature", write=("feature.txt",)))],
            reviewer=[],
        ).run_text(SPEC, run_id="run")
        self.assertIn("auditor", self.workers.roles())
        self.assertNotIn("reviewer", self.workers.roles())
        self.assertEqual(result.state["status"], "published", result.state.get("failure"))

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
            reviewer=[],
        ).run_text(SPEC, run_id="run")
        self.assertIn("auditor", self.workers.roles())
        report = json.loads((self.run_dir() / "cycles/001/audit/001/report.json").read_text())
        self.assertEqual(report["status"], "DONE")
        self.assertEqual(result.state["status"], "published", result.state.get("failure"))

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
        self.assertEqual(result.state["status"], "published", result.state.get("failure"))
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
        self.assertEqual(result.state["status"], "published", result.state.get("failure"))
        report = json.loads((self.run_dir() / "cycles/001/audit/001/report.json").read_text())
        self.assertIn("tests/test_feature.py", report["changed_paths"])

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
        result = self._run()
        self.assertEqual(result.state["failure"]["reason"], "AUDIT_REMAINING")
        self.assertNotEqual(result.state.get("disposition"), "wait_human")
        self.assertNotEqual(result.state.get("disposition"), "wait_external")
        report = json.loads((self.run_dir() / "cycles/001/audit/002/report.json").read_text())
        self.assertEqual(report["remaining"], ["Fix test regression"])

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
            reviewer=[],
        )
        first = orchestrator.run_text(SPEC, run_id="run")
        self.assertEqual(first.state["disposition"], "WAIT_EXTERNAL")
        resumed = orchestrator.resume("run")
        self.assertEqual(resumed.state["status"], "published", resumed.state.get("failure"))
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
