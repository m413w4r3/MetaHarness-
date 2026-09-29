import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metaharness.evidence import EvidenceBundle
from metaharness.orchestration.audit import _evidence_payload
from metaharness.orchestration.audit_prompt import build_audit_payload
from metaharness.planning.protocol import parse_task_plan_v2
from tests.autonomy.support import Step, meta_plan


class AuditPromptTests(unittest.TestCase):
    def test_resumed_gate_keeps_resolvable_log_paths_and_useful_excerpts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "checks/test.stdout.log"
            log.parent.mkdir()
            log.write_text("ImportError: cannot import name old_api\n", encoding="utf-8")
            evidence = EvidenceBundle(
                base_sha="a" * 40, staged_tree_sha="b" * 40, changed_files=(), diff="",
                checks=({"name": "test", "exit_code": 2,
                         "stdout_log_path": "checks/test.stdout.log", "stderr_log_path": "",
                         "stdout_tail": "summary without the root cause"},),
                deterministic_passed=False, failures=("CHECK_FAILED:test",),
            )
            check = _evidence_payload(evidence, directory)["checks"][0]
            self.assertEqual(check["stdout_log"], str(log))
            self.assertIn("cannot import name old_api", check["failure_excerpt"])

    def payload(
        self, *, spec="SPEC exact definition", budget=64_000, excerpt="failure details",
        artifacts_readable=True,
    ):
        plan = parse_task_plan_v2(
            meta_plan(Step(id="S01", title="Feature", write=("feature.txt",)))
        )
        plan = replace(
            plan,
            risks="preserve published format",
            project_remainder="frontend belongs to M02",
            raw="RAW CONTRACT SHOULD NOT BE PASTED" * 10000,
        )
        return build_audit_payload(
            spec=spec,
            plan=plan,
            steps=[
                {
                    "id": "S01",
                    "status": "FAILED_CONTINUED",
                    "reason": "CHECK_FAILED",
                    "out_of_scope_paths": ["fixture.py"],
                    "final": "OLD WORKER MESSAGE" * 10000,
                },
            ],
            baseline={
                "checks": [
                    {
                        "id": "test",
                        "verdict": "REGRESSION",
                        "baseline_status": "PASS",
                        "new_failure_ids": ["tests/test_feature.py::test_feature"],
                    }
                ],
                "unused_evidence": "BASELINE RAW" * 10000,
            },
            gate={
                "passed": False,
                "failures": ["CHECK_FAILED:test"],
                "warnings": [],
                "checks": [
                    {
                        "id": "test",
                        "exit_code": 1,
                        "failure_excerpt": excerpt,
                        "stdout_log": "/run/checks/test.stdout.log",
                    },
                ],
            },
            changed_paths=["feature.txt"],
            diff_base_tree="a" * 40,
            candidate_parent="b" * 40,
            prior_remaining=["complete publication behavior"],
            hard_deny=[".env*"],
            file_refs={"plan": "/run/iterations/01/plan/task_plan.json"},
            budget_bytes=budget,
            artifacts_readable=artifacts_readable,
        )

    def test_large_recoverable_context_is_not_injected(self):
        payload = self.payload()
        self.assertLess(payload.total_bytes, 8_000)
        self.assertEqual(payload.rendered.count("SPEC exact definition"), 1)
        for redundant in (
            "RAW CONTRACT SHOULD NOT BE PASTED",
            "OLD WORKER MESSAGE",
            "BASELINE RAW",
        ):
            self.assertNotIn(redundant, payload.rendered)
        self.assertIn("/run/iterations/01/plan/task_plan.json", payload.rendered)
        self.assertIn("git diff --stat", payload.rendered)
        self.assertIn("preserve published format", payload.rendered)
        self.assertIn("frontend belongs to M02", payload.rendered)
        self.assertIn("Do not implement PROJECT_REMAINDER", payload.rendered)
        self.assertIn("FAILED_CONTINUED", payload.rendered)
        self.assertIn("fixture.py", payload.rendered)

    def test_external_audit_keeps_authority_and_excerpts_even_over_budget(self):
        spec = "SPEC exact external invariant\n" * 1000
        details = "failure with exact reproduction details\n" * 1000
        payload = self.payload(spec=spec, excerpt=details, budget=1000, artifacts_readable=False)
        self.assertIn(spec, payload.rendered)
        self.assertIn(details, payload.rendered)
        self.assertIn('"approved_step_contracts"', payload.rendered)
        self.assertIn("repository access only", payload.rendered)
        self.assertTrue(payload.budget_overrun)
        batch = json.loads(next(section.text for section in payload.sections if section.name == "batch"))
        self.assertEqual(batch["file_refs"], {})
        for section in payload.sections:
            if section.authority:
                self.assertFalse(section.truncated)

    def test_bounding_excerpts_preserves_check_authority_and_log_pointer(self):
        payload = self.payload(budget=6_000, excerpt="evidence " * 10000)
        self.assertLessEqual(payload.total_bytes, 6_000)
        excerpt = next(
            section for section in payload.sections if section.name == "failure_details"
        )
        self.assertTrue(excerpt.truncated)
        self.assertIn("tests/test_feature.py::test_feature", payload.rendered)
        self.assertIn("/run/checks/test.stdout.log", payload.rendered)
        self.assertIn("REGRESSION", payload.rendered)
        self.assertIn(".env*", payload.rendered)
        self.assertIn("complete publication behavior", payload.rendered)
        for section in payload.sections:
            if section.authority:
                self.assertFalse(section.truncated)

    def test_large_spec_is_never_silently_truncated(self):
        spec = "SPEC semantic authority\n" * 10000
        payload = self.payload(spec=spec, budget=6_000)
        self.assertIn(spec, payload.rendered)
        self.assertTrue(payload.budget_overrun)
        self.assertTrue(payload.omitted_sections)
        diagnostics = json.dumps(payload.diagnostics())
        self.assertNotIn("SPEC semantic authority", diagnostics)
        self.assertIn('"budget_overrun": true', diagnostics)

    def test_audit_references_spec_and_bounds_large_gate_diagnostics(self):
        plan = parse_task_plan_v2(meta_plan(Step(id="S01", title="Feature", write=("feature.txt",))))
        failures = ["CHECK_FAILED:test:" + str(index) + ":" + "x" * 1000 for index in range(5000)]
        payload = build_audit_payload(
            spec="EXACT SPEC " * 10_000, plan=plan, steps=[],
            baseline={"checks": [{"id": "test", "new_failure_ids": failures}]},
            gate={"passed": False, "failures": failures, "checks": [
                {"id": "test", "exit_code": 1, "stdout_log": "/run/test.log",
                 "unneeded_raw_log": "RAW" * 100_000},
            ]}, changed_paths=["feature.txt"], diff_base_tree="a" * 40,
            candidate_parent="b" * 40, prior_remaining=[],
            file_refs={"spec": "/run/spec.md", "evidence": "/run/evidence.json", "baseline": "/run/state.json"},
            budget_bytes=20_000,
            artifacts_readable=True,
        )
        self.assertNotIn("EXACT SPEC ", payload.rendered)
        self.assertNotIn("RAWRAW", payload.rendered)
        gate = json.loads(next(section.text for section in payload.sections if section.name == "gate"))
        self.assertEqual(gate["failures_count"], 5000)
        self.assertEqual(gate["failures_omitted"], 4992)
        self.assertEqual(gate["baseline"][0]["new_failure_ids_count"], 5000)
        self.assertIn("/run/evidence.json", payload.rendered)


if __name__ == "__main__":
    unittest.main()
