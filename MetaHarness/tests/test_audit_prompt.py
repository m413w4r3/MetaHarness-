import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from metaharness.evidence import EvidenceBundle
from metaharness.orchestration.audit import _evidence_payload
from metaharness.orchestration.audit import build_audit_payload, parse_audit_report
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
            objective="reuse publication artifacts safely",
            milestone_goal="reuse a prior publication artifact",
            acceptance=("the publication result keeps provenance",),
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
                    "detail": "required change is outside the WRITE_SET",
                    "mismatch": "find_reusable refuses PUBLICATION",
                    "out_of_scope_paths": ["fixture.py"],
                    "changed_paths": ["feature.txt"],
                    "final": "Worker conclusion: publication reuse lacks provenance.",
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
        self.assertIn("reuse publication artifacts safely", payload.rendered)
        self.assertIn("Worker conclusion: publication reuse lacks provenance.", payload.rendered)
        self.assertIn("find_reusable refuses PUBLICATION", payload.rendered)
        self.assertNotIn("frontend belongs to M02", payload.rendered)
        self.assertNotIn('"approved_step_contracts"', payload.rendered)
        self.assertIn("Keep repairs within the current milestone", payload.rendered)
        self.assertIn("FAILED_CONTINUED", payload.rendered)
        self.assertIn('"changed_paths"', payload.rendered)

    def test_external_audit_bounds_context_and_drops_contract_dump(self):
        spec = "SPEC exact external invariant\n" * 1000
        details = "failure with exact reproduction details\n" * 1000
        payload = self.payload(spec=spec, excerpt=details, budget=1000, artifacts_readable=False)
        self.assertIn(spec, payload.rendered)
        self.assertNotIn(details, payload.rendered)
        self.assertNotIn('"approved_step_contracts"', payload.rendered)
        self.assertIn("repository access only", payload.rendered)
        self.assertTrue(payload.budget_overrun)
        self.assertLess(payload.total_bytes, 150_000)
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
        self.assertNotIn("complete publication behavior", payload.rendered)
        for section in payload.sections:
            if section.authority:
                self.assertFalse(section.truncated)

    def test_large_spec_is_never_silently_truncated_and_dispatch_guard_can_reject_it(self):
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
            candidate_parent="b" * 40,
            file_refs={"spec": "/run/spec.md", "evidence": "/run/evidence.json", "baseline": "/run/state.json"},
            budget_bytes=20_000,
            artifacts_readable=True,
        )
        self.assertNotIn("EXACT SPEC ", payload.rendered)
        self.assertNotIn("RAWRAW", payload.rendered)
        gate = json.loads(next(section.text for section in payload.sections if section.name == "gate"))
        self.assertEqual(gate["failures_count"], 5000)
        self.assertEqual(gate["failures_omitted"], 4980)
        self.assertEqual(gate["baseline"][0]["new_failure_ids_count"], 5000)
        self.assertIn("/run/evidence.json", payload.rendered)

    def test_audit_report_parser_recovers_bounded_indented_bullet_continuations(self):
        message = (
            "META AUDIT v1\n\nSTATUS\nDONE\n\nFIXED\n"
            "- Reused the publication artifact while preserving\n"
            "  its provenance fields.\n\nREFACTORED\n- none\n\n"
            "REMAINING\n- none\n\nRISKS\n- none\nEND META AUDIT\n"
        )
        report = parse_audit_report(message)
        self.assertIsNotNone(report)
        self.assertEqual(report.fixed, ("Reused the publication artifact while preserving its provenance fields.",))

    def test_large_secondary_evidence_is_pruned_below_the_absolute_ceiling(self):
        plan = parse_task_plan_v2(meta_plan(Step(id="S01", title="Feature", write=("feature.txt",))))
        payload = build_audit_payload(
            spec="Exact bounded spec.", plan=plan,
            steps=[{"id": f"S{index:02d}", "final": "agent conclusion " * 1_000} for index in range(1, 31)],
            baseline={"checks": []}, gate={"passed": False, "failures": ["failure"] * 1_000,
                "checks": [{"id": "test", "failure_excerpt": "log " * 100_000}]},
            changed_paths=[f"src/file_{index}.py" for index in range(1_000)],
            diff_base_tree="a" * 40, candidate_parent="b" * 40,
            file_refs={}, budget_bytes=150_000,
            candidate_diff="diff --git\n" + "+change\n" * 1_000_000,
        )
        self.assertLessEqual(payload.total_bytes, 150_000)
        self.assertFalse(payload.budget_overrun)


if __name__ == "__main__":
    unittest.main()
