"""Removed gate and review authorities cannot re-enter the normal path."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / "src" / "metaharness"


class AuditArchitectureTests(unittest.TestCase):
    def test_removed_gate_protocols_are_absent_from_source(self) -> None:
        source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.rglob("*.py"))
        for forbidden in (
            "META CHECK REPAIR RESULT", "TARGETED_CHECK", "CHECK_REPAIR_UNAVAILABLE",
            "WAITING_CHECK_REPAIR", "_GATE_LADDER",
        ):
            self.assertNotIn(forbidden, source)


    def test_coordinator_has_one_audit_authority(self) -> None:
        source = (ROOT / "orchestration" / "pipeline_v2.py").read_text(encoding="utf-8")
        coordinator = source.split("class PipelineV2Coordinator:", 1)[1]
        self.assertIn("ops.run_audit(", coordinator)
        self.assertIn("ops.run_gate(", coordinator)
        self.assertNotIn("ops.check_repair_attempt(", coordinator)
        self.assertNotIn("ops.review_candidate(", coordinator)
        self.assertNotIn("ops.semantic_revision(", coordinator)
