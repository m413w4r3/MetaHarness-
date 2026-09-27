"""Removed gate and review authorities cannot re-enter the normal path."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src" / "metaharness"


def test_removed_gate_protocols_are_absent_from_source() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in ROOT.rglob("*.py"))
    for forbidden in (
        "META CHECK REPAIR RESULT", "TARGETED_CHECK", "CHECK_REPAIR_UNAVAILABLE",
        "WAITING_CHECK_REPAIR", "_GATE_LADDER",
    ):
        assert forbidden not in source


def test_coordinator_has_one_audit_authority() -> None:
    source = (ROOT / "orchestration" / "pipeline_v2.py").read_text(encoding="utf-8")
    coordinator = source.split("class PipelineV2Coordinator:", 1)[1]
    assert "ops.run_audit(" in coordinator
    assert "ops.run_gate(" in coordinator
    assert "ops.check_repair_attempt(" not in coordinator
    assert "ops.review_candidate(" not in coordinator
    assert "ops.semantic_revision(" not in coordinator
