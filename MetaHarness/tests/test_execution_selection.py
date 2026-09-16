"""Execution selection capacity: one implementer selection per step, S01..S99.

Schema 3 and schema 4 share ``_canonical_step_items``; both must carry the
authoritative MAX_STEPS and survive a durable write/read/validate cycle.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.execution_selection import (  # noqa: E402
    _MAX_STEP_SELECTIONS,
    ExecutionSelectionError,
    ensure_execution_selection_v3,
    ensure_execution_selection_v4,
    parse_execution_selection_v3,
    parse_execution_selection_v4,
    read_execution_selection_v4_with_sha256,
    resolve_execution_selection_v3,
    resolve_execution_selection_v4,
    validate_execution_selection_v4,
)
from metaharness.models import ExecutionSelectionV3, StepExecutionSelection  # noqa: E402
from metaharness.step_ids import MAX_STEPS  # noqa: E402
from tests.test_p28_full_pipeline import P28Harness  # noqa: E402


def _steps(count: int) -> dict[str, str]:
    return {f"S{number:02d}": "luna" for number in range(1, count + 1)}


def _ids(count: int) -> list[str]:
    return [f"S{number:02d}" for number in range(1, count + 1)]


class ExecutionSelectionCapacityTests(P28Harness):
    def setUp(self) -> None:
        super().setUp()
        self.config = self.load(revision=True)

    def v3(self, steps: dict[str, str]):
        return resolve_execution_selection_v3(
            self.config, planner_profile_id="planner", step_profile_ids=steps,
            reviewer_profile_id="reviewer",
        )

    def v4(self, steps: dict[str, str]):
        return resolve_execution_selection_v4(
            self.config, planner_profile_id="planner", step_profile_ids=steps,
            reviser_profile_id="claude", repair_implementer_profile_id="luna",
            reviewer_profile_id="reviewer",
        )

    def test_capacity_is_the_shared_step_authority(self) -> None:
        self.assertEqual((_MAX_STEP_SELECTIONS, MAX_STEPS), (99, 99))

    def test_large_selections_are_accepted_for_v3_and_v4(self) -> None:
        for count in (9, 16, 32, 99):
            for resolve in (self.v3, self.v4):
                with self.subTest(count=count, schema=resolve.__name__):
                    selection = resolve(_steps(count))
                    self.assertEqual([item.step_id for item in selection.steps], _ids(count))
        # Canonical order never depends on the request's insertion order.
        reversed_request = dict(reversed(list(_steps(32).items())))
        self.assertEqual([item.step_id for item in self.v4(reversed_request).steps], _ids(32))

    def test_s09_nine_selections_and_gaps_are_rejected(self) -> None:
        cases = {
            "S100 id": {"S01": "luna", "S100": "luna"},
            "100 selections": {**_steps(99), "S100": "luna"},
            "gap": {"S01": "luna", "S02": "luna", "S04": "luna"},
            "missing S07": {**_steps(6), "S08": "luna"},
            "no S01": {"S02": "luna"},
            "empty": {},
        }
        for name, steps in cases.items():
            for resolve in (self.v3, self.v4):
                with self.subTest(case=name, schema=resolve.__name__):
                    with self.assertRaises(ExecutionSelectionError):
                        resolve(steps)

    def test_durable_v4_with_32_steps_round_trips_and_validates(self) -> None:
        run_dir = self.root / "run-v4"
        selection = self.v4(_steps(32))
        ensure_execution_selection_v4(run_dir, selection)
        durable, digest = read_execution_selection_v4_with_sha256(run_dir)
        self.assertEqual(durable, selection)
        self.assertEqual([item.step_id for item in durable.steps], _ids(32))
        self.assertEqual(
            digest, hashlib.sha256((run_dir / "execution_selection.json").read_bytes()).hexdigest()
        )
        validate_execution_selection_v4(self.config, durable)
        # Idempotent re-publication (an approval retry or a resume).
        self.assertEqual(ensure_execution_selection_v4(run_dir, self.v4(_steps(32))), selection)

    def test_durable_parsers_reject_s09_nine_and_noncontiguous_payloads(self) -> None:
        for schema, ensure, parse, resolve in (
            (3, ensure_execution_selection_v3, parse_execution_selection_v3, self.v3),
            (4, ensure_execution_selection_v4, parse_execution_selection_v4, self.v4),
        ):
            run_dir = self.root / f"run-parse-{schema}"
            ensure(run_dir, resolve(_steps(32)))
            valid = json.loads((run_dir / "execution_selection.json").read_text(encoding="utf-8"))
            self.assertEqual(parse(json.dumps(valid).encode()).steps[-1].step_id, "S32")
            implementer = valid["steps"][0]["implementer"]
            over = dict(valid, steps=valid["steps"] + [{"step_id": "S100", "implementer": implementer}])
            invalid = dict(valid, steps=valid["steps"][:-1] + [{"step_id": "S100", "implementer": implementer}])
            gap = dict(valid, steps=[item for item in valid["steps"] if item["step_id"] != "S07"])
            for name, payload in (("over", over), ("S100", invalid), ("gap", gap)):
                with self.subTest(schema=schema, case=name):
                    with self.assertRaises(ExecutionSelectionError):
                        parse(json.dumps(payload).encode())

    def test_manual_malformed_v3_selections_are_rejected_before_publication(self) -> None:
        valid = self.v3(_steps(32))
        implementer = valid.steps[0].implementer
        cases = {
            "gap": (valid.steps[0], valid.steps[2]),
            "S02 only": (StepExecutionSelection("S02", implementer),),
            "S100": (StepExecutionSelection("S100", implementer),),
            "more than MAX_STEPS": valid.steps + (StepExecutionSelection("S100", implementer),),
        }
        for name, steps in cases.items():
            with self.subTest(case=name):
                run_dir = self.root / f"run-manual-v3-{name.replace(' ', '-')}"
                malformed = ExecutionSelectionV3(
                    valid.schema_version, valid.planner, steps, valid.reviewer, valid.reviser,
                )
                with self.assertRaises(ExecutionSelectionError):
                    ensure_execution_selection_v3(run_dir, malformed)
                self.assertFalse((run_dir / "execution_selection.json").exists())


if __name__ == "__main__":
    unittest.main()
