"""Execution selection capacity: one implementer selection per step, S01..S08.

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
        self.assertEqual((_MAX_STEP_SELECTIONS, MAX_STEPS), (8, 8))

    def test_seven_and_eight_selections_are_accepted_for_v3_and_v4(self) -> None:
        for count in (7, 8):
            for resolve in (self.v3, self.v4):
                with self.subTest(count=count, schema=resolve.__name__):
                    selection = resolve(_steps(count))
                    self.assertEqual([item.step_id for item in selection.steps], _ids(count))
        # Canonical order never depends on the request's insertion order.
        reversed_request = dict(reversed(list(_steps(8).items())))
        self.assertEqual([item.step_id for item in self.v4(reversed_request).steps], _ids(8))

    def test_s09_nine_selections_and_gaps_are_rejected(self) -> None:
        cases = {
            "S09 id": {"S01": "luna", "S09": "luna"},
            "nine selections": {**_steps(8), "S09": "luna"},
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

    def test_durable_v4_with_eight_steps_round_trips_and_validates(self) -> None:
        run_dir = self.root / "run-v4"
        selection = self.v4(_steps(8))
        ensure_execution_selection_v4(run_dir, selection)
        durable, digest = read_execution_selection_v4_with_sha256(run_dir)
        self.assertEqual(durable, selection)
        self.assertEqual([item.step_id for item in durable.steps], _ids(8))
        self.assertEqual(
            digest, hashlib.sha256((run_dir / "execution_selection.json").read_bytes()).hexdigest()
        )
        validate_execution_selection_v4(self.config, durable)
        # Idempotent re-publication (an approval retry or a resume).
        self.assertEqual(ensure_execution_selection_v4(run_dir, self.v4(_steps(8))), selection)

    def test_durable_parsers_reject_s09_nine_and_noncontiguous_payloads(self) -> None:
        for schema, ensure, parse, resolve in (
            (3, ensure_execution_selection_v3, parse_execution_selection_v3, self.v3),
            (4, ensure_execution_selection_v4, parse_execution_selection_v4, self.v4),
        ):
            run_dir = self.root / f"run-parse-{schema}"
            ensure(run_dir, resolve(_steps(8)))
            valid = json.loads((run_dir / "execution_selection.json").read_text(encoding="utf-8"))
            self.assertEqual(parse(json.dumps(valid).encode()).steps[-1].step_id, "S08")
            implementer = valid["steps"][0]["implementer"]
            nine = dict(valid, steps=valid["steps"] + [{"step_id": "S09", "implementer": implementer}])
            s09 = dict(valid, steps=valid["steps"][:-1] + [{"step_id": "S09", "implementer": implementer}])
            gap = dict(valid, steps=[item for item in valid["steps"] if item["step_id"] != "S07"])
            for name, payload in (("nine", nine), ("S09", s09), ("gap", gap)):
                with self.subTest(schema=schema, case=name):
                    with self.assertRaises(ExecutionSelectionError):
                        parse(json.dumps(payload).encode())


if __name__ == "__main__":
    unittest.main()
