import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness import approval as approval_module  # noqa: E402
from metaharness.approval import (  # noqa: E402
    ApprovalDecision,
    ApprovalError,
    PlanIdentity,
    compute_plan_identity,
    compute_plan_identity_from_run,
    read_check_authority,
    read_plan_approval,
    wait_for_plan_approval,
    write_plan_approval,
    write_check_authority,
)
from metaharness.cli import main  # noqa: E402
from metaharness.models import CheckConfig, RunStatus  # noqa: E402
from metaharness.state import RunStateStore  # noqa: E402


class ApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = "plané\n"
        self.contract = "contract\n"
        self.identity = compute_plan_identity(self.raw, self.contract)

    def test_identity_hashes_exact_utf8_bytes(self) -> None:
        self.assertEqual(
            self.identity.raw_sha256,
            hashlib.sha256(self.raw.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            compute_plan_identity_from_run(self._run_dir("identity")),
            self.identity,
        )

    def test_write_read_and_second_decision_are_exclusive(self) -> None:
        directory = self._run_dir("write")
        write_plan_approval(
            directory,
            decision=ApprovalDecision.APPROVE,
            identity=self.identity,
            source="test",
        )
        approval = read_plan_approval(directory, expected_identity=self.identity)
        self.assertIsNotNone(approval)
        self.assertEqual(approval.decision, ApprovalDecision.APPROVE)
        with self.assertRaises(ApprovalError):
            write_plan_approval(
                directory,
                decision=ApprovalDecision.REJECT,
                identity=self.identity,
                source="test",
            )
        self.assertEqual(
            json.loads((directory / "plan_approval.json").read_text())["decision"],
            "APPROVE",
        )

    def test_invalid_or_wrong_approval_fails_closed(self) -> None:
        cases = (
            {"schema_version": 2},
            {"schema_version": 1, "decision": "MAYBE"},
            {"schema_version": 1, "decision": "APPROVE", "raw_sha256": "bad"},
            {
                "schema_version": 1,
                "decision": "APPROVE",
                "raw_sha256": self.identity.raw_sha256,
                "contract_sha256": "0" * 64,
                "created_at": "now",
                "source": "test",
            },
            {"schema_version": 1, "decision": "APPROVE", "raw_sha256": self.identity.raw_sha256},
        )
        for index, payload in enumerate(cases):
            with self.subTest(index=index):
                directory = self._run_dir(f"invalid-{index}")
                path = directory / "plan_approval.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ApprovalError):
                    read_plan_approval(directory, expected_identity=self.identity)
        directory = self._run_dir("wrong-plan")
        write_plan_approval(
            directory,
            decision=ApprovalDecision.APPROVE,
            identity=self.identity,
            source="test",
        )
        other = compute_plan_identity("other", self.contract)
        with self.assertRaises(ApprovalError):
            read_plan_approval(directory, expected_identity=other)

    def test_check_authority_hash_is_bound_to_approval(self) -> None:
        directory = self._run_dir("check-authority")
        (directory / "planner.raw.md").write_text(self.raw, encoding="utf-8")
        (directory / "implementation_contract.md").write_text(self.contract, encoding="utf-8")
        check = CheckConfig("lint", ("make", "lint"), timeout_seconds=18000)
        write_check_authority(directory, [check], required_check_ids=("lint",))
        identity = compute_plan_identity_from_run(directory)
        write_plan_approval(
            directory, decision=ApprovalDecision.APPROVE,
            identity=identity, source="test",
        )
        payload = json.loads((directory / "plan_approval.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["checks_sha256"], identity.checks_sha256)
        self.assertEqual(read_check_authority(directory)[0], ("lint",))

        changed = json.loads((directory / "check_authority.json").read_text(encoding="utf-8"))
        changed["checks"][0]["argv"] = ["make", "different-lint"]
        (directory / "check_authority.json").write_text(
            json.dumps(changed), encoding="utf-8"
        )
        with self.assertRaises(ApprovalError):
            read_plan_approval(directory, expected_identity=identity)

    def test_schema_2_freezes_the_whole_catalogue_and_names_the_selection(self) -> None:
        directory = self._run_dir("authority-schema2")
        catalogue = [
            CheckConfig("lint", ("make", "lint")),
            CheckConfig("test", ("make", "test")),
            CheckConfig("integration", ("make", "integration")),
        ]
        write_check_authority(directory, catalogue, required_check_ids=("lint", "test"))
        payload = json.loads((directory / "check_authority.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["required_check_ids"], ["lint", "test"])
        self.assertEqual([entry["id"] for entry in payload["checks"]],
                         ["lint", "test", "integration"])
        required_ids, frozen = read_check_authority(directory)
        self.assertEqual(required_ids, ("lint", "test"))
        # The catalogue keeps the check C01 did not select, with its argv.
        self.assertEqual([check.id for check in frozen], ["lint", "test", "integration"])
        self.assertEqual(frozen[2].argv, ("make", "integration"))
        # Republishing the identical bytes stays idempotent; different bytes do not.
        write_check_authority(directory, catalogue, required_check_ids=("lint", "test"))
        with self.assertRaises(ApprovalError):
            write_check_authority(directory, catalogue, required_check_ids=("lint",))

    def test_schema_2_rejects_a_selection_outside_the_frozen_catalogue(self) -> None:
        directory = self._run_dir("authority-outside")
        catalogue = [CheckConfig("lint", ("make", "lint"))]
        with self.assertRaises(ApprovalError):
            write_check_authority(directory, catalogue, required_check_ids=("lint", "test"))
        with self.assertRaises(ApprovalError):
            write_check_authority(directory, catalogue, required_check_ids=("lint", "lint"))
        self.assertFalse((directory / "check_authority.json").exists())
        payload = {
            "schema_version": 2,
            "required_check_ids": ["absent"],
            "checks": [{"id": "lint", "argv": ["make", "lint"], "cwd": ".",
                        "timeout_seconds": 1800, "preflight_argv": [], "required": True}],
        }
        (directory / "check_authority.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ApprovalError):
            read_check_authority(directory)

    def test_authority_round_trip_uses_the_current_schema(self) -> None:
        directory = self._run_dir("authority-schema2-round-trip")
        check = CheckConfig("lint", ("make", "lint"))
        write_check_authority(directory, [check], required_check_ids=("lint",))
        before = (directory / "check_authority.json").read_bytes()
        self.assertEqual(json.loads(before)["schema_version"], 2)
        required_ids, frozen = read_check_authority(directory)
        self.assertEqual((required_ids, [c.id for c in frozen]), (("lint",), ["lint"]))
        read_check_authority(
            directory,
            expected_sha256=hashlib.sha256(before).hexdigest(),
            trusted_check_ids=("lint",),
        )
        self.assertEqual((directory / "check_authority.json").read_bytes(), before)

    def test_wait_returns_decision_and_preserves_keyboard_interrupt(self) -> None:
        directory = self._run_dir("wait")
        with mock.patch("metaharness.approval.time.sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                wait_for_plan_approval(
                    directory,
                    identity=self.identity,
                    poll_interval_seconds=0.01,
                )

        # The decision is published only after the wait has observed that no
        # decision exists yet, so the wait must keep polling to return it.
        first_miss = threading.Event()
        real_read = approval_module.read_plan_approval

        def observed_read(*args, **kwargs):
            result = real_read(*args, **kwargs)
            if result is None:
                first_miss.set()
            return result

        def approve() -> None:
            self.assertTrue(first_miss.wait(timeout=5))
            write_plan_approval(
                directory,
                decision=ApprovalDecision.APPROVE,
                identity=self.identity,
                source="test",
            )

        thread = threading.Thread(target=approve)
        thread.start()
        with mock.patch.object(approval_module, "read_plan_approval", side_effect=observed_read):
            approval = wait_for_plan_approval(
                directory,
                identity=self.identity,
                poll_interval_seconds=0.01,
            )
        thread.join(timeout=5)
        self.assertTrue(first_miss.is_set())
        self.assertEqual(approval.decision, ApprovalDecision.APPROVE)

    def test_cli_requires_waiting_state_and_does_not_write_state(self) -> None:
        directory = self._run_dir("cli")
        store = RunStateStore(directory / "state.json")
        state = store.initialize("cli")
        (directory / "planner.raw.md").write_text(self.raw, encoding="utf-8")
        (directory / "implementation_contract.md").write_text(self.contract, encoding="utf-8")
        state = store.update(
            status=RunStatus.AWAITING_PLAN_APPROVAL,
            plan_identity={
                "raw_sha256": self.identity.raw_sha256,
                "contract_sha256": self.identity.contract_sha256,
            },
        )
        before = (directory / "state.json").read_bytes()
        self.assertEqual(main(["approve-plan", "--run", str(directory)]), 0)
        self.assertEqual((directory / "state.json").read_bytes(), before)
        self.assertEqual(
            read_plan_approval(directory, expected_identity=self.identity).decision,
            ApprovalDecision.APPROVE,
        )
        self.assertEqual(main(["reject-plan", "--run", str(directory)]), 2)
        self.assertEqual(store.load(), state)

    def _run_dir(self, name: str) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root))
        directory = root / name
        directory.mkdir()
        if name.startswith("identity"):
            (directory / "planner.raw.md").write_bytes(self.raw.encode("utf-8"))
            (directory / "implementation_contract.md").write_bytes(self.contract.encode("utf-8"))
        return directory


if __name__ == "__main__":
    unittest.main()
